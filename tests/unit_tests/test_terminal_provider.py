# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import pytest
import requests

from nemo_gym.sandbox import AsyncSandbox, SandboxResources, get_provider_class
from nemo_gym.sandbox.providers.base import SandboxExecResult, SandboxSpec, SandboxStatus
from nemo_gym.sandbox.providers.registry import create_provider, list_providers
from nemo_gym.sandbox.providers.terminal import provider as terminal_provider
from nemo_gym.sandbox.providers.terminal.provider import (
    READY_PROBE_COMMAND,
    READY_PROBE_EXPECTED,
    TerminalCreateRequest,
    TerminalCreateResponse,
    TerminalCreateVerificationError,
    TerminalSandboxProvider,
)


class RecordingTerminalServiceClient:
    def __init__(self, *, session_id: str = "terminal-session-1") -> None:
        self.session_id = session_id
        self.created: list[TerminalCreateRequest] = []
        self.commands: list[dict[str, Any]] = []
        self.uploads: list[tuple[Path, str]] = []
        self.downloads: list[tuple[str, Path]] = []
        self.closed: list[str] = []

    async def create(self, request: TerminalCreateRequest) -> TerminalCreateResponse:
        self.created.append(request)
        return TerminalCreateResponse(session_id=self.session_id, raw={"request_index": len(self.created) - 1})

    async def exec(
        self,
        session_id: str,
        command: str,
        *,
        cwd: str | None,
        env: dict[str, str] | None,
        timeout_s: int | float | None,
        user: str | int | None,
    ) -> SandboxExecResult:
        self.commands.append(
            {
                "session_id": session_id,
                "command": command,
                "cwd": cwd,
                "env": env or {},
                "timeout_s": timeout_s,
                "user": user,
            }
        )
        if command == READY_PROBE_COMMAND:
            return SandboxExecResult(stdout=READY_PROBE_EXPECTED, stderr="", return_code=0)
        return SandboxExecResult(stdout="", stderr="", return_code=0)

    async def exec_long(
        self,
        session_id: str,
        command: str,
        *,
        cwd: str | None,
        env: dict[str, str] | None,
        timeout_s: int | float | None,
        user: str | int | None,
        progress_callback=None,
        idle_timeout_s: int | float | None = None,
    ) -> SandboxExecResult:
        return await self.exec(session_id, command, cwd=cwd, env=env, timeout_s=timeout_s, user=user)

    async def upload_file(self, session_id: str, source_path: Path, target_path: str) -> None:
        self.uploads.append((source_path, target_path))

    async def download_file(self, session_id: str, source_path: str, target_path: Path) -> None:
        self.downloads.append((source_path, target_path))
        target_path.write_text("", encoding="utf-8")

    async def status(self, session_id: str) -> SandboxStatus:
        return SandboxStatus.RUNNING

    async def close(self, session_id: str) -> None:
        self.closed.append(session_id)

    async def aclose(self) -> None:
        return None


def test_terminal_provider_is_registered_as_builtin() -> None:
    assert get_provider_class("terminal") is TerminalSandboxProvider


def test_terminal_provider_create_exec_transfer_and_close(tmp_path: Path) -> None:
    async def run_test() -> None:
        client = RecordingTerminalServiceClient(session_id="sess-123")
        provider = TerminalSandboxProvider(
            client=client,
            metadata={"cluster": "aiic"},
            exec={"default_timeout_s": 99, "concurrency": 2},
            probe={"command": READY_PROBE_COMMAND, "stable_count": 2, "stable_delay_s": 0.0},
        )
        sandbox = await AsyncSandbox(provider).start(
            SandboxSpec(
                image="internal/swebench:django",
                ttl_s=3600,
                ready_timeout_s=1200,
                workdir="/testbed",
                env={"A": "1"},
                metadata={"benchmark": "swebench-verified"},
                resources=SandboxResources(cpu=2, memory_mib=8192),
            )
        )

        try:
            assert client.created[0].image == "internal/swebench:django"
            assert client.created[0].workdir == "/testbed"
            assert client.created[0].metadata == {"cluster": "aiic", "benchmark": "swebench-verified"}
            assert [command["command"] for command in client.commands[:2]] == [
                READY_PROBE_COMMAND,
                READY_PROBE_COMMAND,
            ]

            await sandbox.exec("pwd", env={"B": "2"}, timeout_s=None, user="root")
            command = client.commands[-1]
            assert command["cwd"] == "/testbed"
            assert command["env"] == {"A": "1", "B": "2"}
            assert command["timeout_s"] == 99
            assert command["user"] == "root"

            local_file = tmp_path / "payload.txt"
            local_file.write_text("payload", encoding="utf-8")
            await sandbox.upload(local_file, "/tmp/payload.txt")
            await sandbox.download("/tmp/remote.txt", tmp_path / "downloaded.txt")
            assert client.uploads == [(local_file, "/tmp/payload.txt")]
            assert client.downloads == [("/tmp/remote.txt", tmp_path / "downloaded.txt")]
            assert await sandbox.status() == SandboxStatus.RUNNING
        finally:
            await sandbox.stop()

        assert client.closed == ["sess-123"]

    asyncio.run(run_test())


def test_terminal_provider_probe_failure_closes_session() -> None:
    async def run_test() -> None:
        class FailingProbeClient(RecordingTerminalServiceClient):
            async def exec(self, *args, **kwargs) -> SandboxExecResult:
                self.commands.append({"command": args[1]})
                return SandboxExecResult(stdout="not-ready", stderr="", return_code=0)

        client = FailingProbeClient(session_id="bad-session")
        provider = TerminalSandboxProvider(client=client, probe={"command": READY_PROBE_COMMAND})

        with pytest.raises(TerminalCreateVerificationError):
            await provider.create(SandboxSpec(image="image"))

        assert client.closed == ["bad-session"]

    asyncio.run(run_test())


def test_terminal_provider_returns_runtime_error_for_exec_exceptions() -> None:
    async def run_test() -> None:
        class ExecErrorClient(RecordingTerminalServiceClient):
            async def exec(self, session_id: str, command: str, **kwargs) -> SandboxExecResult:
                if command == READY_PROBE_COMMAND:
                    return SandboxExecResult(stdout=READY_PROBE_EXPECTED, stderr="", return_code=0)
                raise RuntimeError("service unavailable")

        provider = TerminalSandboxProvider(client=ExecErrorClient(), probe={"command": READY_PROBE_COMMAND})
        sandbox = await AsyncSandbox(provider).start(SandboxSpec(image="image"))
        try:
            result = await sandbox.exec("boom")
        finally:
            await sandbox.stop()

        assert result.return_code == 125
        assert result.error_type == "RuntimeError"
        assert result.stderr == "service unavailable"

    asyncio.run(run_test())


class DummyResponse:
    def __init__(
        self,
        status_code: int,
        payload: dict[str, Any] | None = None,
        *,
        text: str = "",
        chunks: list[bytes] | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = text or str(self._payload)
        self._chunks = chunks or []
        self.headers = headers or {}

    def json(self) -> dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error", response=self)

    def iter_content(self, chunk_size: int):
        yield from self._chunks

    def iter_lines(self):
        yield from self._chunks

    def close(self) -> None:
        return None


class FakeUserTokenManager:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def get_user_token(self, username: str, ignore_cache: bool = False) -> terminal_provider.UserToken:
        self.calls.append({"username": username, "ignore_cache": ignore_cache})
        return terminal_provider.UserToken(
            access_token="user-jwt",
            expires_at=terminal_provider.datetime.now() + terminal_provider.timedelta(seconds=3600),
        )


def test_terminal_provider_registered() -> None:
    assert "terminal" in list_providers()
    assert create_provider({"terminal": {"connection": {"creator_jwt_token": "jwt"}}}).name == "terminal"
    assert get_provider_class("terminal") is terminal_provider.TerminalSandboxProvider


def test_region_defaults_match_stargaze(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(terminal_provider, "_stargaze_region_config", lambda: {})

    assert terminal_provider.get_default_region() == "boei18n"
    cfg = terminal_provider.get_region_config("boei18n")
    assert cfg.terminal_sandbox_url == "http://aipaas-gateway-boei18n.byted.org/api/v1"
    assert cfg.terminal_sandbox_id == "lgjk1fy5"


def test_get_codebase_jwt_loads_stargaze_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    root = tmp_path / "StarGazeWorkflow"
    utils_dir = root / "utils"
    utils_dir.mkdir(parents=True)
    (utils_dir / "__init__.py").write_text("", encoding="utf-8")
    (utils_dir / "baseline_assets.py").write_text(
        "def _get_codebase_jwt(*, ignore_cache=False):\n    return 'codebase-jwt'\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("STARGAZE_WORKFLOW_ROOT", str(root))
    monkeypatch.setattr(terminal_provider, "_CODEBASE_JWT_CACHE", "")
    monkeypatch.delitem(sys.modules, "utils", raising=False)
    monkeypatch.delitem(sys.modules, "utils.baseline_assets", raising=False)

    assert terminal_provider._get_codebase_jwt(ignore_cache=True) == "codebase-jwt"
    assert str(root) in sys.path


def test_region_config_prefers_stargaze_oauth_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    stargaze_cfg = terminal_provider.RegionConfig(
        instance_link="stargaze-link",
        terminal_sandbox_url="http://stargaze-control/api/v1",
        _terminal_sandbox_domain=terminal_provider.EnvConfigStr(tce="stargaze.tce", dev="stargaze.dev"),
        terminal_sandbox_id="stargaze-sandbox",
        _oauth_host=terminal_provider.EnvConfigStr(tce="oauth.tce", dev="oauth.dev"),
        oauth_serv_account="codewise_service",
        oauth_serv_token="serv-token",
        oauth_client_id="client-id",
        oauth_client_secret="client-secret",
    )
    monkeypatch.setattr(terminal_provider, "_stargaze_region_config", lambda: {"boei18n": stargaze_cfg})

    cfg = terminal_provider.get_region_config("boei18n")

    assert cfg.terminal_sandbox_url == "http://stargaze-control/api/v1"
    assert cfg.oauth_serv_token == "serv-token"
    assert cfg.oauth_client_id == "client-id"
    assert cfg.oauth_client_secret == "client-secret"


async def test_create_exec_close_protocol(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake_manager = FakeUserTokenManager()
    posts: list[dict[str, Any]] = []
    deletes: list[dict[str, Any]] = []

    def fake_get_service_account_manager(region: str):
        assert region == "boei18n"
        return fake_manager

    def fake_post(url: str, **kwargs: Any) -> DummyResponse:
        posts.append({"url": url, **kwargs})
        if url.endswith("/sandboxes/lgjk1fy5/sessions"):
            return DummyResponse(
                200,
                {
                    "code": 0,
                    "data": {
                        "session_id": "session-123",
                        "faas_pod_name": "pod-1",
                        "faas_function_id": "fn-1",
                        "image": "registry.example/swe:latest",
                    },
                },
            )
        if url == "https://session-123.us-east.ai-sandbox-boei18n.byted.org/api/process/start":
            command = kwargs["json"]["command"]
            assert command["path"] == "/bin/bash"
            assert command["args"][0] == "-c"
            assert kwargs["headers"]["Accept-Encoding"] == "identity"
            assert kwargs["verify"] is False
            script = command["args"][1]
            assert "/root/.nvm/versions/node/*/bin" in script
            if "terminal-sandbox-ready" in script:
                return DummyResponse(200, {"exit_code": 0, "stdout": "terminal-sandbox-ready", "stderr": ""})
            if "printf %s" in script and ".netrc" in script:
                return DummyResponse(200, {"exit_code": 0, "stdout": "", "stderr": ""})
            assert "cd /repo" in script
            assert "export FOO=bar" in script
            assert script.endswith("printf ok")
            return DummyResponse(200, {"exit_code": 0, "stdout": "ok", "stderr": "", "pid": 42})
        raise AssertionError(f"unexpected POST {url}")

    def fake_delete(url: str, **kwargs: Any) -> DummyResponse:
        deletes.append({"url": url, **kwargs})
        return DummyResponse(200, {"code": 0})

    monkeypatch.setattr(terminal_provider, "get_service_account_manager", fake_get_service_account_manager)
    monkeypatch.setattr(terminal_provider, "_get_codebase_jwt", lambda **kwargs: "codebase-jwt")
    monkeypatch.setattr(terminal_provider.requests, "post", fake_post)
    monkeypatch.setattr(terminal_provider.requests, "delete", fake_delete)

    provider = terminal_provider._StarGazeTerminalProvider(
        connection={
            "config_dir": str(tmp_path),
            "creator_email": "dev.user@bytedance.com",
            "refresh_jwt": False,
        },
        probe={"command": terminal_provider.READY_PROBE_COMMAND},
    )
    handle = await provider.create(SandboxSpec(image="registry.example/swe:latest", ttl_s=60, env={"FOO": "bar"}))

    assert handle.sandbox_id == "session-123"
    assert fake_manager.calls == [{"username": "dev.user", "ignore_cache": True}]
    create_call = posts[0]
    assert create_call["headers"]["X-Jwt-Token"] == "user-jwt"
    assert create_call["json"]["ttl"] == 60
    assert create_call["json"]["image"] == "registry.example/swe:latest"
    assert create_call["headers"]["Accept-Encoding"] == "identity"
    assert "verify" not in create_call
    assert create_call["json"]["envs"]["PATH"] == terminal_provider.TERMINAL_PATH
    assert create_call["json"]["envs"]["FOO"] == "bar"
    assert (tmp_path / "terminal_sandbox_config.json").exists()

    result = await provider.exec(handle, "printf ok", cwd="/repo", timeout_s=10)
    assert result.return_code == 0
    assert result.stdout == "ok"

    await provider.close(handle)
    assert deletes[0]["url"].endswith("/sandboxes/lgjk1fy5/sessions/session-123")
    assert not (tmp_path / "terminal_sandbox_config.json").exists()


async def test_upload_download_native(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []

    def fake_post(url: str, **kwargs: Any) -> DummyResponse:
        calls.append({"url": url, **kwargs})
        if url.endswith("/api/process/start"):
            return DummyResponse(200, {"exit_code": 0, "stdout": "", "stderr": ""})
        if url.endswith("/api/fs/upload"):
            assert kwargs["params"] == {"path": "/remote"}
            assert kwargs["headers"]["X-Jwt-Token"] == "jwt"
            assert kwargs["verify"] is False
            return DummyResponse(200, {"ok": True})
        if url.endswith("/api/fs/download"):
            assert kwargs["json"] == {"path": "/remote/out.txt"}
            assert kwargs["verify"] is False
            return DummyResponse(200, chunks=[b"hello", b" world"])
        raise AssertionError(f"unexpected POST {url}")

    monkeypatch.setattr(terminal_provider.requests, "post", fake_post)

    provider = terminal_provider._StarGazeTerminalProvider(
        connection={
            "creator_jwt_token": "jwt",
            "session_id": "session-123",
            "setup_netrc": False,
            "persist_config": False,
        }
    )
    handle = await provider.create(SandboxSpec(image="registry.example/app:latest"))
    source = tmp_path / "in.txt"
    source.write_text("hello", encoding="utf-8")

    await provider.upload_file(handle, source, "/remote/in.txt")
    target = tmp_path / "out.txt"
    await provider.download_file(handle, "/remote/out.txt", target)

    assert target.read_bytes() == b"hello world"
    assert any(call["url"].endswith("/api/fs/upload") for call in calls)
    assert any(call["url"].endswith("/api/fs/download") for call in calls)


async def test_exec_long_uses_nonblocking_process_api(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    progress_messages: list[str] = []

    def event(name: str, payload: dict[str, Any]) -> bytes:
        return f"event: {name}\ndata: {json.dumps(payload)}\n\n".encode()

    def fake_post(url: str, **kwargs: Any) -> DummyResponse:
        calls.append({"url": url, **kwargs})
        if url.endswith("/api/process/start") and kwargs["json"].get("blocking") is False:
            assert kwargs["stream"] is True
            assert kwargs["json"]["timeout"] == 30
            return DummyResponse(
                200,
                chunks=[
                    event("process.start", {"pid": 123}),
                    event("process.data", {"stdout": "streamed"}),
                    event("process.exit", {"pid": 123, "exit_code": 0}),
                ],
            )
        if url.endswith("/api/process/start"):
            script = kwargs["json"]["command"]["args"][1]
            if "cat /tmp/xgym_terminal_long_command_" in script:
                if script.endswith(".stdout 2>/dev/null || true"):
                    return DummyResponse(200, {"exit_code": 0, "stdout": "file stdout", "stderr": ""})
                if script.endswith(".stderr 2>/dev/null || true"):
                    return DummyResponse(200, {"exit_code": 0, "stdout": "", "stderr": ""})
                if script.endswith(".status 2>/dev/null || true"):
                    return DummyResponse(200, {"exit_code": 0, "stdout": "0\n", "stderr": ""})
            return DummyResponse(200, {"exit_code": 0, "stdout": "", "stderr": ""})
        raise AssertionError(f"unexpected POST {url}")

    monkeypatch.setattr(terminal_provider.requests, "post", fake_post)
    provider = terminal_provider._StarGazeTerminalProvider(
        connection={"creator_jwt_token": "jwt", "session_id": "session-123", "setup_netrc": False}
    )
    handle = await provider.create(SandboxSpec())

    result = await provider.exec_long(
        handle,
        "printf ok",
        cwd="/repo",
        timeout_s=30,
        progress_callback=progress_messages.append,
    )

    assert result.return_code == 0
    assert result.stdout == "file stdout"
    assert "streamed" in progress_messages
    start_call = calls[0]
    assert start_call["json"]["blocking"] is False
    assert "cd /repo" in start_call["json"]["command"]["args"][1]


async def test_status_uses_echo_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post(url: str, **kwargs: Any) -> DummyResponse:
        return DummyResponse(200, {"exit_code": 0, "stdout": "hello\n", "stderr": ""})

    monkeypatch.setattr(terminal_provider.requests, "post", fake_post)
    provider = terminal_provider._StarGazeTerminalProvider(
        connection={
            "creator_jwt_token": "jwt",
            "session_id": "session-123",
            "setup_netrc": False,
            "persist_config": False,
        }
    )
    handle = await provider.create(SandboxSpec())
    assert await provider.status(handle) is SandboxStatus.RUNNING


async def test_create_retries_transient_http_response(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = {"post": 0, "sleep": []}

    def fake_post(url: str, **kwargs: Any) -> DummyResponse:
        calls["post"] += 1
        if calls["post"] == 1:
            return DummyResponse(500, {"code": 500, "message": "500 Internal Server Error"})
        return DummyResponse(200, {"code": 0, "data": {"session_id": "session-123"}})

    async def fake_sleep(delay: float) -> None:
        calls["sleep"].append(delay)

    monkeypatch.setattr(terminal_provider.requests, "post", fake_post)
    monkeypatch.setattr(terminal_provider.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(terminal_provider, "_get_codebase_jwt", lambda **kwargs: "")

    provider = terminal_provider._StarGazeTerminalProvider(
        connection={
            "creator_jwt_token": "jwt",
            "setup_netrc": False,
            "refresh_jwt": False,
            "config_dir": str(tmp_path),
        },
        operations={"create_retries": 1},
    )
    handle = await provider.create(SandboxSpec(ttl_s=60))

    assert handle.sandbox_id == "session-123"
    assert calls["post"] == 2
    assert calls["sleep"]


async def test_create_retries_transient_request_exception(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = {"post": 0, "sleep": []}

    def fake_post(url: str, **kwargs: Any) -> DummyResponse:
        calls["post"] += 1
        if calls["post"] == 1:
            raise requests.ConnectionError("connection reset by peer")
        return DummyResponse(200, {"code": 0, "data": {"session_id": "session-123"}})

    async def fake_sleep(delay: float) -> None:
        calls["sleep"].append(delay)

    monkeypatch.setattr(terminal_provider.requests, "post", fake_post)
    monkeypatch.setattr(terminal_provider.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(terminal_provider, "_get_codebase_jwt", lambda **kwargs: "")

    provider = terminal_provider._StarGazeTerminalProvider(
        connection={
            "creator_jwt_token": "jwt",
            "setup_netrc": False,
            "refresh_jwt": False,
            "config_dir": str(tmp_path),
        },
        operations={"create_retries": 1},
    )
    handle = await provider.create(SandboxSpec(ttl_s=60))

    assert handle.sandbox_id == "session-123"
    assert calls["post"] == 2
    assert calls["sleep"]


async def test_create_does_not_retry_non_transient_response(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = {"post": 0, "sleep": []}

    def fake_post(url: str, **kwargs: Any) -> DummyResponse:
        calls["post"] += 1
        return DummyResponse(400, {"code": 400, "message": "bad request"})

    monkeypatch.setattr(terminal_provider.requests, "post", fake_post)
    monkeypatch.setattr(terminal_provider.asyncio, "sleep", lambda delay: calls["sleep"].append(delay))

    provider = terminal_provider._StarGazeTerminalProvider(
        connection={
            "creator_jwt_token": "jwt",
            "setup_netrc": False,
            "refresh_jwt": False,
            "config_dir": str(tmp_path),
        },
        operations={"create_retries": 3},
    )

    with pytest.raises(terminal_provider.TerminalCreateError, match="HTTP 400"):
        await provider.create(SandboxSpec(ttl_s=60))
    assert calls == {"post": 1, "sleep": []}


async def test_create_retries_transient_user_jwt_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = {"token": 0, "post": 0, "sleep": []}

    class FakeServiceAccountManager:
        def get_user_token(self, username: str, ignore_cache: bool = False) -> terminal_provider.UserToken:
            calls["token"] += 1
            assert username == "dev.user"
            assert ignore_cache is True
            if calls["token"] == 1:
                raise terminal_provider.TokenFetchError("connection aborted: EPIPE")
            return terminal_provider.UserToken(
                access_token="fresh-user-jwt",
                expires_at=terminal_provider.datetime.now() + terminal_provider.timedelta(seconds=3600),
            )

    def fake_post(url: str, **kwargs: Any) -> DummyResponse:
        calls["post"] += 1
        assert kwargs["headers"]["X-Jwt-Token"] == "fresh-user-jwt"
        return DummyResponse(200, {"code": 0, "data": {"session_id": "session-123"}})

    monkeypatch.setattr(terminal_provider, "get_service_account_manager", lambda region: FakeServiceAccountManager())
    monkeypatch.setattr(terminal_provider.requests, "post", fake_post)
    monkeypatch.setattr(terminal_provider.time, "sleep", lambda delay: calls["sleep"].append(delay))
    monkeypatch.setattr(terminal_provider, "_get_codebase_jwt", lambda **kwargs: "")

    provider = terminal_provider._StarGazeTerminalProvider(
        connection={
            "creator_email": "dev.user@bytedance.com",
            "setup_netrc": False,
            "refresh_jwt": False,
            "config_dir": str(tmp_path),
        }
    )
    handle = await provider.create(SandboxSpec(ttl_s=60))

    assert handle.sandbox_id == "session-123"
    assert calls["token"] == 2
    assert calls["post"] == 1
    assert calls["sleep"]


async def test_create_does_not_create_session_when_user_not_authorized(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls = {"token": 0, "post": 0}

    class FakeServiceAccountManager:
        def get_user_token(self, username: str, ignore_cache: bool = False) -> terminal_provider.UserToken:
            calls["token"] += 1
            raise terminal_provider.UserNotAuthorizedError(username)

    def fake_post(url: str, **kwargs: Any) -> DummyResponse:
        calls["post"] += 1
        raise AssertionError("sandbox session should not be created without JWT")

    monkeypatch.setattr(terminal_provider, "get_service_account_manager", lambda region: FakeServiceAccountManager())
    monkeypatch.setattr(terminal_provider.requests, "post", fake_post)

    provider = terminal_provider._StarGazeTerminalProvider(
        connection={
            "creator_email": "dev.user@bytedance.com",
            "setup_netrc": False,
            "refresh_jwt": False,
            "config_dir": str(tmp_path),
        }
    )

    with pytest.raises(terminal_provider.TerminalCreateError, match="failed to get user JWT"):
        await provider.create(SandboxSpec(ttl_s=60))
    assert calls == {"token": 1, "post": 0}


def test_session_create_admission_times_out_when_all_slots_taken(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("STARGAZE_TERMINAL_SESSION_CREATE_MAX_PARALLEL", "1")
    monkeypatch.setenv("STARGAZE_TERMINAL_SESSION_CREATE_LOCK_DIR", str(tmp_path))
    monkeypatch.setenv("STARGAZE_TERMINAL_SESSION_CREATE_LOCK_TIMEOUT_SECONDS", "1")
    monkeypatch.setenv("STARGAZE_TERMINAL_SESSION_CREATE_STALE_LOCK_SECONDS", "9999")
    monkeypatch.setattr(terminal_provider.time, "sleep", lambda delay: None)

    admission = terminal_provider._SessionCreateAdmission(group="control|sandbox")
    with admission:
        with pytest.raises(TimeoutError):
            with terminal_provider._SessionCreateAdmission(group="control|sandbox"):
                pass


def test_process_start_admission_times_out_when_all_slots_taken(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("STARGAZE_TERMINAL_PROCESS_START_MAX_PARALLEL", "1")
    monkeypatch.setenv("STARGAZE_TERMINAL_PROCESS_START_LOCK_DIR", str(tmp_path))
    monkeypatch.setenv("STARGAZE_TERMINAL_PROCESS_START_LOCK_TIMEOUT_SECONDS", "1")
    monkeypatch.setenv("STARGAZE_TERMINAL_PROCESS_START_STALE_LOCK_SECONDS", "9999")
    monkeypatch.setattr(terminal_provider.time, "sleep", lambda delay: None)

    admission = terminal_provider._ProcessStartAdmission(group="control|sandbox|process_start")
    with admission:
        with pytest.raises(TimeoutError):
            with terminal_provider._ProcessStartAdmission(group="control|sandbox|process_start"):
                pass


async def test_exec_returns_sandbox_error_when_process_start_admission_times_out(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = {"post": 0}

    def fake_post(url: str, **kwargs: Any) -> DummyResponse:
        calls["post"] += 1
        raise AssertionError("process/start should not be called without admission")

    monkeypatch.setenv("STARGAZE_TERMINAL_PROCESS_START_MAX_PARALLEL", "1")
    monkeypatch.setenv("STARGAZE_TERMINAL_PROCESS_START_LOCK_DIR", str(tmp_path))
    monkeypatch.setenv("STARGAZE_TERMINAL_PROCESS_START_LOCK_TIMEOUT_SECONDS", "1")
    monkeypatch.setenv("STARGAZE_TERMINAL_PROCESS_START_STALE_LOCK_SECONDS", "9999")
    monkeypatch.setattr(terminal_provider.time, "sleep", lambda delay: None)
    monkeypatch.setattr(terminal_provider.requests, "post", fake_post)

    provider = terminal_provider._StarGazeTerminalProvider(
        connection={
            "creator_jwt_token": "jwt",
            "session_id": "session-123",
            "setup_netrc": False,
            "persist_config": False,
        }
    )
    handle = await provider.create(SandboxSpec())

    with terminal_provider._ProcessStartAdmission(group=provider._process_start_admission_group):
        result = await provider.exec(handle, "printf ok", timeout_s=10)

    assert result.return_code == terminal_provider.TERMINAL_RUNTIME_RETURN_CODE
    assert result.error_type == "sandbox"
    assert "process-start admission lock" in (result.stderr or "")
    assert calls["post"] == 0


def test_service_account_get_token_retries_transient_response(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"get": 0, "sleep": []}
    cfg = terminal_provider.RegionConfig(
        instance_link="link",
        terminal_sandbox_url="http://control",
        _terminal_sandbox_domain=terminal_provider.EnvConfigStr(tce="sandbox.tce", dev="sandbox.dev"),
        terminal_sandbox_id="sandbox-id",
        _oauth_host=terminal_provider.EnvConfigStr(tce="oauth.tce", dev="oauth.dev"),
        oauth_serv_account="codewise_service",
        oauth_serv_token="serv-token",
        oauth_client_id="client-id",
        oauth_client_secret="client-secret",
    )

    def fake_get(url: str, **kwargs: Any) -> DummyResponse:
        calls["get"] += 1
        if calls["get"] == 1:
            return DummyResponse(503, {"message": "upstream unavailable"})
        return DummyResponse(200, {}, headers={"X-Jwt-Token": "service-jwt"})

    monkeypatch.setattr(terminal_provider.requests, "get", fake_get)
    monkeypatch.setattr(terminal_provider.time, "sleep", lambda delay: calls["sleep"].append(delay))

    manager = terminal_provider.ServiceAccountManager(cfg)

    assert manager.get_token() == "service-jwt"
    assert calls["get"] == 2
    assert calls["sleep"]


def test_service_account_get_user_token_retries_transient_post(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"post": 0, "sleep": []}
    cfg = terminal_provider.RegionConfig(
        instance_link="link",
        terminal_sandbox_url="http://control",
        _terminal_sandbox_domain=terminal_provider.EnvConfigStr(tce="sandbox.tce", dev="sandbox.dev"),
        terminal_sandbox_id="sandbox-id",
        _oauth_host=terminal_provider.EnvConfigStr(tce="oauth.tce", dev="oauth.dev"),
        oauth_serv_account="codewise_service",
        oauth_serv_token="serv-token",
        oauth_client_id="client-id",
        oauth_client_secret="client-secret",
    )
    manager = terminal_provider.ServiceAccountManager(cfg)
    manager._cached_token = "service-jwt"
    manager._token_fetched_at = terminal_provider.time.monotonic()

    def fake_post(url: str, **kwargs: Any) -> DummyResponse:
        calls["post"] += 1
        if calls["post"] == 1:
            return DummyResponse(502, {"message": "bad gateway"})
        return DummyResponse(200, {"access_token": "user-jwt", "expires_in": 3600})

    monkeypatch.setattr(terminal_provider.requests, "post", fake_post)
    monkeypatch.setattr(terminal_provider.time, "sleep", lambda delay: calls["sleep"].append(delay))

    token = manager.get_user_token("dev.user", ignore_cache=True)

    assert token.access_token == "user-jwt"
    assert calls["post"] == 2
    assert calls["sleep"]


async def test_exec_refreshes_user_jwt_once_on_session_403(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"post": 0}
    fake_manager = FakeUserTokenManager()

    def fake_get_service_account_manager(region: str):
        assert region == "boei18n"
        return fake_manager

    def fake_post(url: str, **kwargs: Any) -> DummyResponse:
        calls["post"] += 1
        if calls["post"] == 1:
            return DummyResponse(403, {"message": "forbidden"})
        return DummyResponse(200, {"exit_code": 0, "stdout": "ok", "stderr": ""})

    monkeypatch.setattr(terminal_provider, "get_service_account_manager", fake_get_service_account_manager)
    monkeypatch.setattr(terminal_provider.requests, "post", fake_post)

    provider = terminal_provider._StarGazeTerminalProvider(
        connection={
            "creator_email": "dev.user@example.com",
            "creator_jwt_token": "stale-jwt",
            "session_id": "session-123",
            "setup_netrc": False,
            "persist_config": False,
        },
        operations={"exec_request_extra_timeout_s": 1},
    )
    handle = await provider.create(SandboxSpec())
    result = await provider.exec(handle, "printf ok", timeout_s=10)

    assert result.return_code == 0
    assert result.stdout == "ok"
    assert calls["post"] == 2
    assert fake_manager.calls == [{"username": "dev.user", "ignore_cache": True}]


async def test_exec_retries_transient_session_503(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"post": 0, "sleep": []}

    def fake_post(url: str, **kwargs: Any) -> DummyResponse:
        calls["post"] += 1
        if calls["post"] == 1:
            return DummyResponse(503, {"message": "Service Unavailable"})
        return DummyResponse(200, {"exit_code": 0, "stdout": "ok", "stderr": ""})

    monkeypatch.setattr(terminal_provider.requests, "post", fake_post)
    monkeypatch.setattr(terminal_provider.time, "sleep", lambda delay: calls["sleep"].append(delay))

    provider = terminal_provider._StarGazeTerminalProvider(
        connection={
            "creator_jwt_token": "jwt",
            "session_id": "session-123",
            "setup_netrc": False,
            "persist_config": False,
        },
        operations={"exec_request_extra_timeout_s": 1},
    )
    handle = await provider.create(SandboxSpec())
    result = await provider.exec(handle, "printf ok", timeout_s=10)

    assert result.return_code == 0
    assert result.stdout == "ok"
    assert calls["post"] == 2
    assert calls["sleep"]
