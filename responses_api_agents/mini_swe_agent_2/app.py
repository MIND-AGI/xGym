# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import hashlib
import json
import os
import shlex
import threading
import time
import traceback
from asyncio import Semaphore
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Literal, Optional, cast
from uuid import uuid4

import ray
import yaml
from fastapi import Body, FastAPI
from minisweagent.config import builtin_config_dir, get_config_path
from pydantic import ConfigDict

from nemo_gym.base_resources_server import (
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
)
from nemo_gym.base_responses_api_agent import (
    BaseResponsesAPIAgentConfig,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.global_config import TASK_INDEX_KEY_NAME
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.reward_profile import compute_pass_majority_metrics, highest_k_metrics
from nemo_gym.sandbox import Sandbox, SandboxResources, SandboxSpec, resolve_provider_config, resolve_provider_metadata
from nemo_gym.server_utils import (
    ServerClient,
    get_first_server_config_dict,
)


OPENSANDBOX_PROVIDER_NAME = "opensandbox"
OPENSANDBOX_API_KEY_ENV = "OPENSANDBOX_API_KEY"  # pragma: allowlist secret
PROGRESS_STATE_FILENAME = "progress_state.json"
PROGRESS_EVENTS_FILENAME = "progress_events.jsonl"
OPENCODE_SESSION_LOG_FILENAME = "opencode_session.log"


def _utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class _StageProgressRecorder:
    """Persist StarGaze-style per-instance progress for long SWE runs."""

    def __init__(self, *, instance_id: str, instance_dir: Path) -> None:
        self.instance_id = instance_id
        self.instance_dir = instance_dir
        self.state_path = instance_dir / PROGRESS_STATE_FILENAME
        self.events_path = instance_dir / PROGRESS_EVENTS_FILENAME
        self.opencode_session_log_path = instance_dir / OPENCODE_SESSION_LOG_FILENAME
        self._lock = threading.RLock()
        self._state: dict[str, Any] = {
            "instance_id": instance_id,
            "status": "running",
            "current_stage": None,
            "current_stage_started_at": None,
            "completed_stages": [],
            "failed_stage": None,
            "error": None,
            "started_at": _utc_timestamp(),
            "updated_at": _utc_timestamp(),
            "progress_state_path": str(self.state_path),
            "progress_events_path": str(self.events_path),
            "opencode_session_log_path": str(self.opencode_session_log_path),
            "opencode_session_log_bytes": 0,
            "opencode_session_log_lines": 0,
            "opencode_session_log_tail": "",
            "last_opencode_session_id": None,
            "recent_events": [],
        }
        self.instance_dir.mkdir(parents=True, exist_ok=True)
        self.events_path.write_text("", encoding="utf-8")
        self.opencode_session_log_path.write_text("", encoding="utf-8")
        self._record_event("instance_start", status="running")

    @property
    def current_stage(self) -> str | None:
        stage = self._state.get("current_stage")
        return str(stage) if stage else None

    def paths(self) -> dict[str, str]:
        return {
            "progress_state_path": str(self.state_path),
            "progress_events_path": str(self.events_path),
            "opencode_session_log_path": str(self.opencode_session_log_path),
        }

    def append_opencode_session_log(self, chunk: str) -> None:
        if not chunk:
            return
        with self._lock:
            with self.opencode_session_log_path.open("a", encoding="utf-8") as fh:
                fh.write(chunk)
                if not chunk.endswith("\n"):
                    fh.write("\n")
            text = self.opencode_session_log_path.read_text(encoding="utf-8", errors="replace")
            self._update_opencode_session_log_state(text, session_id=_extract_opencode_session_id(chunk))

    def write_opencode_session_log(self, text: str, *, session_id: str | None = None) -> None:
        with self._lock:
            self.opencode_session_log_path.write_text(text, encoding="utf-8")
            self._update_opencode_session_log_state(text, session_id=session_id)

    def _update_opencode_session_log_state(self, text: str, *, session_id: str | None = None) -> None:
        lines = text.splitlines()
        self._state["opencode_session_log_bytes"] = len(text.encode("utf-8"))
        self._state["opencode_session_log_lines"] = len(lines)
        self._state["opencode_session_log_tail"] = "\n".join(lines[-20:])
        if session_id:
            self._state["last_opencode_session_id"] = session_id
        self._record_event(
            "opencode_session_log",
            stage="agent_run",
            status="running",
            message=f"session_log_bytes={self._state['opencode_session_log_bytes']} lines={len(lines)}",
        )

    def stage_start(self, stage: str, message: str | None = None) -> None:
        self._state["status"] = "running"
        self._state["current_stage"] = stage
        self._state["current_stage_started_at"] = _utc_timestamp()
        self._record_event("stage_start", stage=stage, status="running", message=message)

    def stage_update(self, stage: str, message: str, *, status: str = "running") -> None:
        self._state["status"] = status
        self._state["current_stage"] = stage
        self._record_event("stage_update", stage=stage, status=status, message=message)

    def stage_finish(self, stage: str, *, status: str = "completed", message: str | None = None) -> None:
        completed = self._state.setdefault("completed_stages", [])
        if status == "completed" and stage not in completed:
            completed.append(stage)
        if status != "completed":
            self._state["failed_stage"] = stage
            self._state["error"] = message
        if self._state.get("current_stage") == stage:
            self._state["current_stage"] = None
            self._state["current_stage_started_at"] = None
        self._record_event("stage_finish", stage=stage, status=status, message=message)

    def fail(self, error: str, *, stage: str | None = None) -> None:
        failed_stage = stage or self.current_stage
        self._state["status"] = "failed"
        self._state["failed_stage"] = failed_stage
        self._state["error"] = error
        self._state["completed_at"] = _utc_timestamp()
        self._record_event("instance_end", stage=failed_stage, status="failed", message=error)

    def finish(self, status: str) -> None:
        self._state["status"] = status
        self._state["current_stage"] = None
        self._state["current_stage_started_at"] = None
        self._state["completed_at"] = _utc_timestamp()
        self._record_event("instance_end", status=status)

    def _record_event(
        self,
        event: str,
        *,
        stage: str | None = None,
        status: str | None = None,
        message: str | None = None,
    ) -> None:
        with self._lock:
            timestamp = _utc_timestamp()
            record = {
                "timestamp": timestamp,
                "event": event,
                "instance_id": self.instance_id,
                "stage": stage,
                "status": status,
                "message": message,
            }
            self._state["updated_at"] = timestamp
            recent_events = self._state.setdefault("recent_events", [])
            recent_events.append(record)
            del recent_events[:-20]
            with self.events_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = self.state_path.with_name(
                f"{self.state_path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
            )
            temp_path.write_text(json.dumps(self._state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            temp_path.replace(self.state_path)


class _ProgressHeartbeat:
    def __init__(
        self,
        *,
        progress: _StageProgressRecorder,
        stage: str,
        message: str,
        interval_s: float = 60.0,
    ) -> None:
        self._progress = progress
        self._stage = stage
        self._message = message
        self._interval_s = interval_s
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> "_ProgressHeartbeat":
        if self._thread is not None:
            return self
        self._thread = threading.Thread(target=self._run, name=f"xgym-progress-heartbeat-{self._stage}", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        while not self._stop_event.wait(self._interval_s):
            self._progress.stage_update(self._stage, self._message)


class MiniSWEAgentConfig(BaseResponsesAPIAgentConfig):
    model_server: ModelServerRef
    env: Literal["sandbox"]
    concurrency: int
    # A sandbox name resolved from a separate provider config (e.g. "sandbox"),
    # or an inline single-key provider mapping ({provider_name: {...}}).
    sandbox_provider: Optional[str | dict[str, Any]] = None
    sandbox_spec: Optional[dict[str, Any]] = None
    sandbox_environment_kwargs: Optional[dict[str, Any]] = None
    run_golden: bool = False
    step_timeout: int = 600
    eval_timeout: int = 1800
    skip_if_exists: bool = False
    step_limit: int = 250
    tool_choice: Optional[str | dict[str, Any]] = None
    sandbox_resource_profiles: Optional[list[dict[str, str]]] = None
    agent_framework: Literal["mini_swe", "opencode"] = "mini_swe"
    image_map_path: Optional[str] = None
    require_image_map: bool = False
    opencode_provider_id: str = "litellm"
    opencode_provider_api_url: Optional[str] = None
    opencode_api_key_env: str = "LITELLM_KEY"
    opencode_api_key: str = "dummy_key"
    opencode_npm_package: str = "opencode-ai@1.14.50"
    opencode_setup_timeout: int = 300
    opencode_idle_timeout: int = 900
    opencode_command_timeout: Optional[int] = None
    opencode_prompt_mode: Literal["inline", "file"] = "inline"
    opencode_infra_retries: int = 0
    opencode_infra_retry_delay_s: float = 5.0


class MiniSWEAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class MiniSWEAgentVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")


class MiniSWEAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


@ray.remote(scheduling_strategy="SPREAD")
def runner_ray_remote(runner: Callable, params: dict[str, Any]) -> Any:
    return runner(**params)


def _json_dict_from_metadata(value: Any, *, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
    raise ValueError(f"responses_create_params.metadata.{field_name} must be a JSON object")


def _responses_create_params_to_model_kwargs(
    params: dict[str, Any],
    *,
    default_tool_choice: Any = None,
) -> dict[str, Any]:
    """Convert Gym Responses API rollout params into mini-swe-agent chat-completions kwargs."""
    model_kwargs: dict[str, Any] = {}
    for key in ("temperature", "top_p", "top_logprobs", "parallel_tool_calls"):
        value = params.get(key)
        if value is not None:
            model_kwargs[key] = value

    max_output_tokens = params.get("max_output_tokens")
    if max_output_tokens is not None:
        model_kwargs["max_tokens"] = max_output_tokens

    metadata = params.get("metadata") or {}
    extra_body = _json_dict_from_metadata(metadata.get("extra_body"), field_name="extra_body")
    chat_template_kwargs = _json_dict_from_metadata(
        metadata.get("chat_template_kwargs"),
        field_name="chat_template_kwargs",
    )
    if chat_template_kwargs:
        extra_body["chat_template_kwargs"] = chat_template_kwargs
    if extra_body:
        model_kwargs["extra_body"] = extra_body

    tool_choice = default_tool_choice if default_tool_choice is not None else params.get("tool_choice")
    if tool_choice == "bash":
        model_kwargs["tool_choice"] = _bash_tool_choice()
    elif tool_choice is not None:
        model_kwargs["tool_choice"] = tool_choice

    return model_kwargs


def _opensandbox_connection(provider: dict[str, Any] | None) -> dict[str, Any] | None:
    if provider is None:
        return None
    provider_config = provider.get(OPENSANDBOX_PROVIDER_NAME)
    if not isinstance(provider_config, dict):
        return None
    connection = provider_config.get("connection")
    if not isinstance(connection, dict):
        return None
    return connection


def _sandbox_provider_for_config_dump(provider: dict[str, Any]) -> dict[str, Any]:
    provider_for_disk = deepcopy(provider)
    connection = _opensandbox_connection(provider_for_disk)
    if connection is not None:
        connection.pop("api_key", None)
    return provider_for_disk


def _sandbox_runtime_env(provider: dict[str, Any] | None) -> dict[str, Any]:
    runtime_env: dict[str, Any] = {}
    connection = _opensandbox_connection(provider)
    if connection is None:
        return runtime_env
    api_key = connection.get("api_key")
    if api_key:
        runtime_env["env_vars"] = {OPENSANDBOX_API_KEY_ENV: str(api_key)}
    return runtime_env


def _restore_sandbox_provider_secrets(config: dict[str, Any]) -> None:
    provider = config.get("environment", {}).get("provider")
    connection = _opensandbox_connection(provider if isinstance(provider, dict) else None)
    if connection is None or connection.get("api_key"):
        return
    api_key = os.getenv(OPENSANDBOX_API_KEY_ENV)
    if api_key:
        connection["api_key"] = api_key


def _bash_tool_choice() -> dict[str, Any]:
    return {"type": "function", "function": {"name": "bash"}}


def _sandbox_spec_for_instance(
    spec: dict[str, Any] | None,
    *,
    resource_profiles: list[dict[str, Any]] | None,
    instance_id: str,
) -> dict[str, Any]:
    instance_spec = dict(spec or {})
    if not resource_profiles:
        return instance_spec

    resources = dict(instance_spec.get("resources") or {})
    digest = hashlib.sha256(instance_id.encode("utf-8")).digest()
    profile = resource_profiles[int.from_bytes(digest[:4], "big") % len(resource_profiles)]
    resources.update(profile)
    instance_spec["resources"] = resources
    return instance_spec


def _swebench_config_path() -> Path:
    for candidate in (
        builtin_config_dir / "extra" / "swebench.yaml",
        builtin_config_dir / "benchmarks" / "swebench.yaml",
    ):
        if candidate.exists():
            return candidate
    return builtin_config_dir / "extra" / "swebench.yaml"


def _load_image_map(path: str | Path | None) -> dict[str, str]:
    if not path:
        return {}
    map_path = Path(path).expanduser()
    payload = json.loads(map_path.read_text(encoding="utf-8"))
    images = payload.get("images", payload)
    if not isinstance(images, dict):
        raise ValueError(f"image map must be a JSON object or contain an 'images' object: {map_path}")

    normalized_images: dict[str, str] = {}
    for instance_id, image in images.items():
        if not isinstance(instance_id, str) or not isinstance(image, str):
            raise ValueError(f"image map entries must be string:string pairs: {map_path}")
        normalized_images[instance_id] = image
    return normalized_images


def _swebench_image_name(
    instance: dict[str, Any],
    subset: str,
    *,
    image_map: dict[str, str] | None = None,
    require_image_map: bool = False,
) -> str:
    instance_id = str(instance["instance_id"])
    explicit_image = instance.get("sandbox_image") or instance.get("swebench_image")
    if explicit_image:
        return str(explicit_image)

    image_map = image_map or {}
    mapped_image = image_map.get(instance_id)
    if mapped_image:
        return mapped_image
    if require_image_map:
        raise ValueError(f"missing sandbox image for {instance_id!r} in configured image map")

    image_name = instance.get("image_name")
    if image_name:
        return str(image_name)

    if subset == "verified":
        docker_compatible_id = instance_id.replace("__", "_1776_")
        return f"docker.io/swebench/sweb.eval.x86_64.{docker_compatible_id}:latest".lower()

    docker_compatible_id = instance_id.replace("__", "_s_")
    return f"docker.io/xingyaoww/sweb.eval.x86_64.{docker_compatible_id}:latest".lower()


def _message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return "" if content is None else str(content)


def _strip_extra(item: Any) -> dict[str, Any]:
    if hasattr(item, "model_dump"):
        item = item.model_dump()
    if not isinstance(item, dict):
        return {"type": "message", "role": "user", "content": str(item)}
    return {key: value for key, value in item.items() if key != "extra"}


def _split_trajectory_for_responses(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    input_messages: list[dict[str, Any]] = []
    output_items: list[dict[str, Any]] = []
    raw_responses: list[dict[str, Any]] = []
    in_initial_prompt = True

    for message in messages:
        role = message.get("role")
        if in_initial_prompt and role in {"system", "user"}:
            input_messages.append(
                {"type": "message", "role": role, "content": _message_content_to_text(message.get("content"))}
            )
            continue

        in_initial_prompt = False
        if message.get("object") == "response":
            response = _strip_extra(message)
            raw_responses.append(response)
            output_items.extend(_strip_extra(item) for item in response.get("output", []))
        elif role == "assistant":
            content = _message_content_to_text(message.get("content"))
            if content:
                output_items.append(
                    {
                        "id": message.get("id") or f"msg_{uuid4()}",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": content, "annotations": []}],
                    }
                )
            for tool_call in message.get("tool_calls") or []:
                function = tool_call.get("function") or {}
                output_items.append(
                    {
                        "id": tool_call.get("id") or f"fc_{uuid4()}",
                        "type": "function_call",
                        "name": function.get("name") or tool_call.get("name") or "",
                        "call_id": tool_call.get("id") or tool_call.get("call_id") or "",
                        "arguments": function.get("arguments") or tool_call.get("arguments") or "{}",
                    }
                )
        elif role == "tool":
            output_items.append(
                {
                    "type": "function_call_output",
                    "call_id": message.get("tool_call_id") or message.get("call_id") or "",
                    "output": _message_content_to_text(message.get("content")),
                }
            )
        elif message.get("type") == "function_call_output":
            output_items.append(_strip_extra(message))

    return input_messages, output_items, raw_responses


def _default_response_object() -> dict[str, Any]:
    return {
        "id": f"resp_{str(uuid4())}",
        "created_at": int(time.time()),
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "metadata": {},
        "object": "response",
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "background": False,
        "max_output_tokens": None,
        "max_tool_calls": None,
        "previous_response_id": None,
        "prompt": None,
        "reasoning": {
            "effort": None,
            "generate_summary": None,
            "summary": None,
        },
        "service_tier": "default",
        "status": "completed",
        "text": {"format": {"type": "text"}, "verbosity": "medium"},
        "top_logprobs": 0,
        "truncation": "disabled",
        "usage": {
            "input_tokens": 0,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 0,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 0,
        },
        "user": None,
        "prompt_cache_key": None,
        "safety_identifier": None,
        "store": True,
    }


def _is_resolved(instance_id: str, eval_report: dict[str, Any]) -> bool:
    try:
        if not eval_report:
            return False
        report = eval_report["eval_report"][instance_id]
        resolved = bool(report["resolved"])
        if not report.get("tests_status"):
            return False

        tests_status = report["tests_status"]
        f2f = tests_status.get("FAIL_TO_PASS", {})
        p2p = tests_status.get("PASS_TO_PASS", {})
        total_reported = (
            len(f2f.get("success", []))
            + len(f2f.get("failure", []))
            + len(p2p.get("success", []))
            + len(p2p.get("failure", []))
        )
        return resolved and total_reported > 0
    except Exception as exc:
        print(f"Error in _is_resolved: {exc}", flush=True)
        return False


def _metadata_dict(verify_response: dict[str, Any]) -> dict[str, Any]:
    metadata = verify_response.get("metadata") or {}
    return metadata if isinstance(metadata, dict) else {}


def _eval_report_map(verify_response: dict[str, Any]) -> dict[str, Any]:
    report = _metadata_dict(verify_response).get("eval_report") or {}
    return report if isinstance(report, dict) else {}


def _eval_instance_report(verify_response: dict[str, Any]) -> dict[str, Any]:
    report_map = _eval_report_map(verify_response)
    instance_id = verify_response.get("instance_id") or _metadata_dict(verify_response).get("instance_id")
    if instance_id is not None:
        report = report_map.get(str(instance_id))
        if isinstance(report, dict):
            return report

    for report in report_map.values():
        if isinstance(report, dict) and "resolved" in report:
            return report
    return {}


def _test_status_counts(verify_response: dict[str, Any]) -> dict[str, int]:
    report = _eval_instance_report(verify_response)
    tests_status = report.get("tests_status") if isinstance(report, dict) else None
    if not isinstance(tests_status, dict):
        return {}

    counts: dict[str, int] = {}
    for suite_name, suite_report in tests_status.items():
        if not isinstance(suite_report, dict):
            continue
        prefix = str(suite_name).lower()
        counts[f"{prefix}_success"] = len(suite_report.get("success") or [])
        counts[f"{prefix}_failure"] = len(suite_report.get("failure") or [])
    return counts


def _run_eval_v2(
    *,
    instance: dict[str, Any],
    env: Any,
    model_patch: str,
    instance_dir: Path,
    run_id: str,
    is_golden: bool,
) -> dict[str, Any]:
    from swebench.harness.constants import SWEbenchInstance
    from swebench.harness.docker_build import setup_logger
    from swebench.harness.grading import get_eval_report
    from swebench.harness.test_spec.test_spec import make_test_spec

    swebench_instance = cast(SWEbenchInstance, instance)
    test_spec = make_test_spec(swebench_instance)
    pred = {"instance_id": test_spec.instance_id, "model_patch": model_patch}

    instance_dir.mkdir(parents=True, exist_ok=True)
    log_file = instance_dir / f"run_instance_{run_id}.log"
    report_path = instance_dir / f"report_{run_id}.json"
    patch_file = instance_dir / f"patch_{run_id}.diff"
    patch_file.write_text(model_patch)

    logger = setup_logger(test_spec.instance_id, log_file)
    logger.info(f"DEBUG test_spec {test_spec}")
    logger.info(f"DEBUG eval_script {test_spec.eval_script}")

    if is_golden:
        env.execute(f"cat > patch.diff <<'EOF'\n{model_patch}\n\nEOF")
        env.execute("git status --porcelain")
        env.execute("git apply --check patch.diff")
        env.execute("git apply patch.diff")

    eval_script = test_spec.eval_script.replace("#!/bin/bash", "")
    result = env.execute(eval_script, is_eval=True)
    test_output = result["output"]
    returncode = result["returncode"]
    print(f"[EVAL]{test_spec.instance_id} returncode: {returncode}", flush=True)

    test_output_path = instance_dir / f"test_output_{run_id}.txt"
    test_output_path.write_text(test_output)
    print(f"[EVAL]{test_spec.instance_id} Test output written to {test_output_path}", flush=True)

    report = get_eval_report(
        test_spec=test_spec,
        prediction=pred,
        test_log_path=str(test_output_path),
        include_tests_status=True,
    )
    print(f"[EVAL]{test_spec.instance_id} Result: resolved: {report[test_spec.instance_id]['resolved']}", flush=True)

    report_path.write_text(json.dumps(report, indent=4))
    return {
        "instance_id": test_spec.instance_id,
        "model_patch": model_patch,
        "eval_report": report,
    }


OPENCODE_REMOTE_CONFIG_DIR = "/root/.config/opencode"
OPENCODE_REMOTE_SESSION_DIR = "/root/.local/share/opencode/storage/session"
OPENCODE_NPM_NO_INSPECT_PREFIX = (
    "env -u NODE_OPTIONS NODE_OPTIONS= npm_config_node_options= NODE_INSPECT_RESUME_ON_START=0"
)
OPENCODE_VERIFY_COMMAND = (
    "command -v opencode >/dev/null 2>&1 && echo OPENCODE_OK || { echo OPENCODE_MISSING >&2; exit 1; }"
)
OPENCODE_MUSL_PATCH_COMMAND = r"""
if command -v apk >/dev/null 2>&1; then
    OPENCODE_GLOBAL_ROOT="$(npm root -g 2>/dev/null)"
    OPENCODE_PKG_DIR="$OPENCODE_GLOBAL_ROOT/opencode-ai"
    if [ ! -d "$OPENCODE_PKG_DIR" ]; then
        echo "opencode-ai package not found under npm root" >&2
        exit 1
    fi
    case "$(uname -m)" in
        x86_64|amd64) OPENCODE_ARCH=x64 ;;
        aarch64|arm64) OPENCODE_ARCH=arm64 ;;
        *) echo "unsupported Alpine arch for opencode musl patch: $(uname -m)" >&2; exit 1 ;;
    esac
    OPENCODE_MUSL_BIN="$OPENCODE_PKG_DIR/node_modules/opencode-linux-${OPENCODE_ARCH}-musl/bin/opencode"
    OPENCODE_TARGET="$OPENCODE_PKG_DIR/bin/.opencode"
    if [ ! -x "$OPENCODE_MUSL_BIN" ]; then
        echo "opencode musl binary not found: $OPENCODE_MUSL_BIN" >&2
        exit 1
    fi
    mkdir -p "$(dirname "$OPENCODE_TARGET")"
    cp "$OPENCODE_MUSL_BIN" "$OPENCODE_TARGET"
    chmod 755 "$OPENCODE_TARGET"
fi
""".strip()


def _opencode_api_key_export(params: dict[str, Any]) -> str:
    api_key_env = str(params.get("opencode_api_key_env") or "LITELLM_KEY")
    api_key = str(os.environ.get(api_key_env) or params.get("opencode_api_key") or "dummy_key")
    return f"export {api_key_env}={shlex.quote(api_key)}"


def _redact_opencode_secrets(text: str, params: dict[str, Any]) -> str:
    api_key_env = str(params.get("opencode_api_key_env") or "LITELLM_KEY")
    candidates = [os.environ.get(api_key_env), params.get("opencode_api_key")]
    redacted = text
    for candidate in candidates:
        if isinstance(candidate, str) and candidate:
            redacted = redacted.replace(candidate, "<redacted>")
    return redacted


def _opencode_responses_create_params(params: dict[str, Any]) -> dict[str, Any]:
    raw_params = params.get("responses_create_params") or {}
    if isinstance(raw_params, str):
        parsed = json.loads(raw_params) if raw_params.strip() else {}
        if isinstance(parsed, dict):
            return parsed
        raise ValueError("responses_create_params must be a JSON object")
    if isinstance(raw_params, dict):
        return raw_params
    raise ValueError("responses_create_params must be a mapping or JSON object string")


def _opencode_issue_text(instance: dict[str, Any]) -> str:
    parts = [
        f"Instance ID: {instance['instance_id']}",
        f"Repository: {instance.get('repo') or 'unknown'}",
        f"Base commit: {instance.get('base_commit') or 'unknown'}",
        "",
        "Issue statement:",
        str(instance.get("problem_statement") or "").strip(),
    ]
    return "\n".join(parts).strip() + "\n"


def _opencode_swebench_prompt(
    instance: dict[str, Any], *, repo_path: str = "/testbed", max_diff_lines: int = 400
) -> str:
    issue_text = _opencode_issue_text(instance)
    return f"""You are working inside a checked-out repository at `{repo_path}`.

You will be given an issue statement that describes a bug to fix. Solve the issue by making real code changes in this repository.

<issue>
{issue_text}</issue>

Requirements:
1. Work directly in `{repo_path}`.
2. You must actually modify files in this repository before you finish. Do not only describe a fix. Do not only print a diff. Do not hand-write a patch without changing the working tree.
3. Keep the change as small and focused as possible. Try to stay within about {max_diff_lines} changed diff lines unless the issue clearly needs more.
4. Run the minimal relevant tests or checks when possible.
5. The evaluation harness will collect the real working-tree diff from `{repo_path}` after your final response. Do not hand-write, print, or save a patch file.

Final response contract:
- Output exactly one JSON object, with no Markdown fences or extra prose.
- Use exactly two keys: `explanation` and `failure_reason`.
- `explanation`: non-empty string summarizing root cause, files changed, and validation.
- `failure_reason`: JSON `null` only if you made a real repository fix; otherwise a short string.

Example:
{{
  "explanation": "Fixed the issue by updating src/example.py to handle empty input before normalization. Validated with pytest tests/test_example.py.",
  "failure_reason": null
}}
"""


def _opencode_model_options(params: dict[str, Any]) -> dict[str, Any]:
    responses_params = _opencode_responses_create_params(params)
    options: dict[str, Any] = {}

    field_map = {
        "temperature": "temperature",
        "top_p": "top_p",
        "top_k": "top_k",
        "min_p": "min_p",
        "presence_penalty": "presence_penalty",
        "frequency_penalty": "frequency_penalty",
        "repetition_penalty": "repetition_penalty",
        "max_output_tokens": "max_tokens",
    }
    for source_key, target_key in field_map.items():
        value = responses_params.get(source_key)
        if value is not None:
            options[target_key] = value

    metadata = responses_params.get("metadata") or {}
    extra_body = _json_dict_from_metadata(metadata.get("extra_body"), field_name="extra_body")
    chat_template_kwargs = _json_dict_from_metadata(
        metadata.get("chat_template_kwargs"),
        field_name="chat_template_kwargs",
    )
    if chat_template_kwargs:
        extra_body["chat_template_kwargs"] = chat_template_kwargs

    extra_body_field_map = {
        "top_k": "top_k",
        "min_p": "min_p",
        "presence_penalty": "presence_penalty",
        "frequency_penalty": "frequency_penalty",
        "repetition_penalty": "repetition_penalty",
        "max_tokens": "max_tokens",
        "max_output_tokens": "max_tokens",
    }
    for source_key, value in extra_body.items():
        if value is None:
            continue
        options[extra_body_field_map.get(source_key, source_key)] = value
    if extra_body:
        # Preserve the raw provider payload for adapters that support passing
        # OpenAI-compatible provider extras through a single object.
        options["extraBody"] = extra_body

    reasoning = responses_params.get("reasoning")
    if isinstance(reasoning, dict):
        effort = reasoning.get("effort")
        if effort:
            options["reasoningEffort"] = effort
    reasoning_effort = responses_params.get("reasoning_effort") or params.get("reasoning_effort")
    if reasoning_effort:
        options["reasoningEffort"] = reasoning_effort

    return options


def _opencode_config_payload(params: dict[str, Any]) -> dict[str, Any]:
    provider_id = str(params.get("opencode_provider_id") or "litellm")
    model_name = str(params["policy_model_name"])
    base_url = str(params.get("opencode_provider_api_url") or params["base_url"])
    api_key_env = str(params.get("opencode_api_key_env") or "LITELLM_KEY")
    model_config: dict[str, Any] = {"id": model_name}
    model_options = _opencode_model_options(params)
    if model_options:
        model_config["options"] = model_options
    return {
        "$schema": "https://opencode.ai/config.json",
        "model": f"{provider_id}/{model_name}",
        "permission": "allow",
        "provider": {
            provider_id: {
                "npm": str(params.get("opencode_provider_npm") or "@ai-sdk/openai-compatible"),
                "options": {
                    "baseURL": base_url,
                    "apiKey": f"{{env:{api_key_env}}}",
                },
                "models": {
                    model_name: model_config,
                },
            },
        },
    }


def _opencode_run_command(
    *,
    params: dict[str, Any],
    provider_id: str,
    model_name: str,
    prompt: str,
    prompt_path: str,
) -> str:
    base = (
        f"{_opencode_api_key_export(params)} && "
        f"{OPENCODE_NPM_NO_INSPECT_PREFIX} opencode run --model {shlex.quote(f'{provider_id}/{model_name}')} "
        "--log-level ERROR --format json"
    )
    if str(params.get("opencode_prompt_mode") or "inline") == "file":
        return (
            f"{base} --file {shlex.quote(prompt_path)} -- "
            f"{shlex.quote('Read the attached SWE-bench task prompt and solve it by editing the repository.') }"
        )
    return f"{base} {shlex.quote(prompt)}"


def _opencode_setup_steps(params: dict[str, Any]) -> list[tuple[str, str]]:
    config_content = json.dumps(_opencode_config_payload(params), ensure_ascii=False, indent=2)
    npm_package = shlex.quote(str(params.get("opencode_npm_package") or "opencode-ai@1.14.50"))
    return [
        ("opencode_setup:clean_storage", "rm -rf /root/.local/share/opencode"),
        ("opencode_setup:npm_install", f"{OPENCODE_NPM_NO_INSPECT_PREFIX} npm install -g {npm_package}"),
        ("opencode_setup:musl_patch", OPENCODE_MUSL_PATCH_COMMAND),
        ("opencode_setup:verify", OPENCODE_VERIFY_COMMAND),
        ("opencode_setup:mkdir", f"mkdir -p {OPENCODE_REMOTE_CONFIG_DIR} {OPENCODE_REMOTE_SESSION_DIR}"),
        (
            "opencode_setup:write_config",
            f"cat > {OPENCODE_REMOTE_CONFIG_DIR}/opencode.json <<'EOF'\n{config_content}\nEOF",
        ),
    ]


def _opencode_setup_commands(params: dict[str, Any]) -> list[str]:
    return [command for _, command in _opencode_setup_steps(params)]


def _process_output(stdout: str | None, stderr: str | None) -> str:
    return "\n".join(part for part in (stdout, stderr) if part)


def _sandbox_exec_result(result: Any) -> dict[str, Any]:
    return {
        "output": _process_output(getattr(result, "stdout", None), getattr(result, "stderr", None)),
        "returncode": int(getattr(result, "return_code", 1)),
        "exception_info": str(getattr(result, "error_type", "") or ""),
    }


def _download_sandbox_text(
    sandbox: Any,
    remote_path: str,
    local_path: Path,
    *,
    cwd: str,
    user: str | int | None,
) -> str:
    try:
        sandbox.download(remote_path, local_path)
        return local_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        text = ""
        try:
            result = sandbox.exec(f"cat {shlex.quote(remote_path)}", cwd=cwd, timeout_s=120, user=user)
            if int(getattr(result, "return_code", 1)) == 0:
                text = str(getattr(result, "stdout", "") or "")
        except Exception:
            text = ""
        local_path.write_text(text, encoding="utf-8", errors="replace")
        return text


def _run_sandbox_background_command(
    sandbox: Any,
    command: str,
    *,
    cwd: str,
    timeout: int,
    user: str | int | None,
    instance_dir: Path,
    progress_callback: Callable[[str], None] | None = None,
    poll_interval: float = 30.0,
) -> dict[str, Any]:
    job_id = uuid4().hex
    remote_prefix = f"/tmp/xgym_swe_command_{job_id}"
    remote_script = f"{remote_prefix}.sh"
    remote_stdout = f"{remote_prefix}.stdout"
    remote_stderr = f"{remote_prefix}.stderr"
    remote_status = f"{remote_prefix}.status"
    script_path = instance_dir / f"long_command_{job_id}.sh"
    script_path.write_text(
        "#!/usr/bin/env bash\n"
        "set +e\n"
        f"rm -f {shlex.quote(remote_status)}\n"
        "(\n"
        f"{command}\n"
        ")\n"
        "rc=$?\n"
        f"printf '%s\\n' \"$rc\" > {shlex.quote(remote_status)}\n"
        'exit "$rc"\n',
        encoding="utf-8",
    )
    sandbox.upload(script_path, remote_script)

    start_command = (
        f"rm -f {shlex.quote(remote_stdout)} {shlex.quote(remote_stderr)} {shlex.quote(remote_status)} && "
        f"chmod +x {shlex.quote(remote_script)} && "
        f"(nohup bash {shlex.quote(remote_script)} > {shlex.quote(remote_stdout)} "
        f"2> {shlex.quote(remote_stderr)} < /dev/null & echo $!)"
    )
    start_result = sandbox.exec(start_command, cwd=cwd, timeout_s=60, user=user)
    if int(getattr(start_result, "return_code", 1)) != 0:
        return _sandbox_exec_result(start_result)
    pid = str(getattr(start_result, "stdout", "") or "").strip().splitlines()[-1:]
    pid_text = pid[0].strip() if pid else ""

    deadline = time.monotonic() + timeout
    stdout_seen = 0
    stderr_seen = 0
    while True:
        status_command = (
            f"if test -f {shlex.quote(remote_status)}; then "
            f"printf 'XGYM_COMMAND_DONE\\n'; cat {shlex.quote(remote_status)}; "
            "else printf 'XGYM_COMMAND_RUNNING\\n'; fi"
        )
        status_result = sandbox.exec(status_command, cwd=cwd, timeout_s=60, user=user)
        status_output = str(getattr(status_result, "stdout", "") or "")
        if progress_callback is not None:
            stdout_snapshot = _download_sandbox_text(
                sandbox,
                remote_stdout,
                instance_dir / f"long_command_{job_id}.stdout",
                cwd=cwd,
                user=user,
            )
            stderr_snapshot = _download_sandbox_text(
                sandbox,
                remote_stderr,
                instance_dir / f"long_command_{job_id}.stderr",
                cwd=cwd,
                user=user,
            )
            if len(stdout_snapshot) > stdout_seen:
                progress_callback(stdout_snapshot[stdout_seen:])
                stdout_seen = len(stdout_snapshot)
            if len(stderr_snapshot) > stderr_seen:
                progress_callback(stderr_snapshot[stderr_seen:])
                stderr_seen = len(stderr_snapshot)
        if int(getattr(status_result, "return_code", 1)) == 0 and "XGYM_COMMAND_DONE" in status_output:
            status_lines = [line.strip() for line in status_output.splitlines() if line.strip()]
            exit_code = 1
            for line in reversed(status_lines):
                if line == "XGYM_COMMAND_DONE":
                    continue
                try:
                    exit_code = int(line)
                    break
                except ValueError:
                    continue

            stdout = _download_sandbox_text(
                sandbox,
                remote_stdout,
                instance_dir / f"long_command_{job_id}.stdout",
                cwd=cwd,
                user=user,
            )
            stderr = _download_sandbox_text(
                sandbox,
                remote_stderr,
                instance_dir / f"long_command_{job_id}.stderr",
                cwd=cwd,
                user=user,
            )
            sandbox.exec(
                "rm -f "
                f"{shlex.quote(remote_script)} {shlex.quote(remote_stdout)} "
                f"{shlex.quote(remote_stderr)} {shlex.quote(remote_status)}",
                cwd=cwd,
                timeout_s=60,
                user=user,
            )
            return {"output": _process_output(stdout, stderr), "returncode": exit_code, "exception_info": ""}

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            stdout = _download_sandbox_text(
                sandbox,
                remote_stdout,
                instance_dir / f"long_command_{job_id}.stdout",
                cwd=cwd,
                user=user,
            )
            stderr = _download_sandbox_text(
                sandbox,
                remote_stderr,
                instance_dir / f"long_command_{job_id}.stderr",
                cwd=cwd,
                user=user,
            )
            status = _download_sandbox_text(
                sandbox,
                remote_status,
                instance_dir / f"long_command_{job_id}.status",
                cwd=cwd,
                user=user,
            )
            if pid_text:
                sandbox.exec(f"kill {shlex.quote(pid_text)} >/dev/null 2>&1 || true", cwd=cwd, timeout_s=30, user=user)
            output_parts = [f"terminal sandbox background command timed out after {timeout}s"]
            if status.strip():
                output_parts.append(f"status_file={status.strip()}")
            if stdout.strip():
                output_parts.append("stdout_tail:\n" + stdout[-4000:])
            if stderr.strip():
                output_parts.append("stderr_tail:\n" + stderr[-4000:])
            return {
                "output": "\n".join(output_parts),
                "returncode": 125,
                "exception_info": "timeout",
            }
        time.sleep(min(poll_interval, max(1.0, remaining)))


def _run_checked(env: Any, command: str, *, timeout: int, cwd: str = "/testbed") -> str:
    result = env.execute(command, cwd=cwd, timeout=timeout)
    if int(result.get("returncode", 1)) != 0:
        output = str(result.get("output") or "")
        raise RuntimeError(output or f"command failed with exit code {result.get('returncode')}: {command}")
    return str(result.get("output") or "")


def _opencode_infra_error_category(error_text: str) -> str | None:
    lowered = error_text.lower()
    if any(
        marker in lowered
        for marker in (
            "sandbox not found",
            "session not found",
            "not found by sandbox",
            "terminal sandbox session",
        )
    ):
        return "sandbox_lost"
    if any(marker in lowered for marker in ("timeout", "timed out", "idle timed out", "idle timeout")):
        return "timeout"
    if "opencode completed without producing repository changes" in lowered:
        return "no_repository_changes"
    if any(
        marker in lowered
        for marker in (
            "terminal sandbox long command stream ended without exit status",
            "terminal sandbox long command reconnect failed",
            "terminal sandbox long command did not report a process id",
            "process/connect",
            "process/start",
        )
    ):
        return "terminal_stream_error"
    return None


def _is_retryable_opencode_infra_error(exc: BaseException) -> bool:
    return _opencode_infra_error_category(str(exc)) is not None


def _extract_opencode_session_id(stdout: str) -> str | None:
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("XGYM_OPENCODE_SESSION_SNAPSHOT "):
            for part in line.split():
                if not part.startswith("path="):
                    continue
                session_id = Path(part.removeprefix("path=")).stem
                if session_id:
                    return session_id
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        session_id = event.get("sessionID") or event.get("sessionId") or event.get("session_id")
        if isinstance(session_id, str) and session_id.strip():
            return session_id.strip()
    return None


def _run_opencode_with_sandbox_attempt(**params: Any) -> dict[str, Any]:
    instance = params.get("instance_dict")
    if isinstance(instance, str):
        instance = json.loads(instance)
    if not isinstance(instance, dict):
        raise ValueError("opencode path requires instance_dict")

    instance = dict(instance)
    instance_id = str(params.get("instance_id") or instance["instance_id"])
    instance["instance_id"] = instance_id
    image_map = _load_image_map(params.get("image_map_path"))
    sandbox_image = _swebench_image_name(
        instance,
        params["subset"],
        image_map=image_map,
        require_image_map=bool(params.get("require_image_map")),
    )

    output_dir = Path(params["output"])
    attempt_index = int(params.get("opencode_attempt") or 0)
    instance_dir = output_dir / instance_id
    if attempt_index:
        instance_dir = instance_dir / f"infra_retry_{attempt_index}"
    instance_dir.mkdir(parents=True, exist_ok=True)

    provider = params.get("sandbox_provider")
    if not isinstance(provider, dict):
        raise ValueError("opencode path requires sandbox_provider")
    spec_config = dict(params.get("sandbox_spec") or {})
    environment_kwargs = dict(params.get("sandbox_environment_kwargs") or {})
    env_config = dict(spec_config.pop("env", {}))
    resources = SandboxResources.from_mapping(spec_config.pop("resources", {}))
    progress = _StageProgressRecorder(instance_id=instance_id, instance_dir=instance_dir)
    sandbox: Sandbox | None = None
    session_id = None
    opencode_output = ""
    try:
        progress.stage_start("sandbox_create", f"image={sandbox_image}")
        sandbox = Sandbox(provider).start(
            SandboxSpec(
                image=sandbox_image,
                ttl_s=spec_config.pop("ttl_s", None),
                ready_timeout_s=spec_config.pop("ready_timeout_s", None),
                workdir=spec_config.pop("workdir", environment_kwargs.get("cwd", "/testbed")),
                env=env_config,
                files=spec_config.pop("files", {}),
                metadata={
                    **spec_config.pop("metadata", {}),
                    "nemo_gym_agent": "swe_agents_terminal_opencode",
                    "instance_id": instance_id[:63],
                },
                resources=resources,
                entrypoint=spec_config.pop("entrypoint", None),
                provider_options=spec_config.pop("provider_options", {}),
            )
        )
        progress.stage_finish("sandbox_create")

        class EnvAdapter:
            def __init__(self, sandbox_obj: Sandbox) -> None:
                self._sandbox = sandbox_obj
                self.cwd = str(environment_kwargs.get("cwd") or "/testbed")
                self.step_timeout = int(params["step_timeout"])
                self.eval_timeout = int(params["eval_timeout"])
                self.user = environment_kwargs.get("user", "root")

            def execute(
                self,
                command: str,
                cwd: str = "",
                is_eval: bool = False,
                timeout: int | None = None,
                background: bool = False,
            ) -> dict[str, Any]:
                effective_cwd = cwd or self.cwd
                effective_timeout = timeout or (self.eval_timeout if is_eval else self.step_timeout)
                if background and hasattr(self._sandbox, "exec_long") and not params.get("opencode_poll_logs"):
                    result = self._sandbox.exec_long(
                        command,
                        cwd=effective_cwd,
                        timeout_s=effective_timeout,
                        user=self.user,
                        progress_callback=params.get("opencode_progress_callback"),
                        idle_timeout_s=params.get("opencode_idle_timeout"),
                    )
                    return _sandbox_exec_result(result)
                if background:
                    return _run_sandbox_background_command(
                        self._sandbox,
                        command,
                        cwd=effective_cwd,
                        timeout=effective_timeout,
                        user=self.user,
                        instance_dir=instance_dir,
                        progress_callback=params.get("opencode_progress_callback"),
                    )
                result = self._sandbox.exec(
                    command,
                    cwd=effective_cwd,
                    timeout_s=effective_timeout,
                    user=self.user,
                )
                return _sandbox_exec_result(result)

        env = EnvAdapter(sandbox)
        setup_timeout = int(params.get("opencode_setup_timeout") or 300)
        for stage, command in _opencode_setup_steps(params):
            progress.stage_start(stage, "running command")
            _run_checked(env, command, timeout=setup_timeout)
            progress.stage_finish(stage)

        progress.stage_start("prompt_upload", "uploading SWE-bench prompt")
        prompt = _opencode_swebench_prompt(
            instance,
            repo_path=str(environment_kwargs.get("cwd") or "/testbed"),
            max_diff_lines=int(params.get("max_diff_lines") or 400),
        )
        prompt_path = "/tmp/xgym_swebench_prompt.txt"
        prompt_file = instance_dir / "prompt.txt"
        prompt_file.write_text(prompt, encoding="utf-8")
        sandbox.upload(prompt_file, prompt_path)
        progress.stage_finish("prompt_upload")

        def opencode_progress_callback(message: str) -> None:
            text = str(message or "")
            if text and not text.startswith(("terminal long command", "waiting for terminal")):
                progress.append_opencode_session_log(text)
            progress.stage_update("agent_run", text or "opencode activity")

        params["opencode_progress_callback"] = opencode_progress_callback
        provider_id = str(params.get("opencode_provider_id") or "litellm")
        model_name = str(params["policy_model_name"])
        raw_opencode_command = _opencode_run_command(
            params=params,
            provider_id=provider_id,
            model_name=model_name,
            prompt=prompt,
            prompt_path=prompt_path,
        )
        run_command = f"""
set +e
before_sessions="$(find {shlex.quote(OPENCODE_REMOTE_SESSION_DIR)} -maxdepth 1 -name '*.json' -type f 2>/dev/null || true)"
(
  last_snapshot=""
  heartbeat_count=0
  while true; do
    current_sessions="$(find {shlex.quote(OPENCODE_REMOTE_SESSION_DIR)} -maxdepth 1 -name '*.json' -type f 2>/dev/null || true)"
    latest_session="$(printf '%s\n' "$current_sessions" | sort | tail -n 1)"
    if [ -n "$latest_session" ]; then
      snapshot="$(wc -c < "$latest_session" 2>/dev/null):$latest_session"
      if [ "$snapshot" != "$last_snapshot" ]; then
        printf 'XGYM_OPENCODE_SESSION_SNAPSHOT path=%s bytes=%s\n' "$latest_session" "${{snapshot%%:*}}"
        tail -c 20000 "$latest_session" 2>/dev/null || true
        printf '\nXGYM_OPENCODE_SESSION_SNAPSHOT_END path=%s\n' "$latest_session"
        last_snapshot="$snapshot"
      fi
      printf 'XGYM_OPENCODE_HEARTBEAT count=%s path=%s bytes=%s time=%s\n' "$heartbeat_count" "$latest_session" "${{snapshot%%:*}}" "$(date -Iseconds 2>/dev/null || date)"
    else
      printf 'XGYM_OPENCODE_HEARTBEAT count=%s path=- bytes=0 time=%s\n' "$heartbeat_count" "$(date -Iseconds 2>/dev/null || date)"
    fi
    heartbeat_count=$((heartbeat_count + 1))
    sleep 30
  done
) &
monitor_pid=$!
{raw_opencode_command}
opencode_rc=$?
kill "$monitor_pid" >/dev/null 2>&1 || true
wait "$monitor_pid" >/dev/null 2>&1 || true
latest_session="$(find {shlex.quote(OPENCODE_REMOTE_SESSION_DIR)} -maxdepth 1 -name '*.json' -type f 2>/dev/null | sort | tail -n 1)"
if [ -n "$latest_session" ]; then
  printf 'XGYM_OPENCODE_SESSION_SNAPSHOT path=%s bytes=%s final=1\n' "$latest_session" "$(wc -c < "$latest_session" 2>/dev/null || printf 0)"
  cat "$latest_session" 2>/dev/null || true
  printf '\nXGYM_OPENCODE_SESSION_SNAPSHOT_END path=%s final=1\n' "$latest_session"
fi
exit "$opencode_rc"
""".strip()
        progress.stage_start("agent_run", f"opencode run --model {provider_id}/{model_name}")
        heartbeat = _ProgressHeartbeat(
            progress=progress,
            stage="agent_run",
            message="waiting for terminal sandbox/opencode activity",
            interval_s=60.0,
        ).start()
        try:
            opencode_result = env.execute(
                run_command,
                timeout=int(params.get("opencode_command_timeout") or params["step_timeout"]),
                cwd=str(environment_kwargs.get("cwd") or "/testbed"),
                background=True,
            )
        finally:
            heartbeat.stop()
        opencode_output = str(opencode_result.get("output") or "")
        session_id = _extract_opencode_session_id(opencode_output)
        if int(opencode_result.get("returncode", 1)) != 0:
            progress.stage_update(
                "agent_run",
                f"opencode exited with {opencode_result.get('returncode')}; attempting partial patch recovery",
            )
            partial_patch_result = env.execute(
                "git add -A && git diff --binary --no-color HEAD",
                timeout=300,
                cwd=str(environment_kwargs.get("cwd") or "/testbed"),
            )
            partial_patch = str(partial_patch_result.get("output") or "")
            if int(partial_patch_result.get("returncode", 1)) != 0 or not partial_patch.strip():
                output = opencode_output
                raise RuntimeError(
                    _redact_opencode_secrets(
                        output or f"command failed with exit code {opencode_result.get('returncode')}: {run_command}",
                        params,
                    )
                )
            progress.stage_finish(
                "agent_run",
                status="completed",
                message=f"partial patch recovered after agent error; session_id={session_id or '-'}",
            )
        else:
            progress.stage_finish("agent_run", message=f"session_id={session_id or '-'}")
        if session_id:
            export_path = f"{OPENCODE_REMOTE_SESSION_DIR}/{session_id}.json"
            progress.stage_start("session_export", f"opencode export {session_id}")
            export_result = env.execute(
                f"{_opencode_api_key_export(params)} && {OPENCODE_NPM_NO_INSPECT_PREFIX} opencode export {shlex.quote(session_id)} > {shlex.quote(export_path)}",
                timeout=60,
            )
            exported_session = ""
            if int(export_result.get("returncode", 1)) == 0:
                read_export_result = env.execute(f"cat {shlex.quote(export_path)}", timeout=60)
                if int(read_export_result.get("returncode", 1)) == 0:
                    exported_session = str(read_export_result.get("output") or "")
                    progress.write_opencode_session_log(exported_session, session_id=session_id)
            progress.stage_finish("session_export", message=export_path)

        progress.stage_start("patch_collect", "git diff")
        model_patch = _run_checked(
            env,
            "git add -A && git diff --binary --no-color HEAD",
            timeout=300,
            cwd=str(environment_kwargs.get("cwd") or "/testbed"),
        )
        if not model_patch.strip():
            raise RuntimeError("opencode completed without producing repository changes")
        progress.stage_finish("patch_collect", message=f"bytes={len(model_patch.encode('utf-8'))}")

        run_id = f"{int(time.time())}_{uuid4()}"
        progress.stage_start("eval", "running SWE-bench eval script")
        eval_report = _run_eval_v2(
            instance=instance,
            env=env,
            model_patch=model_patch,
            instance_dir=instance_dir,
            run_id=run_id,
            is_golden=False,
        )
        progress.stage_finish("eval")
        patch_path = instance_dir / f"opencode_patch_{run_id}.diff"
        patch_path.write_text(model_patch, encoding="utf-8")

        response_output = []
        if opencode_output.strip():
            response_output.append(
                {
                    "id": f"msg_{uuid4()}",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": opencode_output[-4000:], "annotations": []}],
                }
            )
        response = _default_response_object()
        response["id"] = f"resp_{uuid4()}"
        response["output"] = response_output
        return {
            instance_id: {
                "input_messages": [
                    {"type": "message", "role": "user", "content": str(prompt)},
                ],
                "response_output": response_output,
                "responses": [response],
                "eval_report": {
                    **eval_report,
                    "sandbox_image": sandbox_image,
                    "agent_framework": "opencode",
                    "opencode_session_id": session_id,
                    "model_patch": model_patch,
                    **progress.paths(),
                },
                "exit_status": "Submitted",
            }
        }
    except Exception as exc:
        progress.fail(str(exc))
        raise
    finally:
        had_failed = progress._state.get("status") == "failed"
        try:
            progress.stage_start("cleanup", "stopping sandbox")
        except Exception:
            pass
        try:
            if sandbox is not None:
                sandbox.stop()
        finally:
            progress.stage_finish("cleanup")
            if had_failed:
                progress._state["status"] = "failed"
                progress._record_event("instance_end", status="failed", message=progress._state.get("error"))
            else:
                progress.finish("completed")


def _run_opencode_with_sandbox(**params: Any) -> dict[str, Any]:
    max_retries = max(0, int(params.get("opencode_infra_retries") or 0))
    retry_delay = max(0.0, float(params.get("opencode_infra_retry_delay_s") or 0.0))
    last_exc: BaseException | None = None
    for attempt in range(max_retries + 1):
        attempt_params = dict(params)
        attempt_params["opencode_attempt"] = attempt
        try:
            return _run_opencode_with_sandbox_attempt(**attempt_params)
        except Exception as exc:
            last_exc = exc
            category = _opencode_infra_error_category(str(exc))
            if attempt >= max_retries or category is None:
                raise
            instance_id = str(params.get("instance_id") or "unknown")
            print(
                f"[EVAL]{instance_id} retrying opencode after {category} "
                f"failure on attempt {attempt + 1}/{max_retries + 1}: {exc}",
                flush=True,
            )
            if retry_delay:
                time.sleep(retry_delay)
    assert last_exc is not None
    raise last_exc


def _run_mini_swe_v2(**params: Any) -> dict[str, Any]:
    from minisweagent.agents.default import DefaultAgent
    from minisweagent.environments import get_environment
    from minisweagent.models import get_model

    instance = params.get("instance_dict")
    if isinstance(instance, str):
        instance = json.loads(instance)
    if not isinstance(instance, dict):
        raise ValueError("mini-swe-agent v2 path requires instance_dict")

    instance = dict(instance)
    instance_id = str(params.get("instance_id") or instance["instance_id"]).lower()
    instance["instance_id"] = instance_id

    output_dir = Path(params["output"])
    instance_dir = output_dir / instance_id
    output_dir.mkdir(parents=True, exist_ok=True)
    instance_dir.mkdir(parents=True, exist_ok=True)
    progress = _StageProgressRecorder(instance_id=instance_id, instance_dir=instance_dir)

    config = yaml.safe_load(get_config_path(params["config"]).read_text())
    _restore_sandbox_provider_secrets(config)
    model_config = config.setdefault("model", {})
    model_config["model_class"] = "litellm"
    model_config["model_name"] = params["model"]
    model_config.setdefault("cost_tracking", "ignore_errors")
    model_kwargs = model_config.setdefault("model_kwargs", {})
    model_kwargs["api_key"] = params["api_key"]
    model_kwargs["base_url"] = params["base_url"]
    model_kwargs.pop("api_base", None)
    max_output_tokens = model_kwargs.pop("max_output_tokens", None)
    if max_output_tokens is not None and "max_tokens" not in model_kwargs:
        model_kwargs["max_tokens"] = max_output_tokens

    environment_config = config.setdefault("environment", {})
    image_map = _load_image_map(params.get("image_map_path"))
    environment_config["image"] = _swebench_image_name(
        instance,
        params["subset"],
        image_map=image_map,
        require_image_map=bool(params.get("require_image_map")),
    )
    environment_config["step_timeout"] = params["step_timeout"]
    environment_config["eval_timeout"] = params["eval_timeout"]
    environment_config["instance_id"] = instance_id
    environment_config["environment_class"] = (
        "responses_api_agents.mini_swe_agent_2.sandbox_environment.MiniSWESandboxEnvironment"
    )

    agent_config = config.get("agent", {})
    agent_config["step_limit"] = params["step_limit"]
    agent_config.pop("collapse_limit", None)

    run_id = f"{int(time.time())}_{uuid4()}"
    trajectory_path = instance_dir / f"{instance_id}_{run_id}.traj.json"
    agent_config["output_path"] = trajectory_path
    env = None
    agent = None
    try:
        progress.stage_start("environment_create", f"image={environment_config['image']}")
        print(f"[EVAL]{instance_id} Creating environment...", flush=True)
        env = get_environment(environment_config)
        print(f"[EVAL]{instance_id} Environment created", flush=True)
        progress.stage_finish("environment_create")

        progress.stage_start("agent_init", "initializing mini-swe-agent")
        model = get_model(config=model_config)
        agent = DefaultAgent(model, env, **agent_config)
        progress.stage_finish("agent_init")

        if params["run_golden"]:
            progress.stage_start("agent_run", "using golden patch")
            exit_status = "Gold Patch Applied"
            model_patch = instance.get("patch", "")
            data = agent.save(None, {"messages": []})
            progress.stage_finish("agent_run", message=exit_status)
        else:
            progress.stage_start("agent_run", "running mini-swe-agent v2")
            print(f"[EVAL]{instance_id} Running mini-swe-agent v2...", flush=True)
            info = agent.run(instance["problem_statement"])
            exit_status = info.get("exit_status", "")
            model_patch = info.get("submission", "")
            data = agent.save(
                trajectory_path,
                {"instance_id": instance_id},
            )
            progress.stage_finish("agent_run", message=str(exit_status or "submitted"))

        progress.stage_start("eval", "running SWE-bench eval script")
        print(f"[EVAL]{instance_id} Running eval", flush=True)
        eval_report = _run_eval_v2(
            instance=instance,
            env=env,
            model_patch=model_patch,
            instance_dir=instance_dir,
            run_id=run_id,
            is_golden=params["run_golden"],
        )
        progress.stage_finish("eval")
        print(f"[EVAL]{instance_id} Eval completed", flush=True)

        input_messages, response_output, responses = _split_trajectory_for_responses(data.get("messages", []))

        return {
            instance_id: {
                "input_messages": input_messages,
                "response_output": response_output,
                "responses": responses,
                "eval_report": {**eval_report, **progress.paths()},
                "exit_status": exit_status,
            }
        }
    except Exception as exc:
        progress.fail(str(exc))
        raise
    finally:
        had_failed = progress._state.get("status") == "failed"
        if env and hasattr(env, "cleanup"):
            progress.stage_start("cleanup", "cleaning up environment")
            env.cleanup()
            progress.stage_finish("cleanup")
        if had_failed:
            progress._state["status"] = "failed"
            progress._record_event("instance_end", status="failed", message=progress._state.get("error"))
        else:
            progress.finish("completed")


def run_mini_swe_with_sandbox(**params: Any) -> Any:
    if params.get("agent_framework") == "opencode":
        return _run_opencode_with_sandbox(**params)
    return _run_mini_swe_v2(**params)


class MiniSWEAgent(SimpleResponsesAPIAgent):
    config: MiniSWEAgentConfig
    sem: Semaphore = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        self.sem = Semaphore(self.config.concurrency)

    def setup_webserver(self) -> FastAPI:
        app = FastAPI()
        self.setup_session_middleware(app)
        app.post("/v1/responses")(self.responses)
        app.post("/run")(self.run)
        app.get("/progress/{instance_id}")(self.progress)
        app.post("/aggregate_metrics")(self.aggregate_metrics)
        return app

    async def progress(self, instance_id: str) -> dict[str, Any]:
        candidates = sorted(
            Path.cwd().glob(f"results/**/{instance_id}/{PROGRESS_STATE_FILENAME}"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            return {"instance_id": instance_id, "status": "not_found"}
        state_path = candidates[0]
        try:
            payload = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return {
                "instance_id": instance_id,
                "status": "error",
                "error": str(exc),
                "progress_state_path": str(state_path),
            }
        if isinstance(payload, dict):
            return payload
        return {
            "instance_id": instance_id,
            "status": "error",
            "error": "progress state is not a JSON object",
            "progress_state_path": str(state_path),
        }

    def compute_metrics(self, tasks: list[list[dict[str, Any]]]) -> dict[str, Any]:
        metrics, _, _, max_k = compute_pass_majority_metrics(tasks)
        metrics.pop("per_sample_aggregate", None)

        all_rollouts = [rollout for task in tasks for rollout in task]
        rollout_count = len(all_rollouts)
        resolved_task_count = sum(1 for task in tasks if any(float(r.get("reward", 0.0) or 0.0) >= 1.0 for r in task))
        eval_error_count = sum(1 for rollout in all_rollouts if _metadata_dict(rollout).get("error"))
        eval_report_count = sum(1 for rollout in all_rollouts if _eval_report_map(rollout))
        tests_status_count = sum(1 for rollout in all_rollouts if _eval_instance_report(rollout).get("tests_status"))
        patch_applied_count = sum(
            1 for rollout in all_rollouts if _eval_instance_report(rollout).get("patch_successfully_applied")
        )

        metrics.update(
            {
                "task_count": len(tasks),
                "rollout_count": rollout_count,
                "max_rollouts_per_task": max_k,
                "resolved_task_count": resolved_task_count,
                "resolved_task_rate": 100.0 * resolved_task_count / len(tasks) if tasks else 0.0,
                "eval_error_rollout_count": eval_error_count,
                "eval_error_rate": 100.0 * eval_error_count / rollout_count if rollout_count else 0.0,
                "eval_report_rollout_count": eval_report_count,
                "eval_report_rate": 100.0 * eval_report_count / rollout_count if rollout_count else 0.0,
                "tests_status_rollout_count": tests_status_count,
                "tests_status_rate": 100.0 * tests_status_count / rollout_count if rollout_count else 0.0,
                "patch_applied_rollout_count": patch_applied_count,
                "patch_applied_rate": 100.0 * patch_applied_count / rollout_count if rollout_count else 0.0,
                "per_task_metrics": self._compute_per_task_eval_metrics(tasks),
            }
        )

        test_status_totals: dict[str, int] = {}
        for rollout in all_rollouts:
            for key, value in _test_status_counts(rollout).items():
                test_status_totals[key] = test_status_totals.get(key, 0) + value
        metrics.update({f"tests_status/{key}": value for key, value in sorted(test_status_totals.items())})

        return metrics

    def _compute_per_task_eval_metrics(self, tasks: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
        per_task_metrics: list[dict[str, Any]] = []
        for fallback_idx, rollouts in enumerate(tasks):
            if not rollouts:
                continue

            first = rollouts[0]
            task_index = first.get(TASK_INDEX_KEY_NAME, fallback_idx)
            instance_id = first.get("instance_id") or _metadata_dict(first).get("instance_id")
            resolved_count = sum(1 for rollout in rollouts if float(rollout.get("reward", 0.0) or 0.0) >= 1.0)
            error_count = sum(1 for rollout in rollouts if _metadata_dict(rollout).get("error"))
            eval_report_count = sum(1 for rollout in rollouts if _eval_report_map(rollout))
            tests_status_count = sum(1 for rollout in rollouts if _eval_instance_report(rollout).get("tests_status"))
            patch_applied_count = sum(
                1 for rollout in rollouts if _eval_instance_report(rollout).get("patch_successfully_applied")
            )

            task_metrics: dict[str, Any] = {
                TASK_INDEX_KEY_NAME: task_index,
                "instance_id": instance_id,
                "rollout_count": len(rollouts),
                "resolved": resolved_count > 0,
                "resolved_rollout_count": resolved_count,
                "eval_error_rollout_count": error_count,
                "eval_report_rollout_count": eval_report_count,
                "tests_status_rollout_count": tests_status_count,
                "patch_applied_rollout_count": patch_applied_count,
            }

            test_status_totals: dict[str, int] = {}
            for rollout in rollouts:
                for key, value in _test_status_counts(rollout).items():
                    test_status_totals[key] = test_status_totals.get(key, 0) + value
            task_metrics.update({f"tests_status/{key}": value for key, value in sorted(test_status_totals.items())})
            per_task_metrics.append(task_metrics)

        return per_task_metrics

    def get_key_metrics(self, agent_metrics: dict[str, Any]) -> dict[str, Any]:
        key_metrics: dict[str, Any] = {}
        key_metrics.update(highest_k_metrics(agent_metrics, "pass@{k}", score_names=["accuracy"]))
        key_metrics.update(highest_k_metrics(agent_metrics, "pass@1[avg-of-{k}]", score_names=["accuracy"]))
        for key in (
            "mean/reward",
            "resolved_task_count",
            "task_count",
            "resolved_task_rate",
            "eval_error_rate",
            "tests_status_rate",
        ):
            if key in agent_metrics:
                key_metrics[key] = agent_metrics[key]
        return key_metrics

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> NeMoGymResponse:
        raise NotImplementedError

    async def run(self, body: MiniSWEAgentRunRequest) -> MiniSWEAgentVerifyResponse:
        async with self.sem:
            model_server_name = self.config.model_server.name
            global_config_dict = ServerClient.load_from_global_config().global_config_dict

            model_server_config = get_first_server_config_dict(
                global_config_dict,
                model_server_name,
            )

            policy_model_name = global_config_dict["policy_model_name"]

            ##### MINI-SWE-AGENT CONFIG #####
            subset = body.subset
            split = body.split
            workers = 1
            run_golden = self.config.run_golden
            base_url = f"http://{model_server_config['host']}:{model_server_config['port']}/v1"
            dummy_key = "dummy_key"
            model_name = f"hosted_vllm/{policy_model_name}"
            step_timeout = self.config.step_timeout
            eval_timeout = self.config.eval_timeout
            step_limit = self.config.step_limit

            instance_id = body.instance_id

            mini_swe_config_path = _swebench_config_path()
            config = yaml.safe_load(get_config_path(mini_swe_config_path).read_text())
            responses_create_params_dict = body.responses_create_params.model_dump(exclude_none=True)

            default_model_kwargs = config["model"].get("model_kwargs") or {}
            temperature = (
                body.responses_create_params.temperature
                if body.responses_create_params.temperature is not None
                else default_model_kwargs.get("temperature")
            )
            top_p = (
                body.responses_create_params.top_p
                if body.responses_create_params.top_p is not None
                else default_model_kwargs.get("top_p")
            )
            model_kwargs = _responses_create_params_to_model_kwargs(
                responses_create_params_dict,
                default_tool_choice=self.config.tool_choice,
            )
            if model_kwargs:
                config.setdefault("model", {}).setdefault("model_kwargs", {}).update(model_kwargs)

            output_file_dir = f"{Path.cwd()}/results/{subset}/{policy_model_name}"
            config_path = mini_swe_config_path
            should_write_config = bool(model_kwargs)
            if self.config.sandbox_provider is None:
                raise ValueError("mini_swe_agent_2 requires sandbox_provider")
            resolved_sandbox_provider = resolve_provider_config(self.config.sandbox_provider, global_config_dict)
            provider_default_metadata = resolve_provider_metadata(self.config.sandbox_provider, global_config_dict)
            config.setdefault("environment", {}).update(self.config.sandbox_environment_kwargs or {})
            config["environment"]["provider"] = _sandbox_provider_for_config_dump(resolved_sandbox_provider)
            instance_spec = _sandbox_spec_for_instance(
                self.config.sandbox_spec,
                resource_profiles=self.config.sandbox_resource_profiles,
                instance_id=instance_id,
            )
            if provider_default_metadata:
                # Provider defaults first; the agent's own spec metadata wins on conflict.
                instance_spec["metadata"] = {**provider_default_metadata, **(instance_spec.get("metadata") or {})}
            config["environment"]["spec"] = instance_spec
            should_write_config = True

            if should_write_config:
                config_output_dir = Path(output_file_dir) / "_configs"
                config_output_dir.mkdir(parents=True, exist_ok=True)
                config_path = config_output_dir / f"{instance_id}.sandbox.yaml"
                config_path.write_text(yaml.safe_dump(config, sort_keys=False))

            if self.config.skip_if_exists:
                if Path(f"{output_file_dir}/{instance_id}/{instance_id}.json").exists():
                    with open(f"{output_file_dir}/{instance_id}/{instance_id}.json", "r") as f:
                        print(f"Skipping {instance_id} because it already exists")
                        verify_response = MiniSWEAgentVerifyResponse.model_validate_json(f.read())
                    return verify_response

            #### RUN MINI-SWE-AGENT #####
            try:
                params = dict(
                    subset=subset,
                    split=split,
                    workers=workers,
                    output=output_file_dir,
                    model=model_name,
                    policy_model_name=policy_model_name,
                    api_key=dummy_key,
                    base_url=base_url,
                    env="sandbox",
                    run_golden=run_golden,
                    instance_id=instance_id,
                    config=config_path,
                    # TODO: add this later
                    instance_dict=body.model_dump(),
                    responses_create_params=json.dumps(responses_create_params_dict),
                    step_timeout=step_timeout,
                    eval_timeout=eval_timeout,
                    step_limit=step_limit,
                    agent_framework=self.config.agent_framework,
                    image_map_path=self.config.image_map_path,
                    require_image_map=self.config.require_image_map,
                    sandbox_provider=resolved_sandbox_provider,
                    sandbox_spec=instance_spec,
                    sandbox_environment_kwargs=self.config.sandbox_environment_kwargs or {},
                    opencode_provider_id=self.config.opencode_provider_id,
                    opencode_provider_api_url=self.config.opencode_provider_api_url,
                    opencode_api_key_env=self.config.opencode_api_key_env,
                    opencode_api_key=self.config.opencode_api_key,
                    opencode_npm_package=self.config.opencode_npm_package,
                    opencode_setup_timeout=self.config.opencode_setup_timeout,
                    opencode_idle_timeout=self.config.opencode_idle_timeout,
                    opencode_command_timeout=self.config.opencode_command_timeout,
                    opencode_prompt_mode=self.config.opencode_prompt_mode,
                    opencode_infra_retries=self.config.opencode_infra_retries,
                    opencode_infra_retry_delay_s=self.config.opencode_infra_retry_delay_s,
                    max_diff_lines=400,
                )
                runner = runner_ray_remote
                runtime_env = _sandbox_runtime_env(resolved_sandbox_provider)
                if runtime_env:
                    runner = runner.options(runtime_env=runtime_env)
                future = runner.remote(run_mini_swe_with_sandbox, params)
                result = await future
                result = result[instance_id]
                input_messages = result["input_messages"]
                response_output = result["response_output"]
                responses = result["responses"]
                reward = 1.0 if _is_resolved(instance_id, result["eval_report"]) else 0.0

            except Exception as e:
                error_info = {"error": str(e), "traceback": traceback.format_exc()}
                print(f"Error running mini-swe-agent: {e}\n{error_info['traceback']}", flush=True)
                result = {"eval_report": error_info}
                input_messages = []
                response_output = []
                responses = []
                reward = 0.0

            body.responses_create_params.input = input_messages
            response = _default_response_object()
            if responses:
                response.update(dict(responses[-1]))
            response.pop("extra", None)
            response["model"] = policy_model_name
            response["temperature"] = temperature
            response["top_p"] = top_p
            response["output"] = response_output

            verify_response = MiniSWEAgentVerifyResponse(
                responses_create_params=body.responses_create_params,
                reward=reward,
                response=response,
                instance_id=instance_id,
                metadata=result.get("eval_report", {}) if result else {},
            )

            output_path = Path(f"{output_file_dir}/{instance_id}")
            output_path.mkdir(parents=True, exist_ok=True)

            with open(f"{output_file_dir}/{instance_id}/{instance_id}.json", "w") as f:
                json.dump(verify_response.model_dump(), f)

            return verify_response


if __name__ == "__main__":
    MiniSWEAgent.run_webserver()
