# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""StarGaze terminal sandbox provider implementation.

This provider intentionally mirrors StarGaze's terminal sandbox manager:
region endpoints are hardcoded, auth is carried in ``X-Jwt-Token``, and the
default region is ``boei18n``.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import importlib.util
import json
import logging
import os
import posixpath
import re
import shlex
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Protocol

import requests
import urllib3

from nemo_gym.sandbox.providers.base import (
    SandboxCreateError,
    SandboxCreateVerificationError,
    SandboxExecResult,
    SandboxHandle,
    SandboxResources,
    SandboxSpec,
    SandboxStatus,
    coerce_config,
)


LOGGER = logging.getLogger(__name__)

# Terminal sandbox session endpoints currently present a certificate whose SAN
# does not match the per-session ai-sandbox hostname in BOE. Keep this scoped to
# session-domain requests; control-plane and auth requests still use default TLS.
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

TERMINAL_PATH = ":".join(
    [
        "/opt/tiger/tce/tce_tools/bin",
        "/root/.nvm/versions/node/v22.22.2/bin",
        "/opt/nvm/versions/node/v22.22.1/bin",
        "/openhands/bin",
        "/opt/tiger/tango_adhoc/bin",
        "/go/bin",
        "/opt/tiger/tango_ftf_1_22/bin",
        "/usr/local/lib/go/bin",
        "/root/go/bin",
        "/root/.local/bin",
        "/usr/local/bin",
        "/home/tiger/system_op/bin",
        "/usr/local/sbin",
        "/usr/sbin",
        "/usr/bin",
        "/sbin",
        "/bin",
    ]
)

TERMINAL_DYNAMIC_PATH_BOOTSTRAP = r"""
for __stargaze_bin in /root/.nvm/versions/node/*/bin /opt/nvm/versions/node/*/bin /root/.bun/bin /home/tiger/.bun/bin; do
  if [ -d "$__stargaze_bin" ]; then
    case ":$PATH:" in
      *":$__stargaze_bin:"*) ;;
      *) PATH="$__stargaze_bin:$PATH" ;;
    esac
  fi
done
export PATH
unset __stargaze_bin
""".strip()

TERMINAL_RUNTIME_RETURN_CODE = 125
DEFAULT_TTL_SECONDS = 240 * 60
READY_PROBE_COMMAND = "printf terminal-sandbox-ready"
READY_PROBE_EXPECTED = "terminal-sandbox-ready"
_TOKEN_CACHE_TTL_SECONDS = 60 * 55
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_JWT_NETRC_MACHINES = ("goproxy.byted.org", "luban-source.byted.org")
_CODEBASE_NETRC_MACHINE = "code.byted.org"
_CODEBASE_JWT_CACHE = ""

_USER_JWT_RETRY_ATTEMPTS_ENV = "STARGAZE_TERMINAL_USER_JWT_RETRY_ATTEMPTS"
_USER_JWT_RETRY_ATTEMPTS = 3
_USER_JWT_RETRY_DELAYS_SECONDS = (1.0, 2.0)
_TRANSIENT_USER_JWT_MARKERS = (
    "connection aborted",
    "epipe",
    "connection reset",
    "read timed out",
    "i/o timeout",
    "ssl",
    "ssleoferror",
    "eof occurred in violation of protocol",
    "max retries exceeded",
    "compliance gateway",
    "forward to backend error",
    "temporarily unavailable",
    "http 408",
    "http 429",
    "http 500",
    "http 502",
    "http 503",
    "http 504",
)
_AUTH_RETRY_ATTEMPTS_ENV = "STARGAZE_TERMINAL_AUTH_RETRY_ATTEMPTS"
_AUTH_RETRY_ATTEMPTS = 3
_AUTH_RETRY_DELAYS_SECONDS = (1.0, 2.0)
_TRANSIENT_AUTH_MARKERS = (
    "connection aborted",
    "epipe",
    "connection reset",
    "read timed out",
    "i/o timeout",
    "ssl",
    "ssleoferror",
    "eof occurred in violation of protocol",
    "max retries exceeded",
    "compliance gateway",
    "forward to backend error",
    "temporarily unavailable",
    "upstream",
)

_SESSION_CREATE_MAX_PARALLEL_ENV = "STARGAZE_TERMINAL_SESSION_CREATE_MAX_PARALLEL"
_SESSION_CREATE_LOCK_DIR_ENV = "STARGAZE_TERMINAL_SESSION_CREATE_LOCK_DIR"
_SESSION_CREATE_LOCK_TIMEOUT_ENV = "STARGAZE_TERMINAL_SESSION_CREATE_LOCK_TIMEOUT_SECONDS"
_SESSION_CREATE_STALE_LOCK_ENV = "STARGAZE_TERMINAL_SESSION_CREATE_STALE_LOCK_SECONDS"
_SESSION_CREATE_RETRY_ATTEMPTS_ENV = "STARGAZE_TERMINAL_SESSION_CREATE_RETRY_ATTEMPTS"
_SESSION_CREATE_JITTER_ENV = "STARGAZE_TERMINAL_SESSION_CREATE_JITTER_SECONDS"
_SESSION_CREATE_RETRY_JITTER_ENV = "STARGAZE_TERMINAL_SESSION_CREATE_RETRY_JITTER_SECONDS"
_SESSION_CREATE_DEFAULT_MAX_PARALLEL = 16
_SESSION_CREATE_DEFAULT_LOCK_DIR = "/tmp/stargaze_terminal_session_create_locks"
_SESSION_CREATE_DEFAULT_LOCK_TIMEOUT_SECONDS = 900.0
_SESSION_CREATE_DEFAULT_STALE_LOCK_SECONDS = 900.0
_SESSION_CREATE_DEFAULT_RETRY_ATTEMPTS = 6
_SESSION_CREATE_DEFAULT_JITTER_SECONDS = 0.0
_SESSION_CREATE_DEFAULT_RETRY_JITTER_SECONDS = 1.0
_SESSION_CREATE_TRANSIENT_STATUS_CODES = {408, 429, 500, 502, 503, 504}
_SESSION_CREATE_TRANSIENT_MARKERS = (
    "function_cold_start_timeout",
    "connection aborted",
    "connection reset",
    "read timed out",
    "i/o timeout",
    "epipe",
    "compliance gateway",
    "forward to backend error",
    "temporarily unavailable",
    "too many requests",
    "rate limit",
    "internal server error",
    "bad gateway",
    "service unavailable",
    "gateway timeout",
    "upstream",
)
_SESSION_EXEC_RETRY_ATTEMPTS_ENV = "STARGAZE_TERMINAL_SESSION_EXEC_RETRY_ATTEMPTS"
_SESSION_EXEC_RETRY_DELAY_ENV = "STARGAZE_TERMINAL_SESSION_EXEC_RETRY_DELAY_SECONDS"
_SESSION_EXEC_DEFAULT_RETRY_ATTEMPTS = 4
_SESSION_EXEC_DEFAULT_RETRY_DELAY_SECONDS = 2.0
_SESSION_EXEC_TRANSIENT_STATUS_CODES = {403, 408, 429, 500, 502, 503, 504}
_SESSION_EXEC_TRANSIENT_MARKERS = _SESSION_CREATE_TRANSIENT_MARKERS + (
    "service unavailable",
    "process/start",
    "process/connect",
)
_PROCESS_START_MAX_PARALLEL_ENV = "STARGAZE_TERMINAL_PROCESS_START_MAX_PARALLEL"
_PROCESS_START_LOCK_DIR_ENV = "STARGAZE_TERMINAL_PROCESS_START_LOCK_DIR"
_PROCESS_START_LOCK_TIMEOUT_ENV = "STARGAZE_TERMINAL_PROCESS_START_LOCK_TIMEOUT_SECONDS"
_PROCESS_START_STALE_LOCK_ENV = "STARGAZE_TERMINAL_PROCESS_START_STALE_LOCK_SECONDS"
_PROCESS_START_JITTER_ENV = "STARGAZE_TERMINAL_PROCESS_START_JITTER_SECONDS"
_PROCESS_START_DEFAULT_MAX_PARALLEL = 32
_PROCESS_START_DEFAULT_LOCK_DIR = "/tmp/stargaze_terminal_process_start_locks"
_PROCESS_START_DEFAULT_LOCK_TIMEOUT_SECONDS = 900.0
_PROCESS_START_DEFAULT_STALE_LOCK_SECONDS = 900.0
_PROCESS_START_DEFAULT_JITTER_SECONDS = 0.0


class TerminalCreateError(SandboxCreateError):
    """Raised when terminal sandbox cannot create a session."""


class TerminalCreateVerificationError(SandboxCreateVerificationError):
    """Raised when a created terminal session fails a readiness probe."""


class ServiceAccountError(Exception):
    """Base error for service-account auth."""


class UserNotAuthorizedError(ServiceAccountError):
    """Raised when the user has not authorized the service account."""

    def __init__(self, username: str, message: str = "") -> None:
        self.username = username
        super().__init__(message or f"user {username!r} has not authorized yet")


class TokenFetchError(ServiceAccountError):
    """Raised when auth token retrieval fails."""


@dataclass(frozen=True)
class EnvConfigStr:
    tce: str
    dev: str

    @property
    def value(self) -> str:
        if os.getenv("RUNTIME_IDC_NAME"):
            return self.tce
        return self.dev


@dataclass(frozen=True)
class RegionConfig:
    instance_link: str
    terminal_sandbox_url: str
    _terminal_sandbox_domain: EnvConfigStr
    terminal_sandbox_id: str
    _oauth_host: EnvConfigStr
    oauth_serv_account: str
    oauth_serv_token: str
    oauth_client_id: str
    oauth_client_secret: str

    @property
    def terminal_sandbox_domain(self) -> str:
        return self._terminal_sandbox_domain.value

    @property
    def oauth_host(self) -> str:
        return self._oauth_host.value


REGION_CONFIG: dict[str, RegionConfig] = {
    "i18n": RegionConfig(
        instance_link=(
            "https://cloud-i18n.bytedance.net/faas/function/dxgbx4v9/cluster/instances"
            "?cluster=faas-us-east&page=1&page_size=10&region=us-east"
        ),
        terminal_sandbox_url="http://controlplane.sg.ai-sandbox-i18n.tiktok-row.org/api/v1",
        _terminal_sandbox_domain=EnvConfigStr(
            tce="sg.ai-sandbox-i18n.byted.org",
            dev="sg.ai-sandbox-i18n.tiktok-row.org",
        ),
        terminal_sandbox_id="dxgbx4v9",
        _oauth_host=EnvConfigStr(
            tce="cloud-i18n.bytedance.net",
            dev="cloud-i18n.bytedance.net",
        ),
        oauth_serv_account="codewise_service",
        oauth_serv_token="",
        oauth_client_id="",
        oauth_client_secret="",
    ),
    "boei18n": RegionConfig(
        instance_link=(
            "https://cloud-boe-i18n.bytedance.net/faas/function/w8lpv49e/cluster/instances"
            "?cluster=faas-us-east&page=1&page_size=10&region=us-east"
        ),
        terminal_sandbox_url="http://aipaas-gateway-boei18n.byted.org/api/v1",
        _terminal_sandbox_domain=EnvConfigStr(
            tce="us-east.ai-sandbox-boei18n.byted.org",
            dev="us-east.ai-sandbox-boei18n.byted.org",
        ),
        terminal_sandbox_id="lgjk1fy5",
        _oauth_host=EnvConfigStr(
            tce="cloud-i18n.bytedance.net",
            dev="cloud-i18n.bytedance.net",
        ),
        oauth_serv_account="codewise_service",
        oauth_serv_token="",
        oauth_client_id="",
        oauth_client_secret="",
    ),
}


@lru_cache(maxsize=1)
def _stargaze_region_config() -> dict[str, RegionConfig]:
    """Load StarGaze's region config when available.

    StarGaze carries the internal OAuth service-account credentials in its
    sandbox_config module. xGym keeps checked-in defaults empty, but local users
    often have StarGazeWorkflow next to xGym. Reusing that module matches
    StarGaze's terminal-sandbox behavior without requiring STARGAZE_OAUTH_* env
    vars in this repo.
    """
    stargaze_region_config: Mapping[str, Any] | None = None
    try:
        from server.config.sandbox_config import (
            REGION_CONFIG as imported_region_config,  # type: ignore[import-not-found]
        )

        stargaze_region_config = imported_region_config
    except Exception:
        current_path = Path(__file__).resolve()
        for parent in current_path.parents:
            candidate = parent.parent / "StarGazeWorkflow" / "server" / "config" / "sandbox_config.py"
            if not candidate.is_file():
                continue
            spec = importlib.util.spec_from_file_location("_stargaze_sandbox_config", candidate)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules.setdefault(spec.name, module)
            try:
                spec.loader.exec_module(module)
            except Exception as exc:
                LOGGER.warning("failed to load StarGaze sandbox_config from %s: %s", candidate, exc)
                continue
            loaded = getattr(module, "REGION_CONFIG", None)
            if isinstance(loaded, Mapping):
                stargaze_region_config = loaded
                break
    if not stargaze_region_config:
        return {}

    converted: dict[str, RegionConfig] = {}
    for name, cfg in stargaze_region_config.items():
        try:
            converted[name] = RegionConfig(
                instance_link=str(cfg.instance_link),
                terminal_sandbox_url=str(cfg.terminal_sandbox_url),
                _terminal_sandbox_domain=EnvConfigStr(
                    tce=str(getattr(cfg._terminal_sandbox_domain, "tce")),
                    dev=str(getattr(cfg._terminal_sandbox_domain, "dev")),
                ),
                terminal_sandbox_id=str(cfg.terminal_sandbox_id),
                _oauth_host=EnvConfigStr(
                    tce=str(getattr(cfg._oauth_host, "tce")),
                    dev=str(getattr(cfg._oauth_host, "dev")),
                ),
                oauth_serv_account=str(cfg.oauth_serv_account),
                oauth_serv_token=str(cfg.oauth_serv_token),
                oauth_client_id=str(cfg.oauth_client_id),
                oauth_client_secret=str(cfg.oauth_client_secret),
            )
        except Exception as exc:
            LOGGER.warning("failed to load StarGaze terminal region config for %s: %s", name, exc)
    return converted


def get_region_config(region: str) -> RegionConfig:
    stargaze_cfg = _stargaze_region_config().get(region)
    if stargaze_cfg is not None:
        return stargaze_cfg
    if region not in REGION_CONFIG:
        raise KeyError(f"Unknown region: {region}. Available regions: {list(REGION_CONFIG.keys())}")
    return REGION_CONFIG[region]


def get_all_regions() -> list[str]:
    return list(REGION_CONFIG.keys())


def get_default_region() -> str:
    return "boei18n"


@dataclass(frozen=True)
class UserToken:
    access_token: str
    expires_at: datetime


@dataclass(frozen=True)
class UserTokenResponse:
    access_token: str | None = None
    expires_in: int | None = None
    token_type: str | None = None
    error: str | None = None
    error_code: int | None = None
    error_description: str | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "UserTokenResponse":
        return cls(
            access_token=data.get("access_token"),
            expires_in=data.get("expires_in"),
            token_type=data.get("token_type"),
            error=data.get("error"),
            error_code=data.get("error_code"),
            error_description=data.get("error_description"),
        )

    @property
    def is_authorized(self) -> bool:
        return bool(self.access_token)

    @property
    def is_waiting(self) -> bool:
        return self.error_code == 1024


class ServiceAccountManager:
    """StarGaze-style service-account auth helper."""

    def __init__(self, config: RegionConfig) -> None:
        self._config = config
        self._cached_token: str | None = None
        self._token_fetched_at = 0.0
        self._lock = threading.RLock()

    def _oauth_secret(self, field_name: str, env_name: str) -> str:
        value = str(os.environ.get(env_name) or getattr(self._config, field_name) or "").strip()
        if not value:
            raise TokenFetchError(
                f"missing terminal OAuth credential {field_name!r}: set {env_name} or install StarGazeWorkflow"
            )
        return value

    def get_token(self) -> str:
        with self._lock:
            now = time.monotonic()
            if self._cached_token and (now - self._token_fetched_at) < _TOKEN_CACHE_TTL_SECONDS:
                LOGGER.debug("service account JWT cache hit")
                return self._cached_token

            url = f"https://{self._config.oauth_host}/auth/api/v1/jwt"
            headers = {
                "Authorization": f"Bearer {self._oauth_secret('oauth_serv_token', 'STARGAZE_OAUTH_SERV_TOKEN')}"
            }
            last_error: BaseException | None = None
            last_response: requests.Response | None = None
            retry_attempts = _auth_retry_attempts()
            for attempt in range(retry_attempts):
                try:
                    response = requests.get(url, headers=headers, timeout=30)
                except requests.RequestException as exc:
                    last_error = exc
                    if attempt + 1 < retry_attempts and _is_transient_auth_exception(exc):
                        _log_auth_retry("get service account JWT", attempt, retry_attempts, exc)
                        continue
                    raise TokenFetchError(f"GET {url} failed: {exc}") from exc

                last_response = response
                if response.status_code == 200:
                    jwt_token = response.headers.get("X-Jwt-Token")
                    if not jwt_token:
                        raise TokenFetchError(f"GET {url} succeeded but response is missing X-Jwt-Token header")

                    self._cached_token = jwt_token
                    self._token_fetched_at = now
                    LOGGER.info("service account JWT refreshed successfully")
                    return jwt_token

                if attempt + 1 < retry_attempts and _is_transient_auth_response(response):
                    _log_auth_retry(
                        "get service account JWT",
                        attempt,
                        retry_attempts,
                        f"HTTP {response.status_code}: {response.text[:300]}",
                    )
                    continue
                raise TokenFetchError(
                    f"GET {url} returned HTTP {response.status_code}: {response.text} {response.headers}"
                )

            if last_response is not None:
                raise TokenFetchError(
                    f"GET {url} returned HTTP {last_response.status_code}: {last_response.text} {last_response.headers}"
                )
            assert last_error is not None
            raise TokenFetchError(f"GET {url} failed: {last_error}") from last_error

    def get_user_auth(self, username: str, ignore_cache: bool = False) -> UserTokenResponse:
        return self._call_token_api(username, lark_auth_req=True, ignore_cache=ignore_cache)

    def get_user_token(self, username: str, ignore_cache: bool = False) -> UserToken:
        resp = self._call_token_api(username, lark_auth_req=True, ignore_cache=ignore_cache)
        if resp.is_authorized:
            return UserToken(
                access_token=str(resp.access_token),
                expires_at=datetime.now() + timedelta(seconds=resp.expires_in or 0),
            )
        if resp.is_waiting:
            raise UserNotAuthorizedError(username)
        raise TokenFetchError(f"get_user_token for {username!r} returned unexpected response: {resp}")

    def _invalidate_token_cache(self) -> None:
        self._cached_token = None
        self._token_fetched_at = 0.0

    def _call_token_api(self, username: str, lark_auth_req: bool, ignore_cache: bool = False) -> UserTokenResponse:
        url = f"https://{self._config.oauth_host}/auth/api/v1/token"
        payload = {
            "client_id": self._oauth_secret("oauth_client_id", "STARGAZE_OAUTH_CLIENT_ID"),
            "client_secret": self._oauth_secret("oauth_client_secret", "STARGAZE_OAUTH_CLIENT_SECRET"),
            "redirect_uri": "",
            "grant_type": "authorization_code",
            "auth_type": "custom",
            "username": username,
            "lark_auth_req": lark_auth_req,
            "ignore_cache": ignore_cache,
        }

        last_response: UserTokenResponse | None = None
        jwt_rejected_retried = False
        attempt = 0
        retry_attempts = _auth_retry_attempts()
        max_attempts = retry_attempts + 1
        while attempt < max_attempts:
            service_jwt = self.get_token()
            headers = {
                "X-Jwt-Token": service_jwt,
                "Content-Type": "application/json",
            }
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=30)
            except requests.RequestException as exc:
                if attempt + 1 < retry_attempts and _is_transient_auth_exception(exc):
                    _log_auth_retry("get user JWT", attempt, retry_attempts, exc)
                    attempt += 1
                    continue
                raise TokenFetchError(f"POST {url} failed: {exc}") from exc

            if response.status_code not in (200, 400):
                if attempt + 1 < retry_attempts and _is_transient_auth_response(response):
                    _log_auth_retry(
                        "get user JWT",
                        attempt,
                        retry_attempts,
                        f"HTTP {response.status_code}: {response.text[:300]}",
                    )
                    attempt += 1
                    continue
                raise TokenFetchError(
                    f"POST {url} returned unexpected HTTP {response.status_code}: {response.text} {response.headers}"
                )

            try:
                data = response.json()
            except Exception as exc:
                raise TokenFetchError(
                    f"POST {url} returned non-JSON response (HTTP {response.status_code}): {response.text}"
                ) from exc

            last_response = UserTokenResponse.from_mapping(data)
            if last_response.error and "jwt" in last_response.error.lower() and not jwt_rejected_retried:
                LOGGER.warning("service JWT rejected by auth server; invalidating cache and retrying")
                jwt_rejected_retried = True
                with self._lock:
                    if self._cached_token == service_jwt:
                        self._invalidate_token_cache()
                attempt += 1
                continue
            return last_response

        assert last_response is not None
        return last_response


_SERVICE_ACCOUNT_MANAGERS: dict[str, ServiceAccountManager] = {
    "boei18n": ServiceAccountManager(get_region_config("boei18n")),
    "i18n": ServiceAccountManager(get_region_config("i18n")),
}


def get_service_account_manager(region: str = "boei18n") -> ServiceAccountManager:
    if region not in _SERVICE_ACCOUNT_MANAGERS:
        raise ValueError(f"Invalid region: {region}")
    return _SERVICE_ACCOUNT_MANAGERS[region]


def _auth_retry_attempts() -> int:
    return _env_int(_AUTH_RETRY_ATTEMPTS_ENV, _AUTH_RETRY_ATTEMPTS, minimum=1)


def _auth_retry_delay(attempt_index: int) -> float:
    return _AUTH_RETRY_DELAYS_SECONDS[min(attempt_index, len(_AUTH_RETRY_DELAYS_SECONDS) - 1)]


def _is_transient_auth_exception(exc: requests.RequestException) -> bool:
    if isinstance(exc, (requests.ConnectionError, requests.Timeout)):
        return True
    text = str(exc).lower()
    return any(marker in text for marker in _TRANSIENT_AUTH_MARKERS)


def _is_transient_auth_response(response: requests.Response) -> bool:
    status_code = int(response.status_code)
    if status_code in (408, 429, 500, 502, 503, 504):
        return True
    if status_code != 403:
        return False
    text = f"{response.text} {dict(response.headers)}".lower()
    return any(marker in text for marker in _TRANSIENT_AUTH_MARKERS)


def _log_auth_retry(action: str, attempt_index: int, retry_attempts: int, reason: object) -> None:
    delay = _auth_retry_delay(attempt_index)
    LOGGER.warning(
        "%s hit transient error, attempt %d/%d failed; retrying in %.1fs: %s",
        action,
        attempt_index + 1,
        retry_attempts,
        delay,
        reason,
    )
    time.sleep(delay)


def _is_transient_user_jwt_error(exc: Exception) -> bool:
    if isinstance(exc, TokenFetchError):
        text = str(exc).lower()
        return any(marker in text for marker in _TRANSIENT_USER_JWT_MARKERS)
    if isinstance(exc, requests.RequestException):
        return True
    return False


def _user_jwt_retry_attempts() -> int:
    return _env_int(_USER_JWT_RETRY_ATTEMPTS_ENV, _USER_JWT_RETRY_ATTEMPTS, minimum=1)


def _user_jwt_retry_delay(attempt_index: int) -> float:
    return _USER_JWT_RETRY_DELAYS_SECONDS[min(attempt_index, len(_USER_JWT_RETRY_DELAYS_SECONDS) - 1)]


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        return max(minimum, int(raw_value))
    except ValueError:
        LOGGER.warning("ignoring invalid integer environment variable %s=%r; using %s", name, raw_value, default)
        return default


def _env_float(name: str, default: float, *, minimum: float = 0.0) -> float:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        return max(minimum, float(raw_value))
    except ValueError:
        LOGGER.warning("ignoring invalid float environment variable %s=%r; using %s", name, raw_value, default)
        return default


def _session_create_retry_attempts(default_attempts: int) -> int:
    return _env_int(
        _SESSION_CREATE_RETRY_ATTEMPTS_ENV,
        default_attempts,
        minimum=1,
    )


def _stable_jitter(*, key: object, max_seconds: float) -> float:
    if max_seconds <= 0:
        return 0.0
    digest = hashlib.md5(str(key).encode("utf-8", errors="replace")).hexdigest()[:8]
    return (int(digest, 16) / 0xFFFFFFFF) * max_seconds


def _session_create_initial_jitter(key: object) -> float:
    return _stable_jitter(
        key=key,
        max_seconds=_env_float(
            _SESSION_CREATE_JITTER_ENV,
            _SESSION_CREATE_DEFAULT_JITTER_SECONDS,
            minimum=0.0,
        ),
    )


def _session_create_retry_delay(attempt_index: int, reason: object) -> float:
    base_delay = min(30.0, 2.0 * (2 ** min(attempt_index, 5)))
    retry_jitter = _env_float(
        _SESSION_CREATE_RETRY_JITTER_ENV,
        _SESSION_CREATE_DEFAULT_RETRY_JITTER_SECONDS,
        minimum=0.0,
    )
    return base_delay + _stable_jitter(key=f"{attempt_index}:{reason}", max_seconds=retry_jitter)


def _is_transient_session_create_response(status_code: int, response_text: str) -> bool:
    if int(status_code) in _SESSION_CREATE_TRANSIENT_STATUS_CODES:
        return True
    text = str(response_text or "").lower()
    return any(marker in text for marker in _SESSION_CREATE_TRANSIENT_MARKERS)


def _is_transient_session_create_exception(exc: Exception) -> bool:
    if isinstance(exc, (requests.ConnectionError, requests.Timeout)):
        return True
    if isinstance(exc, requests.RequestException):
        text = str(exc).lower()
        return any(marker in text for marker in _SESSION_CREATE_TRANSIENT_MARKERS)
    return False


def _session_exec_retry_attempts() -> int:
    return _env_int(
        _SESSION_EXEC_RETRY_ATTEMPTS_ENV,
        _SESSION_EXEC_DEFAULT_RETRY_ATTEMPTS,
        minimum=1,
    )


def _session_exec_retry_delay(attempt_index: int, reason: object) -> float:
    base_delay = _env_float(
        _SESSION_EXEC_RETRY_DELAY_ENV,
        _SESSION_EXEC_DEFAULT_RETRY_DELAY_SECONDS,
        minimum=0.0,
    )
    reason_hash = int(hashlib.md5(str(reason).encode("utf-8", errors="replace")).hexdigest()[:4], 16)
    jitter = (reason_hash % 500) / 1000.0
    return min(30.0, base_delay * (2 ** min(attempt_index, 5))) + jitter


def _is_transient_session_exec_exception(exc: Exception) -> bool:
    if isinstance(exc, requests.HTTPError):
        response = getattr(exc, "response", None)
        if response is not None:
            status_code = int(getattr(response, "status_code", 0))
            response_text = str(getattr(response, "text", "") or "")
            if status_code in _SESSION_EXEC_TRANSIENT_STATUS_CODES:
                return status_code != 403 or any(
                    marker in response_text.lower() for marker in _SESSION_EXEC_TRANSIENT_MARKERS
                )
    if isinstance(exc, requests.ConnectionError):
        return True
    if isinstance(exc, requests.Timeout):
        # A client-side read timeout may mean the command is still running, so do
        # not blindly replay it. Let the caller see the timeout result.
        return False
    if isinstance(exc, requests.RequestException):
        text = str(exc).lower()
        return any(marker in text for marker in _SESSION_EXEC_TRANSIENT_MARKERS)
    return False


def _safe_lock_group(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()[:16]


def _session_create_lock_root() -> str:
    return (
        os.environ.get(_SESSION_CREATE_LOCK_DIR_ENV, _SESSION_CREATE_DEFAULT_LOCK_DIR).strip()
        or _SESSION_CREATE_DEFAULT_LOCK_DIR
    )


def _process_start_lock_root() -> str:
    return (
        os.environ.get(_PROCESS_START_LOCK_DIR_ENV, _PROCESS_START_DEFAULT_LOCK_DIR).strip()
        or _PROCESS_START_DEFAULT_LOCK_DIR
    )


def _process_start_initial_jitter(key: object) -> float:
    return _stable_jitter(
        key=key,
        max_seconds=_env_float(
            _PROCESS_START_JITTER_ENV,
            _PROCESS_START_DEFAULT_JITTER_SECONDS,
            minimum=0.0,
        ),
    )


class _SlotAdmission:
    def __init__(
        self,
        *,
        group: str,
        label: str,
        lock_root: str,
        max_parallel: int,
        timeout_seconds: float,
        stale_seconds: float,
    ) -> None:
        self.group = _safe_lock_group(group)
        self.label = label
        self.max_parallel = max_parallel
        self.timeout_seconds = timeout_seconds
        self.stale_seconds = stale_seconds
        self.lock_dir = os.path.join(lock_root, self.group)
        self.lock_path: str | None = None

    def __enter__(self) -> "_SlotAdmission":
        os.makedirs(self.lock_dir, exist_ok=True)
        deadline = time.monotonic() + self.timeout_seconds
        while True:
            self._cleanup_stale_locks()
            for index in range(self.max_parallel):
                candidate = os.path.join(self.lock_dir, f"slot_{index}.lock")
                try:
                    fd = os.open(candidate, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                except FileExistsError:
                    continue
                except OSError as exc:
                    LOGGER.warning("failed to acquire terminal sandbox %s lock; retrying: %s", self.label, exc)
                    continue
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.write(json.dumps({"pid": os.getpid(), "created_at": time.time()}) + "\n")
                self.lock_path = candidate
                return self
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"timed out waiting for terminal sandbox {self.label} admission lock: "
                    f"group={self.group}, max_parallel={self.max_parallel}, timeout={self.timeout_seconds}s"
                )
            time.sleep(0.2)

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        if not self.lock_path:
            return False
        try:
            os.remove(self.lock_path)
        except FileNotFoundError:
            pass
        except OSError as remove_exc:
            LOGGER.warning("failed to release terminal sandbox %s lock: %s", self.label, remove_exc)
        self.lock_path = None
        return False

    def _cleanup_stale_locks(self) -> None:
        now = time.time()
        try:
            entries = os.listdir(self.lock_dir)
        except FileNotFoundError:
            return
        except OSError as exc:
            LOGGER.warning("failed to scan terminal sandbox %s lock directory: %s", self.label, exc)
            return
        for name in entries:
            if not name.endswith(".lock"):
                continue
            path = os.path.join(self.lock_dir, name)
            try:
                stat = os.stat(path)
            except (FileNotFoundError, OSError):
                continue
            if now - stat.st_mtime < self.stale_seconds:
                continue
            try:
                os.remove(path)
                LOGGER.warning("removed stale terminal sandbox %s lock: %s", self.label, path)
            except FileNotFoundError:
                pass
            except OSError as exc:
                LOGGER.warning("failed to remove stale terminal sandbox %s lock: %s", self.label, exc)


class _SessionCreateAdmission(_SlotAdmission):
    def __init__(self, *, group: str) -> None:
        super().__init__(
            group=group,
            label="session-create",
            lock_root=_session_create_lock_root(),
            max_parallel=_env_int(
                _SESSION_CREATE_MAX_PARALLEL_ENV,
                _SESSION_CREATE_DEFAULT_MAX_PARALLEL,
                minimum=1,
            ),
            timeout_seconds=_env_float(
                _SESSION_CREATE_LOCK_TIMEOUT_ENV,
                _SESSION_CREATE_DEFAULT_LOCK_TIMEOUT_SECONDS,
                minimum=1.0,
            ),
            stale_seconds=_env_float(
                _SESSION_CREATE_STALE_LOCK_ENV,
                _SESSION_CREATE_DEFAULT_STALE_LOCK_SECONDS,
                minimum=1.0,
            ),
        )


class _ProcessStartAdmission(_SlotAdmission):
    def __init__(self, *, group: str) -> None:
        super().__init__(
            group=group,
            label="process-start",
            lock_root=_process_start_lock_root(),
            max_parallel=_env_int(
                _PROCESS_START_MAX_PARALLEL_ENV,
                _PROCESS_START_DEFAULT_MAX_PARALLEL,
                minimum=1,
            ),
            timeout_seconds=_env_float(
                _PROCESS_START_LOCK_TIMEOUT_ENV,
                _PROCESS_START_DEFAULT_LOCK_TIMEOUT_SECONDS,
                minimum=1.0,
            ),
            stale_seconds=_env_float(
                _PROCESS_START_STALE_LOCK_ENV,
                _PROCESS_START_DEFAULT_STALE_LOCK_SECONDS,
                minimum=1.0,
            ),
        )


@dataclass(frozen=True)
class TerminalServiceConfig:
    """Transport/auth settings for a cluster terminal service."""

    endpoint: str | None = None
    region: str | None = None
    config_dir: str | None = None
    auth: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TerminalCreateConfig:
    """Provider-neutral settings for creating terminal sandbox sessions."""

    request_timeout_s: float | None = 300
    retries: int = 2
    retry_delay_s: float = 2.0

    def __post_init__(self) -> None:
        if self.request_timeout_s is not None and self.request_timeout_s <= 0:
            raise ValueError("create.request_timeout_s must be > 0")
        if self.retries < 0:
            raise ValueError("create.retries must be >= 0")
        if self.retry_delay_s < 0:
            raise ValueError("create.retry_delay_s must be >= 0")


@dataclass(frozen=True)
class TerminalExecConfig:
    """Provider-neutral settings for command execution."""

    default_timeout_s: float | None = 180
    request_extra_timeout_s: float = 60
    concurrency: int = 32

    def __post_init__(self) -> None:
        if self.default_timeout_s is not None and self.default_timeout_s <= 0:
            raise ValueError("exec.default_timeout_s must be > 0")
        if self.request_extra_timeout_s < 0:
            raise ValueError("exec.request_extra_timeout_s must be >= 0")
        if self.concurrency < 1:
            raise ValueError("exec.concurrency must be >= 1")


@dataclass(frozen=True)
class TerminalConnectionConfig:
    """Settings matching StarGaze TerminalSandboxManager constructor fields."""

    config_dir: str = "config"
    creator_jwt_token: str = ""
    creator_email: str = ""
    model_name: str = "gpt-5-codex-2025-09-15"
    sandbox_id: str | None = None
    session_id: str | None = None
    sandbox_image: str | None = None
    default_image: str | None = None
    vregion: str | None = None
    persist_config: bool = True
    setup_netrc: bool = True
    refresh_jwt: bool = True

    def __post_init__(self) -> None:
        region = self.vregion or get_default_region()
        get_region_config(region)


@dataclass(frozen=True)
class TerminalOperationConfig:
    """Operation timeouts and limits copied from StarGaze defaults."""

    create_request_timeout_s: float = 300
    create_retries: int = 2
    create_retry_delay_s: float = 2.0
    exec_request_extra_timeout_s: float = 60
    delete_timeout_s: float = 60
    upload_timeout_s: float = 120
    download_connect_timeout_s: float = 10
    download_read_timeout_s: float = 120
    native_file_size_limit: int = 10 * 1024 * 1024
    fallback_chunk_size: int = 8000
    concurrency: int = 32

    def __post_init__(self) -> None:
        if self.create_request_timeout_s <= 0:
            raise ValueError("operations.create_request_timeout_s must be > 0")
        if self.create_retries < 0:
            raise ValueError("operations.create_retries must be >= 0")
        if self.create_retry_delay_s < 0:
            raise ValueError("operations.create_retry_delay_s must be >= 0")
        if self.exec_request_extra_timeout_s < 0:
            raise ValueError("operations.exec_request_extra_timeout_s must be >= 0")
        if self.delete_timeout_s <= 0:
            raise ValueError("operations.delete_timeout_s must be > 0")
        if self.upload_timeout_s <= 0:
            raise ValueError("operations.upload_timeout_s must be > 0")
        if self.download_connect_timeout_s <= 0:
            raise ValueError("operations.download_connect_timeout_s must be > 0")
        if self.download_read_timeout_s <= 0:
            raise ValueError("operations.download_read_timeout_s must be > 0")
        if self.native_file_size_limit < 0:
            raise ValueError("operations.native_file_size_limit must be >= 0")
        if self.fallback_chunk_size <= 0:
            raise ValueError("operations.fallback_chunk_size must be > 0")
        if self.concurrency < 1:
            raise ValueError("operations.concurrency must be >= 1")


@dataclass(frozen=True)
class TerminalProbeConfig:
    """Optional post-create probe."""

    command: str | None = None
    expected_stdout: str | None = READY_PROBE_EXPECTED
    timeout_s: int = 30
    deadline_s: float | None = None
    stable_count: int = 1
    stable_delay_s: float = 0.0

    def __post_init__(self) -> None:
        if self.command is not None and self.timeout_s <= 0:
            raise ValueError("probe.timeout_s must be > 0")
        if self.deadline_s is not None and self.deadline_s <= 0:
            raise ValueError("probe.deadline_s must be > 0")
        if self.stable_count < 1:
            raise ValueError("probe.stable_count must be >= 1")
        if self.stable_delay_s < 0:
            raise ValueError("probe.stable_delay_s must be >= 0")


@dataclass
class _TerminalSession:
    session_id: str
    sandbox_id: str
    base_url: str
    domain: str
    vregion: str
    sandbox_image: str | None = None
    session_created_at: datetime | None = None
    session_ttl: int | None = None
    instance_pod_name: str | None = None
    instance_function_id: str | None = None
    env: dict[str, str] = field(default_factory=dict)


def _with_terminal_runtime_path(command: str) -> str:
    return f"{TERMINAL_DYNAMIC_PATH_BOOTSTRAP}\n{command}"


def _infer_creator_email_from_git_config() -> str:
    for args in (
        ["git", "config", "--get", "user.email"],
        ["git", "config", "--global", "--get", "user.email"],
    ):
        try:
            completed = subprocess.run(
                args,
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if completed.returncode != 0:
            continue
        email = completed.stdout.strip()
        if email and _EMAIL_RE.match(email):
            return email
    return ""


def _stargaze_workflow_root() -> Path | None:
    configured = os.environ.get("STARGAZE_WORKFLOW_ROOT")
    candidates: list[Path] = []
    if configured:
        candidates.append(Path(configured).expanduser())
    current_path = Path(__file__).resolve()
    candidates.extend(parent.parent / "StarGazeWorkflow" for parent in current_path.parents)
    for candidate in candidates:
        if (candidate / "utils" / "baseline_assets.py").is_file():
            return candidate
    return None


def _get_codebase_jwt(*, ignore_cache: bool = False) -> str:
    global _CODEBASE_JWT_CACHE
    token = (os.environ.get("CODEBASE_JWT_TOKEN") or os.environ.get("STARGAZE_CODEBASE_JWT_TOKEN") or "").strip()
    if token:
        return token
    if _CODEBASE_JWT_CACHE and not ignore_cache:
        return _CODEBASE_JWT_CACHE
    stargaze_root = _stargaze_workflow_root()
    if stargaze_root is not None and str(stargaze_root) not in sys.path:
        # StarGaze's baseline_assets imports sibling modules such as
        # server.utils.service_account_manager at call time. Put the project root
        # on sys.path instead of requiring users to launch xGym from StarGaze.
        sys.path.insert(0, str(stargaze_root))
    try:
        from utils.baseline_assets import _get_codebase_jwt as get_codebase_jwt  # type: ignore[import-not-found]

        token = get_codebase_jwt(ignore_cache=ignore_cache)
    except Exception as exc:
        LOGGER.warning("failed to get Codebase JWT via StarGaze baseline_assets: %s", exc)
        return ""
    if not token:
        LOGGER.warning("failed to get Codebase JWT: token is empty")
        return ""
    _CODEBASE_JWT_CACHE = token
    return token


def _build_jwt_netrc_content_for_hosts(jwt_token: str) -> str:
    lines: list[str] = []
    for machine in _JWT_NETRC_MACHINES:
        lines.extend(
            [
                f"machine {machine}",
                "  login x-jwt-token",
                f"  password {jwt_token}",
            ]
        )
    return "\n".join(lines) + "\n"


def _build_jwt_netrc_content(jwt_token: str, *, codebase_jwt_token: str = "") -> str:
    content = _build_jwt_netrc_content_for_hosts(jwt_token)
    if codebase_jwt_token:
        content += f"machine {_CODEBASE_NETRC_MACHINE}\n  login x-jwt-token\n  password {codebase_jwt_token}\n"
    return content


def _build_jwt_netrc_command(jwt_token: str, *, codebase_jwt_token: str = "", append: bool = False) -> str:
    redirect = ">>" if append else ">"
    content = _build_jwt_netrc_content(jwt_token, codebase_jwt_token=codebase_jwt_token)
    return f"printf %s {shlex.quote(content)} {redirect} ~/.netrc"


def _quote_env_exports(env: Mapping[str, str]) -> str:
    lines: list[str] = []
    for key, value in env.items():
        if not _ENV_NAME_RE.match(key):
            raise ValueError(f"Invalid environment variable name: {key!r}")
        lines.append(f"export {key}={shlex.quote(str(value))}")
    return "\n".join(lines)


def _hex_to_binary_command(remote_hex_path: str, remote_path: str) -> str:
    return (
        "python3 -S -c "
        f"\"import sys; data=''.join(open({remote_hex_path!r}).read().split()); "
        f'sys.stdout.buffer.write(bytes.fromhex(data))" > {shlex.quote(remote_path)} '
        f"&& rm {shlex.quote(remote_hex_path)}"
    )


def _inline_hex_to_binary_command(hex_data: str, remote_path: str) -> str:
    return (
        f"printf %s {shlex.quote(hex_data)} | "
        'python3 -S -c "import sys; sys.stdout.buffer.write(bytes.fromhex(sys.stdin.read()))" '
        f"> {shlex.quote(remote_path)}"
    )


def _binary_to_hex_command(remote_path: str, start_byte: int | None = None, byte_count: int | None = None) -> str:
    if start_byte is None or byte_count is None:
        script = f"from pathlib import Path; import sys; sys.stdout.write(Path({remote_path!r}).read_bytes().hex())"
        return f"python3 -S -c {shlex.quote(script)}"
    script = (
        f"import sys; f=open({remote_path!r}, 'rb'); f.seek({int(start_byte)}); "
        f"sys.stdout.write(f.read({int(byte_count)}).hex())"
    )
    return f"python3 -S -c {shlex.quote(script)}"


@dataclass(frozen=True)
class TerminalCreateRequest:
    """Provider-to-client create request."""

    image: str | None
    ttl_s: int | float | None
    ready_timeout_s: int | float | None
    workdir: str | None
    env: dict[str, str]
    files: dict[str, str]
    metadata: dict[str, str]
    resources: SandboxResources
    entrypoint: list[str] | None
    provider_options: dict[str, Any]


@dataclass(frozen=True)
class TerminalCreateResponse:
    """Client-to-provider create response."""

    session_id: str
    raw: Any = None


class TerminalServiceClient(Protocol):
    """Cluster-specific terminal service client protocol."""

    async def create(self, request: TerminalCreateRequest) -> TerminalCreateResponse:
        """Create a terminal session and return its service id."""
        ...

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
        """Run a command in a terminal session."""
        ...

    async def exec_long(
        self,
        session_id: str,
        command: str,
        *,
        cwd: str | None,
        env: dict[str, str] | None,
        timeout_s: int | float | None,
        user: str | int | None,
        progress_callback: Callable[[str], None] | None = None,
        idle_timeout_s: int | float | None = None,
    ) -> SandboxExecResult:
        """Run a long command in a terminal session."""
        ...

    async def upload_file(self, session_id: str, source_path: Path, target_path: str) -> None:
        """Upload one local file into a terminal session."""
        ...

    async def download_file(self, session_id: str, source_path: str, target_path: Path) -> None:
        """Download one file from a terminal session."""
        ...

    async def status(self, session_id: str) -> SandboxStatus:
        """Return terminal session status."""
        ...

    async def close(self, session_id: str) -> None:
        """Close a terminal session."""
        ...

    async def aclose(self) -> None:
        """Close client-scoped resources."""
        ...


@dataclass
class _ProviderTerminalSession:
    """Provider-private state stashed on ``SandboxHandle.raw``."""

    session_id: str
    image: str | None
    workdir: str | None
    env: dict[str, str]
    metadata: dict[str, str]
    raw: Any = None


class _UnconfiguredTerminalServiceClient:
    """Default placeholder client used until a cluster adapter is supplied."""

    def __init__(self, config: TerminalServiceConfig) -> None:
        self.config = config

    async def create(self, request: TerminalCreateRequest) -> TerminalCreateResponse:
        raise TerminalCreateError(
            "terminal sandbox provider needs a TerminalServiceClient implementation for this cluster. "
            "Pass client=... in tests, or configure the StarGaze terminal adapter."
        )

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
        raise RuntimeError("terminal sandbox client is not configured")

    async def exec_long(
        self,
        session_id: str,
        command: str,
        *,
        cwd: str | None,
        env: dict[str, str] | None,
        timeout_s: int | float | None,
        user: str | int | None,
        progress_callback: Callable[[str], None] | None = None,
        idle_timeout_s: int | float | None = None,
    ) -> SandboxExecResult:
        raise RuntimeError("terminal sandbox client is not configured")

    async def upload_file(self, session_id: str, source_path: Path, target_path: str) -> None:
        raise RuntimeError("terminal sandbox client is not configured")

    async def download_file(self, session_id: str, source_path: str, target_path: Path) -> None:
        raise RuntimeError("terminal sandbox client is not configured")

    async def status(self, session_id: str) -> SandboxStatus:
        return SandboxStatus.UNKNOWN

    async def close(self, session_id: str) -> None:
        return None

    async def aclose(self) -> None:
        return None


def _merge_env(base: Mapping[str, str], override: Mapping[str, str] | None) -> dict[str, str]:
    env = {str(key): str(value) for key, value in base.items()}
    if override:
        env.update({str(key): str(value) for key, value in override.items()})
    return env


def _normalize_remote_path(path: str) -> str:
    if not path.startswith("/"):
        raise ValueError(f"terminal sandbox paths must be absolute: {path!r}")
    return posixpath.normpath(path)


def _is_probe_success(result: SandboxExecResult, expected_stdout: str | None) -> bool:
    if result.return_code != 0:
        return False
    if expected_stdout is None:
        return True
    return (result.stdout or "").strip() == expected_stdout


class _StarGazeTerminalProvider:
    """Sandbox provider backed by StarGaze's terminal sandbox API."""

    name = "terminal"

    def __init__(
        self,
        *,
        connection: TerminalConnectionConfig | Mapping[str, Any] | None = None,
        operations: TerminalOperationConfig | Mapping[str, Any] | None = None,
        probe: TerminalProbeConfig | Mapping[str, Any] | None = None,
    ) -> None:
        self._connection = coerce_config(connection, TerminalConnectionConfig)
        self._operations = coerce_config(operations, TerminalOperationConfig)
        self._probe = coerce_config(probe, TerminalProbeConfig)
        self._semaphore = asyncio.Semaphore(self._operations.concurrency)

        self._vregion = self._connection.vregion or get_default_region()
        self._region_config = get_region_config(self._vregion)
        self._base_url = self._region_config.terminal_sandbox_url
        self._domain = self._region_config.terminal_sandbox_domain
        self._sandbox_id = self._connection.sandbox_id or self._region_config.terminal_sandbox_id
        self._sandbox_image = (
            str(self._connection.sandbox_image or self._connection.default_image or "").strip() or None
        )

        effective_creator_email = str(self._connection.creator_email or "").strip()
        if not effective_creator_email and not self._connection.creator_jwt_token:
            effective_creator_email = (
                os.environ.get("STARGAZE_CREATOR_EMAIL") or os.environ.get("CREATOR_EMAIL") or ""
            ).strip()
        if not effective_creator_email and not self._connection.creator_jwt_token:
            effective_creator_email = _infer_creator_email_from_git_config()

        self._creator_email = effective_creator_email
        self._creator_username = effective_creator_email.split("@", 1)[0] if effective_creator_email else ""
        self._jwt_token = self._connection.creator_jwt_token
        self._codebase_jwt_token = ""
        self._config_file = Path(self._connection.config_dir) / "terminal_sandbox_config.json"
        self._create_admission_group = f"{self._base_url}|{self._sandbox_id}"
        self._process_start_admission_group = f"{self._base_url}|{self._sandbox_id}|process_start"
        self._jwt_refresh_stop_event: threading.Event | None = None
        self._jwt_refresh_thread: threading.Thread | None = None

    def _request_headers(self, *, json_content: bool = True) -> dict[str, str]:
        headers = {
            "X-Jwt-Token": self._jwt_token,
            # Some terminal sandbox responses advertise brotli while returning
            # payloads that requests cannot decode reliably. Plain responses are
            # small here, so disable compression for control/session calls.
            "Accept-Encoding": "identity",
        }
        if json_content:
            headers["Content-Type"] = "application/json"
        return headers

    def _session_process_start_url(self, session: _TerminalSession) -> str:
        return f"https://{session.session_id}.{session.domain}/api/process/start"

    def _process_start_admission_error(self, exc: TimeoutError) -> SandboxExecResult:
        return SandboxExecResult(
            stdout=None,
            stderr=f"failed to acquire terminal sandbox process-start admission lock: {exc}",
            return_code=TERMINAL_RUNTIME_RETURN_CODE,
            error_type="sandbox",
        )

    def _process_start_jitter_key(
        self,
        session: _TerminalSession,
        *,
        blocking: bool,
        command: str,
    ) -> str:
        return "|".join(
            str(part)
            for part in (
                self._process_start_admission_group,
                session.session_id,
                "blocking" if blocking else "stream",
                hashlib.md5(command.encode("utf-8", errors="replace")).hexdigest()[:12],
            )
        )

    def _session_process_connect_url(self, session: _TerminalSession) -> str:
        return f"https://{session.session_id}.{session.domain}/api/process/connect"

    def _session_upload_url(self, session: _TerminalSession) -> str:
        return f"https://{session.session_id}.{session.domain}/api/fs/upload"

    def _session_download_url(self, session: _TerminalSession) -> str:
        return f"https://{session.session_id}.{session.domain}/api/fs/download"

    def _create_session_url(self) -> str:
        return f"{self._base_url}/sandboxes/{self._sandbox_id}/sessions"

    def _delete_session_url(self, session: _TerminalSession) -> str:
        return f"{session.base_url}/sandboxes/{session.sandbox_id}/sessions/{session.session_id}"

    def _refresh_user_jwt(self, *, ignore_cache: bool = False) -> None:
        if not self._creator_username:
            return
        last_error: Exception | None = None
        retry_attempts = _user_jwt_retry_attempts()
        for attempt in range(retry_attempts):
            try:
                user_token = get_service_account_manager(self._vregion).get_user_token(
                    self._creator_username,
                    ignore_cache=ignore_cache,
                )
                self._jwt_token = user_token.access_token
                return
            except Exception as exc:
                last_error = exc
                if attempt + 1 >= retry_attempts or not _is_transient_user_jwt_error(exc):
                    raise
                delay = _user_jwt_retry_delay(attempt)
                LOGGER.warning(
                    "failed to get terminal sandbox user JWT (region=%s, user=%s), attempt %d/%d; retrying in %.1fs: %s",
                    self._vregion,
                    self._creator_username,
                    attempt + 1,
                    retry_attempts,
                    delay,
                    exc,
                )
                time.sleep(delay)
        assert last_error is not None
        raise last_error

    def _persist_config(self, session: _TerminalSession) -> None:
        if not self._connection.persist_config:
            return
        created_at = session.session_created_at or datetime.now(timezone.utc)
        config = {
            "backend": "terminal",
            "id": session.session_id,
            "session_id": session.session_id,
            "sandbox_id": session.sandbox_id,
            "sandbox_image": session.sandbox_image,
            "base_url": session.base_url,
            "domain": session.domain,
            "vregion": session.vregion,
            "created_at": created_at.isoformat(),
            "ttl_seconds": session.session_ttl,
            "instance_pod_name": session.instance_pod_name,
            "instance_function_id": session.instance_function_id,
        }
        try:
            self._config_file.parent.mkdir(parents=True, exist_ok=True)
            self._config_file.write_text(json.dumps(config, indent=2), encoding="utf-8")
        except OSError as exc:
            LOGGER.warning("failed to write terminal sandbox config: %s", exc)

    def _coerce_session(self, handle: SandboxHandle) -> _TerminalSession:
        if handle.provider_name != self.name:
            raise ValueError(f"handle provider {handle.provider_name!r} does not match {self.name!r}")
        if not isinstance(handle.raw, _TerminalSession):
            raise TypeError("StarGaze terminal handle has invalid raw session")
        return handle.raw

    def _effective_command(
        self,
        session: _TerminalSession,
        command: str,
        *,
        cwd: str | None,
        env: dict[str, str] | None,
    ) -> str:
        parts: list[str] = []
        merged_env = dict(session.env)
        if env:
            merged_env.update(env)
        if merged_env:
            parts.append(_quote_env_exports(merged_env))
        if cwd:
            parts.append(f"cd {shlex.quote(cwd)}")
        parts.append(command)
        return "\n".join(parts)

    def _iter_sse_events(self, response: requests.Response):
        event_type = ""
        for raw_line in response.iter_lines():
            raw_text = raw_line.decode("utf-8", errors="replace") if isinstance(raw_line, bytes) else str(raw_line)
            for line in raw_text.splitlines():
                if not line:
                    continue
                if line.startswith("event:"):
                    event_type = line.split(":", 1)[1].strip()
                    continue
                if not line.startswith("data:"):
                    continue
                data_text = line.split(":", 1)[1].strip()
                if not data_text:
                    continue
                try:
                    yield event_type, json.loads(data_text)
                except json.JSONDecodeError:
                    LOGGER.debug("ignoring invalid terminal sandbox SSE data: %s", data_text[:300])

    def _exec_blocking(
        self,
        session: _TerminalSession,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: int | float | None = None,
        refresh_jwt_on_forbidden: bool = True,
    ) -> SandboxExecResult:
        effective_timeout = int(timeout_s if timeout_s is not None else 180)
        if effective_timeout <= 0:
            raise ValueError("timeout_s must be > 0")
        effective_command = self._effective_command(session, command, cwd=cwd, env=env)
        payload = {
            "command": {
                "path": "/bin/bash",
                "args": ["-c", _with_terminal_runtime_path(effective_command)],
            },
            "timeout": effective_timeout,
            "blocking": True,
        }
        max_attempts = _session_exec_retry_attempts()
        last_error: str | None = None
        jwt_refreshed_after_forbidden = False
        for attempt in range(max_attempts):
            try:
                jitter = _process_start_initial_jitter(
                    self._process_start_jitter_key(session, blocking=True, command=effective_command)
                )
                if jitter:
                    LOGGER.info("terminal sandbox process-start jitter %.2fs before blocking command", jitter)
                    time.sleep(jitter)
                with _ProcessStartAdmission(group=self._process_start_admission_group):
                    response = requests.post(
                        self._session_process_start_url(session),
                        headers=self._request_headers(),
                        json=payload,
                        timeout=effective_timeout + self._operations.exec_request_extra_timeout_s,
                        verify=False,
                    )
                response.raise_for_status()
                data = response.json()
                break
            except TimeoutError as exc:
                return self._process_start_admission_error(exc)
            except requests.exceptions.Timeout as exc:
                return SandboxExecResult(
                    stdout=None,
                    stderr=f"terminal sandbox command timed out after {effective_timeout:g}s: {exc}",
                    return_code=TERMINAL_RUNTIME_RETURN_CODE,
                    error_type="timeout",
                )
            except requests.RequestException as exc:
                last_error = f"terminal sandbox command request failed: {exc}"
                response = getattr(exc, "response", None)
                status_code = int(getattr(response, "status_code", 0)) if response is not None else 0
                if (
                    status_code == 403
                    and refresh_jwt_on_forbidden
                    and self._creator_username
                    and not jwt_refreshed_after_forbidden
                    and attempt + 1 < max_attempts
                ):
                    try:
                        self._refresh_user_jwt(ignore_cache=True)
                        jwt_refreshed_after_forbidden = True
                        LOGGER.warning("terminal sandbox command returned 403; refreshed user JWT and retrying once")
                        continue
                    except Exception as refresh_exc:
                        LOGGER.warning("failed to refresh terminal sandbox user JWT after 403: %s", refresh_exc)
                if attempt + 1 < max_attempts and _is_transient_session_exec_exception(exc):
                    delay = _session_exec_retry_delay(attempt, exc)
                    LOGGER.warning(
                        "terminal sandbox command hit transient runtime error, attempt %d/%d; retrying in %.2fs: %s",
                        attempt + 1,
                        max_attempts,
                        delay,
                        exc,
                    )
                    time.sleep(delay)
                    continue
                return SandboxExecResult(
                    stdout=None,
                    stderr=last_error,
                    return_code=TERMINAL_RUNTIME_RETURN_CODE,
                    error_type="sandbox",
                )
            except json.JSONDecodeError as exc:
                last_error = f"terminal sandbox command returned non-JSON response: {exc}"
                return SandboxExecResult(
                    stdout=None,
                    stderr=last_error,
                    return_code=TERMINAL_RUNTIME_RETURN_CODE,
                    error_type="sandbox",
                )
        else:
            return SandboxExecResult(
                stdout=None,
                stderr=last_error or "terminal sandbox command failed after retries",
                return_code=TERMINAL_RUNTIME_RETURN_CODE,
                error_type="sandbox",
            )

        if "exit_code" not in data:
            message = data.get("message", "terminal sandbox command response missing exit_code")
            return SandboxExecResult(
                stdout=data.get("stdout"),
                stderr=message,
                return_code=TERMINAL_RUNTIME_RETURN_CODE,
                error_type="sandbox",
            )

        exit_code = int(data.get("exit_code", data.get("return_code", 0)))
        error_type = "timeout" if data.get("is_timeout") else None
        return SandboxExecResult(
            stdout=data.get("stdout", ""),
            stderr=data.get("stderr", ""),
            return_code=exit_code,
            error_type=error_type,
        )

    def _long_command_paths(self) -> dict[str, str]:
        job_id = uuid.uuid4().hex
        prefix = f"/tmp/xgym_terminal_long_command_{job_id}"
        return {
            "job_id": job_id,
            "stdout": f"{prefix}.stdout",
            "stderr": f"{prefix}.stderr",
            "status": f"{prefix}.status",
        }

    def _read_remote_text_blocking(
        self,
        session: _TerminalSession,
        remote_path: str,
        *,
        timeout_s: int | float = 120,
    ) -> str:
        result = self._exec_blocking(
            session,
            f"cat {shlex.quote(remote_path)} 2>/dev/null || true",
            timeout_s=timeout_s,
        )
        if result.return_code == 0:
            return str(result.stdout or "")
        return ""

    def _finalize_long_exec_result(
        self,
        session: _TerminalSession,
        paths: dict[str, str],
        *,
        exit_code: int | None,
        stdout_fallback: str = "",
        stderr_fallback: str = "",
        error_type: str | None = None,
    ) -> SandboxExecResult:
        status_text = self._read_remote_text_blocking(session, paths["status"]).strip()
        if status_text:
            for line in reversed([line.strip() for line in status_text.splitlines() if line.strip()]):
                try:
                    exit_code = int(line)
                    break
                except ValueError:
                    continue
        if exit_code is None:
            exit_code = TERMINAL_RUNTIME_RETURN_CODE

        stdout = self._read_remote_text_blocking(session, paths["stdout"], timeout_s=30) or stdout_fallback
        stderr = self._read_remote_text_blocking(session, paths["stderr"], timeout_s=30) or stderr_fallback
        cleanup = "rm -f " + " ".join(shlex.quote(paths[key]) for key in ("stdout", "stderr", "status"))
        self._exec_blocking(session, cleanup, timeout_s=30)
        return SandboxExecResult(
            stdout=stdout,
            stderr=stderr,
            return_code=int(exit_code),
            error_type=error_type,
        )

    def _long_exec_progress(
        self,
        progress_callback: Callable[[str], None] | None,
        message: str,
        *,
        last_reported: float,
        throttle_s: float = 30.0,
    ) -> float:
        now = time.monotonic()
        if progress_callback is None or (throttle_s > 0 and now - last_reported < throttle_s):
            return last_reported
        try:
            progress_callback(message)
        except Exception:
            LOGGER.debug("terminal long-command progress callback failed", exc_info=True)
        return now

    def _poll_long_exec_files(
        self,
        session: _TerminalSession,
        paths: dict[str, str],
        *,
        effective_timeout: int,
        command_started: float,
        progress_callback: Callable[[str], None] | None = None,
        stdout_chunks: list[str] | None = None,
        stderr_chunks: list[str] | None = None,
        pid: int | str | None = None,
        timeout_detail: str = "",
        poll_interval: float = 30.0,
    ) -> SandboxExecResult:
        stdout_chunks = stdout_chunks if stdout_chunks is not None else []
        stderr_chunks = stderr_chunks if stderr_chunks is not None else []
        stdout_seen = len("".join(stdout_chunks))
        stderr_seen = len("".join(stderr_chunks))
        deadline = command_started + effective_timeout
        last_progress = 0.0
        last_output_activity = time.monotonic()
        output_idle_timeout = 20 * 60.0
        while True:
            stdout_snapshot = self._read_remote_text_blocking(session, paths["stdout"], timeout_s=30)
            stderr_snapshot = self._read_remote_text_blocking(session, paths["stderr"], timeout_s=30)
            if len(stdout_snapshot) > stdout_seen:
                chunk = stdout_snapshot[stdout_seen:]
                stdout_chunks.append(chunk)
                stdout_seen = len(stdout_snapshot)
                last_output_activity = time.monotonic()
                last_progress = self._long_exec_progress(
                    progress_callback,
                    chunk,
                    last_reported=last_progress,
                    throttle_s=0.0,
                )
            if len(stderr_snapshot) > stderr_seen:
                chunk = stderr_snapshot[stderr_seen:]
                stderr_chunks.append(chunk)
                stderr_seen = len(stderr_snapshot)
                last_output_activity = time.monotonic()
                last_progress = self._long_exec_progress(
                    progress_callback,
                    chunk,
                    last_reported=last_progress,
                    throttle_s=0.0,
                )

            if time.monotonic() - last_output_activity >= output_idle_timeout:
                if pid is not None:
                    self._exec_blocking(session, f"kill {shlex.quote(str(pid))} >/dev/null 2>&1 || true", timeout_s=30)
                stderr = "".join(stderr_chunks)
                idle_message = "terminal sandbox long command output idle timed out after 1200s"
                stderr = (stderr + "\n" if stderr else "") + idle_message
                return self._finalize_long_exec_result(
                    session,
                    paths,
                    exit_code=TERMINAL_RUNTIME_RETURN_CODE,
                    stdout_fallback="".join(stdout_chunks),
                    stderr_fallback=stderr,
                    error_type="timeout",
                )

            status_text = self._read_remote_text_blocking(session, paths["status"], timeout_s=30).strip()
            if status_text:
                return self._finalize_long_exec_result(
                    session,
                    paths,
                    exit_code=None,
                    stdout_fallback="".join(stdout_chunks),
                    stderr_fallback="".join(stderr_chunks),
                )

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                if pid is not None:
                    self._exec_blocking(session, f"kill {shlex.quote(str(pid))} >/dev/null 2>&1 || true", timeout_s=30)
                stderr = "".join(stderr_chunks)
                if timeout_detail:
                    stderr = (stderr + "\n" if stderr else "") + timeout_detail
                return self._finalize_long_exec_result(
                    session,
                    paths,
                    exit_code=TERMINAL_RUNTIME_RETURN_CODE,
                    stdout_fallback="".join(stdout_chunks),
                    stderr_fallback=stderr or f"terminal sandbox long command timed out after {effective_timeout:g}s",
                    error_type="timeout",
                )
            time.sleep(min(poll_interval, max(1.0, remaining)))

    def _exec_long_blocking(
        self,
        session: _TerminalSession,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: int | float | None = None,
        progress_callback: Callable[[str], None] | None = None,
        idle_timeout_s: int | float | None = None,
    ) -> SandboxExecResult:
        effective_timeout = int(timeout_s if timeout_s is not None else 180)
        if effective_timeout <= 0:
            raise ValueError("timeout_s must be > 0")

        effective_command = self._effective_command(session, command, cwd=cwd, env=env)
        paths = self._long_command_paths()
        stdout_path = shlex.quote(paths["stdout"])
        stderr_path = shlex.quote(paths["stderr"])
        status_path = shlex.quote(paths["status"])
        wrapped_command = (
            f"rm -f {stdout_path} {stderr_path} {status_path}; "
            f"( bash -lc {shlex.quote(_with_terminal_runtime_path(effective_command))}; rc=$?; "
            f'printf \'%s\\n\' "$rc" > {status_path}; exit "$rc" '
            f") < /dev/null > >(tee {stdout_path}) 2> >(tee {stderr_path} >&2)"
        )
        payload = {
            "command": {"path": "/bin/bash", "args": ["-c", wrapped_command]},
            "timeout": effective_timeout,
            "blocking": False,
        }
        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []
        stdout_bytes = 0
        stderr_bytes = 0
        pid: int | str | None = None
        exit_code: int | None = None
        last_activity = time.monotonic()
        last_progress = 0.0
        idle_timeout = float(idle_timeout_s or 0)
        read_timeout = min(60.0, idle_timeout) if idle_timeout > 0 else 60.0
        response: requests.Response | None = None
        try:
            jitter = _process_start_initial_jitter(
                self._process_start_jitter_key(session, blocking=False, command=effective_command)
            )
            if jitter:
                LOGGER.info("terminal sandbox process-start jitter %.2fs before long command", jitter)
                time.sleep(jitter)
            with _ProcessStartAdmission(group=self._process_start_admission_group):
                response = requests.post(
                    self._session_process_start_url(session),
                    headers=self._request_headers(),
                    json=payload,
                    stream=True,
                    timeout=(10, read_timeout),
                    verify=False,
                )
            response.raise_for_status()
            for event_type, data in self._iter_sse_events(response):
                if event_type == "process.start":
                    pid = data.get("pid")
                    last_activity = time.monotonic()
                    last_progress = self._long_exec_progress(
                        progress_callback,
                        f"terminal long command started pid={pid}",
                        last_reported=last_progress,
                    )
                    continue
                if event_type == "process.data":
                    stdout_value = data.get("stdout")
                    stderr_value = data.get("stderr")
                    if stdout_value:
                        stdout_text = str(stdout_value)
                        stdout_chunks.append(stdout_text)
                        stdout_bytes += len(stdout_text.encode("utf-8"))
                        last_activity = time.monotonic()
                        last_progress = self._long_exec_progress(
                            progress_callback,
                            str(stdout_value),
                            last_reported=last_progress,
                            throttle_s=0.0,
                        )
                    if stderr_value:
                        stderr_text = str(stderr_value)
                        stderr_chunks.append(stderr_text)
                        stderr_bytes += len(stderr_text.encode("utf-8"))
                        last_activity = time.monotonic()
                        last_progress = self._long_exec_progress(
                            progress_callback,
                            str(stderr_value),
                            last_reported=last_progress,
                            throttle_s=0.0,
                        )
                    last_progress = self._long_exec_progress(
                        progress_callback,
                        f"terminal long command activity pid={pid or '-'} stdout_bytes={stdout_bytes} stderr_bytes={stderr_bytes}",
                        last_reported=last_progress,
                    )
                    continue
                if event_type == "process.exit":
                    if pid is None:
                        pid = data.get("pid")
                    last_activity = time.monotonic()
                    exit_code = int(data.get("exit_code", 1))
                    error_type = "timeout" if data.get("is_timeout") else None
                    return self._finalize_long_exec_result(
                        session,
                        paths,
                        exit_code=exit_code,
                        stdout_fallback="".join(stdout_chunks),
                        stderr_fallback="".join(stderr_chunks),
                        error_type=error_type,
                    )
        except TimeoutError as exc:
            return self._process_start_admission_error(exc)
        except requests.exceptions.Timeout as exc:
            LOGGER.warning("terminal sandbox long command start stream timed out; polling job files: %s", exc)
            return self._poll_long_exec_files(
                session,
                paths,
                effective_timeout=effective_timeout,
                command_started=last_activity,
                progress_callback=progress_callback,
                stdout_chunks=stdout_chunks,
                stderr_chunks=stderr_chunks,
                pid=pid,
                timeout_detail=f"terminal sandbox long command start stream timed out: {exc}",
            )
        except requests.RequestException as exc:
            if pid is None:
                LOGGER.warning(
                    "terminal sandbox long command start stream failed before pid; polling job files: %s", exc
                )
                return self._poll_long_exec_files(
                    session,
                    paths,
                    effective_timeout=effective_timeout,
                    command_started=last_activity,
                    progress_callback=progress_callback,
                    stdout_chunks=stdout_chunks,
                    stderr_chunks=stderr_chunks,
                    timeout_detail=f"terminal sandbox long command start stream failed: {exc}",
                )
            LOGGER.warning("terminal sandbox long command stream disconnected; reconnecting to pid=%s: %s", pid, exc)
        finally:
            if response is not None:
                response.close()

        if exit_code is not None:
            return self._finalize_long_exec_result(
                session,
                paths,
                exit_code=exit_code,
                stdout_fallback="".join(stdout_chunks),
                stderr_fallback="".join(stderr_chunks),
            )
        if idle_timeout > 0 and time.monotonic() - last_activity >= idle_timeout:
            if pid is not None:
                self._exec_blocking(session, f"kill {shlex.quote(str(pid))} >/dev/null 2>&1 || true", timeout_s=30)
            return self._finalize_long_exec_result(
                session,
                paths,
                exit_code=TERMINAL_RUNTIME_RETURN_CODE,
                stdout_fallback="".join(stdout_chunks),
                stderr_fallback=f"terminal sandbox long command idle timed out after {idle_timeout:g}s",
                error_type="timeout",
            )
        if pid is None:
            return self._finalize_long_exec_result(
                session,
                paths,
                exit_code=TERMINAL_RUNTIME_RETURN_CODE,
                stdout_fallback="".join(stdout_chunks),
                stderr_fallback="terminal sandbox long command did not report a process id",
                error_type="sandbox",
            )

        response = None
        try:
            response = requests.post(
                self._session_process_connect_url(session),
                headers=self._request_headers(),
                json={"pid": pid},
                stream=True,
                timeout=(10, read_timeout),
                verify=False,
            )
            response.raise_for_status()
            for event_type, data in self._iter_sse_events(response):
                if event_type == "process.data":
                    stdout_value = data.get("stdout")
                    stderr_value = data.get("stderr")
                    if stdout_value:
                        stdout_text = str(stdout_value)
                        stdout_chunks.append(stdout_text)
                        stdout_bytes += len(stdout_text.encode("utf-8"))
                        last_activity = time.monotonic()
                        last_progress = self._long_exec_progress(
                            progress_callback,
                            str(stdout_value),
                            last_reported=last_progress,
                            throttle_s=0.0,
                        )
                    if stderr_value:
                        stderr_text = str(stderr_value)
                        stderr_chunks.append(stderr_text)
                        stderr_bytes += len(stderr_text.encode("utf-8"))
                        last_activity = time.monotonic()
                        last_progress = self._long_exec_progress(
                            progress_callback,
                            str(stderr_value),
                            last_reported=last_progress,
                            throttle_s=0.0,
                        )
                    last_progress = self._long_exec_progress(
                        progress_callback,
                        f"terminal long command activity pid={pid or '-'} stdout_bytes={stdout_bytes} stderr_bytes={stderr_bytes}",
                        last_reported=last_progress,
                    )
                    continue
                if event_type == "process.exit":
                    last_activity = time.monotonic()
                    exit_code = int(data.get("exit_code", 1))
                    error_type = "timeout" if data.get("is_timeout") else None
                    return self._finalize_long_exec_result(
                        session,
                        paths,
                        exit_code=exit_code,
                        stdout_fallback="".join(stdout_chunks),
                        stderr_fallback="".join(stderr_chunks),
                        error_type=error_type,
                    )
        except requests.exceptions.Timeout as exc:
            self._exec_blocking(session, f"kill {shlex.quote(str(pid))} >/dev/null 2>&1 || true", timeout_s=30)
            return self._finalize_long_exec_result(
                session,
                paths,
                exit_code=TERMINAL_RUNTIME_RETURN_CODE,
                stdout_fallback="".join(stdout_chunks),
                stderr_fallback=f"terminal sandbox long command timed out after {effective_timeout:g}s: {exc}",
                error_type="timeout",
            )
        except requests.RequestException as exc:
            status_text = self._read_remote_text_blocking(session, paths["status"]).strip()
            if status_text:
                return self._finalize_long_exec_result(
                    session,
                    paths,
                    exit_code=None,
                    stdout_fallback="".join(stdout_chunks),
                    stderr_fallback="".join(stderr_chunks),
                )
            return self._finalize_long_exec_result(
                session,
                paths,
                exit_code=TERMINAL_RUNTIME_RETURN_CODE,
                stdout_fallback="".join(stdout_chunks),
                stderr_fallback=f"terminal sandbox long command reconnect failed: {exc}",
                error_type="sandbox",
            )
        finally:
            if response is not None:
                response.close()

        return self._finalize_long_exec_result(
            session,
            paths,
            exit_code=None,
            stdout_fallback="".join(stdout_chunks),
            stderr_fallback="terminal sandbox long command stream ended without exit status",
            error_type="sandbox",
        )

    async def create(self, spec: SandboxSpec) -> SandboxHandle:
        """Create or attach to a terminal sandbox session."""
        requested_image = str(spec.image or self._sandbox_image or "").strip() or None
        ttl_seconds = int(spec.ttl_s if spec.ttl_s is not None else DEFAULT_TTL_SECONDS)
        if ttl_seconds <= 0:
            raise TerminalCreateError("spec.ttl_s must be > 0 for the terminal provider")

        if self._connection.session_id:
            session = _TerminalSession(
                session_id=self._connection.session_id,
                sandbox_id=self._sandbox_id,
                base_url=self._base_url,
                domain=self._domain,
                vregion=self._vregion,
                sandbox_image=requested_image,
                session_ttl=ttl_seconds,
                env=dict(spec.env),
            )
            handle = SandboxHandle(session.session_id, self.name, session)
            await self._verify_created_handle(handle)
            return handle

        envs = {"PATH": TERMINAL_PATH}
        envs.update(spec.env)
        payload: dict[str, Any] = {
            "ttl": ttl_seconds,
            "envs": envs,
        }
        if requested_image:
            payload["image"] = requested_image

        create_key = "|".join(
            str(part)
            for part in (
                self._create_admission_group,
                requested_image or "",
                ttl_seconds,
                sorted((spec.metadata or {}).items()),
            )
        )
        last_error = "unknown error"
        max_attempts = _session_create_retry_attempts(self._operations.create_retries + 1)
        for attempt in range(max_attempts):
            response: requests.Response | None = None
            try:
                if attempt == 0:
                    jitter = _session_create_initial_jitter(create_key)
                    if jitter:
                        LOGGER.info("terminal sandbox create jitter %.2fs before JWT/create", jitter)
                        await asyncio.sleep(jitter)
                self._refresh_user_jwt(ignore_cache=True)
                with _SessionCreateAdmission(group=self._create_admission_group):
                    response = await asyncio.to_thread(
                        requests.post,
                        self._create_session_url(),
                        headers=self._request_headers(),
                        json=payload,
                        timeout=self._operations.create_request_timeout_s,
                    )
                data = response.json()
            except TimeoutError as exc:
                raise TerminalCreateError(
                    f"failed to acquire terminal sandbox session-create admission lock: {exc}"
                ) from exc
            except UserNotAuthorizedError as exc:
                raise TerminalCreateError(f"failed to get user JWT: {exc}") from exc
            except requests.RequestException as exc:
                last_error = f"create terminal sandbox session request failed: {exc}"
                if attempt + 1 < max_attempts and _is_transient_session_create_exception(exc):
                    await asyncio.sleep(_session_create_retry_delay(attempt, exc))
                    continue
                break
            except json.JSONDecodeError as exc:
                response_text = getattr(response, "text", "")
                status_code = getattr(response, "status_code", 0)
                last_error = f"create terminal sandbox session returned non-JSON response: {exc}"
                if attempt + 1 < max_attempts and _is_transient_session_create_response(status_code, response_text):
                    await asyncio.sleep(_session_create_retry_delay(attempt, response_text or exc))
                    continue
                break

            if response.status_code == 200 and data.get("code") == 0:
                session_data = data.get("data", {})
                session_id = str(session_data.get("session_id") or "").strip()
                if not session_id:
                    raise TerminalCreateError("create terminal sandbox session did not return session_id")

                returned_image = str(session_data.get("image") or "").strip() or None
                if requested_image and returned_image and returned_image != requested_image:
                    session = _TerminalSession(
                        session_id=session_id,
                        sandbox_id=self._sandbox_id,
                        base_url=self._base_url,
                        domain=self._domain,
                        vregion=self._vregion,
                        sandbox_image=returned_image,
                    )
                    await self.close(SandboxHandle(session_id, self.name, session))
                    raise TerminalCreateError(
                        "created terminal sandbox session returned a different image: "
                        f"requested={requested_image}, returned={returned_image}"
                    )

                session = _TerminalSession(
                    session_id=session_id,
                    sandbox_id=self._sandbox_id,
                    base_url=self._base_url,
                    domain=self._domain,
                    vregion=self._vregion,
                    sandbox_image=returned_image or requested_image,
                    session_created_at=datetime.now(timezone.utc),
                    session_ttl=ttl_seconds,
                    instance_pod_name=session_data.get("faas_pod_name"),
                    instance_function_id=session_data.get("faas_function_id"),
                    env=dict(spec.env),
                )
                self._persist_config(session)
                handle = SandboxHandle(session.session_id, self.name, session)
                try:
                    await self._setup_session_auth(handle)
                    await self._verify_created_handle(handle)
                except Exception:
                    await self.close(handle)
                    raise
                if self._connection.refresh_jwt and self._creator_username:
                    self._start_jwt_refresh_thread(session)
                return handle

            message = data.get("message") or getattr(response, "text", "") or "unknown error"
            last_error = f"create terminal sandbox session API returned error: HTTP {response.status_code}, {message}"
            if attempt + 1 < max_attempts and _is_transient_session_create_response(
                response.status_code, getattr(response, "text", "") or message
            ):
                await asyncio.sleep(_session_create_retry_delay(attempt, message))
                continue
            break

        raise TerminalCreateError(last_error)

    async def _setup_session_auth(self, handle: SandboxHandle) -> None:
        if not self._connection.setup_netrc:
            return
        if not self._jwt_token:
            return
        self._codebase_jwt_token = _get_codebase_jwt()
        command = _build_jwt_netrc_command(
            self._jwt_token,
            codebase_jwt_token=self._codebase_jwt_token,
            append=True,
        )
        result = await self.exec(handle, command, timeout_s=30)
        if result.return_code != 0:
            LOGGER.warning("failed to write JWT .netrc in terminal sandbox: %s", result.stderr)

    async def _verify_created_handle(self, handle: SandboxHandle) -> None:
        probe = self._probe
        if probe.command is None:
            return

        loop = asyncio.get_running_loop()
        deadline = loop.time() + probe.deadline_s if probe.deadline_s is not None else None
        consecutive = 0
        last_detail = "no probe attempt completed"

        while True:
            result = await self.exec(handle, probe.command, timeout_s=probe.timeout_s)
            passed = result.return_code == 0 and (
                probe.expected_stdout is None or probe.expected_stdout in (result.stdout or "")
            )
            if passed:
                consecutive += 1
                if consecutive >= probe.stable_count:
                    return
            else:
                consecutive = 0
                last_detail = f"return_code={result.return_code}, stderr={(result.stderr or '').strip()!r}"
                if deadline is None:
                    raise TerminalCreateVerificationError(
                        f"terminal sandbox {handle.sandbox_id!r} failed readiness probe: {last_detail}"
                    )

            if deadline is not None and loop.time() >= deadline:
                raise TerminalCreateVerificationError(
                    f"terminal sandbox {handle.sandbox_id!r} did not pass readiness probe within "
                    f"{probe.deadline_s:g}s: {last_detail}"
                )
            if probe.stable_delay_s > 0:
                await asyncio.sleep(probe.stable_delay_s)

    async def exec(
        self,
        handle: SandboxHandle,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: int | float | None = None,
        user: str | int | None = None,
        refresh_jwt_on_forbidden: bool = True,
    ) -> SandboxExecResult:
        """Run a command in a terminal sandbox session."""
        if user not in (None, "root", 0):
            LOGGER.warning("StarGaze terminal client ignores unsupported user=%r", user)
        session = self._coerce_session(handle)
        async with self._semaphore:
            return await asyncio.to_thread(
                self._exec_blocking,
                session,
                command,
                cwd=cwd,
                env=env,
                timeout_s=timeout_s,
                refresh_jwt_on_forbidden=refresh_jwt_on_forbidden,
            )

    async def exec_long(
        self,
        handle: SandboxHandle,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: int | float | None = None,
        user: str | int | None = None,
        progress_callback: Callable[[str], None] | None = None,
        idle_timeout_s: int | float | None = None,
    ) -> SandboxExecResult:
        """Run a long command using terminal sandbox non-blocking process APIs."""
        if user not in (None, "root", 0):
            LOGGER.warning("StarGaze terminal client ignores unsupported user=%r", user)
        session = self._coerce_session(handle)
        async with self._semaphore:
            return await asyncio.to_thread(
                self._exec_long_blocking,
                session,
                command,
                cwd=cwd,
                env=env,
                timeout_s=timeout_s,
                progress_callback=progress_callback,
                idle_timeout_s=idle_timeout_s,
            )

    async def upload_file(self, handle: SandboxHandle, source_path: Path, target_path: str) -> None:
        session = self._coerce_session(handle)
        source_path = Path(source_path)
        if not source_path.exists():
            raise FileNotFoundError(f"local file does not exist: {source_path}")
        if source_path.stat().st_size <= self._operations.native_file_size_limit:
            try:
                uploaded = await asyncio.to_thread(self._upload_native_blocking, session, source_path, target_path)
            except Exception as exc:
                LOGGER.warning("terminal fs/upload failed; falling back to command transfer: %s", exc)
                uploaded = False
            if uploaded:
                return
        await self._upload_via_commands(handle, source_path, target_path)

    def _upload_native_blocking(self, session: _TerminalSession, source_path: Path, target_path: str) -> bool:
        remote_dir = os.path.dirname(target_path) or "."
        remote_name = os.path.basename(target_path)
        mkdir_result = self._exec_blocking(session, f"mkdir -p {shlex.quote(remote_dir)}", timeout_s=30)
        if mkdir_result.return_code != 0:
            return False

        with source_path.open("rb") as file_obj:
            files = {"file": (remote_name, file_obj, "application/octet-stream")}
            response = requests.post(
                self._session_upload_url(session),
                headers=self._request_headers(json_content=False),
                params={"path": remote_dir},
                files=files,
                timeout=self._operations.upload_timeout_s,
                verify=False,
            )
        response.raise_for_status()
        return True

    async def _upload_via_commands(self, handle: SandboxHandle, source_path: Path, target_path: str) -> None:
        remote_dir = os.path.dirname(target_path)
        if remote_dir:
            result = await self.exec(handle, f"mkdir -p {shlex.quote(remote_dir)}", timeout_s=30)
            if result.return_code != 0:
                raise RuntimeError(f"failed to create remote directory {remote_dir!r}: {result.stderr}")

        source_bytes = source_path.read_bytes()
        if len(source_bytes) * 2 <= self._operations.fallback_chunk_size:
            result = await self.exec(
                handle, _inline_hex_to_binary_command(source_bytes.hex(), target_path), timeout_s=120
            )
            if result.return_code != 0:
                raise RuntimeError(f"terminal upload convert for {target_path!r} failed: {result.stderr}")
            return

        remote_hex = f"{target_path}.hex"
        init_result = await self.exec(
            handle,
            f"rm -f {shlex.quote(target_path)} {shlex.quote(remote_hex)} && touch {shlex.quote(remote_hex)}",
            timeout_s=60,
        )
        if init_result.return_code != 0:
            raise RuntimeError(f"terminal upload init for {target_path!r} failed: {init_result.stderr}")

        chunk_size = max(1, self._operations.fallback_chunk_size // 2)
        total_chunks = 0
        for index in range(0, len(source_bytes), chunk_size):
            total_chunks += 1
            result = await self.exec(
                handle,
                f"printf %s {shlex.quote(source_bytes[index : index + chunk_size].hex())} >> {shlex.quote(remote_hex)}",
                timeout_s=60,
            )
            if result.return_code != 0:
                raise RuntimeError(f"terminal upload chunk {total_chunks} failed: {result.stderr}")

        convert_result = await self.exec(handle, _hex_to_binary_command(remote_hex, target_path), timeout_s=120)
        if convert_result.return_code != 0:
            raise RuntimeError(f"terminal upload convert for {target_path!r} failed: {convert_result.stderr}")

    async def download_file(self, handle: SandboxHandle, source_path: str, target_path: Path) -> None:
        session = self._coerce_session(handle)
        target_path = Path(target_path)
        try:
            downloaded = await asyncio.to_thread(self._download_native_blocking, session, source_path, target_path)
        except FileNotFoundError:
            raise
        except Exception as exc:
            LOGGER.warning("terminal fs/download failed; falling back to command transfer: %s", exc)
            downloaded = False
        if downloaded:
            return
        await self._download_via_commands(handle, source_path, target_path)

    def _download_native_blocking(self, session: _TerminalSession, source_path: str, target_path: Path) -> bool:
        response = requests.post(
            self._session_download_url(session),
            headers=self._request_headers(),
            json={"path": source_path},
            stream=True,
            timeout=(self._operations.download_connect_timeout_s, self._operations.download_read_timeout_s),
            verify=False,
        )
        if response.status_code == 404:
            raise FileNotFoundError(f"remote file does not exist: {source_path}")
        response.raise_for_status()
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with target_path.open("wb") as output:
            for chunk in response.iter_content(chunk_size=1024 * 256):
                if chunk:
                    output.write(chunk)
        return True

    async def _download_via_commands(self, handle: SandboxHandle, source_path: str, target_path: Path) -> None:
        exists = await self.exec(
            handle,
            f"test -f {shlex.quote(source_path)} && echo exists || echo not_found",
            timeout_s=30,
        )
        if exists.return_code != 0 or "not_found" in (exists.stdout or ""):
            raise FileNotFoundError(f"remote file does not exist: {source_path}")

        size_result = await self.exec(handle, f"wc -c < {shlex.quote(source_path)}", timeout_s=30)
        file_size: int | None = None
        if size_result.return_code == 0 and size_result.stdout:
            try:
                file_size = int(size_result.stdout.strip())
            except ValueError:
                file_size = None

        target_path.parent.mkdir(parents=True, exist_ok=True)
        if file_size is not None and file_size > 100000:
            with target_path.open("wb") as output:
                chunk_size = 50000
                for start in range(0, file_size, chunk_size):
                    byte_count = min(chunk_size, file_size - start)
                    result = await self.exec(
                        handle,
                        _binary_to_hex_command(source_path, start, byte_count),
                        timeout_s=120,
                    )
                    if result.return_code != 0:
                        raise RuntimeError(f"terminal download chunk failed: {result.stderr}")
                    output.write(bytes.fromhex((result.stdout or "").strip()))
            return

        result = await self.exec(handle, _binary_to_hex_command(source_path), timeout_s=120)
        if result.return_code != 0:
            raise RuntimeError(f"terminal download from {source_path!r} failed: {result.stderr}")
        target_path.write_bytes(bytes.fromhex((result.stdout or "").strip()))

    async def status(self, handle: SandboxHandle) -> SandboxStatus:
        result = await self.exec(handle, "echo hello", timeout_s=10)
        if result.return_code == 0 and "hello" in (result.stdout or ""):
            return SandboxStatus.RUNNING
        if result.error_type == "sandbox":
            return SandboxStatus.UNKNOWN
        return SandboxStatus.ERROR

    async def close(self, handle: SandboxHandle) -> None:
        session = self._coerce_session(handle)
        self._stop_jwt_refresh_thread()
        try:
            response = await asyncio.to_thread(
                requests.delete,
                self._delete_session_url(session),
                headers=self._request_headers(json_content=False),
                timeout=self._operations.delete_timeout_s,
            )
        except requests.RequestException as exc:
            raise RuntimeError(f"delete terminal sandbox session failed: {exc}") from exc

        if response.status_code == 404:
            return
        try:
            data = response.json()
        except Exception:
            data = {}
        if response.status_code != 200 or data.get("code") not in (None, 0):
            raise RuntimeError(
                f"delete terminal sandbox session failed: HTTP {response.status_code}, {getattr(response, 'text', '')}"
            )

        if self._connection.persist_config:
            try:
                if self._config_file.exists():
                    self._config_file.unlink()
            except OSError:
                pass

    async def aclose(self) -> None:
        self._stop_jwt_refresh_thread()

    def _start_jwt_refresh_thread(self, session: _TerminalSession) -> None:
        self._jwt_refresh_stop_event = threading.Event()
        self._jwt_refresh_thread = threading.Thread(
            target=self._jwt_refresh_loop,
            args=(session,),
            daemon=True,
            name=f"terminal-jwt-refresh-{session.session_id}",
        )
        self._jwt_refresh_thread.start()

    def _stop_jwt_refresh_thread(self) -> None:
        if self._jwt_refresh_stop_event:
            self._jwt_refresh_stop_event.set()
        if self._jwt_refresh_thread and self._jwt_refresh_thread.is_alive():
            self._jwt_refresh_thread.join(timeout=5)

    def _jwt_refresh_loop(self, session: _TerminalSession) -> None:
        ttl_seconds = session.session_ttl or DEFAULT_TTL_SECONDS
        deadline = time.monotonic() + ttl_seconds
        assert self._jwt_refresh_stop_event is not None

        while not self._jwt_refresh_stop_event.is_set():
            try:
                user_token = get_service_account_manager(session.vregion).get_user_token(self._creator_username)
                self._jwt_token = user_token.access_token
                self._codebase_jwt_token = _get_codebase_jwt(ignore_cache=True)
                command = _build_jwt_netrc_command(
                    self._jwt_token,
                    codebase_jwt_token=self._codebase_jwt_token,
                )
                self._exec_blocking(session, command, timeout_s=30)
                remaining_token = (user_token.expires_at - datetime.now()).total_seconds()
                refresh_in = max(remaining_token - 300, 60)
            except Exception as exc:
                LOGGER.warning("terminal sandbox JWT refresh failed: %s", exc)
                refresh_in = 60

            remaining_ttl = deadline - time.monotonic()
            if remaining_ttl <= 0:
                break
            self._jwt_refresh_stop_event.wait(timeout=min(refresh_in, remaining_ttl))


class StarGazeTerminalServiceClient:
    """Terminal service client backed by StarGaze's terminal sandbox API."""

    def __init__(
        self,
        *,
        connection: TerminalConnectionConfig | Mapping[str, Any] | None = None,
        operations: TerminalOperationConfig | Mapping[str, Any] | None = None,
        probe: TerminalProbeConfig | Mapping[str, Any] | None = None,
    ) -> None:
        self._provider = _StarGazeTerminalProvider(connection=connection, operations=operations, probe=probe)
        self._handles: dict[str, SandboxHandle] = {}

    async def create(self, request: TerminalCreateRequest) -> TerminalCreateResponse:
        handle = await self._provider.create(
            SandboxSpec(
                image=request.image,
                ttl_s=request.ttl_s,
                ready_timeout_s=request.ready_timeout_s,
                workdir=request.workdir,
                env=request.env,
                files=request.files,
                metadata=request.metadata,
                resources=request.resources,
                entrypoint=request.entrypoint,
                provider_options=request.provider_options,
            )
        )
        self._handles[handle.sandbox_id] = handle
        return TerminalCreateResponse(session_id=handle.sandbox_id, raw=handle.raw)

    def _handle(self, session_id: str) -> SandboxHandle:
        try:
            return self._handles[session_id]
        except KeyError as exc:
            raise RuntimeError(f"terminal session is not tracked by this client: {session_id}") from exc

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
        return await self._provider.exec(
            self._handle(session_id),
            command,
            cwd=cwd,
            env=env,
            timeout_s=timeout_s,
            user=user,
        )

    async def exec_long(
        self,
        session_id: str,
        command: str,
        *,
        cwd: str | None,
        env: dict[str, str] | None,
        timeout_s: int | float | None,
        user: str | int | None,
        progress_callback: Callable[[str], None] | None = None,
        idle_timeout_s: int | float | None = None,
    ) -> SandboxExecResult:
        return await self._provider.exec_long(
            self._handle(session_id),
            command,
            cwd=cwd,
            env=env,
            timeout_s=timeout_s,
            user=user,
            progress_callback=progress_callback,
            idle_timeout_s=idle_timeout_s,
        )

    async def upload_file(self, session_id: str, source_path: Path, target_path: str) -> None:
        await self._provider.upload_file(self._handle(session_id), source_path, target_path)

    async def download_file(self, session_id: str, source_path: str, target_path: Path) -> None:
        await self._provider.download_file(self._handle(session_id), source_path, target_path)

    async def status(self, session_id: str) -> SandboxStatus:
        return await self._provider.status(self._handle(session_id))

    async def close(self, session_id: str) -> None:
        handle = self._handles.pop(session_id, None)
        if handle is None:
            return
        await self._provider.close(handle)

    async def aclose(self) -> None:
        await self._provider.aclose()


def _stargaze_connection_from_service(service: TerminalServiceConfig) -> TerminalConnectionConfig:
    auth = service.auth or {}
    extra = service.extra or {}
    allowed_extra = set(TerminalConnectionConfig.__dataclass_fields__) - {"config_dir", "vregion"}
    connection_kwargs = {key: value for key, value in extra.items() if key in allowed_extra}
    if "creator_jwt_token" in auth:
        connection_kwargs["creator_jwt_token"] = auth["creator_jwt_token"]
    if "creator_email" in auth:
        connection_kwargs["creator_email"] = auth["creator_email"]
    return TerminalConnectionConfig(
        config_dir=service.config_dir or str(connection_kwargs.pop("config_dir", "config")),
        vregion=service.region or connection_kwargs.pop("vregion", None),
        **connection_kwargs,
    )


def _stargaze_operations_from_configs(
    create: TerminalCreateConfig,
    exec_config: TerminalExecConfig,
    service: TerminalServiceConfig,
) -> TerminalOperationConfig:
    extra = service.extra or {}
    allowed_extra = set(TerminalOperationConfig.__dataclass_fields__)
    operation_kwargs = {key: value for key, value in extra.items() if key in allowed_extra}
    if create.request_timeout_s is not None:
        operation_kwargs.setdefault("create_request_timeout_s", create.request_timeout_s)
    # TerminalSandboxProvider owns create retries for wrapped service clients; keep
    # the StarGaze adapter to one create attempt unless service.extra overrides it.
    operation_kwargs.setdefault("create_retries", 0)
    operation_kwargs.setdefault("create_retry_delay_s", create.retry_delay_s)
    operation_kwargs.setdefault("exec_request_extra_timeout_s", exec_config.request_extra_timeout_s)
    operation_kwargs.setdefault("concurrency", exec_config.concurrency)
    return TerminalOperationConfig(**operation_kwargs)


class TerminalSandboxProvider:
    """Provider-neutral adapter for terminal/session sandbox services."""

    name = "terminal"

    def __init__(
        self,
        *,
        service: TerminalServiceConfig | Mapping[str, Any] | None = None,
        create: TerminalCreateConfig | Mapping[str, Any] | None = None,
        exec: TerminalExecConfig | Mapping[str, Any] | None = None,
        probe: TerminalProbeConfig | Mapping[str, Any] | None = None,
        client: TerminalServiceClient | None = None,
        metadata: Mapping[str, str] | None = None,
        connection: TerminalConnectionConfig | Mapping[str, Any] | None = None,
        operations: TerminalOperationConfig | Mapping[str, Any] | None = None,
    ) -> None:
        self.service = coerce_config(service, TerminalServiceConfig)
        self.create_config = coerce_config(create, TerminalCreateConfig)
        self.exec_config = coerce_config(exec, TerminalExecConfig)
        self.probe_config = coerce_config(probe, TerminalProbeConfig)
        self.metadata = {str(key): str(value) for key, value in (metadata or {}).items()}
        if client is not None:
            self._client = client
        elif connection is not None or operations is not None:
            self._client = StarGazeTerminalServiceClient(connection=connection, operations=operations, probe=probe)
        elif self.service.region is not None or self.service.endpoint == "stargaze":
            self._client = StarGazeTerminalServiceClient(
                connection=_stargaze_connection_from_service(self.service),
                operations=_stargaze_operations_from_configs(self.create_config, self.exec_config, self.service),
                probe=probe,
            )
        else:
            self._client = _UnconfiguredTerminalServiceClient(self.service)
        self._exec_sem = asyncio.Semaphore(self.exec_config.concurrency)

    async def create(self, spec: SandboxSpec) -> SandboxHandle:
        request = TerminalCreateRequest(
            image=spec.image,
            ttl_s=spec.ttl_s,
            ready_timeout_s=spec.ready_timeout_s,
            workdir=spec.workdir,
            env={str(key): str(value) for key, value in spec.env.items()},
            files=dict(spec.files),
            metadata={**self.metadata, **{str(key): str(value) for key, value in spec.metadata.items()}},
            resources=spec.resources,
            entrypoint=list(spec.entrypoint) if spec.entrypoint else None,
            provider_options=dict(spec.provider_options),
        )
        response = await self._create_with_retries(request)
        session = _ProviderTerminalSession(
            session_id=response.session_id,
            image=spec.image,
            workdir=spec.workdir,
            env=request.env,
            metadata=request.metadata,
            raw=response.raw,
        )
        handle = SandboxHandle(sandbox_id=response.session_id, provider_name=self.name, raw=session)
        try:
            await self._verify_ready(handle)
        except Exception:
            with contextlib.suppress(Exception):
                await self.close(handle)
            raise
        return handle

    async def _create_with_retries(self, request: TerminalCreateRequest) -> TerminalCreateResponse:
        last_error: BaseException | None = None
        for attempt in range(self.create_config.retries + 1):
            try:
                return await asyncio.wait_for(
                    self._client.create(request), timeout=self.create_config.request_timeout_s
                )
            except Exception as exc:
                last_error = exc
                if attempt >= self.create_config.retries:
                    break
                LOGGER.warning(
                    "Retrying terminal sandbox create after attempt %s/%s: %r",
                    attempt + 1,
                    self.create_config.retries + 1,
                    exc,
                )
                if self.create_config.retry_delay_s:
                    await asyncio.sleep(self.create_config.retry_delay_s)
        raise TerminalCreateError(f"terminal sandbox create failed: {last_error}") from last_error

    async def _verify_ready(self, handle: SandboxHandle) -> None:
        command = self.probe_config.command
        if command is None:
            return
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.probe_config.deadline_s if self.probe_config.deadline_s is not None else None
        consecutive = 0
        last_detail = "no probe attempt completed"
        while True:
            result = await self.exec(handle, command, timeout_s=self.probe_config.timeout_s)
            if _is_probe_success(result, self.probe_config.expected_stdout):
                consecutive += 1
                if consecutive >= self.probe_config.stable_count:
                    return
            else:
                consecutive = 0
                last_detail = f"return_code={result.return_code}, stdout={result.stdout!r}, stderr={result.stderr!r}"
                if deadline is None:
                    raise TerminalCreateVerificationError(f"terminal sandbox readiness probe failed: {last_detail}")
            if deadline is not None and loop.time() >= deadline:
                raise TerminalCreateVerificationError(
                    f"terminal sandbox readiness probe did not pass within {self.probe_config.deadline_s:g}s: "
                    f"{last_detail}"
                )
            if self.probe_config.stable_delay_s > 0:
                await asyncio.sleep(self.probe_config.stable_delay_s)

    def _session(self, handle: SandboxHandle) -> _ProviderTerminalSession:
        if handle.provider_name != self.name:
            raise ValueError(f"Handle provider {handle.provider_name!r} does not match {self.name!r}")
        if not isinstance(handle.raw, _ProviderTerminalSession):
            raise TypeError("Terminal sandbox handle has unexpected raw state")
        return handle.raw

    async def exec(
        self,
        handle: SandboxHandle,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: int | float | None = None,
        user: str | int | None = None,
    ) -> SandboxExecResult:
        session = self._session(handle)
        merged_env = _merge_env(session.env, env)
        timeout = timeout_s if timeout_s is not None else self.exec_config.default_timeout_s
        async with self._exec_sem:
            try:
                return await self._client.exec(
                    session.session_id,
                    command,
                    cwd=cwd or session.workdir,
                    env=merged_env,
                    timeout_s=timeout,
                    user=user,
                )
            except Exception as exc:
                return SandboxExecResult(
                    stdout=None,
                    stderr=str(exc),
                    return_code=TERMINAL_RUNTIME_RETURN_CODE,
                    error_type=type(exc).__name__,
                )

    async def exec_long(
        self,
        handle: SandboxHandle,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: int | float | None = None,
        user: str | int | None = None,
        progress_callback: Callable[[str], None] | None = None,
        idle_timeout_s: int | float | None = None,
    ) -> SandboxExecResult:
        session = self._session(handle)
        merged_env = _merge_env(session.env, env)
        timeout = timeout_s if timeout_s is not None else self.exec_config.default_timeout_s
        exec_long = getattr(self._client, "exec_long", None)
        if not callable(exec_long):
            return await self.exec(handle, command, cwd=cwd, env=env, timeout_s=timeout_s, user=user)
        async with self._exec_sem:
            try:
                return await exec_long(
                    session.session_id,
                    command,
                    cwd=cwd or session.workdir,
                    env=merged_env,
                    timeout_s=timeout,
                    user=user,
                    progress_callback=progress_callback,
                    idle_timeout_s=idle_timeout_s,
                )
            except Exception as exc:
                return SandboxExecResult(
                    stdout=None,
                    stderr=str(exc),
                    return_code=TERMINAL_RUNTIME_RETURN_CODE,
                    error_type=type(exc).__name__,
                )

    async def upload_file(self, handle: SandboxHandle, source_path: Path, target_path: str) -> None:
        session = self._session(handle)
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        await self._client.upload_file(session.session_id, source_path, _normalize_remote_path(target_path))

    async def download_file(self, handle: SandboxHandle, source_path: str, target_path: Path) -> None:
        session = self._session(handle)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        await self._client.download_file(session.session_id, _normalize_remote_path(source_path), target_path)

    async def status(self, handle: SandboxHandle) -> SandboxStatus:
        session = self._session(handle)
        return await self._client.status(session.session_id)

    async def close(self, handle: SandboxHandle) -> None:
        session = self._session(handle)
        await self._client.close(session.session_id)

    async def aclose(self) -> None:
        await self._client.aclose()


TerminalProvider = TerminalSandboxProvider


__all__ = [
    "StarGazeTerminalServiceClient",
    "TerminalConnectionConfig",
    "TerminalCreateConfig",
    "TerminalCreateError",
    "TerminalCreateRequest",
    "TerminalCreateResponse",
    "TerminalCreateVerificationError",
    "TerminalExecConfig",
    "TerminalOperationConfig",
    "TerminalProbeConfig",
    "TerminalProvider",
    "TerminalSandboxProvider",
    "TerminalServiceClient",
    "TerminalServiceConfig",
    "UserNotAuthorizedError",
    "UserToken",
    "TokenFetchError",
    "get_all_regions",
    "get_default_region",
    "get_region_config",
]
