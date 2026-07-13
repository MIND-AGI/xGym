#!/usr/bin/env python3
"""SGF-style direct SWE-bench Pro terminal/opencode rollout runner.

This runner is intentionally separate from the SWE-bench Verified direct runner.
It uses the SWE-bench Pro StarGaze/FaaS image map as the source of truth,
creates one terminal sandbox per instance with a 4h TTL, runs Opencode in
/app, sanitizes the collected patch, and evaluates with SWE-bench Pro run and
parser scripts. Infrastructure failures are tracked separately from valid model
outcomes so failed sandbox creation/upload does not masquerade as model score.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shlex
import subprocess
import sys
import threading
import time
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen
from uuid import uuid4

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from datasets import load_dataset  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional for datasets-library loading mode
    load_dataset = None


def _import_runtime_modules() -> dict[str, Any]:
    from omegaconf import OmegaConf

    from nemo_gym.global_config import GlobalConfigDictParser
    from nemo_gym.sandbox import (
        Sandbox,
        SandboxResources,
        SandboxSpec,
        resolve_provider_config,
        resolve_provider_metadata,
    )
    from responses_api_agents.mini_swe_agent_2.app import (
        _StageProgressRecorder,
        _default_response_object,
        _extract_opencode_session_id,
        _opencode_run_command,
        _opencode_setup_steps,
        _redact_opencode_secrets,
        _sandbox_exec_result,
    )

    return {
        "OmegaConf": OmegaConf,
        "GlobalConfigDictParser": GlobalConfigDictParser,
        "Sandbox": Sandbox,
        "SandboxResources": SandboxResources,
        "SandboxSpec": SandboxSpec,
        "resolve_provider_config": resolve_provider_config,
        "resolve_provider_metadata": resolve_provider_metadata,
        "_StageProgressRecorder": _StageProgressRecorder,
        "_default_response_object": _default_response_object,
        "_extract_opencode_session_id": _extract_opencode_session_id,
        "_opencode_run_command": _opencode_run_command,
        "_opencode_setup_steps": _opencode_setup_steps,
        "_redact_opencode_secrets": _redact_opencode_secrets,
        "_sandbox_exec_result": _sandbox_exec_result,
    }

DEFAULT_IMAGE_MAP = Path("evaluation/swebench_pro/configs/faas_instance_images.json")
DEFAULT_OFFICIAL_ASSETS_DIR = Path("evaluation/swebench_pro/official/SWE-bench_Pro-os")
DATASET_ROWS_ENDPOINT = "https://datasets-server.huggingface.co/rows"
PRO_REPO_PATH = "/app"
PRO_RUNTIME_PATCH_PATHS = {"auth.yaml"}
PRO_RUNTIME_PATCH_PREFIXES = ("appendonlydir/",)
_PRINT_LOCK = threading.Lock()


class ProRunError(RuntimeError):
    def __init__(self, stage: str, reason: str, *, category: str = "model_failure") -> None:
        super().__init__(reason)
        self.stage = stage
        self.category = category


def _log(message: str) -> None:
    with _PRINT_LOCK:
        print(message, flush=True)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False).encode("utf-8") + b"\n")
        fh.flush()


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _parse_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(text)
            except Exception:
                continue
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        return [text]
    return [str(value)]


def _load_image_map(path: Path) -> dict[str, Any]:
    payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
    images = payload.get("images")
    if not isinstance(images, dict):
        raise ValueError(f"image map must contain an images object: {path}")
    return payload


def _dataset_request(*, dataset_name: str, split: str, dataset_config: str, offset: int, length: int) -> list[dict[str, Any]]:
    query = urlencode(
        {
            "dataset": dataset_name,
            "config": dataset_config,
            "split": split,
            "offset": offset,
            "length": length,
        }
    )
    with urlopen(f"{DATASET_ROWS_ENDPOINT}?{query}", timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))
    normalized: list[dict[str, Any]] = []
    for item in payload.get("rows") or []:
        if isinstance(item, dict) and isinstance(item.get("row"), dict):
            normalized.append(item["row"])
    return normalized


def _load_rows_for_instances(args: argparse.Namespace, instance_ids: list[str]) -> list[dict[str, Any]]:
    if args.input is not None:
        by_id = {str(row.get("instance_id")): row for row in _iter_jsonl(args.input)}
        missing = [iid for iid in instance_ids if iid not in by_id]
        if missing:
            raise RuntimeError(f"input file is missing {len(missing)} requested instances; first missing={missing[0]}")
        return [dict(by_id[iid]) for iid in instance_ids]

    if args.use_datasets_library:
        ds = load_dataset(args.dataset, args.dataset_config, split=args.split)
        by_id = {str(item["instance_id"]): dict(item) for item in ds}
        missing = [iid for iid in instance_ids if iid not in by_id]
        if missing:
            raise RuntimeError(f"dataset is missing {len(missing)} requested instances; first missing={missing[0]}")
        return [dict(by_id[iid]) for iid in instance_ids]

    pending = set(instance_ids)
    found: dict[str, dict[str, Any]] = {}
    offset = 0
    while pending:
        batch = _dataset_request(
            dataset_name=args.dataset,
            split=args.split,
            dataset_config=args.dataset_config,
            offset=offset,
            length=100,
        )
        if not batch:
            break
        for row in batch:
            iid = row.get("instance_id")
            if isinstance(iid, str) and iid in pending:
                found[iid] = row
                pending.remove(iid)
        offset += len(batch)
    if pending:
        first = sorted(pending)[0]
        raise RuntimeError(f"failed to load {len(pending)} dataset rows; first missing={first}")
    return [dict(found[iid]) for iid in instance_ids]


def _materialize_rows(rows: list[dict[str, Any]], image_map: dict[str, str], args: argparse.Namespace) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for i, raw in enumerate(rows):
        row = dict(raw)
        instance_id = str(row.get("instance_id") or "")
        image = image_map.get(instance_id)
        if not image:
            raise RuntimeError(f"missing SWE-bench Pro StarGaze image for {instance_id}")
        row["sandbox_image"] = image
        row["subset"] = str(args.subset or row.get("subset") or "swebench_pro")
        row["split"] = str(row.get("split") or args.split)
        row["responses_create_params"] = {
            "input": [],
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_output_tokens": args.max_output_tokens,
            "metadata": {
                "extra_body": json.dumps(
                    {
                        "top_k": args.top_k,
                        "min_p": args.min_p,
                        "presence_penalty": args.presence_penalty,
                        "repetition_penalty": args.repetition_penalty,
                    }
                ),
                "chat_template_kwargs": json.dumps({"enable_thinking": args.enable_thinking}),
            },
        }
        row["agent_ref"] = {"name": args.agent_name}
        row["_ng_task_index"] = i
        row["_ng_rollout_index"] = 0
        prepared.append(row)
    return prepared


def _merge_config(args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    cfg = OmegaConf.merge(OmegaConf.load(args.agent_config), OmegaConf.load(args.model_config))
    cfg.policy_model_name = args.model
    if args.model_url is not None:
        cfg.policy_base_url = args.model_url
    cfg.policy_api_key = args.api_key or os.environ.get("LITELLM_KEY") or os.environ.get("POLICY_API_KEY") or "dummy_key"
    parsed = GlobalConfigDictParser().parse_no_environment(cfg)
    agent_cfg = OmegaConf.to_container(parsed[args.agent_name]["responses_api_agents"][args.agent_name], resolve=True)
    agent_cfg["concurrency"] = args.concurrency
    agent_cfg["sandbox_provider"]["terminal"]["operations"]["concurrency"] = args.terminal_concurrency
    agent_cfg.setdefault("sandbox_spec", {})["ttl_s"] = args.sandbox_ttl_s
    agent_cfg.setdefault("sandbox_spec", {})["ready_timeout_s"] = args.ready_timeout_s
    agent_cfg.setdefault("sandbox_environment_kwargs", {})["cwd"] = args.workdir
    agent_cfg.setdefault("sandbox_environment_kwargs", {})["user"] = args.user
    agent_cfg["opencode_api_key"] = args.api_key or os.environ.get("LITELLM_KEY") or agent_cfg.get("opencode_api_key")
    if args.model_url is not None:
        agent_cfg["opencode_provider_api_url"] = args.model_url
    agent_cfg["opencode_idle_timeout"] = args.opencode_idle_timeout
    agent_cfg["opencode_command_timeout"] = args.opencode_command_timeout
    agent_cfg["opencode_infra_retries"] = args.infra_retries
    return parsed, dict(agent_cfg)


def _pro_prompt(instance: dict[str, Any], *, repo_path: str, max_diff_lines: int) -> str:
    requirements = str(instance.get("requirements") or "").strip()
    interface = str(instance.get("interface") or "").strip()
    return f"""You are working inside a checked-out repository at `{repo_path}`.

You will be given a SWE-bench Pro issue. Solve it by making real code changes in this repository.

<issue>
Instance ID: {instance.get('instance_id')}
Repository: {instance.get('repo')}
Base commit: {instance.get('base_commit')}

Issue statement:
{str(instance.get('problem_statement') or '').strip()}
</issue>

<requirements>
{requirements}
</requirements>

<interface>
{interface}
</interface>

Requirements:
1. Work directly in `{repo_path}`.
2. You must actually modify files in this repository before you finish. Do not only describe a fix.
3. Keep the change as small and focused as possible. Try to stay within about {max_diff_lines} changed diff lines unless the issue clearly needs more.
4. Run the minimal relevant tests or checks when possible.
5. The evaluation harness will collect the real working-tree diff from `{repo_path}` after your final response. Do not hand-write, print, or save a patch file.

Final response contract:
- Output exactly one JSON object, with no Markdown fences or extra prose.
- Use exactly two keys: `explanation` and `failure_reason`.
- `failure_reason`: JSON `null` only if you made a real repository fix; otherwise a short string.
"""


class EnvAdapter:
    def __init__(
        self,
        sandbox_obj: Sandbox,
        *,
        cwd: str,
        user: str,
        step_timeout: int,
        eval_timeout: int,
        idle_timeout_s: int,
        instance_dir: Path,
        progress: _StageProgressRecorder,
    ) -> None:
        self._sandbox = sandbox_obj
        self.cwd = cwd
        self.user = user
        self.step_timeout = step_timeout
        self.eval_timeout = eval_timeout
        self.idle_timeout_s = idle_timeout_s
        self.instance_dir = instance_dir
        self.progress = progress

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
        if background and hasattr(self._sandbox, "exec_long"):
            result = self._sandbox.exec_long(
                command,
                cwd=effective_cwd,
                timeout_s=effective_timeout,
                user=self.user,
                progress_callback=lambda chunk: self.progress.append_opencode_session_log(str(chunk or "")),
                idle_timeout_s=self.idle_timeout_s,
            )
            return _sandbox_exec_result(result)
        result = self._sandbox.exec(command, cwd=effective_cwd, timeout_s=effective_timeout, user=self.user)
        return _sandbox_exec_result(result)


def _run_checked(env: EnvAdapter, command: str, *, timeout: int, cwd: str | None = None, is_eval: bool = False) -> str:
    result = env.execute(command, cwd=cwd or env.cwd, timeout=timeout, is_eval=is_eval)
    if int(result.get("returncode", 1)) != 0:
        output = str(result.get("output") or "")
        raise RuntimeError(output or f"command failed with exit code {result.get('returncode')}")
    return str(result.get("output") or "")


def _here_doc(path: str, content: str) -> str:
    marker = f"XGYM_EOF_{uuid4().hex}"
    return f"cat > {shlex.quote(path)} <<'{marker}'\n{content}\n{marker}"


def _remote_python3_fallback_script() -> str:
    version_check = shlex.quote("import sys; raise SystemExit(0 if sys.version_info[0] == 3 else 1)")
    return (
        "if ! command -v python3 >/dev/null 2>&1; then "
        "candidate=''; "
        "for path in /openhands/micromamba/envs/openhands/bin/python /usr/local/bin/python /usr/bin/python /opt/conda/bin/python /root/.local/bin/python; do "
        f"if [ -x \"$path\" ] && \"$path\" -c {version_check} >/dev/null 2>&1; then candidate=\"$path\"; break; fi; "
        "done; "
        f"if [ -z \"$candidate\" ] && command -v python >/dev/null 2>&1 && python -c {version_check} >/dev/null 2>&1; then candidate=\"$(command -v python)\"; fi; "
        "if [ -n \"$candidate\" ]; then mkdir -p /usr/local/bin 2>/dev/null || true; ln -sf \"$candidate\" /usr/local/bin/python3 2>/dev/null || true; fi; "
        "fi; command -v python3"
    )


def _toolchain_command_link_script() -> str:
    return (
        "mkdir -p /usr/local/bin; "
        "for dir in /usr/local/go/bin /usr/local/cargo/bin /usr/local/mvnd/bin /opt/java/openjdk/bin; do "
        "[ -d \"$dir\" ] || continue; "
        "for src in \"$dir\"/*; do [ -f \"$src\" ] && [ -x \"$src\" ] || continue; "
        "ln -sf \"$src\" \"/usr/local/bin/$(basename \"$src\")\"; done; done"
    )


def _go_sqlite3_alpine_cgo_script() -> str:
    return (
        "if [ -f /etc/alpine-release ] && command -v go >/dev/null 2>&1 && "
        "[ -f /app/go.sum ] && grep -q 'github.com/mattn/go-sqlite3' /app/go.sum; then "
        "case \" ${CGO_CFLAGS:-} \" in *\" -D_LARGEFILE64_SOURCE \"*) ;; "
        "*) export CGO_CFLAGS=\"${CGO_CFLAGS:+$CGO_CFLAGS }-D_LARGEFILE64_SOURCE\" ;; esac; fi"
    )


def _preflight_steps() -> list[tuple[str, str, int]]:
    return [
        ("preflight:repo", f"test -d {shlex.quote(PRO_REPO_PATH)} && git -C {shlex.quote(PRO_REPO_PATH)} status --short --branch", 120),
        ("preflight:toolchain_links", _toolchain_command_link_script(), 120),
        ("preflight:python3", _remote_python3_fallback_script(), 120),
        ("preflight:go_sqlite3", _go_sqlite3_alpine_cgo_script(), 120),
        ("preflight:node", "command -v node && command -v npm", 120),
    ]


def _strip_binary_hunks(patch: str) -> str:
    sections = re.split(r"(?=^diff --git )", patch, flags=re.MULTILINE)
    kept: list[str] = []
    for section in sections:
        if not section.strip():
            continue
        if re.search(r"^Binary files .* differ$", section, re.MULTILINE):
            continue
        if re.search(r"^GIT binary patch$", section, re.MULTILINE):
            continue
        kept.append(section)
    return "".join(kept)


def _normalize_diff_path(path: str) -> str:
    path = path.strip()
    if path in {"/dev/null", "dev/null"}:
        return ""
    if path.startswith(("a/", "b/")):
        return path[2:]
    return path


def _diff_section_paths(section: str) -> set[str]:
    paths: set[str] = set()
    lines = section.splitlines()
    if not lines:
        return paths
    match = re.match(r"^diff --git (.+?) (.+)$", lines[0])
    if match:
        paths.update(_normalize_diff_path(raw) for raw in match.groups() if _normalize_diff_path(raw))
    for line in lines:
        if line.startswith(("--- ", "+++ ")):
            raw_path = line[4:].split("\t", 1)[0]
            normalized = _normalize_diff_path(raw_path)
            if normalized:
                paths.add(normalized)
    return paths


def _is_runtime_patch_path(path: str) -> bool:
    return path in PRO_RUNTIME_PATCH_PATHS or any(path.startswith(prefix) for prefix in PRO_RUNTIME_PATCH_PREFIXES)


def _sanitize_swebench_pro_patch(patch: str) -> str:
    patch = _strip_binary_hunks(patch)
    sections = re.split(r"(?=^diff --git )", patch, flags=re.MULTILINE)
    kept: list[str] = []
    for section in sections:
        if not section.strip():
            continue
        if any(_is_runtime_patch_path(path) for path in _diff_section_paths(section)):
            continue
        kept.append(section)
    return "".join(kept)


def _read_official_asset(args: argparse.Namespace, relative_path: str) -> str:
    path = Path(args.official_assets_dir).expanduser() / relative_path
    return path.read_text(encoding="utf-8")


def _official_or_row_asset(instance: dict[str, Any], args: argparse.Namespace, key: str, relative_path: str) -> str:
    raw = str(instance.get(key) or "")
    if raw.strip():
        return raw
    return _read_official_asset(args, relative_path)


def _extract_env_commands(*dockerfile_contents: str) -> str:
    env_cmds: list[str] = []
    for dockerfile_content in dockerfile_contents:
        for line in dockerfile_content.splitlines():
            stripped = line.strip()
            if stripped.startswith("ENV"):
                env_cmds.append(stripped.replace("ENV", "export", 1))
    return "\n".join(env_cmds)


def _run_pro_eval(instance: dict[str, Any], env: EnvAdapter, instance_dir: Path, model_patch: str, args: argparse.Namespace) -> dict[str, Any]:
    run_id = f"{int(time.time())}_{uuid4()}"
    sanitized_patch = _sanitize_swebench_pro_patch(model_patch)
    if not sanitized_patch.strip():
        raise ProRunError("patch_collect", "sanitized SWE-bench Pro patch is empty", category="model_failure")
    patch_path = instance_dir / f"opencode_patch_{run_id}.diff"
    patch_path.write_text(sanitized_patch, encoding="utf-8")

    instance_id = str(instance["instance_id"])
    run_script = _official_or_row_asset(instance, args, "run_script", f"run_scripts/{instance_id}/run_script.sh")
    parser_script = _official_or_row_asset(instance, args, "parser_script", f"run_scripts/{instance_id}/parser.py")
    base_dockerfile = ""
    instance_dockerfile = ""
    try:
        base_dockerfile = _read_official_asset(args, f"dockerfiles/base_dockerfile/{instance_id}/Dockerfile")
        instance_dockerfile = _read_official_asset(args, f"dockerfiles/instance_dockerfile/{instance_id}/Dockerfile")
    except FileNotFoundError:
        pass
    before_repo_set_cmd = str(instance.get("before_repo_set_cmd") or "").strip().splitlines()
    setup_cmd = before_repo_set_cmd[-1] if before_repo_set_cmd else ":"
    selected_tests = ",".join(_parse_list(instance.get("selected_test_files_to_run")))
    env_cmds = _extract_env_commands(base_dockerfile, instance_dockerfile)

    _run_checked(env, "mkdir -p /workspace", timeout=60, cwd=args.workdir)
    _run_checked(env, _here_doc("/workspace/patch.diff", sanitized_patch), timeout=120, cwd=args.workdir)
    _run_checked(env, _here_doc("/workspace/run_script.sh", run_script), timeout=120, cwd=args.workdir)
    _run_checked(env, _here_doc("/workspace/parser.py", parser_script), timeout=120, cwd=args.workdir)
    entryscript = f"""set -e
{env_cmds}
cd {shlex.quote(args.workdir)}
git config --global --add safe.directory {shlex.quote(args.workdir)} || true
if [ -s /workspace/patch.diff ]; then
  git apply -v /workspace/patch.diff
fi
{_go_sqlite3_alpine_cgo_script()}
{setup_cmd}
set +e
bash /workspace/run_script.sh {shlex.quote(selected_tests)} > /workspace/stdout.log 2> /workspace/stderr.log
rc=$?
printf '%s\n' "$rc" > /workspace/status.txt
python3 /workspace/parser.py /workspace/stdout.log /workspace/stderr.log /workspace/parsed_tests.json > /workspace/parser_stdout.log 2> /workspace/parser_stderr.log
parser_rc=$?
printf '%s\n' "$parser_rc" > /workspace/parser_status.txt
exit 0
"""
    _run_checked(env, _here_doc("/workspace/entryscript.sh", entryscript), timeout=120, cwd=args.workdir)
    _run_checked(env, "chmod +x /workspace/run_script.sh /workspace/entryscript.sh", timeout=60, cwd=args.workdir)
    eval_result = env.execute(
        "bash /workspace/entryscript.sh",
        cwd=args.workdir,
        timeout=args.eval_timeout,
        is_eval=True,
        background=bool(args.eval_background),
    )
    if int(eval_result.get("returncode", 1)) not in (0,):
        raise ProRunError("eval_execute", str(eval_result.get("output") or "SWE-bench Pro eval command failed"), category="eval_failure")

    stdout_local = instance_dir / f"test_stdout_{run_id}.txt"
    stderr_local = instance_dir / f"test_stderr_{run_id}.txt"
    parsed_local = instance_dir / f"parsed_tests_{run_id}.json"
    parser_stdout_local = instance_dir / f"parser_stdout_{run_id}.txt"
    parser_stderr_local = instance_dir / f"parser_stderr_{run_id}.txt"
    for remote, local in (
        ("/workspace/stdout.log", stdout_local),
        ("/workspace/stderr.log", stderr_local),
        ("/workspace/parsed_tests.json", parsed_local),
        ("/workspace/parser_stdout.log", parser_stdout_local),
        ("/workspace/parser_stderr.log", parser_stderr_local),
    ):
        try:
            env._sandbox.download(remote, local)
        except Exception:
            local.write_text(env.execute(f"cat {shlex.quote(remote)}", timeout=120).get("output", ""), encoding="utf-8", errors="replace")

    if not parsed_local.read_text(encoding="utf-8", errors="replace").strip():
        raise ProRunError("eval_parse", "SWE-bench Pro parser produced no parsed_tests.json", category="eval_failure")
    parsed = json.loads(parsed_local.read_text(encoding="utf-8"))
    tests = parsed.get("tests") if isinstance(parsed, dict) else []
    if not isinstance(tests, list):
        tests = []
    status_by_name = {str(t.get("name")): str(t.get("status")) for t in tests if isinstance(t, dict)}
    fail_to_pass = _parse_list(instance.get("fail_to_pass") or instance.get("FAIL_TO_PASS"))
    pass_to_pass = _parse_list(instance.get("pass_to_pass") or instance.get("PASS_TO_PASS"))
    f2p_success = [name for name in fail_to_pass if status_by_name.get(name) == "PASSED"]
    f2p_failure = [name for name in fail_to_pass if status_by_name.get(name) != "PASSED"]
    p2p_success = [name for name in pass_to_pass if status_by_name.get(name) == "PASSED"]
    p2p_failure = [name for name in pass_to_pass if status_by_name.get(name) != "PASSED"]
    resolved = bool(fail_to_pass) and not f2p_failure and not p2p_failure
    return {
        "instance_id": instance_id,
        "resolved": resolved,
        "tests_status": {
            "FAIL_TO_PASS": {"success": f2p_success, "failure": f2p_failure},
            "PASS_TO_PASS": {"success": p2p_success, "failure": p2p_failure},
        },
        "model_patch": sanitized_patch,
        "raw_model_patch_bytes": len(model_patch.encode("utf-8")),
        "sanitized_model_patch_bytes": len(sanitized_patch.encode("utf-8")),
        "num_parsed_tests": len(tests),
        "test_output_path": str(stdout_local),
        "test_stderr_path": str(stderr_local),
        "parsed_tests_path": str(parsed_local),
    }


def _classify_error(exc: BaseException, stage: str | None = None) -> tuple[str, str]:
    if isinstance(exc, ProRunError):
        return exc.category, exc.stage
    normalized_stage = stage or "unknown"
    if normalized_stage == "sandbox_create" or normalized_stage.startswith(("preflight:", "opencode_setup:", "prompt_write")):
        return "infra_failure", normalized_stage
    text = str(exc).lower()
    infra_markers = (
        "terminal sandbox create failed",
        "terminal sandbox session",
        "sandbox_infra_failure",
        "sandbox.upload",
        "terminal upload",
        "terminal fs/upload",
        "terminal fs/download",
        "504 server error",
        "503 server error",
        "502 server error",
        "500 server error",
        "read timed out",
        "timed out waiting for the sync loop",
        "ssleoferror",
        "connectionerror",
        "max retries exceeded",
        "process/start",
        "process/connect",
        "long command stream",
        "jwt",
    )
    if any(marker in text for marker in infra_markers):
        return "infra_failure", stage or "infra"
    if "opencode completed without producing repository changes" in text or "sanitized swe-bench pro patch is empty" in text:
        return "model_failure", stage or "patch_collect"
    if "parser" in text or "test patch" in text or "parsed_tests" in text:
        return "eval_failure", stage or "eval"
    return "model_failure", stage or "unknown"


def _build_failure_result(
    *,
    row: dict[str, Any],
    args: argparse.Namespace,
    exc: BaseException,
    category: str,
    stage: str,
    progress: _StageProgressRecorder | None,
    attempt_count: int,
) -> dict[str, Any]:
    instance_id = str(row.get("instance_id") or "unknown")
    response_obj = _default_response_object()
    response_obj["model"] = args.model
    response_obj["temperature"] = args.temperature
    response_obj["top_p"] = args.top_p
    response_obj["output"] = []
    metadata = {
        "failure_category": category,
        "failure_stage": stage,
        "attempt_count": attempt_count,
        "eval_report": {"error": str(exc), "traceback": traceback.format_exc()},
        "sandbox_image": row.get("sandbox_image"),
        "agent_framework": "opencode",
    }
    if progress is not None:
        metadata.update(progress.paths())
    return {
        "responses_create_params": {**row.get("responses_create_params", {}), "input": []},
        "reward": 0.0,
        "response": response_obj,
        "instance_id": instance_id,
        "metadata": metadata,
        "_ng_task_index": row.get("_ng_task_index", 0),
        "_ng_rollout_index": row.get("_ng_rollout_index", 0),
        "agent_ref": row.get("agent_ref", {"name": args.agent_name}),
    }


def _run_one_attempt(
    *,
    row: dict[str, Any],
    global_config_dict: Any,
    agent_cfg: dict[str, Any],
    args: argparse.Namespace,
    attempt_number: int,
) -> tuple[dict[str, Any], _StageProgressRecorder]:
    instance = deepcopy(row)
    instance_id = str(instance["instance_id"])
    output_file_dir = str(Path.cwd() / "responses_api_agents" / "swe_agents_terminal" / "results" / args.subset / args.model)
    instance_dir = Path(output_file_dir) / instance_id / f"attempt_{attempt_number}"
    progress = _StageProgressRecorder(instance_id=instance_id, instance_dir=instance_dir)
    sandbox: Sandbox | None = None
    response_output: list[dict[str, Any]] = []
    model_patch = ""
    eval_report: dict[str, Any] = {}
    reward = 0.0
    current_stage = "start"
    try:
        resolved_sandbox_provider = resolve_provider_config(agent_cfg["sandbox_provider"], global_config_dict)
        provider_default_metadata = resolve_provider_metadata(agent_cfg["sandbox_provider"], global_config_dict)
        spec_config = dict(agent_cfg.get("sandbox_spec") or {})
        spec_config.setdefault("ttl_s", args.sandbox_ttl_s)
        spec_config.setdefault("ready_timeout_s", args.ready_timeout_s)
        spec_config.setdefault("metadata", {})
        spec_config["metadata"] = {
            **(provider_default_metadata or {}),
            **spec_config.get("metadata", {}),
            "benchmark": "swebench-pro",
            "harness": "opencode-direct-pro-sgf-style",
            "auto_cleanup": "true",
            "instance_id": instance_id[:63],
        }
        resources = SandboxResources.from_mapping(spec_config.pop("resources", {}))
        image = str(instance.get("sandbox_image") or "")
        if not image:
            raise ProRunError("sandbox_create", f"missing sandbox image for {instance_id}", category="infra_failure")
        current_stage = "sandbox_create"
        progress.stage_start(current_stage, f"image={image}")
        sandbox = Sandbox(resolved_sandbox_provider).start(
            SandboxSpec(
                image=image,
                ttl_s=spec_config.pop("ttl_s", None),
                ready_timeout_s=spec_config.pop("ready_timeout_s", None),
                workdir=args.workdir,
                env=dict(spec_config.pop("env", {})),
                files=spec_config.pop("files", {}),
                metadata=spec_config.pop("metadata", {}),
                resources=resources,
                entrypoint=spec_config.pop("entrypoint", None),
                provider_options=spec_config.pop("provider_options", {}),
            )
        )
        progress.stage_finish(current_stage)
        env = EnvAdapter(
            sandbox,
            cwd=args.workdir,
            user=args.user,
            step_timeout=args.step_timeout,
            eval_timeout=args.eval_timeout,
            idle_timeout_s=args.opencode_idle_timeout,
            instance_dir=instance_dir,
            progress=progress,
        )

        for stage, command, timeout in _preflight_steps():
            current_stage = stage
            progress.stage_start(stage, "running command")
            _run_checked(env, command, timeout=timeout, cwd=args.workdir)
            progress.stage_finish(stage)

        params = {
            "policy_model_name": args.model,
            "base_url": str(args.model_url).rstrip("/") + "/v1",
            "responses_create_params": row["responses_create_params"],
            "opencode_provider_id": agent_cfg.get("opencode_provider_id", "litellm"),
            "opencode_provider_api_url": agent_cfg.get("opencode_provider_api_url") or args.model_url,
            "opencode_api_key_env": agent_cfg.get("opencode_api_key_env", "LITELLM_KEY"),
            "opencode_api_key": args.api_key or agent_cfg.get("opencode_api_key"),
            "opencode_npm_package": args.opencode_npm_package or agent_cfg.get("opencode_npm_package", "opencode-ai@1.14.50"),
            "opencode_prompt_mode": args.opencode_prompt_mode,
        }
        for stage, command in _opencode_setup_steps(params):
            current_stage = stage
            progress.stage_start(stage, "running command")
            _run_checked(env, command, timeout=args.opencode_setup_timeout, cwd=args.workdir)
            progress.stage_finish(stage)

        current_stage = "prompt_write"
        progress.stage_start(current_stage, "writing SWE-bench Pro prompt")
        prompt = _pro_prompt(instance, repo_path=args.workdir, max_diff_lines=args.max_diff_lines)
        prompt_path = "/tmp/xgym_swebench_pro_prompt.txt"
        prompt_file = instance_dir / "prompt.txt"
        prompt_file.parent.mkdir(parents=True, exist_ok=True)
        prompt_file.write_text(prompt, encoding="utf-8")
        _run_checked(env, _here_doc(prompt_path, prompt), timeout=120, cwd=args.workdir)
        progress.stage_finish(current_stage)

        provider_id = str(agent_cfg.get("opencode_provider_id") or "litellm")
        raw_command = _opencode_run_command(
            params=params,
            provider_id=provider_id,
            model_name=args.model,
            prompt=prompt,
            prompt_path=prompt_path,
        )
        current_stage = "agent_run"
        progress.stage_start(current_stage, f"opencode run --model {provider_id}/{args.model}")
        opencode_result = env.execute(raw_command, timeout=args.opencode_command_timeout, cwd=args.workdir, background=True)
        opencode_output = str(opencode_result.get("output") or "")
        session_id = _extract_opencode_session_id(opencode_output)
        if int(opencode_result.get("returncode", 1)) != 0:
            progress.stage_update(current_stage, f"opencode exited with {opencode_result.get('returncode')}; attempting partial patch recovery")
            partial = env.execute("git add -A && git diff --no-color HEAD", timeout=300, cwd=args.workdir)
            model_patch = _sanitize_swebench_pro_patch(str(partial.get("output") or ""))
            if int(partial.get("returncode", 1)) != 0 or not model_patch.strip():
                raise RuntimeError(_redact_opencode_secrets(opencode_output or "opencode failed without patch", params))
        progress.stage_finish(current_stage, message=f"session_id={session_id or '-'}")

        if opencode_output.strip():
            response_output = [
                {
                    "id": f"msg_{uuid4()}",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": opencode_output[-4000:], "annotations": []}],
                }
            ]

        if not model_patch:
            current_stage = "patch_collect"
            progress.stage_start(current_stage, "git diff")
            raw_patch = _run_checked(env, "git add -A && git diff --no-color HEAD", timeout=300, cwd=args.workdir)
            model_patch = _sanitize_swebench_pro_patch(raw_patch)
            if not model_patch.strip():
                raise ProRunError(current_stage, "opencode completed without producing repository changes", category="model_failure")
            progress.stage_finish(current_stage, message=f"bytes={len(model_patch.encode('utf-8'))}")

        current_stage = "eval"
        progress.stage_start(current_stage, "running SWE-bench Pro eval")
        instance_report = _run_pro_eval(instance, env, instance_dir, model_patch, args)
        progress.stage_finish(current_stage)
        eval_report = {instance_id: instance_report}
        reward = 1.0 if bool(instance_report.get("resolved")) else 0.0

        response_obj = _default_response_object()
        response_obj["id"] = f"resp_{uuid4()}"
        response_obj["model"] = args.model
        response_obj["temperature"] = args.temperature
        response_obj["top_p"] = args.top_p
        response_obj["output"] = response_output
        result = {
            "responses_create_params": {**row["responses_create_params"], "input": [{"type": "message", "role": "user", "content": prompt}]},
            "reward": reward,
            "response": response_obj,
            "instance_id": instance_id,
            "metadata": {
                "failure_category": None,
                "failure_stage": None,
                "valid_eval": True,
                "eval_report": eval_report,
                "sandbox_image": instance.get("sandbox_image"),
                "agent_framework": "opencode",
                "model_patch": model_patch,
                "opencode_session_id": session_id,
                **progress.paths(),
            },
            "_ng_task_index": row["_ng_task_index"],
            "_ng_rollout_index": row["_ng_rollout_index"],
            "agent_ref": row["agent_ref"],
        }
        return result, progress
    except Exception as exc:
        category, stage = _classify_error(exc, current_stage)
        progress.fail(str(exc))
        raise ProRunError(stage, str(exc), category=category) from exc
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
            try:
                progress.stage_finish("cleanup")
                if had_failed:
                    progress._state["status"] = "failed"
                    progress._record_event("instance_end", status="failed", message=progress._state.get("error"))
                else:
                    progress.finish("completed")
            except Exception:
                pass


def _run_one(*, row: dict[str, Any], global_config_dict: Any, agent_cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    last_exc: BaseException | None = None
    last_category = "model_failure"
    last_stage = "unknown"
    last_progress: _StageProgressRecorder | None = None
    attempts = max(1, int(args.infra_retries) + 1)
    for attempt_number in range(1, attempts + 1):
        try:
            result, progress = _run_one_attempt(
                row=row,
                global_config_dict=global_config_dict,
                agent_cfg=agent_cfg,
                args=args,
                attempt_number=attempt_number,
            )
            result.setdefault("metadata", {})["attempt_count"] = attempt_number
            return result
        except Exception as exc:
            last_exc = exc
            last_category, last_stage = _classify_error(exc)
            if isinstance(exc, ProRunError):
                last_category, last_stage = exc.category, exc.stage
            _log(f"[direct-pro] {row.get('instance_id')} attempt {attempt_number}/{attempts} failed: {last_category}:{last_stage}: {exc}")
            if last_category == "infra_failure" and attempt_number < attempts:
                if args.infra_retry_delay_s:
                    time.sleep(args.infra_retry_delay_s)
                continue
            break
    assert last_exc is not None
    return _build_failure_result(
        row=row,
        args=args,
        exc=last_exc,
        category=last_category,
        stage=last_stage,
        progress=last_progress,
        attempt_count=attempt_number,
    )


def _compute_metrics(results: list[dict[str, Any]], total: int) -> dict[str, Any]:
    resolved_ids = sorted(str(r.get("instance_id")) for r in results if float(r.get("reward", 0.0) or 0.0) >= 1.0)
    categories: Counter[str] = Counter()
    stages: Counter[str] = Counter()
    nonempty_patch = 0
    valid_eval = 0
    for row in results:
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        category = metadata.get("failure_category")
        stage = metadata.get("failure_stage")
        if category:
            categories[str(category)] += 1
        else:
            valid_eval += 1
        if stage:
            stages[str(stage)] += 1
        if str(metadata.get("model_patch") or "").strip():
            nonempty_patch += 1
    completed = len(results)
    resolved = len(resolved_ids)
    valid_unresolved = max(valid_eval - resolved, 0)
    return {
        "total": total,
        "completed": completed,
        "pending": max(total - completed, 0),
        "valid_eval": valid_eval,
        "resolved": resolved,
        "valid_unresolved": valid_unresolved,
        "infra_failed": categories.get("infra_failure", 0),
        "model_failed": categories.get("model_failure", 0),
        "eval_failed": categories.get("eval_failure", 0),
        "nonempty_model_patch": nonempty_patch,
        "resolved_rate_valid_eval": resolved / valid_eval if valid_eval else 0.0,
        "resolved_rate_total": resolved / total if total else 0.0,
        "valid_eval_rate_total": valid_eval / total if total else 0.0,
        "resolved_ids": resolved_ids,
        "failure_categories": dict(categories),
        "failure_stages": dict(stages),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="ScaleAI/SWE-bench_Pro")
    parser.add_argument("--dataset-config", default="default")
    parser.add_argument("--split", default="test")
    parser.add_argument("--image-map", type=Path, default=DEFAULT_IMAGE_MAP)
    parser.add_argument("--official-assets-dir", type=Path, default=DEFAULT_OFFICIAL_ASSETS_DIR)
    parser.add_argument("--instance-id", action="append", default=None)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--agent-config", default="responses_api_agents/swe_agents_terminal/configs/swebench_terminal_opencode_latest.yaml")
    parser.add_argument("--model-config", default="responses_api_models/vllm_model/configs/vllm_model.yaml")
    parser.add_argument("--agent-name", default="swe_agents_terminal")
    parser.add_argument("--model", default="qwen3.5-35b-a3b")
    parser.add_argument("--model-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--subset", default="swebench_pro_qwen35_maas_opencode_sgf_style")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--terminal-concurrency", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run-materialize", action="store_true")
    parser.add_argument("--use-datasets-library", action="store_true")
    parser.add_argument("--workdir", default=PRO_REPO_PATH)
    parser.add_argument("--user", default="root")
    parser.add_argument("--sandbox-ttl-s", type=int, default=14400)
    parser.add_argument("--ready-timeout-s", type=int, default=1200)
    parser.add_argument("--step-timeout", type=int, default=7200)
    parser.add_argument("--eval-timeout", type=int, default=1800)
    parser.add_argument("--eval-background", action="store_true")
    parser.add_argument("--opencode-setup-timeout", type=int, default=300)
    parser.add_argument("--opencode-idle-timeout", type=int, default=1200)
    parser.add_argument("--opencode-command-timeout", type=int, default=7200)
    parser.add_argument("--opencode-prompt-mode", default="file", choices=["file", "inline"])
    parser.add_argument("--opencode-npm-package", default=None)
    parser.add_argument("--infra-retries", type=int, default=2)
    parser.add_argument("--infra-retry-delay-s", type=float, default=5.0)
    parser.add_argument("--max-diff-lines", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-output-tokens", type=int, default=32768)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--enable-thinking", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_map_payload = _load_image_map(args.image_map)
    if args.dataset == "ScaleAI/SWE-bench_Pro" and image_map_payload.get("dataset_name"):
        args.dataset = str(image_map_payload["dataset_name"])
    if args.split == "test" and image_map_payload.get("split"):
        args.split = str(image_map_payload["split"])
    if args.dataset_config == "default" and image_map_payload.get("dataset_config"):
        args.dataset_config = str(image_map_payload["dataset_config"])
    images = {str(k): str(v) for k, v in image_map_payload["images"].items()}
    requested_instances = list(args.instance_id or sorted(images.keys()))
    if args.limit is not None:
        requested_instances = requested_instances[: args.limit]
    rows = _materialize_rows(_load_rows_for_instances(args, requested_instances), images, args)
    total = len(rows)

    materialized = args.output.with_name(args.output.stem + "_materialized_inputs.jsonl")
    if not materialized.exists() or not args.resume:
        materialized.unlink(missing_ok=True)
        for row in rows:
            _append_jsonl(materialized, row)

    if args.dry_run_materialize:
        _write_json(
            args.output.with_name(args.output.stem + "_materialized_summary.json"),
            {
                "dataset": args.dataset,
                "dataset_config": args.dataset_config,
                "split": args.split,
                "image_map": str(args.image_map),
                "total": total,
                "first_instance_id": rows[0].get("instance_id") if rows else None,
                "first_sandbox_image": rows[0].get("sandbox_image") if rows else None,
                "materialized": str(materialized),
            },
        )
        _log(f"[direct-pro] dry-run materialized {total} rows -> {materialized}")
        return

    globals().update(_import_runtime_modules())
    global_config_dict, agent_cfg = _merge_config(args)
    completed_ids: set[str] = set()
    results: list[dict[str, Any]] = []
    if args.resume and args.output.exists():
        for row in _iter_jsonl(args.output):
            results.append(row)
            if row.get("instance_id") is not None:
                completed_ids.add(str(row["instance_id"]))
        rows = [row for row in rows if str(row.get("instance_id")) not in completed_ids]
        _log(f"[direct-pro] resume: {len(results)} existing, {len(rows)} remaining")
    elif args.output.exists():
        args.output.unlink()

    started = time.time()
    completed = len(results)
    lock = threading.Lock()
    _log(
        f"[direct-pro] running {len(rows)} remaining / {total} total with "
        f"concurrency={args.concurrency}, terminal_concurrency={args.terminal_concurrency}, ttl_s={args.sandbox_ttl_s}"
    )
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {executor.submit(_run_one, row=row, global_config_dict=global_config_dict, agent_cfg=agent_cfg, args=args): row for row in rows}
        for future in as_completed(futures):
            row = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                category, stage = _classify_error(exc)
                result = _build_failure_result(
                    row=row,
                    args=args,
                    exc=exc,
                    category=category,
                    stage=stage,
                    progress=None,
                    attempt_count=max(1, args.infra_retries + 1),
                )
            with lock:
                _append_jsonl(args.output, result)
                results.append(result)
                completed += 1
                if completed % 5 == 0 or completed == total:
                    metrics = _compute_metrics(results, total)
                    elapsed = max(time.time() - started, 1e-6)
                    _log(
                        f"[direct-pro] completed={completed}/{total} valid_eval={metrics['valid_eval']} "
                        f"resolved={metrics['resolved']} infra_failed={metrics['infra_failed']} "
                        f"elapsed={elapsed / 60:.1f}m"
                    )

    results.sort(key=lambda r: (int(r.get("_ng_task_index", 0)), int(r.get("_ng_rollout_index", 0))))
    args.output.unlink(missing_ok=True)
    for row in results:
        _append_jsonl(args.output, row)
    metrics = _compute_metrics(results, total)
    metrics_path = args.output.with_name(args.output.stem + "_aggregate_metrics.json")
    _write_json(metrics_path, metrics)
    _log(f"[direct-pro] done rollouts={args.output} metrics={metrics_path}")
    _log(json.dumps(metrics, ensure_ascii=False, indent=2)[:4000])


if __name__ == "__main__":
    main()
