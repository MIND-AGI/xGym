#!/usr/bin/env python3
"""Direct SWE-bench terminal/opencode rollout runner.

This bypasses Ray/Gym server startup and calls the terminal sandbox opencode
runner in-process with a ThreadPoolExecutor. It writes NeMo Gym-compatible
rollout JSONL rows plus aggregate metrics.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import Any
from uuid import uuid4

from omegaconf import OmegaConf

from nemo_gym.global_config import GlobalConfigDictParser
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox import resolve_provider_config, resolve_provider_metadata
from responses_api_agents.mini_swe_agent_2.app import (
    MiniSWEAgentVerifyResponse,
    _default_response_object,
    _is_resolved,
    _opencode_responses_create_params,
    _responses_create_params_to_model_kwargs,
    _sandbox_runtime_env,
    _sandbox_spec_for_instance,
    _swebench_config_path,
    run_mini_swe_with_sandbox,
)

_PRINT_LOCK = threading.Lock()


def _log(message: str) -> None:
    with _PRINT_LOCK:
        print(message, flush=True)


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line_number, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            rows.append(payload)
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False).encode("utf-8") + b"\n")
        fh.flush()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False).encode("utf-8") + b"\n")


def _merge_config(args: argparse.Namespace):
    cfg = OmegaConf.merge(
        OmegaConf.load(args.agent_config),
        OmegaConf.load(args.model_config),
    )
    cfg.policy_model_name = args.model
    if args.model_url is not None:
        cfg.policy_base_url = args.model_url
    if args.api_key is not None:
        cfg.policy_api_key = args.api_key
    parsed = GlobalConfigDictParser().parse_no_environment(cfg)
    agent_cfg = OmegaConf.to_container(
        parsed[args.agent_name]["responses_api_agents"][args.agent_name],
        resolve=True,
    )
    agent_cfg["concurrency"] = args.concurrency
    agent_cfg["sandbox_provider"]["terminal"]["operations"]["concurrency"] = args.terminal_concurrency
    return parsed, agent_cfg


def _prepare_row(raw_row: dict[str, Any], task_index: int, args: argparse.Namespace) -> dict[str, Any]:
    row = deepcopy(raw_row)
    rcp = dict(row.get("responses_create_params") or {})
    rcp.setdefault("input", [])
    rcp.update(
        {
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
    )
    row["responses_create_params"] = rcp
    row["agent_ref"] = {"name": args.agent_name}
    row["_ng_task_index"] = task_index
    row["_ng_rollout_index"] = 0
    return row


def _run_one(
    *,
    row: dict[str, Any],
    global_config_dict: Any,
    agent_cfg: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    instance_id = str(row["instance_id"])
    subset = str(row.get("subset") or "verified")
    split = str(row.get("split") or "test")
    output_file_dir = str(Path.cwd() / "responses_api_agents" / "swe_agents_terminal" / "results" / subset / args.model)

    responses_params = NeMoGymResponseCreateParamsNonStreaming.model_validate(row["responses_create_params"])
    responses_create_params_dict = responses_params.model_dump(exclude_none=True)

    import yaml

    mini_swe_config_path = _swebench_config_path()
    config = yaml.safe_load(mini_swe_config_path.read_text())
    model_kwargs = _responses_create_params_to_model_kwargs(responses_create_params_dict)
    if model_kwargs:
        config.setdefault("model", {}).setdefault("model_kwargs", {}).update(model_kwargs)

    resolved_sandbox_provider = resolve_provider_config(agent_cfg["sandbox_provider"], global_config_dict)
    # Make the API key visible to provider SDKs if needed in future direct mode.
    runtime_env = _sandbox_runtime_env(resolved_sandbox_provider)
    for key, value in (runtime_env.get("env_vars") or {}).items():
        os.environ[str(key)] = str(value)

    provider_default_metadata = resolve_provider_metadata(agent_cfg["sandbox_provider"], global_config_dict)
    config.setdefault("environment", {}).update(dict(agent_cfg.get("sandbox_environment_kwargs") or {}))
    instance_spec = _sandbox_spec_for_instance(
        dict(agent_cfg.get("sandbox_spec") or {}),
        resource_profiles=agent_cfg.get("sandbox_resource_profiles"),
        instance_id=instance_id,
    )
    if provider_default_metadata:
        instance_spec["metadata"] = {**provider_default_metadata, **(instance_spec.get("metadata") or {})}
    config["environment"]["spec"] = instance_spec

    config_output_dir = Path(output_file_dir) / "_configs"
    config_output_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_output_dir / f"{instance_id}.sandbox.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    base_url = str(args.model_url).rstrip("/") + "/v1"
    params = dict(
        subset=subset,
        split=split,
        workers=1,
        output=output_file_dir,
        model=f"hosted_vllm/{args.model}",
        policy_model_name=args.model,
        api_key="dummy_key",
        base_url=base_url,
        env="sandbox",
        run_golden=bool(agent_cfg.get("run_golden", False)),
        instance_id=instance_id,
        config=config_path,
        instance_dict=row,
        responses_create_params=json.dumps(responses_create_params_dict),
        step_timeout=int(agent_cfg.get("step_timeout", 7200)),
        eval_timeout=int(agent_cfg.get("eval_timeout", 1800)),
        step_limit=int(agent_cfg.get("step_limit", 250)),
        agent_framework=str(agent_cfg.get("agent_framework", "opencode")),
        image_map_path=agent_cfg.get("image_map_path"),
        require_image_map=bool(agent_cfg.get("require_image_map", False)),
        sandbox_provider=resolved_sandbox_provider,
        sandbox_spec=instance_spec,
        sandbox_environment_kwargs=dict(agent_cfg.get("sandbox_environment_kwargs") or {}),
        opencode_provider_id=str(agent_cfg.get("opencode_provider_id", "litellm")),
        opencode_provider_api_url=agent_cfg.get("opencode_provider_api_url"),
        opencode_api_key_env=str(agent_cfg.get("opencode_api_key_env", "LITELLM_KEY")),
        opencode_api_key=str(agent_cfg.get("opencode_api_key", args.api_key)),
        opencode_npm_package=str(agent_cfg.get("opencode_npm_package", "opencode-ai@latest")),
        opencode_setup_timeout=int(agent_cfg.get("opencode_setup_timeout", 300)),
        opencode_idle_timeout=int(agent_cfg.get("opencode_idle_timeout", 1200)),
        opencode_command_timeout=agent_cfg.get("opencode_command_timeout"),
        opencode_prompt_mode=str(agent_cfg.get("opencode_prompt_mode", "inline")),
        opencode_infra_retries=int(agent_cfg.get("opencode_infra_retries", 1)),
        opencode_infra_retry_delay_s=float(agent_cfg.get("opencode_infra_retry_delay_s", 5.0)),
        max_diff_lines=400,
    )

    try:
        result_map = run_mini_swe_with_sandbox(**params)
        result = result_map[instance_id]
        input_messages = result["input_messages"]
        response_output = result["response_output"]
        responses = result["responses"]
        reward = 1.0 if _is_resolved(instance_id, result["eval_report"]) else 0.0
    except Exception as exc:
        _log(f"[direct-eval] {instance_id} failed: {exc}")
        result = {"eval_report": {"error": str(exc), "traceback": traceback.format_exc()}}
        input_messages = []
        response_output = []
        responses = []
        reward = 0.0

    response_obj = _default_response_object()
    if responses:
        response_obj.update(dict(responses[-1]))
    response_obj.pop("extra", None)
    response_obj["model"] = args.model
    response_obj["temperature"] = args.temperature
    response_obj["top_p"] = args.top_p
    response_obj["output"] = response_output

    row_rcp = dict(row["responses_create_params"])
    row_rcp["input"] = input_messages
    verify_response = MiniSWEAgentVerifyResponse(
        responses_create_params=row_rcp,
        reward=reward,
        response=response_obj,
        instance_id=instance_id,
        metadata=result.get("eval_report", {}) if result else {},
    ).model_dump(mode="json")
    verify_response["_ng_task_index"] = row["_ng_task_index"]
    verify_response["_ng_rollout_index"] = row["_ng_rollout_index"]
    verify_response["agent_ref"] = row["agent_ref"]
    return verify_response


def _compute_metrics(results: list[dict[str, Any]], total: int) -> dict[str, Any]:
    resolved_ids: list[str] = []
    error_counts: Counter[str] = Counter()
    for row in results:
        iid = str(row.get("instance_id"))
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        if float(row.get("reward", 0.0) or 0.0) >= 1.0:
            resolved_ids.append(iid)
        if metadata.get("error"):
            text = str(metadata.get("error") or "").lower()
            if "timeout" in text or "timed out" in text:
                error_counts["timeout"] += 1
            elif "sandbox" in text and "not found" in text:
                error_counts["sandbox_lost"] += 1
            elif "completed without producing repository changes" in text:
                error_counts["no_repository_changes"] += 1
            else:
                error_counts["other_error"] += 1
    completed = len(results)
    resolved = len(resolved_ids)
    return {
        "total": total,
        "completed": completed,
        "resolved": resolved,
        "unresolved_completed": completed - resolved,
        "pending": max(total - completed, 0),
        "resolved_rate_completed": resolved / completed if completed else 0.0,
        "resolved_rate_total": resolved / total if total else 0.0,
        "resolved_ids": sorted(resolved_ids),
        "error_counts": dict(error_counts),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--agent-config", default="responses_api_agents/swe_agents_terminal/configs/swebench_terminal_opencode_latest.yaml")
    parser.add_argument("--model-config", default="responses_api_models/vllm_model/configs/vllm_model.yaml")
    parser.add_argument("--agent-name", default="swe_agents_terminal")
    parser.add_argument("--model", default="qwen3.5-35b-a3b")
    parser.add_argument("--model-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--concurrency", type=int, default=128)
    parser.add_argument("--terminal-concurrency", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
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
    global_config_dict, agent_cfg = _merge_config(args)
    raw_rows = _iter_jsonl(args.input)
    if args.limit is not None:
        raw_rows = raw_rows[: args.limit]
    rows = [_prepare_row(row, i, args) for i, row in enumerate(raw_rows)]
    total = len(rows)

    completed_ids: set[str] = set()
    results: list[dict[str, Any]] = []
    if args.resume and args.output.exists():
        for row in _iter_jsonl(args.output):
            results.append(row)
            if row.get("instance_id") is not None:
                completed_ids.add(str(row["instance_id"]))
        rows = [row for row in rows if str(row.get("instance_id")) not in completed_ids]
        _log(f"[direct-eval] resume: {len(results)} existing, {len(rows)} remaining")
    elif args.output.exists():
        args.output.unlink()

    materialized = args.output.with_name(args.output.stem + "_materialized_inputs.jsonl")
    if not materialized.exists() or not args.resume:
        materialized.unlink(missing_ok=True)
        _write_jsonl(materialized, [_prepare_row(row, i, args) for i, row in enumerate(raw_rows)])

    started = time.time()
    completed = len(results)
    lock = threading.Lock()

    _log(f"[direct-eval] running {len(rows)} remaining / {total} total with concurrency={args.concurrency}")
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(_run_one, row=row, global_config_dict=global_config_dict, agent_cfg=agent_cfg, args=args): row
            for row in rows
        }
        for future in as_completed(futures):
            row = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                iid = str(row.get("instance_id") or "unknown")
                _log(f"[direct-eval] outer failure {iid}: {exc}")
                result = {
                    "responses_create_params": row["responses_create_params"],
                    "reward": 0.0,
                    "response": _default_response_object(),
                    "instance_id": iid,
                    "metadata": {"error": str(exc), "traceback": traceback.format_exc()},
                    "_ng_task_index": row["_ng_task_index"],
                    "_ng_rollout_index": row["_ng_rollout_index"],
                    "agent_ref": row["agent_ref"],
                }
            with lock:
                _append_jsonl(args.output, result)
                results.append(result)
                completed += 1
                elapsed = max(time.time() - started, 1e-6)
                rate = completed / total if total else 1.0
                if completed % 10 == 0 or completed == total:
                    resolved = sum(1 for r in results if float(r.get("reward", 0.0) or 0.0) >= 1.0)
                    _log(
                        f"[direct-eval] completed={completed}/{total} resolved={resolved} "
                        f"rate={rate:.1%} elapsed={elapsed/60:.1f}m"
                    )

    results.sort(key=lambda r: (int(r.get("_ng_task_index", 0)), int(r.get("_ng_rollout_index", 0))))
    args.output.unlink(missing_ok=True)
    for row in results:
        _append_jsonl(args.output, row)
    metrics = _compute_metrics(results, total)
    metrics_path = args.output.with_name(args.output.stem + "_aggregate_metrics.json")
    _write_json(metrics_path, metrics)
    _log(f"[direct-eval] done rollouts={args.output} metrics={metrics_path}")
    _log(json.dumps(metrics, ensure_ascii=False, indent=2)[:4000])


if __name__ == "__main__":
    # Make imports work when executed from this server directory's venv.
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    main()
