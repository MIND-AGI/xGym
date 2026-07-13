#!/usr/bin/env python3
"""Write StarGaze-style live progress files for xGym SWE-bench rollouts."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ERROR_PATTERNS = {
    "sync_loop_timeout_3600": [("timed out waiting for the sync loop after 3600s",)],
    "terminal_background_timeout_3600": [("terminal sandbox background command timed out after 3600s",)],
    "idle_timeout": [
        ("long command output idle timed out",),
        ("idle timed out",),
        ("idle timeout",),
    ],
    "no_repository_changes": [("completed without producing repository changes",)],
    "patch_apply_failed": [("patch", "apply", "failed")],
    "sandbox_lost": [
        ("sandbox", "not found"),
        ("session", "not found"),
        ("process/connect",),
        ("process/start",),
    ],
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


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


def _iter_rollouts(path: Path) -> list[dict[str, Any]]:
    return _iter_jsonl(path)


def _input_instance_ids(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    return {_row_instance_id(row) for row in _iter_jsonl(path)}


def _row_instance_id(row: dict[str, Any]) -> str:
    return str(
        row.get("instance_id")
        or (row.get("metadata") or {}).get("instance_id")
        or (row.get("verifier_metadata") or {}).get("instance_id")
        or "unknown"
    )


def _row_eval_report(row: dict[str, Any]) -> dict[str, Any]:
    result = row.get("result") or {}
    report = result.get("eval_report") or row.get("eval_report") or {}
    return report if isinstance(report, dict) else {}


def _is_resolved(row: dict[str, Any]) -> bool:
    report = _row_eval_report(row)
    return bool(report.get("resolved") is True or row.get("reward") in {1, 1.0, True})


def _row_error_text(row: dict[str, Any]) -> str:
    report = _row_eval_report(row)
    parts = [
        (row.get("metadata") or {}).get("error"),
        (row.get("metadata") or {}).get("traceback"),
        row.get("error"),
        report.get("error"),
        report.get("failure_reason"),
        report.get("unresolved_reason"),
    ]
    return "\n".join(str(part) for part in parts if part)


def _classify_error(text: str) -> str | None:
    lowered = text.lower()
    for category, alternatives in ERROR_PATTERNS.items():
        if any(all(needle in lowered for needle in needles) for needles in alternatives):
            return category
    if "timeout" in lowered or "timed out" in lowered:
        return "other_timeout"
    if lowered.strip():
        return "other_error"
    return None


def _state_is_fresh(state_path: Path, *, state_max_age_s: float | None) -> bool:
    if state_max_age_s is None or state_max_age_s <= 0:
        return True
    try:
        return time.time() - state_path.stat().st_mtime <= state_max_age_s
    except OSError:
        return False


def _discover_instance_roots(
    instance_root: Path,
    *,
    active_ids: set[str] | None,
    state_max_age_s: float | None,
) -> list[Path]:
    """Find roots that are actually receiving per-instance progress.

    Runs can be relaunched with a new subset name while old in-flight requests are
    still writing under their original subset root. Prefer the configured root, but
    discover sibling roots for the same model and active instance ids so live
    monitors do not sit at 0/running when progress is being written elsewhere.
    """
    roots = [instance_root]
    if not active_ids:
        return roots

    try:
        results_root = instance_root.parent.parent
    except IndexError:
        return roots
    if not results_root.is_dir():
        return roots

    model_dir_name = instance_root.name
    discovered: list[tuple[float, Path]] = []
    for subset_dir in results_root.iterdir():
        candidate_root = subset_dir / model_dir_name
        if candidate_root == instance_root or not candidate_root.is_dir():
            continue
        latest_mtime = 0.0
        for instance_id in active_ids:
            state_path = candidate_root / instance_id / "progress_state.json"
            if not state_path.exists() or not _state_is_fresh(state_path, state_max_age_s=state_max_age_s):
                continue
            try:
                latest_mtime = max(latest_mtime, state_path.stat().st_mtime)
            except OSError:
                continue
        if latest_mtime > 0:
            discovered.append((latest_mtime, candidate_root))

    roots.extend(root for _, root in sorted(discovered, reverse=True))
    return roots


def _collect_running_states(
    instance_roots: list[Path],
    completed_ids: set[str],
    active_ids: set[str] | None = None,
    state_max_age_s: float | None = None,
) -> list[dict[str, Any]]:
    states_by_instance: dict[str, tuple[float, Path, dict[str, Any]]] = {}
    for instance_root in instance_roots:
        if not instance_root.exists():
            continue
        for state_path in instance_root.glob("*/progress_state.json"):
            if not _state_is_fresh(state_path, state_max_age_s=state_max_age_s):
                continue
            instance_id = state_path.parent.name
            if active_ids is not None and instance_id not in active_ids:
                continue
            if instance_id in completed_ids:
                continue
            state = _load_json(state_path)
            if not state:
                continue
            status = str(state.get("status") or "")
            if status in {"completed", "failed", "error"}:
                continue
            try:
                mtime = state_path.stat().st_mtime
            except OSError:
                continue
            previous = states_by_instance.get(instance_id)
            if previous is not None and previous[0] >= mtime:
                continue
            states_by_instance[instance_id] = (mtime, state_path, state)

    running: list[dict[str, Any]] = []
    for instance_id, (_, state_path, state) in states_by_instance.items():
        status = str(state.get("status") or "")
        running.append(
            {
                "instance_id": instance_id,
                "status": status or state.get("current_stage") or "running",
                "current_stage": state.get("current_stage"),
                "updated_at": state.get("updated_at"),
                "progress_state_path": str(state_path),
                "instance_root": str(state_path.parent.parent),
                "opencode_session_log_path": state.get("opencode_session_log_path"),
                "opencode_session_log_lines": state.get("opencode_session_log_lines"),
                "opencode_session_log_bytes": state.get("opencode_session_log_bytes"),
                "opencode_session_log_tail": state.get("opencode_session_log_tail"),
            }
        )
    running.sort(key=lambda item: str(item.get("updated_at") or ""), reverse=True)
    return running


def build_snapshot(
    *,
    rollout_path: Path,
    instance_root: Path,
    total: int,
    input_path: Path | None = None,
    state_max_age_s: float | None = 6 * 60 * 60,
    discover_roots: bool = True,
) -> dict[str, Any]:
    rows = _iter_rollouts(rollout_path)
    active_ids = _input_instance_ids(input_path)
    instance_roots = (
        _discover_instance_roots(instance_root, active_ids=active_ids, state_max_age_s=state_max_age_s)
        if discover_roots
        else [instance_root]
    )
    latest_by_instance: dict[str, dict[str, Any]] = {}
    for row in rows:
        latest_by_instance[_row_instance_id(row)] = row

    completed_ids = set(latest_by_instance)
    resolved_ids = sorted(iid for iid, row in latest_by_instance.items() if _is_resolved(row))
    unresolved_ids = sorted(completed_ids - set(resolved_ids))

    error_counts: Counter[str] = Counter()
    error_examples: dict[str, list[dict[str, str]]] = defaultdict(list)
    for iid, row in latest_by_instance.items():
        category = _classify_error(_row_error_text(row))
        if category:
            error_counts[category] += 1
            if len(error_examples[category]) < 8:
                error_examples[category].append({"instance_id": iid})

    completed = len(completed_ids)
    resolved = len(resolved_ids)
    running_instances = _collect_running_states(
        instance_roots,
        completed_ids,
        active_ids,
        state_max_age_s=state_max_age_s,
    )
    running_count = len(running_instances)
    pending = max(total - completed - running_count, 0)
    elapsed_s = None
    eta_s = None
    if completed:
        mtimes = [rollout_path.stat().st_mtime] if rollout_path.exists() else []
        if instance_root.exists():
            mtimes.extend(p.stat().st_mtime for p in instance_root.glob("*/progress_state.json") if p.exists())
        # Conservative ETA using rollout file age if available; monitor loop adds precise started_at separately.
        if mtimes and rollout_path.exists():
            elapsed_s = max(0.0, time.time() - rollout_path.stat().st_ctime)
            if elapsed_s > 0:
                eta_s = elapsed_s / completed * max(total - completed, 0)

    rates = {
        "completed_resolve_rate": resolved / completed if completed else 0.0,
        "full_resolve_lower_bound": resolved / total if total else 0.0,
        "completion_rate": completed / total if total else 0.0,
    }
    return {
        "updated_at": _now_iso(),
        "rollout_path": str(rollout_path),
        "instance_root": str(instance_root),
        "discovered_instance_roots": [str(root) for root in instance_roots],
        "total_instances": total,
        "completed_instances": completed,
        "running_instances_count": running_count,
        "pending_instances": pending,
        "resolved_instances": resolved,
        "unresolved_instances": len(unresolved_ids),
        "rates": rates,
        "error_counts": dict(error_counts),
        "error_examples": dict(error_examples),
        "instances_by_exit_status": {
            "resolved": resolved_ids,
            "unresolved": unresolved_ids,
        },
        "running_instances": running_instances[:128],
        "timing": {
            "elapsed_s_estimate": elapsed_s,
            "eta_s_estimate": eta_s,
        },
    }


def summary_text(snapshot: dict[str, Any]) -> str:
    rates = snapshot["rates"]
    lines = [
        f"updated_at: {snapshot['updated_at']}",
        f"completed: {snapshot['completed_instances']} / {snapshot['total_instances']}",
        f"running: {snapshot['running_instances_count']}",
        f"pending: {snapshot['pending_instances']}",
        f"resolved: {snapshot['resolved_instances']}",
        f"unresolved completed: {snapshot['unresolved_instances']}",
        f"completed_resolve_rate: {rates['completed_resolve_rate']:.4%}",
        f"full_resolve_lower_bound: {rates['full_resolve_lower_bound']:.4%}",
        f"completion_rate: {rates['completion_rate']:.4%}",
        "",
        "error_counts:",
    ]
    if snapshot["error_counts"]:
        lines.extend(f"  {key}: {value}" for key, value in sorted(snapshot["error_counts"].items()))
    else:
        lines.append("  none")
    lines.append("")
    lines.append("recent/running instances:")
    for item in snapshot["running_instances"][:20]:
        stage = item.get("current_stage") or item.get("status") or "running"
        lines.append(f"  {item['instance_id']}: {stage} updated_at={item.get('updated_at')}")
    return "\n".join(lines) + "\n"


def write_snapshot(snapshot: dict[str, Any], out_dir: Path, previous: dict[str, Any] | None = None) -> None:
    _atomic_write_json(out_dir / "progress_state.json", snapshot)
    _atomic_write_json(
        out_dir / "live_metrics.json",
        {
            "updated_at": snapshot["updated_at"],
            "total_instances": snapshot["total_instances"],
            "completed_instances": snapshot["completed_instances"],
            "running_instances_count": snapshot["running_instances_count"],
            "pending_instances": snapshot["pending_instances"],
            "resolved_instances": snapshot["resolved_instances"],
            "unresolved_instances": snapshot["unresolved_instances"],
            "rates": snapshot["rates"],
            "error_counts": snapshot["error_counts"],
        },
    )
    _atomic_write_text(out_dir / "live_summary.txt", summary_text(snapshot))

    if previous is None or previous.get("completed_instances") != snapshot.get("completed_instances"):
        event = {
            "updated_at": snapshot["updated_at"],
            "completed_instances": snapshot["completed_instances"],
            "resolved_instances": snapshot["resolved_instances"],
            "completed_resolve_rate": snapshot["rates"]["completed_resolve_rate"],
            "full_resolve_lower_bound": snapshot["rates"]["full_resolve_lower_bound"],
            "error_counts": snapshot["error_counts"],
        }
        with (out_dir / "progress_events.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--input", type=Path, default=None, help="Optional input JSONL to scope running instances.")
    parser.add_argument("--instance-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--total", type=int, default=500)
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument(
        "--state-max-age-s",
        type=float,
        default=6 * 60 * 60,
        help="Ignore per-instance progress files older than this many seconds; <=0 disables the age filter.",
    )
    parser.add_argument(
        "--no-discover-roots",
        action="store_true",
        help="Only scan --instance-root; by default sibling roots for the same model are discovered from active IDs.",
    )
    parser.add_argument("--watch", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    previous: dict[str, Any] | None = None
    while True:
        snapshot = build_snapshot(
            rollout_path=args.rollout,
            instance_root=args.instance_root,
            total=args.total,
            input_path=args.input,
            state_max_age_s=args.state_max_age_s,
            discover_roots=not args.no_discover_roots,
        )
        write_snapshot(snapshot, args.out_dir, previous=previous)
        print(summary_text(snapshot), flush=True)
        previous = snapshot
        if not args.watch or snapshot["completed_instances"] >= args.total:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
