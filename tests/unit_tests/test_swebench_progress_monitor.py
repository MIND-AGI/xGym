from __future__ import annotations

import json
from pathlib import Path

from responses_api_agents.swe_agents_terminal.monitor_swebench_progress import _classify_error, build_snapshot


def test_classify_idle_timeout_variants() -> None:
    assert _classify_error("terminal sandbox long command output idle timed out after 1200s") == "idle_timeout"
    assert _classify_error("terminal sandbox long command idle timed out after 1200s") == "idle_timeout"
    assert _classify_error("idle timeout waiting for opencode output") == "idle_timeout"


def test_classify_timed_out_as_timeout() -> None:
    assert _classify_error("command timed out after 14400s") == "other_timeout"


def test_classify_sandbox_lost_variants() -> None:
    assert _classify_error("sandbox abc not found") == "sandbox_lost"
    assert _classify_error("session abc not found") == "sandbox_lost"
    assert _classify_error("terminal sandbox long command reconnect failed: /api/process/connect") == "sandbox_lost"


def test_build_snapshot_discovers_sibling_instance_root(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    input_path.write_text(json.dumps({"instance_id": "django__django-1"}) + "\n", encoding="utf-8")
    rollout_path = tmp_path / "rollouts.jsonl"
    rollout_path.write_text("", encoding="utf-8")

    configured_root = tmp_path / "results" / "new_subset" / "model"
    actual_root = tmp_path / "results" / "old_subset" / "model"
    state_dir = actual_root / "django__django-1"
    state_dir.mkdir(parents=True)
    (state_dir / "progress_state.json").write_text(
        json.dumps(
            {
                "status": "running",
                "current_stage": "agent_run",
                "updated_at": "2026-07-09T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )

    snapshot = build_snapshot(
        rollout_path=rollout_path,
        instance_root=configured_root,
        total=1,
        input_path=input_path,
    )

    assert snapshot["running_instances_count"] == 1
    assert snapshot["running_instances"][0]["instance_id"] == "django__django-1"
    assert snapshot["running_instances"][0]["current_stage"] == "agent_run"
    assert str(actual_root) in snapshot["discovered_instance_roots"]
