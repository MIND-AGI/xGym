#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


DATASET_ROWS_ENDPOINT = "https://datasets-server.huggingface.co/rows"
DEFAULT_DATASET_NAME = "princeton-nlp/SWE-bench_Verified"
DEFAULT_DATASET_CONFIG = "default"
DEFAULT_SPLIT = "test"
DEFAULT_IMAGE_MAP = Path("evaluation/swebench/configs/faas_instance_images.json")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            rows.append(payload)
    return rows


def _load_instance_ids_from_image_map(path: Path) -> list[str]:
    payload = _load_json(path)
    images = payload.get("images")
    if not isinstance(images, dict):
        raise ValueError(f"image map must contain an 'images' object: {path}")
    instance_ids = sorted(str(instance_id) for instance_id in images)
    if not instance_ids:
        raise ValueError(f"image map has no instances: {path}")
    return instance_ids


def _dataset_request(
    *,
    dataset_name: str,
    dataset_config: str,
    split: str,
    offset: int,
    length: int,
) -> list[dict[str, Any]]:
    query = urllib.parse.urlencode(
        {
            "dataset": dataset_name,
            "config": dataset_config,
            "split": split,
            "offset": offset,
            "length": length,
        }
    )
    with urllib.request.urlopen(f"{DATASET_ROWS_ENDPOINT}?{query}", timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise RuntimeError("datasets-server returned an unexpected payload")
    normalized_rows: list[dict[str, Any]] = []
    for item in rows:
        if isinstance(item, dict) and isinstance(item.get("row"), dict):
            normalized_rows.append(item["row"])
    return normalized_rows


def _load_instances(
    *,
    dataset_name: str,
    dataset_config: str,
    split: str,
    instance_ids: list[str],
) -> list[dict[str, Any]]:
    pending = set(instance_ids)
    found: dict[str, dict[str, Any]] = {}
    offset = 0
    length = 100
    while pending:
        rows = _dataset_request(
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            split=split,
            offset=offset,
            length=length,
        )
        if not rows:
            break
        for row in rows:
            instance_id = row.get("instance_id")
            if isinstance(instance_id, str) and instance_id in pending:
                normalized = dict(row)
                normalized["responses_create_params"] = dict(normalized.get("responses_create_params") or {})
                normalized["responses_create_params"]["input"] = []
                normalized["subset"] = "verified"
                normalized["split"] = split
                found[instance_id] = normalized
                pending.discard(instance_id)
        offset += len(rows)
    if pending:
        raise RuntimeError(f"failed to load dataset rows for: {', '.join(sorted(pending))}")
    return [found[instance_id] for instance_id in instance_ids]


def _metadata(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def _instance_id(row: dict[str, Any]) -> str | None:
    value = row.get("instance_id") or _metadata(row).get("instance_id")
    return str(value) if value is not None else None


def _is_resolved_rollout(row: dict[str, Any]) -> bool:
    try:
        if float(row.get("reward", 0.0) or 0.0) >= 1.0:
            return True
    except (TypeError, ValueError):
        pass
    metadata = _metadata(row)
    instance_id = _instance_id(row)
    report_map = metadata.get("eval_report")
    if isinstance(report_map, dict):
        if instance_id and isinstance(report_map.get(instance_id), dict):
            return bool(report_map[instance_id].get("resolved"))
        for report in report_map.values():
            if isinstance(report, dict) and bool(report.get("resolved")):
                return True
    return False


def _best_rollouts_by_instance(paths: list[Path]) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in _read_jsonl(path):
            instance_id = _instance_id(row)
            if not instance_id:
                continue
            current = best.get(instance_id)
            if current is None or (not _is_resolved_rollout(current) and _is_resolved_rollout(row)):
                best[instance_id] = row
    return best


def _make_input(args: argparse.Namespace) -> None:
    image_map = Path(args.image_map).expanduser()
    instance_ids = [str(item) for item in args.instance_id] or _load_instance_ids_from_image_map(image_map)
    rows = _load_instances(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.split,
        instance_ids=instance_ids,
    )
    _write_jsonl(Path(args.output), rows)
    _write_json(
        Path(args.output).with_suffix(Path(args.output).suffix + ".metadata.json"),
        {
            "dataset_name": args.dataset_name,
            "dataset_config": args.dataset_config,
            "split": args.split,
            "image_map": str(image_map),
            "count": len(rows),
        },
    )


def _make_retry(args: argparse.Namespace) -> None:
    source_rows = _read_jsonl(Path(args.input))
    selected_rows = []
    resolved = _best_rollouts_by_instance([Path(path) for path in args.rollout])
    for row in source_rows:
        instance_id = _instance_id(row)
        if not instance_id:
            continue
        if _is_resolved_rollout(resolved.get(instance_id, {})):
            continue
        selected_rows.append(row)
    _write_jsonl(Path(args.output), selected_rows)
    _write_json(
        Path(args.output).with_suffix(Path(args.output).suffix + ".metadata.json"),
        {
            "source_input": args.input,
            "rollouts": args.rollout,
            "source_count": len(source_rows),
            "retry_count": len(selected_rows),
        },
    )


def _combine(args: argparse.Namespace) -> None:
    best = _best_rollouts_by_instance([Path(path) for path in args.rollout])
    instance_ids = sorted(best)
    resolved_ids = sorted(instance_id for instance_id, row in best.items() if _is_resolved_rollout(row))
    unresolved_ids = sorted(set(instance_ids) - set(resolved_ids))
    total = len(instance_ids)
    summary = {
        "total": total,
        "resolved": len(resolved_ids),
        "unresolved": len(unresolved_ids),
        "resolved_rate": round(len(resolved_ids) / total, 4) if total else 0.0,
        "rollouts": args.rollout,
        "resolved_ids": resolved_ids,
        "unresolved_ids": unresolved_ids,
    }
    output = Path(args.output)
    _write_json(output, summary)
    report = output.with_suffix(".md")
    report.write_text(
        "\n".join(
            [
                "# SWE-bench Verified Union Report",
                "",
                f"- total: `{total}`",
                f"- resolved: `{len(resolved_ids)}`",
                f"- unresolved: `{len(unresolved_ids)}`",
                f"- resolved rate: `{summary['resolved_rate'] * 100:.2f}%`",
                "",
                "## Inputs",
                *[f"- `{path}`" for path in args.rollout],
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare and combine xGym SWE-bench Verified reproduction artifacts.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    make_input = subparsers.add_parser("make-input", help="Create xGym JSONL from SWE-bench Verified rows.")
    make_input.add_argument("--image-map", default=str(DEFAULT_IMAGE_MAP))
    make_input.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    make_input.add_argument("--dataset-config", default=DEFAULT_DATASET_CONFIG)
    make_input.add_argument("--split", default=DEFAULT_SPLIT)
    make_input.add_argument("--instance-id", action="append", default=[])
    make_input.add_argument("--output", required=True)
    make_input.set_defaults(func=_make_input)

    make_retry = subparsers.add_parser("make-retry", help="Create retry JSONL for unresolved/failed rows.")
    make_retry.add_argument("--input", required=True)
    make_retry.add_argument("--rollout", action="append", required=True)
    make_retry.add_argument("--output", required=True)
    make_retry.set_defaults(func=_make_retry)

    combine = subparsers.add_parser("combine", help="Compute union resolved metrics from rollout JSONL files.")
    combine.add_argument("--rollout", action="append", required=True)
    combine.add_argument("--output", required=True)
    combine.set_defaults(func=_combine)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
