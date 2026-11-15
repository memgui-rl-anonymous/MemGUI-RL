#!/usr/bin/env python3
"""Build a trajectory-level subset from MobileSE/MemGUI verl JSON files.

The training JSON can be large, so this script prefers streaming with ijson when
available. It falls back to json.load if ijson is not installed.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable


def split_paths(paths: str) -> list[Path]:
    out = [Path(part.strip()) for part in paths.split(",") if part.strip()]
    if not out:
        raise ValueError("--input is empty")
    return out


def iter_json_array(path: Path) -> Iterable[dict[str, Any]]:
    try:
        import ijson  # type: ignore

        with path.open("rb") as f:
            try:
                yield from ijson.items(f, "item", use_float=True)
            except TypeError:
                yield from ijson.items(f, "item")
    except ImportError:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON list: {path}")
        yield from data


def get_nested_value(obj: dict[str, Any], dotted_key: str) -> Any:
    cur: Any = obj
    for part in dotted_key.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def trajectory_id(sample: dict[str, Any], group_key: str, index: int) -> str:
    if group_key != "auto":
        value = get_nested_value(sample, group_key)
        if value not in (None, ""):
            return str(value)

    metadata = sample.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}

    for key in ("trajectory_id", "traj_id"):
        value = sample.get(key) or metadata.get(key)
        if value not in (None, ""):
            return str(value)

    task_id = sample.get("task_id") or metadata.get("task_id")
    attempt_id = sample.get("attempt_id") or metadata.get("attempt_id")
    session = metadata.get("session")
    agent = metadata.get("agent")
    if task_id and (attempt_id or session or agent):
        return "/".join(str(x) for x in (session, task_id, agent, attempt_id) if x not in (None, ""))
    if task_id:
        return str(task_id)

    return f"sample-{index}"


def collect_trajectory_ids(paths: list[Path], group_key: str) -> list[str]:
    seen = set()
    ordered = []
    index = 0
    for path in paths:
        for sample in iter_json_array(path):
            group_id = trajectory_id(sample, group_key=group_key, index=index)
            index += 1
            if group_id in seen:
                continue
            seen.add(group_id)
            ordered.append(group_id)
    return ordered


def write_subset(
    paths: list[Path],
    output: Path,
    selected_ids: set[str],
    group_key: str,
) -> tuple[int, int]:
    output.parent.mkdir(parents=True, exist_ok=True)
    sample_count = 0
    selected_trajectory_ids = set()
    first = True
    index = 0

    with output.open("w", encoding="utf-8") as f:
        f.write("[\n")
        for path in paths:
            for sample in iter_json_array(path):
                group_id = trajectory_id(sample, group_key=group_key, index=index)
                index += 1
                if group_id not in selected_ids:
                    continue
                if not first:
                    f.write(",\n")
                json.dump(sample, f, ensure_ascii=False, default=json_default)
                first = False
                sample_count += 1
                selected_trajectory_ids.add(group_id)
        f.write("\n]\n")

    return sample_count, len(selected_trajectory_ids)


def json_default(value: Any) -> Any:
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else float(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="JSON path or comma-separated JSON paths")
    parser.add_argument("--output", required=True, help="Output subset JSON path")
    parser.add_argument("--max-trajectories", type=int, required=True, help="Number of trajectories to keep")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--group-key",
        default="auto",
        help="Trajectory id key. Use 'auto' or a dotted key such as metadata.trajectory_id.",
    )
    parser.add_argument("--no-shuffle", action="store_true", help="Keep the first N trajectories in file order")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_trajectories <= 0:
        raise ValueError("--max-trajectories must be positive")

    input_paths = split_paths(args.input)
    for path in input_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    trajectory_ids = collect_trajectory_ids(input_paths, group_key=args.group_key)
    if args.no_shuffle:
        ordered_ids = trajectory_ids
    else:
        ordered_ids = list(trajectory_ids)
        random.Random(args.seed).shuffle(ordered_ids)

    selected = set(ordered_ids[: args.max_trajectories])
    output = Path(args.output)
    sample_count, trajectory_count = write_subset(
        paths=input_paths,
        output=output,
        selected_ids=selected,
        group_key=args.group_key,
    )

    print("[subset] input_files:", ",".join(os.fspath(p) for p in input_paths))
    print(f"[subset] available_trajectories: {len(trajectory_ids)}")
    print(f"[subset] requested_trajectories: {args.max_trajectories}")
    print(f"[subset] selected_trajectories: {trajectory_count}")
    print(f"[subset] selected_samples: {sample_count}")
    print(f"[subset] output: {output}")


if __name__ == "__main__":
    main()
