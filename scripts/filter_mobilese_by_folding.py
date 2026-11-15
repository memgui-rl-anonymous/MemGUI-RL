#!/usr/bin/env python3
"""Filter MobileSE/MemGUI verl JSON by folding supervision type.

Supported modes:
  - span_only: keep samples whose gold <folding> range spans more than one step.
  - span_with_step_mix: keep all span-level samples, then mix in step-level
    samples according to a Span:Step ratio such as 9:1.
  - valid_natural_matched: use the same positive span/step support and total
    sample count as span_with_step_mix, but preserve the natural class prior.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
from collections import Counter
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable


FOLDING_RE = re.compile(r"<folding>\s*(.*?)\s*</folding>", re.DOTALL | re.IGNORECASE)


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


def json_default(value: Any) -> Any:
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else float(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got: {value}")


def normalize_mode(mode: str) -> str:
    normalized = mode.strip().lower().replace("-", "_")
    aliases = {
        "all": "all",
        "none": "all",
        "span": "span_only",
        "span_only": "span_only",
        "span_level": "span_only",
        "span_level_abstraction": "span_only",
        "span_with_step": "span_with_step_mix",
        "span_step_mix": "span_with_step_mix",
        "span_with_step_mix": "span_with_step_mix",
        "span_step_9_1": "span_with_step_mix",
        "natural_matched": "valid_natural_matched",
        "valid_natural": "valid_natural_matched",
        "valid_natural_matched": "valid_natural_matched",
    }
    if normalized not in aliases:
        raise ValueError(
            "Unknown mode: "
            f"{mode}. Expected one of: all, span_only, span_with_step_mix, valid_natural_matched."
        )
    return aliases[normalized]


def parse_ratio(ratio: str) -> tuple[int, int]:
    if ":" in ratio:
        left, right = ratio.split(":", 1)
    elif "," in ratio:
        left, right = ratio.split(",", 1)
    else:
        left, right = ratio, "1"
    span_parts = int(left)
    step_parts = int(right)
    if span_parts <= 0 or step_parts < 0:
        raise ValueError(f"Invalid Span:Step ratio: {ratio}")
    return span_parts, step_parts


def extract_text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "".join(parts)
    return ""


def assistant_response(sample: dict[str, Any]) -> str:
    raw_response = sample.get("raw_response")
    if raw_response:
        return str(raw_response)

    for message in sample.get("conversations", []):
        if isinstance(message, dict) and message.get("role") == "assistant":
            return extract_text_from_content(message.get("content", ""))

    return ""


def folding_from_ground_truth(sample: dict[str, Any]) -> Any:
    ground_truth = sample.get("ground_truth")
    if isinstance(ground_truth, str):
        try:
            ground_truth = json.loads(ground_truth)
        except json.JSONDecodeError:
            return None
    if isinstance(ground_truth, dict):
        return ground_truth.get("folding")
    return None


def extract_folding(sample: dict[str, Any]) -> Any:
    response = assistant_response(sample)
    match = FOLDING_RE.search(response)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            return None
    return folding_from_ground_truth(sample)


def classify_folding(sample: dict[str, Any]) -> str:
    folding = extract_folding(sample)
    if not isinstance(folding, dict):
        return "none"

    fold_range = folding.get("range")
    if not (isinstance(fold_range, list) and len(fold_range) == 2):
        return "invalid"

    try:
        start, end = int(fold_range[0]), int(fold_range[1])
    except (TypeError, ValueError):
        return "invalid"

    if end > start:
        return "span"
    if end == start:
        return "step"
    return "invalid"


def sample_is_positive(sample: dict[str, Any]) -> bool:
    if isinstance(sample.get("is_positive"), bool):
        return bool(sample["is_positive"])
    metadata = sample.get("metadata", {})
    if isinstance(metadata, dict):
        if isinstance(metadata.get("is_reasonable"), bool):
            return bool(metadata["is_reasonable"])
        return metadata.get("impact") == "positive"
    return False


def iter_eligible_samples(paths: list[Path], positive_only: bool) -> Iterable[dict[str, Any]]:
    for path in paths:
        for sample in iter_json_array(path):
            if positive_only and not sample_is_positive(sample):
                continue
            yield sample


def count_folding_types(paths: list[Path], positive_only: bool) -> Counter:
    counts: Counter = Counter()
    for sample in iter_eligible_samples(paths, positive_only=positive_only):
        counts[classify_folding(sample)] += 1
    return counts


def selected_ordinals(available_count: int, target_count: int, seed: int) -> set[int]:
    if target_count <= 0 or available_count <= 0:
        return set()
    target_count = min(available_count, target_count)
    return set(random.Random(seed).sample(range(available_count), target_count))


def matched_natural_class_counts(span_count: int, step_count: int, matched_total: int) -> tuple[int, int]:
    """Allocate a fixed sample budget according to the eligible natural class prior."""
    eligible_total = span_count + step_count
    if matched_total < 0 or matched_total > eligible_total:
        raise ValueError(f"matched_total must be in [0, {eligible_total}], got {matched_total}")
    if eligible_total == 0 or matched_total == 0:
        return 0, 0

    target_span = round(matched_total * span_count / eligible_total)
    min_span = max(0, matched_total - step_count)
    max_span = min(span_count, matched_total)
    target_span = min(max(target_span, min_span), max_span)
    return target_span, matched_total - target_span


def should_keep(
    folding_type: str,
    mode: str,
    span_ordinal: int | None,
    step_ordinal: int | None,
    selected_span_ids: set[int],
    selected_step_ids: set[int],
) -> bool:
    if mode == "span_only":
        return folding_type == "span"
    if mode == "span_with_step_mix":
        if folding_type == "span":
            return True
        return folding_type == "step" and step_ordinal in selected_step_ids
    if mode == "valid_natural_matched":
        if folding_type == "span":
            return span_ordinal in selected_span_ids
        return folding_type == "step" and step_ordinal in selected_step_ids
    if mode == "all":
        return True
    raise ValueError(f"Unsupported mode: {mode}")


def write_filtered(
    paths: list[Path],
    output: Path,
    mode: str,
    selected_span_ids: set[int],
    selected_step_ids: set[int],
    positive_only: bool,
) -> Counter:
    output.parent.mkdir(parents=True, exist_ok=True)
    stats: Counter = Counter()
    first = True
    span_ordinal = 0
    step_ordinal = 0

    with output.open("w", encoding="utf-8") as f:
        f.write("[\n")
        for sample in iter_eligible_samples(paths, positive_only=positive_only):
            folding_type = classify_folding(sample)
            current_span_ordinal = None
            current_step_ordinal = None
            if folding_type == "span":
                current_span_ordinal = span_ordinal
                span_ordinal += 1
            elif folding_type == "step":
                current_step_ordinal = step_ordinal
                step_ordinal += 1

            stats[f"seen_{folding_type}"] += 1
            if not should_keep(
                folding_type,
                mode,
                current_span_ordinal,
                current_step_ordinal,
                selected_span_ids,
                selected_step_ids,
            ):
                continue

            if not first:
                f.write(",\n")
            json.dump(sample, f, ensure_ascii=False, default=json_default)
            first = False
            stats[f"kept_{folding_type}"] += 1
            stats["kept_total"] += 1
        f.write("\n]\n")

    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="JSON path or comma-separated JSON paths")
    parser.add_argument("--output", required=True, help="Output filtered JSON path")
    parser.add_argument(
        "--mode",
        default="span_only",
        help="Filtering mode: all, span_only, span_with_step_mix, valid_natural_matched.",
    )
    parser.add_argument(
        "--span-step-ratio",
        default="9:1",
        help=(
            "Reference Span:Step ratio. For 9:1, span_with_step_mix keeps all span and ceil(span/9) "
            "step samples; valid_natural_matched uses that same total size with the natural class prior."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--positive-only", type=str_to_bool, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mode = normalize_mode(args.mode)
    input_paths = split_paths(args.input)
    for path in input_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    counts = count_folding_types(input_paths, positive_only=args.positive_only)
    selected_spans: set[int] = set()
    selected_steps: set[int] = set()
    target_step_count = 0
    matched_total = 0
    natural_span_count = 0
    natural_step_count = 0
    if mode in {"span_with_step_mix", "valid_natural_matched"}:
        span_parts, step_parts = parse_ratio(args.span_step_ratio)
        target_step_count = math.ceil(counts["span"] * step_parts / span_parts)
        target_step_count = min(counts["step"], target_step_count)
        matched_total = counts["span"] + target_step_count

    if mode == "span_with_step_mix":
        selected_steps = selected_ordinals(counts["step"], target_step_count, seed=args.seed)
    elif mode == "valid_natural_matched":
        natural_span_count, natural_step_count = matched_natural_class_counts(
            span_count=counts["span"],
            step_count=counts["step"],
            matched_total=matched_total,
        )
        selected_spans = selected_ordinals(counts["span"], natural_span_count, seed=args.seed)
        selected_steps = selected_ordinals(counts["step"], natural_step_count, seed=args.seed + 1)

    stats = write_filtered(
        paths=input_paths,
        output=Path(args.output),
        mode=mode,
        selected_span_ids=selected_spans,
        selected_step_ids=selected_steps,
        positive_only=args.positive_only,
    )

    print("[fold-filter] input_files:", ",".join(os.fspath(p) for p in input_paths))
    print(f"[fold-filter] mode: {mode}")
    print(f"[fold-filter] positive_only: {args.positive_only}")
    print(f"[fold-filter] available_span: {counts['span']}")
    print(f"[fold-filter] available_step: {counts['step']}")
    print(f"[fold-filter] available_none: {counts['none']}")
    print(f"[fold-filter] available_invalid: {counts['invalid']}")
    if mode in {"span_with_step_mix", "valid_natural_matched"}:
        print(f"[fold-filter] span_step_ratio: {args.span_step_ratio}")
        print(f"[fold-filter] matched_total: {matched_total}")
    if mode == "span_with_step_mix":
        print(f"[fold-filter] requested_step_mix: {target_step_count}")
    elif mode == "valid_natural_matched":
        print(f"[fold-filter] natural_matched_span: {natural_span_count}")
        print(f"[fold-filter] natural_matched_step: {natural_step_count}")
    print(f"[fold-filter] kept_span: {stats['kept_span']}")
    print(f"[fold-filter] kept_step: {stats['kept_step']}")
    print(f"[fold-filter] kept_total: {stats['kept_total']}")
    print(f"[fold-filter] output: {args.output}")


if __name__ == "__main__":
    main()
