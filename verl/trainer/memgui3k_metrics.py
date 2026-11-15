"""MemGUI-3K offline-style validation metrics for verl training.

This module ports the core matching logic from
`MemGUI-Rollout/memgui3k_test_eval/evaluate_predictions.py` into a pure in-memory
form so validation can log the same action, memory, folding, format, and
trajectory-level metrics during training.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from typing import Any, Optional


SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 1000
CLICK_THRESHOLD = math.sqrt((SCREEN_WIDTH * 0.14) ** 2 + (SCREEN_HEIGHT * 0.14) ** 2)
SWIPE_THRESHOLD = CLICK_THRESHOLD * 1.5
TEXT_F1_THRESHOLD = 0.5
MEMORY_CONTENT_F1_THRESHOLD = 0.5
FOLD_RANGE_TOLERANCE = 2

UI_ACTIONS = {"click", "long_press", "swipe", "type", "answer", "system_button", "wait", "terminate"}
MEMORY_ACTIONS = {"memory_add", "memory_update", "memory_delete"}
REQUIRED_TAGS = ["thinking", "folding", "tool_call", "ui_observation", "action_intent"]


def calculate_f1_score(predicted_str: str, ground_truth_str: str) -> float:
    if not predicted_str or not ground_truth_str:
        return 0.0
    predicted_tokens = set(predicted_str.lower().split())
    ground_truth_tokens = set(ground_truth_str.lower().split())
    common = predicted_tokens & ground_truth_tokens
    precision = len(common) / len(predicted_tokens) if predicted_tokens else 0.0
    recall = len(common) / len(ground_truth_tokens) if ground_truth_tokens else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def text_matching(gt_text: str, pred_text: str) -> bool:
    if gt_text.strip() == pred_text.strip():
        return True
    return calculate_f1_score(pred_text, gt_text) > TEXT_F1_THRESHOLD


def coord_distance(c1: list[Any], c2: list[Any]) -> float:
    if not c1 or not c2 or len(c1) < 2 or len(c2) < 2:
        return float("inf")
    try:
        return math.sqrt((float(c1[0]) - float(c2[0])) ** 2 + (float(c1[1]) - float(c2[1])) ** 2)
    except (TypeError, ValueError):
        return float("inf")


def _normalize_tool_call(tool_call: dict[str, Any]) -> dict[str, Any]:
    if "arguments" in tool_call and isinstance(tool_call.get("arguments"), dict):
        return tool_call
    if "action" in tool_call:
        return {
            "name": tool_call.get("name", "mobile_use"),
            "arguments": {k: v for k, v in tool_call.items() if k != "name"},
        }
    return tool_call


def parse_tool_call(response: str) -> Optional[dict[str, Any]]:
    m = re.search(r"<tool_call>(.*?)</tool_call>", str(response), re.DOTALL)
    if not m:
        return parse_ground_truth_as_response(response).get("tool_call")
    try:
        parsed = json.loads(m.group(1).strip())
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return _normalize_tool_call(parsed)


def parse_folding(response: str) -> Optional[dict[str, Any]]:
    m = re.search(r"<folding>(.*?)</folding>", str(response), re.DOTALL)
    if not m:
        return parse_ground_truth_as_response(response).get("folding")
    try:
        parsed = json.loads(m.group(1).strip())
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def parse_ground_truth_as_response(response: str) -> dict[str, Any]:
    """Fallback for older validation batches where only ground_truth JSON exists."""
    try:
        gt = json.loads(str(response))
    except (TypeError, json.JSONDecodeError):
        return {}
    if not isinstance(gt, dict) or "action" not in gt:
        return {}

    action = gt.get("action", "")
    args: dict[str, Any] = {"action": action}
    if gt.get("gt_bbox"):
        args["coordinate"] = gt.get("gt_bbox")
    if gt.get("input_text"):
        if action in {"type", "answer"}:
            args["text"] = gt.get("input_text", "")
        elif action in {"complete", "terminate"}:
            args["status"] = gt.get("input_text", "")
        else:
            args["text"] = gt.get("input_text", "")
    if action in MEMORY_ACTIONS:
        args["memory_id"] = gt.get("memory_id", "")
        args["content"] = gt.get("content", gt.get("memory_content", ""))
        args["description"] = gt.get("description", gt.get("memory_description", ""))

    return {
        "tool_call": {"name": "mobile_use", "arguments": args},
        "folding": gt.get("folding"),
    }


def extract_action_info(tool_call: Optional[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    if not tool_call:
        return "", {}
    args = tool_call.get("arguments", {})
    if not isinstance(args, dict):
        return "", {}
    return str(args.get("action", "")), args


def match_ui_action(pred_type: str, pred_args: dict[str, Any], gold_type: str, gold_args: dict[str, Any]) -> dict[str, Any]:
    type_match = pred_type == gold_type
    if not type_match:
        return {"type_match": False, "match_success": False, "info": f"type_mismatch:{pred_type}_vs_{gold_type}"}

    if gold_type in ("click", "long_press"):
        dist = coord_distance(pred_args.get("coordinate"), gold_args.get("coordinate"))
        return {"type_match": True, "match_success": dist <= CLICK_THRESHOLD, "info": f"coord_dist={dist:.1f}"}

    if gold_type == "swipe":
        d1 = coord_distance(pred_args.get("coordinate"), gold_args.get("coordinate"))
        d2 = coord_distance(pred_args.get("coordinate2"), gold_args.get("coordinate2"))
        return {"type_match": True, "match_success": d1 <= SWIPE_THRESHOLD and d2 <= SWIPE_THRESHOLD, "info": f"swipe_d1={d1:.1f},d2={d2:.1f}"}

    if gold_type in ("type", "answer"):
        gt_text = str(gold_args.get("text", "")).strip()
        pred_text = str(pred_args.get("text", "")).strip()
        f1 = calculate_f1_score(pred_text, gt_text)
        return {"type_match": True, "match_success": text_matching(gt_text, pred_text), "info": f"text_f1={f1:.3f}"}

    if gold_type == "system_button":
        success = pred_args.get("button") == gold_args.get("button")
        return {"type_match": True, "match_success": success, "info": f"button_match={success}"}

    if gold_type == "terminate":
        success = pred_args.get("status") == gold_args.get("status")
        return {"type_match": True, "match_success": success, "info": f"status_match={success}"}

    if gold_type == "wait":
        return {"type_match": True, "match_success": True, "info": "wait_ok"}

    return {"type_match": True, "match_success": pred_args == gold_args, "info": f"generic_eq={pred_args == gold_args}"}


def match_memory_action(
    pred_type: str,
    pred_args: dict[str, Any],
    gold_type: str,
    gold_args: dict[str, Any],
) -> dict[str, Any]:
    type_match = pred_type == gold_type
    if not type_match:
        return {"type_match": False, "match_success": False, "info": f"mem_type_mismatch:{pred_type}_vs_{gold_type}"}

    if gold_type == "memory_delete":
        id_match = str(pred_args.get("memory_id", "")).strip() == str(gold_args.get("memory_id", "")).strip()
        return {"type_match": True, "match_success": id_match, "info": f"mem_delete_id_match={id_match}"}

    pred_content = str(pred_args.get("content", "")).strip()
    gold_content = str(gold_args.get("content", "")).strip()
    content_f1 = calculate_f1_score(pred_content, gold_content)
    pred_desc = str(pred_args.get("description", "")).strip()
    gold_desc = str(gold_args.get("description", "")).strip()
    desc_f1 = calculate_f1_score(pred_desc, gold_desc) if gold_desc else 1.0
    success = desc_f1 > MEMORY_CONTENT_F1_THRESHOLD or content_f1 > MEMORY_CONTENT_F1_THRESHOLD
    return {
        "type_match": True,
        "match_success": success,
        "info": f"mem_content_f1={content_f1:.3f},desc_f1={desc_f1:.3f}",
        "content_f1": content_f1,
        "desc_f1": desc_f1,
    }


def match_action(pred_tc: Optional[dict[str, Any]], gold_tc: Optional[dict[str, Any]]) -> dict[str, Any]:
    if pred_tc is None and gold_tc is None:
        return {"type_match": True, "match_success": True, "info": "both_none", "category": "none"}
    if pred_tc is None:
        gold_type, _ = extract_action_info(gold_tc)
        cat = "memory" if gold_type in MEMORY_ACTIONS else "ui"
        return {"type_match": False, "match_success": False, "info": "pred_none", "category": cat, "gold_action": gold_type, "pred_action": ""}
    if gold_tc is None:
        pred_type, _ = extract_action_info(pred_tc)
        cat = "memory" if pred_type in MEMORY_ACTIONS else "ui"
        return {"type_match": False, "match_success": False, "info": "gold_none", "category": cat, "gold_action": "", "pred_action": pred_type}

    pred_type, pred_args = extract_action_info(pred_tc)
    gold_type, gold_args = extract_action_info(gold_tc)
    if gold_type in MEMORY_ACTIONS or pred_type in MEMORY_ACTIONS:
        result = match_memory_action(pred_type, pred_args, gold_type, gold_args)
        result["category"] = "memory"
    else:
        result = match_ui_action(pred_type, pred_args, gold_type, gold_args)
        result["category"] = "ui"
    result["gold_action"] = gold_type
    result["pred_action"] = pred_type
    return result


def match_folding(pred_fold: Optional[dict[str, Any]], gold_fold: Optional[dict[str, Any]]) -> dict[str, Any]:
    pred_has = pred_fold is not None
    gold_has = gold_fold is not None
    if not gold_has and not pred_has:
        return {"presence_match": True, "range_match": True, "depth_type_match": True, "info": "both_absent"}
    if gold_has and not pred_has:
        return {"presence_match": False, "range_match": False, "depth_type_match": False, "info": "pred_missing"}
    if not gold_has and pred_has:
        return {"presence_match": False, "range_match": False, "depth_type_match": False, "info": "pred_extra"}

    pred_range = pred_fold.get("range", []) if isinstance(pred_fold, dict) else []
    gold_range = gold_fold.get("range", []) if isinstance(gold_fold, dict) else []
    range_match = False
    if len(pred_range) == 2 and len(gold_range) == 2:
        try:
            range_match = (
                abs(int(pred_range[0]) - int(gold_range[0])) <= FOLD_RANGE_TOLERANCE
                and abs(int(pred_range[1]) - int(gold_range[1])) <= FOLD_RANGE_TOLERANCE
            )
        except (TypeError, ValueError):
            range_match = False

    pred_is_deep = len(pred_range) == 2 and pred_range[1] > pred_range[0]
    gold_is_deep = len(gold_range) == 2 and gold_range[1] > gold_range[0]
    return {
        "presence_match": True,
        "range_match": range_match,
        "depth_type_match": pred_is_deep == gold_is_deep,
        "gold_is_deep": gold_is_deep,
        "pred_is_deep": pred_is_deep,
        "info": f"range:pred={pred_range},gold={gold_range}",
    }


def check_format_compliance(response: str, step: int) -> dict[str, bool]:
    results = {}
    for tag in REQUIRED_TAGS:
        if tag == "folding" and step <= 1:
            results[tag] = True
            continue
        results[tag] = f"<{tag}>" in str(response) and f"</{tag}>" in str(response)
    return results


def _safe_pct(numerator: float, denominator: float) -> float:
    return round(numerator / denominator * 100, 4) if denominator else 0.0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def evaluate_memgui3k_responses(
    predictions: list[str],
    gold_responses: list[str],
    step_numbers: Optional[list[Any]] = None,
    trajectory_ids: Optional[list[Any]] = None,
) -> dict[str, Any]:
    ui_type_match = 0
    ui_match_success = 0
    ui_total = 0
    ui_action_stats = defaultdict(lambda: {"total": 0, "type_match": 0, "match_success": 0})

    mem_type_match = 0
    mem_match_success = 0
    mem_total = 0
    mem_action_stats = defaultdict(lambda: {"total": 0, "type_match": 0, "match_success": 0})
    mem_content_f1s: list[float] = []
    mem_trigger_gold = 0
    mem_trigger_pred = 0
    mem_trigger_tp = 0

    all_type_match = 0
    all_match_success = 0
    fold_presence: list[bool] = []
    fold_range: list[bool] = []
    fold_depth_type: list[bool] = []
    fold_range_shallow: list[bool] = []
    fold_range_deep: list[bool] = []
    fold_pred_deep_count = 0
    fold_pred_shallow_count = 0
    format_tag_scores = defaultdict(list)
    format_all_correct_count = 0
    step_results = []

    total = min(len(predictions), len(gold_responses))
    for idx in range(total):
        pred_response = predictions[idx] or ""
        gold_response = gold_responses[idx] or ""
        try:
            step = int(step_numbers[idx]) if step_numbers is not None and idx < len(step_numbers) else idx + 1
        except (TypeError, ValueError):
            step = idx + 1
        trajectory_id = (
            str(trajectory_ids[idx])
            if trajectory_ids is not None and idx < len(trajectory_ids) and trajectory_ids[idx] not in (None, "")
            else f"sample-{idx}"
        )

        action_result = match_action(parse_tool_call(pred_response), parse_tool_call(gold_response))
        all_type_match += int(bool(action_result["type_match"]))
        all_match_success += int(bool(action_result["match_success"]))

        gold_action = action_result.get("gold_action", "")
        pred_action = action_result.get("pred_action", "")
        cat = action_result.get("category", "ui")
        if cat == "ui" and gold_action:
            ui_total += 1
            ui_type_match += int(bool(action_result["type_match"]))
            ui_match_success += int(bool(action_result["match_success"]))
            ui_action_stats[gold_action]["total"] += 1
            ui_action_stats[gold_action]["type_match"] += int(bool(action_result["type_match"]))
            ui_action_stats[gold_action]["match_success"] += int(bool(action_result["match_success"]))

        if cat == "memory" and (gold_action in MEMORY_ACTIONS or pred_action in MEMORY_ACTIONS):
            if gold_action in MEMORY_ACTIONS:
                mem_total += 1
                mem_trigger_gold += 1
                mem_type_match += int(bool(action_result["type_match"]))
                mem_match_success += int(bool(action_result["match_success"]))
                mem_action_stats[gold_action]["total"] += 1
                mem_action_stats[gold_action]["type_match"] += int(bool(action_result["type_match"]))
                mem_action_stats[gold_action]["match_success"] += int(bool(action_result["match_success"]))
                if "content_f1" in action_result:
                    mem_content_f1s.append(float(action_result["content_f1"]))
            if pred_action in MEMORY_ACTIONS:
                mem_trigger_pred += 1
            if gold_action in MEMORY_ACTIONS and pred_action in MEMORY_ACTIONS:
                mem_trigger_tp += 1

        fold_result = match_folding(parse_folding(pred_response), parse_folding(gold_response))
        fold_presence.append(bool(fold_result["presence_match"]))
        fold_range.append(bool(fold_result["range_match"]))
        fold_depth_type.append(bool(fold_result["depth_type_match"]))
        if fold_result.get("gold_is_deep") is not None:
            if fold_result["gold_is_deep"]:
                fold_range_deep.append(bool(fold_result["range_match"]))
            else:
                fold_range_shallow.append(bool(fold_result["range_match"]))
        if fold_result.get("pred_is_deep") is not None:
            if fold_result["pred_is_deep"]:
                fold_pred_deep_count += 1
            else:
                fold_pred_shallow_count += 1

        fmt = check_format_compliance(pred_response, step)
        all_ok = all(fmt.values())
        for tag, ok in fmt.items():
            format_tag_scores[tag].append(bool(ok))
        format_all_correct_count += int(all_ok)

        step_results.append(
            {
                "trajectory_id": trajectory_id,
                "match_success": bool(action_result["match_success"]),
            }
        )

    mem_trigger_precision = mem_trigger_tp / mem_trigger_pred if mem_trigger_pred else 0.0
    mem_trigger_recall = mem_trigger_tp / mem_trigger_gold if mem_trigger_gold else 0.0
    mem_trigger_f1 = (
        2 * mem_trigger_precision * mem_trigger_recall / (mem_trigger_precision + mem_trigger_recall)
        if (mem_trigger_precision + mem_trigger_recall)
        else 0.0
    )

    traj_groups = defaultdict(list)
    for step_result in step_results:
        traj_groups[step_result["trajectory_id"]].append(step_result)
    traj_full_match = sum(1 for steps in traj_groups.values() if all(s["match_success"] for s in steps))

    return {
        "total_steps": total,
        "total_trajectories": len(traj_groups),
        "overall": {
            "type_match": all_type_match,
            "match_success": all_match_success,
            "type_acc": _safe_pct(all_type_match, total),
            "match_acc": _safe_pct(all_match_success, total),
        },
        "ui_actions": {
            "total": ui_total,
            "type_match": ui_type_match,
            "match_success": ui_match_success,
            "type_acc": _safe_pct(ui_type_match, ui_total),
            "match_acc": _safe_pct(ui_match_success, ui_total),
            "by_action": dict(ui_action_stats),
        },
        "memory_actions": {
            "total": mem_total,
            "type_match": mem_type_match,
            "match_success": mem_match_success,
            "type_acc": _safe_pct(mem_type_match, mem_total),
            "match_acc": _safe_pct(mem_match_success, mem_total),
            "by_action": dict(mem_action_stats),
            "content_f1_mean": round(_mean(mem_content_f1s), 4),
            "trigger": {
                "gold_count": mem_trigger_gold,
                "pred_count": mem_trigger_pred,
                "tp": mem_trigger_tp,
                "precision": round(mem_trigger_precision * 100, 4),
                "recall": round(mem_trigger_recall * 100, 4),
                "f1": round(mem_trigger_f1 * 100, 4),
            },
        },
        "folding": {
            "presence_acc": _safe_pct(sum(fold_presence), len(fold_presence)),
            "range_acc": _safe_pct(sum(fold_range), len(fold_range)),
            "range_acc_shallow": _safe_pct(sum(fold_range_shallow), len(fold_range_shallow)),
            "range_acc_deep": _safe_pct(sum(fold_range_deep), len(fold_range_deep)),
            "depth_type_acc": _safe_pct(sum(fold_depth_type), len(fold_depth_type)),
            "pred_deep_count": fold_pred_deep_count,
            "pred_shallow_count": fold_pred_shallow_count,
            "pred_deep_ratio": _safe_pct(fold_pred_deep_count, fold_pred_deep_count + fold_pred_shallow_count),
            "gold_shallow_count": len(fold_range_shallow),
            "gold_deep_count": len(fold_range_deep),
        },
        "format_compliance": {
            "all_correct": _safe_pct(format_all_correct_count, total),
            "by_tag": {tag: _safe_pct(sum(scores), len(scores)) for tag, scores in format_tag_scores.items()},
        },
        "trajectory_level": {
            "total": len(traj_groups),
            "full_match": traj_full_match,
            "full_match_rate": _safe_pct(traj_full_match, len(traj_groups)),
        },
    }


def _safe_metric_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip()) or "unknown"


def flatten_report(report: dict[str, Any], prefix: str = "val/memgui3k") -> dict[str, float]:
    metrics: dict[str, float] = {
        f"{prefix}/total_steps": float(report["total_steps"]),
        f"{prefix}/total_trajectories": float(report["total_trajectories"]),
        f"{prefix}/overall_type_acc_pct": float(report["overall"]["type_acc"]),
        f"{prefix}/overall_match_acc_pct": float(report["overall"]["match_acc"]),
        f"{prefix}/ui_type_acc_pct": float(report["ui_actions"]["type_acc"]),
        f"{prefix}/ui_match_acc_pct": float(report["ui_actions"]["match_acc"]),
        f"{prefix}/memory_type_acc_pct": float(report["memory_actions"]["type_acc"]),
        f"{prefix}/memory_match_acc_pct": float(report["memory_actions"]["match_acc"]),
        f"{prefix}/memory_content_f1_mean": float(report["memory_actions"]["content_f1_mean"]),
        f"{prefix}/memory_trigger_precision_pct": float(report["memory_actions"]["trigger"]["precision"]),
        f"{prefix}/memory_trigger_recall_pct": float(report["memory_actions"]["trigger"]["recall"]),
        f"{prefix}/memory_trigger_f1_pct": float(report["memory_actions"]["trigger"]["f1"]),
        f"{prefix}/fold_presence_acc_pct": float(report["folding"]["presence_acc"]),
        f"{prefix}/fold_range_acc_pct": float(report["folding"]["range_acc"]),
        f"{prefix}/fold_range_acc_shallow_pct": float(report["folding"]["range_acc_shallow"]),
        f"{prefix}/fold_range_acc_deep_pct": float(report["folding"]["range_acc_deep"]),
        f"{prefix}/fold_depth_type_acc_pct": float(report["folding"]["depth_type_acc"]),
        f"{prefix}/fold_pred_deep_ratio_pct": float(report["folding"]["pred_deep_ratio"]),
        f"{prefix}/format_all_correct_pct": float(report["format_compliance"]["all_correct"]),
        f"{prefix}/trajectory_full_match_rate_pct": float(report["trajectory_level"]["full_match_rate"]),
    }

    for tag, value in report["format_compliance"]["by_tag"].items():
        metrics[f"{prefix}/format_{_safe_metric_name(tag)}_pct"] = float(value)

    for action, stats in report["ui_actions"]["by_action"].items():
        safe_action = _safe_metric_name(action)
        total = float(stats["total"])
        metrics[f"{prefix}/ui_{safe_action}_total"] = total
        metrics[f"{prefix}/ui_{safe_action}_type_acc_pct"] = _safe_pct(stats["type_match"], stats["total"])
        metrics[f"{prefix}/ui_{safe_action}_match_acc_pct"] = _safe_pct(stats["match_success"], stats["total"])

    for action, stats in report["memory_actions"]["by_action"].items():
        safe_action = _safe_metric_name(action)
        total = float(stats["total"])
        metrics[f"{prefix}/memory_{safe_action}_total"] = total
        metrics[f"{prefix}/memory_{safe_action}_type_acc_pct"] = _safe_pct(stats["type_match"], stats["total"])
        metrics[f"{prefix}/memory_{safe_action}_match_acc_pct"] = _safe_pct(stats["match_success"], stats["total"])

    return metrics


def compute_memgui3k_validation_metrics(
    predictions: list[str],
    gold_responses: list[str],
    step_numbers: Optional[list[Any]] = None,
    trajectory_ids: Optional[list[Any]] = None,
    prefix: str = "val/memgui3k",
) -> dict[str, float]:
    if not predictions or not gold_responses:
        return {}
    report = evaluate_memgui3k_responses(
        predictions=predictions,
        gold_responses=gold_responses,
        step_numbers=step_numbers,
        trajectory_ids=trajectory_ids,
    )
    return flatten_report(report, prefix=prefix)
