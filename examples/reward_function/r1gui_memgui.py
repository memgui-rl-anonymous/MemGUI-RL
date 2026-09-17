"""
MemGUI Reward Function - GUI Agent with Memory and Proactive Context Folding

Reward function for the MemGUI (ConAct) agent: scores UI actions, memory operations and context-folding directives.

Required output structure (in order):
1. <thinking></thinking>: reasoning (required)
2. <folding></folding>: context-folding directive (required from step 2 on)
3. <tool_call></tool_call>: tool-call JSON (required)
4. <ui_observation></ui_observation>: UI observation (required)
5. <action_intent></action_intent>: action intent (required)

Tool-call format example:
<thinking>I need to click the button to proceed.</thinking>
<folding>{"range": [1, 1], "summary": "[Step 1] Opened Settings app"}</folding>
<tool_call>
{"name": "mobile_use", "arguments": {"action": "click", "coordinate": [500, 300]}}
</tool_call>
<ui_observation>The screen shows a Settings menu with Wi-Fi, Bluetooth, and Display options.</ui_observation>
<action_intent>Click the Wi-Fi option to configure network settings.</action_intent>

Coordinates: normalised to 0-1000 (virtual 1000x1000 screen)

Supported actions:
- UI actions: click, long_press, swipe, type, answer, system_button, wait, terminate
- Memory operations: memory_add, memory_update, memory_delete

Scoring (v2):
- overall = 0.1 * format + 0.45 * action_type + 0.45 * action_params (default weights; see compute_score)
- memory operations are scored against memory ground truth
- folding directives are scored when the ground truth carries a folding annotation
"""

import json
import re
from typing import Any


# Metadata required by the EasyR1 framework
REWARD_NAME = "r1gui_memgui"
REWARD_TYPE = "batch"  # batch mode

# Default weights (override through reward_function_kwargs)
DEFAULT_FORMAT_WEIGHT = 0.1
DEFAULT_ACTION_TYPE_WEIGHT = 0.4
DEFAULT_ACTION_PARAMS_WEIGHT = 0.4
DEFAULT_FOLDING_WEIGHT = 0.1


def calculate_f1_score(predicted_str: str, ground_truth_str: str) -> float:
    """Token-level F1 between two strings."""
    predicted_str = predicted_str.replace("[", "").replace("]", "")
    ground_truth_str = ground_truth_str.replace("[", "").replace("]", "")
    predicted_tokens = set(predicted_str.lower().split())
    ground_truth_tokens = set(ground_truth_str.lower().split())

    if len(predicted_tokens) == 1 and len(ground_truth_tokens) == 1:
        predicted_token = list(predicted_tokens)[0]
        ground_truth_token = list(ground_truth_tokens)[0]
        if predicted_token in ground_truth_token or ground_truth_token in predicted_token:
            return 1.0

    common_tokens = predicted_tokens.intersection(ground_truth_tokens)
    if len(predicted_tokens) == 0:
        precision = 0.0
    else:
        precision = len(common_tokens) / len(predicted_tokens)
    if len(ground_truth_tokens) == 0:
        recall = 0.0
    else:
        recall = len(common_tokens) / len(ground_truth_tokens)

    if precision + recall == 0:
        f1_score = 0.0
    else:
        f1_score = 2.0 * (precision * recall) / (precision + recall)
    return f1_score


def extract_tool_call(content: str) -> dict | None:
    """Extract the tool_call JSON from a ConAct response."""
    tool_call_pattern = r"<tool_call>\s*(.*?)\s*</tool_call>"
    match = re.search(tool_call_pattern, content, re.DOTALL)
    if match:
        try:
            tool_call_str = match.group(1).strip()
            tool_call = json.loads(tool_call_str)
            # Two accepted layouts:
            # 1. {"name": "mobile_use", "arguments": {...}}
            # 2. {"name": "mobile_use", "action": "...", ...}
            if "arguments" in tool_call:
                return tool_call["arguments"]
            else:
                return {k: v for k, v in tool_call.items() if k != "name"}
        except json.JSONDecodeError:
            return None
    return None


def extract_action(content: str) -> str:
    """Extract the action type from a ConAct response."""
    tool_call = extract_tool_call(content)
    if tool_call:
        return tool_call.get("action", "no action")
    return "no action"


def extract_coordinate(content: str) -> tuple[list[int], bool]:
    """Extract the (0-1000 normalised) coordinate from a ConAct response."""
    tool_call = extract_tool_call(content)
    if tool_call:
        coord = tool_call.get("coordinate", None)
        if coord and len(coord) >= 2:
            try:
                return [int(coord[0]), int(coord[1])], True
            except (ValueError, TypeError):
                pass
    return [0, 0], False


def extract_text(content: str) -> str:
    """Extract the text argument from a ConAct response."""
    tool_call = extract_tool_call(content)
    if tool_call:
        return tool_call.get("text", "")
    return ""


def extract_button(content: str) -> str:
    """Extract the system button from a ConAct response."""
    tool_call = extract_tool_call(content)
    if tool_call:
        return tool_call.get("button", "")
    return ""


def extract_direction(content: str) -> str:
    """Extract the swipe direction (inferred from coordinate and coordinate2)."""
    tool_call = extract_tool_call(content)
    if tool_call:
        # explicit direction first
        if "direction" in tool_call:
            return tool_call["direction"]

        # otherwise infer the direction from coordinate and coordinate2
        coord1 = tool_call.get("coordinate", [])
        coord2 = tool_call.get("coordinate2", [])
        if len(coord1) >= 2 and len(coord2) >= 2:
            dx = coord2[0] - coord1[0]
            dy = coord2[1] - coord1[1]
            if abs(dx) > abs(dy):
                return "right" if dx > 0 else "left"
            else:
                return "down" if dy > 0 else "up"
    return ""


def extract_status(content: str) -> str:
    """Extract the status of a terminate action."""
    tool_call = extract_tool_call(content)
    if tool_call:
        return tool_call.get("status", "")
    return ""


def extract_time(content: str) -> float:
    """Extract the time argument (wait / long_press)."""
    tool_call = extract_tool_call(content)
    if tool_call:
        time_val = tool_call.get("time", 0)
        try:
            return float(time_val)
        except (ValueError, TypeError):
            return 0.0
    return 0.0


def extract_folding(content: str) -> dict | None:
    """Extract the folding directive from a ConAct response."""
    folding_pattern = r"<folding>\s*(.*?)\s*</folding>"
    match = re.search(folding_pattern, content, re.DOTALL | re.IGNORECASE)
    if match:
        try:
            folding_str = match.group(1).strip()
            if folding_str.startswith("{"):
                return json.loads(folding_str)
            else:
                # not JSON: treat the text as the summary
                return {"range": [0, 0], "summary": folding_str}
        except json.JSONDecodeError:
            return {"range": [0, 0], "summary": match.group(1).strip()}
    return None


def extract_ui_observation(content: str) -> str:
    """Extract the UI observation from a ConAct response."""
    pattern = r"<ui_observation>\s*(.*?)\s*</ui_observation>"
    match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""


def extract_action_intent(content: str) -> str:
    """Extract the action intent from a ConAct response."""
    pattern = r"<action_intent>\s*(.*?)\s*</action_intent>"
    match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""


def folding_depth(folding: dict | None) -> int:
    """Number of steps covered by a folding range; 0 when the range is invalid."""
    if not isinstance(folding, dict):
        return 0

    fold_range = folding.get("range", [])
    if not (isinstance(fold_range, list) and len(fold_range) == 2):
        return 0

    try:
        start, end = int(fold_range[0]), int(fold_range[1])
    except (TypeError, ValueError):
        return 0

    if start < 1 or end < start:
        return 0

    return end - start + 1


def folding_range_iou(gt_folding: dict | None, pred_folding: dict | None) -> float:
    """IoU between the ground-truth and predicted folding ranges."""
    if folding_depth(gt_folding) == 0 or folding_depth(pred_folding) == 0:
        return 0.0

    gt_start, gt_end = [int(x) for x in gt_folding["range"]]
    pred_start, pred_end = [int(x) for x in pred_folding["range"]]

    overlap_start = max(gt_start, pred_start)
    overlap_end = min(gt_end, pred_end)
    if overlap_end < overlap_start:
        return 0.0

    overlap_len = overlap_end - overlap_start + 1
    union_len = (gt_end - gt_start + 1) + (pred_end - pred_start + 1) - overlap_len
    return overlap_len / union_len if union_len > 0 else 0.0


def folding_observation_metrics(pred: dict, gt: dict) -> dict[str, float]:
    """Per-sample GT / predicted folding observations, aggregated by the training loop for logging."""
    gt_folding = gt.get("folding", None)
    pred_folding = pred.get("folding", None)

    gt_has_folding = gt_folding is not None
    pred_has_folding = pred_folding is not None
    gt_depth = folding_depth(gt_folding)
    pred_depth = folding_depth(pred_folding)
    gt_range_valid = gt_has_folding and gt_depth > 0
    pred_range_valid = pred_has_folding and pred_depth > 0
    both_valid = gt_range_valid and pred_range_valid

    iou = folding_range_iou(gt_folding, pred_folding) if both_valid else 0.0
    depth_signed_error = float(pred_depth - gt_depth) if both_valid else 0.0
    exact_range_match = bool(
        both_valid
        and gt_folding.get("range", []) == pred_folding.get("range", [])
    )

    return {
        "gt_has_folding": float(gt_has_folding),
        "pred_has_folding": float(pred_has_folding),
        "gt_fold_depth": float(gt_depth),
        "pred_fold_depth": float(pred_depth),
        "gt_fold_range_valid": float(gt_range_valid),
        "pred_fold_range_valid": float(pred_range_valid),
        "fold_both_present": float(gt_has_folding and pred_has_folding),
        "fold_both_valid_range": float(both_valid),
        "fold_missing": float(gt_has_folding and not pred_has_folding),
        "fold_unexpected": float((not gt_has_folding) and pred_has_folding),
        "fold_range_iou": float(iou),
        "fold_exact_range_match": float(exact_range_match),
        "fold_depth_abs_error": abs(depth_signed_error) if both_valid else 0.0,
        "fold_depth_signed_error": depth_signed_error,
        "fold_pred_deeper_or_equal": float(both_valid and pred_depth >= gt_depth),
    }


def memgui_format_reward(predict_str: str, step_number: int = 0) -> float:
    """
    Check that predict_str follows the ConAct output format:
    1. contains <thinking></thinking>
    2. contains <tool_call></tool_call> with valid JSON inside
    3. contains <ui_observation></ui_observation>
    4. contains <action_intent></action_intent>
    5. contains <folding></folding> from step 2 on

    Expected order: <thinking> -> <folding> -> <tool_call> -> <ui_observation> -> <action_intent>

    Args:
        predict_str: model response
        step_number: current step (1-based); folding is optional at step 1 and required afterwards

    Returns:
        format score in [0, 1]
    """
    # folding is checked from step 2 on
    need_folding_check = step_number >= 2
    total_checks = 5 if need_folding_check else 4
    score = 0.0

    # <thinking>
    thinking_pattern = re.compile(r"<thinking>.*?</thinking>", re.DOTALL | re.IGNORECASE)
    if thinking_pattern.search(predict_str):
        score += 1.0

    # <tool_call>
    tool_call_pattern = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE)
    tool_call_match = tool_call_pattern.search(predict_str)
    if tool_call_match:
        # the JSON inside tool_call must parse
        try:
            tool_call_json = json.loads(tool_call_match.group(1).strip())

            # required fields
            if "arguments" in tool_call_json:
                action_args = tool_call_json["arguments"]
            else:
                action_args = {k: v for k, v in tool_call_json.items() if k != "name"}

            if "action" in action_args:
                action = action_args["action"]

                # per-action required arguments
                valid_action = True
                if action in ["click", "long_press"]:
                    if "coordinate" not in action_args:
                        valid_action = False
                    else:
                        coord = action_args["coordinate"]
                        if not (isinstance(coord, list) and len(coord) >= 2):
                            valid_action = False

                if action == "swipe":
                    coord = action_args.get("coordinate", [])
                    coord2 = action_args.get("coordinate2", [])
                    direction = action_args.get("direction", "")
                    if not direction and not (len(coord) >= 2 and len(coord2) >= 2):
                        valid_action = False

                if action == "type":
                    if "text" not in action_args or not action_args["text"]:
                        valid_action = False

                if action == "system_button":
                    if "button" not in action_args:
                        valid_action = False

                if action == "terminate":
                    if "status" not in action_args:
                        valid_action = False

                # memory operations
                if action in ["memory_add", "memory_update"]:
                    if "memory_id" not in action_args or "content" not in action_args:
                        valid_action = False

                if action == "memory_delete":
                    if "memory_id" not in action_args:
                        valid_action = False

                if valid_action:
                    score += 1.0
        except json.JSONDecodeError:
            pass

    # <ui_observation>
    ui_obs_pattern = re.compile(r"<ui_observation>.*?</ui_observation>", re.DOTALL | re.IGNORECASE)
    if ui_obs_pattern.search(predict_str):
        score += 1.0

    # <action_intent>
    action_intent_pattern = re.compile(r"<action_intent>.*?</action_intent>", re.DOTALL | re.IGNORECASE)
    if action_intent_pattern.search(predict_str):
        score += 1.0

    # <folding> (required from step 2 on)
    if need_folding_check:
        folding_pattern = re.compile(r"<folding>\s*\{.*?\}\s*</folding>", re.DOTALL | re.IGNORECASE)
        if folding_pattern.search(predict_str):
            # the JSON inside folding must parse
            folding_match = re.search(r"<folding>\s*(.*?)\s*</folding>", predict_str, re.DOTALL | re.IGNORECASE)
            if folding_match:
                try:
                    folding_json = json.loads(folding_match.group(1).strip())
                    # required fields
                    if "range" in folding_json and "summary" in folding_json:
                        folding_range = folding_json.get("range", [])
                        if isinstance(folding_range, list) and len(folding_range) == 2:
                            score += 1.0
                except json.JSONDecodeError:
                    pass  # unparsable JSON: no credit

    return score / total_checks


def pixel_to_normalized(coord: list, image_size: list) -> list:
    """Convert pixel coordinates to the 0-1000 normalised frame."""
    if not image_size or len(image_size) < 2:
        return coord
    w, h = image_size
    if w <= 0 or h <= 0:
        return coord
    return [coord[0] * 1000 / w, coord[1] * 1000 / h] if len(coord) >= 2 else coord


def _parse_prediction(predict_str: str) -> dict:
    """Parse the model response into an action type and its arguments."""
    tool_call = extract_tool_call(predict_str)

    if not tool_call:
        return {
            "action": "no action",
            "normalized_action": "no action",
            "coord": [0, 0],
            "coord_valid": False,
            "text": "",
            "button": "",
            "direction": "",
            "time": 0.0,
            "status": "",
            "memory_id": "",
            "memory_content": "",
            "memory_description": "",
            "folding": None,
        }

    pred_action = tool_call.get("action", "no action").lower()

    # coordinate
    coord = tool_call.get("coordinate", None)
    if coord and len(coord) >= 2:
        try:
            pred_coord = [int(coord[0]), int(coord[1])]
            coord_valid = True
        except (ValueError, TypeError):
            pred_coord = [0, 0]
            coord_valid = False
    else:
        pred_coord = [0, 0]
        coord_valid = False

    pred_text = tool_call.get("text", "")
    pred_button = tool_call.get("button", "")
    pred_time = float(tool_call.get("time", 0) or 0)
    pred_status = tool_call.get("status", "")

    # direction (inferred from coordinate and coordinate2)
    pred_direction = ""
    if "direction" in tool_call:
        pred_direction = tool_call["direction"]
    else:
        coord1 = tool_call.get("coordinate", [])
        coord2 = tool_call.get("coordinate2", [])
        if len(coord1) >= 2 and len(coord2) >= 2:
            dx = coord2[0] - coord1[0]
            dy = coord2[1] - coord1[1]
            if abs(dx) > abs(dy):
                pred_direction = "right" if dx > 0 else "left"
            else:
                pred_direction = "down" if dy > 0 else "up"

    # memory-operation fields
    memory_id = tool_call.get("memory_id", "")
    memory_content = tool_call.get("content", "")
    memory_description = tool_call.get("description", "")

    # folding directive
    folding = extract_folding(predict_str)

    # action-name mapping: ConAct action -> canonical GUI-R1 action
    action_mapping = {
        "click": "click",
        "long_press": "long_press",
        "swipe": "scroll",
        "type": "type",
        "system_button": None,
        "wait": "wait",
        "terminate": "complete",
        "answer": "answer",
        "open": "open_app",
        # memory operations keep their names
        "memory_add": "memory_add",
        "memory_update": "memory_update",
        "memory_delete": "memory_delete",
    }

    # system_button mapping
    if pred_action == "system_button":
        button_mapping = {
            "back": "press_back",
            "home": "press_home",
            "menu": "press_recent",
            "enter": "enter",
        }
        pred_action = button_mapping.get(pred_button.lower(), pred_action)

    # canonical action name
    normalized_pred_action = action_mapping.get(pred_action, pred_action)

    return {
        "action": pred_action,
        "normalized_action": normalized_pred_action,
        "coord": pred_coord,
        "coord_valid": coord_valid,
        "text": pred_text,
        "button": pred_button,
        "direction": pred_direction,
        "time": pred_time,
        "status": pred_status,
        "memory_id": memory_id,
        "memory_content": memory_content,
        "memory_description": memory_description,
        "folding": folding,
    }


def _parse_ground_truth(ground_truth: str) -> dict:
    """Parse the ground truth.

    Accepted formats:
    - UI action: {"action": "click", "gt_bbox": [...], "input_text": "...", ...}
    - memory operation: {"action": "memory_add", "memory_id": "...", "content": "...", ...}
    - folding directive: {"folding": {"range": [start, end], "summary": "..."}}
    """
    ground_truth_dict = json.loads(ground_truth)

    result = {
        "action": ground_truth_dict.get("action", "").lower(),
        "gt_bbox": ground_truth_dict.get("gt_bbox", []),
        "input_text": ground_truth_dict.get("input_text", ""),
        "image_size": ground_truth_dict.get("image_size", None),
        "bbox_valid": ground_truth_dict.get("bbox_valid", True),
        "is_normalized": ground_truth_dict.get("is_normalized", False),
        "time": ground_truth_dict.get("time", 0),
        # memory-operation fields
        "memory_id": ground_truth_dict.get("memory_id", ""),
        "memory_content": ground_truth_dict.get("content", ""),
        "memory_description": ground_truth_dict.get("description", ""),
        # folding fields
        "folding": ground_truth_dict.get("folding", None),
    }

    return result


def memgui_action_type_reward(pred: dict, gt: dict) -> float:
    """
    Action-type reward.

    1.0 when the predicted action type matches the ground truth, else 0.0.
    

    Memory operations are matched against memory ground truth.
    """
    gt_action = gt["action"]
    pred_action = pred["action"]
    normalized_pred_action = pred["normalized_action"]

    # memory operations
    memory_actions = ["memory_add", "memory_update", "memory_delete"]
    if pred_action in memory_actions:
        # GT is a memory operation too: compare
        if gt_action in memory_actions:
            return 1.0 if pred_action == gt_action else 0.0
        else:
            # GT is a UI action but the prediction is a memory operation
            return 0.0

    # GT is a memory operation but the prediction is a UI action
    if gt_action in memory_actions:
        return 0.0

    # action-type match
    if normalized_pred_action == gt_action or pred_action == gt_action:
        return 1.0
    return 0.0


def memgui_action_params_reward(pred: dict, gt: dict) -> float:
    """
    Action-parameter reward.

    Checks the arguments that matter for the given action type.
    Returns a value in [0, 1].
    """
    gt_action = gt["action"]
    gt_bbox = gt["gt_bbox"]
    gt_input_text = gt["input_text"]
    image_size = gt["image_size"]
    gt_bbox_valid = gt["bbox_valid"]
    is_normalized = gt["is_normalized"]
    gt_time = gt["time"]

    pred_coord = pred["coord"]
    coord_valid = pred["coord_valid"]
    pred_text = pred["text"]
    pred_direction = pred["direction"]
    pred_time = pred["time"]
    pred_status = pred["status"]

    # argument-free actions: press_back, press_home, press_recent, enter
    # a correct action type earns full parameter credit
    no_params_actions = ["press_back", "press_home", "press_recent", "enter"]
    if gt_action in no_params_actions:
        return 1.0

    # click-like actions: coordinate
    if gt_action in ["click", "long_press", "moveto", "doubleclick", "rightclick"]:
        # did the model output a valid coordinate?
        if not coord_valid:
            return 0.0

        # is the ground-truth coordinate valid?
        if not gt_bbox_valid:
            coord_score = 1.0
        elif len(gt_bbox) == 2:
            if is_normalized:
                gt_normalized = gt_bbox
            else:
                gt_normalized = pixel_to_normalized(gt_bbox, image_size)
            dist = ((pred_coord[0] - gt_normalized[0]) ** 2 + (pred_coord[1] - gt_normalized[1]) ** 2) ** 0.5
            coord_score = 1.0 if dist < 140 else 0.0
        elif len(gt_bbox) == 4:
            if is_normalized:
                gt_normalized_min = [gt_bbox[0], gt_bbox[1]]
                gt_normalized_max = [gt_bbox[2], gt_bbox[3]]
            else:
                gt_normalized_min = pixel_to_normalized([gt_bbox[0], gt_bbox[1]], image_size)
                gt_normalized_max = pixel_to_normalized([gt_bbox[2], gt_bbox[3]], image_size)
            if (gt_normalized_min[0] < pred_coord[0] < gt_normalized_max[0]) and (
                gt_normalized_min[1] < pred_coord[1] < gt_normalized_max[1]
            ):
                coord_score = 1.0
            else:
                coord_score = 0.0
        else:
            coord_score = 0.0

        # long_press: when the GT has a duration, the prediction must provide one too
        if gt_action == "long_press" and gt_time > 0:
            if pred_time > 0:
                return coord_score
            else:
                return coord_score * 0.8

        return coord_score

    # scroll / swipe: direction
    elif gt_action in ["scroll", "swipe"]:
        gt_direction = gt_input_text.lower()
        pred_dir = pred_direction.lower()

        # direction conventions:
        # - MemGUI-3K describes the finger motion (swipe down = finger moves down = content scrolls up)
        # - GUI-R1 describes the content motion (scroll down = content moves down = finger moves up)
        # GUI-R1 ground truth (is_normalized=False) is therefore flipped before comparison
        if not is_normalized:
            # GUI-R1 format: flip the GT direction to the finger-motion convention
            direction_invert = {
                "up": "down",
                "down": "up",
                "left": "right",
                "right": "left",
            }
            gt_direction = direction_invert.get(gt_direction, gt_direction)

        if pred_dir == gt_direction:
            return 1.0
        else:
            return 0.0

    # text input: text similarity
    elif gt_action in ["type", "open_app", "select"]:
        if calculate_f1_score(pred_text, gt_input_text) >= 0.5:
            return 1.0
        else:
            return 0.0

    # answer: answer-text similarity
    elif gt_action == "answer":
        if gt_input_text:
            if calculate_f1_score(pred_text, gt_input_text) >= 0.5:
                return 1.0
            else:
                return 0.0
        else:
            return 1.0

    # terminate / complete: status
    elif gt_action == "complete":
        gt_status = gt_input_text.lower()
        if gt_status:
            if pred_status.lower() == gt_status:
                return 1.0
            else:
                return 0.0
        else:
            return 1.0

    # wait: a duration must be present
    elif gt_action == "wait":
        if pred_time > 0:
            return 1.0
        else:
            return 0.5

    # memory operations: memory_id and content
    elif gt_action in ["memory_add", "memory_update"]:
        gt_memory_id = gt.get("memory_id", "")
        gt_memory_content = gt.get("memory_content", "")
        pred_memory_id = pred.get("memory_id", "")
        pred_memory_content = pred.get("memory_content", "")

        score = 0.0
        # memory_id (token F1)
        if gt_memory_id and pred_memory_id:
            id_score = calculate_f1_score(pred_memory_id, gt_memory_id)
            score += 0.3 * (1.0 if id_score >= 0.5 else 0.0)
        elif not gt_memory_id:
            score += 0.3  # GT has no memory_id: any id is accepted

        # content (token F1)
        if gt_memory_content and pred_memory_content:
            content_score = calculate_f1_score(pred_memory_content, gt_memory_content)
            score += 0.7 * content_score  # content matters more and is scored continuously
        elif not gt_memory_content:
            score += 0.7  # GT has no content

        return score

    elif gt_action == "memory_delete":
        gt_memory_id = gt.get("memory_id", "")
        pred_memory_id = pred.get("memory_id", "")

        if gt_memory_id and pred_memory_id:
            id_score = calculate_f1_score(pred_memory_id, gt_memory_id)
            return 1.0 if id_score >= 0.5 else 0.0
        elif not gt_memory_id:
            return 1.0  # GT has no memory_id
        else:
            return 0.0  # GT has one, the prediction does not

    # other actions: no argument requirements
    else:
        return 1.0


def memgui_folding_reward(
    pred: dict, 
    gt: dict, 
    include_summary: bool = True,
    depth_bonus: float = 0.1,
) -> float:
    """
    Folding reward.

    Compares the predicted folding directive with the annotated one.
    Returns a value in [0, 1].

    Rules:
    - an exact range match earns full credit
    - a partial match is scored by IoU times depth accuracy
    - the closer the predicted depth to the annotated depth, the higher the reward
    - an optional bonus rewards predictions at least as deep as the annotation (disabled in the paper)

    Args:
        pred: parsed prediction
        gt: parsed ground truth
        include_summary: also score the summary text (default True; the paper uses False)
        depth_bonus: depth-bonus coefficient (default 0.1; the paper uses 0.0)
            awarded when the predicted depth is at least the annotated depth
            which encourages deeper folds even when they slightly overshoot
    """
    gt_folding = gt.get("folding", None)
    pred_folding = pred.get("folding", None)

    # no folding annotation in the GT
    if not gt_folding:
        # a predicted fold is not penalised (folding is optional here)
        return 1.0

    # GT has a fold but the prediction does not
    if not pred_folding:
        return 0.0

    score = 0.0

    # range match
    gt_range = gt_folding.get("range", [])
    pred_range = pred_folding.get("range", [])

    if len(gt_range) == 2 and len(pred_range) == 2:
        gt_start, gt_end = gt_range
        pred_start, pred_end = pred_range
        
        gt_depth = gt_end - gt_start + 1
        pred_depth = pred_end - pred_start + 1
        
        # range overlap (IoU)
        overlap_start = max(gt_start, pred_start)
        overlap_end = min(gt_end, pred_end)
        
        if overlap_end >= overlap_start:
            overlap_len = overlap_end - overlap_start + 1
            union_len = gt_depth + pred_depth - overlap_len
            iou = overlap_len / union_len if union_len > 0 else 0
        else:
            iou = 0.0
        
        # depth accuracy
        # depth ratio = min(pred_depth, gt_depth) / max(pred_depth, gt_depth)
        # only an exact depth earns full credit
        if gt_depth > 0 and pred_depth > 0:
            depth_accuracy = min(pred_depth, gt_depth) / max(pred_depth, gt_depth)
        else:
            depth_accuracy = 0.0
        
        # combined score: IoU times depth accuracy
        if gt_range == pred_range:
            # exact range match
            range_score = 1.0
        else:
            # partial match: IoU * depth_accuracy
            # the model needs both the right overlap and the right depth
            range_score = iou * depth_accuracy
        
        if include_summary:
            score += 0.3 * range_score  # range weight 30%
        else:
            score += 1.0 * range_score  # range weight 100% (summary not scored)
        
        # depth bonus: awarded when the predicted depth is at least the annotated depth
        # (encourages deeper folds)
        # gated on IoU: overshooting only makes sense when the ranges overlap
        if pred_depth >= gt_depth and iou > 0.5:
            # predicted depth >= annotated depth and IoU > 0.5
            # bonus scaled by the depth ratio
            depth_exceed_ratio = min(pred_depth / gt_depth, 2.0) if gt_depth > 0 else 0
            score += depth_bonus * min(depth_exceed_ratio * iou, 1.0)
            
    elif not gt_range:
        # GT has no range
        if include_summary:
            score += 0.3
        else:
            score += 1.0

    # summary similarity (only when include_summary=True)
    if include_summary:
        gt_summary = gt_folding.get("summary", "")
        pred_summary = pred_folding.get("summary", "")

        if gt_summary and pred_summary:
            summary_score = calculate_f1_score(pred_summary, gt_summary)
            score += 0.7 * summary_score
        elif not gt_summary:
            score += 0.7  # GT has no summary

    return min(score, 1.0)  # clip to 1.0


def memgui_accuracy_reward(predict_str: str, ground_truth: str) -> float:
    """
    Combined accuracy reward (kept for backward compatibility).

    overall = 0.5 * action_type_reward + 0.5 * action_params_reward
    """
    try:
        pred = _parse_prediction(predict_str)
        gt = _parse_ground_truth(ground_truth)

        action_type_score = memgui_action_type_reward(pred, gt)

        # a wrong action type zeroes the parameter reward
        if action_type_score == 0.0:
            action_params_score = 0.0
        else:
            action_params_score = memgui_action_params_reward(pred, gt)

        return 0.5 * action_type_score + 0.5 * action_params_score

    except Exception:
        return 0.0


def memgui_compute_score(
    predict_str: str,
    ground_truth: str,
    step_number: int = 0,
    format_weight: float = DEFAULT_FORMAT_WEIGHT,
    action_type_weight: float = DEFAULT_ACTION_TYPE_WEIGHT,
    action_params_weight: float = DEFAULT_ACTION_PARAMS_WEIGHT,
    folding_weight: float = DEFAULT_FOLDING_WEIGHT,
    folding_include_summary: bool = True,
    folding_depth_bonus: float = 0.1,
) -> dict[str, float]:
    """
    Score one sample.

    Formula:
    - overall = format_weight * format + action_type_weight * action_type + action_params_weight * action_params
    - when the GT carries a folding annotation the folding score is added and the weights re-normalised

    Weights:
    - format: output format (default 10%)
    - action_type: action type (default 40%)
    - action_params: action arguments (default 40%)
    - folding: context-folding directive (default 10%)

    Args:
        predict_str: model response
        ground_truth: ground-truth JSON string
        step_number: current step (1-based); decides whether the folding tag is required
        format_weight: weight of the format score
        action_type_weight: weight of the action-type score
        action_params_weight: weight of the action-parameter score
        folding_weight: weight of the folding score (when the GT has a fold)
        folding_include_summary: score the summary text as well (default True)
        folding_depth_bonus: depth-bonus coefficient (default 0.1)
            awarded when the predicted depth is at least the annotated depth and IoU > 0.5
    """
    format_score = memgui_format_reward(predict_str, step_number=step_number)
    fold_metrics = {
        "gt_has_folding": 0.0,
        "pred_has_folding": 0.0,
        "gt_fold_depth": 0.0,
        "pred_fold_depth": 0.0,
        "gt_fold_range_valid": 0.0,
        "pred_fold_range_valid": 0.0,
        "fold_both_present": 0.0,
        "fold_both_valid_range": 0.0,
        "fold_missing": 0.0,
        "fold_unexpected": 0.0,
        "fold_range_iou": 0.0,
        "fold_exact_range_match": 0.0,
        "fold_depth_abs_error": 0.0,
        "fold_depth_signed_error": 0.0,
        "fold_pred_deeper_or_equal": 0.0,
    }

    try:
        pred = _parse_prediction(predict_str)
        gt = _parse_ground_truth(ground_truth)
        fold_metrics = folding_observation_metrics(pred, gt)

        action_type_score = memgui_action_type_reward(pred, gt)

        # a wrong action type zeroes the parameter reward
        if action_type_score == 0.0:
            action_params_score = 0.0
        else:
            action_params_score = memgui_action_params_reward(pred, gt)

        # folding score (when the GT has a fold)
        folding_score = memgui_folding_reward(
            pred, gt, 
            include_summary=folding_include_summary,
            depth_bonus=folding_depth_bonus,
        )
        has_folding_gt = gt.get("folding") is not None

        # overall score
        if has_folding_gt:
            # with a folding annotation, folding_weight is taken proportionally from the other components
            # e.g. format_weight=0.1, action_type_weight=0.45, action_params_weight=0.45, folding_weight=0.1
            # becomes format=0.09, action_type=0.405, action_params=0.405, folding=0.1
            adjusted_format_weight = format_weight * (1 - folding_weight)
            adjusted_action_type_weight = action_type_weight * (1 - folding_weight)
            adjusted_action_params_weight = action_params_weight * (1 - folding_weight)

            overall = (
                adjusted_format_weight * format_score +
                adjusted_action_type_weight * action_type_score +
                adjusted_action_params_weight * action_params_score +
                folding_weight * folding_score
            )
        else:
            # no folding annotation: standard weights
            overall = format_weight * format_score + action_type_weight * action_type_score + action_params_weight * action_params_score

    except Exception:
        action_type_score = 0.0
        action_params_score = 0.0
        folding_score = 0.0
        overall = 0.0

    return {
        "overall": overall,
        "action_type": action_type_score,
        "action_params": action_params_score,
        "format": format_score,
        "folding": folding_score,
        **fold_metrics,
    }


def compute_score(
    reward_inputs: list[dict[str, Any]],
    debug: bool = False,
    format_weight: float = DEFAULT_FORMAT_WEIGHT,
    action_type_weight: float = DEFAULT_ACTION_TYPE_WEIGHT,
    action_params_weight: float = DEFAULT_ACTION_PARAMS_WEIGHT,
    folding_weight: float = DEFAULT_FOLDING_WEIGHT,
    folding_include_summary: bool = True,
    folding_depth_bonus: float = 0.1,
) -> list[dict[str, float]]:
    """
    Score a batch of samples (entry point expected by EasyR1).

    Args:
        reward_inputs: list of samples, each with:
            - response: model response
            - response_length: response length
            - ground_truth: ground-truth JSON string in one of the formats:
                - UI action: {"action": "click", "gt_bbox": [...], ...}
                - memory operation: {"action": "memory_add", "memory_id": "...", "content": "..."}
                - folding annotation: {"folding": {"range": [start, end], "summary": "..."}}
            - step_number: (optional) current step (1-based), decides whether the folding tag is required
        debug: print per-sample details
        format_weight: weight of the format score
        action_type_weight: weight of the action-type score
        action_params_weight: weight of the action-parameter score
        folding_weight: weight of the folding score
        folding_include_summary: score the summary text as well (default True)
        folding_depth_bonus: depth-bonus coefficient (default 0.1)
            awarded when the predicted depth is at least the annotated depth and IoU > 0.5

    Returns:
        list of per-sample score dicts with:
            - overall: weighted total
            - format: format score in [0, 1]
            - action_type: action-type score (0 or 1)
            - action_params: action-parameter score in [0, 1]
            - folding: folding score in [0, 1]
    """
    scores = []

    for i, reward_input in enumerate(reward_inputs):
        response = reward_input.get("response", "")
        ground_truth = reward_input.get("ground_truth", "{}")
        step_number = reward_input.get("step_number", 0)

        score = memgui_compute_score(
            response, ground_truth, step_number=step_number,
            format_weight=format_weight,
            action_type_weight=action_type_weight,
            action_params_weight=action_params_weight,
            folding_weight=folding_weight,
            folding_include_summary=folding_include_summary,
            folding_depth_bonus=folding_depth_bonus,
        )
        scores.append(score)

        # debug: print details of the first ten samples
        if debug and i < 10:
            try:
                gt_dict = json.loads(ground_truth)
                pred_coord, coord_valid = extract_coordinate(response)
                pred_action = extract_action(response)
                gt_action = gt_dict.get("action", "unknown")
                gt_bbox = gt_dict.get("gt_bbox", [])
                is_normalized = gt_dict.get("is_normalized", False)

                print(f"\n[DEBUG] Sample {i}:")
                print(f"  pred_action: {pred_action}, pred_coord: {pred_coord}")
                print(f"  gt_action: {gt_action}, gt_bbox: {gt_bbox}, is_normalized: {is_normalized}")
                print(f"  score: {score}")

                # click-like actions: also print the coordinate distance
                if gt_action in ["click", "long_press"] and coord_valid and len(gt_bbox) == 2:
                    if is_normalized:
                        dist = ((pred_coord[0] - gt_bbox[0]) ** 2 + (pred_coord[1] - gt_bbox[1]) ** 2) ** 0.5
                        print(f"  distance: {dist:.2f} (threshold: 140)")
            except Exception as e:
                print(f"[DEBUG] Error in sample {i}: {e}")

    return scores


# Self-test
if __name__ == "__main__":
    test_cases = [
        # test 1: correct click
        {
            "response": """<thinking>I need to click the button.</thinking>
<folding>{"range": [1, 1], "summary": "[Step 1] Opened Settings app"}</folding>
<tool_call>
{"name": "mobile_use", "arguments": {"action": "click", "coordinate": [500, 300]}}
</tool_call>
<ui_observation>The screen shows a Settings menu with Wi-Fi option.</ui_observation>
<action_intent>Click the Wi-Fi button to configure network.</action_intent>""",
            "response_length": 350,
            "ground_truth": json.dumps(
                {"action": "click", "gt_bbox": [450, 250, 550, 350], "input_text": "no input text"}
            ),
        },
        # test 2: correct swipe
        {
            "response": """<thinking>I need to swipe down to see more content.</thinking>
<tool_call>
{"name": "mobile_use", "arguments": {"action": "swipe", "coordinate": [500, 300], "coordinate2": [500, 600]}}
</tool_call>
<ui_observation>The screen shows a list that continues below.</ui_observation>
<action_intent>Swipe down to reveal more items in the list.</action_intent>""",
            "response_length": 300,
            "ground_truth": json.dumps({"action": "scroll", "gt_bbox": [], "input_text": "down"}),
        },
        # test 3: memory operation against a UI ground truth (expects 0)
        {
            "response": """<thinking>I will store this information in memory.</thinking>
<tool_call>
{"name": "mobile_use", "arguments": {"action": "memory_add", "memory_id": "price", "content": "$99.99"}}
</tool_call>
<ui_observation>The screen shows a product with price $99.99.</ui_observation>
<action_intent>Store the price in memory for later use.</action_intent>""",
            "response_length": 300,
            "ground_truth": json.dumps({"action": "click", "gt_bbox": [450, 250], "input_text": ""}),
        },
        # test 4: correct memory operation
        {
            "response": """<thinking>I need to save the product price.</thinking>
<tool_call>
{"name": "mobile_use", "arguments": {"action": "memory_add", "memory_id": "product_price", "content": "$99.99"}}
</tool_call>
<ui_observation>The screen shows iPhone 15 Pro with price $99.99.</ui_observation>
<action_intent>Store the price in memory for comparison.</action_intent>""",
            "response_length": 300,
            "ground_truth": json.dumps({"action": "memory_add", "memory_id": "product_price", "content": "$99.99"}),
        },
        # test 5: folding annotation, matching depth
        {
            "response": """<thinking>I need to click the submit button.</thinking>
<folding>{"range": [1, 3], "summary": "[Steps 1-3] Navigated to checkout page and filled form"}</folding>
<tool_call>
{"name": "mobile_use", "arguments": {"action": "click", "coordinate": [500, 800]}}
</tool_call>
<ui_observation>The checkout form is filled, submit button visible at bottom.</ui_observation>
<action_intent>Click submit to complete the purchase.</action_intent>""",
            "response_length": 400,
            "ground_truth": json.dumps(
                {
                    "action": "click",
                    "gt_bbox": [450, 750, 550, 850],
                    "input_text": "",
                    "folding": {"range": [1, 3], "summary": "[Steps 1-3] Navigated to checkout and filled the form"},
                }
            ),
        },
        # test 6: folding annotation, depth mismatch (folding_include_summary=false)
        {
            "response": """<thinking>I need to click the submit button.</thinking>
<folding>{"range": [1, 1], "summary": "[Step 1] Did something"}</folding>
<tool_call>
{"name": "mobile_use", "arguments": {"action": "click", "coordinate": [500, 800]}}
</tool_call>
<ui_observation>The checkout form is filled, submit button visible at bottom.</ui_observation>
<action_intent>Click submit to complete the purchase.</action_intent>""",
            "response_length": 400,
            "ground_truth": json.dumps(
                {
                    "action": "click",
                    "gt_bbox": [450, 750, 550, 850],
                    "input_text": "",
                    "folding": {"range": [1, 3], "summary": "[Steps 1-3] Navigated to checkout and filled the form"},
                }
            ),
        },
        # test 7: folding annotation, depth mismatch (folding_include_summary=true)
        {
            "response": """<thinking>I need to click the submit button.</thinking>
<folding>{"range": [1, 2], "summary": "[Steps 1-2] Navigated to checkout"}</folding>
<tool_call>
{"name": "mobile_use", "arguments": {"action": "click", "coordinate": [500, 800]}}
</tool_call>
<ui_observation>The checkout form is filled, submit button visible at bottom.</ui_observation>
<action_intent>Click submit to complete the purchase.</action_intent>""",
            "response_length": 400,
            "ground_truth": json.dumps(
                {
                    "action": "click",
                    "gt_bbox": [450, 750, 550, 850],
                    "input_text": "",
                    "folding": {"range": [1, 3], "summary": "[Steps 1-3] Navigated to checkout and filled the form"},
                }
            ),
        },
        # test 8: partially correct format (missing ui_observation)
        {
            "response": """<thinking>I need to type some text.</thinking>
<tool_call>
{"name": "mobile_use", "arguments": {"action": "type", "text": "hello world"}}
</tool_call>
<action_intent>Type the search query.</action_intent>""",
            "response_length": 200,
            "ground_truth": json.dumps({"action": "type", "gt_bbox": [], "input_text": "hello world"}),
        },
    ]

    print("=" * 60)
    print("Tests 1-5: standard (folding_include_summary=True)")
    print("=" * 60)
    scores = compute_score(test_cases[:5], debug=False, folding_include_summary=True)

    for i, (test, score) in enumerate(zip(test_cases[:5], scores), 1):
        gt_dict = json.loads(test["ground_truth"])
        has_folding = "folding" in gt_dict
        gt_folding = gt_dict.get("folding", {})
        pred_folding_match = re.search(r'<folding>\s*\{.*?\}\s*</folding>', test["response"], re.DOTALL)
        pred_range = "N/A"
        if pred_folding_match:
            try:
                folding_json = json.loads(pred_folding_match.group(0).replace("<folding>", "").replace("</folding>", ""))
                pred_range = folding_json.get("range", "N/A")
            except:
                pass
        print(f"\nTest {i}:")
        print(f"  GT Action: {gt_dict.get('action', 'N/A')}, GT Folding: {gt_folding.get('range', 'N/A')}")
        print(f"  Pred Folding: {pred_range}")
        print(f"  Has GT Folding: {has_folding}")
        print(
            f"  score: overall={score['overall']:.3f}, format={score['format']:.2f}, "
            f"action_type={score['action_type']:.2f}, action_params={score['action_params']:.2f}, "
            f"folding={score['folding']:.2f}"
        )

    print("\n" + "=" * 60)
    print("Tests 6-7: fold depth (folding_include_summary=False)")
    print("=" * 60)
    scores_no_summary = compute_score(test_cases[5:7], debug=False, folding_include_summary=False)

    for i, (test, score) in enumerate(zip(test_cases[5:7], scores_no_summary), 6):
        gt_dict = json.loads(test["ground_truth"])
        gt_folding = gt_dict.get("folding", {})
        pred_folding_match = re.search(r'<folding>\s*\{.*?\}\s*</folding>', test["response"], re.DOTALL)
        pred_range = "N/A"
        if pred_folding_match:
            try:
                folding_json = json.loads(pred_folding_match.group(0).replace("<folding>", "").replace("</folding>", ""))
                pred_range = folding_json.get("range", "N/A")
            except:
                pass
        print(f"\nTest {i} (depth mismatch):")
        print(f"  GT Action: {gt_dict.get('action', 'N/A')}, GT Folding: {gt_folding.get('range', 'N/A')}")
        print(f"  Pred Folding: {pred_range}")
        print(f"  GT Depth: {gt_folding.get('range', [0,0])[1] - gt_folding.get('range', [0,0])[0] + 1 if gt_folding.get('range') else 0}")
        print(
            f"  score: overall={score['overall']:.3f}, folding={score['folding']:.4f}"
        )
        print(f"  analysis: GT depth={gt_folding.get('range', [0,0])[1] - gt_folding.get('range', [0,0])[0] + 1}, pred depth={pred_range[1] - pred_range[0] + 1 if pred_range != 'N/A' and len(pred_range)==2 else 0}")
        print(f"  expected: a depth mismatch lowers the folding score (IoU * depth_accuracy < 1.0)")

    print("\n" + "=" * 60)
    print("Depth reward analysis")
    print("=" * 60)
    print("""
For GT folding = [1, 3] (depth 3):
- prediction [1, 3] (depth 3): IoU=1.0, depth_acc=1.0 -> score=1.0
- prediction [1, 2] (depth 2): IoU=0.67, depth_acc=0.67 -> score=0.44
- prediction [1, 1] (depth 1): IoU=0.33, depth_acc=0.33 -> score=0.11

The model has to predict the right depth to score well.
""")
