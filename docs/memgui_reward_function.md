# ConAct verifier (reward function)

`examples/reward_function/r1gui_memgui.py` scores one ConAct response against the
annotated state of MemGUI-3K. It returns four components in `[0, 1]` per response,
`format`, `action_type`, `action_params` and `folding`, plus a scalar `overall` that is
only used for monitoring. The reward-decoupled estimator (`--algorithm gdpo`, the FARPO
estimator) consumes the four raw components; the matched scalar baseline
(`--algorithm scalar_grpo_matched`) first aggregates them with the same weights and then
normalises the sum.

## Response format

A ConAct response carries five tagged fields, in this order:

```
<thinking> ... </thinking>                          reasoning (required)
<folding> {"range": [s, t], "summary": "..."} </folding>   context-folding directive (required from step 2 on)
<tool_call> {"name": "mobile_use", "arguments": {...}} </tool_call>   one UI action or memory operation (required)
<ui_observation> ... </ui_observation>              description of the current screen (required)
<action_intent> ... </action_intent>                what the next step should achieve (required)
```

Coordinates are normalised to a virtual 1000 x 1000 screen. Supported UI actions are
`click`, `long_press`, `swipe`, `type`, `answer`, `system_button`, `wait` and `terminate`;
memory operations are `memory_add`, `memory_update` and `memory_delete`.

## Components

**Format** — fraction of applicable checks that pass: the reasoning, UI-observation and
intent tags, a tool-call JSON with the arguments required by its action, and, from step two
on, a folding JSON with a two-endpoint `range` and a `summary`.

**Action type** — 1 when the operation family matches the annotation, else 0
(`swipe` <-> `scroll` and `terminate` <-> `complete` are mapped onto each other; a memory
operation never matches a UI action and vice versa).

**Action parameters** — dispatched on the annotated type and 0 whenever the type is wrong:

| action | rule |
| --- | --- |
| `click`, `long_press` | Euclidean distance between the predicted point and the annotated target on the 0–1000 screen, threshold 140 (point-in-box when a box is annotated); a long press without a duration receives 0.8x the coordinate score |
| `swipe` | direction agreement (finger-motion convention) |
| `system_button` | button agreement |
| `wait` | 1 for a positive duration, otherwise 0.5 |
| `type`, `answer`, `open` | exact normalised match or token-F1 >= 0.5 |
| `memory_add`, `memory_update` | 0.3 * [F1(id) >= 0.5] + 0.7 * F1(content) |
| `memory_delete` | [F1(id) >= 0.5] |
| argument-free actions | 1 |

**Folding** — compares the predicted interval `[s, t]` with the annotated one (inclusive
integer steps). An exact match gives 1, a missing or invalid interval 0, and a partial
overlap is scored by `IoU x depth accuracy` where the depth accuracy is
`min(pred_depth, gt_depth) / max(pred_depth, gt_depth)`. The optional summary-similarity
term (`folding_include_summary`) and depth bonus (`folding_depth_bonus`) are disabled in
every run reported in the paper. For parsed responses whose state has no annotated fold the
component is the constant 1, which group centering removes.

**Overall (monitoring only)** — `0.1 * format + 0.4 * action_type + 0.4 * action_params`,
rescaled by 0.9 and combined with `0.1 * folding` when a fold is annotated. The weights are
set with `--reward_weights "format,action_type,action_params,folding"` in
`examples/memgui_8b_farpo.sh` (default `0.1,0.4,0.4,0.1`).

## Ground-truth format

The annotated state is parsed from the assistant turn of each MemGUI-3K record by
`verl/utils/dataset.py::parse_ground_truth_from_conversations` into

```json
{"action": "click", "gt_bbox": [x, y], "input_text": "", "is_normalized": true, "bbox_valid": true,
 "folding": {"range": [s, t], "summary": "..."}, "memory_id": "", "content": "", "description": ""}
```

## Self-test

```bash
python3 examples/reward_function/r1gui_memgui.py
```
