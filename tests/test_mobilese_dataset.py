import json

from verl.utils.dataset import load_mobilese_data


def _sample(task_id: str) -> dict:
    return {
        "task_id": task_id,
        "step_number": 0,
        "is_positive": True,
        "ground_truth": {"action": "click"},
        "raw_response": "click",
        "metadata": {
            "impact": "positive",
            "attempt_stats": {"total_attempts": 1, "pass@attempts": 1},
        },
        "conversations": [],
    }


def test_load_mobilese_data_supports_comma_separated_json_files(tmp_path):
    first_path = tmp_path / "train_first.json"
    second_path = tmp_path / "train_second.json"
    first_path.write_text(json.dumps([_sample("task_1"), _sample("task_2")]), encoding="utf-8")
    second_path.write_text(json.dumps([_sample("task_3")]), encoding="utf-8")

    samples, task_ids = load_mobilese_data(
        data_path=f"{first_path}, {second_path}",
        load_all=True,
    )

    assert [sample["task_id"] for sample in samples] == ["task_1", "task_2", "task_3"]
    assert set(task_ids) == {"task_1", "task_2", "task_3"}
