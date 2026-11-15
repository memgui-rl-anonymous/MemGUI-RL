import json

from scripts.filter_mobilese_by_folding import (
    matched_natural_class_counts,
    selected_ordinals,
    write_filtered,
)


def _sample(start: int, end: int) -> dict:
    return {
        "is_positive": True,
        "raw_response": f'<folding>{{"range":[{start},{end}],"summary":"x"}}</folding>',
    }


def test_matched_natural_class_counts_preserve_budget_and_natural_prior():
    span_count, step_count = matched_natural_class_counts(
        span_count=9,
        step_count=21,
        matched_total=10,
    )

    assert (span_count, step_count) == (3, 7)
    assert span_count + step_count == 10


def test_natural_and_foldaware_outputs_have_equal_size(tmp_path):
    samples = [_sample(1, 2) for _ in range(9)] + [_sample(1, 1) for _ in range(21)]
    input_path = tmp_path / "input.json"
    input_path.write_text(json.dumps(samples), encoding="utf-8")

    foldaware_output = tmp_path / "foldaware.json"
    write_filtered(
        paths=[input_path],
        output=foldaware_output,
        mode="span_with_step_mix",
        selected_span_ids=set(),
        selected_step_ids=selected_ordinals(21, 1, seed=7),
        positive_only=True,
    )

    natural_output = tmp_path / "natural.json"
    natural_span_count, natural_step_count = matched_natural_class_counts(9, 21, matched_total=10)
    write_filtered(
        paths=[input_path],
        output=natural_output,
        mode="valid_natural_matched",
        selected_span_ids=selected_ordinals(9, natural_span_count, seed=7),
        selected_step_ids=selected_ordinals(21, natural_step_count, seed=8),
        positive_only=True,
    )

    foldaware = json.loads(foldaware_output.read_text(encoding="utf-8"))
    natural = json.loads(natural_output.read_text(encoding="utf-8"))
    assert len(foldaware) == len(natural) == 10
    assert sum('"range":[1,2]' in sample["raw_response"] for sample in foldaware) == 9
    assert sum('"range":[1,1]' in sample["raw_response"] for sample in foldaware) == 1
    assert sum('"range":[1,2]' in sample["raw_response"] for sample in natural) == 3
    assert sum('"range":[1,1]' in sample["raw_response"] for sample in natural) == 7
