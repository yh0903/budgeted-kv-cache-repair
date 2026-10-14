import math

import pytest

from kv_cache_repair.semantic_units import (
    best_forward_causal_window,
    score_semantic_units,
    select_semantic_unit_support,
    semantic_char_units,
    semantic_unit_token_ranges,
)


class CharacterTokenizer:
    """Lossless character offsets, with a deliberately unusable decoder."""

    def __call__(self, text, *, add_special_tokens=False, return_offsets_mapping=False):
        assert add_special_tokens is False
        encoded = {"input_ids": [ord(character) for character in text]}
        if return_offsets_mapping:
            encoded["offset_mapping"] = [(index, index + 1) for index in range(len(text))]
        return encoded

    def decode(self, token_ids):
        raise AssertionError("unit construction must use canonical text, not decoded tokens")


def _unit(start, end, container=0):
    return {
        "start": start,
        "end": end,
        "container_id": container,
        "container_kind": "ordinary_text",
        "boundary_kind": "generic_sentence_or_paragraph",
    }


@pytest.mark.parametrize("peak, expected", [(None, (0, 5)), (2, (2, 7)), (10, (10, 15))])
def test_partial_window_tracks_peak_without_tail_bias(peak, expected):
    scores = [0.0] * 20 if peak is None else [-100.0] * 20
    if peak is not None:
        scores[peak] = 0.0

    window = best_forward_causal_window(start=0, end=20, width=5, token_scores=scores)

    assert (window["start"], window["end"]) == expected
    assert window["decay_tokens"] == 16.0


@pytest.mark.parametrize("width", [0, -1, 21])
def test_partial_window_rejects_invalid_width(width):
    with pytest.raises(ValueError, match="partial-window width"):
        best_forward_causal_window(start=0, end=20, width=width, token_scores=[0.0] * 20)


def test_semantic_units_use_canonical_offsets_and_clip_delayed_query():
    tokenizer = CharacterTokenizer()
    text = "Alpha support sentence.\n\nQuestion: Where?"
    token_ids = tokenizer(text)["input_ids"]
    token_end = text.index("\n\nQuestion:")

    units = semantic_unit_token_ranges(
        tokenizer=tokenizer,
        text=text,
        expected_token_ids=token_ids,
        suffix_start=0,
        token_end=token_end,
    )
    support, _ = select_semantic_unit_support(
        units=units,
        token_scores=[float(index) for index in range(token_end)],
        budget_tokens=5,
    )

    assert [(unit["start"], unit["end"]) for unit in units] == [(0, token_end)]
    assert support == list(range(token_end - 5, token_end))


def test_semantic_units_reject_tokenizer_identity_drift():
    with pytest.raises(RuntimeError, match="tokenizer identity drift"):
        semantic_unit_token_ranges(
            tokenizer=CharacterTokenizer(),
            text="A sentence.",
            expected_token_ids=[0],
            suffix_start=0,
        )


def test_long_units_use_balanced_chunks():
    text = "A" * 257
    units = semantic_unit_token_ranges(
        tokenizer=CharacterTokenizer(),
        text=text,
        expected_token_ids=[ord("A")] * 257,
        suffix_start=0,
    )

    assert [(unit["start"], unit["end"]) for unit in units] == [(0, 86), (86, 172), (172, 257)]


def test_structured_units_keep_tool_and_json_containers():
    text = '<tool_result>{"place": "Oslo."}</tool_result>\n\nMetadata: {"count": 2}\n'

    units = semantic_char_units(text)
    structured = [
        (unit["boundary_kind"], text[unit["start"] : unit["end"]])
        for unit in units
        if unit["boundary_kind"] != "generic_sentence_or_paragraph"
    ]

    assert structured == [
        ("tool_container", '<tool_result>{"place": "Oslo."}</tool_result>'),
        ("json_container", '{"count": 2}'),
    ]
    assert "".join(text[unit["start"] : unit["end"]] for unit in units) == text


def test_unit_ranking_uses_top_four_not_all_token_mean():
    units = [_unit(0, 8), _unit(8, 16)]
    scores = [4.0] * 4 + [-100.0] * 4 + [3.0] * 8

    ranked = score_semantic_units(units=units, token_scores=scores)
    support, metadata = select_semantic_unit_support(units=units, token_scores=scores, budget_tokens=8)

    assert [unit["unit_score"] for unit in ranked] == [4.0, 3.0]
    assert support == list(range(8))
    assert metadata["selections"][0]["selection_kind"] == "complete_unit"
    assert metadata["kv_probe_token_calls"] == metadata["planning_expansion_token_calls"] == 0


def test_unit_ties_prefer_earlier_position_and_fill_final_window():
    units = [_unit(4, 12), _unit(12, 20)]

    support, metadata = select_semantic_unit_support(
        units=units,
        token_scores=[0.0] * 20,
        budget_tokens=11,
    )

    assert support == list(range(4, 15))
    assert [entry["selection_kind"] for entry in metadata["selections"]] == [
        "complete_unit",
        "partial_forward_causal_window",
    ]
    assert metadata["budget_tokens"] == 11


@pytest.mark.parametrize("bad_score", [math.inf, -math.inf, math.nan])
def test_unit_scoring_rejects_nonfinite_values(bad_score):
    with pytest.raises(RuntimeError, match="invalid semantic unit token scores"):
        score_semantic_units(units=[_unit(0, 2)], token_scores=[1.0, bad_score])
