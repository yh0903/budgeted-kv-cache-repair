import math
import random

import pytest

from kv_cache_repair.baselines import (
    _unit_scores_to_token_scores,
    bm25_unit_scores,
    max_relevance_contiguous_positions,
    random_semantic_unit_positions,
    top_b_positions,
)
from kv_cache_repair.semantic_units import select_semantic_unit_support


@pytest.mark.parametrize("budget, expected", [(0, []), (2, [2, 3]), (20, [1, 2, 3, 4])])
def test_dense_top_tokens_respect_suffix_budget_and_earlier_ties(budget, expected):
    assert top_b_positions([100, 0, 2, 2, 1], low=1, high=5, budget=budget) == expected


@pytest.mark.parametrize("budget, expected", [(0, []), (2, [2, 3]), (20, [1, 2, 3, 4])])
def test_single_window_maximizes_sum_inside_suffix(budget, expected):
    assert max_relevance_contiguous_positions([100, -10, 5, 4, 0], low=1, high=5, budget=budget) == expected


def test_single_window_equal_scores_choose_earliest_start():
    assert max_relevance_contiguous_positions([1.0] * 8, low=2, high=8, budget=3) == [2, 3, 4]


def test_bm25_scoring_and_shared_allocator_use_identical_units():
    vocabulary = {0: "apple", 1: "pear", 2: "plum"}

    class Tokenizer:
        def decode(self, token_ids):
            return " ".join(vocabulary[token] for token in token_ids)

    units = [{"start": 0, "end": 3}, {"start": 3, "end": 6}]
    scores = bm25_unit_scores(
        tokenizer=Tokenizer(),
        edited_ids=[0, 0, 1, 1, 2, 2],
        units=units,
        query_text="APPLE",
    )
    token_scores = _unit_scores_to_token_scores(length=6, units=units, unit_scores=scores)
    support, _ = select_semantic_unit_support(units=units, token_scores=token_scores, budget_tokens=3)

    assert scores == pytest.approx([math.log(2.0) * 2 * 2.2 / 3.2, 0.0])
    assert token_scores == [scores[0]] * 3 + [0.0] * 3
    assert support == [0, 1, 2]


def test_bm25_empty_units_return_empty_scores():
    assert bm25_unit_scores(tokenizer=None, edited_ids=[], units=[], query_text="query") == []


def test_unit_score_conversion_rejects_mismatched_counts():
    with pytest.raises(ValueError, match="unit score count"):
        _unit_scores_to_token_scores(length=4, units=[{"start": 0, "end": 4}], unit_scores=[])


def test_random_units_are_seeded_and_leave_global_rng_untouched():
    units = [{"start": 4, "end": 8}, {"start": 8, "end": 12}, {"start": 12, "end": 16}]
    global_state = random.getstate()

    support, metadata = random_semantic_unit_positions(units=units, budget=6, seed=7)
    repeated = random_semantic_unit_positions(units=units, budget=6, seed=7)

    assert (support, metadata) == repeated
    assert support == [5, 6, 12, 13, 14, 15]
    assert metadata["unit_order"] == [2, 0, 1]
    assert random.getstate() == global_state


def test_random_units_zero_budget_returns_empty_support():
    assert random_semantic_unit_positions(units=[], budget=0, seed=7) == (
        [],
        {"unit_order": [], "partial_unit": None},
    )
