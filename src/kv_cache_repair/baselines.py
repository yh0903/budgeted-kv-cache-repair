"""Simple allocation controls using the same suffix-token budget as SUR."""

from __future__ import annotations

import math
import random
import re
from collections import Counter
from typing import Any

_WORD = re.compile(r"[a-z0-9]+")
_BM25_K1 = 1.2
_BM25_B = 0.75


def top_b_positions(scores: list[float], *, low: int, high: int, budget: int) -> list[int]:
    budget = min(max(0, int(budget)), max(0, int(high) - int(low)))
    ranked = sorted(
        range(int(low), int(high)),
        key=lambda position: (-float(scores[position]), position),
    )
    return sorted(ranked[:budget])


def max_relevance_contiguous_positions(scores: list[float], *, low: int, high: int, budget: int) -> list[int]:
    budget = min(max(0, int(budget)), max(0, int(high) - int(low)))
    if budget == 0:
        return []
    best_start = int(low)
    window = sum(float(scores[position]) for position in range(low, low + budget))
    best_score = window
    for start in range(int(low) + 1, int(high) - budget + 1):
        window += float(scores[start + budget - 1]) - float(scores[start - 1])
        if window > best_score:
            best_start = start
            best_score = window
    return list(range(best_start, best_start + budget))


def _lexical_tokens(text: str) -> list[str]:
    return _WORD.findall(str(text).lower())


def bm25_unit_scores(
    *,
    tokenizer: Any,
    edited_ids: list[int],
    units: list[dict[str, Any]],
    query_text: str,
) -> list[float]:
    documents = [
        _lexical_tokens(tokenizer.decode(edited_ids[int(unit["start"]) : int(unit["end"])])) for unit in units
    ]
    if not documents:
        return []
    average_length = sum(len(document) for document in documents) / len(documents)
    average_length = average_length or 1.0
    query_terms = Counter(_lexical_tokens(query_text))
    document_frequency = {term: sum(term in document for document in documents) for term in query_terms}
    output: list[float] = []
    for document in documents:
        frequency = Counter(document)
        normalization = 1.0 - _BM25_B + _BM25_B * (len(document) / average_length)
        score = 0.0
        for term, query_frequency in query_terms.items():
            count = frequency.get(term, 0)
            if count == 0:
                continue
            containing = document_frequency[term]
            inverse_frequency = math.log(1.0 + (len(documents) - containing + 0.5) / (containing + 0.5))
            score += (
                query_frequency
                * inverse_frequency
                * count
                * (_BM25_K1 + 1.0)
                / (count + _BM25_K1 * normalization)
            )
        output.append(score)
    return output


def _unit_scores_to_token_scores(
    *,
    length: int,
    units: list[dict[str, Any]],
    unit_scores: list[float],
) -> list[float]:
    if len(units) != len(unit_scores):
        raise ValueError("unit score count does not match semantic-unit count")
    scores = [0.0] * int(length)
    for unit, score in zip(units, unit_scores, strict=True):
        for position in range(int(unit["start"]), int(unit["end"])):
            scores[position] = float(score)
    return scores


def random_semantic_unit_positions(
    *,
    units: list[dict[str, Any]],
    budget: int,
    seed: int,
) -> tuple[list[int], dict[str, Any]]:
    budget = min(
        max(0, int(budget)),
        int(units[-1]["end"]) - int(units[0]["start"]) if units else 0,
    )
    if budget == 0:
        return [], {"unit_order": [], "partial_unit": None}
    rng = random.Random(int(seed))
    order = list(range(len(units)))
    rng.shuffle(order)
    selected: list[int] = []
    partial = None
    for unit_index in order:
        unit = units[unit_index]
        positions = list(range(int(unit["start"]), int(unit["end"])))
        remaining = budget - len(selected)
        if remaining <= 0:
            break
        if len(positions) <= remaining:
            selected.extend(positions)
            continue
        start_offset = rng.randrange(0, len(positions) - remaining + 1)
        selected.extend(positions[start_offset : start_offset + remaining])
        partial = {
            "unit_index": unit_index,
            "start_offset": start_offset,
            "selected_tokens": remaining,
        }
    return sorted(selected), {"unit_order": order, "partial_unit": partial}
