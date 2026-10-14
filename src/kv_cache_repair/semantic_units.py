"""Semantic-unit partitioning and exact-budget SUR V1 allocation."""

from __future__ import annotations

import json
import math
import re
from typing import Any


SEMANTIC_UNIT_MIN_TOKENS = 8
SEMANTIC_UNIT_MAX_TOKENS = 128
SEMANTIC_UNIT_SCORE_TOP_K = 4
FORWARD_CAUSAL_DECAY_TOKENS = 16.0

_CHAT_BLOCK = re.compile(
    r"<\|im_start\|>.*?<\|im_end\|>",
    flags=re.DOTALL,
)
_TOOL_BLOCK = re.compile(
    r"<(?P<tag>tool_call|tool_response|tool_result|function_call|"
    r"function_response)(?:\s[^>]*)?>.*?</(?P=tag)>",
    flags=re.DOTALL | re.IGNORECASE,
)
_SENTENCE_BOUNDARY = re.compile(r"\n[ \t]*\n+|[.!?][\"')\]]*(?:[ \t\r\n]+|(?=[A-Z0-9<]))")


def _json_block_ranges(
    text: str,
    *,
    start: int,
    end: int,
) -> list[tuple[int, int]]:
    decoder = json.JSONDecoder()
    output = []
    cursor = start
    while cursor < end:
        match = re.search(r"[\[{]", text[cursor:end])
        if match is None:
            break
        candidate_start = cursor + match.start()
        line_start = text.rfind("\n", start, candidate_start) + 1
        prefix = text[line_start:candidate_start].strip()
        if prefix and not prefix.endswith((":", "=", ">")):
            cursor = candidate_start + 1
            continue
        try:
            value, relative_end = decoder.raw_decode(text[candidate_start:end])
        except json.JSONDecodeError:
            cursor = candidate_start + 1
            continue
        candidate_end = candidate_start + relative_end
        if not isinstance(value, (dict, list)):
            cursor = candidate_start + 1
            continue
        output.append((candidate_start, candidate_end))
        cursor = candidate_end
    return output


def _base_container_ranges(text: str) -> list[dict[str, Any]]:
    chat_ranges = [(match.start(), match.end()) for match in _CHAT_BLOCK.finditer(text)]
    output = []
    cursor = 0
    container_index = 0
    for start, end in chat_ranges:
        if cursor < start:
            if output and not text[cursor:start].strip():
                output[-1]["end"] = start
            else:
                output.append(
                    {
                        "start": cursor,
                        "end": start,
                        "container_id": container_index,
                        "container_kind": "ordinary_text",
                    }
                )
                container_index += 1
        output.append(
            {
                "start": start,
                "end": end,
                "container_id": container_index,
                "container_kind": "chat_message",
            }
        )
        container_index += 1
        cursor = end
    if cursor < len(text):
        if output and not text[cursor:].strip():
            output[-1]["end"] = len(text)
        else:
            output.append(
                {
                    "start": cursor,
                    "end": len(text),
                    "container_id": container_index,
                    "container_kind": "ordinary_text",
                }
            )
    elif not output:
        output.append(
            {
                "start": 0,
                "end": len(text),
                "container_id": container_index,
                "container_kind": "ordinary_text",
            }
        )
    return [row for row in output if int(row["end"]) > int(row["start"])]


def _structured_subranges(
    text: str,
    container: dict[str, Any],
) -> list[dict[str, Any]]:
    start = int(container["start"])
    end = int(container["end"])
    candidates = [
        {
            "start": match.start(),
            "end": match.end(),
            "kind": "tool_container",
        }
        for match in _TOOL_BLOCK.finditer(text, start, end)
    ]
    candidates.extend(
        {
            "start": json_start,
            "end": json_end,
            "kind": "json_container",
        }
        for json_start, json_end in _json_block_ranges(
            text,
            start=start,
            end=end,
        )
    )
    selected = []
    for candidate in sorted(
        candidates,
        key=lambda row: (
            int(row["start"]),
            -(int(row["end"]) - int(row["start"])),
        ),
    ):
        if any(
            int(candidate["start"]) < int(prior["end"]) and int(prior["start"]) < int(candidate["end"])
            for prior in selected
        ):
            continue
        selected.append(candidate)
    selected.sort(key=lambda row: int(row["start"]))

    output = []
    cursor = start
    child_index = 0
    for candidate in selected:
        candidate_start = int(candidate["start"])
        candidate_end = int(candidate["end"])
        if cursor < candidate_start:
            output.append(
                {
                    "start": cursor,
                    "end": candidate_start,
                    "container_id": container["container_id"],
                    "container_kind": container["container_kind"],
                    "boundary_kind": "generic_text",
                }
            )
        output.append(
            {
                "start": candidate_start,
                "end": candidate_end,
                "container_id": (f"{container['container_id']}:structured:{child_index}"),
                "container_kind": candidate["kind"],
                "boundary_kind": candidate["kind"],
            }
        )
        child_index += 1
        cursor = candidate_end
    if cursor < end:
        output.append(
            {
                "start": cursor,
                "end": end,
                "container_id": container["container_id"],
                "container_kind": container["container_kind"],
                "boundary_kind": "generic_text",
            }
        )
    return [row for row in output if int(row["end"]) > int(row["start"])]


def _sentence_spans(
    *,
    text: str,
    start: int,
    end: int,
) -> list[tuple[int, int]]:
    output = []
    cursor = start
    for match in _SENTENCE_BOUNDARY.finditer(text, start, end):
        boundary = match.end()
        if boundary > cursor:
            output.append((cursor, boundary))
        cursor = boundary
    if cursor < end:
        output.append((cursor, end))
    return output or [(start, end)]


def semantic_char_units(text: str) -> list[dict[str, Any]]:
    output = []
    for container in _base_container_ranges(text):
        for subrange in _structured_subranges(text, container):
            if subrange["boundary_kind"] == "generic_text":
                spans = _sentence_spans(
                    text=text,
                    start=int(subrange["start"]),
                    end=int(subrange["end"]),
                )
                boundary_kind = "generic_sentence_or_paragraph"
            else:
                spans = [
                    (
                        int(subrange["start"]),
                        int(subrange["end"]),
                    )
                ]
                boundary_kind = str(subrange["boundary_kind"])
            output.extend(
                {
                    "start": start,
                    "end": end,
                    "container_id": subrange["container_id"],
                    "container_kind": subrange["container_kind"],
                    "boundary_kind": boundary_kind,
                }
                for start, end in spans
                if end > start
            )
    return output


def _balanced_chunks(
    *,
    start: int,
    end: int,
) -> list[tuple[int, int]]:
    length = int(end) - int(start)
    if length <= 0:
        return []
    count = math.ceil(length / SEMANTIC_UNIT_MAX_TOKENS)
    base, extra = divmod(length, count)
    output = []
    cursor = int(start)
    for index in range(count):
        width = base + (1 if index < extra else 0)
        output.append((cursor, cursor + width))
        cursor += width
    return output


def _split_long_units(
    units: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for unit in units:
        chunks = _balanced_chunks(
            start=int(unit["start"]),
            end=int(unit["end"]),
        )
        for chunk_index, (start, end) in enumerate(chunks):
            output.append(
                {
                    **unit,
                    "start": start,
                    "end": end,
                    "fixed_chunk_index": chunk_index,
                    "fixed_chunk_count": len(chunks),
                }
            )
    return output


def _merge_short_units(
    units: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output = list(units)
    while True:
        merge = None
        for short_index, unit in enumerate(output):
            if int(unit["end"]) - int(unit["start"]) >= SEMANTIC_UNIT_MIN_TOKENS:
                continue
            candidates = []
            if short_index:
                prior = output[short_index - 1]
                if prior["container_id"] == unit["container_id"] and int(prior["end"]) == int(unit["start"]):
                    candidates.append(
                        (
                            int(prior["end"]) - int(prior["start"]),
                            0,
                            short_index - 1,
                        )
                    )
            if short_index + 1 < len(output):
                following = output[short_index + 1]
                if following["container_id"] == unit["container_id"] and int(unit["end"]) == int(
                    following["start"]
                ):
                    candidates.append(
                        (
                            int(following["end"]) - int(following["start"]),
                            1,
                            short_index + 1,
                        )
                    )
            if candidates:
                merge = (
                    short_index,
                    min(candidates)[2],
                )
                break
        if merge is None:
            return output
        short_index, neighbor_index = merge
        left_index = min(short_index, neighbor_index)
        right_index = max(short_index, neighbor_index)
        merged = {
            **output[left_index],
            "start": int(output[left_index]["start"]),
            "end": int(output[right_index]["end"]),
            "boundary_kind": "short_unit_merged_within_container",
        }
        replacement = [
            {
                **merged,
                "start": start,
                "end": end,
                "fixed_chunk_index": chunk_index,
                "fixed_chunk_count": len(
                    _balanced_chunks(
                        start=int(merged["start"]),
                        end=int(merged["end"]),
                    )
                ),
            }
            for chunk_index, (start, end) in enumerate(
                _balanced_chunks(
                    start=int(merged["start"]),
                    end=int(merged["end"]),
                )
            )
        ]
        output[left_index : right_index + 1] = replacement


def semantic_unit_token_ranges(
    *,
    tokenizer: Any,
    text: str,
    expected_token_ids: list[int],
    suffix_start: int,
    token_end: int | None = None,
) -> list[dict[str, Any]]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    token_ids = [int(value) for value in encoded["input_ids"]]
    if token_ids != [int(value) for value in expected_token_ids]:
        raise RuntimeError("semantic unit tokenizer identity drift")
    token_end = len(token_ids) if token_end is None else int(token_end)
    offsets = [(int(start), int(end)) for start, end in encoded["offset_mapping"]]
    if not 0 <= int(suffix_start) < token_end <= len(token_ids):
        raise ValueError("semantic unit suffix start out of range")

    char_units = semantic_char_units(text)
    if not char_units:
        raise RuntimeError("semantic unit construction produced no units")
    token_units = []
    char_index = 0
    current_key = None
    current_unit = None
    for token_index in range(int(suffix_start), token_end):
        start, end = offsets[token_index]
        point = start if end <= start else (start + end - 1) / 2.0
        while char_index + 1 < len(char_units) and point >= int(char_units[char_index]["end"]):
            char_index += 1
        char_unit = char_units[char_index]
        key = (
            char_unit["container_id"],
            int(char_unit["start"]),
            int(char_unit["end"]),
        )
        if key != current_key:
            current_unit = {
                "start": token_index,
                "end": token_index + 1,
                "container_id": char_unit["container_id"],
                "container_kind": char_unit["container_kind"],
                "boundary_kind": char_unit["boundary_kind"],
            }
            token_units.append(current_unit)
            current_key = key
        else:
            current_unit["end"] = token_index + 1

    units = _merge_short_units(_split_long_units(token_units))
    if (
        not units
        or int(units[0]["start"]) != int(suffix_start)
        or int(units[-1]["end"]) != token_end
        or any(int(left["end"]) != int(right["start"]) for left, right in zip(units, units[1:], strict=False))
        or any(int(unit["end"]) - int(unit["start"]) > SEMANTIC_UNIT_MAX_TOKENS for unit in units)
    ):
        raise RuntimeError("semantic unit token partition drift")
    return units


def score_semantic_units(
    *,
    units: list[dict[str, Any]],
    token_scores: list[float],
) -> list[dict[str, Any]]:
    output = []
    for unit_index, unit in enumerate(units):
        start = int(unit["start"])
        end = int(unit["end"])
        scores = [float(value) for value in token_scores[start:end]]
        if not scores or any(not math.isfinite(value) for value in scores):
            raise RuntimeError("invalid semantic unit token scores")
        top_count = min(SEMANTIC_UNIT_SCORE_TOP_K, len(scores))
        top_scores = sorted(scores, reverse=True)[:top_count]
        output.append(
            {
                **unit,
                "unit_index": unit_index,
                "token_count": end - start,
                "unit_score": sum(top_scores) / top_count,
                "score_top_k": top_count,
            }
        )
    return output


def best_forward_causal_window(
    *,
    start: int,
    end: int,
    width: int,
    token_scores: list[float],
) -> dict[str, Any]:
    width = int(width)
    length = int(end) - int(start)
    if not 0 < width <= length:
        raise ValueError("invalid semantic unit partial-window width")
    scores = [float(value) for value in token_scores[start:end]]
    maximum = max(scores)
    decay = math.exp(-1.0 / FORWARD_CAUSAL_DECAY_TOKENS)
    forward_utility = []
    running = 0.0
    for score in scores:
        running = max(
            math.exp(score - maximum),
            decay * running,
        )
        forward_utility.append(running)
    window_sum = sum(forward_utility[:width])
    best_start = 0
    best_utility = window_sum
    for candidate_start in range(1, length - width + 1):
        window_sum += forward_utility[candidate_start + width - 1] - forward_utility[candidate_start - 1]
        if window_sum > best_utility:
            best_start = candidate_start
            best_utility = window_sum
    return {
        "start": int(start) + best_start,
        "end": int(start) + best_start + width,
        "utility": best_utility,
        "decay_tokens": FORWARD_CAUSAL_DECAY_TOKENS,
    }


def select_semantic_unit_support(
    *,
    units: list[dict[str, Any]],
    token_scores: list[float],
    budget_tokens: int,
) -> tuple[list[int], dict[str, Any]]:
    if not units:
        raise ValueError("semantic unit refresh requires units")
    suffix_start = int(units[0]["start"])
    suffix_end = int(units[-1]["end"])
    budget = min(
        max(1, int(budget_tokens)),
        suffix_end - suffix_start,
    )
    scored = score_semantic_units(
        units=units,
        token_scores=token_scores,
    )
    ranked = sorted(
        scored,
        key=lambda unit: (
            -float(unit["unit_score"]),
            int(unit["start"]),
        ),
    )
    support = []
    selections = []
    for rank, unit in enumerate(ranked):
        remaining = budget - len(support)
        if remaining <= 0:
            break
        start = int(unit["start"])
        end = int(unit["end"])
        if end - start <= remaining:
            positions = list(range(start, end))
            selection_kind = "complete_unit"
            partial_window = None
        else:
            partial_window = best_forward_causal_window(
                start=start,
                end=end,
                width=remaining,
                token_scores=token_scores,
            )
            positions = list(
                range(
                    int(partial_window["start"]),
                    int(partial_window["end"]),
                )
            )
            selection_kind = "partial_forward_causal_window"
        support.extend(positions)
        selections.append(
            {
                "rank": rank,
                "unit_index": int(unit["unit_index"]),
                "unit_start": start,
                "unit_end": end,
                "unit_score": float(unit["unit_score"]),
                "selection_kind": selection_kind,
                "selected_start": min(positions),
                "selected_end": max(positions) + 1,
                "selected_tokens": len(positions),
                "partial_window": partial_window,
            }
        )
    if len(support) != budget or len(set(support)) != budget:
        raise RuntimeError("semantic unit refresh support budget drift")
    return sorted(support), {
        "revision": "SEMANTIC_UNIT_REFRESH_V1",
        "information_scope": (
            "current query, edited prompt tokens, and deployable generic container/sentence boundaries only"
        ),
        "kv_probe_token_calls": 0,
        "planning_expansion_token_calls": 0,
        "budget_tokens": budget,
        "unit_count": len(scored),
        "unit_score_rule": "mean of top min(4, unit_token_count) token scores",
        "long_unit_rule": "balanced fixed chunks with maximum 128 tokens",
        "short_unit_rule": (
            "units shorter than 8 tokens merge with an adjacent unit in the same container when one exists"
        ),
        "partial_unit_rule": (
            "maximum-utility contiguous remaining-budget window under "
            "forward exponential causal decay with 16-token scale"
        ),
        "units": scored,
        "selections": selections,
    }
