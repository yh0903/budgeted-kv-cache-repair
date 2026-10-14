"""Align unchanged token prefixes and suffixes across a context edit."""

from __future__ import annotations


def shared_prefix_len(left: list[int], right: list[int]) -> int:
    index = 0
    while index < min(len(left), len(right)) and left[index] == right[index]:
        index += 1
    return index


def shared_suffix_len(left: list[int], right: list[int], prefix_len: int) -> int:
    maximum = min(len(left) - prefix_len, len(right) - prefix_len)
    count = 0
    while count < maximum and left[-1 - count] == right[-1 - count]:
        count += 1
    return count


def token_alignment(old_ids: list[int], new_ids: list[int]) -> dict[str, int]:
    prefix = shared_prefix_len(old_ids, new_ids)
    suffix = shared_suffix_len(old_ids, new_ids, prefix)
    return {
        "prefix": prefix,
        "suffix": suffix,
        "old_suffix_start": len(old_ids) - suffix,
        "new_suffix_start": len(new_ids) - suffix,
        "mandatory_changed_tokens": len(new_ids) - prefix - suffix,
        "position_shift": (len(new_ids) - suffix) - (len(old_ids) - suffix),
    }
