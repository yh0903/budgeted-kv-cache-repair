"""Reference schedules for refreshing an explicitly selected support set."""

from __future__ import annotations

import time
from typing import Any

import torch
from transformers.cache_utils import StaticCache

from .cache import clone_cache, repair_window


def positions_to_intervals(positions: list[int]) -> list[tuple[int, int]]:
    """Coalesce sorted unique positions into maximal contiguous runs."""

    intervals: list[tuple[int, int]] = []
    for position in sorted(set(int(p) for p in positions)):
        if intervals and position == intervals[-1][1]:
            intervals[-1] = (intervals[-1][0], position + 1)
        else:
            intervals.append((position, position + 1))
    return intervals


def _validate_intervals(intervals: list[tuple[int, int]], length: int) -> None:
    previous_end = 0
    for start, end in intervals:
        if not previous_end <= start < end <= length:
            raise ValueError("intervals must be sorted, disjoint, and within the edited sequence")
        previous_end = end


@torch.no_grad()
def sequential_repair(
    shim: Any,
    base_cache: Any,
    edited_ids_t: torch.Tensor,
    intervals: list[tuple[int, int]],
) -> tuple[Any, dict[str, Any]]:
    """Refresh intervals in order, conditioning each on earlier refreshed rows.

    The returned seconds count host elapsed time. CUDA benchmarks must use
    synchronized timing around the complete call.
    """

    _validate_intervals(intervals, int(edited_ids_t.shape[1]))
    cache = base_cache
    started = time.perf_counter()
    for start, end in intervals:
        cache = repair_window(shim, cache, edited_ids_t, start, end)
    elapsed = time.perf_counter() - started
    return cache, {
        "executor": "sequential_interval_repair",
        "interval_count": len(intervals),
        "recomputed_tokens": sum(end - start for start, end in intervals),
        "seconds": elapsed,
        "conditioned_on_earlier_intervals": True,
    }


@torch.no_grad()
def oneshot_sparse_repair(
    shim: Any,
    base_cache: Any,
    edited_ids_t: torch.Tensor,
    intervals: list[tuple[int, int]],
) -> tuple[Any, dict[str, Any]]:
    """Single forward over all selected positions, absolute position ids.

    This reference control uses Hugging Face StaticCache indexed writeback.
    It requires a model supporting absolute cache positions and full causal
    attention. The returned seconds are host elapsed time, not CUDA latency.
    """

    _validate_intervals(intervals, int(edited_ids_t.shape[1]))
    positions = [p for start, end in intervals for p in range(start, end)]
    if not positions:
        return base_cache, {
            "executor": "oneshot_sparse_repair",
            "interval_count": 0,
            "recomputed_tokens": 0,
            "seconds": 0.0,
        }
    device = edited_ids_t.device
    model = shim.generator
    base = shim._base_transformer(model)
    seq_len = int(edited_ids_t.shape[1])
    position_tensor = torch.tensor(positions, dtype=torch.long, device=device)
    full_positions = torch.arange(seq_len, dtype=torch.long, device=device)

    started = time.perf_counter()
    # A StaticCache addressed by absolute `cache_position` is what keeps this
    # correct: rows are rewritten in place, and a token at absolute position p
    # still attends to 0..p rather than to the whole cache. Appending to a
    # DynamicCache instead would let recomputed tokens attend to their own
    # future, which is not a coherence trade-off but a bug.
    static_cache = StaticCache(config=model.config, max_cache_len=seq_len)
    for layer_index in range(len(base_cache)):
        key, value = base_cache[layer_index]
        # Under ``device_map="auto"`` each layer's KV lives on that layer's
        # device, so a single shared ``cache_position`` built on the input's
        # device makes ``index_copy_`` fail. The index is layer-local metadata,
        # not data, so materialising it per layer is exact.
        static_cache.update(
            key.clone(),
            value.clone(),
            layer_index,
            {"cache_position": full_positions.to(key.device)},
        )
    base(
        input_ids=edited_ids_t.index_select(1, position_tensor),
        attention_mask=torch.ones((1, seq_len), dtype=torch.long, device=device),
        position_ids=position_tensor.unsqueeze(0),
        past_key_values=static_cache,
        use_cache=True,
        cache_position=position_tensor,
        return_dict=True,
    )
    output = clone_cache(base_cache)
    for layer_index in range(len(output)):
        out_key, out_value = output[layer_index]
        static_key, static_value = static_cache[layer_index]
        for target, source in ((out_key, static_key), (out_value, static_value)):
            # Same sharding hazard as the cache_position above: the write-back
            # index must live on the tensor being written, which under
            # ``device_map="auto"`` is this layer's device, not the input's.
            index = position_tensor.to(target.device)
            target.index_copy_(-2, index, source.index_select(-2, index.to(source.device)))
    elapsed = time.perf_counter() - started
    if int(clone_cache(output).get_seq_length()) != seq_len:
        raise RuntimeError("one-shot recompute changed the logical cache length")
    return output, {
        "executor": "oneshot_sparse_repair",
        "interval_count": len(intervals),
        "recomputed_tokens": len(positions),
        "rows_written": len(positions),
        "seconds": elapsed,
        "absolute_position_ids_used": True,
    }
