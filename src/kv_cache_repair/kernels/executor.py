"""Experimental sparse KV refresh with the reference interval-forward schedule.

The reference model forward runs once per maximal interval, in causal order.
Cache ownership, prefix materialization, and writeback are explicit choices.
The base Shift cache is borrowed and is never mutated. This module does not
implement SUR selection or RoPE correction.
"""

from __future__ import annotations

import hashlib
import json
from numbers import Integral
from typing import Any, Sequence

import torch


def _validate_cache(cache: Any, logical_length: int) -> int:
    try:
        layer_count = len(cache)
    except TypeError as exc:
        raise ValueError("base_cache must expose indexed K/V layers") from exc
    if layer_count == 0:
        raise ValueError("base_cache must contain at least one K/V layer")
    expected_shape = None
    expected_dtype = None
    for layer in range(layer_count):
        try:
            key, value = cache[layer]
        except (TypeError, ValueError, IndexError) as exc:
            raise ValueError(f"cache layer {layer} must contain one K/V pair") from exc
        if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
            raise ValueError(f"cache layer {layer} K/V must be tensors")
        if key.ndim != 4 or value.ndim != 4:
            raise ValueError(f"cache layer {layer} K/V must have rank four")
        if key.shape != value.shape or key.dtype != value.dtype or key.device != value.device:
            raise ValueError(f"cache layer {layer} K/V geometry, dtype, and device must agree")
        if key.shape[0] != 1 or key.shape[2] != logical_length:
            raise ValueError(f"cache layer {layer} must match the single-batch edited token length")
        if key.shape[1] <= 0 or key.shape[3] <= 0 or not key.is_floating_point():
            raise ValueError(f"cache layer {layer} requires nonempty floating-point KV heads")
        if expected_shape is None:
            expected_shape, expected_dtype = key.shape, key.dtype
        elif key.shape != expected_shape or key.dtype != expected_dtype:
            raise ValueError("cache layers must have regular shapes and matching dtypes")
    return layer_count


def _validate_intervals(
    intervals: Sequence[tuple[int, int]], logical_length: int
) -> tuple[tuple[int, int], ...]:
    if not isinstance(intervals, (list, tuple)):
        raise ValueError("intervals must be a list or tuple of maximal half-open ranges")
    result = []
    previous_end = -1
    for interval in intervals:
        if not isinstance(interval, (tuple, list)) or len(interval) != 2:
            raise ValueError("each interval must contain exactly a start and end")
        start, end = interval
        if any(isinstance(value, bool) or not isinstance(value, Integral) for value in interval):
            raise ValueError("interval bounds must be integers, not booleans")
        start, end = int(start), int(end)
        if not 0 <= start < end <= logical_length:
            raise ValueError("interval bounds must satisfy 0 <= start < end <= token length")
        if start <= previous_end:
            raise ValueError("intervals must be sorted, disjoint, and maximal (no adjacent ranges)")
        result.append((start, end))
        previous_end = end
    return tuple(result)


def _require_no_clone_support(cache: Any) -> None:
    """Fail closed outside the inspected DynamicLayer materialization contract."""
    import transformers
    from transformers.cache_utils import DynamicCache, DynamicLayer

    if transformers.__version__ != "4.57.6":
        raise ValueError("prefix_backend='no_clone' requires transformers==4.57.6")
    if type(cache) is not DynamicCache or getattr(cache, "offloading", True):
        raise ValueError("prefix_backend='no_clone' requires a standard non-offloaded DynamicCache")
    if any(type(layer) is not DynamicLayer for layer in cache.layers):
        raise ValueError("prefix_backend='no_clone' requires standard full-attention DynamicLayer entries")


def _materialize_prefix_without_clone(cache: Any, length: int) -> Any:
    """Feed views to public update; its initial cat still copies every prefix.

    In transformers 4.57.6, DynamicLayer.lazy_initialization starts with empty
    tensors and DynamicLayer.update always assigns torch.cat([old, incoming]).
    The new cache therefore owns its storage before the decoder is called.
    Removing the preceding tensor.clone() removes one redundant copy, not all
    prefix copying. No borrowed prefix storage reaches the model forward.
    """
    from transformers.cache_utils import DynamicCache

    prefix = DynamicCache()
    for layer in range(len(cache)):
        key, value = cache[layer]
        prefix.update(key[:, :, :length, :], value[:, :, :length, :], layer)
    return prefix


@torch.no_grad()
def interval_scatter_repair(
    shim: Any,
    base_cache: Any,
    edited_ids_t: torch.Tensor,
    intervals: Sequence[tuple[int, int]],
    *,
    writeback_backend: str = "torch",
    prefix_backend: str = "reference",
) -> tuple[Any, dict[str, Any]]:
    """Refresh selected intervals while preserving the reference arithmetic.

    Nonempty support returns independently owned K/V. Empty support returns the
    borrowed ``base_cache`` itself without importing the reference or launching
    a forward. No production dispatch is changed by this experimental API.
    ``triton`` is explicit and has no silent fallback.

    ``prefix_backend='reference'`` retains the original clone helper.
    ``'no_clone'`` omits its redundant explicit clone before DynamicCache.update
    materializes the prefix. This option is pinned to transformers 4.57.6's
    ordinary non-offloaded full DynamicCache and keeps prefix storage isolated.
    """
    if writeback_backend not in ("torch", "triton"):
        raise ValueError("writeback_backend must be 'torch' or 'triton'")
    if prefix_backend not in ("reference", "no_clone"):
        raise ValueError("prefix_backend must be 'reference' or 'no_clone'")
    if not isinstance(edited_ids_t, torch.Tensor) or edited_ids_t.ndim != 2 or edited_ids_t.shape[0] != 1:
        raise ValueError("edited_ids_t must be a single-batch rank-two tensor")
    if edited_ids_t.dtype not in (torch.int32, torch.int64):
        raise ValueError("edited_ids_t must contain int32 or int64 token IDs")
    logical_length = int(edited_ids_t.shape[1])
    layer_count = _validate_cache(base_cache, logical_length)
    validated = _validate_intervals(intervals, logical_length)
    positions = [position for start, end in validated for position in range(start, end)]
    diagnostics = {
        "executor": "interval_scatter_repair",
        "schedule": "reference_sequential_maximal_intervals",
        "writeback_backend": writeback_backend,
        "prefix_backend": prefix_backend,
        "prefix_initialization": (
            "reference_clone_then_dynamic_update" if prefix_backend == "reference"
            else "dynamic_update_materialization_without_explicit_clone"
        ),
        "interval_count": len(validated),
        "recomputed_tokens": len(positions),
        "rows_written_per_layer": len(positions),
        "layer_count": layer_count,
        "logical_length": logical_length,
        "selection_positions_sha256": hashlib.sha256(
            json.dumps(positions, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "conditioned_on_earlier_intervals": True,
        "base_cache_ownership": "borrowed_immutable",
        "output_cache_ownership": "independent" if validated else "borrowed_base",
    }
    if not validated:
        return base_cache, diagnostics
    if getattr(shim.generator, "training", True):
        raise ValueError("interval_scatter_repair requires an eval-mode inference model")
    if prefix_backend == "no_clone":
        _require_no_clone_support(base_cache)

    from kv_cache_repair.cache import clone_cache, run_tokens_from_prefix

    if writeback_backend == "triton":
        from .writeback import copy_kv_interval_

    output = clone_cache(base_cache)
    for start, end in validated:
        prefix = (
            clone_cache(output, start) if prefix_backend == "reference"
            else _materialize_prefix_without_clone(output, start)
        )
        recomputed = run_tokens_from_prefix(
            shim, prefix, edited_ids_t[:, start:end], end
        )
        if len(recomputed) != layer_count:
            raise RuntimeError("reference forward changed the K/V layer count")
        for layer in range(layer_count):
            out_key, out_value = output[layer]
            new_key, new_value = recomputed[layer]
            for target, source in ((out_key, new_key), (out_value, new_value)):
                expected_shape = (*target.shape[:2], end, target.shape[3])
                if source.shape != expected_shape or source.dtype != target.dtype or source.device != target.device:
                    raise RuntimeError("reference forward changed the K/V geometry, dtype, or device")
            if writeback_backend == "torch":
                out_key[:, :, start:end, :].copy_(new_key[:, :, start:end, :])
                out_value[:, :, start:end, :].copy_(new_value[:, :, start:end, :])
            else:
                copy_kv_interval_(out_key, out_value, new_key, new_value, start, end)
        # Do not hold the previous interval's all-layer prefix during the next
        # forward: that would add an avoidable live cache to the memory peak.
        del prefix, recomputed, new_key, new_value, source, target
    return output, diagnostics
