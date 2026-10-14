"""Experimental interval repair that writes native fresh K/V directly in place.

Pinned to the inspected Transformers 4.57.6 full-attention cache API. Native
model forwards retain the reference interval schedule. Prefix views can change
attention strides, so this candidate requires separate numerical qualification.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Sequence

import torch
import transformers
from transformers.cache_utils import Cache, DynamicLayer

from .executor import _validate_cache, _validate_intervals


class _DirectIntervalLayer(DynamicLayer):
    """Own no storage: update the matching independently owned output layer."""

    def __init__(self, key, value, writeback_backend, attention_layout):
        super().__init__()
        self.keys, self.values = key, value
        self.dtype, self.device = key.dtype, key.device
        self.is_initialized = True
        self._backend = writeback_backend
        self._attention_layout = attention_layout
        self._interval = None
        self._updated = False
        self._scatter_fresh = None
        if writeback_backend == "triton":
            # clone_cache creates independent contiguous output storage. Qualify
            # it once for the private launch; no target geometry changes during
            # this adapter's lifetime. The public copy API stays fully checked.
            if key.ndim != 4 or key.shape != value.shape:
                raise ValueError("direct Triton outputs require matching rank-four K/V")
            if not key.is_cuda or value.device != key.device:
                raise ValueError("direct Triton outputs must share one CUDA device")
            if key.dtype not in (torch.float16, torch.bfloat16, torch.float32) or value.dtype != key.dtype:
                raise ValueError("direct Triton outputs require matching float16, bfloat16, or float32 dtypes")
            if not key.is_contiguous() or not value.is_contiguous():
                raise ValueError("direct Triton outputs must be contiguous")
            if key.untyped_storage().data_ptr() == value.untyped_storage().data_ptr():
                raise ValueError("direct Triton outputs require independent K/V storage")
            from ._compiled_scatter import scatter_fresh_kv

            self._scatter_fresh = scatter_fresh_kv

    def begin(self, start, end):
        if self._interval is not None:
            raise RuntimeError("previous interval transaction is still active")
        self._interval = (start, end)
        self._updated = False

    def finish(self):
        if self._interval is None or not self._updated:
            raise RuntimeError("native forward did not update every cache layer")
        self._interval = None

    def get_seq_length(self):
        if self._interval is None:
            return self.keys.shape[-2]
        return self._interval[1] if self._updated else self._interval[0]

    def get_mask_sizes(self, cache_position):
        if self._interval is None:
            raise RuntimeError("mask requested outside an interval transaction")
        start, end = self._interval
        if cache_position.ndim != 1 or cache_position.shape[0] != end - start:
            raise RuntimeError("native forward changed the interval position count")
        return end, 0

    def update(self, key_states, value_states, cache_kwargs=None):
        if self._interval is None or self._updated:
            raise RuntimeError("expected exactly one update per layer and interval")
        start, end = self._interval
        expected = (*self.keys.shape[:2], end - start, self.keys.shape[-1])
        for source, target in ((key_states, self.keys), (value_states, self.values)):
            if source.shape != expected or source.dtype != target.dtype or source.device != target.device:
                raise RuntimeError("native cache update changed K/V shape, dtype, or device")
        cache_position = (cache_kwargs or {}).get("cache_position")
        if cache_position is None or cache_position.ndim != 1 or cache_position.shape[0] != end - start:
            raise RuntimeError("native cache update omitted interval cache positions")
        # The inspected native model constructs arange(start, end) from this
        # layer's get_seq_length(). Avoid copying positions to the host here.
        if self._backend == "torch":
            self.keys[:, :, start:end, :].copy_(key_states)
            self.values[:, :, start:end, :].copy_(value_states)
        else:
            # Native projections are fresh allocations, independent of owned
            # output. Their geometry/device/dtype was checked above. The kernel
            # maps source row zero to absolute output start directly, avoiding
            # destination views and repeated public-wrapper qualification.
            with torch.cuda.device(self.device):
                self._scatter_fresh(
                    self.keys, self.values, key_states, value_states, start, end
                )
        self._updated = True
        key_prefix = self.keys[:, :, :end, :]
        value_prefix = self.values[:, :, :end, :]
        if self._attention_layout == "contiguous":
            key_prefix = key_prefix.contiguous()
            value_prefix = value_prefix.contiguous()
        return key_prefix, value_prefix


def _qualify_native_model(shim, layer_count):
    if transformers.__version__ != "4.57.6":
        raise ValueError("direct cache prototype requires transformers==4.57.6")
    generator = shim.generator
    if getattr(generator, "training", True):
        raise ValueError("direct interval repair requires an eval-mode inference model")
    config = generator.config
    if config.model_type not in ("llama", "qwen3", "qwen3_moe"):
        raise ValueError("direct cache supports inspected Llama/Qwen3/Qwen3-MoE models only")
    from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaModel
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM, Qwen3Model
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM, Qwen3MoeModel

    causal_type, base_type = {
        "llama": (LlamaForCausalLM, LlamaModel),
        "qwen3": (Qwen3ForCausalLM, Qwen3Model),
        "qwen3_moe": (Qwen3MoeForCausalLM, Qwen3MoeModel),
    }[config.model_type]
    # Config identity alone cannot establish how cache_position is constructed.
    # Keep this prototype scoped to the exact inspected native classes, excluding
    # wrappers/subclasses which can override model forward or position handling.
    if type(generator) is not causal_type or type(shim._base_transformer(generator)) is not base_type:
        raise ValueError("direct cache requires the exact inspected native model and base classes")
    if config._attn_implementation not in ("eager", "sdpa"):
        raise ValueError("direct cache supports eager or SDPA attention only")
    if config.num_hidden_layers != layer_count:
        raise ValueError("model and cache layer counts must match")
    if any(kind != "full_attention" for kind in getattr(config, "layer_types", []) or []):
        raise ValueError("direct cache does not support sliding/hybrid attention")
    if getattr(config, "use_sliding_window", False):
        raise ValueError("direct cache does not support sliding-window attention")
    if config.model_type == "qwen3_moe" and getattr(config, "sliding_window", None) is not None:
        # Qwen3-MoE selects its mask from sliding_window directly; unlike dense
        # Qwen3, it does not consult use_sliding_window or per-layer types.
        raise ValueError("direct cache does not support Qwen3-MoE sliding-window attention")


@torch.no_grad()
def direct_interval_repair(
    shim: Any,
    base_cache: Any,
    edited_ids_t: torch.Tensor,
    intervals: Sequence[tuple[int, int]],
    *,
    writeback_backend: str = "torch",
    attention_layout: str = "view",
) -> tuple[Any, dict[str, Any]]:
    """Keep interval forwards; remove prefix clone/cat and final interval copy.

    Returns an independently owned ordinary DynamicCache for nonempty support;
    zero support returns the borrowed base cache. A failed transaction never
    exposes the partially updated owned output. This API is experimental and
    does not claim that attention-stride changes are numerically harmless. Only
    the exact inspected native model/base classes are accepted; custom forward
    monkeypatches or replaced attention modules are outside this API contract.
    """
    if writeback_backend not in ("torch", "triton"):
        raise ValueError("writeback_backend must be 'torch' or 'triton'")
    if attention_layout not in ("view", "contiguous"):
        raise ValueError("attention_layout must be 'view' or 'contiguous'")
    if not isinstance(edited_ids_t, torch.Tensor) or edited_ids_t.ndim != 2 or edited_ids_t.shape[0] != 1:
        raise ValueError("edited_ids_t must be a single-batch rank-two tensor")
    if edited_ids_t.dtype not in (torch.int32, torch.int64):
        raise ValueError("edited_ids_t must contain int32 or int64 token IDs")
    logical_length = int(edited_ids_t.shape[1])
    layer_count = _validate_cache(base_cache, logical_length)
    validated = _validate_intervals(intervals, logical_length)
    positions = [p for start, end in validated for p in range(start, end)]
    diagnostics = {
        "executor": "direct_interval_repair",
        "schedule": "reference_sequential_maximal_intervals",
        "writeback_backend": writeback_backend,
        "writeback_validation": (
            "owned_output_once_and_native_projection_per_update" if writeback_backend == "triton"
            else "native_projection_per_update"
        ),
        "writeback_launch": (
            "bounded_compiled_runner_with_jit_fallback" if writeback_backend == "triton"
            else "torch_slice_copy"
        ),
        "attention_layout": attention_layout,
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
        "attention_view_requires_numerical_qualification": attention_layout == "view",
    }
    if not validated:
        return base_cache, diagnostics
    _qualify_native_model(shim, layer_count)

    from kv_cache_repair.cache import clone_cache, run_tokens_from_prefix

    output = clone_cache(base_cache)
    layers = [
        _DirectIntervalLayer(*output[index], writeback_backend, attention_layout)
        for index in range(layer_count)
    ]
    interval_cache = Cache(layers=layers)
    for start, end in validated:
        for layer in layers:
            layer.begin(start, end)
        returned_cache = run_tokens_from_prefix(
            shim, interval_cache, edited_ids_t[:, start:end], end
        )
        if returned_cache is not interval_cache:
            raise RuntimeError("native forward replaced the direct cache adapter")
        for layer in layers:
            layer.finish()
    return output, diagnostics
