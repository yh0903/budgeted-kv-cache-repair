"""Packed selected-row refresh through isolated native decoder module shells.

The same selected rows pass through every layer. Cache updates scatter fresh
K/V before attention reads the full mixed history. This changes GEMM/attention
shapes versus interval replay and requires its own numerical qualification.
SDPA with Torch writeback is the default. The explicit Triton attention backend
is experimental: local operator checks do not establish complete-model or
task-level equivalence, and a checkpoint-level numerical check remains unmet.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from functools import lru_cache
from threading import Lock
from typing import Any, Sequence

import torch
import transformers
from transformers.cache_utils import Cache, DynamicLayer

from .executor import _validate_cache, _validate_intervals


_ATTENTION_NAME = "sur_kernel_packed_absolute_attention_v1"
_REGISTRATION_LOCK = Lock()


@dataclass
class _PackedContext:
    positions: torch.Tensor
    logical_length: int
    backend: str
    device_plan: Any = None
    mask: Any = None


@lru_cache(maxsize=1)
def _compiled_flex():
    from torch.nn.attention.flex_attention import flex_attention

    # Cache executable dispatch only, never a model, request, or tensor.
    return torch.compile(flex_attention, dynamic=False)


def _packed_attention(
    module, query, key, value, attention_mask,
    *, scaling=None, dropout=0.0, sur_packed_context=None, **kwargs,
):
    context = sur_packed_context
    if not isinstance(context, _PackedContext):
        raise RuntimeError("packed attention requires explicit per-call request context")
    if attention_mask is not None:
        raise RuntimeError("native dense mask construction must be bypassed for packed attention")
    if dropout:
        raise ValueError("packed attention supports inference without dropout only")
    if query.shape[-2] != context.positions.numel() or key.shape[-2] != context.logical_length:
        raise RuntimeError("packed attention query/cache lengths drifted")
    if context.backend == "sdpa":
        from transformers.integrations.sdpa_attention import sdpa_attention_forward

        # Explicit absolute-position mask; reuse the pinned native GQA path.
        return sdpa_attention_forward(
            module, query, key, value, context.mask,
            dropout=0.0, scaling=scaling, is_causal=False,
        )
    if context.backend == "triton":
        from .indexed_attention import indexed_causal_attention

        result = indexed_causal_attention(
            query, key, value, context.device_plan, scale=scaling
        )
    elif context.backend == "flex":
        result = _compiled_flex()(
            query, key, value, block_mask=context.mask,
            scale=scaling, enable_gqa=True,
        )
    else:
        raise RuntimeError("unknown packed attention backend")
    # Native attention modules consume [batch, query, query_head, head_dim].
    return result.transpose(1, 2).contiguous(), None


def _register_attention():
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    with _REGISTRATION_LOCK:
        if _ATTENTION_NAME in ALL_MASK_ATTENTION_FUNCTIONS:
            raise RuntimeError("packed backend name unexpectedly has a native mask interface")
        if _ATTENTION_NAME in ALL_ATTENTION_FUNCTIONS:
            if ALL_ATTENTION_FUNCTIONS[_ATTENTION_NAME] is not _packed_attention:
                raise RuntimeError("packed backend registry name is already owned by another implementation")
        else:
            ALL_ATTENTION_FUNCTIONS.register(_ATTENTION_NAME, _packed_attention)


def _make_native_view(base):
    """Share parameters without modifying the caller's modules/configuration."""
    _register_attention()
    config = copy.deepcopy(base.config)
    config._attn_implementation = _ATTENTION_NAME
    view = copy.copy(base)
    view._modules = base._modules.copy()
    view.config = config
    layers = []
    for original in base.layers:
        layer = copy.copy(original)
        layer._modules = original._modules.copy()
        if hasattr(layer, "config"):
            layer.config = config
        attention = copy.copy(original.self_attn)
        attention._modules = original.self_attn._modules.copy()
        attention.config = config
        layer._modules["self_attn"] = attention
        layers.append(layer)
    view._modules["layers"] = torch.nn.ModuleList(layers)
    view.layers.training = False
    return view


def _qualify(shim, edited_ids, base_cache, attention_backend, writeback_backend):
    if transformers.__version__ != "4.57.6":
        raise ValueError("packed prototype requires transformers==4.57.6")
    from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaModel
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM, Qwen3Model

    model = shim.generator
    base = shim._base_transformer(model)
    pairs = {(LlamaForCausalLM, LlamaModel), (Qwen3ForCausalLM, Qwen3Model)}
    if (type(model), type(base)) not in pairs:
        raise ValueError("packed prototype requires exact native dense Llama or Qwen3 classes")
    if model.training or base.training or any(module.training for module in base.modules()):
        raise ValueError("packed prototype requires eval mode on all decoder modules")
    config = model.config
    if config.num_hidden_layers != len(base_cache) or len(base.layers) != len(base_cache):
        raise ValueError("model and cache layer counts must agree")
    if config.rope_scaling is not None or getattr(base.rotary_emb, "rope_type", None) != "default":
        raise ValueError("packed prototype supports default fixed RoPE only")
    if getattr(config, "use_sliding_window", False) or getattr(base, "has_sliding_layers", False):
        raise ValueError("packed prototype requires full attention")
    if any(kind != "full_attention" for kind in getattr(config, "layer_types", []) or []):
        raise ValueError("packed prototype rejects sliding/hybrid attention")
    device = edited_ids.device
    if device.type not in ("cpu", "cuda"):
        raise ValueError("packed prototype supports one CPU test device or one CUDA device")
    if any(parameter.device != device for parameter in model.parameters()):
        raise ValueError("packed prototype requires all model parameters on the request device")
    if any(key.device != device or value.device != device for key, value in base_cache):
        raise ValueError("packed prototype requires all cache layers on the request device")
    if any(getattr(module, "_hf_hook", None) is not None for module in model.modules()):
        raise ValueError("packed prototype does not support dispatch/offload hooks")
    if any("forward" in module.__dict__ for module in base.modules()):
        raise ValueError("packed prototype does not support per-instance decoder forward overrides")
    if attention_backend in ("triton", "flex") or writeback_backend == "triton":
        if device.type != "cuda":
            raise ValueError("Triton/Flex packed paths require CUDA")
    if attention_backend == "triton" or writeback_backend == "triton":
        key = base_cache[0][0]
        if key.dtype not in (torch.float16, torch.bfloat16) or key.shape[-1] not in (64, 128):
            raise ValueError("indexed Triton prototype requires fp16/bf16 and head dimension 64 or 128")
    return base


class _PackedLayer(DynamicLayer):
    def __init__(self, key, value, context, backend):
        super().__init__()
        self.keys, self.values = key, value
        self.dtype, self.device = key.dtype, key.device
        self.is_initialized = True
        self._context = context
        self._backend = backend
        self.updated = False

    def get_mask_sizes(self, cache_position):
        return self._context.logical_length, 0

    def update(self, key_states, value_states, cache_kwargs=None):
        if self.updated:
            raise RuntimeError("packed native forward updated one layer more than once")
        positions = self._context.positions
        expected = (*self.keys.shape[:2], positions.numel(), self.keys.shape[-1])
        for source, target in ((key_states, self.keys), (value_states, self.values)):
            if source.shape != expected or source.dtype != target.dtype or source.device != target.device:
                raise RuntimeError("packed native projection geometry/dtype/device drifted")
        if (cache_kwargs or {}).get("cache_position") is not positions:
            raise RuntimeError("native forward replaced the explicit packed cache positions")
        if self._backend == "torch":
            self.keys.index_copy_(2, positions, key_states)
            self.values.index_copy_(2, positions, value_states)
        else:
            from .indexed_attention import indexed_kv_scatter_

            indexed_kv_scatter_(
                self.keys, self.values, key_states, value_states,
                self._context.device_plan,
            )
        self.updated = True
        return self.keys, self.values


def _context(host_plan, positions, attention_backend, device_plan):
    context = _PackedContext(positions, host_plan.logical_length, attention_backend, device_plan)
    if attention_backend == "sdpa":
        key_positions = torch.arange(host_plan.logical_length, device=positions.device)
        context.mask = (key_positions[None, :] <= positions[:, None])[None, None]
    elif attention_backend == "flex":
        from torch.nn.attention.flex_attention import create_block_mask

        count = len(host_plan.positions)
        length = host_plan.logical_length
        def mask_mod(batch, head, query_index, key_index):
            # create_block_mask visits padded query tiles too. Clamp before the
            # indexed read, then explicitly exclude those padded query lanes.
            absolute_query = positions[query_index.clamp(max=count - 1)]
            return (query_index < count) & (key_index < length) & (key_index <= absolute_query)
        context.mask = create_block_mask(
            mask_mod, B=1, H=None, Q_LEN=count, KV_LEN=length,
            device=str(positions.device), BLOCK_SIZE=(128, 128), _compile=True,
        )
    return context


@torch.no_grad()
def packed_interval_repair(
    shim: Any, base_cache: Any, edited_ids_t: torch.Tensor,
    intervals: Sequence[tuple[int, int]], *,
    attention_backend: str = "sdpa", writeback_backend: str = "torch",
    plan: Any = None,
) -> tuple[Any, dict[str, Any]]:
    """Recompute sorted selected rows in one native layer-major forward.

    ``plan`` optionally supplies a reusable host IndexedPlan for precisely these
    intervals and this logical length. Device descriptors, attention masks,
    isolated native module shells, and the independent output clone are prepared
    inside this call. No request/model tensors are retained in a global cache.
    CPU support is a reference-test path limited to SDPA + torch writeback.
    Triton attention is an explicit experimental option, not a recommended
    replacement for the default SDPA backend.
    """
    if attention_backend not in ("sdpa", "triton", "flex"):
        raise ValueError("attention_backend must be sdpa, triton, or flex")
    if writeback_backend not in ("torch", "triton"):
        raise ValueError("writeback_backend must be torch or triton")
    if not isinstance(edited_ids_t, torch.Tensor) or edited_ids_t.ndim != 2 or edited_ids_t.shape[0] != 1:
        raise ValueError("edited_ids_t must be a single-batch rank-two tensor")
    if edited_ids_t.dtype not in (torch.int32, torch.int64):
        raise ValueError("edited_ids_t must contain int32 or int64 IDs")
    length = int(edited_ids_t.shape[1])
    layer_count = _validate_cache(base_cache, length)
    intervals = _validate_intervals(intervals, length)
    selected = tuple(p for start, end in intervals for p in range(start, end))
    from .indexed_attention import IndexedPlan

    if plan is not None:
        if not isinstance(plan, IndexedPlan) or tuple(plan.positions) != selected or plan.logical_length != length:
            raise ValueError("supplied IndexedPlan must match exact selected positions and cache length")
    receipt = {
        "executor": "packed_interval_repair",
        "schedule": "packed_selected_rows_layer_major",
        "attention_backend": attention_backend,
        "writeback_backend": writeback_backend,
        "interval_count": len(intervals),
        "recomputed_tokens": len(selected),
        "logical_length": length,
        "layer_count": layer_count,
        "base_cache_ownership": "borrowed_immutable",
        "output_cache_ownership": "independent" if selected else "borrowed_base",
        "selected_absolute_positions": list(selected),
        "native_model_mutated": False,
        "numerical_contract": "separate_schedule_qualification_required",
    }
    if not selected:
        return base_cache, receipt
    base = _qualify(shim, edited_ids_t, base_cache, attention_backend, writeback_backend)
    host_plan = plan if plan is not None else IndexedPlan(selected, length)
    device_plan = host_plan.to(edited_ids_t.device) if edited_ids_t.is_cuda else None
    positions = (
        device_plan.positions if device_plan is not None
        else torch.tensor(selected, dtype=torch.long, device=edited_ids_t.device)
    )
    context = _context(host_plan, positions, attention_backend, device_plan)
    native_view = _make_native_view(base)
    from kv_cache_repair.cache import clone_cache

    output = clone_cache(base_cache)
    layers = [_PackedLayer(*output[index], context, writeback_backend) for index in range(layer_count)]
    cache = Cache(layers=layers)
    result = native_view(
        input_ids=edited_ids_t.index_select(1, positions),
        position_ids=positions.unsqueeze(0), cache_position=positions,
        past_key_values=cache, use_cache=True, return_dict=True,
        output_attentions=False, sur_packed_context=context,
    )
    if result.past_key_values is not cache or not all(layer.updated for layer in layers):
        raise RuntimeError("native packed forward did not update each cache layer exactly once")
    return output, receipt
