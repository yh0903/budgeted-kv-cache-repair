"""Small public interface over the paper's allocation and repair operators.

Text is encoded with ``add_special_tokens=False`` throughout. If a chat template
or beginning-of-sequence marker is needed, include it in the canonical text
before prefilling. Deletion spans are character offsets in that same string.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from .cache import OperatorShim, build_halo_cache
from .execution import positions_to_intervals, sequential_repair
from .scoring import dense_query_relevance_scores, query_text_token_ids
from .semantic_units import semantic_unit_token_ranges, select_semantic_unit_support


@dataclass(frozen=True)
class Selection:
    """Selected absolute positions in the edited prompt, with half-open intervals."""

    positions: tuple[int, ...]
    intervals: tuple[tuple[int, int], ...]
    suffix_start: int
    suffix_tokens: int
    budget_tokens: int
    metadata: dict[str, Any]


@dataclass(frozen=True)
class RepairResult:
    """A repaired cache and the edited prompt it represents."""

    cache: Any
    edited_text: str
    input_ids: torch.Tensor
    selection: Selection
    metadata: dict[str, Any]


def _check_budget(budget_percent: float) -> float:
    if isinstance(budget_percent, bool):
        raise ValueError("budget_percent must be a number between 0 and 100")
    value = float(budget_percent)
    if not math.isfinite(value) or not 0.0 <= value <= 100.0:
        raise ValueError("budget_percent must be finite and between 0 and 100")
    return value


def _encode(model: Any, tokenizer: Any, text: str) -> torch.Tensor:
    ids = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"]
    if not isinstance(ids, torch.Tensor) or ids.ndim != 2 or ids.shape[0] != 1:
        raise ValueError("tokenizer must return input_ids with shape [1, tokens]")
    device = model.get_input_embeddings().weight.device
    return ids.to(device=device, dtype=torch.long)


def _select_encoded(
    model: Any,
    tokenizer: Any,
    text: str,
    query: str,
    ids: torch.Tensor,
    suffix_start: int,
    budget_percent: float,
) -> Selection:
    if isinstance(suffix_start, bool) or not isinstance(suffix_start, int):
        raise ValueError("suffix_start must be an integer token position")
    length = int(ids.shape[1])
    if not 0 <= suffix_start <= length:
        raise ValueError("suffix_start must lie inside the edited prompt")
    suffix_tokens = length - suffix_start
    budget = min(suffix_tokens, math.ceil(suffix_tokens * budget_percent / 100.0))
    metadata: dict[str, Any] = {"budget_percent": budget_percent}
    if budget == 0:
        positions: list[int] = []
        metadata["selection"] = "shift_only"
    elif budget == suffix_tokens:
        positions = list(range(suffix_start, length))
        metadata["selection"] = "full_suffix"
    else:
        units = semantic_unit_token_ranges(
            tokenizer=tokenizer,
            text=text,
            expected_token_ids=ids[0].tolist(),
            suffix_start=suffix_start,
        )
        scores, score_metadata = dense_query_relevance_scores(
            model=OperatorShim(model, tokenizer),
            edited_input_ids=ids,
            query_token_ids=query_text_token_ids(tokenizer, query),
        )
        positions, allocator_metadata = select_semantic_unit_support(
            units=units, token_scores=scores, budget_tokens=budget
        )
        metadata.update({"selection": "sur", "scoring": score_metadata, "allocator": allocator_metadata})
    if len(positions) != budget or any(not suffix_start <= index < length for index in positions):
        raise RuntimeError("selected support violates the suffix budget")
    return Selection(
        positions=tuple(positions),
        intervals=tuple(positions_to_intervals(positions)),
        suffix_start=suffix_start,
        suffix_tokens=suffix_tokens,
        budget_tokens=budget,
        metadata=metadata,
    )


@torch.no_grad()
def select_support(
    model: Any,
    tokenizer: Any,
    edited_text: str,
    query: str,
    *,
    suffix_start: int,
    budget_percent: float = 5.0,
) -> Selection:
    """Allocate ``ceil(budget_percent / 100 * suffix_tokens)`` positions.

    ``suffix_start`` is a token index, not a character offset. The tokenizer must
    support exact offset mappings for the original string. No model forward or
    cache is needed: relevance uses only the model's frozen input embeddings.
    A 0% budget selects nothing; a 100% budget selects the entire suffix.
    """
    budget_percent = _check_budget(budget_percent)
    ids = _encode(model, tokenizer, edited_text)
    return _select_encoded(model, tokenizer, edited_text, query, ids, suffix_start, budget_percent)


def _check_model(model: Any) -> None:
    import transformers
    from transformers import LlamaForCausalLM, LlamaModel, Qwen3ForCausalLM, Qwen3Model

    if transformers.__version__ != "4.57.6":
        raise ValueError("repair requires transformers==4.57.6")
    base = getattr(model, "base_model", None)
    native_pairs = {(LlamaForCausalLM, LlamaModel), (Qwen3ForCausalLM, Qwen3Model)}
    if (type(model), type(base)) not in native_pairs:
        raise ValueError("repair_after_deletion supports native dense Llama and Qwen3 models")
    if any(module.training for module in model.modules()):
        raise ValueError("call model.eval() before repairing its cache")
    config = model.config
    if config.rope_scaling is not None or getattr(base.rotary_emb, "rope_type", None) != "default":
        raise ValueError("repair requires default fixed RoPE")
    if (
        getattr(config, "use_sliding_window", False)
        or getattr(base, "has_sliding_layers", False)
        or any(kind != "full_attention" for kind in getattr(config, "layer_types", []) or [])
    ):
        raise ValueError("repair requires full attention, not sliding or hybrid attention")
    device = model.get_input_embeddings().weight.device
    if device.type not in ("cpu", "cuda") or any(
        parameter.device != device for parameter in model.parameters()
    ):
        raise ValueError("repair requires one CPU test device or one CUDA device")
    if any(getattr(module, "_hf_hook", None) is not None for module in model.modules()):
        raise ValueError("repair does not support dispatch or offload hooks")
    if any("forward" in module.__dict__ for module in model.modules()):
        raise ValueError("repair does not support per-instance forward overrides")
    if getattr(model, "is_quantized", False):
        raise ValueError("repair does not support quantized models")


@torch.no_grad()
def repair_after_deletion(
    model: Any,
    tokenizer: Any,
    old_cache: Any,
    original_text: str,
    *,
    delete_span: tuple[int, int],
    query: str,
    budget_percent: float = 5.0,
    executor: str = "packed",
) -> RepairResult:
    """Delete a character span, correct positions, and refresh selected suffix KV.

    ``old_cache`` must come from an unpadded, batch-one prefill of exactly
    ``tokenizer(original_text, add_special_tokens=False)`` with the same model.
    The caller owns this correspondence; a cache does not store its source text.
    Input cache tensors are not modified.

    Tokenization may change at the edit boundary. Newly changed token positions
    are recomputed before suffix selection and are not charged to the retained
    suffix budget. Metadata reports this work separately. The query is used for
    selection only and is not appended to the returned cache.

    ``packed`` uses SDPA and Torch writes; ``sequential`` is the interval-wise
    reference. The result is an approximate mixed cache, not secure forgetting.
    """
    budget_percent = _check_budget(budget_percent)
    if executor not in ("packed", "sequential"):
        raise ValueError("executor must be 'packed' or 'sequential'")
    if (
        not isinstance(delete_span, (tuple, list))
        or len(delete_span) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in delete_span)
    ):
        raise ValueError("delete_span must contain two integer character offsets")
    start, end = delete_span
    if not 0 <= start < end <= len(original_text):
        raise ValueError("delete_span must be a nonempty half-open span in original_text")
    _check_model(model)
    old_ids = _encode(model, tokenizer, original_text)
    old_length = int(old_ids.shape[1])
    if not old_length or len(old_cache) != model.config.num_hidden_layers:
        raise ValueError("old_cache must contain every model layer for a nonempty prompt")
    for key, value in old_cache:
        if key.ndim != 4 or key.shape != value.shape or key.shape[0] != 1 or key.shape[2] != old_length:
            raise ValueError("old_cache shape does not match the batch-one original prompt")
        if key.device != old_ids.device or value.device != old_ids.device:
            raise ValueError("cache and model must be on the same device")
    edited_text = original_text[:start] + original_text[end:]
    edited_ids = _encode(model, tokenizer, edited_text)
    shim = OperatorShim(model, tokenizer)
    shifted_cache, alignment = build_halo_cache(shim, old_cache, old_ids, edited_ids, halo_tokens=0)
    selection = _select_encoded(
        model,
        tokenizer,
        edited_text,
        query,
        edited_ids,
        int(alignment["new_suffix_start"]),
        budget_percent,
    )
    intervals = list(selection.intervals)
    if executor == "packed":
        from .kernels.packed_cache import packed_interval_repair

        cache, execution = packed_interval_repair(shim, shifted_cache, edited_ids, intervals)
    else:
        cache, execution = sequential_repair(shim, shifted_cache, edited_ids, intervals)
    return RepairResult(
        cache=cache,
        edited_text=edited_text,
        input_ids=edited_ids,
        selection=selection,
        metadata={"alignment": alignment, "execution": execution, "executor": executor},
    )
