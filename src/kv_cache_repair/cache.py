"""Reference eager cache editing and interval refresh for full-attention models."""

from __future__ import annotations

from typing import Any

import torch
from transformers.cache_utils import DynamicCache

from .alignment import token_alignment


class OperatorShim:
    """Expose the small model surface used by the reference repair operators."""

    def __init__(self, model: Any, tokenizer: Any = None) -> None:
        self.generator = model
        self.tokenizer = tokenizer

    @staticmethod
    def _base_transformer(model: Any) -> Any:
        return getattr(model, "base_model", model)


def rotate_half(value: torch.Tensor) -> torch.Tensor:
    left = value[..., : value.shape[-1] // 2]
    right = value[..., value.shape[-1] // 2 :]
    return torch.cat((-right, left), dim=-1)


def clone_cache(cache: Any, length: int | None = None) -> DynamicCache:
    """Clone all layers, optionally retaining only a prefix; never alias storage."""
    if length is not None and (int(length) < 0 or (len(cache) and int(length) > cache[0][0].shape[2])):
        raise ValueError("prefix length is outside the cache")
    output = DynamicCache()
    for layer_index in range(len(cache)):
        key, value = cache[layer_index]
        end = int(key.shape[2]) if length is None else int(length)
        output.update(
            key[:, :, :end, :].clone(),
            value[:, :, :end, :].clone(),
            layer_index,
        )
    return output


def shifted_suffix_keys(
    model: Any,
    key: torch.Tensor,
    shift: int,
) -> torch.Tensor:
    """Apply the reference eager delta-RoPE rotation to retained suffix keys."""
    if not key.shape[2] or not shift:
        return key.clone()
    base = model._base_transformer(model.generator)
    positions = torch.full(
        (1, int(key.shape[2])),
        int(shift),
        dtype=torch.long,
        device=key.device,
    )
    cos, sin = base.rotary_emb(key, positions)
    # ``rotary_emb`` lives wherever the embedding module was placed, which under
    # ``device_map="auto"`` is cuda:0, while this layer's KV may sit on cuda:1.
    # RoPE factors depend only on position, so moving them to the key's device
    # is exact -- and without it the whole repair operator is single-device.
    cos = cos.unsqueeze(1).to(device=key.device, dtype=key.dtype)
    sin = sin.unsqueeze(1).to(device=key.device, dtype=key.dtype)

    rotary_dim = int(cos.shape[-1])
    head_dim = int(key.shape[-1])
    if rotary_dim >= head_dim:
        return key * cos + rotate_half(key) * sin
    rotated = key[..., :rotary_dim]
    passthrough = key[..., rotary_dim:]
    rotated = rotated * cos + rotate_half(rotated) * sin
    return torch.cat((rotated, passthrough), dim=-1)


@torch.no_grad()
def run_tokens_from_prefix(
    model: Any,
    prefix_cache: Any,
    input_ids: torch.Tensor,
    end_position: int,
) -> Any:
    if not int(input_ids.shape[1]):
        return prefix_cache
    attention_mask = torch.ones(
        (1, int(end_position)),
        dtype=torch.long,
        device=input_ids.device,
    )
    return model._base_transformer(model.generator)(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=prefix_cache,
        use_cache=True,
        return_dict=True,
    ).past_key_values


@torch.no_grad()
def build_halo_cache(
    model: Any,
    old_cache: Any,
    old_input_ids: torch.Tensor,
    new_input_ids: torch.Tensor,
    halo_tokens: int,
) -> tuple[Any, dict[str, int]]:
    """Preserve the common prefix, refresh changed tokens/halo, then shift the tail.

    A zero halo gives the Shift reference. A halo spanning the retained suffix
    gives Replay. Inputs use batch size one and the old cache is not mutated.
    """
    old_ids = [int(value) for value in old_input_ids[0].tolist()]
    new_ids = [int(value) for value in new_input_ids[0].tolist()]
    alignment = token_alignment(old_ids, new_ids)
    halo = min(max(int(halo_tokens), 0), alignment["suffix"])
    recompute_end = alignment["new_suffix_start"] + halo
    prefix_cache = clone_cache(old_cache, alignment["prefix"])
    recomputed = run_tokens_from_prefix(
        model,
        prefix_cache,
        new_input_ids[:, alignment["prefix"] : recompute_end],
        recompute_end,
    )
    output = DynamicCache()
    old_tail_start = alignment["old_suffix_start"] + halo
    for layer_index in range(len(old_cache)):
        exact_key, exact_value = recomputed[layer_index]
        old_key, old_value = old_cache[layer_index]
        old_key_tail = old_key[:, :, old_tail_start:, :]
        old_value_tail = old_value[:, :, old_tail_start:, :]
        tail_key = shifted_suffix_keys(model, old_key_tail, alignment["position_shift"])
        tail_value = old_value_tail.clone()
        output_key = torch.cat((exact_key[:, :, :recompute_end, :], tail_key), dim=2)
        output_value = torch.cat((exact_value[:, :, :recompute_end, :], tail_value), dim=2)
        output.update(output_key, output_value, layer_index)
    return output, {
        **alignment,
        "halo_tokens": halo,
        "recomputed_tokens": alignment["mandatory_changed_tokens"] + halo,
    }


@torch.no_grad()
def repair_window(
    model: Any,
    base_cache: Any,
    new_input_ids: torch.Tensor,
    start: int,
    end: int,
) -> Any:
    """Refresh one half-open interval and preserve all other cache entries."""
    start = int(start)
    end = int(end)
    if start < 0 or end < start or end > int(new_input_ids.shape[1]):
        raise ValueError("repair interval is outside the edited sequence")
    if end <= start:
        return base_cache
    prefix_cache = clone_cache(base_cache, start)
    recomputed = run_tokens_from_prefix(
        model,
        prefix_cache,
        new_input_ids[:, start:end],
        end,
    )
    output = DynamicCache()
    for layer_index in range(len(base_cache)):
        old_key, old_value = base_cache[layer_index]
        new_key, new_value = recomputed[layer_index]
        output.update(
            torch.cat(
                (
                    old_key[:, :, :start, :],
                    new_key[:, :, start:end, :],
                    old_key[:, :, end:, :],
                ),
                dim=2,
            ),
            torch.cat(
                (
                    old_value[:, :, :start, :],
                    new_value[:, :, start:end, :],
                    old_value[:, :, end:, :],
                ),
                dim=2,
            ),
            layer_index,
        )
    return output


def merge_intervals(
    intervals: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    output = []
    for start, end in sorted((int(start), int(end)) for start, end in intervals if int(end) > int(start)):
        if output and start <= output[-1][1]:
            output[-1] = (output[-1][0], max(output[-1][1], end))
        else:
            output.append((start, end))
    return output
