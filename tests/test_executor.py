"""Real tiny-model certification of the unchanged interval-forward schedule."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    Qwen3Config,
    Qwen3ForCausalLM,
    Qwen3MoeConfig,
    Qwen3MoeForCausalLM,
)
from transformers.cache_utils import DynamicCache

from kv_cache_repair.execution import sequential_repair
from kv_cache_repair.cache import OperatorShim
from kv_cache_repair.cache import clone_cache
from kv_cache_repair.kernels.executor import interval_scatter_repair


@pytest.fixture(scope="module", autouse=True)
def _small_cpu_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _model(family, dtype, attention):
    torch.manual_seed(41)
    options = dict(
        vocab_size=97,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        attention_dropout=0.0,
        use_cache=True,
    )
    if family == "llama":
        model = LlamaForCausalLM(LlamaConfig(**options))
    elif family == "qwen3":
        model = Qwen3ForCausalLM(Qwen3Config(**options, layer_types=["full_attention"] * 2))
    else:
        model = Qwen3MoeForCausalLM(Qwen3MoeConfig(
            **options, num_experts=4, num_experts_per_tok=2,
            moe_intermediate_size=32, norm_topk_prob=True,
        ))
    model = model.to(dtype=dtype).eval()
    model.set_attn_implementation(attention)
    return model


def _equal_bits(left, right):
    assert left.shape == right.shape
    assert left.dtype == right.dtype
    assert torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8))


def _cache_bits(left, right):
    assert len(left) == len(right)
    for layer in range(len(left)):
        for component in (0, 1):
            _equal_bits(left[layer][component], right[layer][component])


@pytest.mark.parametrize("family", ["llama", "qwen3", "qwen3_moe"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("attention", ["eager", "sdpa"])
@pytest.mark.parametrize("prefix_backend", ["reference", "no_clone"])
@pytest.mark.parametrize("intervals", [[], [(4, 5)], [(3, 5), (8, 10)], [(0, 1), (3, 4), (7, 8), (11, 12)], [(2, 12)], [(0, 12)]])
def test_tiny_models_bitwise_reference(family, dtype, attention, prefix_backend, intervals):
    model = _model(family, dtype, attention)
    shim = OperatorShim(model, SimpleNamespace(eos_token_id=None))
    old_ids = torch.arange(4, 16, dtype=torch.long).unsqueeze(0)
    edited_ids = old_ids.clone()
    edited_ids[:, 2:] = torch.arange(31, 41, dtype=torch.long)
    with torch.no_grad():
        base = model(input_ids=old_ids, use_cache=True).past_key_values
        snapshot = clone_cache(base)
        expected, _ = sequential_repair(shim, base, edited_ids, intervals)
    actual, receipt = interval_scatter_repair(shim, base, edited_ids, intervals, prefix_backend=prefix_backend)
    _cache_bits(actual, expected)
    _cache_bits(base, snapshot)
    selected = {position for start, end in intervals for position in range(start, end)}
    untouched = [position for position in range(old_ids.shape[1]) if position not in selected]
    for layer in range(len(base)):
        for component in (0, 1):
            _equal_bits(actual[layer][component][:, :, untouched], base[layer][component][:, :, untouched])
            assert not actual[layer][component].requires_grad
            if intervals:
                assert actual[layer][component].data_ptr() != base[layer][component].data_ptr()
    assert receipt["recomputed_tokens"] == len(selected)
    assert receipt["interval_count"] == len(intervals)
    assert receipt["logical_length"] == old_ids.shape[1]
    assert receipt["conditioned_on_earlier_intervals"] is True
    assert receipt["prefix_backend"] == prefix_backend
    if not intervals:
        assert actual is base
    # The cache remains usable for the ordinary next-token forward.
    query = torch.tensor([[43, 44]])
    with torch.no_grad():
        expected_logits = model(input_ids=query, past_key_values=clone_cache(expected)).logits
        actual_logits = model(input_ids=query, past_key_values=clone_cache(actual)).logits
    _equal_bits(actual_logits, expected_logits)


def _fake_state(length=8):
    cache = DynamicCache()
    for layer in range(2):
        cache.update(torch.ones(1, 2, length, 4), torch.ones(1, 2, length, 4), layer)
    return SimpleNamespace(generator=SimpleNamespace(training=False)), cache, torch.ones((1, length), dtype=torch.long)


@pytest.mark.parametrize("intervals", [
    [(True, 3)], [(1, False)], [(1.0, 3)], [(-1, 2)], [(3, 3)], [(4, 2)],
    [(1, 9)], [(3, 4), (1, 2)], [(1, 3), (2, 4)], [(1, 3), (3, 5)],
    [(1,)], [(1, 2, 3)], [None], "1,2",
])
def test_invalid_intervals_rejected_without_mutation(intervals):
    shim, base, ids = _fake_state()
    snapshot = clone_cache(base)
    with pytest.raises(ValueError):
        interval_scatter_repair(shim, base, ids, intervals)
    _cache_bits(base, snapshot)


@pytest.mark.parametrize("fault", ["batch", "float_ids", "length", "kv_shape", "rank", "layer_shape", "kv_dtype", "empty_cache", "backend", "prefix_backend", "training"])
def test_invalid_inputs_rejected(fault):
    shim, base, ids = _fake_state()
    backend = "torch"
    prefix_backend = "reference"
    if fault == "batch":
        ids = ids.expand(2, -1)
    elif fault == "float_ids":
        ids = ids.float()
    elif fault == "length":
        ids = ids[:, :-1]
    elif fault == "kv_shape":
        base = [(base[0][0], base[0][1][:, :, :-1])]
    elif fault == "rank":
        base = [(base[0][0][0], base[0][1][0])]
    elif fault == "layer_shape":
        base = [base[0], (torch.ones(1, 1, 8, 4), torch.ones(1, 1, 8, 4))]
    elif fault == "kv_dtype":
        base = [(base[0][0], base[0][1].double())]
    elif fault == "empty_cache":
        base = DynamicCache()
    elif fault == "backend":
        backend = "unknown"
    elif fault == "prefix_backend":
        prefix_backend = "unknown"
    else:
        shim.generator.training = True
    with pytest.raises(ValueError):
        interval_scatter_repair(shim, base, ids, [(1, 3)], writeback_backend=backend, prefix_backend=prefix_backend)


@pytest.mark.parametrize("prefix_backend", ["reference", "no_clone"])
def test_zero_support_never_calls_reference_or_loads_triton(monkeypatch, prefix_backend):
    import builtins
    import kv_cache_repair.cache as panel

    shim, base, ids = _fake_state()
    def forbidden(*args, **kwargs):
        raise AssertionError("zero support must not call reference helpers")
    monkeypatch.setattr(panel, "clone_cache", forbidden)
    monkeypatch.setattr(panel, "run_tokens_from_prefix", forbidden)
    original_import = builtins.__import__
    def checked_import(name, *args, **kwargs):
        if name == "writeback" or "triton" in name:
            raise AssertionError("zero support must not import a writeback kernel")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", checked_import)
    actual, receipt = interval_scatter_repair(shim, base, ids, [], writeback_backend="triton", prefix_backend=prefix_backend)
    assert actual is base
    assert receipt["recomputed_tokens"] == 0


@pytest.mark.parametrize("prefix_backend", ["reference", "no_clone"])
def test_failed_forward_does_not_mutate_borrowed_base(monkeypatch, prefix_backend):
    import kv_cache_repair.cache as panel

    shim, base, ids = _fake_state()
    snapshot = clone_cache(base)
    def fail_after_prefix_mutation(shim, prefix, tokens, end):
        prefix[0][0].fill_(999)
        raise RuntimeError("injected forward failure")
    monkeypatch.setattr(panel, "run_tokens_from_prefix", fail_after_prefix_mutation)
    with pytest.raises(RuntimeError, match="injected forward failure"):
        interval_scatter_repair(shim, base, ids, [(2, 4)], prefix_backend=prefix_backend)
    _cache_bits(base, snapshot)


@pytest.mark.parametrize("fault", ["version", "cache_subclass", "layer_subclass", "offloading"])
def test_no_clone_rejects_unqualified_cache_contracts(monkeypatch, fault):
    import transformers
    from transformers.cache_utils import DynamicLayer

    shim, base, ids = _fake_state()
    if fault == "version":
        monkeypatch.setattr(transformers, "__version__", "0.0.0-unsupported")
    elif fault == "cache_subclass":
        class AlternateCache(DynamicCache):
            pass
        base.__class__ = AlternateCache
    elif fault == "layer_subclass":
        class AlternateLayer(DynamicLayer):
            pass
        base.layers[0].__class__ = AlternateLayer
    else:
        base.offloading = True
    with pytest.raises(ValueError, match="no_clone"):
        interval_scatter_repair(shim, base, ids, [(2, 4)], prefix_backend="no_clone")


@pytest.mark.parametrize("length", [0, 3, 8])
def test_no_clone_prefix_is_materialized_and_cannot_modify_input(length):
    from kv_cache_repair.kernels.executor import _materialize_prefix_without_clone

    _, base, _ = _fake_state()
    snapshot = clone_cache(base)
    prefix = _materialize_prefix_without_clone(base, length)
    expected = clone_cache(base, length)
    _cache_bits(prefix, expected)
    for layer in range(len(prefix)):
        for component in range(2):
            if length:
                assert prefix[layer][component].untyped_storage().data_ptr() != base[layer][component].untyped_storage().data_ptr()
            prefix[layer][component].fill_(999)
    _cache_bits(base, snapshot)
