"""Direct cache candidate: CPU semantics, layouts, and native-model outputs."""

from types import SimpleNamespace

import pytest
import torch
from transformers import (
    LlamaConfig, LlamaForCausalLM,
    Qwen3Config, Qwen3ForCausalLM,
    Qwen3MoeConfig, Qwen3MoeForCausalLM,
)
from transformers.cache_utils import DynamicCache

from kv_cache_repair.execution import sequential_repair
from kv_cache_repair.cache import OperatorShim
from kv_cache_repair.cache import clone_cache
from kv_cache_repair.kernels.direct_cache import _DirectIntervalLayer, direct_interval_repair


@pytest.fixture(scope="module", autouse=True)
def small_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make_model(family, dtype, attention, kv_heads):
    torch.manual_seed(71)
    options = dict(
        vocab_size=97, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=kv_heads,
        head_dim=8, max_position_embeddings=64, attention_dropout=0.0,
    )
    if family == "llama":
        model = LlamaForCausalLM(LlamaConfig(**options))
    elif family == "qwen3":
        model = Qwen3ForCausalLM(Qwen3Config(**options, layer_types=["full_attention"] * 2))
    else:
        model = Qwen3MoeForCausalLM(Qwen3MoeConfig(
            **options, num_experts=4, num_experts_per_tok=2, moe_intermediate_size=32,
        ))
    model.to(dtype=dtype).eval()
    model.set_attn_implementation(attention)
    return model


def assert_bits(left, right):
    assert left.shape == right.shape and left.dtype == right.dtype
    assert torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8))


def assert_cache_bits(left, right):
    assert len(left) == len(right)
    for layer in range(len(left)):
        for component in (0, 1):
            assert_bits(left[layer][component], right[layer][component])


@pytest.mark.parametrize("family", ["llama", "qwen3", "qwen3_moe"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("attention", ["eager", "sdpa"])
@pytest.mark.parametrize("kv_heads", [1, 2, 4])
@pytest.mark.parametrize("layout", ["view", "contiguous"])
@pytest.mark.parametrize("intervals", [[], [(0, 12)], [(3, 5), (8, 10)], [(0, 1), (3, 4), (8, 9), (11, 12)]])
def test_native_model_bitwise_schedule(family, dtype, attention, kv_heads, layout, intervals):
    model = make_model(family, dtype, attention, kv_heads)
    shim = OperatorShim(model, SimpleNamespace())
    original_ids = torch.arange(4, 16).unsqueeze(0)
    edited_ids = original_ids + 13
    with torch.no_grad():
        base = model(input_ids=original_ids, use_cache=True).past_key_values
        snapshot = clone_cache(base)
        expected, _ = sequential_repair(shim, base, edited_ids, intervals)
    actual, receipt = direct_interval_repair(
        shim, base, edited_ids, intervals, attention_layout=layout
    )
    assert isinstance(actual, DynamicCache)
    assert actual.get_seq_length() == 12
    assert_cache_bits(actual, expected)
    assert_cache_bits(base, snapshot)
    selected = {p for start, end in intervals for p in range(start, end)}
    untouched = [p for p in range(12) if p not in selected]
    for layer in range(len(base)):
        for component in (0, 1):
            assert_bits(actual[layer][component][:, :, untouched], base[layer][component][:, :, untouched])
            if intervals:
                assert actual[layer][component].data_ptr() != base[layer][component].data_ptr()
    assert receipt["recomputed_tokens"] == len(selected)
    if not intervals:
        assert actual is base
    query_ids = torch.tensor([[47, 48]])
    with torch.no_grad():
        actual_logits = model(input_ids=query_ids, past_key_values=clone_cache(actual)).logits
        expected_logits = model(input_ids=query_ids, past_key_values=clone_cache(expected)).logits
    assert_bits(actual_logits, expected_logits)


@pytest.mark.parametrize("layout", ["view", "contiguous"])
def test_layer_writes_before_attention_and_excludes_future(layout):
    key = torch.arange(2 * 12 * 4).reshape(1, 2, 12, 4).float()
    value = key + 1000
    original_key, original_value = key.clone(), value.clone()
    layer = _DirectIntervalLayer(key, value, "torch", layout)
    layer.begin(3, 5)
    assert layer.get_seq_length() == 3
    assert layer.get_mask_sizes(torch.arange(3, 5)) == (5, 0)
    new_key, new_value = torch.full((1, 2, 2, 4), -7.0), torch.full((1, 2, 2, 4), -9.0)
    read_key, read_value = layer.update(new_key, new_value, {"cache_position": torch.arange(3, 5)})
    assert read_key.shape[-2] == 5 and read_value.shape[-2] == 5
    assert_bits(read_key[:, :, 3:5], new_key)
    assert_bits(read_value[:, :, 3:5], new_value)
    assert_bits(key[:, :, 5:], original_key[:, :, 5:])
    assert_bits(value[:, :, 5:], original_value[:, :, 5:])
    if layout == "view":
        assert read_key.data_ptr() == key.data_ptr()
        assert read_key.stride(1) == 12 * 4
    else:
        assert read_key.is_contiguous()
        assert read_key.stride(1) == 5 * 4
    assert layer.get_seq_length() == 5
    with pytest.raises(RuntimeError, match="exactly one"):
        layer.update(new_key, new_value, {"cache_position": torch.arange(3, 5)})
    layer.finish()
    assert layer.get_seq_length() == 12
    # Later intervals see earlier fresh selected rows and intervening stale rows.
    layer.begin(8, 9)
    read_key, _ = layer.update(new_key[:, :, :1], new_value[:, :, :1], {"cache_position": torch.arange(8, 9)})
    assert_bits(read_key[:, :, 3:5], new_key)
    assert_bits(read_key[:, :, 5:8], original_key[:, :, 5:8])
    layer.finish()


@pytest.mark.parametrize("failure", ["forward_raises", "missing_layer", "replaced_cache"])
def test_failed_transaction_preserves_borrowed_base(monkeypatch, failure):
    import kv_cache_repair.cache as panel
    model = make_model("llama", torch.float32, "eager", 2)
    shim = OperatorShim(model, SimpleNamespace())
    ids = torch.arange(4, 16).unsqueeze(0)
    with torch.no_grad():
        base = model(input_ids=ids).past_key_values
    snapshot = clone_cache(base)
    def failing_forward(shim, cache, interval_ids, end):
        fresh = torch.full((1, 2, interval_ids.shape[-1], 8), 999.0)
        cache.update(fresh, fresh, 0, {"cache_position": torch.arange(end - interval_ids.shape[-1], end)})
        if failure == "forward_raises":
            raise RuntimeError("injected failure")
        return DynamicCache() if failure == "replaced_cache" else cache
    monkeypatch.setattr(panel, "run_tokens_from_prefix", failing_forward)
    with pytest.raises(RuntimeError):
        direct_interval_repair(shim, base, ids, [(3, 5)])
    assert_cache_bits(base, snapshot)


@pytest.mark.parametrize("fault", ["version", "family", "model_subclass", "base_class", "backend", "sliding", "layout", "training", "interval"])
def test_fail_closed_contract(monkeypatch, fault):
    import kv_cache_repair.kernels.direct_cache as module
    model = make_model("llama", torch.float32, "eager", 2)
    shim = OperatorShim(model, SimpleNamespace())
    ids = torch.arange(4, 16).unsqueeze(0)
    with torch.no_grad():
        base = model(input_ids=ids).past_key_values
    snapshot = clone_cache(base)
    kwargs, intervals = {}, [(3, 5)]
    if fault == "version":
        monkeypatch.setattr(module.transformers, "__version__", "unsupported")
    elif fault == "family":
        model.config.model_type = "uninspected"
    elif fault == "model_subclass":
        class UninspectedLlama(LlamaForCausalLM):
            pass
        model.__class__ = UninspectedLlama
    elif fault == "base_class":
        monkeypatch.setattr(shim, "_base_transformer", lambda generator: SimpleNamespace())
    elif fault == "backend":
        model.config._attn_implementation = "flash_attention_2"
    elif fault == "sliding":
        model.config.layer_types = ["full_attention", "sliding_attention"]
    elif fault == "layout":
        kwargs["attention_layout"] = "unknown"
    elif fault == "training":
        model.train()
    else:
        intervals = [(3, 5), (5, 6)]
    with pytest.raises(ValueError):
        direct_interval_repair(shim, base, ids, intervals, **kwargs)
    assert_cache_bits(base, snapshot)


def test_moe_sliding_mask_without_use_sliding_flag_is_rejected():
    model = make_model("qwen3_moe", torch.float32, "eager", 2)
    shim = OperatorShim(model, SimpleNamespace())
    ids = torch.arange(4, 16).unsqueeze(0)
    with torch.no_grad():
        base = model(input_ids=ids).past_key_values
    model.config.sliding_window = 4
    model.config.use_sliding_window = False
    with pytest.raises(ValueError, match="Qwen3-MoE sliding-window"):
        direct_interval_repair(shim, base, ids, [(3, 5)])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA qualification requires a GPU")
@pytest.mark.parametrize("family", ["llama", "qwen3", "qwen3_moe"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("attention", ["eager", "sdpa"])
@pytest.mark.parametrize("kv_heads", [1, 2, 4])
def test_cuda_direct_writeback_matches_reference(family, dtype, attention, kv_heads):
    pytest.importorskip("triton")
    model = make_model(family, dtype, attention, kv_heads).cuda()
    shim = OperatorShim(model, SimpleNamespace())
    ids = torch.arange(4, 16, device="cuda").unsqueeze(0)
    edited = ids + 13
    intervals = [(0, 1), (3, 5), (8, 9), (11, 12)]
    with torch.no_grad():
        base = model(input_ids=ids).past_key_values
        snapshot = clone_cache(base)
        expected, _ = sequential_repair(shim, base, edited, intervals)
        expected_logits = model(input_ids=ids[:, :2], past_key_values=clone_cache(expected)).logits
    for backend in ("torch", "triton"):
        for layout in ("view", "contiguous"):
            actual, _ = direct_interval_repair(
                shim, base, edited, intervals,
                writeback_backend=backend, attention_layout=layout,
            )
            assert_cache_bits(actual, expected)
            assert_cache_bits(base, snapshot)
            with torch.no_grad():
                actual_logits = model(input_ids=ids[:, :2], past_key_values=clone_cache(actual)).logits
            assert_bits(actual_logits, expected_logits)


def test_triton_layer_rejects_cpu_output_before_importing_kernel():
    key = torch.zeros((1, 2, 8, 4))
    with pytest.raises(ValueError, match="CUDA"):
        _DirectIntervalLayer(key, key.clone(), "triton", "view")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA qualification requires a GPU")
@pytest.mark.parametrize("fault", ["alias", "noncontiguous", "dtype", "dtype_mismatch", "shape"])
def test_triton_layer_qualifies_owned_output_once(fault):
    key = torch.zeros((1, 2, 8, 4), device="cuda")
    value = key.clone()
    if fault == "alias":
        value = key.view_as(key)
    elif fault == "noncontiguous":
        key = key.transpose(1, 2).contiguous().transpose(1, 2)
    elif fault == "dtype":
        key, value = key.double(), value.double()
    elif fault == "dtype_mismatch":
        value = value.half()
    else:
        value = value[:, :, :-1, :]
    with pytest.raises(ValueError):
        _DirectIntervalLayer(key, value, "triton", "view")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA qualification requires a GPU")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("source_layout", ["transposed", "strided"])
def test_triton_direct_scatter_maps_fresh_strided_rows_to_absolute_output(dtype, source_layout):
    pytest.importorskip("triton")
    torch.manual_seed(811)
    batch, heads, length, dim = 2, 3, 29, 7
    start, end = 5, 12
    width = end - start
    key = torch.randn((batch, heads, length, dim), dtype=dtype, device="cuda")
    value = torch.randn_like(key)
    if source_layout == "transposed":
        source_key = torch.randn((batch, width, heads, dim), dtype=dtype, device="cuda").transpose(1, 2)
        source_value = torch.randn((batch, width, heads, dim), dtype=dtype, device="cuda").transpose(1, 2)
    else:
        backing_shape = (batch, heads, 2 * width + 3, 2 * dim + 5)
        source_key = torch.randn(backing_shape, dtype=dtype, device="cuda")[:, :, 1:1 + 2 * width:2, 2:2 + 2 * dim:2]
        source_value = torch.randn(backing_shape, dtype=dtype, device="cuda")[:, :, 1:1 + 2 * width:2, 2:2 + 2 * dim:2]
    assert not source_key.is_contiguous() and not source_value.is_contiguous()
    expected_key, expected_value = key.clone(), value.clone()
    expected_key[:, :, start:end, :].copy_(source_key)
    expected_value[:, :, start:end, :].copy_(source_value)
    source_snapshots = (source_key.clone(), source_value.clone())
    layer = _DirectIntervalLayer(key, value, "triton", "view")
    layer.begin(start, end)
    read_key, read_value = layer.update(
        source_key, source_value,
        {"cache_position": torch.arange(start, end, device="cuda")},
    )
    torch.cuda.synchronize()
    layer.finish()
    assert_bits(key, expected_key)
    assert_bits(value, expected_value)
    assert_bits(source_key, source_snapshots[0])
    assert_bits(source_value, source_snapshots[1])
    assert_bits(read_key, expected_key[:, :, :end, :])
    assert_bits(read_value, expected_value[:, :, :end, :])
