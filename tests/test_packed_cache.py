"""Packed refresh semantics, model isolation, and independent cache ownership."""

import copy
from types import SimpleNamespace

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM, Qwen3Config, Qwen3ForCausalLM
from transformers.cache_utils import DynamicCache

from kv_cache_repair.execution import sequential_repair
from kv_cache_repair.cache import OperatorShim
from kv_cache_repair.cache import clone_cache
from kv_cache_repair.kernels.indexed_attention import IndexedPlan
from kv_cache_repair.kernels.packed_cache import (
    _PackedContext, _PackedLayer, _context, _packed_attention, packed_interval_repair,
)


@pytest.fixture(scope="module", autouse=True)
def small_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make_model(family="llama", attention="sdpa", kv_heads=2, *, head_dim=8, dtype=torch.float32, seed=71):
    torch.manual_seed(seed)
    options = dict(
        vocab_size=97, hidden_size=4 * head_dim, intermediate_size=8 * head_dim,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=kv_heads,
        head_dim=head_dim, max_position_embeddings=512, attention_dropout=0.0,
    )
    if family == "llama":
        model = LlamaForCausalLM(LlamaConfig(**options))
    else:
        model = Qwen3ForCausalLM(Qwen3Config(**options, layer_types=["full_attention"] * 2))
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


def make_request(model, length=12):
    ids = (torch.arange(length, device=model.device) % 60 + 4).unsqueeze(0)
    with torch.no_grad():
        base = model(input_ids=ids, use_cache=True).past_key_values
    return OperatorShim(model, SimpleNamespace()), ids, base


def assert_ownership_and_untouched(actual, base, snapshot, intervals):
    assert_cache_bits(base, snapshot)
    selected = {position for start, end in intervals for position in range(start, end)}
    untouched = [position for position in range(base.get_seq_length()) if position not in selected]
    for layer in range(len(base)):
        for component in (0, 1):
            assert_bits(actual[layer][component][:, :, untouched], base[layer][component][:, :, untouched])
            if selected:
                assert actual[layer][component].data_ptr() != base[layer][component].data_ptr()


@pytest.mark.parametrize("family", ["llama", "qwen3"])
@pytest.mark.parametrize("attention", ["eager", "sdpa"])
@pytest.mark.parametrize("kv_heads", [1, 2, 4])
@pytest.mark.parametrize("intervals", [[], [(0, 12)], [(2, 12)], [(3, 5), (8, 10)], [(0, 1), (3, 4), (8, 9), (11, 12)]])
def test_cpu_packed_matches_interval_recurrence(family, attention, kv_heads, intervals):
    model = make_model(family, attention, kv_heads)
    shim, ids, base = make_request(model)
    snapshot = clone_cache(base)
    with torch.no_grad():
        expected, _ = sequential_repair(shim, base, ids + 13, intervals)
    actual, receipt = packed_interval_repair(shim, base, ids + 13, intervals)
    assert isinstance(actual, DynamicCache)
    assert actual.get_seq_length() == ids.shape[-1]
    for layer in range(len(base)):
        for component in (0, 1):
            torch.testing.assert_close(actual[layer][component], expected[layer][component], rtol=1e-5, atol=2e-6)
    assert_ownership_and_untouched(actual, base, snapshot, intervals)
    assert receipt["recomputed_tokens"] == sum(end - start for start, end in intervals)
    assert receipt["schedule"] == "packed_selected_rows_layer_major"
    if not intervals:
        assert actual is base
    with torch.no_grad():
        query = torch.tensor([[77, 78]])
        actual_logits = model(input_ids=query, past_key_values=clone_cache(actual)).logits
        expected_logits = model(input_ids=query, past_key_values=clone_cache(expected)).logits
    torch.testing.assert_close(actual_logits, expected_logits, rtol=1e-5, atol=2e-6)


@pytest.mark.parametrize("family", ["llama", "qwen3"])
def test_full_support_matches_clean_prefill(family):
    model = make_model(family)
    shim, ids, base = make_request(model)
    actual, _ = packed_interval_repair(shim, base, ids + 13, [(0, ids.shape[-1])])
    with torch.no_grad():
        clean = model(input_ids=ids + 13).past_key_values
    for layer in range(len(base)):
        for component in (0, 1):
            torch.testing.assert_close(actual[layer][component], clean[layer][component], rtol=1e-5, atol=2e-6)


@pytest.mark.parametrize("family", ["llama", "qwen3"])
def test_requests_share_no_state_and_leave_native_model_unchanged(family):
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    model = make_model(family, "eager")
    shim, ids, base = make_request(model)
    modules = dict(model.named_modules())
    configs = {name: module.config.to_dict() for name, module in modules.items() if hasattr(module, "config")}
    forwards = {name: module.forward.__func__ for name, module in modules.items()}
    parameters = {name: parameter.data_ptr() for name, parameter in model.named_parameters()}
    implementations = dict(ALL_ATTENTION_FUNCTIONS)
    mask_implementations = dict(ALL_MASK_ATTENTION_FUNCTIONS)
    with torch.no_grad():
        original_logits = model(input_ids=ids).logits.clone()
    first, _ = packed_interval_repair(shim, base, ids + 13, [(3, 5), (8, 10)])
    first_snapshot = clone_cache(first)
    second, _ = packed_interval_repair(shim, base, ids + 19, [(0, 2), (10, 12)])
    repeated, _ = packed_interval_repair(shim, base, ids + 13, [(3, 5), (8, 10)])
    assert_cache_bits(first, first_snapshot)
    assert_cache_bits(first, repeated)
    assert first is not second
    assert dict(model.named_modules()) == modules
    assert {name: parameter.data_ptr() for name, parameter in model.named_parameters()} == parameters
    for name, module in model.named_modules():
        assert module.forward.__func__ is forwards[name]
        if name in configs:
            assert module.config.to_dict() == configs[name]
    for name, implementation in implementations.items():
        assert ALL_ATTENTION_FUNCTIONS[name] is implementation
    assert dict(ALL_MASK_ATTENTION_FUNCTIONS) == mask_implementations
    with torch.no_grad():
        assert_bits(model(input_ids=ids).logits, original_logits)


def test_supplied_host_plan_matches_exact_support():
    model = make_model()
    shim, ids, base = make_request(model)
    intervals = [(3, 5), (8, 10)]
    plan = IndexedPlan((3, 4, 8, 9), 12)
    actual, _ = packed_interval_repair(shim, base, ids + 13, intervals, plan=plan)
    expected, _ = packed_interval_repair(shim, base, ids + 13, intervals)
    assert_cache_bits(actual, expected)
    for invalid in (IndexedPlan((3, 4, 8), 12), IndexedPlan((3, 4, 8, 9), 13), object()):
        with pytest.raises(ValueError, match="IndexedPlan"):
            packed_interval_repair(shim, base, ids, intervals, plan=invalid)


def test_empty_support_does_no_model_or_descriptor_work(monkeypatch):
    import kv_cache_repair.kernels.packed_cache as module
    def forbidden(*args, **kwargs):
        raise AssertionError("empty support must not prepare or execute work")
    monkeypatch.setattr(module, "_qualify", forbidden)
    monkeypatch.setattr(module, "_make_native_view", forbidden)
    monkeypatch.setattr(IndexedPlan, "to", forbidden)
    base = DynamicCache()
    for layer in range(2):
        base.update(torch.zeros(1, 2, 0, 8), torch.zeros(1, 2, 0, 8), layer)
    actual, receipt = packed_interval_repair(object(), base, torch.empty((1, 0), dtype=torch.long), [])
    assert actual is base
    assert receipt["output_cache_ownership"] == "borrowed_base"


def test_layer_scatter_happens_before_full_cache_is_returned():
    positions = torch.tensor([3, 4, 8, 9])
    key = torch.arange(2 * 12 * 8).reshape(1, 2, 12, 8).float()
    value = key + 1000
    original_key, original_value = key.clone(), value.clone()
    context = _PackedContext(positions, 12, "sdpa")
    layer = _PackedLayer(key, value, context, "torch")
    fresh_key = torch.full((1, 2, 4, 8), -7.0)
    fresh_value = torch.full_like(fresh_key, -9.0)
    returned_key, returned_value = layer.update(fresh_key, fresh_value, {"cache_position": positions})
    assert returned_key is key and returned_value is value
    assert layer.get_seq_length() == 12 and layer.get_mask_sizes(positions) == (12, 0)
    assert_bits(key[:, :, positions], fresh_key)
    assert_bits(value[:, :, positions], fresh_value)
    untouched = [0, 1, 2, 5, 6, 7, 10, 11]
    assert_bits(key[:, :, untouched], original_key[:, :, untouched])
    assert_bits(value[:, :, untouched], original_value[:, :, untouched])
    with pytest.raises(RuntimeError, match="more than once"):
        layer.update(fresh_key, fresh_value, {"cache_position": positions})


def test_layer_rejects_replaced_absolute_positions_before_write():
    positions = torch.tensor([3, 4])
    key = torch.zeros(1, 2, 12, 8)
    layer = _PackedLayer(key, key.clone(), _PackedContext(positions, 12, "sdpa"), "torch")
    with pytest.raises(RuntimeError, match="replaced"):
        layer.update(torch.ones(1, 2, 2, 8), torch.ones(1, 2, 2, 8), {"cache_position": positions.clone()})
    assert torch.count_nonzero(key) == 0


def test_absolute_attention_reads_unselected_past_and_excludes_future():
    plan = IndexedPlan((3, 4, 8, 9), 12)
    positions = torch.tensor(plan.positions)
    context = _context(plan, positions, "sdpa", None)
    module = SimpleNamespace(num_key_value_groups=2, is_causal=True)
    query, key = torch.zeros(1, 4, 4, 8), torch.zeros(1, 2, 12, 8)
    value = torch.arange(12).reshape(1, 1, 12, 1).float().expand(1, 2, 12, 8).clone()
    def attend(values):
        return _packed_attention(module, query, key, values, None, sur_packed_context=context)[0]
    actual = attend(value)
    for row, position in enumerate(plan.positions):
        torch.testing.assert_close(actual[:, row], torch.full((1, 4, 8), position / 2))
    changed_future = value.clone()
    changed_future[:, :, 10:] = 10000
    assert_bits(attend(changed_future), actual)
    changed_legal_unselected = value.clone()
    changed_legal_unselected[:, :, 1] += 100
    assert torch.all(attend(changed_legal_unselected) > actual)


@pytest.mark.parametrize("failure", ["forward_raises", "missing_layer", "replaced_cache"])
def test_failed_transaction_keeps_original_model_and_base_unchanged(monkeypatch, failure):
    import kv_cache_repair.kernels.packed_cache as module
    model = make_model()
    shim, ids, base = make_request(model)
    snapshot = clone_cache(base)
    config = model.config.to_dict()
    modules = dict(model.named_modules())
    with torch.no_grad():
        original_logits = model(input_ids=ids).logits.clone()
    def failing_forward(**kwargs):
        cache = kwargs["past_key_values"]
        count = kwargs["input_ids"].shape[-1]
        fresh = torch.full((1, 2, count, 8), 999.0)
        cache.update(fresh, fresh, 0, {"cache_position": kwargs["cache_position"]})
        if failure == "forward_raises":
            raise RuntimeError("injected failure")
        return SimpleNamespace(past_key_values=DynamicCache() if failure == "replaced_cache" else cache)
    monkeypatch.setattr(module, "_make_native_view", lambda base: failing_forward)
    with pytest.raises(RuntimeError):
        packed_interval_repair(shim, base, ids + 13, [(3, 5)])
    assert_cache_bits(base, snapshot)
    assert model.config.to_dict() == config
    assert dict(model.named_modules()) == modules
    with torch.no_grad():
        assert_bits(model(input_ids=ids).logits, original_logits)


@pytest.mark.parametrize("fault", ["version", "class", "base_forward", "training", "rope", "sliding", "attention_backend", "writeback_backend", "interval", "cuda_attention", "cuda_scatter"])
def test_fail_closed_supported_contract(monkeypatch, fault):
    import kv_cache_repair.kernels.packed_cache as module
    model = make_model()
    shim, ids, base = make_request(model)
    snapshot = clone_cache(base)
    kwargs, intervals = {}, [(3, 5)]
    if fault == "version":
        monkeypatch.setattr(module.transformers, "__version__", "unsupported")
    elif fault == "class":
        class CustomLlama(LlamaForCausalLM):
            pass
        model.__class__ = CustomLlama
    elif fault == "base_forward":
        monkeypatch.setattr(model.model, "forward", lambda **kwargs: None)
    elif fault == "training":
        model.train()
    elif fault == "rope":
        model.config.rope_scaling = {"rope_type": "dynamic", "factor": 2.0}
    elif fault == "sliding":
        model.config.layer_types = ["full_attention", "sliding_attention"]
    elif fault == "attention_backend":
        kwargs["attention_backend"] = "unknown"
    elif fault == "writeback_backend":
        kwargs["writeback_backend"] = "unknown"
    elif fault == "interval":
        intervals = [(3, 5), (5, 6)]
    elif fault == "cuda_attention":
        kwargs["attention_backend"] = "triton"
    else:
        kwargs["writeback_backend"] = "triton"
    with pytest.raises(ValueError):
        packed_interval_repair(shim, base, ids, intervals, **kwargs)
    assert_cache_bits(base, snapshot)


def assert_within_native_precision_floor(candidate, native_sdpa, matched_fp32):
    """Frozen controlled-numerics gate; deliberately not task certification.

    The factor 2 and FP32 epsilon floor were fixed before calibration outcomes.
    Unlike pointwise allclose between two low-precision schedules, this compares
    both schedules to the same higher-precision computation and bounds both
    aggregate RMS error and the largest absolute error. Seed 191 is a holdout
    from the original seed-71 fixture. Standalone FP64 kernel limits are unchanged.
    """
    reference = matched_fp32.double()
    candidate, native_sdpa = candidate.double(), native_sdpa.double()
    assert torch.isfinite(reference).all()
    assert torch.isfinite(candidate).all()
    assert torch.isfinite(native_sdpa).all()
    candidate_error, native_error = candidate - reference, native_sdpa - reference
    floor = torch.finfo(torch.float32).eps * max(1.0, reference.abs().max().item())
    for metric, candidate_value, native_value in (
        ("max_abs", candidate_error.abs().max().item(), native_error.abs().max().item()),
        ("rmse", candidate_error.square().mean().sqrt().item(), native_error.square().mean().sqrt().item()),
    ):
        budget = 2.0 * max(native_value, floor)
        assert candidate_value <= budget, (
            f"{metric}: candidate error {candidate_value} exceeds frozen budget {budget}; "
            f"native SDPA error {native_value}, FP32 floor {floor}"
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA qualification requires a GPU")
@pytest.mark.parametrize("seed", [71, 191])
@pytest.mark.parametrize("family", ["llama", "qwen3"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("kv_heads", [1, 2, 4])
@pytest.mark.parametrize("attention_backend", ["triton", "flex"])
def test_cuda_packed_backend_integration(family, dtype, kv_heads, attention_backend, seed):
    pytest.importorskip("triton")
    previous_precision = torch.get_float32_matmul_precision()
    previous_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    previous_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    try:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model = make_model(family, kv_heads=kv_heads, head_dim=64, dtype=dtype, seed=seed).cuda()
        shim, ids, base = make_request(model, length=137)
        intervals = [(0, 1), (3, 5), (62, 68), (126, 132), (136, 137)]
        positions = torch.tensor([p for start, end in intervals for p in range(start, end)], device="cuda")
        snapshot = clone_cache(base)
        # Promote existing quantized weights, buffers, and initial mixed cache.
        # A newly initialized FP32 model would change the test's operands.
        control_model = copy.deepcopy(model).float()
        control_shim = OperatorShim(control_model, SimpleNamespace())
        control_base = DynamicCache()
        for layer in range(len(base)):
            control_base.update(base[layer][0].float(), base[layer][1].float(), layer)
        control, _ = packed_interval_repair(control_shim, control_base, ids + 13, intervals)
        native_sdpa, _ = packed_interval_repair(shim, base, ids + 13, intervals)
        torch_scatter, _ = packed_interval_repair(
            shim, base, ids + 13, intervals,
            attention_backend=attention_backend, writeback_backend="torch",
        )
        actual, _ = packed_interval_repair(
            shim, base, ids + 13, intervals,
            attention_backend=attention_backend, writeback_backend="triton",
        )
        assert_ownership_and_untouched(actual, base, snapshot, intervals)
        assert_ownership_and_untouched(torch_scatter, base, snapshot, intervals)
        assert_cache_bits(actual, torch_scatter)
        for layer in range(len(base)):
            for component in (0, 1):
                assert_within_native_precision_floor(
                    actual[layer][component].index_select(2, positions),
                    native_sdpa[layer][component].index_select(2, positions),
                    control[layer][component].index_select(2, positions),
                )
        query = torch.tensor([[77, 78]], device="cuda")
        with torch.no_grad():
            actual_logits = model(input_ids=query, past_key_values=clone_cache(actual)).logits
            torch_scatter_logits = model(input_ids=query, past_key_values=clone_cache(torch_scatter)).logits
            native_logits = model(input_ids=query, past_key_values=clone_cache(native_sdpa)).logits
            control_logits = control_model(input_ids=query, past_key_values=clone_cache(control)).logits
        assert_bits(actual_logits, torch_scatter_logits)
        assert_within_native_precision_floor(actual_logits, native_logits, control_logits)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_matmul_tf32
        torch.backends.cudnn.allow_tf32 = previous_cudnn_tf32
        torch.set_float32_matmul_precision(previous_precision)
