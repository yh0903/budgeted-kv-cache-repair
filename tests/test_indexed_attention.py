"""Independent absolute-causal oracle and indexed-write storage invariants."""

from dataclasses import FrozenInstanceError, replace

import pytest
import torch


from kv_cache_repair.kernels.indexed_attention import (
    BLOCK_M, BLOCK_N, DeviceIndexedPlan, IndexedPlan, indexed_causal_attention, indexed_kv_scatter_,
)
import kv_cache_repair.kernels.indexed_attention as indexed_module


CUDA_ONLY = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires actual CUDA hardware and Triton")


def test_host_plan_owns_immutable_sorted_metadata():
    positions = [3, 4, 8, 9]
    plan = IndexedPlan(positions, 10)
    positions[0] = 0
    assert plan.positions == (3, 4, 8, 9)
    assert plan.selected_count == 4 and plan.logical_length == 10
    assert plan.full_prefix_tiles == (0,) and plan.total_tiles == (1,)
    with pytest.raises(FrozenInstanceError):
        plan.logical_length = 12


@pytest.mark.parametrize("count", [0, 1, 15, 16, 17, 64])
def test_host_plan_bounds_cover_exactly_each_sorted_query_tile(count):
    positions = tuple(63 + i * 67 for i in range(count))
    plan = IndexedPlan(positions, 5000)
    assert len(plan.full_prefix_tiles) == (count + BLOCK_M - 1) // BLOCK_M
    for tile, start in enumerate(range(0, count, BLOCK_M)):
        rows = positions[start:start + BLOCK_M]
        full, total = plan.full_prefix_tiles[tile], plan.total_tiles[tile]
        assert full * BLOCK_N <= min(rows) + 1 < (full + 1) * BLOCK_N
        assert (total - 1) * BLOCK_N < max(rows) + 1 <= total * BLOCK_N
    assert IndexedPlan([], 0).selected_count == 0


@pytest.mark.parametrize("positions,length", [
    ([2, 1], 4), ([1, 1], 4), ([-1], 4), ([4], 4),
    ([True], 4), ([1.0], 4), (torch.tensor([1]), 4),
    ([0], -1), ([], True), ([], 4.0), ([], 1 << 63), (None, 4),
])
def test_invalid_host_plans_fail_before_any_device_operation(positions, length):
    with pytest.raises(ValueError):
        IndexedPlan(positions, length)


def test_host_plan_upload_rejects_cpu():
    with pytest.raises(ValueError, match="CUDA"):
        IndexedPlan([0], 1).to("cpu")


def test_attention_and_scatter_reject_cpu_without_fallback():
    q = torch.zeros((1, 1, 1, 64), dtype=torch.float16)
    with pytest.raises(ValueError, match="CUDA"):
        indexed_causal_attention(q, q, q, IndexedPlan([0], 1))
    with pytest.raises(ValueError, match="CUDA"):
        indexed_kv_scatter_(q, q, q, q, IndexedPlan([0], 1))


@pytest.mark.parametrize("capability,accepted", [((7, 5), False), ((8, 0), True), ((9, 0), True)])
def test_attention_architecture_contract_uses_host_capability_metadata(monkeypatch, capability, accepted):
    indexed_module._cuda_capability.cache_clear()
    monkeypatch.setattr(torch.version, "hip", None)
    calls = []
    def capability_for(index):
        calls.append(index)
        return capability
    monkeypatch.setattr(torch.cuda, "get_device_capability", capability_for)
    try:
        if accepted:
            indexed_module._require_tf32x3_device(torch.device("cuda", 2))
        else:
            with pytest.raises(ValueError, match="compute capability"):
                indexed_module._require_tf32x3_device(torch.device("cuda", 2))
        assert calls == [2]
    finally:
        indexed_module._cuda_capability.cache_clear()


def test_capability_cache_distinguishes_devices_and_performs_no_repeated_lookup(monkeypatch):
    indexed_module._cuda_capability.cache_clear()
    monkeypatch.setattr(torch.version, "hip", None)
    calls = []
    def capability_for(index):
        calls.append(index)
        return (8 + index, 0)
    monkeypatch.setattr(torch.cuda, "get_device_capability", capability_for)
    try:
        for index in (0, 0, 1, 1, 0):
            indexed_module._require_tf32x3_device(torch.device("cuda", index))
        assert calls == [0, 1]
        assert indexed_module._cuda_capability.cache_info().currsize == 2
    finally:
        indexed_module._cuda_capability.cache_clear()


def _positions(count, length):
    return [length - 1] if count == 1 else [i * (length - 1) // (count - 1) for i in range(count)]


def _oracle(q, k, v, positions, scale=None):
    """Dense FP64 attention, independent of kernel tiling and online softmax."""
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    groups = q.shape[1] // k.shape[1]
    keys = k.double().repeat_interleave(groups, dim=1)
    values = v.double().repeat_interleave(groups, dim=1)
    scores = torch.matmul(q.double(), keys.transpose(-1, -2)) * scale
    absolute = torch.tensor(positions, device=q.device, dtype=torch.int64)
    allowed = torch.arange(k.shape[2], device=q.device)[None, :] <= absolute[:, None]
    scores = scores.masked_fill(~allowed[None, None, :, :], float("-inf"))
    return torch.matmul(scores.softmax(-1), values)


def _assert_attention_close(actual, expected):
    # Fixed before the first kernel's GPU evaluation; unchanged for the TF32x3
    # correction. The independent FP64 oracle rounds neither P nor P@V.
    atol, rtol = ((0.002, 0.02) if actual.dtype == torch.bfloat16 else (0.00025, 0.003))
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.double(), expected, atol=atol, rtol=rtol)


@CUDA_ONLY
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_nonuniform_attention_with_cancelling_large_value_channels(dtype):
    pytest.importorskip("triton")
    positions = [0, 1, 3, 7, 16, 31, 63, 64, 95, 127, 191, 255, 256]
    plan = IndexedPlan(positions, 257).to("cuda")
    query = torch.zeros((1, 4, len(positions), 64), device="cuda", dtype=dtype)
    key = torch.zeros((1, 2, 257, 64), device="cuda", dtype=dtype)
    # Two nonzero coordinates make QK accurately representable while creating
    # nonuniform attention. Values cancel in alternating large/small channels;
    # this stresses P@V beyond the low-amplitude random-input test matrix.
    for head in range(4):
        query[:, head, :, 0] = 1 + head / 4
        query[:, head, :, 1] = torch.arange(len(positions), device="cuda", dtype=torch.float32) / 16 - 0.5
    key[:, :, :, 0] = torch.linspace(-3, 3, 257, device="cuda", dtype=torch.float32)
    key[:, :, :, 1] = ((torch.arange(257, device="cuda") % 9).float() - 4) / 8
    signs = torch.where(torch.arange(257, device="cuda") % 2 == 0, 1.0, -1.0)
    magnitude = torch.tensor([64.0] * 16 + [1.0] * 16 + [0.03125] * 32, device="cuda")
    value = (signs[:, None] * magnitude[None, :]).to(dtype)[None, None].repeat(1, 2, 1, 1)
    value[:, 1].mul_(1.5)
    prior_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        # This flag constrains the PyTorch reference; explicit Triton TF32x3
        # remains the selected arithmetic. The frozen tolerances are unchanged.
        torch.backends.cuda.matmul.allow_tf32 = False
        actual = indexed_causal_attention(query, key, value, plan, scale=1.0)
        _assert_attention_close(actual, _oracle(query, key, value, positions, 1.0))
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prior_tf32


def _bits_equal(a, b):
    return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def _view(shape, dtype, layout):
    _, heads, tokens, dim = shape
    if layout == "native":
        storage = torch.randn((1, tokens, heads, dim), device="cuda", dtype=dtype) * 0.5
        return storage, storage.transpose(1, 2)
    if layout == "holes":
        storage = torch.randn((1, heads, 2 * tokens + 3, 2 * dim + 5), device="cuda", dtype=dtype) * 0.5
        return storage, storage[:, :, 1:1 + 2 * tokens:2, 2:2 + 2 * dim:2]
    storage = torch.randn(shape, device="cuda", dtype=dtype) * 0.5
    return storage, storage


@CUDA_ONLY
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dim", [64, 128])
@pytest.mark.parametrize("count", [1, 15, 16, 17, 64])
def test_attention_matches_independent_fp64_oracle(count, dim, dtype):
    pytest.importorskip("triton")
    torch.manual_seed(8200 + count + dim)
    length = 1025
    positions = _positions(count, length)
    plan = IndexedPlan(positions, length).to("cuda")
    q = torch.randn((1, 4, count, dim), device="cuda", dtype=dtype) * 0.5
    k = torch.randn((1, 2, length, dim), device="cuda", dtype=dtype) * 0.5
    v = torch.randn_like(k) * 0.5
    before = [tensor.clone() for tensor in (q, k, v)]
    actual = indexed_causal_attention(q, k, v, plan)
    assert actual.shape == q.shape and actual.dtype == dtype
    _assert_attention_close(actual, _oracle(q, k, v, positions))
    assert all(_bits_equal(tensor, saved) for tensor, saved in zip((q, k, v), before))


@CUDA_ONLY
@pytest.mark.parametrize("layout", ["native", "holes"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_attention_accepts_native_transposes_storage_offsets_and_custom_scale(layout, dtype):
    pytest.importorskip("triton")
    positions = [0, 1, 63, 64, 65, 66, 71, 99, 111, 127, 128, 193, 249, 250, 255, 256, 258]
    plan = IndexedPlan(positions, 259).to("cuda")
    q = _view((1, 6, len(positions), 128), dtype, layout)[1]
    k = _view((1, 3, 259, 128), dtype, layout)[1]
    v = _view((1, 3, 259, 128), dtype, "holes")[1]
    actual = indexed_causal_attention(q, k, v, plan, scale=0.17)
    _assert_attention_close(actual, _oracle(q, k, v, positions, 0.17))


@CUDA_ONLY
@pytest.mark.parametrize("heads,kv_heads", [(1, 1), (4, 1), (4, 4), (8, 2)])
def test_grouped_query_head_mapping(heads, kv_heads):
    pytest.importorskip("triton")
    positions = [0, 64, 130]
    plan = IndexedPlan(positions, 131).to("cuda")
    q = torch.zeros((1, heads, 3, 64), device="cuda", dtype=torch.bfloat16)
    k = torch.zeros((1, kv_heads, 131, 64), device="cuda", dtype=torch.bfloat16)
    v = torch.empty_like(k)
    for head in range(kv_heads):
        v[:, head].fill_(head + 1)
    actual = indexed_causal_attention(q, k, v, plan)
    _assert_attention_close(actual, _oracle(q, k, v, positions))
    for head in range(heads):
        torch.testing.assert_close(actual[:, head].float(),
                                   torch.full_like(actual[:, head].float(), head // (heads // kv_heads) + 1),
                                   atol=0.02, rtol=0.01)


@CUDA_ONLY
def test_fresh_selected_visibility_and_future_exclusion():
    pytest.importorskip("triton")
    positions = [3, 4, 8, 9]
    plan = IndexedPlan(positions, 131).to("cuda")
    key = torch.zeros((1, 2, 131, 64), device="cuda", dtype=torch.bfloat16)
    value = torch.zeros_like(key)
    fresh_key = torch.zeros((1, 2, 4, 64), device="cuda", dtype=torch.bfloat16)
    fresh_value = torch.empty_like(fresh_key)
    for row in range(4):
        fresh_value[:, :, row].fill_(10 + row)
    query = torch.zeros((1, 4, 4, 64), device="cuda", dtype=torch.bfloat16)
    indexed_kv_scatter_(key, value, fresh_key, fresh_value, plan)
    output = indexed_causal_attention(query, key, value, plan)
    _assert_attention_close(output, _oracle(query, key, value, positions))
    assert float(output[0, 0, 2, 0]) > 3.5  # query 8 sees fresh 3, 4, and 8
    preserved = output.clone()
    key[:, :, 9:].fill_(50)
    value[:, :, 9:].fill_(500)
    changed = indexed_causal_attention(query, key, value, plan)
    assert _bits_equal(changed[:, :, :3], preserved[:, :, :3])
    assert not _bits_equal(changed[:, :, 3:], preserved[:, :, 3:])


@CUDA_ONLY
def test_query_at_zero_reads_only_itself_and_isolated_tail_row():
    pytest.importorskip("triton")
    plan = IndexedPlan([0], 130).to("cuda")
    q = torch.randn((1, 4, 1, 64), device="cuda", dtype=torch.float16)
    k = torch.randn((1, 2, 130, 64), device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)
    output = indexed_causal_attention(q, k, v, plan)
    assert _bits_equal(output, v[:, :, :1].repeat_interleave(2, dim=1))


@CUDA_ONLY
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_indexed_scatter_preserves_entire_backings_including_holes(dtype):
    pytest.importorskip("triton")
    positions = [0, 3, 4, 8, 9, 63, 64, 129]
    plan = IndexedPlan(positions, 130).to("cuda")
    pairs = [
        _view((1, 3, 130, 128), dtype, "holes"),
        _view((1, 3, 130, 128), dtype, "native"),
        _view((1, 3, len(positions), 128), dtype, "native"),
        _view((1, 3, len(positions), 128), dtype, "holes"),
    ]
    snapshots = [backing.clone() for backing, _ in pairs]
    views = [snapshot.as_strided(view.shape, view.stride(), view.storage_offset())
             for snapshot, (_, view) in zip(snapshots, pairs)]
    for target, source in zip(views[:2], views[2:]):
        target.index_copy_(2, plan.positions, source)
    indexed_kv_scatter_(*(view for _, view in pairs), plan)
    torch.cuda.synchronize()
    assert all(_bits_equal(backing, expected) for (backing, _), expected in zip(pairs, snapshots))


@CUDA_ONLY
@pytest.mark.parametrize("dtype,bits", [
    (torch.float16, [0, -32768, 31744, -1024, 32257, 1]),
    (torch.bfloat16, [0, -32768, 32640, -128, 32705, 1]),
])
def test_scatter_is_bitwise_copy_including_nan_payloads(dtype, bits):
    pytest.importorskip("triton")
    plan = IndexedPlan([0, 3, 8], 9).to("cuda")
    raw = torch.tensor(bits, dtype=torch.int16, device="cuda").repeat(32).reshape(1, 1, 3, 64)
    fresh_key = raw.view(dtype)
    fresh_value = fresh_key.clone()
    key = torch.zeros((1, 1, 9, 64), device="cuda", dtype=dtype)
    value = key.clone()
    indexed_kv_scatter_(key, value, fresh_key, fresh_value, plan)
    assert _bits_equal(key.index_select(2, plan.positions), fresh_key)
    assert _bits_equal(value.index_select(2, plan.positions), fresh_value)
    assert torch.count_nonzero(key[:, :, [1, 2, 4, 5, 6, 7]].view(torch.int16)) == 0


@CUDA_ONLY
@pytest.mark.parametrize("length", [0, 17])
def test_empty_plan_has_no_writes_and_empty_attention_result(length):
    pytest.importorskip("triton")
    plan = IndexedPlan([], length).to("cuda")
    key = torch.randn((1, 2, length, 64), device="cuda", dtype=torch.float16)
    value, saved = key.clone(), key.clone()
    fresh = torch.empty((1, 2, 0, 64), device="cuda", dtype=torch.float16)
    indexed_kv_scatter_(key, value, fresh, fresh, plan)
    result = indexed_causal_attention(torch.empty((1, 4, 0, 64), device="cuda", dtype=key.dtype), key, value, plan)
    assert result.shape == (1, 4, 0, 64)
    assert _bits_equal(key, saved) and _bits_equal(value, saved)


@CUDA_ONLY
def test_upload_scatter_attention_use_the_callers_nondefault_stream():
    pytest.importorskip("triton")
    # Warm compilation outside the ordered race probe.
    host = IndexedPlan([0, 3, 4, 8, 9], 130)
    plan = host.to("cuda")
    q = torch.zeros((1, 4, 5, 64), device="cuda", dtype=torch.bfloat16)
    k = torch.zeros((1, 2, 130, 64), device="cuda", dtype=torch.bfloat16)
    v = k.clone()
    fresh = torch.zeros((1, 2, 5, 64), device="cuda", dtype=torch.bfloat16)
    indexed_kv_scatter_(k, v, fresh, fresh, plan)
    indexed_causal_attention(q, k, v, plan)
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        torch.cuda._sleep(2_000_000)
        plan = host.to("cuda")
        q = torch.randn_like(q) * 0.5
        k, v = torch.randn_like(k) * 0.5, torch.randn_like(v) * 0.5
        fresh_k, fresh_v = torch.randn_like(fresh), torch.randn_like(fresh)
        indexed_kv_scatter_(k, v, fresh_k, fresh_v, plan)
        output = indexed_causal_attention(q, k, v, plan)
    torch.cuda.current_stream().wait_stream(stream)
    _assert_attention_close(output, _oracle(q, k, v, host.positions))


@CUDA_ONLY
@pytest.mark.parametrize("problem", ["dtype", "batch", "dimension", "gqa", "overlap", "length", "plan", "scale", "grad"])
def test_attention_public_contract_rejects_unsupported_inputs(problem):
    plan = IndexedPlan([0, 3, 8], 9).to("cuda")
    q = torch.zeros((1, 4, 3, 64), device="cuda", dtype=torch.float16)
    k = torch.zeros((1, 2, 9, 64), device="cuda", dtype=torch.float16)
    v = k.clone()
    scale = None
    if problem == "dtype":
        q, k, v = q.float(), k.float(), v.float()
    elif problem == "batch":
        q = q.repeat(2, 1, 1, 1)
    elif problem == "dimension":
        q, k, v = q[:, :, :, :32], k[:, :, :, :32], v[:, :, :, :32]
    elif problem == "gqa":
        q = q[:, :3]
    elif problem == "overlap":
        q = q.as_strided(q.shape, (768, 1, 1, 1))
    elif problem == "length":
        q = q[:, :, :2]
    elif problem == "plan":
        plan = replace(plan, positions=plan.positions.to(torch.int32))
    elif problem == "scale":
        scale = float("nan")
    elif problem == "grad":
        q.requires_grad_(True)
    with pytest.raises(ValueError):
        indexed_causal_attention(q, k, v, plan, scale)


@CUDA_ONLY
@pytest.mark.parametrize("alias", ["outputs", "source"])
def test_scatter_rejects_aliasing_storage(alias):
    plan = IndexedPlan([0, 3, 8], 9).to("cuda")
    key = torch.zeros((1, 2, 9, 64), device="cuda", dtype=torch.float16)
    value = key if alias == "outputs" else key.clone()
    fresh_key = key[:, :, :3] if alias == "source" else torch.zeros((1, 2, 3, 64), device="cuda", dtype=key.dtype)
    fresh_value = fresh_key.clone()
    with pytest.raises(ValueError, match="independent"):
        indexed_kv_scatter_(key, value, fresh_key, fresh_value, plan)


@CUDA_ONLY
@pytest.mark.parametrize("change", ["positions", "full_prefix_tiles", "total_tiles", "replacement", "manufactured"])
def test_device_plan_rejects_mutated_or_manufactured_metadata(change):
    plan = IndexedPlan([0, 3, 8], 9).to("cuda")
    q = torch.zeros((1, 4, 3, 64), device="cuda", dtype=torch.float16)
    k = torch.zeros((1, 2, 9, 64), device="cuda", dtype=torch.float16)
    if change in ("positions", "full_prefix_tiles", "total_tiles"):
        getattr(plan, change).fill_(-1)
    elif change == "replacement":
        plan = replace(plan, positions=plan.positions.clone())
    else:
        plan = DeviceIndexedPlan(plan.host_plan, plan.positions, plan.full_prefix_tiles, plan.total_tiles)
    with pytest.raises(ValueError, match="metadata"):
        indexed_causal_attention(q, k, k, plan)


@CUDA_ONLY
def test_plan_upload_under_inference_mode_still_tracks_mutations():
    pytest.importorskip("triton")
    with torch.inference_mode():
        plan = IndexedPlan([0], 1).to("cuda")
        q = torch.ones((1, 1, 1, 64), device="cuda", dtype=torch.float16)
        actual = indexed_causal_attention(q, q, q, plan)
        assert _bits_equal(actual, q)
        plan.positions.add_(0)
        with pytest.raises(ValueError, match="mutated"):
            indexed_causal_attention(q, q, q, plan)
