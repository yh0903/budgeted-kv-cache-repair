"""Real-CUDA tests for reusable code plans; no cached data or stale streams."""

import gc
import math
import weakref

import pytest
import torch


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="compiled launch plans require real CUDA")


@pytest.fixture
def module():
    triton = pytest.importorskip("triton")
    if triton.__version__ != "3.5.1":
        pytest.skip("compiled-runner API is pinned to Triton 3.5.1")
    from kv_cache_repair.kernels import _compiled_scatter

    _compiled_scatter._clear_runner_cache()
    yield _compiled_scatter
    _compiled_scatter._clear_runner_cache()


def _tensor(shape, dtype, device, offset=0):
    count = math.prod(shape)
    backing = torch.randn(count + offset, dtype=dtype, device=device)
    return backing[offset:].view(shape)


def _case(module, *, start=5, width=7, dtype=torch.bfloat16, offset=0,
          device="cuda:0", transposed=False):
    output_shape = (1, 3, 64, 7)
    source_shape = (1, 3, width, 7)
    key = _tensor(output_shape, dtype, device, offset)
    value = _tensor(output_shape, dtype, device, offset)
    if transposed:
        new_key = _tensor((1, width, 3, 7), dtype, device, offset).transpose(1, 2)
        new_value = _tensor((1, width, 3, 7), dtype, device, offset).transpose(1, 2)
    else:
        new_key = _tensor(source_shape, dtype, device, offset)
        new_value = _tensor(source_shape, dtype, device, offset)
    expected_key, expected_value = key.clone(), value.clone()
    expected_key[:, :, start:start + width, :].copy_(new_key)
    expected_value[:, :, start:start + width, :].copy_(new_value)
    source_before = (new_key.clone(), new_value.clone())
    with torch.cuda.device(device):
        module.scatter_fresh_kv(key, value, new_key, new_value, start, start + width)
    return (key, value, new_key, new_value), (expected_key, expected_value, *source_before)


def _assert_bits(actual, expected):
    for left, right in zip(actual, expected):
        assert left.shape == right.shape and left.dtype == right.dtype
        assert torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8))


def test_changed_data_starts_and_pointer_alignment_share_code_only(module):
    torch.manual_seed(955)
    for start, offset in ((0, 0), (1, 1), (5, 3), (16, 0), (33, 1)):
        actual, expected = _case(module, start=start, offset=offset)
        torch.cuda.synchronize()
        _assert_bits(actual, expected)
    info = module._runner_cache_info()
    assert info["size"] == 1
    assert info["misses"] == 1 and info["hits"] == 4
    assert dict(info["signatures"][0])["OUT_START"] == "i64"


def test_launch_plan_does_not_retain_any_input_or_output_tensor(module):
    actual, expected = _case(module)
    torch.cuda.synchronize()
    _assert_bits(actual, expected)
    references = [weakref.ref(tensor) for tensor in actual]
    del actual, expected
    gc.collect()
    assert module._runner_cache_info()["size"] == 1
    assert all(reference() is None for reference in references)


def test_nondefault_stream_reuses_plan_and_observes_current_stream(module):
    actual, expected = _case(module)
    torch.cuda.synchronize()
    _assert_bits(actual, expected)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        # Delay new-stream initialization so an accidentally cached default
        # stream launch cannot pass merely because its inputs were ready early.
        torch.cuda._sleep(2_000_000)
        actual, expected = _case(module, start=1, offset=1)
        # Capture results on this stream before its event completes. A runner
        # accidentally using a stale stream can race this same-stream copy.
        snapshots = tuple(tensor.clone() for tensor in actual)
    stream.synchronize()
    _assert_bits(snapshots, expected)
    info = module._runner_cache_info()
    assert info["size"] == 1 and info["hits"] == 1


def test_dtype_and_strides_partition_specializations(module):
    for dtype, transposed in ((torch.float32, False), (torch.float16, False),
                              (torch.bfloat16, False), (torch.bfloat16, True)):
        actual, expected = _case(module, dtype=dtype, transposed=transposed)
        torch.cuda.synchronize()
        _assert_bits(actual, expected)
    info = module._runner_cache_info()
    assert info["size"] == 4 and info["misses"] == 4
    assert {key[0] for key in info["keys"]} == {0}


def test_metadata_cache_has_bounded_lru_eviction(module, monkeypatch):
    monkeypatch.setattr(module, "_MAX_RUNNERS", 2)
    for width in (3, 5, 3, 7):
        actual, expected = _case(module, width=width)
        torch.cuda.synchronize()
        _assert_bits(actual, expected)
    info = module._runner_cache_info()
    assert info["size"] == 2
    assert info["hits"] == 1 and info["misses"] == 3
    assert [key[7] for key in info["keys"]] == [3, 7]


def test_live_launch_hooks_are_preserved_on_cached_runner(module):
    actual, expected = _case(module)
    torch.cuda.synchronize()
    _assert_bits(actual, expected)
    events = []
    def enter(*args):
        events.append("enter")
    def leave(*args):
        events.append("exit")
    module.knobs.runtime.launch_enter_hook.add(enter)
    module.knobs.runtime.launch_exit_hook.add(leave)
    try:
        actual, expected = _case(module, start=1)
        torch.cuda.synchronize()
        _assert_bits(actual, expected)
    finally:
        module.knobs.runtime.launch_enter_hook.remove(enter)
        module.knobs.runtime.launch_exit_hook.remove(leave)
    assert events == ["enter", "exit"]
    assert module._runner_cache_info()["hits"] == 1


def test_jit_pre_run_hooks_use_ordinary_launch(module):
    events = []
    def pre_run(*args, **kwargs):
        events.append("pre_run")
    # Compatibility concerns belong to the ordinary fallback kernel's JIT too:
    # a hook attached to our dedicated kernel must still execute on fallback.
    module._scatter_kernel.pre_run_hooks.append(pre_run)
    try:
        actual, expected = _case(module)
        torch.cuda.synchronize()
        _assert_bits(actual, expected)
    finally:
        module._scatter_kernel.pre_run_hooks.remove(pre_run)
    assert events == ["pre_run"]
    assert module._runner_cache_info()["size"] == 0


def test_debug_mode_uses_ordinary_launch(module, monkeypatch):
    monkeypatch.setattr(module.knobs.runtime, "debug", True)
    actual, expected = _case(module)
    torch.cuda.synchronize()
    _assert_bits(actual, expected)
    assert module._runner_cache_info()["size"] == 0
    assert module._runner_cache_info()["fallbacks"] == 1


def test_always_compile_mode_uses_ordinary_launch(module, monkeypatch):
    monkeypatch.setattr(module.knobs.compilation, "always_compile", True)
    actual, expected = _case(module)
    torch.cuda.synchronize()
    _assert_bits(actual, expected)
    assert module._runner_cache_info()["size"] == 0
    assert module._runner_cache_info()["fallbacks"] == 1


def test_missing_compiled_executable_fails_instead_of_skipping_write(module, monkeypatch):
    monkeypatch.setattr(module._scatter_kernel, "warmup", lambda *args, **kwargs: None)
    with pytest.raises(RuntimeError, match="compiled scatter executable"):
        _case(module)
    assert module._runner_cache_info()["size"] == 0
