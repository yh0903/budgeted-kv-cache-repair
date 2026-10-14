"""CUDA copy correctness, including storage holes and absolute source rows."""


import pytest
import torch


from kv_cache_repair.kernels.writeback import copy_kv_interval_


CUDA_ONLY = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires real CUDA hardware and Triton")


def _view(shape, dtype, layout):
    batch, heads, tokens, dim = shape
    if layout == "contiguous":
        backing = torch.randn(shape, device="cuda", dtype=dtype)
        return backing, backing
    if layout == "holes":
        backing = torch.randn((batch, heads, 2 * tokens + 3, 2 * dim + 5), device="cuda", dtype=dtype)
        return backing, backing[:, :, 1:1 + 2 * tokens:2, 2:2 + 2 * dim:2]
    if layout == "transposed":
        backing = torch.randn((batch, tokens, heads, dim), device="cuda", dtype=dtype)
        return backing, backing.permute(0, 2, 1, 3)
    if layout == "batch_head_holes":
        backing = torch.randn((2 * batch + 1, 2 * heads + 1, tokens + 2, dim + 4), device="cuda", dtype=dtype)
        return backing, backing[1:1 + 2 * batch:2, 1:1 + 2 * heads:2, 1:1 + tokens, 2:2 + dim]
    raise AssertionError(layout)


@CUDA_ONLY
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape,start,end", [
    ((1, 8, 37, 128), 3, 4),
    ((2, 3, 71, 37), 5, 22),
    ((2, 5, 33, 7), 0, 33),
    ((1, 1, 9, 13), 0, 0),
])
@pytest.mark.parametrize("strided", [False, True])
def test_writeback_matches_slice_copy_and_preserves_storage(shape, start, end, dtype, strided):
    pytest.importorskip("triton")
    torch.manual_seed(701 + end)
    source_shape = (shape[0], shape[1], max(1, end), shape[3])
    layouts = ("holes", "transposed", "batch_head_holes", "holes") if strided else ("contiguous",) * 4
    pairs = [_view(s, dtype, layout) for s, layout in zip((shape, shape, source_shape, source_shape), layouts)]
    actual_backing = [pair[0] for pair in pairs]
    tensors = [pair[1] for pair in pairs]
    snapshots = [backing.clone() for backing in actual_backing]
    expected_views = [
        backing.as_strided(tensor.shape, tensor.stride(), tensor.storage_offset())
        for backing, tensor in zip(snapshots, tensors)
    ]
    expected_views[0][:, :, start:end, :].copy_(expected_views[2][:, :, start:end, :])
    expected_views[1][:, :, start:end, :].copy_(expected_views[3][:, :, start:end, :])

    copy_kv_interval_(*tensors, start, end)
    torch.cuda.synchronize()

    # Comparing entire backing storage also checks unselected rows, view holes,
    # nonzero storage offsets, and unchanged source storage.
    for actual, expected in zip(actual_backing, snapshots):
        assert torch.equal(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8))


def test_cpu_inputs_are_rejected_instead_of_silently_falling_back():
    tensors = [torch.zeros((1, 1, 3, 4)) for _ in range(4)]
    with pytest.raises(ValueError, match="CUDA"):
        copy_kv_interval_(*tensors, 0, 1)


@CUDA_ONLY
@pytest.mark.parametrize("start,end", [(-1, 2), (2, 1), (0, 5), (True, 2), (0, 1.0)])
def test_invalid_interval_is_rejected(start, end):
    tensors = [torch.zeros((1, 2, 4, 8), device="cuda") for _ in range(4)]
    with pytest.raises(ValueError, match="interval"):
        copy_kv_interval_(*tensors, start, end)


@CUDA_ONLY
@pytest.mark.parametrize("kind", ["source_alias", "output_alias", "dtype", "shape", "zero_stride", "overlap"])
def test_invalid_tensor_contract_is_rejected(kind):
    tensors = [torch.zeros((1, 2, 4, 8), device="cuda") for _ in range(4)]
    if kind == "source_alias":
        tensors[2] = tensors[0][:, :, :, :]
    elif kind == "output_alias":
        tensors[1] = tensors[0][:, :, :, :]
    elif kind == "dtype":
        tensors[2] = tensors[2].to(torch.float64)
    elif kind == "shape":
        tensors[2] = torch.zeros((1, 3, 4, 8), device="cuda")
    elif kind == "zero_stride":
        tensors[0] = torch.zeros((1, 1, 4, 8), device="cuda").expand(1, 2, 4, 8)
    elif kind == "overlap":
        tensors[0] = tensors[0].as_strided((1, 2, 4, 8), (64, 1, 1, 1))
    with pytest.raises(ValueError):
        copy_kv_interval_(*tensors, 0, 2)


@CUDA_ONLY
@pytest.mark.parametrize("dtype,bits", [
    (torch.float16, [0, -32768, 31744, -1024, 32257, 1]),
    (torch.bfloat16, [0, -32768, 32640, -128, 32705, 1]),
])
def test_writeback_preserves_raw_bits_including_nan_payloads(dtype, bits):
    pytest.importorskip("triton")
    raw = torch.tensor(bits, dtype=torch.int16, device="cuda").reshape(1, 1, 2, 3)
    source_key = raw.view(dtype)
    source_value = source_key.clone()
    target_key = torch.zeros((1, 1, 4, 3), device="cuda", dtype=dtype)
    target_value = target_key.clone()
    copy_kv_interval_(target_key, target_value, source_key, source_value, 0, 2)
    torch.cuda.synchronize()
    for target in (target_key, target_value):
        assert torch.equal(target[:, :, :2, :].view(torch.int16), raw)
        assert torch.count_nonzero(target[:, :, 2:, :].view(torch.int16)) == 0
