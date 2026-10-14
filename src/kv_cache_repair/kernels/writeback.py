"""Copy fresh K/V rows without changing any floating-point arithmetic."""

import torch


def _nonoverlapping_layout(tensor):
    # Sufficient proof for ordinary contiguous, permuted and strided slices.
    # Reject unusual layouts rather than allow concurrent overlapping stores.
    extent = 1
    for stride, size in sorted(zip(tensor.stride(), tensor.shape)):
        if size > 1:
            if stride < extent:
                return False
            extent += (size - 1) * stride
    return True


def copy_kv_interval_(out_key, out_value, new_key, new_value, start, end):
    """Overwrite [start:end) in two independent CUDA output tensors.

    All tensors use [batch, kv_head, token, head_dim] layout, with arbitrary
    positive nonoverlapping strides. Source sequence lengths may differ from output lengths.
    Inputs and outputs must have separate storage (as provided by the executor).
    This is a pure copy; no RoPE, arithmetic, or attention is performed.
    """
    tensors = (out_key, out_value, new_key, new_value)
    if any(t.ndim != 4 for t in tensors):
        raise ValueError("K/V tensors must have four dimensions")
    if any(t.device != out_key.device or not t.is_cuda for t in tensors):
        raise ValueError("K/V tensors must be on the same CUDA device")
    if out_key.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("supported dtypes are float16, bfloat16, float32")
    if any(t.dtype != out_key.dtype for t in tensors):
        raise ValueError("K/V dtypes must match")
    if any(type(p) is not int for p in (start, end)):
        raise ValueError("interval bounds must be integers")
    if not 0 <= start <= end <= min(t.shape[2] for t in tensors):
        raise ValueError("interval exceeds a K/V sequence length")
    batch, heads, _, dim = out_key.shape
    if not batch or not heads or not dim:
        raise ValueError("batch, heads, and head dimension must be positive")
    if any((t.shape[0], t.shape[1], t.shape[3]) != (batch, heads, dim) for t in tensors):
        raise ValueError("K/V batch, head count, and head dimensions must match")
    if any(any(s <= 0 for s in t.stride()) for t in tensors):
        raise ValueError("K/V strides must be positive")
    if not all(_nonoverlapping_layout(t) for t in tensors):
        raise ValueError("K/V layouts must be provably nonoverlapping")
    if any(dst.untyped_storage().data_ptr() == src.untyped_storage().data_ptr()
           for dst in tensors[:2] for src in tensors[2:]):
        raise ValueError("source and output storage must be separate")
    if out_key.untyped_storage().data_ptr() == out_value.untyped_storage().data_ptr():
        raise ValueError("output K and V storage must be separate")
    if start == end:
        return

    # Lazy import keeps the reference/CPU tests usable without Triton installed.
    from ._writeback_triton import copy_kv_interval

    with torch.cuda.device(out_key.device):
        copy_kv_interval(out_key, out_value, new_key, new_value, start, end)
