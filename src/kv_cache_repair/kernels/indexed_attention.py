"""Validated plans and CUDA APIs for selected-query causal attention.

The host plan owns only immutable integer metadata. Upload once per request;
device-plan tensors are read-only by contract and must not be modified while a
consumer is pending. Neither API copies metadata back to the host. Attention
uses the already updated full cache; scatter and attention must be ordered on
the same stream (or by an explicit caller-owned event).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import math
from numbers import Integral, Real
from typing import Sequence

import torch

from .writeback import _nonoverlapping_layout


BLOCK_M = 16
BLOCK_N = 64
_DTYPES = (torch.float16, torch.bfloat16)


@lru_cache(maxsize=32)
def _cuda_capability(device_index: int) -> tuple[int, int]:
    """Cache host device properties only; never retain a request or tensor."""
    return tuple(torch.cuda.get_device_capability(device_index))


def _require_tf32x3_device(device: torch.device) -> None:
    # Inputs have already established a live CUDA device. Reading capability
    # metadata requires no tensor readback; subsequent calls reuse the tuple.
    if (torch.version.hip is not None or device.type != "cuda" or device.index is None
            or _cuda_capability(device.index)[0] < 8):
        raise ValueError("indexed attention TF32x3 requires NVIDIA CUDA compute capability >= 8.0")


@dataclass(frozen=True)
class IndexedPlan:
    """Sorted absolute positions, validated entirely on the CPU.

    Tensor positions are deliberately rejected: creating a plan must not hide a
    device-to-host synchronization. A caller with GPU selection output must make
    its transfer explicit before constructing this host plan.
    """

    positions: Sequence[int]
    logical_length: int
    full_prefix_tiles: tuple[int, ...] = field(init=False)
    total_tiles: tuple[int, ...] = field(init=False)

    def __post_init__(self):
        if isinstance(self.logical_length, bool) or not isinstance(self.logical_length, Integral):
            raise ValueError("logical_length must be a host integer")
        length = int(self.logical_length)
        if length < 0 or length > (1 << 63) - 1:
            raise ValueError("logical_length must fit nonnegative signed int64")
        if isinstance(self.positions, torch.Tensor):
            raise ValueError("positions must be host integers, not a tensor")
        try:
            original = tuple(self.positions)
        except TypeError as error:
            raise ValueError("positions must be a sequence of host integers") from error
        if any(isinstance(p, bool) or not isinstance(p, Integral) for p in original):
            raise ValueError("positions must contain host integers")
        positions = tuple(int(p) for p in original)
        if any(p < 0 or p >= length for p in positions):
            raise ValueError("positions must lie within logical_length")
        if any(a >= b for a, b in zip(positions, positions[1:])):
            raise ValueError("positions must be strictly increasing and unique")
        full, total = [], []
        for row in range(0, len(positions), BLOCK_M):
            tile = positions[row:row + BLOCK_M]
            full.append((tile[0] + 1) // BLOCK_N)
            total.append((tile[-1] + BLOCK_N) // BLOCK_N)
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "logical_length", length)
        object.__setattr__(self, "full_prefix_tiles", tuple(full))
        object.__setattr__(self, "total_tiles", tuple(total))

    @property
    def selected_count(self) -> int:
        return len(self.positions)

    def to(self, device: torch.device | str | int) -> "DeviceIndexedPlan":
        """Upload O(B) metadata on the caller's current stream for this device."""
        if isinstance(device, int):
            device = torch.device("cuda", device)
        else:
            device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("device plan requires a CUDA device")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        # Use int64 bounds as well as addresses. No arithmetic is allowed to
        # wrap when a caller constructs a large logical-position plan.
        # Metadata needs version counters even if the caller uses inference
        # mode. This lets every launch reject ordinary in-place mutations with
        # CPU metadata checks rather than a device-to-host read.
        with torch.cuda.device(device), torch.inference_mode(False):
            positions = torch.tensor(self.positions, device=device, dtype=torch.int64)
            full = torch.tensor(self.full_prefix_tiles, device=device, dtype=torch.int64)
            total = torch.tensor(self.total_tiles, device=device, dtype=torch.int64)
        tensors = (positions, full, total)
        seal = (id(self), tuple(id(t) for t in tensors), tuple(t._version for t in tensors))
        return DeviceIndexedPlan(self, positions, full, total, seal)


@dataclass(frozen=True)
class DeviceIndexedPlan:
    """Device metadata from IndexedPlan.to; tensors are immutable by contract.

    Factory identity/version checks reject replacement or ordinary in-place
    mutation. Bypassing PyTorch versions via .data or external pointer writes is
    unsupported. Moving between streams requires the usual wait/record-stream
    lifetime handling, just like other CUDA input tensors.
    """

    host_plan: IndexedPlan
    positions: torch.Tensor
    full_prefix_tiles: torch.Tensor
    total_tiles: torch.Tensor
    _seal: tuple | None = field(default=None, repr=False, compare=False)

    @property
    def logical_length(self) -> int:
        return self.host_plan.logical_length

    @property
    def selected_count(self) -> int:
        return self.host_plan.selected_count

    @property
    def device(self) -> torch.device:
        return self.positions.device


def _validate_plan(plan, device):
    if not isinstance(plan, DeviceIndexedPlan) or not isinstance(plan.host_plan, IndexedPlan):
        raise ValueError("device_plan must be produced by IndexedPlan.to")
    tiles = (plan.selected_count + BLOCK_M - 1) // BLOCK_M
    for tensor, count in ((plan.positions, plan.selected_count),
                          (plan.full_prefix_tiles, tiles), (plan.total_tiles, tiles)):
        if (not isinstance(tensor, torch.Tensor) or tensor.device != device
                or not tensor.is_cuda or tensor.dtype != torch.int64
                or tensor.shape != (count,) or not tensor.is_contiguous()):
            raise ValueError("device plan has incompatible metadata shape, dtype, or device")
    tensors = (plan.positions, plan.full_prefix_tiles, plan.total_tiles)
    try:
        seal = (id(plan.host_plan), tuple(id(t) for t in tensors), tuple(t._version for t in tensors))
    except RuntimeError as error:
        raise ValueError("device plan must be produced by IndexedPlan.to") from error
    if plan._seal != seal:
        raise ValueError("device plan metadata was replaced, mutated, or not factory-created")


def _validate_tensors(tensors):
    if any(not isinstance(t, torch.Tensor) or t.ndim != 4 for t in tensors):
        raise ValueError("attention/cache tensors must have four dimensions")
    first = tensors[0]
    if not first.is_cuda or any(t.device != first.device for t in tensors):
        raise ValueError("all tensors must share one CUDA device")
    if first.dtype not in _DTYPES or any(t.dtype != first.dtype for t in tensors):
        raise ValueError("all tensors must share float16 or bfloat16 dtype")
    if any(t.shape[0] != 1 for t in tensors):
        raise ValueError("indexed kernels currently require batch size one")
    if any(t.shape[1] <= 0 or t.shape[3] not in (64, 128) for t in tensors):
        raise ValueError("head counts must be positive and head dimension must be 64 or 128")
    if any(any(s <= 0 for s in t.stride()) or not _nonoverlapping_layout(t) for t in tensors):
        raise ValueError("tensor layouts must have positive provably nonoverlapping strides")
    if any(t.requires_grad for t in tensors):
        raise ValueError("indexed kernels are inference-only; detach gradient-bearing inputs")


def indexed_kv_scatter_(out_k, out_v, fresh_k, fresh_v, device_plan):
    """Copy selected K/V rows exactly, preserving every other storage byte.

    Output K/V storage must be independent of each other and both fresh inputs.
    Source and destination token strides can differ; no contiguous conversion is
    performed. Valid unique positions guarantee one writer per output element.
    """
    tensors = (out_k, out_v, fresh_k, fresh_v)
    _validate_tensors(tensors)
    _validate_plan(device_plan, out_k.device)
    _, heads, _, dim = out_k.shape
    if any((t.shape[1], t.shape[3]) != (heads, dim) for t in tensors):
        raise ValueError("scatter K/V head geometry must match")
    if any(t.shape[2] != device_plan.logical_length for t in tensors[:2]):
        raise ValueError("output sequence length must match the plan")
    if any(t.shape[2] != device_plan.selected_count for t in tensors[2:]):
        raise ValueError("fresh sequence length must match selected_count")
    if not device_plan.selected_count:
        return
    storage = [t.untyped_storage().data_ptr() for t in tensors]
    if storage[0] == storage[1] or any(a == b for a in storage[:2] for b in storage[2:]):
        raise ValueError("output K/V storage must be independent of outputs and fresh sources")
    from ._indexed_triton import scatter

    with torch.cuda.device(out_k.device):
        scatter(out_k, out_v, fresh_k, fresh_v, device_plan.positions)


def indexed_causal_attention(q, k, v, device_plan, scale=None):
    """Attend packed queries to their entire absolute causal prefix.

    q=[1,Hq,B,D], k/v=[1,Hkv,N,D], result=[1,Hq,B,D]. Each selected
    row attends key positions <= its absolute position, including itself.
    K/V must already contain all fresh selected rows. The first implementation
    supports unpadded finite model-scale inputs, no dropout or backward. It uses
    native fp16/bf16 QK products, FP32 online-softmax/accumulation, and an explicit
    TF32x3 P@V product with FP32 probabilities and FP32-promoted V. This avoids
    rounding probabilities to the input dtype, but does not promise IEEE-exact
    FP32 arithmetic or bitwise equality with native SDPA. Attention requires
    NVIDIA SM80+; scatter has no additional architecture restriction. Explicit
    Triton TF32x3 is unaffected by PyTorch's allow_tf32 flags.
    """
    _validate_tensors((q, k, v))
    _validate_plan(device_plan, q.device)
    if k.shape != v.shape or q.shape[3] != k.shape[3]:
        raise ValueError("attention K/V shapes and Q/K/V head dimensions must match")
    if q.shape[1] % k.shape[1]:
        raise ValueError("query head count must be divisible by KV head count")
    if q.shape[2] != device_plan.selected_count or k.shape[2] != device_plan.logical_length:
        raise ValueError("attention sequence lengths must match the plan")
    if scale is None:
        scale = q.shape[-1] ** -0.5
    if isinstance(scale, bool) or not isinstance(scale, Real) or not math.isfinite(scale):
        raise ValueError("attention scale must be a finite host real number")
    _require_tf32x3_device(q.device)
    output = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    if not device_plan.selected_count:
        return output
    from ._indexed_triton import attention

    with torch.cuda.device(q.device):
        attention(q, k, v, output, device_plan, float(scale))
    return output
