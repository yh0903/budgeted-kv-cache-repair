"""One launch copies both K and V; each output element has one writer."""

import triton
import triton.language as tl


@triton.jit
def _copy_kv_interval(
    OUT_K, OUT_V, IN_K, IN_V,
    TOTAL: tl.constexpr, HEADS: tl.constexpr, WIDTH: tl.constexpr,
    DIM: tl.constexpr, OUT_START, IN_START,
    OK: tl.constexpr, OV: tl.constexpr, IK: tl.constexpr, IV: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Large H100 caches can have valid head offsets beyond signed int32 even
    # when the selected interval is tiny. Promote before offset arithmetic.
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    d = i % DIM
    p = (i // DIM) % WIDTH
    source_p = p + IN_START
    target_p = p + OUT_START
    h = (i // (DIM * WIDTH)) % HEADS
    b = i // (DIM * WIDTH * HEADS)
    k = tl.load(IN_K + b * IK[0] + h * IK[1] + source_p * IK[2] + d * IK[3], i < TOTAL)
    v = tl.load(IN_V + b * IV[0] + h * IV[1] + source_p * IV[2] + d * IV[3], i < TOTAL)
    tl.store(OUT_K + b * OK[0] + h * OK[1] + target_p * OK[2] + d * OK[3], k, i < TOTAL)
    tl.store(OUT_V + b * OV[0] + h * OV[1] + target_p * OV[2] + d * OV[3], v, i < TOTAL)


def copy_kv_interval(out_key, out_value, new_key, new_value, start, end):
    batch, heads, _, dim = out_key.shape
    width = end - start
    total = batch * heads * width * dim
    _copy_kv_interval[(triton.cdiv(total, 1024),)](
        out_key, out_value, new_key, new_value,
        total, heads, width, dim, start, start,
        out_key.stride(), out_value.stride(), new_key.stride(), new_value.stride(),
        1024, num_warps=4,
    )


def scatter_fresh_kv(out_key, out_value, new_key, new_value, start, end):
    """Private trusted launch: fresh rows [0:width) overwrite [start:end).

    The direct executor qualifies independently owned contiguous CUDA outputs
    once, restricts execution to inspected native model classes, and checks
    each projection's shape/dtype/device before calling this helper on the
    correct device. It does not construct destination interval views or repeat
    the public copy API's full validation. Source strides remain explicit.
    """
    batch, heads, _, dim = out_key.shape
    width = end - start
    total = batch * heads * width * dim
    _copy_kv_interval[(triton.cdiv(total, 1024),)](
        out_key, out_value, new_key, new_value,
        total, heads, width, dim, start, 0,
        out_key.stride(), out_value.stride(), new_key.stride(), new_value.stride(),
        1024, num_warps=4,
    )
