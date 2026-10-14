"""Experimental indexed scatter and absolute-causal online-softmax kernels.

No mask materialization, persistent work scheduling, split-K, or graph replay.
All data pointer arithmetic is signed int64, including strided head offsets.
The attention kernel is opt-in and not fully checkpoint-qualified; the public
packed executor defaults to SDPA instead.
"""

import triton
import triton.language as tl


@triton.jit
def _scatter(OK, OV, FK, FV, POS,
             HEADS: tl.constexpr, ROWS: tl.constexpr, N: tl.constexpr, DIM: tl.constexpr,
             SK: tl.constexpr, SV: tl.constexpr, TK: tl.constexpr, TV: tl.constexpr,
             BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    keep = i < HEADS * ROWS * DIM
    d = i % DIM
    row = i // DIM % ROWS
    head = i // (DIM * ROWS)
    position = tl.load(POS + row, keep, other=0)
    target_valid = keep & (position >= 0) & (position < N)
    key = tl.load(FK + head * TK[1] + row * TK[2] + d * TK[3], keep, other=0)
    value = tl.load(FV + head * TV[1] + row * TV[2] + d * TV[3], keep, other=0)
    tl.store(OK + head * SK[1] + position * SK[2] + d * SK[3], key, target_valid)
    tl.store(OV + head * SV[1] + position * SV[2] + d * SV[3], value, target_valid)


@triton.jit
def _blocks(q, q_positions, K, V, head, lo, hi, m, z, acc,
            N: tl.constexpr, DIM: tl.constexpr, KS: tl.constexpr, VS: tl.constexpr,
            SCALE_LOG2: tl.constexpr, MASK_CAUSAL: tl.constexpr,
            BM: tl.constexpr, BN: tl.constexpr):
    d = tl.arange(0, DIM).to(tl.int64)
    column = tl.arange(0, BN).to(tl.int64)
    for block in range(lo, hi):
        positions = block.to(tl.int64) * BN + column
        key = tl.load(K + head * KS[1] + d[:, None] * KS[3]
                      + positions[None, :] * KS[2], positions[None, :] < N, other=0)
        value = tl.load(V + head * VS[1] + positions[:, None] * VS[2]
                        + d[None, :] * VS[3], positions[:, None] < N, other=0)
        scores = tl.dot(q, key) * SCALE_LOG2
        if MASK_CAUSAL:
            allowed = (positions[None, :] < N) & (positions[None, :] <= q_positions[:, None])
            scores = tl.where(allowed, scores, float("-inf"))
        # Full-prefix blocks are entirely within N and <= every valid query.
        next_m = tl.maximum(m, tl.max(scores, 1))
        alpha = tl.exp2(m - next_m)
        probability = tl.exp2(scores - next_m[:, None])
        z = z * alpha + tl.sum(probability, 1)
        acc = acc * alpha[:, None]
        # Keep softmax probabilities in FP32. Rounding P to BF16 before P@V
        # can amplify cancellation for the real checkpoint's large V channels.
        # TF32x3 is explicit, independent of PyTorch's global TF32 flags; the
        # public attention wrapper qualifies NVIDIA SM80+ before this launch.
        acc = tl.dot(probability, value.to(tl.float32), acc, input_precision="tf32x3")
        m = next_m
    return m, z, acc


@triton.jit
def _attention(Q, K, V, OUTPUT, POS, FULL_TILES, TOTAL_TILES,
               ROWS: tl.constexpr, N: tl.constexpr, DIM: tl.constexpr,
               GROUP: tl.constexpr, QS: tl.constexpr, KS: tl.constexpr,
               VS: tl.constexpr, OS: tl.constexpr, SCALE_LOG2: tl.constexpr,
               BM: tl.constexpr, BN: tl.constexpr):
    tile = tl.program_id(0).to(tl.int64)
    query_head = tl.program_id(1).to(tl.int64)
    kv_head = query_head // GROUP
    row = tile * BM + tl.arange(0, BM).to(tl.int64)
    d = tl.arange(0, DIM).to(tl.int64)
    # Dummy rows use absolute position 0, keeping their softmax finite even in
    # a partially occupied query tile. They never store output or affect peers.
    positions = tl.load(POS + row, row < ROWS, other=0)
    q = tl.load(Q + query_head * QS[1] + row[:, None] * QS[2]
                + d[None, :] * QS[3], row[:, None] < ROWS, other=0)
    total = tl.minimum(tl.maximum(tl.load(TOTAL_TILES + tile), 0), tl.cdiv(N, BN))
    full = tl.minimum(tl.maximum(tl.load(FULL_TILES + tile), 0), total)
    m = tl.full((BM,), float("-inf"), tl.float32)
    z = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, DIM), tl.float32)
    m, z, acc = _blocks(q, positions, K, V, kv_head, 0, full, m, z, acc,
                        N, DIM, KS, VS, SCALE_LOG2, False, BM, BN)
    m, z, acc = _blocks(q, positions, K, V, kv_head, full, total, m, z, acc,
                        N, DIM, KS, VS, SCALE_LOG2, True, BM, BN)
    result = acc / z[:, None]
    tl.store(OUTPUT + query_head * OS[1] + row[:, None] * OS[2]
             + d[None, :] * OS[3], result, row[:, None] < ROWS)


def scatter(out_k, out_v, fresh_k, fresh_v, positions):
    _, heads, rows, dim = fresh_k.shape
    _scatter[(triton.cdiv(heads * rows * dim, 1024),)](
        out_k, out_v, fresh_k, fresh_v, positions,
        heads, rows, out_k.shape[2], dim,
        out_k.stride(), out_v.stride(), fresh_k.stride(), fresh_v.stride(),
        1024, num_warps=4,
    )


def attention(q, k, v, output, plan, scale):
    _, heads, rows, dim = q.shape
    _attention[(triton.cdiv(rows, 16), heads)](
        q, k, v, output, plan.positions, plan.full_prefix_tiles, plan.total_tiles,
        rows, k.shape[2], dim, heads // k.shape[1],
        q.stride(), k.stride(), v.stride(), output.stride(), scale * 1.4426950408889634,
        16, 64, num_warps=4, num_stages=2,
    )
