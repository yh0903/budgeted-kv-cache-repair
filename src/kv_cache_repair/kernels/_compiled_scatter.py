"""Bounded compiled launch plans for the qualified direct-cache scatter path.

This is pinned to Triton 3.5.1's CompiledKernel[grid] runner. That runner obtains
the current stream and calls the live launch hooks on every invocation. Plans
retain code and immutable specialization metadata only, never tensor arguments,
pointer values, outputs, streams, or previously computed results.
"""

from collections import OrderedDict
from dataclasses import dataclass
from threading import Lock
from typing import Callable

import triton
import triton.language as tl

from ._writeback_triton import scatter_fresh_kv as _ordinary_scatter


_PINNED_TRITON = "3.5.1"
_MAX_RUNNERS = 128
_RUNNERS = OrderedDict()
_CACHE_LOCK = Lock()
_HITS = 0
_MISSES = 0
_FALLBACKS = 0


if triton.__version__ == _PINNED_TRITON:
    from triton import knobs

    @triton.jit(
        do_not_specialize=["OUT_START"],
        do_not_specialize_on_alignment=["OUT_K", "OUT_V", "IN_K", "IN_V", "OUT_START"],
    )
    def _scatter_kernel(
        OUT_K, OUT_V, IN_K, IN_V,
        TOTAL: tl.constexpr, HEADS: tl.constexpr, WIDTH: tl.constexpr,
        DIM: tl.constexpr, OUT_START: tl.int64,
        OK: tl.constexpr, OV: tl.constexpr, IK: tl.constexpr, IV: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        # Explicit int64 arithmetic and annotation avoid both offset overflow
        # and the JIT's special handling of integer value 1. Pointer alignment
        # is not specialized, so same-stride offset views can reuse this code.
        i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
        d = i % DIM
        source_p = (i // DIM) % WIDTH
        target_p = source_p + OUT_START
        h = (i // (DIM * WIDTH)) % HEADS
        b = i // (DIM * WIDTH * HEADS)
        k = tl.load(IN_K + b * IK[0] + h * IK[1] + source_p * IK[2] + d * IK[3], i < TOTAL)
        v = tl.load(IN_V + b * IV[0] + h * IV[1] + source_p * IV[2] + d * IV[3], i < TOTAL)
        tl.store(OUT_K + b * OK[0] + h * OK[1] + target_p * OK[2] + d * OK[3], k, i < TOTAL)
        tl.store(OUT_V + b * OV[0] + h * OV[1] + target_p * OV[2] + d * OV[3], v, i < TOTAL)
else:
    knobs = None
    _scatter_kernel = None


@dataclass(frozen=True)
class _LaunchPlan:
    runner: Callable
    signature: tuple


def _clear_runner_cache():
    """Clear our bounded metadata cache (not Triton's own compiler cache)."""
    global _HITS, _MISSES, _FALLBACKS
    with _CACHE_LOCK:
        _RUNNERS.clear()
        _HITS = _MISSES = _FALLBACKS = 0


def _runner_cache_info():
    with _CACHE_LOCK:
        return {
            "size": len(_RUNNERS),
            "max_size": _MAX_RUNNERS,
            "hits": _HITS,
            "misses": _MISSES,
            "fallbacks": _FALLBACKS,
            "keys": tuple(_RUNNERS),
            "signatures": tuple(plan.signature for plan in _RUNNERS.values()),
        }


def _cache_compatible():
    # The compiled runner preserves launch hooks, but intentionally bypasses
    # JIT.run. Use ordinary JIT whenever its pre-run/debug/compilation behavior
    # is requested. Interpreter mode can also replace the JITFunction entirely.
    return (
        triton.__version__ == _PINNED_TRITON
        and _scatter_kernel is not None
        and not knobs.runtime.interpret
        and not knobs.runtime.debug
        and not getattr(_scatter_kernel, "debug", False)
        and not getattr(_scatter_kernel, "pre_run_hooks", ())
        and knobs.runtime.jit_cache_hook is None
        and knobs.runtime.jit_post_compile_hook is None
        and not knobs.compilation.always_compile
        and not getattr(_scatter_kernel, "used_global_vals", {})
    )


def scatter_fresh_kv(out_key, out_value, new_key, new_value, start, end):
    """Scatter using current tensor arguments and the caller's CUDA device.

    Preconditions are supplied by the direct adapter: independently owned
    contiguous CUDA outputs, valid interval bounds, and native fresh projections
    with matching shape/dtype/device. The caller retains its device context.
    """
    global _HITS, _MISSES, _FALLBACKS
    if triton.__version__ != _PINNED_TRITON or _scatter_kernel is None:
        _FALLBACKS += 1
        return _ordinary_scatter(out_key, out_value, new_key, new_value, start, end)

    batch, heads, _, dim = out_key.shape
    width = end - start
    total = batch * heads * width * dim
    out_key_stride, out_value_stride = out_key.stride(), out_value.stride()
    new_key_stride, new_value_stride = new_key.stride(), new_value.stride()
    grid = (triton.cdiv(total, 1024), 1, 1)
    args = (
        out_key, out_value, new_key, new_value,
        total, heads, width, dim, start,
        out_key_stride, out_value_stride, new_key_stride, new_value_stride,
        1024,
    )
    if not _cache_compatible():
        _FALLBACKS += 1
        # Use this same JITFunction so its pre-run hooks and global checks are
        # preserved, rather than switching to a different uncached kernel.
        return _scatter_kernel[grid](*args, num_warps=4)
    key = (
        out_key.device.index,
        out_key.dtype, out_value.dtype, new_key.dtype, new_value.dtype,
        batch, heads, width, dim,
        out_key_stride, out_value_stride, new_key_stride, new_value_stride,
    )
    with _CACHE_LOCK:
        plan = _RUNNERS.get(key)
        if plan is not None:
            _RUNNERS.move_to_end(key)
            _HITS += 1
        else:
            _MISSES += 1
    if plan is None:
        # warmup compiles without touching data. The CompiledKernel runner
        # captures the executable and grid, while each call supplies fresh args.
        compiled = _scatter_kernel.warmup(*args, grid=grid, num_warps=4)
        if compiled is None:
            raise RuntimeError("Triton did not return a compiled scatter executable")
        if compiled.src.signature["OUT_START"] != "i64":
            raise RuntimeError("compiled scatter requires an int64 OUT_START ABI")
        if not _cache_compatible():
            # Compilation may discover a global dependency. Preserve JIT.run's
            # global-value checks rather than caching a runner for such a kernel.
            _FALLBACKS += 1
            return _scatter_kernel[grid](*args, num_warps=4)
        plan = _LaunchPlan(compiled[grid], tuple(compiled.src.signature.items()))
        with _CACHE_LOCK:
            _RUNNERS[key] = plan
            _RUNNERS.move_to_end(key)
            while len(_RUNNERS) > _MAX_RUNNERS:
                _RUNNERS.popitem(last=False)
    # CompiledKernel.__getitem__ in 3.5.1 dynamically obtains current stream and
    # launch hooks. Do not replace it with a cached stream or raw driver call.
    plan.runner(*args)
