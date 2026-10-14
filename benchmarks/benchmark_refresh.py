#!/usr/bin/env python3
"""Benchmark refresh-only execution over fixed sparse supports on one GPU.

The fixture removes eight prompt tokens and applies positional correction before
timing. Each backend receives identical selected positions and the same borrowed
cache. Timings include the executor's output ownership, planning, mask creation,
and model traversal. Initial prefill, selection, positional correction, answer
decoding, and first compilation are excluded. These are synthetic execution
measurements, not task accuracy or end-to-end request speedups.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
from pathlib import Path
import random
import statistics
import time

import torch
from transformers import AutoModelForCausalLM

from kv_cache_repair.cache import OperatorShim, build_halo_cache, clone_cache
from kv_cache_repair.execution import oneshot_sparse_repair, sequential_repair
from kv_cache_repair.kernels.direct_cache import direct_interval_repair
from kv_cache_repair.kernels.packed_cache import packed_interval_repair


DEFAULT_BACKENDS = ("sequential", "direct", "oneshot", "packed_sdpa")


def selected_intervals(length: int, budget: int, count: int) -> list[tuple[int, int]]:
    """Distribute a fixed token count symmetrically around the context midpoint."""
    if length <= 0 or budget <= 0 or count <= 0 or budget % count:
        raise ValueError("length and budget must be positive; budget must be divisible by interval count")
    width = budget // count
    if count == 1:
        starts = [(length - width) // 2]
    else:
        starts = [round(length / 4 + i * length / (2 * (count - 1)) - width / 2) for i in range(count)]
    intervals = [(start, start + width) for start in starts]
    if intervals[0][0] < 0 or intervals[-1][1] > length:
        raise ValueError("selected support does not fit the sequence")
    if any(left[1] >= right[0] for left, right in zip(intervals, intervals[1:])):
        raise ValueError("intervals must be separated; increase length or reduce budget/count")
    return intervals


@torch.no_grad()
def deletion_fixture(model, length: int, device):
    """Produce a genuinely stale cache, rather than reusing clean-prefill state."""
    old_ids = torch.randint(4, model.config.vocab_size, (1, length + 8), device=device)
    delete_at = length // 8
    edited_ids = torch.cat((old_ids[:, :delete_at], old_ids[:, delete_at + 8:]), dim=1)
    shim = OperatorShim(model, None)
    old_cache = model(input_ids=old_ids, use_cache=True, return_dict=True).past_key_values
    shifted, alignment = build_halo_cache(shim, old_cache, old_ids, edited_ids, 0)
    return shim, shifted, edited_ids, alignment


def _bits_equal(left, right) -> bool:
    return (left.shape == right.shape and left.dtype == right.dtype
            and torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8)))


def _check_cache_ownership(actual, base, snapshot, positions):
    untouched = torch.ones(base.get_seq_length(), dtype=torch.bool, device=base[0][0].device)
    untouched[positions] = False
    if actual.get_seq_length() != base.get_seq_length() or len(actual) != len(base):
        raise AssertionError("executor changed the logical cache geometry")
    for layer in range(len(base)):
        for component in (0, 1):
            source = base[layer][component]
            result = actual[layer][component]
            if not _bits_equal(source, snapshot[layer][component]):
                raise AssertionError("executor mutated the borrowed input cache")
            if result.untyped_storage().data_ptr() == source.untyped_storage().data_ptr():
                raise AssertionError("nonempty repair must own its output storage")
            if not _bits_equal(result[:, :, untouched], source[:, :, untouched]):
                raise AssertionError("executor changed unselected cache rows")


def _difference(actual, reference, positions):
    max_absolute, max_relative_l2, all_bits = 0.0, 0.0, True
    index = torch.tensor(positions, device=actual[0][0].device)
    for layer in range(len(reference)):
        for component in (0, 1):
            result = actual[layer][component].index_select(2, index)
            expected = reference[layer][component].index_select(2, index)
            all_bits = all_bits and _bits_equal(result, expected)
            difference = result.double() - expected.double()
            max_absolute = max(max_absolute, difference.abs().max().item())
            max_relative_l2 = max(max_relative_l2, (difference.norm() / expected.double().norm().clamp_min(1e-12)).item())
    return {"bitwise_equal": all_bits, "max_abs": max_absolute, "max_layer_relative_l2": max_relative_l2}


def _summary(samples):
    ordered = sorted(samples)
    return {"raw_ms": samples, "median_ms": statistics.median(samples),
            "mean_ms": statistics.mean(samples), "p95_ms": ordered[math.ceil(0.95 * len(ordered)) - 1]}


def _time_calls(calls, device, warmups, repeats, seed):
    names = list(calls)
    for _ in range(warmups):
        for name in names:
            output = calls[name]()
            torch.cuda.synchronize(device)
            del output
    samples = {name: [] for name in names}
    rng, order = random.Random(seed), []
    for _ in range(repeats):
        round_order = names.copy()
        rng.shuffle(round_order)
        order.append(round_order)
        for name in round_order:
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            output = calls[name]()
            torch.cuda.synchronize(device)
            samples[name].append((time.perf_counter() - started) * 1000)
            del output
    return {name: _summary(values) for name, values in samples.items()}, order


@torch.no_grad()
def benchmark_case(model, length, interval_count, args):
    shim, base, ids, alignment = deletion_fixture(model, length, args.device)
    intervals = selected_intervals(length, args.token_budget, interval_count)
    positions = [position for start, end in intervals for position in range(start, end)]
    snapshot = clone_cache(base)
    calls = {
        "sequential": lambda: sequential_repair(shim, base, ids, intervals)[0],
        "direct": lambda: direct_interval_repair(shim, base, ids, intervals)[0],
        "oneshot": lambda: oneshot_sparse_repair(shim, base, ids, intervals)[0],
        "packed_sdpa": lambda: packed_interval_repair(shim, base, ids, intervals)[0],
    }
    calls = {name: calls[name] for name in args.backends}
    reference = sequential_repair(shim, base, ids, intervals)[0]
    for start, end in intervals:
        if not any(not _bits_equal(reference[layer][part][:, :, start:end], base[layer][part][:, :, start:end])
                   for layer in range(len(base)) for part in (0, 1)):
            raise AssertionError("degenerate fixture: every interval must change cached states")
    numerics = {}
    for name, call in calls.items():
        output = call()
        _check_cache_ownership(output, base, snapshot, positions)
        numerics[name] = _difference(output, reference, positions)
        if name in ("sequential", "direct") and not numerics[name]["bitwise_equal"]:
            raise AssertionError(f"{name}: same-schedule cache differs from the sequential reference")
        del output
    del reference
    timings, order = _time_calls(calls, args.device, args.warmups, args.repeats, args.seed)
    for layer in range(len(base)):
        for part in (0, 1):
            if not _bits_equal(base[layer][part], snapshot[layer][part]):
                raise AssertionError("timed execution mutated the input cache")
    return {"length": length, "token_budget": args.token_budget, "interval_count": interval_count,
            "intervals": intervals, "causal_query_key_pairs": sum(position + 1 for position in positions),
            "alignment": alignment, "cache_numerics_vs_sequential": numerics,
            "latency": timings, "measurement_order": order}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local checkpoint directory or Hugging Face model ID")
    parser.add_argument("--revision", default="main", help="Checkpoint revision; use a commit ID for repeatability")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--lengths", type=int, nargs="+", default=[2048, 8192])
    parser.add_argument("--interval-counts", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--token-budget", type=int, default=64)
    parser.add_argument("--backends", nargs="+", choices=DEFAULT_BACKENDS, default=list(DEFAULT_BACKENDS))
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--output", type=Path, help="Optional JSON output; raw measurements are also printed")
    args = parser.parse_args(argv)
    if args.warmups < 0 or args.repeats <= 0:
        parser.error("warmups must be nonnegative and repeats positive")
    for length in args.lengths:
        for count in args.interval_counts:
            try:
                selected_intervals(length, args.token_budget, count)
            except ValueError as error:
                parser.error(str(error))
    args.device = torch.device(args.device)
    if args.device.type != "cuda" or not torch.cuda.is_available():
        parser.error("timing requires a real CUDA GPU; use the tiny-model tests for CPU correctness checks")
    if args.device.index is None:
        args.device = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.set_device(args.device)
    torch.manual_seed(args.seed)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, local_files_only=args.local_files_only,
        trust_remote_code=False, torch_dtype=getattr(torch, args.dtype), attn_implementation="sdpa",
    ).to(args.device).eval()
    payload = {
        "scope": "refresh_only",
        "includes": ["owned cache initialization", "support planning/upload", "mask construction", "sparse model traversal"],
        "excludes": ["prefill", "selection", "positional correction", "answer decoding", "first compilation"],
        "workload": "synthetic deletion; fixed supports; batch one; one GPU",
        "quality_status": "cache numerical diagnostics only; not task-quality certification",
        "model": args.model, "revision": args.revision, "dtype": args.dtype,
        "device": torch.cuda.get_device_name(args.device), "cuda": torch.version.cuda,
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers")},
        "warmups": args.warmups, "repeats": args.repeats, "seed": args.seed, "results": [],
    }
    for length in args.lengths:
        for count in args.interval_counts:
            record = benchmark_case(model, length, count, args)
            payload["results"].append(record)
            print(json.dumps(record), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
