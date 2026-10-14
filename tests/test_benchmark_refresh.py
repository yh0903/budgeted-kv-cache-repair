"""The public timing fixture fixes work and rejects meaningless measurements."""

from pathlib import Path
import runpy

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from kv_cache_repair.cache import clone_cache
from kv_cache_repair.execution import sequential_repair


BENCHMARK = runpy.run_path(str(Path(__file__).resolve().parents[1] / "benchmarks" / "benchmark_refresh.py"))


@pytest.mark.parametrize("length", [2048, 8192])
def test_geometry_preserves_selected_work_and_causal_pairs(length):
    pair_counts = []
    for count in (1, 4, 16):
        intervals = BENCHMARK["selected_intervals"](length, 64, count)
        assert len(intervals) == count
        positions = [position for start, end in intervals for position in range(start, end)]
        assert len(positions) == len(set(positions)) == 64
        assert 0 <= min(positions) <= max(positions) < length
        pair_counts.append(sum(position + 1 for position in positions))
    assert pair_counts == [pair_counts[0]] * 3


@pytest.mark.parametrize("length,budget,count", [(0, 64, 1), (128, 0, 1), (128, 64, 0), (128, 5, 2), (8, 64, 1), (64, 64, 16)])
def test_geometry_rejects_empty_overlapping_or_out_of_bounds_support(length, budget, count):
    with pytest.raises(ValueError):
        BENCHMARK["selected_intervals"](length, budget, count)


def test_deletion_fixture_requires_nontrivial_refresh():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(71)
        model = LlamaForCausalLM(LlamaConfig(
            vocab_size=97, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            head_dim=8, max_position_embeddings=128, attention_dropout=0.0,
        )).eval()
        model.set_attn_implementation("sdpa")
        shim, base, ids, alignment = BENCHMARK["deletion_fixture"](model, 64, "cpu")
        snapshot = clone_cache(base)
        intervals = BENCHMARK["selected_intervals"](64, 8, 4)
        positions = [position for start, end in intervals for position in range(start, end)]
        assert ids.shape == (1, 64)
        assert alignment["position_shift"] == -8
        with torch.no_grad():
            repaired, _ = sequential_repair(shim, base, ids, intervals)
        BENCHMARK["_check_cache_ownership"](repaired, base, snapshot, positions)
        for start, end in intervals:
            assert any(not BENCHMARK["_bits_equal"](repaired[layer][part][:, :, start:end], base[layer][part][:, :, start:end])
                       for layer in range(len(base)) for part in (0, 1))
    finally:
        torch.set_num_threads(previous)


def test_latency_summary_retains_raw_measurements():
    result = BENCHMARK["_summary"]([3.0, 1.0, 2.0])
    assert result == {"raw_ms": [3.0, 1.0, 2.0], "median_ms": 2.0, "mean_ms": 2.0, "p95_ms": 3.0}


def test_help_does_not_load_a_model(monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("--help must not load or download model weights")
    monkeypatch.setattr(BENCHMARK["AutoModelForCausalLM"], "from_pretrained", forbidden)
    with pytest.raises(SystemExit) as exit_info:
        BENCHMARK["main"](["--help"])
    assert exit_info.value.code == 0
    assert "refresh-only" in capsys.readouterr().out
