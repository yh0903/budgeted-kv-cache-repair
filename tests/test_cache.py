import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.cache_utils import DynamicCache

from kv_cache_repair.alignment import token_alignment
from kv_cache_repair.cache import (
    OperatorShim,
    build_halo_cache,
    clone_cache,
    repair_window,
    rotate_half,
    run_tokens_from_prefix,
    shifted_suffix_keys,
)
from kv_cache_repair.execution import oneshot_sparse_repair, positions_to_intervals, sequential_repair


@pytest.fixture
def deletion_case():
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        attention_dropout=0.0,
    )
    with torch.random.fork_rng():
        torch.manual_seed(3)
        model = LlamaForCausalLM(config).eval()
    model.set_attn_implementation("eager")
    shim = OperatorShim(model)
    old_ids = torch.tensor([[1, 2, 9, 10, 3, 4, 5, 6]])
    edited_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    old = run_tokens_from_prefix(shim, DynamicCache(), old_ids, old_ids.shape[1])
    clean = run_tokens_from_prefix(shim, DynamicCache(), edited_ids, edited_ids.shape[1])
    return shim, old, old_ids, edited_ids, clean


@pytest.mark.parametrize(
    "old, new, expected",
    [
        ([1, 2, 9, 3, 4], [1, 2, 3, 4], (2, 2, 3, 2, 0, -1)),
        ([1, 2], [1, 2], (2, 0, 2, 2, 0, 0)),
        ([1, 2], [], (0, 0, 2, 0, 0, -2)),
        ([1, 2, 3], [1, 8, 9, 3], (1, 1, 2, 3, 2, 1)),
    ],
)
def test_alignment_finds_nonoverlapping_common_prefix_and_suffix(old, new, expected):
    result = token_alignment(old, new)

    assert (
        tuple(
            result[key]
            for key in (
                "prefix",
                "suffix",
                "old_suffix_start",
                "new_suffix_start",
                "mandatory_changed_tokens",
                "position_shift",
            )
        )
        == expected
    )


def test_clone_cache_prefix_has_independent_storage(deletion_case):
    _, old, _, _, _ = deletion_case
    snapshot = old[0][0].clone()

    cloned = clone_cache(old, 3)
    cloned[0][0].zero_()

    assert cloned.get_seq_length() == 3
    torch.testing.assert_close(old[0][0], snapshot, rtol=0, atol=0)
    assert old[0][0].data_ptr() != cloned[0][0].data_ptr()


@pytest.mark.parametrize("length", [-1, 9])
def test_clone_cache_rejects_out_of_range_prefix(deletion_case, length):
    with pytest.raises(ValueError, match="prefix length"):
        clone_cache(deletion_case[1], length)


def test_rotate_half_uses_split_half_convention():
    assert rotate_half(torch.tensor([1.0, 2.0, 3.0, 4.0])).tolist() == [-3.0, -4.0, 1.0, 2.0]


def test_zero_shift_clones_instead_of_aliasing(deletion_case):
    shim, old, _, _, _ = deletion_case
    key = old[0][0]

    shifted = shifted_suffix_keys(shim, key, 0)

    torch.testing.assert_close(shifted, key, rtol=0, atol=0)
    assert shifted.data_ptr() != key.data_ptr()


def test_shift_removes_span_preserves_values_and_corrects_first_layer_positions(deletion_case):
    shim, old, old_ids, edited_ids, clean = deletion_case
    before = clone_cache(old)

    shifted, metadata = build_halo_cache(shim, old, old_ids, edited_ids, 0)

    assert shifted.get_seq_length() == 6
    assert metadata["recomputed_tokens"] == 0
    assert metadata["position_shift"] == -2
    for layer in range(len(old)):
        for kind in (0, 1):
            torch.testing.assert_close(
                shifted[layer][kind][:, :, :2], old[layer][kind][:, :, :2], rtol=0, atol=0
            )
            torch.testing.assert_close(old[layer][kind], before[layer][kind], rtol=0, atol=0)
        torch.testing.assert_close(shifted[layer][1][:, :, 2:], old[layer][1][:, :, 4:], rtol=0, atol=0)
    torch.testing.assert_close(shifted[0][0], clean[0][0], rtol=1e-5, atol=1e-6)


def test_replay_matches_clean_cache_without_mutating_old_cache(deletion_case):
    shim, old, old_ids, edited_ids, clean = deletion_case
    before = clone_cache(old)

    replay, metadata = build_halo_cache(shim, old, old_ids, edited_ids, 4)

    assert metadata["recomputed_tokens"] == 4
    for layer in range(len(old)):
        for kind in (0, 1):
            torch.testing.assert_close(replay[layer][kind], clean[layer][kind], rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(old[layer][kind], before[layer][kind], rtol=0, atol=0)


def test_repair_window_changes_only_selected_rows(deletion_case):
    shim, old, old_ids, edited_ids, clean = deletion_case
    shifted, _ = build_halo_cache(shim, old, old_ids, edited_ids, 0)
    before = clone_cache(shifted)

    repaired = repair_window(shim, shifted, edited_ids, 2, 4)

    for layer in range(len(shifted)):
        for kind in (0, 1):
            torch.testing.assert_close(
                repaired[layer][kind][:, :, 2:4], clean[layer][kind][:, :, 2:4], rtol=1e-5, atol=1e-6
            )
            for region in (slice(0, 2), slice(4, None)):
                torch.testing.assert_close(
                    repaired[layer][kind][:, :, region], before[layer][kind][:, :, region], rtol=0, atol=0
                )
            torch.testing.assert_close(shifted[layer][kind], before[layer][kind], rtol=0, atol=0)


@pytest.mark.parametrize("start, end", [(-1, 3), (4, 3), (2, 7)])
def test_repair_window_rejects_invalid_interval(deletion_case, start, end):
    shim, old, _, edited_ids, _ = deletion_case
    with pytest.raises(ValueError, match="repair interval"):
        repair_window(shim, old, edited_ids, start, end)


def test_positions_are_sorted_deduplicated_and_coalesced():
    assert positions_to_intervals([5, 2, 3, 2, 8, 6]) == [(2, 4), (5, 7), (8, 9)]


def test_reference_schedules_match_on_tiny_fragmented_support(deletion_case):
    shim, old, old_ids, edited_ids, _ = deletion_case
    shifted, _ = build_halo_cache(shim, old, old_ids, edited_ids, 0)
    intervals = [(2, 3), (4, 6)]

    sequential, seq_metadata = sequential_repair(shim, shifted, edited_ids, intervals)
    oneshot, one_metadata = oneshot_sparse_repair(shim, shifted, edited_ids, intervals)

    assert seq_metadata["recomputed_tokens"] == one_metadata["recomputed_tokens"] == 3
    for layer in range(len(shifted)):
        for kind in (0, 1):
            torch.testing.assert_close(oneshot[layer][kind], sequential[layer][kind], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("executor", [sequential_repair, oneshot_sparse_repair])
def test_reference_schedule_rejects_unsorted_intervals(deletion_case, executor):
    shim, old, _, edited_ids, _ = deletion_case
    with pytest.raises(ValueError, match="sorted, disjoint"):
        executor(shim, old, edited_ids, [(4, 5), (2, 3)])
