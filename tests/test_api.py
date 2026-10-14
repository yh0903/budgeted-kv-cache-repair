"""Offline public-API behavior on small native models and an in-memory tokenizer."""

import math

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
    Qwen3Config,
    Qwen3ForCausalLM,
)

from kv_cache_repair import repair_after_deletion, select_support
from kv_cache_repair.cache import clone_cache


@pytest.fixture
def tokenizer():
    words = [
        "[UNK]",
        "prefix",
        "obsolete",
        "discard",
        "alpha",
        "beta",
        "gamma",
        ".",
        "red",
        "blue",
        "redblue",
    ]
    backend = Tokenizer(WordLevel({word: index for index, word in enumerate(words)}, unk_token="[UNK]"))
    backend.pre_tokenizer = WhitespaceSplit()
    return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")


@pytest.fixture(params=["llama", "qwen3"])
def model(request):
    config_args = dict(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=256,
        attention_dropout=0.0,
    )
    with torch.random.fork_rng():
        torch.manual_seed(23)
        if request.param == "llama":
            instance = LlamaForCausalLM(LlamaConfig(**config_args))
        else:
            instance = Qwen3ForCausalLM(Qwen3Config(**config_args, layer_types=["full_attention"] * 2))
    instance.eval()
    instance.set_attn_implementation("eager")
    return instance


def _prefill(model, tokenizer, text):
    ids = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        return model.base_model(input_ids=ids, use_cache=True).past_key_values


def _assert_cache_equal(actual, expected, *, exact=False):
    assert len(actual) == len(expected)
    assert actual.get_seq_length() == expected.get_seq_length()
    for layer in range(len(actual)):
        for component in (0, 1):
            torch.testing.assert_close(
                actual[layer][component],
                expected[layer][component],
                rtol=0.0 if exact else 1e-5,
                atol=0.0 if exact else 2e-6,
            )


@pytest.fixture
def deletion_request(model, tokenizer):
    prefix = "prefix "
    obsolete = "obsolete discard "
    suffix = "alpha beta gamma . " * 25
    original = prefix + obsolete + suffix
    return dict(
        model=model,
        tokenizer=tokenizer,
        old_cache=_prefill(model, tokenizer, original),
        original_text=original,
        delete_span=(len(prefix), len(prefix) + len(obsolete)),
        query="alpha",
    )


@pytest.mark.parametrize("budget, expected", [(0, 0), (2, 2), (5, 5), (100, 100)])
def test_select_support_obeys_suffix_budget_without_model_forward(model, tokenizer, budget, expected):
    text = "prefix " + "alpha beta gamma . " * 25

    def reject_forward(module, args):
        raise AssertionError("selection must not run a transformer forward")

    handles = [module.register_forward_pre_hook(reject_forward) for module in (model, model.base_model)]
    try:
        result = select_support(model, tokenizer, text, "alpha", suffix_start=1, budget_percent=budget)
    finally:
        for handle in handles:
            handle.remove()

    assert result.suffix_start == 1
    assert result.suffix_tokens == 100
    assert result.budget_tokens == expected
    assert len(result.positions) == len(set(result.positions)) == expected
    assert list(result.positions) == sorted(result.positions)
    assert [position for start, end in result.intervals for position in range(start, end)] == list(
        result.positions
    )
    assert all(1 <= position < 101 for position in result.positions)
    if budget == 0:
        assert result.positions == result.intervals == ()
        assert result.metadata["selection"] == "shift_only"
    elif budget == 100:
        assert result.positions == tuple(range(1, 101))
        assert result.metadata["selection"] == "full_suffix"
    else:
        assert result.metadata["selection"] == "sur"
        assert result.metadata["scoring"]["exact_token_boost"] == 0.25


@pytest.mark.parametrize("budget", [2, 5])
def test_small_suffix_uses_ceiling_not_nearest_integer(model, tokenizer, budget):
    result = select_support(
        model,
        tokenizer,
        "prefix alpha beta gamma",
        "alpha",
        suffix_start=1,
        budget_percent=budget,
    )

    assert result.suffix_tokens == 3
    assert result.budget_tokens == 1
    assert len(result.positions) == 1
    assert 1 <= result.positions[0] < 4


@pytest.mark.parametrize("budget", [0, 2, 5, 100])
def test_repair_packed_and_sequential_agree_and_preserve_old_cache(deletion_request, budget):
    old_cache = deletion_request["old_cache"]
    old_snapshot = clone_cache(old_cache)

    sequential = repair_after_deletion(**deletion_request, budget_percent=budget, executor="sequential")
    packed = repair_after_deletion(**deletion_request, budget_percent=budget, executor="packed")

    assert packed.edited_text == "prefix " + "alpha beta gamma . " * 25
    assert packed.input_ids.shape == (1, 101)
    assert packed.selection.positions == sequential.selection.positions
    assert packed.selection.budget_tokens == budget
    assert packed.metadata["alignment"]["mandatory_changed_tokens"] == 0
    assert packed.metadata["alignment"]["position_shift"] == -2
    _assert_cache_equal(packed.cache, sequential.cache)
    _assert_cache_equal(old_cache, old_snapshot, exact=True)
    for result in (packed, sequential):
        for layer in range(len(old_cache)):
            for component in (0, 1):
                assert result.cache[layer][component].data_ptr() != old_cache[layer][component].data_ptr()


@pytest.mark.parametrize("executor", ["sequential", "packed"])
def test_full_suffix_repair_matches_clean_prefill(deletion_request, executor):
    result = repair_after_deletion(**deletion_request, budget_percent=100, executor=executor)
    clean = _prefill(deletion_request["model"], deletion_request["tokenizer"], result.edited_text)

    assert result.selection.positions == tuple(range(1, 101))
    _assert_cache_equal(result.cache, clean)


@pytest.mark.parametrize("delete_span, edited_text", [((7, 21), "prefix "), ((0, 21), "")])
@pytest.mark.parametrize("executor", ["sequential", "packed"])
def test_trailing_and_full_deletions_have_empty_suffix(model, tokenizer, delete_span, edited_text, executor):
    original = "prefix obsolete alpha"
    old = _prefill(model, tokenizer, original)
    snapshot = clone_cache(old)

    result = repair_after_deletion(
        model,
        tokenizer,
        old,
        original,
        delete_span=delete_span,
        query="",
        budget_percent=5,
        executor=executor,
    )

    assert result.edited_text == edited_text
    assert result.selection.suffix_tokens == result.selection.budget_tokens == 0
    assert result.selection.positions == result.selection.intervals == ()
    assert result.cache.get_seq_length() == (1 if edited_text else 0)
    _assert_cache_equal(old, snapshot, exact=True)
    for layer in range(len(old)):
        for component in (0, 1):
            torch.testing.assert_close(
                result.cache[layer][component],
                old[layer][component][:, :, : result.cache.get_seq_length()],
                rtol=0,
                atol=0,
            )


@pytest.mark.parametrize("executor", ["sequential", "packed"])
def test_boundary_retokenization_is_recomputed_outside_suffix_budget(model, tokenizer, executor):
    original = "prefix red blue alpha beta"
    join_position = original.index(" blue")
    old = _prefill(model, tokenizer, original)

    result = repair_after_deletion(
        model,
        tokenizer,
        old,
        original,
        delete_span=(join_position, join_position + 1),
        query="alpha",
        budget_percent=0,
        executor=executor,
    )
    clean = _prefill(model, tokenizer, result.edited_text)

    assert result.edited_text == "prefix redblue alpha beta"
    assert result.input_ids.tolist() == [[1, 10, 4, 5]]
    assert result.selection.suffix_start == 2
    assert result.selection.suffix_tokens == 2
    assert result.selection.budget_tokens == 0
    assert result.metadata["alignment"]["mandatory_changed_tokens"] == 1
    assert result.metadata["alignment"]["recomputed_tokens"] == 1
    for layer in range(len(old)):
        for component in (0, 1):
            torch.testing.assert_close(
                result.cache[layer][component][:, :, :2],
                clean[layer][component][:, :, :2],
                rtol=1e-5,
                atol=2e-6,
            )


@pytest.mark.parametrize("budget", [-1, 101, math.nan, math.inf, True])
def test_select_support_rejects_invalid_budget_before_model_access(budget):
    with pytest.raises(ValueError, match="budget_percent"):
        select_support(None, None, "", "", suffix_start=0, budget_percent=budget)


@pytest.mark.parametrize("delete_span", [None, 7, (0, 0), (-1, 2), (0, 99), (True, 2), (0, 1.5), (0,)])
def test_repair_rejects_invalid_deletion_span_before_model_access(delete_span):
    with pytest.raises(ValueError, match="delete_span"):
        repair_after_deletion(None, None, None, "alpha", delete_span=delete_span, query="alpha")


def test_repair_rejects_unknown_executor_before_model_access():
    with pytest.raises(ValueError, match="executor"):
        repair_after_deletion(
            None, None, None, "alpha", delete_span=(0, 1), query="alpha", executor="unknown"
        )


@pytest.mark.parametrize("suffix_start", [-1, 2, True, 0.5])
def test_select_support_rejects_invalid_suffix_boundary(model, tokenizer, suffix_start):
    with pytest.raises(ValueError, match="suffix_start"):
        select_support(model, tokenizer, "alpha", "alpha", suffix_start=suffix_start)


def test_repair_rejects_training_mode(deletion_request):
    deletion_request["model"].train()
    with pytest.raises(ValueError, match="model.eval"):
        repair_after_deletion(**deletion_request)


def test_repair_rejects_cache_prompt_length_mismatch(deletion_request):
    deletion_request["old_cache"] = clone_cache(deletion_request["old_cache"], 5)
    with pytest.raises(ValueError, match="old_cache shape"):
        repair_after_deletion(**deletion_request)


@pytest.mark.parametrize("executor", ["sequential", "packed"])
def test_zero_budget_rejects_overridden_forward_before_boundary_recomputation(
    model, tokenizer, executor, monkeypatch
):
    original = "prefix red blue alpha beta"
    join_position = original.index(" blue")
    old = _prefill(model, tokenizer, original)
    snapshot = clone_cache(old)

    def forbidden_forward(*args, **kwargs):
        raise AssertionError("unsupported model must be rejected before boundary recomputation")

    monkeypatch.setattr(model.base_model, "forward", forbidden_forward)

    with pytest.raises(ValueError, match="per-instance forward overrides"):
        repair_after_deletion(
            model,
            tokenizer,
            old,
            original,
            delete_span=(join_position, join_position + 1),
            query="alpha",
            budget_percent=0,
            executor=executor,
        )
    _assert_cache_equal(old, snapshot, exact=True)


@pytest.mark.parametrize(
    "flag, expected_error",
    [
        ("quantized", "quantized models"),
        ("runtime_rope", "default fixed RoPE"),
        ("runtime_sliding", "full attention"),
    ],
)
def test_repair_rejects_unsupported_runtime_flags_before_forward(
    deletion_request, monkeypatch, flag, expected_error
):
    model = deletion_request["model"]
    if flag == "quantized":
        monkeypatch.setattr(model, "is_quantized", True, raising=False)
    elif flag == "runtime_rope":
        monkeypatch.setattr(model.base_model.rotary_emb, "rope_type", "dynamic")
    else:
        monkeypatch.setattr(model.base_model, "has_sliding_layers", True, raising=False)

    def forbidden_forward(module, args):
        raise AssertionError("unsupported runtime must be rejected before a transformer forward")

    handle = model.base_model.register_forward_pre_hook(forbidden_forward)
    try:
        with pytest.raises(ValueError, match=expected_error):
            repair_after_deletion(**deletion_request, budget_percent=0)
    finally:
        handle.remove()
