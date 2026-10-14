"""Offline repair-API example with random weights; not a task-quality benchmark."""

from __future__ import annotations

import argparse
import math

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from kv_cache_repair import repair_after_deletion


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget-percent", type=float, default=5.0)
    parser.add_argument("--executor", choices=("packed", "sequential"), default="packed")
    args = parser.parse_args()
    if not math.isfinite(args.budget_percent) or not 0 <= args.budget_percent <= 100:
        parser.error("--budget-percent must be between 0 and 100")

    original_text = (
        "Project notes. The obsolete schedule starts Friday. "
        "The revised schedule starts Monday. The location is the east office. "
        "The project owner will send the agenda tomorrow."
    )
    removed_text = "The obsolete schedule starts Friday. "
    query = "When does the revised schedule start?"
    pre_tokenizer = Whitespace()
    words = sorted({word for word, _ in pre_tokenizer.pre_tokenize_str(original_text + " " + query)})
    vocabulary = {"[UNK]": 0, "[PAD]": 1, **{word: index + 2 for index, word in enumerate(words)}}
    backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizer
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]")

    torch.manual_seed(7)
    config = LlamaConfig(
        vocab_size=len(vocabulary),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        attention_dropout=0.0,
        pad_token_id=1,
        bos_token_id=None,
        eos_token_id=None,
    )
    model = LlamaForCausalLM(config).eval()
    model.set_attn_implementation("sdpa")
    original_ids = tokenizer(original_text, add_special_tokens=False, return_tensors="pt")["input_ids"]
    old_cache = model(input_ids=original_ids, use_cache=True).past_key_values
    start = original_text.index(removed_text)
    result = repair_after_deletion(
        model,
        tokenizer,
        old_cache,
        original_text,
        delete_span=(start, start + len(removed_text)),
        query=query,
        budget_percent=args.budget_percent,
        executor=args.executor,
    )
    print(f"Executor: {args.executor}")
    print(f"Prompt tokens: {original_ids.shape[1]} -> {result.input_ids.shape[1]}")
    print(f"Suffix rows selected: {result.selection.budget_tokens}/{result.selection.suffix_tokens}")
    print(f"Selected intervals: {result.selection.intervals}")
    print(f"Repaired cache length: {result.cache.get_seq_length()}")
    print("Random-weight CPU smoke test only; no answer-quality or GPU-speed claim.")


if __name__ == "__main__":
    main()
