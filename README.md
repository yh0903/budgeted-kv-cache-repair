# Budgeted KV-Cache Repair for Evolving Contexts

Code for [Budgeted KV-Cache Repair for Evolving Contexts](https://arxiv.org/abs/2610.17338).

Deleting text from a prefetched context leaves two repair tasks: correcting the retained suffix's rotary positions and refreshing states that still depend on the deleted span. **Semantic-Unit Refresh (SUR)** allocates a user-specified suffix-token budget to query-relevant regions. The packed executor processes selected rows together at each layer while keeping their full causal history visible.

This repository contains the allocation and execution code, simple allocation controls, offline tests, and a refresh-stage benchmark. It does not bundle model weights, benchmark datasets, experiment outputs, or the manuscript. It is not a complete reproduction harness for every experiment or adapted baseline in the paper.

## Install

Use Python 3.11 or newer. The reference implementation targets **PyTorch 2.9.1 and Transformers 4.57.6**; cache and attention APIs are version-sensitive.

```bash
git clone https://github.com/yh0903/budgeted-kv-cache-repair.git
cd budgeted-kv-cache-repair
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

Use a CUDA-enabled PyTorch installation for GPU execution. CPU execution is available for small correctness tests, not performance measurements.

## Quick start

The following example creates a tiny random model and tokenizer locally. It needs no model download and tests the repair API, not answer accuracy:

```bash
python examples/minimal_repair.py --budget-percent 5 --executor packed
```

For a supported pretrained model, the same interface accepts its existing prompt cache:

```python
import torch
from kv_cache_repair import repair_after_deletion

# model: an eval-mode native dense Llama or Qwen3 model on one device.
# tokenizer: its fast tokenizer, with exact offset mappings.
model.eval()
original_text = "Project notes. Obsolete note. The revised schedule starts Monday."
removed_text = "Obsolete note. "
start = original_text.index(removed_text)
device = model.get_input_embeddings().weight.device
input_ids = tokenizer(
    original_text, add_special_tokens=False, return_tensors="pt"
)["input_ids"].to(device)

with torch.no_grad():
    old_cache = model(input_ids=input_ids, use_cache=True).past_key_values

result = repair_after_deletion(
    model, tokenizer, old_cache, original_text,
    delete_span=(start, start + len(removed_text)),  # half-open character offsets
    query="When does the revised schedule start?",
    budget_percent=5,
    executor="packed",  # or "sequential" for interval-wise reference execution
)

print(result.selection.positions)
print(result.selection.intervals)
repaired_cache = result.cache
edited_input_ids = result.input_ids
```

The returned cache contains the **edited prompt only**. The query guides allocation; it is not appended or decoded by this function. Continue inference by passing only new tokens together with this cache, using the edited prompt length for subsequent positions and attention masks. Do not prefill the edited prompt into the returned cache again.

### Budget and cache semantics

- The budget is `ceil(budget_percent / 100 * retained_suffix_tokens)`. A 0% budget performs positional correction only; 100% refreshes the full retained suffix. Selected rows are recomputed through **all layers**.
- The edit boundary can change tokenization. Changed boundary tokens must be recomputed separately; `result.metadata["alignment"]` reports this work. They are not part of the retained-suffix budget.
- SUR uses frozen input-embedding relevance, an identical-token bonus, and text-boundary units. Whole units are selected first; a final partial unit fills the exact token budget. No auxiliary transformer forward is used for selection.
- The old cache must correspond exactly to the supplied canonical text encoded with `add_special_tokens=False`. Include any required chat template or special-token strings before prefilling. Use an unpadded batch of one; a cache cannot independently verify which text created it.
- Repair does not mutate the input cache. Unselected suffix rows retain their old contextual states, so partial repair is approximate and is **not a secure-forgetting guarantee**.

To allocate without executing, use `select_support(model, tokenizer, edited_text, query, suffix_start=..., budget_percent=...)`. Here `suffix_start` is an absolute **token index** in the edited prompt, unlike the character offsets in `delete_span`.

## Supported execution

The public repair API supports native dense `LlamaForCausalLM` and `Qwen3ForCausalLM` models with default fixed RoPE, full attention, batch size one, and a single CPU test device or CUDA device. Models must be in evaluation mode. MoE, sliding-window/hybrid attention, quantized or offloaded/dispatched models, and custom model implementations are outside this release's validated scope.

`packed` defaults to **PyTorch SDPA with Torch cache writes**. The low-level kernel modules also contain experimental Triton paths. Triton attention is not the default: it has known full-model numerical-check failures and is not certified for task-quality preservation. Packed and sequential schedules need not be bitwise identical in reduced precision; the offline suite checks tiny-model numerical behavior, not pretrained task-level equivalence.

## Refresh-stage benchmark

Run on one CUDA GPU with a compatible local checkpoint:

```bash
python benchmarks/benchmark_refresh.py \
  --model /path/to/local/checkpoint --local-files-only \
  --lengths 2048 8192 --interval-counts 1 4 16 --token-budget 64 \
  --warmups 10 --repeats 20 --output outputs/refresh.json
```

An explicit Hugging Face model ID is also accepted; pin `--revision` to a model commit for repeatability. Remote model code is disabled. The benchmark compares interval-wise sequential, direct sequential, one-shot, and packed SDPA execution on identical selected positions. It synchronizes CUDA, interleaves repeated measurements, and records numerical diagnostics, hardware, versions, and raw timing samples.

Timing covers **refresh execution**: output-cache ownership, support planning, masks, and sparse model traversal. Initial prefill, selection, positional correction, answer decoding, and first compilation are excluded. These measurements are not end-to-end request speedups. The lower-level reference functions' `seconds` metadata is host elapsed time and should not be used as synchronized GPU latency.

## Tests and layout

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest -q
python -m ruff check src tests examples benchmarks
```

Tests construct tiny models locally. GPU-dependent cases skip when their hardware or optional dependencies are unavailable. Experimental Triton tests additionally require the `triton` extra on supported Linux/CUDA systems.

```text
src/kv_cache_repair/
  api.py                 Public allocation and deletion-repair interface
  semantic_units.py      Text-boundary units and exact-budget SUR allocation
  scoring.py             Frozen-embedding query relevance
  baselines.py           Same-score tokens, single window, BM25, random units
  alignment.py, cache.py Token alignment, positional correction, cache operations
  execution.py           Sequential and one-shot reference schedules
  kernels/               Packed execution, indexed attention, and cache writes
examples/                Offline API example
benchmarks/              Refresh-only GPU timing
tests/                   Allocation, cache, execution, and interface tests
```

The included baseline helpers are allocation controls under this repair interface, not reproductions of other systems' complete serving stacks.

## Citation

```bibtex
@misc{yu2026budgetedkvcacherepair,
  title = {Budgeted {KV}-Cache Repair for Evolving Contexts},
  author = {Han Yu and Wenhui Zhu and Jincheng Cao and Rui Cai and Yuanda Xu and Zhengze Zhou and Xiwen Chen and Hao Wang and Xuanzhao Dong and Han Shi and Menglin Zhou and Shayan Mohajer Hamidi and Lin Su and Mina Razghandi and Sayan Ghosh and Qing Lan and Rohit Jain and Alborz Geramifard},
  year = {2026},
  eprint = {2610.17338},
  archivePrefix = {arXiv},
  primaryClass = {cs.AI},
  url = {https://arxiv.org/abs/2610.17338}
}
```

Licensed under [Apache-2.0](LICENSE). Model weights and third-party dependencies retain their own licenses.
