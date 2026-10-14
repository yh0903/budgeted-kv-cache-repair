"""Frozen-embedding query relevance without an auxiliary model forward."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


@torch.no_grad()
def dense_query_relevance_scores(
    *,
    model: Any,
    edited_input_ids: torch.Tensor,
    query_token_ids: list[int],
) -> tuple[list[float], dict[str, Any]]:
    if not query_token_ids:
        raise ValueError("dense query scorer requires query tokens")
    device = edited_input_ids.device
    query_tensor = torch.tensor(
        [query_token_ids],
        dtype=torch.long,
        device=device,
    )
    embedding = model.generator.get_input_embeddings()
    query_vectors = F.normalize(
        embedding(query_tensor)[0].float(),
        dim=-1,
    )
    prompt_ids = edited_input_ids[0]
    scores = torch.full(
        (int(prompt_ids.shape[0]),),
        -float("inf"),
        dtype=torch.float32,
        device=device,
    )
    exact_query_ids = set(query_token_ids)
    chunk_size = 1024
    for start in range(0, int(prompt_ids.shape[0]), chunk_size):
        end = min(start + chunk_size, int(prompt_ids.shape[0]))
        chunk_ids = prompt_ids[start:end]
        chunk_vectors = F.normalize(
            embedding(chunk_ids).float(),
            dim=-1,
        )
        similarity = (
            torch.matmul(
                chunk_vectors,
                query_vectors.transpose(0, 1),
            )
            .max(dim=1)
            .values
        )
        exact_boost = torch.tensor(
            [0.25 if int(token_id) in exact_query_ids else 0.0 for token_id in chunk_ids.tolist()],
            dtype=torch.float32,
            device=device,
        )
        scores[start:end] = similarity + exact_boost
    hidden_size = int(embedding.weight.shape[1])
    return [float(value) for value in scores.tolist()], {
        "query_token_ids": [int(value) for value in query_token_ids],
        "query_token_count": len(query_token_ids),
        "exact_token_boost": 0.25,
        "embedding_dot_flops": (2 * int(prompt_ids.shape[0]) * len(query_token_ids) * hidden_size),
        "embedding_dot_flops_scope": (
            "prompt-token by query-token embedding dot products only; "
            "excludes lookup, normalization, max reduction, and "
            "exact-token boost"
        ),
    }


def query_text_token_ids(
    tokenizer: Any,
    query_text: str,
) -> list[int]:
    token_ids = [
        int(value)
        for value in tokenizer(
            query_text,
            add_special_tokens=False,
        )["input_ids"]
    ]
    if not token_ids:
        raise ValueError("query text tokenization is empty")
    return token_ids
