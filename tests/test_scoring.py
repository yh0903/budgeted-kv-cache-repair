from types import SimpleNamespace

import pytest
import torch

from kv_cache_repair.scoring import dense_query_relevance_scores, query_text_token_ids


@pytest.fixture
def embedding_model():
    vectors = torch.tensor([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]])
    embedding = torch.nn.Embedding.from_pretrained(vectors, freeze=True)
    return SimpleNamespace(generator=SimpleNamespace(get_input_embeddings=lambda: embedding))


def test_dense_relevance_exact_bonus_applies_only_to_identical_ids(embedding_model):
    scores, metadata = dense_query_relevance_scores(
        model=embedding_model,
        edited_input_ids=torch.tensor([[0, 1, 2, 3]]),
        query_token_ids=[1],
    )

    assert scores == pytest.approx([1.0, 1.25, -1.0, 0.0])
    assert metadata["query_token_ids"] == [1]
    assert metadata["exact_token_boost"] == 0.25
    assert metadata["embedding_dot_flops"] == 16


def test_dense_relevance_preserves_scores_across_chunk_boundary(embedding_model):
    prompt = [2] * 1024 + [3]
    scores, _ = dense_query_relevance_scores(
        model=embedding_model,
        edited_input_ids=torch.tensor([prompt]),
        query_token_ids=[1, 3],
    )

    assert scores == pytest.approx([0.0] * 1024 + [1.25])


def test_dense_relevance_rejects_empty_query(embedding_model):
    with pytest.raises(ValueError, match="requires query tokens"):
        dense_query_relevance_scores(
            model=embedding_model,
            edited_input_ids=torch.tensor([[0]]),
            query_token_ids=[],
        )


def test_query_tokenization_retains_repeated_ids_and_stopwords():
    class Tokenizer:
        def __call__(self, text, *, add_special_tokens):
            assert text == "what is what" and add_special_tokens is False
            return {"input_ids": [1, 2, 1]}

    assert query_text_token_ids(Tokenizer(), "what is what") == [1, 2, 1]


def test_query_tokenization_rejects_empty_output():
    with pytest.raises(ValueError, match="tokenization is empty"):
        query_text_token_ids(lambda *args, **kwargs: {"input_ids": []}, "")
