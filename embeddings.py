"""
Embedding backends used for novelty scoring and corpus retrieval.

Default backend is TF-IDF (via scikit-learn): zero network dependency, fast,
fully deterministic, and good enough to demonstrate the pipeline end-to-end.
An optional sentence-transformers backend is provided for higher-quality
semantic similarity when the package/model is available locally.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class EmbeddingBackend(ABC):
    name: str = "abstract"

    @abstractmethod
    def fit(self, corpus_texts: list[str]) -> None:
        """Fit any corpus-dependent state (e.g., TF-IDF vocabulary)."""

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (n_texts, dim) float array of embeddings."""

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


class TfidfEmbeddingBackend(EmbeddingBackend):
    """
    Deterministic, dependency-light embedding backend using TF-IDF +
    cosine-ready L2-normalized vectors. This is the default backend: it
    requires no downloaded model weights and runs fully offline.
    """

    name = "tfidf"

    def __init__(self, max_features: int = 4000, ngram_range: tuple[int, int] = (1, 2)):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words="english",
        )
        self._fitted = False

    def fit(self, corpus_texts: list[str]) -> None:
        self._vectorizer.fit(corpus_texts)
        self._fitted = True

    def embed(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TfidfEmbeddingBackend.fit() must be called before embed().")
        matrix = self._vectorizer.transform(texts)
        return matrix.toarray().astype(np.float32)


class SentenceTransformerEmbeddingBackend(EmbeddingBackend):
    """
    Optional higher-quality semantic embedding backend. Requires the
    `sentence-transformers` package and (typically) a one-time model
    download, so it is opt-in rather than the default.
    """

    name = "sentence-transformers"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. Run "
                "`pip install sentence-transformers` or use the 'tfidf' backend."
            ) from exc
        self._model = SentenceTransformer(model_name)

    def fit(self, corpus_texts: list[str]) -> None:
        # Sentence-transformer embeddings don't depend on corpus fitting.
        return

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self._model.encode(texts, normalize_embeddings=False))


def cosine_similarity_matrix(query_vecs: np.ndarray, corpus_vecs: np.ndarray) -> np.ndarray:
    """Return (n_query, n_corpus) cosine similarities, robust to zero vectors."""
    q_norm = np.linalg.norm(query_vecs, axis=1, keepdims=True)
    c_norm = np.linalg.norm(corpus_vecs, axis=1, keepdims=True)
    q_norm[q_norm == 0] = 1e-9
    c_norm[c_norm == 0] = 1e-9
    q_unit = query_vecs / q_norm
    c_unit = corpus_vecs / c_norm
    return q_unit @ c_unit.T


def build_backend(name: str) -> EmbeddingBackend:
    if name == "tfidf":
        return TfidfEmbeddingBackend()
    if name == "sentence-transformers":
        return SentenceTransformerEmbeddingBackend()
    raise ValueError(f"Unknown embedding backend: {name}")
