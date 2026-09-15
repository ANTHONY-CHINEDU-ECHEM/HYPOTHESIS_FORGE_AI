"""
Local literature corpus + similarity search tool.

This is the "tool use" surface for the Literature Scout agent: rather than
hitting a live paper-search API (arXiv/Semantic Scholar etc. — which would
make the demo network-dependent and non-reproducible), HypothesisForge ships
a curated, domain-grounded local corpus of 48 synthetic-but-realistic
abstracts spanning the solid-state-electrolyte literature landscape (see
data/corpus/materials_science_corpus.json + build_corpus.py for how it was
generated). Swapping this module for a real API client (keeping the same
`CorpusIndex.search()` interface) is a drop-in change; see README.md.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..schemas import LiteratureRecord
from .embeddings import EmbeddingBackend, build_backend, cosine_similarity_matrix


@dataclass
class _CorpusDoc:
    doc_id: str
    title: str
    year: int
    tags: list[str]
    abstract: str


class CorpusIndex:
    """Loads a JSON corpus and exposes embedding-based similarity search."""

    def __init__(self, corpus_path: Path, embedding_backend: str = "tfidf"):
        self.corpus_path = Path(corpus_path)
        self._docs: list[_CorpusDoc] = []
        self._backend: EmbeddingBackend = build_backend(embedding_backend)
        self._doc_vectors: np.ndarray | None = None
        self._load_and_index()

    def _load_and_index(self) -> None:
        with open(self.corpus_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._docs = [
            _CorpusDoc(
                doc_id=r["doc_id"],
                title=r["title"],
                year=r["year"],
                tags=r["tags"],
                abstract=r["abstract"],
            )
            for r in data["records"]
        ]
        texts = [f"{d.title}. {d.abstract}" for d in self._docs]
        self._backend.fit(texts)
        self._doc_vectors = self._backend.embed(texts)

    def __len__(self) -> int:
        return len(self._docs)

    def embed_query(self, text: str) -> np.ndarray:
        return self._backend.embed_one(text)

    def search(self, query: str, top_k: int = 5) -> list[LiteratureRecord]:
        """Return the top_k most similar corpus records to `query`."""
        q_vec = self.embed_query(query).reshape(1, -1)
        sims = cosine_similarity_matrix(q_vec, self._doc_vectors)[0]
        order = np.argsort(-sims)[:top_k]
        results = []
        for idx in order:
            doc = self._docs[int(idx)]
            results.append(
                LiteratureRecord(
                    doc_id=doc.doc_id,
                    title=doc.title,
                    year=doc.year,
                    tags=doc.tags,
                    abstract=doc.abstract,
                    similarity=float(sims[idx]),
                )
            )
        return results

    def similarity_stats(self, query: str, top_k: int = 5) -> dict:
        """Return max/mean similarity + nearest neighbor ids, for novelty scoring."""
        records = self.search(query, top_k=top_k)
        sims = [r.similarity for r in records]
        return {
            "max_sim": max(sims) if sims else 0.0,
            "mean_sim": (sum(sims) / len(sims)) if sims else 0.0,
            "records": records,
        }
