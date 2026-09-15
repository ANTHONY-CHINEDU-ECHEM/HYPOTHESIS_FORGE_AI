"""
Novelty scoring: fuses two independent signals.

1. Embedding similarity against the local corpus (objective, cheap, but
   purely lexical/semantic-surface — a paraphrase of an existing abstract
   would score as "similar" even if the underlying idea combination is new).
2. LLM-as-judge score, which reads the hypothesis + Literature Scout's
   qualitative gap assessment and reasons about *conceptual* novelty (e.g.
   two individually-known ideas combined in a new way should score higher
   than pure embedding similarity alone would suggest).

combined_score is a tunable weighted blend, expressed on a 0-10 scale to
match the Critic's other scores.
"""

from __future__ import annotations

from ..schemas import NoveltyScore
from .corpus_search import CorpusIndex

# Weight given to the embedding-derived novelty signal vs. the LLM judge
# signal when producing combined_score. Exposed as a module constant (rather
# than buried in a function) so the ablation harness can vary it explicitly.
DEFAULT_EMBEDDING_WEIGHT = 0.4
DEFAULT_JUDGE_WEIGHT = 0.6


def embedding_novelty_component(max_sim: float) -> float:
    """Map max cosine similarity in [0,1] to a 0-10 novelty sub-score.

    max_sim near 1.0 (near-duplicate of existing literature) -> novelty near 0.
    max_sim near 0.0 (nothing similar in corpus) -> novelty near 10.
    A mild nonlinearity (sqrt) keeps mid-range similarity from being scored
    too harshly, since moderate topical overlap is expected and healthy.
    """
    max_sim = max(0.0, min(1.0, max_sim))
    return round((1.0 - max_sim**0.7) * 10.0, 2)


def compute_novelty(
    corpus: CorpusIndex,
    query_text: str,
    llm_judge_score: float,
    llm_judge_rationale: str,
    top_k: int = 5,
    embedding_weight: float = DEFAULT_EMBEDDING_WEIGHT,
    judge_weight: float = DEFAULT_JUDGE_WEIGHT,
) -> NoveltyScore:
    stats = corpus.similarity_stats(query_text, top_k=top_k)
    embed_component = embedding_novelty_component(stats["max_sim"])
    total_weight = embedding_weight + judge_weight
    combined = (embed_component * embedding_weight + llm_judge_score * judge_weight) / total_weight
    return NoveltyScore(
        embedding_max_similarity=stats["max_sim"],
        embedding_mean_similarity=stats["mean_sim"],
        nearest_neighbor_ids=[r.doc_id for r in stats["records"]],
        llm_judge_score=llm_judge_score,
        llm_judge_rationale=llm_judge_rationale,
        combined_score=round(max(0.0, min(10.0, combined)), 2),
    )
