"""
Critic agent: checks novelty, feasibility, and contradictions.

Novelty is computed by fusing an embedding-similarity signal (via
tools.novelty_scorer, which itself calls the corpus tool) with an
LLM-judge signal produced in the same call as feasibility/contradiction
analysis. This keeps the "LLM-as-judge" skill and the embedding-similarity
skill both genuinely exercised rather than one being decorative.
"""

from __future__ import annotations

from ..prompts import PERSONA_DESCRIPTIONS, format_critic_prompts
from ..schemas import CritiqueVerdict, Hypothesis, LiteratureFinding, Recommendation
from ..tools.corpus_search import CorpusIndex
from ..tools.novelty_scorer import compute_novelty
from .base import Agent


class CriticAgent(Agent):
    name = "critic"

    def __init__(self, llm, config, corpus: CorpusIndex):
        super().__init__(llm, config)
        self.corpus = corpus

    def critique(self, hypothesis: Hypothesis, finding: LiteratureFinding) -> CritiqueVerdict:
        query = f"{hypothesis.statement} {hypothesis.mechanism}"
        stats = self.corpus.similarity_stats(query, top_k=5)
        novelty_partial = stats  # has max_sim, mean_sim, records

        system, user = format_critic_prompts(
            self.domain, hypothesis, finding, novelty_partial, self.config.judge_persona
        )
        payload = self.llm.complete_json(system, user, max_tokens=1000, temperature=0.3)

        novelty = compute_novelty(
            corpus=self.corpus,
            query_text=query,
            llm_judge_score=float(payload.get("llm_judge_novelty_score", 5.0)),
            llm_judge_rationale=payload.get("llm_judge_novelty_rationale", ""),
        )

        feasibility_score = float(payload.get("feasibility_score", 5.0))
        contradictions = list(payload.get("contradictions", []))
        overall = self._recompute_overall(novelty.combined_score, feasibility_score, contradictions)

        recommendation_raw = payload.get("recommendation", "revise")
        try:
            recommendation = Recommendation(recommendation_raw)
        except ValueError:
            recommendation = self._threshold_recommendation(overall)

        return CritiqueVerdict(
            hypothesis_id=hypothesis.id,
            novelty=novelty,
            feasibility_score=round(feasibility_score, 2),
            feasibility_rationale=payload.get("feasibility_rationale", ""),
            contradictions=contradictions,
            identified_flaws=list(payload.get("identified_flaws", [])),
            suggested_revision=payload.get("suggested_revision"),
            overall_score=overall,
            recommendation=recommendation,
        )

    def _recompute_overall(
        self, novelty_score: float, feasibility_score: float, contradictions: list[str]
    ) -> float:
        """
        Recompute overall_score from config-driven weights rather than
        trusting the LLM's arithmetic verbatim — this is what makes the
        Critic's scoring policy programmatically tunable for ablations
        (novelty_weight / feasibility_weight / contradiction_penalty_weight
        in RunConfig) instead of being whatever the model felt like.
        """
        cfg = self.config
        weighted_sum = cfg.novelty_weight * novelty_score + cfg.feasibility_weight * feasibility_score
        # Normalize by weight total (novelty_weight + feasibility_weight need not
        # sum to 1 if a caller changes only one of them) to keep base on 0-10.
        weight_total = cfg.novelty_weight + cfg.feasibility_weight
        base = weighted_sum / weight_total if weight_total > 0 else 0.0
        penalty = cfg.contradiction_penalty_weight * len(contradictions)
        return round(max(0.0, min(10.0, base - penalty)), 2)

    def _threshold_recommendation(self, overall: float) -> Recommendation:
        if overall < self.config.critic_reject_threshold:
            return Recommendation.REJECT
        if overall < self.config.critic_revise_threshold:
            return Recommendation.REVISE
        return Recommendation.ACCEPT
