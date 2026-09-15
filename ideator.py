"""Ideator agent: proposes new, specific, mechanistic hypotheses."""

from __future__ import annotations

from ..prompts import format_ideator_prompts
from ..schemas import Hypothesis
from .base import Agent


class IdeatorAgent(Agent):
    name = "ideator"

    def generate(
        self,
        n: int,
        iteration: int,
        existing: list[Hypothesis] | None = None,
        critic_feedback: list[str] | None = None,
    ) -> list[Hypothesis]:
        existing = existing or []
        critic_feedback = critic_feedback or []
        existing_statements = [h.statement for h in existing]

        system, user = format_ideator_prompts(self.domain, n, existing_statements, critic_feedback)
        payload = self.llm.complete_json(system, user, max_tokens=1800, temperature=0.9)

        raw_hyps = payload.get("hypotheses", [])
        hypotheses: list[Hypothesis] = []
        for raw in raw_hyps:
            try:
                hyp = Hypothesis(
                    statement=raw["statement"],
                    mechanism=raw["mechanism"],
                    material_family=raw["material_family"],
                    target_metric=raw["target_metric"],
                    rationale=raw["rationale"],
                    iteration=iteration,
                )
                hypotheses.append(hyp)
            except KeyError:
                # Skip malformed entries rather than failing the whole batch;
                # the pipeline is robust to occasional LLM formatting slips.
                continue
        return hypotheses

    def revise(self, hypothesis: Hypothesis, suggested_revision: str, iteration: int) -> Hypothesis:
        """Produce a revised hypothesis incorporating Critic feedback."""
        system, user = format_ideator_prompts(
            self.domain,
            n=1,
            existing=[hypothesis.statement],
            feedback=[suggested_revision],
        )
        payload = self.llm.complete_json(system, user, max_tokens=900, temperature=0.7)
        raw_list = payload.get("hypotheses", [])
        if not raw_list:
            # Fall back to a lightly annotated version of the original so the
            # pipeline can continue even if revision generation fails.
            revised = hypothesis.model_copy(
                update={
                    "id": f"{hypothesis.id}_r",
                    "parent_id": hypothesis.id,
                    "iteration": iteration,
                    "rationale": hypothesis.rationale + f" [Revision note: {suggested_revision}]",
                }
            )
            return revised
        raw = raw_list[0]
        revised = Hypothesis(
            parent_id=hypothesis.id,
            iteration=iteration,
            statement=raw.get("statement", hypothesis.statement),
            mechanism=raw.get("mechanism", hypothesis.mechanism),
            material_family=raw.get("material_family", hypothesis.material_family),
            target_metric=raw.get("target_metric", hypothesis.target_metric),
            rationale=raw.get("rationale", hypothesis.rationale),
        )
        return revised
