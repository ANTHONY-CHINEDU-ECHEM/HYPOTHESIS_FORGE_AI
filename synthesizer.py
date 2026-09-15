"""
Synthesizer agent: produces experiment design sketches (including a toy
simulation), executes the toy simulation via the sandboxed tool, and
computes each surviving hypothesis's final ranking score.
"""

from __future__ import annotations

from ..prompts import format_synthesizer_prompts
from ..schemas import CritiqueVerdict, ExperimentDesign, Hypothesis, RankedHypothesis
from ..tools.simulation import run_toy_simulation
from .base import Agent


class SynthesizerAgent(Agent):
    name = "synthesizer"

    def design_experiment(self, hypothesis: Hypothesis, critique: CritiqueVerdict) -> ExperimentDesign:
        system, user = format_synthesizer_prompts(self.domain, hypothesis, critique)
        payload = self.llm.complete_json(system, user, max_tokens=1200, temperature=0.5)

        design = ExperimentDesign(
            hypothesis_id=hypothesis.id,
            protocol_steps=list(payload.get("protocol_steps", [])),
            independent_variables=list(payload.get("independent_variables", [])),
            dependent_variables=list(payload.get("dependent_variables", [])),
            controls=list(payload.get("controls", [])),
            toy_simulation_code=payload.get("toy_simulation_code", ""),
            predicted_outcome=payload.get("predicted_outcome", ""),
        )

        if design.toy_simulation_code.strip():
            design.simulation_result = run_toy_simulation(design.toy_simulation_code)

        return design

    def rank(
        self,
        survivors: list[tuple[Hypothesis, CritiqueVerdict, ExperimentDesign]],
    ) -> list[RankedHypothesis]:
        scored: list[tuple[float, Hypothesis, CritiqueVerdict, ExperimentDesign]] = []
        for hyp, critique, design in survivors:
            sim_bonus = 0.0
            if design.simulation_result and design.simulation_result.executed:
                sim_bonus = 0.3  # small reward for a hypothesis whose toy model actually runs
            final_score = round(min(10.0, critique.overall_score + sim_bonus), 2)
            scored.append((final_score, hyp, critique, design))

        scored.sort(key=lambda t: t[0], reverse=True)

        ranked = []
        for rank, (score, hyp, critique, design) in enumerate(scored, start=1):
            ranked.append(
                RankedHypothesis(
                    hypothesis=hyp,
                    critique=critique,
                    experiment_design=design,
                    final_score=score,
                    rank=rank,
                )
            )
        return ranked
