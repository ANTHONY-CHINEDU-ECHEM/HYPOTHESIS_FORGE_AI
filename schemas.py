"""
Structured data contracts passed between agents.

Every agent reads and returns one of these Pydantic models. Keeping the
contracts explicit (rather than passing around free-text) is what lets the
Critic, Synthesizer, and evaluation harness reason about the pipeline
programmatically instead of re-parsing prose at every hop.
"""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


def _short_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


class HypothesisStatus(str, Enum):
    PROPOSED = "proposed"
    REVISED = "revised"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class Recommendation(str, Enum):
    ACCEPT = "accept"
    REVISE = "revise"
    REJECT = "reject"


class Hypothesis(BaseModel):
    id: str = Field(default_factory=lambda: _short_id("hyp"))
    parent_id: Optional[str] = None
    iteration: int = 0
    statement: str = Field(..., description="One or two sentence hypothesis statement.")
    mechanism: str = Field(..., description="Proposed physical/chemical mechanism.")
    material_family: str = Field(..., description="Which known family this targets or extends.")
    target_metric: str = Field(..., description="Primary metric this hypothesis aims to improve.")
    rationale: str = Field(..., description="Why this is plausible given known chemistry/physics.")
    status: HypothesisStatus = HypothesisStatus.PROPOSED

    def short(self) -> str:
        return f"[{self.id}] {self.statement}"


class LiteratureRecord(BaseModel):
    doc_id: str
    title: str
    year: int
    tags: list[str]
    abstract: str
    similarity: float = Field(0.0, ge=0.0, le=1.0)


class LiteratureFinding(BaseModel):
    hypothesis_id: str
    query: str
    records: list[LiteratureRecord]
    scout_summary: str = Field(
        ..., description="Literature Scout's synthesis of what is already known/attempted."
    )
    closest_prior_work_gap: str = Field(
        ..., description="Scout's assessment of the gap between this hypothesis and prior work."
    )


class NoveltyScore(BaseModel):
    embedding_max_similarity: float = Field(..., ge=0.0, le=1.0)
    embedding_mean_similarity: float = Field(..., ge=0.0, le=1.0)
    nearest_neighbor_ids: list[str]
    llm_judge_score: float = Field(..., ge=0.0, le=10.0)
    llm_judge_rationale: str
    combined_score: float = Field(..., ge=0.0, le=10.0)

    @field_validator("embedding_max_similarity", "embedding_mean_similarity")
    @classmethod
    def _round_sim(cls, v: float) -> float:
        return round(v, 4)


class CritiqueVerdict(BaseModel):
    hypothesis_id: str
    novelty: NoveltyScore
    feasibility_score: float = Field(..., ge=0.0, le=10.0)
    feasibility_rationale: str
    contradictions: list[str] = Field(default_factory=list)
    identified_flaws: list[str] = Field(default_factory=list)
    suggested_revision: Optional[str] = None
    overall_score: float = Field(..., ge=0.0, le=10.0)
    recommendation: Recommendation


class SimulationResult(BaseModel):
    executed: bool
    stdout: str = ""
    error: Optional[str] = None
    parsed_metrics: dict[str, float] = Field(default_factory=dict)
    runtime_seconds: float = 0.0


class ExperimentDesign(BaseModel):
    hypothesis_id: str
    protocol_steps: list[str]
    independent_variables: list[str]
    dependent_variables: list[str]
    controls: list[str]
    toy_simulation_code: str = Field(
        ..., description="Small, self-contained Python snippet approximating expected behavior."
    )
    predicted_outcome: str
    simulation_result: Optional[SimulationResult] = None


class RankedHypothesis(BaseModel):
    hypothesis: Hypothesis
    critique: CritiqueVerdict
    experiment_design: ExperimentDesign
    final_score: float = Field(..., ge=0.0, le=10.0)
    rank: int


class IterationLog(BaseModel):
    iteration_number: int
    proposed: list[Hypothesis]
    findings: list[LiteratureFinding]
    critiques: list[CritiqueVerdict]
    accepted_ids: list[str]
    revised_ids: list[str]
    rejected_ids: list[str]
    notes: str = ""


class RunResult(BaseModel):
    domain: str
    config_snapshot: dict
    iterations: list[IterationLog]
    ranked_hypotheses: list[RankedHypothesis]
    total_hypotheses_considered: int
    wall_clock_seconds: float
