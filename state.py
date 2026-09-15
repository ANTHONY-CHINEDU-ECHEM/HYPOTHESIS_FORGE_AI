"""Mutable state threaded through the orchestration graph."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..schemas import CritiqueVerdict, Hypothesis, IterationLog, LiteratureFinding


@dataclass
class PipelineState:
    """
    The single object passed between graph nodes. Each node reads what it
    needs and writes its output back onto the state (LangGraph convention:
    nodes are pure functions of `state -> state`).
    """

    iteration: int = 0
    max_iterations: int = 3

    # Working set for the CURRENT iteration.
    active_hypotheses: list[Hypothesis] = field(default_factory=list)
    findings_by_hyp_id: dict[str, LiteratureFinding] = field(default_factory=dict)
    critiques_by_hyp_id: dict[str, CritiqueVerdict] = field(default_factory=dict)

    # Cumulative pools across all iterations.
    all_hypotheses: list[Hypothesis] = field(default_factory=list)
    accepted: list[Hypothesis] = field(default_factory=list)
    rejected: list[Hypothesis] = field(default_factory=list)
    revision_queue: list[tuple[Hypothesis, str]] = field(default_factory=list)  # (hyp, suggestion)

    iteration_logs: list[IterationLog] = field(default_factory=list)

    # Accumulated critic feedback fed back to the Ideator for future rounds.
    feedback_pool: list[str] = field(default_factory=list)

    done: bool = False

    def register_hypotheses(self, hyps: list[Hypothesis]) -> None:
        self.active_hypotheses = hyps
        self.all_hypotheses.extend(hyps)
