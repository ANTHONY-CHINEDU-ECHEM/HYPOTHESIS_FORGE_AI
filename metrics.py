"""Metric extraction from a RunResult, used by the ablation harness."""

from __future__ import annotations

from statistics import mean, pstdev

from ..schemas import RunResult


def _safe_mean(xs: list[float]) -> float:
    return round(mean(xs), 3) if xs else 0.0


def _safe_std(xs: list[float]) -> float:
    return round(pstdev(xs), 3) if len(xs) > 1 else 0.0


def summarize_run(result: RunResult) -> dict:
    all_critiques = [c for log in result.iterations for c in log.critiques]
    novelty_scores = [c.novelty.combined_score for c in all_critiques]
    feasibility_scores = [c.feasibility_score for c in all_critiques]
    overall_scores = [c.overall_score for c in all_critiques]
    contradiction_counts = [len(c.contradictions) for c in all_critiques]

    n_accepted = sum(len(log.accepted_ids) for log in result.iterations)
    n_rejected = sum(len(log.rejected_ids) for log in result.iterations)
    n_revised = sum(len(log.revised_ids) for log in result.iterations)
    n_total_critiqued = len(all_critiques)

    # Per-iteration mean overall score, to show whether quality improves
    # across iterations (the "measurable improvement" the brief asks for).
    per_iteration_mean_overall = [
        _safe_mean([c.overall_score for c in log.critiques]) for log in result.iterations
    ]
    per_iteration_mean_novelty = [
        _safe_mean([c.novelty.combined_score for c in log.critiques]) for log in result.iterations
    ]

    final_scores = [rh.final_score for rh in result.ranked_hypotheses]
    sim_success = [
        1.0
        for rh in result.ranked_hypotheses
        if rh.experiment_design.simulation_result and rh.experiment_design.simulation_result.executed
    ]

    return {
        "total_hypotheses_considered": result.total_hypotheses_considered,
        "total_critiqued": n_total_critiqued,
        "acceptance_rate": round(n_accepted / n_total_critiqued, 3) if n_total_critiqued else 0.0,
        "rejection_rate": round(n_rejected / n_total_critiqued, 3) if n_total_critiqued else 0.0,
        "revision_rate": round(n_revised / n_total_critiqued, 3) if n_total_critiqued else 0.0,
        "mean_novelty_score": _safe_mean(novelty_scores),
        "std_novelty_score": _safe_std(novelty_scores),
        "mean_feasibility_score": _safe_mean(feasibility_scores),
        "mean_overall_score": _safe_mean(overall_scores),
        "mean_contradictions_per_hypothesis": _safe_mean([float(c) for c in contradiction_counts]),
        "per_iteration_mean_overall_score": per_iteration_mean_overall,
        "per_iteration_mean_novelty_score": per_iteration_mean_novelty,
        "iteration_over_iteration_delta": (
            round(per_iteration_mean_overall[-1] - per_iteration_mean_overall[0], 3)
            if len(per_iteration_mean_overall) > 1
            else 0.0
        ),
        "final_ranked_mean_score": _safe_mean(final_scores),
        "toy_simulation_success_rate": (
            round(len(sim_success) / len(result.ranked_hypotheses), 3)
            if result.ranked_hypotheses
            else 0.0
        ),
        "wall_clock_seconds": result.wall_clock_seconds,
    }
