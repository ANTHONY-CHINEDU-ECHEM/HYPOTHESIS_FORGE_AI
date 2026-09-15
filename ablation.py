"""
Ablation study harness.

Runs HypothesisForge under several configuration variants and reports
comparative metrics, directly addressing the brief's request for:
"ablation studies (with/without critic, different judge models)".

Variants included by default:
  - baseline:           use_critic=True,  judge_persona="balanced"
  - no_critic:           use_critic=False (critic step neutralized)
  - strict_judge:        use_critic=True,  judge_persona="strict"
  - lenient_judge:       use_critic=True,  judge_persona="lenient"

Each variant runs the full pipeline independently (same domain/corpus,
same random seed for the mock LLM so hypothesis proposals are comparable)
and metrics.summarize_run() is used to produce a comparison table.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

from ..config import RunConfig
from ..llm_client import LLMClient, MockLLMClient
from ..orchestration.pipeline import HypothesisForgePipeline
from ..tools.corpus_search import CorpusIndex
from .metrics import summarize_run


@dataclass
class AblationVariant:
    key: str
    description: str
    use_critic: bool = True
    judge_persona: str = "balanced"


DEFAULT_VARIANTS: list[AblationVariant] = [
    AblationVariant("baseline", "Full pipeline: critic enabled, balanced judge persona.", True, "balanced"),
    AblationVariant("no_critic", "Critic step disabled — every hypothesis auto-accepted.", False, "balanced"),
    AblationVariant("strict_judge", "Critic enabled, strict judge persona.", True, "strict"),
    AblationVariant("lenient_judge", "Critic enabled, lenient judge persona.", True, "lenient"),
]


def _make_llm(base_llm: LLMClient, variant: AblationVariant) -> LLMClient:
    """
    Different "judge models" are simulated for the offline MockLLMClient via
    persona (see llm_client.MockLLMClient docstring). When running against
    the live Anthropic API, swap this for genuinely different model strings
    (e.g. a smaller/larger model) passed through RunConfig.llm_model per
    variant — the harness structure is identical either way.
    """
    if isinstance(base_llm, MockLLMClient):
        return MockLLMClient(model=f"mock-llm-v1[{variant.judge_persona}]", persona=variant.judge_persona)
    return base_llm  # live client: persona is carried via the prompt instead


def run_ablation_study(
    base_config: RunConfig,
    llm: LLMClient,
    variants: list[AblationVariant] | None = None,
    corpus: CorpusIndex | None = None,
) -> dict:
    variants = variants or DEFAULT_VARIANTS
    shared_corpus = corpus or CorpusIndex(
        base_config.domain_profile().corpus_path, embedding_backend=base_config.embedding_backend
    )

    results = {}
    for variant in variants:
        cfg = copy.deepcopy(base_config)
        cfg.use_critic = variant.use_critic
        cfg.judge_persona = variant.judge_persona

        variant_llm = _make_llm(llm, variant)
        pipeline = HypothesisForgePipeline(variant_llm, cfg, corpus=shared_corpus)
        run_result = pipeline.run()
        metrics = summarize_run(run_result)

        results[variant.key] = {
            "description": variant.description,
            "config": {"use_critic": variant.use_critic, "judge_persona": variant.judge_persona},
            "metrics": metrics,
            "run_result": run_result,
        }
    return results


def render_ablation_report_markdown(results: dict) -> str:
    lines = ["# HypothesisForge Ablation Report", ""]
    metric_keys = [
        "total_hypotheses_considered",
        "acceptance_rate",
        "rejection_rate",
        "revision_rate",
        "mean_novelty_score",
        "mean_feasibility_score",
        "mean_overall_score",
        "mean_contradictions_per_hypothesis",
        "iteration_over_iteration_delta",
        "final_ranked_mean_score",
        "toy_simulation_success_rate",
        "wall_clock_seconds",
    ]

    header = "| Metric | " + " | ".join(results.keys()) + " |"
    sep = "|---|" + "---|" * len(results)
    lines.append(header)
    lines.append(sep)
    for mk in metric_keys:
        row = [mk] + [str(results[k]["metrics"].get(mk, "")) for k in results]
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append("## Per-iteration score trajectory (mean overall_score)")
    lines.append("")
    for key, payload in results.items():
        traj = payload["metrics"]["per_iteration_mean_overall_score"]
        lines.append(f"- **{key}**: {traj}")

    lines.append("")
    lines.append("## Variant descriptions")
    lines.append("")
    for key, payload in results.items():
        lines.append(f"- **{key}**: {payload['description']}")

    lines.append("")
    lines.append("## Interpretation notes")
    lines.append("")
    lines.append(
        "- `no_critic` is expected to show a higher (or unchanged) acceptance rate "
        "and zero revision activity, since every hypothesis bypasses feasibility/"
        "contradiction screening — compare its `mean_overall_score` and "
        "`mean_contradictions_per_hypothesis` against `baseline` to see what the "
        "critic step actually catches."
    )
    lines.append(
        "- `strict_judge` vs `lenient_judge` isolates how much of the final score "
        "is judge-calibration-dependent rather than hypothesis-quality-dependent — "
        "large swings suggest the scoring rubric needs tightening (e.g. more "
        "concrete anchor examples in the Critic prompt)."
    )
    lines.append(
        "- `iteration_over_iteration_delta` > 0 indicates the revise-and-resubmit "
        "loop is measurably improving hypothesis quality across iterations, "
        "which is the core claim this ablation is meant to test."
    )
    return "\n".join(lines)
