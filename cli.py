"""
Command-line interface for HypothesisForge.

Examples
--------
Run the full pipeline (offline, mock LLM, no API key needed):
    python -m hypothesisforge.cli run --iterations 3 --seeds 4

Run against the live Anthropic API:
    export ANTHROPIC_API_KEY=sk-...
    python -m hypothesisforge.cli run --live --iterations 3

Run the ablation study and write a markdown report:
    python -m hypothesisforge.cli ablate --out examples/ablation_report.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import DEFAULT_DOMAIN, DOMAIN_REGISTRY, RunConfig, has_live_api_key
from .evaluation.ablation import render_ablation_report_markdown, run_ablation_study
from .evaluation.metrics import summarize_run
from .llm_client import AnthropicLLMClient, LLMClient, MockLLMClient
from .orchestration.pipeline import HypothesisForgePipeline


def _build_llm(live: bool, model: str, persona: str) -> LLMClient:
    if live:
        return AnthropicLLMClient(model=model)
    return MockLLMClient(model="mock-llm-v1", persona=persona)


def _print_run_summary(result) -> None:
    print(f"\n=== HypothesisForge run summary ===")
    print(f"Domain: {result.domain}")
    print(f"Hypotheses considered: {result.total_hypotheses_considered}")
    print(f"Iterations run: {len(result.iterations)}")
    print(f"Wall clock: {result.wall_clock_seconds}s\n")

    print(f"Top {len(result.ranked_hypotheses)} ranked hypotheses:\n")
    for rh in result.ranked_hypotheses:
        print(f"#{rh.rank} [score={rh.final_score}] {rh.hypothesis.statement}")
        print(f"    mechanism: {rh.hypothesis.mechanism}")
        print(
            f"    novelty={rh.critique.novelty.combined_score} "
            f"feasibility={rh.critique.feasibility_score} "
            f"contradictions={len(rh.critique.contradictions)}"
        )
        if rh.experiment_design.simulation_result:
            sr = rh.experiment_design.simulation_result
            status = "ok" if sr.executed else f"FAILED: {sr.error}"
            print(f"    toy simulation: {status} metrics={sr.parsed_metrics}")
        print()

    metrics = summarize_run(result)
    print("Aggregate metrics:")
    print(json.dumps(metrics, indent=2))


def cmd_run(args: argparse.Namespace) -> None:
    if args.live and not has_live_api_key():
        print(
            "ERROR: --live requires ANTHROPIC_API_KEY to be set in the environment.",
            file=sys.stderr,
        )
        sys.exit(1)

    config = RunConfig(
        domain=args.domain,
        n_seed_hypotheses=args.seeds,
        max_iterations=args.iterations,
        top_k_final=args.top_k,
        use_critic=not args.no_critic,
        judge_persona=args.persona,
        embedding_backend=args.embedding_backend,
    )
    llm = _build_llm(args.live, config.llm_model, args.persona)
    pipeline = HypothesisForgePipeline(llm, config)
    result = pipeline.run()

    _print_run_summary(result)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(result.model_dump_json(indent=2))
        print(f"\nFull structured result written to {out_path}")


def cmd_ablate(args: argparse.Namespace) -> None:
    if args.live and not has_live_api_key():
        print(
            "ERROR: --live requires ANTHROPIC_API_KEY to be set in the environment.",
            file=sys.stderr,
        )
        sys.exit(1)

    config = RunConfig(
        domain=args.domain,
        n_seed_hypotheses=args.seeds,
        max_iterations=args.iterations,
        top_k_final=args.top_k,
        embedding_backend=args.embedding_backend,
    )
    llm = _build_llm(args.live, config.llm_model, "balanced")
    results = run_ablation_study(config, llm)

    report_md = render_ablation_report_markdown(results)
    print(report_md)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report_md)
        print(f"\nAblation report written to {out_path}")

    if args.json_out:
        json_path = Path(args.json_out)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        serializable = {
            key: {
                "description": payload["description"],
                "config": payload["config"],
                "metrics": payload["metrics"],
            }
            for key, payload in results.items()
        }
        json_path.write_text(json.dumps(serializable, indent=2))
        print(f"Ablation metrics JSON written to {json_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hypothesisforge",
        description="Multi-agent scientific hypothesis generator & critic.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    common_args = argparse.ArgumentParser(add_help=False)
    common_args.add_argument("--domain", default=DEFAULT_DOMAIN, choices=list(DOMAIN_REGISTRY.keys()))
    common_args.add_argument("--seeds", type=int, default=4, help="Seed hypotheses per iteration.")
    common_args.add_argument("--iterations", type=int, default=3, help="Max iterations.")
    common_args.add_argument("--top-k", type=int, default=3, help="Final ranked hypotheses to keep.")
    common_args.add_argument(
        "--embedding-backend", default="tfidf", choices=["tfidf", "sentence-transformers"]
    )
    common_args.add_argument("--live", action="store_true", help="Use the real Anthropic API.")

    run_parser = sub.add_parser("run", parents=[common_args], help="Run the full pipeline once.")
    run_parser.add_argument("--no-critic", action="store_true", help="Ablation: disable the Critic.")
    run_parser.add_argument(
        "--persona", default="balanced", choices=["balanced", "strict", "lenient"]
    )
    run_parser.add_argument("--out", help="Write full structured JSON result to this path.")
    run_parser.set_defaults(func=cmd_run)

    ablate_parser = sub.add_parser(
        "ablate", parents=[common_args], help="Run the with/without-critic + judge-persona ablation study."
    )
    ablate_parser.add_argument("--out", help="Write markdown report to this path.")
    ablate_parser.add_argument("--json-out", help="Write metrics JSON to this path.")
    ablate_parser.set_defaults(func=cmd_ablate)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
