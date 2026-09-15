from hypothesisforge.evaluation.ablation import (
    AblationVariant,
    render_ablation_report_markdown,
    run_ablation_study,
)


def test_run_ablation_study_produces_all_variants(mock_llm, run_config, corpus):
    variants = [
        AblationVariant("baseline", "baseline desc", True, "balanced"),
        AblationVariant("no_critic", "no critic desc", False, "balanced"),
    ]
    results = run_ablation_study(run_config, mock_llm, variants=variants, corpus=corpus)
    assert set(results.keys()) == {"baseline", "no_critic"}
    for payload in results.values():
        assert "metrics" in payload
        assert "run_result" in payload


def test_no_critic_variant_has_zero_rejections(mock_llm, run_config, corpus):
    variants = [AblationVariant("no_critic", "desc", False, "balanced")]
    results = run_ablation_study(run_config, mock_llm, variants=variants, corpus=corpus)
    metrics = results["no_critic"]["metrics"]
    assert metrics["rejection_rate"] == 0.0
    assert metrics["revision_rate"] == 0.0


def test_render_ablation_report_markdown_contains_all_variant_keys(mock_llm, run_config, corpus):
    variants = [
        AblationVariant("baseline", "d1", True, "balanced"),
        AblationVariant("strict_judge", "d2", True, "strict"),
    ]
    results = run_ablation_study(run_config, mock_llm, variants=variants, corpus=corpus)
    report = render_ablation_report_markdown(results)
    assert "baseline" in report
    assert "strict_judge" in report
    assert "# HypothesisForge Ablation Report" in report


def test_different_personas_yield_different_mean_scores(mock_llm, run_config, corpus):
    variants = [
        AblationVariant("strict_judge", "d", True, "strict"),
        AblationVariant("lenient_judge", "d", True, "lenient"),
    ]
    results = run_ablation_study(run_config, mock_llm, variants=variants, corpus=corpus)
    strict_score = results["strict_judge"]["metrics"]["mean_overall_score"]
    lenient_score = results["lenient_judge"]["metrics"]["mean_overall_score"]
    assert lenient_score > strict_score
