from hypothesisforge.agents.critic import CriticAgent
from hypothesisforge.agents.ideator import IdeatorAgent
from hypothesisforge.agents.literature_scout import LiteratureScoutAgent
from hypothesisforge.agents.synthesizer import SynthesizerAgent
from hypothesisforge.config import RunConfig
from hypothesisforge.schemas import Hypothesis


def test_ideator_generates_requested_count(mock_llm, run_config):
    agent = IdeatorAgent(mock_llm, run_config)
    hyps = agent.generate(n=3, iteration=1)
    assert len(hyps) == 3
    for h in hyps:
        assert h.statement
        assert h.mechanism
        assert h.iteration == 1


def test_ideator_avoids_verbatim_duplicate_of_existing(mock_llm, run_config):
    agent = IdeatorAgent(mock_llm, run_config)
    first_batch = agent.generate(n=2, iteration=1)
    existing_statements = {h.statement for h in first_batch}
    second_batch = agent.generate(n=2, iteration=2, existing=first_batch)
    # Mock generator is deterministic per-seed but seed includes existing
    # hypothesis text via the user prompt, so batches should differ.
    second_statements = {h.statement for h in second_batch}
    assert second_statements != existing_statements


def test_ideator_revise_produces_linked_hypothesis(mock_llm, run_config):
    agent = IdeatorAgent(mock_llm, run_config)
    original = Hypothesis(
        statement="Doping X improves Y.",
        mechanism="m",
        material_family="garnet-type oxides",
        target_metric="ionic conductivity",
        rationale="r",
    )
    revised = agent.revise(original, "Be more specific about dopant concentration.", iteration=2)
    assert revised.parent_id == original.id
    assert revised.iteration == 2


def test_literature_scout_returns_records_and_summary(mock_llm, run_config, corpus):
    agent = LiteratureScoutAgent(mock_llm, run_config, corpus)
    hyp = Hypothesis(
        statement="Doping improves conductivity in garnet electrolytes.",
        mechanism="aliovalent doping",
        material_family="garnet-type oxides",
        target_metric="ionic conductivity",
        rationale="r",
    )
    finding = agent.investigate(hyp)
    assert finding.hypothesis_id == hyp.id
    assert len(finding.records) > 0
    assert finding.scout_summary
    assert finding.closest_prior_work_gap


def test_critic_produces_bounded_scores_and_valid_recommendation(mock_llm, run_config, corpus):
    scout = LiteratureScoutAgent(mock_llm, run_config, corpus)
    critic = CriticAgent(mock_llm, run_config, corpus)
    hyp = Hypothesis(
        statement="Doping improves conductivity in garnet electrolytes.",
        mechanism="aliovalent doping",
        material_family="garnet-type oxides",
        target_metric="ionic conductivity",
        rationale="r",
    )
    finding = scout.investigate(hyp)
    verdict = critic.critique(hyp, finding)
    assert 0.0 <= verdict.overall_score <= 10.0
    assert 0.0 <= verdict.novelty.combined_score <= 10.0
    assert verdict.recommendation.value in {"accept", "revise", "reject"}


def test_critic_overall_score_respects_config_weights(mock_llm, corpus):
    from dataclasses import replace

    cfg_novelty_heavy = replace(
        RunConfig(),
        novelty_weight=1.0,
        feasibility_weight=0.0,
        contradiction_penalty_weight=0.0,
    )
    critic = CriticAgent(mock_llm, cfg_novelty_heavy, corpus)
    hyp = Hypothesis(
        statement="A totally novel halide chemistry never explored before.",
        mechanism="exotic anion framework",
        material_family="lithium halides and halide-doped analogues",
        target_metric="critical current density",
        rationale="r",
    )
    scout = LiteratureScoutAgent(mock_llm, cfg_novelty_heavy, corpus)
    finding = scout.investigate(hyp)
    verdict = critic.critique(hyp, finding)
    # With feasibility weight = 0, overall_score should equal novelty score
    # (minus contradiction penalty, which is also zeroed here).
    assert abs(verdict.overall_score - verdict.novelty.combined_score) < 1e-6


def test_synthesizer_designs_and_executes_simulation(mock_llm, run_config, corpus):
    scout = LiteratureScoutAgent(mock_llm, run_config, corpus)
    critic = CriticAgent(mock_llm, run_config, corpus)
    synth = SynthesizerAgent(mock_llm, run_config)
    hyp = Hypothesis(
        statement="Doping improves conductivity.",
        mechanism="aliovalent doping",
        material_family="garnet-type oxides",
        target_metric="ionic conductivity",
        rationale="r",
    )
    finding = scout.investigate(hyp)
    verdict = critic.critique(hyp, finding)
    design = synth.design_experiment(hyp, verdict)
    assert design.protocol_steps
    assert design.simulation_result is not None
    assert design.simulation_result.executed


def test_synthesizer_rank_orders_by_final_score(mock_llm, run_config):
    synth = SynthesizerAgent(mock_llm, run_config)
    from hypothesisforge.schemas import CritiqueVerdict, ExperimentDesign, NoveltyScore, Recommendation

    def make(score):
        h = Hypothesis(statement="s", mechanism="m", material_family="f", target_metric="t", rationale="r")
        nov = NoveltyScore(
            embedding_max_similarity=0.1, embedding_mean_similarity=0.1,
            nearest_neighbor_ids=[], llm_judge_score=score,
            llm_judge_rationale="x", combined_score=score,
        )
        c = CritiqueVerdict(
            hypothesis_id=h.id, novelty=nov, feasibility_score=score,
            feasibility_rationale="x", overall_score=score, recommendation=Recommendation.ACCEPT,
        )
        d = ExperimentDesign(
            hypothesis_id=h.id, protocol_steps=["step"], independent_variables=[],
            dependent_variables=[], controls=[], toy_simulation_code="print('x=1')",
            predicted_outcome="p",
        )
        return h, c, d

    survivors = [make(3.0), make(8.0), make(5.5)]
    ranked = synth.rank(survivors)
    assert [rh.rank for rh in ranked] == [1, 2, 3]
    assert ranked[0].final_score >= ranked[1].final_score >= ranked[2].final_score
