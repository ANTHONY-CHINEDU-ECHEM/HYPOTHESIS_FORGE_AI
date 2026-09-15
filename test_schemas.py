import pytest
from pydantic import ValidationError

from hypothesisforge.schemas import Hypothesis, HypothesisStatus, NoveltyScore, Recommendation


def test_hypothesis_defaults_and_id_generation():
    h1 = Hypothesis(
        statement="s", mechanism="m", material_family="f", target_metric="t", rationale="r"
    )
    h2 = Hypothesis(
        statement="s", mechanism="m", material_family="f", target_metric="t", rationale="r"
    )
    assert h1.id != h2.id
    assert h1.status == HypothesisStatus.PROPOSED
    assert h1.iteration == 0


def test_hypothesis_short():
    h = Hypothesis(
        statement="Doping improves conductivity.",
        mechanism="m",
        material_family="f",
        target_metric="t",
        rationale="r",
    )
    assert h.id in h.short()
    assert "Doping improves conductivity." in h.short()


def test_novelty_score_bounds_enforced():
    with pytest.raises(ValidationError):
        NoveltyScore(
            embedding_max_similarity=1.5,  # out of [0,1]
            embedding_mean_similarity=0.2,
            nearest_neighbor_ids=[],
            llm_judge_score=5.0,
            llm_judge_rationale="x",
            combined_score=5.0,
        )


def test_novelty_score_similarity_rounding():
    ns = NoveltyScore(
        embedding_max_similarity=0.123456789,
        embedding_mean_similarity=0.1,
        nearest_neighbor_ids=[],
        llm_judge_score=5.0,
        llm_judge_rationale="x",
        combined_score=5.0,
    )
    assert ns.embedding_max_similarity == 0.1235


def test_recommendation_enum_values():
    assert Recommendation("accept") == Recommendation.ACCEPT
    assert Recommendation("revise") == Recommendation.REVISE
    assert Recommendation("reject") == Recommendation.REJECT
    with pytest.raises(ValueError):
        Recommendation("maybe")
