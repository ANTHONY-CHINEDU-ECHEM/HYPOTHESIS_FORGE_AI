from hypothesisforge.tools.novelty_scorer import compute_novelty, embedding_novelty_component


def test_embedding_novelty_component_monotonic_decreasing():
    low_sim_score = embedding_novelty_component(0.0)
    mid_sim_score = embedding_novelty_component(0.5)
    high_sim_score = embedding_novelty_component(1.0)
    assert low_sim_score > mid_sim_score > high_sim_score
    assert high_sim_score == 0.0
    assert low_sim_score == 10.0


def test_embedding_novelty_component_clamps_out_of_range_inputs():
    assert embedding_novelty_component(-0.5) == embedding_novelty_component(0.0)
    assert embedding_novelty_component(1.5) == embedding_novelty_component(1.0)


def test_compute_novelty_combines_signals_within_bounds(corpus):
    result = compute_novelty(
        corpus=corpus,
        query_text="A completely novel exotic crystal structure never before studied.",
        llm_judge_score=9.0,
        llm_judge_rationale="Highly novel per judge.",
    )
    assert 0.0 <= result.combined_score <= 10.0
    assert 0.0 <= result.embedding_max_similarity <= 1.0
    assert len(result.nearest_neighbor_ids) > 0


def test_compute_novelty_weighting_shifts_score_toward_judge(corpus):
    query = "garnet LLZO doping ionic conductivity"  # likely to have real corpus overlap
    judge_heavy = compute_novelty(
        corpus, query, llm_judge_score=10.0, llm_judge_rationale="r",
        embedding_weight=0.0, judge_weight=1.0,
    )
    embed_heavy = compute_novelty(
        corpus, query, llm_judge_score=10.0, llm_judge_rationale="r",
        embedding_weight=1.0, judge_weight=0.0,
    )
    assert judge_heavy.combined_score == 10.0
    assert embed_heavy.combined_score != 10.0
