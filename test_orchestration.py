import pytest

from hypothesisforge.orchestration.graph import END, GraphError, StateGraph
from hypothesisforge.orchestration.pipeline import HypothesisForgePipeline


# ---- graph engine unit tests -----------------------------------------
def test_linear_graph_executes_in_order():
    trace = []

    def a(s):
        trace.append("a")
        return s

    def b(s):
        trace.append("b")
        return s

    g = StateGraph()
    g.add_node("a", a).add_node("b", b)
    g.set_entry_point("a")
    g.add_edge("a", "b")
    g.add_edge("b", END)
    compiled = g.compile()
    compiled.invoke({})
    assert trace == ["a", "b"]


def test_conditional_edge_routes_correctly():
    def a(s):
        s["count"] = s.get("count", 0) + 1
        return s

    def b(s):
        s["visited_b"] = True
        return s

    g = StateGraph()
    g.add_node("a", a).add_node("b", b)
    g.set_entry_point("a")
    g.add_conditional_edges(
        "a", lambda s: "loop" if s["count"] < 3 else "done", {"loop": "a", "done": "b"}
    )
    g.add_edge("b", END)
    compiled = g.compile()
    result = compiled.invoke({})
    assert result["count"] == 3
    assert result["visited_b"] is True


def test_graph_raises_on_undefined_entry_edge():
    g = StateGraph()
    g.add_node("a", lambda s: s)
    g.set_entry_point("a")
    g.add_edge("a", "nonexistent")
    with pytest.raises(GraphError):
        g.compile()


def test_graph_raises_without_entry_point():
    g = StateGraph()
    g.add_node("a", lambda s: s)
    with pytest.raises(GraphError):
        g.compile()


def test_graph_max_steps_guard_against_infinite_loop():
    g = StateGraph()
    g.add_node("a", lambda s: s)
    g.set_entry_point("a")
    g.add_edge("a", "a")  # deliberate infinite loop
    compiled = g.compile(max_steps=10)
    with pytest.raises(GraphError):
        compiled.invoke({})


def test_router_invalid_decision_raises():
    g = StateGraph()
    g.add_node("a", lambda s: s)
    g.add_node("b", lambda s: s)
    g.set_entry_point("a")
    g.add_conditional_edges("a", lambda s: "nope", {"ok": "b"})
    g.add_edge("b", END)
    compiled = g.compile()
    with pytest.raises(GraphError):
        compiled.invoke({})


# ---- full pipeline integration tests -----------------------------------
def test_pipeline_end_to_end_produces_ranked_hypotheses(mock_llm, run_config, corpus):
    pipeline = HypothesisForgePipeline(mock_llm, run_config, corpus=corpus)
    result = pipeline.run()

    assert result.total_hypotheses_considered > 0
    assert len(result.iterations) <= run_config.max_iterations
    assert len(result.ranked_hypotheses) <= run_config.top_k_final
    ranks = [rh.rank for rh in result.ranked_hypotheses]
    assert ranks == sorted(ranks)
    scores = [rh.final_score for rh in result.ranked_hypotheses]
    assert scores == sorted(scores, reverse=True)


def test_pipeline_respects_max_iterations(mock_llm, run_config, corpus):
    pipeline = HypothesisForgePipeline(mock_llm, run_config, corpus=corpus)
    result = pipeline.run()
    assert len(result.iterations) <= run_config.max_iterations


def test_pipeline_every_ranked_hypothesis_has_experiment_design(mock_llm, run_config, corpus):
    pipeline = HypothesisForgePipeline(mock_llm, run_config, corpus=corpus)
    result = pipeline.run()
    for rh in result.ranked_hypotheses:
        assert rh.experiment_design.hypothesis_id == rh.hypothesis.id
        assert rh.experiment_design.toy_simulation_code.strip() != ""


def test_pipeline_without_critic_skips_rejection(mock_llm, run_config, corpus):
    from dataclasses import replace

    cfg = replace(run_config, use_critic=False)
    pipeline = HypothesisForgePipeline(mock_llm, cfg, corpus=corpus)
    result = pipeline.run()
    for log in result.iterations:
        assert log.rejected_ids == []
        assert log.revised_ids == []
