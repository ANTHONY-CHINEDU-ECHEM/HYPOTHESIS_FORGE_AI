"""
Wires the four agents into the graph engine and implements the
ideate -> scout -> critique -> (revise loop) -> synthesize control flow.

Graph shape:

    ideate --> scout --> critic --(router)--> ideate      [more revision/iterations needed]
                                           `-> synthesize --> END

The router at `critic` decides, per iteration:
  - if there is remaining iteration budget AND (revision queue is non-empty
    OR fewer accepted hypotheses than requested) -> loop back to `ideate`
  - else -> proceed to `synthesize`

`use_critic=False` (the ablation switch) short-circuits the critic node to a
neutral pass-through so every proposed hypothesis is accepted without
LLM-judged feasibility/contradiction screening or revision — this isolates
the Critic's contribution when compared against a `use_critic=True` run.
"""

from __future__ import annotations

import time

from ..agents.critic import CriticAgent
from ..agents.ideator import IdeatorAgent
from ..agents.literature_scout import LiteratureScoutAgent
from ..agents.synthesizer import SynthesizerAgent
from ..config import RunConfig
from ..llm_client import LLMClient
from ..schemas import (
    CritiqueVerdict,
    Hypothesis,
    IterationLog,
    NoveltyScore,
    Recommendation,
    RunResult,
)
from ..tools.corpus_search import CorpusIndex
from .graph import END, StateGraph
from .state import PipelineState


def _neutral_critique(hyp: Hypothesis, corpus: CorpusIndex) -> CritiqueVerdict:
    """Used only when use_critic=False: accept everything, no LLM judging."""
    stats = corpus.similarity_stats(f"{hyp.statement} {hyp.mechanism}", top_k=5)
    from ..tools.novelty_scorer import embedding_novelty_component

    embed_score = embedding_novelty_component(stats["max_sim"])
    novelty = NoveltyScore(
        embedding_max_similarity=stats["max_sim"],
        embedding_mean_similarity=stats["mean_sim"],
        nearest_neighbor_ids=[r.doc_id for r in stats["records"]],
        llm_judge_score=embed_score,  # no separate LLM judge call in this ablation branch
        llm_judge_rationale="(critic disabled for this run — embedding-only novelty proxy used)",
        combined_score=embed_score,
    )
    return CritiqueVerdict(
        hypothesis_id=hyp.id,
        novelty=novelty,
        feasibility_score=5.0,
        feasibility_rationale="(critic disabled for this run — no feasibility review performed)",
        contradictions=[],
        identified_flaws=[],
        suggested_revision=None,
        overall_score=round((novelty.combined_score + 5.0) / 2, 2),
        recommendation=Recommendation.ACCEPT,
    )


class HypothesisForgePipeline:
    def __init__(self, llm: LLMClient, config: RunConfig, corpus: CorpusIndex | None = None):
        self.llm = llm
        self.config = config
        self.corpus = corpus or CorpusIndex(
            config.domain_profile().corpus_path, embedding_backend=config.embedding_backend
        )
        self.ideator = IdeatorAgent(llm, config)
        self.scout = LiteratureScoutAgent(llm, config, self.corpus)
        self.critic = CriticAgent(llm, config, self.corpus)
        self.synthesizer = SynthesizerAgent(llm, config)
        self._graph = self._build_graph()

    # ---- graph nodes --------------------------------------------------
    def _node_ideate(self, state: PipelineState) -> PipelineState:
        state.iteration += 1

        revised: list[Hypothesis] = []
        for hyp, suggestion in state.revision_queue:
            revised.append(self.ideator.revise(hyp, suggestion, iteration=state.iteration))
        state.revision_queue = []

        n_fresh = max(0, self.config.n_seed_hypotheses - len(revised))
        fresh: list[Hypothesis] = []
        if n_fresh > 0:
            fresh = self.ideator.generate(
                n=n_fresh,
                iteration=state.iteration,
                existing=state.all_hypotheses,
                critic_feedback=state.feedback_pool[-5:],
            )

        state.register_hypotheses(revised + fresh)
        return state

    def _node_scout(self, state: PipelineState) -> PipelineState:
        state.findings_by_hyp_id = {}
        for hyp in state.active_hypotheses:
            state.findings_by_hyp_id[hyp.id] = self.scout.investigate(hyp)
        return state

    def _node_critic(self, state: PipelineState) -> PipelineState:
        state.critiques_by_hyp_id = {}
        accepted_ids, revised_ids, rejected_ids = [], [], []

        for hyp in state.active_hypotheses:
            finding = state.findings_by_hyp_id[hyp.id]
            if self.config.use_critic:
                verdict = self.critic.critique(hyp, finding)
            else:
                verdict = _neutral_critique(hyp, self.corpus)
            state.critiques_by_hyp_id[hyp.id] = verdict

            if verdict.recommendation == Recommendation.ACCEPT:
                hyp.status = hyp.status.__class__.ACCEPTED
                state.accepted.append(hyp)
                accepted_ids.append(hyp.id)
            elif verdict.recommendation == Recommendation.REVISE:
                hyp.status = hyp.status.__class__.REVISED
                if verdict.suggested_revision and state.iteration < state.max_iterations:
                    state.revision_queue.append((hyp, verdict.suggested_revision))
                    state.feedback_pool.append(verdict.suggested_revision)
                    revised_ids.append(hyp.id)
                else:
                    # No budget left to revise further; accept as-is with a note,
                    # rather than silently discarding work.
                    state.accepted.append(hyp)
                    accepted_ids.append(hyp.id)
            else:
                hyp.status = hyp.status.__class__.REJECTED
                state.rejected.append(hyp)
                rejected_ids.append(hyp.id)
                if verdict.identified_flaws:
                    state.feedback_pool.extend(verdict.identified_flaws[:1])

        state.iteration_logs.append(
            IterationLog(
                iteration_number=state.iteration,
                proposed=list(state.active_hypotheses),
                findings=list(state.findings_by_hyp_id.values()),
                critiques=list(state.critiques_by_hyp_id.values()),
                accepted_ids=accepted_ids,
                revised_ids=revised_ids,
                rejected_ids=rejected_ids,
            )
        )
        return state

    def _node_synthesize(self, state: PipelineState) -> PipelineState:
        state.done = True
        return state

    # ---- routing --------------------------------------------------------
    def _route_after_critic(self, state: PipelineState) -> str:
        budget_left = state.iteration < state.max_iterations
        needs_more = len(state.accepted) < self.config.top_k_final
        if budget_left and (state.revision_queue or needs_more):
            return "continue"
        return "finish"

    def _build_graph(self) -> StateGraph[PipelineState]:
        graph: StateGraph[PipelineState] = StateGraph()
        graph.add_node("ideate", self._node_ideate)
        graph.add_node("scout", self._node_scout)
        graph.add_node("critic", self._node_critic)
        graph.add_node("synthesize", self._node_synthesize)

        graph.set_entry_point("ideate")
        graph.add_edge("ideate", "scout")
        graph.add_edge("scout", "critic")
        graph.add_conditional_edges(
            "critic",
            self._route_after_critic,
            {"continue": "ideate", "finish": "synthesize"},
        )
        graph.add_edge("synthesize", END)
        return graph.compile()

    # ---- public entrypoint ------------------------------------------
    def run(self) -> RunResult:
        start = time.monotonic()
        state = PipelineState(max_iterations=self.config.max_iterations)
        state = self._graph.invoke(state)

        # Synthesize: design + rank experiments for accepted hypotheses.
        survivors = []
        for hyp in state.accepted:
            critique = state.critiques_by_hyp_id.get(hyp.id)
            if critique is None:
                # Hypothesis accepted in an earlier iteration; critique is
                # still recorded in that iteration's log.
                for log in state.iteration_logs:
                    for c in log.critiques:
                        if c.hypothesis_id == hyp.id:
                            critique = c
                            break
                    if critique:
                        break
            if critique is None:
                continue
            design = self.synthesizer.design_experiment(hyp, critique)
            survivors.append((hyp, critique, design))

        ranked = self.synthesizer.rank(survivors)
        ranked = ranked[: self.config.top_k_final]

        wall_clock = round(time.monotonic() - start, 3)
        return RunResult(
            domain=self.config.domain,
            config_snapshot=self._config_snapshot(),
            iterations=state.iteration_logs,
            ranked_hypotheses=ranked,
            total_hypotheses_considered=len(state.all_hypotheses),
            wall_clock_seconds=wall_clock,
        )

    def _config_snapshot(self) -> dict:
        c = self.config
        return {
            "domain": c.domain,
            "n_seed_hypotheses": c.n_seed_hypotheses,
            "max_iterations": c.max_iterations,
            "top_k_final": c.top_k_final,
            "novelty_weight": c.novelty_weight,
            "feasibility_weight": c.feasibility_weight,
            "contradiction_penalty_weight": c.contradiction_penalty_weight,
            "critic_reject_threshold": c.critic_reject_threshold,
            "critic_revise_threshold": c.critic_revise_threshold,
            "embedding_backend": c.embedding_backend,
            "llm_model": self.llm.model_name,
            "judge_persona": c.judge_persona,
            "use_critic": c.use_critic,
        }
