"""
Literature Scout agent: uses the local corpus-search tool to retrieve
related work, then has the LLM synthesize a summary and gap assessment.

This agent demonstrates the "tool use" skill explicitly: it calls
CorpusIndex.search() (a deterministic, non-LLM tool) and feeds the
structured results into an LLM prompt for qualitative synthesis, rather
than letting the LLM hallucinate literature context from parametric memory.
"""

from __future__ import annotations

from ..prompts import format_literature_scout_prompts
from ..schemas import Hypothesis, LiteratureFinding
from ..tools.corpus_search import CorpusIndex
from .base import Agent


class LiteratureScoutAgent(Agent):
    name = "literature_scout"

    def __init__(self, llm, config, corpus: CorpusIndex, top_k: int = 5):
        super().__init__(llm, config)
        self.corpus = corpus
        self.top_k = top_k

    def investigate(self, hypothesis: Hypothesis) -> LiteratureFinding:
        query = f"{hypothesis.statement} {hypothesis.mechanism} {hypothesis.material_family}"
        records = self.corpus.search(query, top_k=self.top_k)

        system, user = format_literature_scout_prompts(self.domain, hypothesis, records)
        payload = self.llm.complete_json(system, user, max_tokens=700, temperature=0.4)

        return LiteratureFinding(
            hypothesis_id=hypothesis.id,
            query=query,
            records=records,
            scout_summary=payload.get("scout_summary", ""),
            closest_prior_work_gap=payload.get("closest_prior_work_gap", ""),
        )
