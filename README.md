# HypothesisForge

**Multi-agent scientific hypothesis generator & critic**, domain-grounded in
**solid-state electrolyte materials for lithium-metal batteries**.

Four specialized agents — **Ideator**, **Literature Scout**, **Critic**, and
**Synthesizer** — run in an iterative loop, orchestrated by a small
LangGraph-style state graph, to propose, ground, critique, revise, and rank
research hypotheses, then sketch an experimental protocol (including a toy
numerical simulation) for the survivors.

This is not a generic "research agent" template: it is built around
**measuring whether the multi-agent loop actually improves hypothesis
quality across iterations**, with a built-in ablation harness to test that
claim (with/without Critic, and across judge personas).

```
                 ┌────────────┐
      ┌─────────▶│  Ideator   │  proposes / revises hypotheses
      │          └─────┬──────┘
      │                │
 loop back        ┌────▼──────┐
 (revise or       │ Lit Scout │  local corpus search (tool use) + LLM synthesis
 more budget)     └────┬──────┘
      │                │
      │          ┌─────▼──────┐
      └──────────┤   Critic   │  novelty (embeddings + LLM judge) + feasibility
                 └─────┬──────┘   + contradiction check
                       │ accept
                 ┌─────▼───────┐
                 │ Synthesizer │  experiment design + toy simulation + ranking
                 └─────────────┘
```

## Why this domain, and why it's grounded rather than generic

The domain profile (`hypothesisforge/config.py::DomainProfile`) encodes:
- A precise **scope statement** the Ideator is instructed to respect (in/out
  of scope material classes).
- A fixed vocabulary of **known material families** (garnet oxides,
  sulfide argyrodites, lithium halides, polymer composites, etc.) and
  **key metrics** (ionic conductivity, critical current density,
  activation energy, ...) that every hypothesis must engage with.
- A **48-record local literature corpus**
  (`data/corpus/materials_science_corpus.json`) of original, synthetically
  generated but domain-realistic abstracts spanning these material
  families and common experimental angles (bulk conductivity, grain
  boundary resistance, dopant chemistry, interfacial stability, dendrite
  suppression, processing effects, mechanical properties, air
  sensitivity, electrochemical stability window). See
  `build_corpus.py` for the generation logic — this corpus is intentionally
  synthetic (not scraped/copied from real papers) so the project ships
  with no licensing ambiguity and fully reproducible outputs.

Swapping `DomainProfile` + corpus retargets the whole system at a different
narrow domain (cognitive psychology, climate modeling, ...) with **zero
changes to agent or orchestration code** — see "Retargeting to a new
domain" below.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Fully offline demo run (deterministic mock LLM, no API key required):
python -m hypothesisforge.cli run --iterations 3 --seeds 4 --top-k 3

# Save the full structured result:
python -m hypothesisforge.cli run --iterations 3 --out examples/my_run.json

# Run against the real Anthropic API instead:
export ANTHROPIC_API_KEY=sk-...
python -m hypothesisforge.cli run --live --iterations 3

# Ablation study (with/without critic, judge personas):
python -m hypothesisforge.cli ablate --out examples/ablation_report.md \
                                       --json-out examples/ablation_metrics.json
```

Sample outputs from an offline run are checked in at
`examples/sample_run_output.json` and `examples/sample_ablation_report.md`
so you can see the shape of the output without running anything.

## Running the tests

```bash
pip install -r requirements.txt
pytest -q
```

43 tests cover: schema validation, embedding/corpus search, novelty scoring
math, the sandboxed simulation tool (including timeout and import-blocking
behavior), the graph engine in isolation, each agent individually, the full
pipeline end-to-end, and the ablation harness.

## The four agents

### Ideator (`agents/ideator.py`)
Proposes new, specific, mechanistic hypotheses constrained to the domain's
scope, material families, and metrics — and explicitly avoids duplicating
hypotheses already under consideration. On a "revise" verdict from the
Critic, `IdeatorAgent.revise()` produces a new hypothesis linked via
`parent_id`, incorporating the Critic's `suggested_revision`.

### Literature Scout (`agents/literature_scout.py`)
The explicit **tool-use** agent: calls `tools/corpus_search.py`
(`CorpusIndex.search`, a deterministic TF-IDF/embedding similarity search
over the local corpus — no LLM involved) to retrieve the most related
records, then prompts the LLM to *synthesize* — summarize what's already
known and identify the specific gap between the hypothesis and prior work.
The LLM never invents literature from parametric memory; it only reasons
over what the tool actually retrieved.

### Critic (`agents/critic.py`)
Checks **novelty**, **feasibility**, and **contradictions**:
- Novelty is computed by `tools/novelty_scorer.py`, which fuses an
  **embedding-similarity signal** (objective, lexical/semantic) with an
  **LLM-as-judge score** (conceptual/mechanistic novelty) — the classic
  failure mode of embedding-only novelty scoring (a paraphrase of existing
  work scores as "similar" even if never explicitly framed that way) is
  mitigated by the judge component, and vice versa (LLM judges are
  over-generous without a grounding signal).
- Feasibility and contradiction-checking are pure LLM-judge calls, reading
  the Literature Scout's findings.
- `overall_score` is **recomputed programmatically** from `RunConfig`
  weights (`novelty_weight`, `feasibility_weight`,
  `contradiction_penalty_weight`) rather than trusted verbatim from the
  LLM's own arithmetic — this is what makes the scoring policy tunable for
  the ablation study instead of being an opaque model output.

### Synthesizer (`agents/synthesizer.py`)
For every hypothesis that survives critique, produces a concrete
experimental protocol (steps, independent/dependent variables, controls)
**and** a short, self-contained **toy simulation** (e.g. an Arrhenius
conductivity model) which is actually executed via the sandboxed
`tools/simulation.py` tool. Successfully-executing simulations receive a
small ranking bonus, and parsed `key=value` metrics from simulation stdout
are attached to the result. Finally ranks all survivors by final score.

## Multi-agent orchestration

`orchestration/graph.py` implements a small, dependency-free graph engine
deliberately modeled on **LangGraph's** programming model: named nodes
(`state -> state` functions), `add_edge`, `add_conditional_edges` (a router
function returning a path key), a reserved `END` sentinel, and
`.compile().invoke(state)`. No external orchestration package is required
to run this project — a `max_steps` guard also prevents runaway loops if a
router is misconfigured.

`orchestration/pipeline.py` (`HypothesisForgePipeline`) wires the four
agents into this graph:

```
ideate → scout → critic ──router──▶ ideate      (more revision/iteration budget)
                              └────▶ synthesize → END
```

The router (`_route_after_critic`) loops back to `ideate` while iteration
budget remains **and** either (a) hypotheses are queued for revision, or
(b) fewer hypotheses have been accepted than `top_k_final` requests.
Revised hypotheses are re-proposed by the Ideator (linked via `parent_id`)
and re-run through Scout → Critic in the next iteration, closing the loop.

### Swapping in real LangGraph

Because `StateGraph.add_node` / `add_edge` / `add_conditional_edges` /
`compile()` mirror LangGraph's actual API surface, replacing
`orchestration/graph.py`'s implementation with:

```python
from langgraph.graph import StateGraph, END
```

and adapting `PipelineState` to a `TypedDict` (LangGraph's expected state
shape) is a small, mechanical change — the rest of `pipeline.py` does not
need to change.

## Novelty scoring in detail

`tools/novelty_scorer.compute_novelty()`:

```
embedding_component = 10 * (1 - max_similarity ** 0.7)   # in [0, 10]
combined_score = (embedding_component * embedding_weight
                   + llm_judge_score * judge_weight) / (embedding_weight + judge_weight)
```

Default weights: `embedding_weight=0.4`, `judge_weight=0.6` (the judge gets
more say since it can detect conceptual novelty the embedding can't). Both
weights, and the embedding backend itself, are configurable — see
`--embedding-backend sentence-transformers` for a higher-fidelity (but
model-download-requiring) alternative to the default TF-IDF backend.

## Simulation hooks

`tools/simulation.run_toy_simulation()` executes Synthesizer-authored
Python snippets in a restricted namespace: a curated builtins allowlist, an
import allowlist (`math`, `statistics`, `random`, `itertools`,
`functools`, plus `numpy` if installed), a wall-clock timeout enforced via
a daemon worker thread, and stdout-only capture (no filesystem/network
access is exposed through the allowed builtins). `key=value` lines printed
to stdout are parsed into `SimulationResult.parsed_metrics`. This is
intentionally narrow — it is a toy-model sanity-check tool, not a
general-purpose code execution sandbox, and is not intended to execute
untrusted code from outside this pipeline.

## Ablation studies

```bash
python -m hypothesisforge.cli ablate --iterations 3 --seeds 4
```

Runs four variants against the same corpus and (for the mock LLM) the same
random seed basis, so proposals are comparable:

| Variant | What changes |
|---|---|
| `baseline` | Critic enabled, balanced judge persona |
| `no_critic` | Critic step neutralized — every hypothesis auto-accepted with only an embedding-based novelty proxy, no feasibility/contradiction screening, no revision loop |
| `strict_judge` | Critic enabled, harsher scoring persona |
| `lenient_judge` | Critic enabled, more generous scoring persona |

`evaluation/metrics.py` computes acceptance/rejection/revision rates, mean
novelty/feasibility/overall scores, mean contradictions per hypothesis,
**per-iteration mean overall score** (to show whether the revise-and-
resubmit loop measurably improves quality across iterations —
`iteration_over_iteration_delta`), toy-simulation success rate, and wall
clock time. `evaluation/ablation.render_ablation_report_markdown()` turns
this into a comparison table + interpretation notes (see
`examples/sample_ablation_report.md` for real output).

To use genuinely different judge *models* (rather than personas) against
the live API, set a different `RunConfig.llm_model` per variant in your own
script calling `run_ablation_study()` — the harness structure is identical.

## Retargeting to a new domain

1. Add a new `DomainProfile` to `hypothesisforge/config.py` and register it
   in `DOMAIN_REGISTRY`.
2. Write (or generate, see `build_corpus.py` as a template) a local corpus
   JSON file at the path your new profile's `corpus_path` points to, with
   `{"domain": ..., "records": [{"doc_id", "title", "year", "tags",
   "abstract"}, ...]}`.
3. Nothing in `agents/`, `orchestration/`, or `tools/` needs to change —
   prompts pull scope/families/metrics from `DomainProfile` at call time.

## Project layout

```
hypothesisforge/
  config.py            domain profile(s) + RunConfig (all tunable knobs)
  schemas.py            Pydantic contracts passed between agents
  llm_client.py          AnthropicLLMClient (live) + MockLLMClient (offline/deterministic)
  prompts.py             per-agent system/user prompt templates
  agents/                Ideator, Literature Scout, Critic, Synthesizer
  tools/                 embeddings, corpus search, novelty scorer, sandboxed simulation
  orchestration/         graph.py (LangGraph-style engine), state.py, pipeline.py
  evaluation/            metrics.py, ablation.py
  cli.py                 `run` and `ablate` commands
data/corpus/              48-record synthetic literature corpus (materials science)
build_corpus.py            script that generated the corpus (for reference/regeneration)
tests/                     43 tests across schemas, tools, agents, orchestration, ablation
examples/                  sample run output + sample ablation report (offline mode)
```

## Known limitations

- The offline `MockLLMClient` produces plausible, schema-valid, but
  templated text — it demonstrates the *pipeline mechanics* (agent
  hand-offs, scoring math, iteration loop, ablation comparisons)
  faithfully, but hypothesis *content* quality should be evaluated using
  `--live` against a real model.
- The local corpus is synthetic (see above) — for a research-grade
  deployment, swap `tools/corpus_search.py` for a real literature API
  (Semantic Scholar, arXiv) behind the same `CorpusIndex.search()`
  interface.
- The toy simulations are order-of-magnitude sanity checks (e.g. a bare
  Arrhenius model), not substitutes for DFT/MD or real lab work.
