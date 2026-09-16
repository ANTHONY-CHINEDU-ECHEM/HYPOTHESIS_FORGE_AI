**HYPOTHESISFORGE**

A rigorously engineered multi-agent system for the automated generation, literature grounding, critical evaluation, iterative refinement, and ranked prioritisation of scientific research hypotheses.Specialised for the high-stakes domain of solid-state electrolyte materials for lithium-metal batteries, HypothesisForge does more than propose ideas. It subjects every candidate hypothesis to a disciplined cycle of evidence retrieval, multi-criteria critique, revision under explicit feedback, and experimental sketching complete with executable toy numerical simulations. 

The entire process is orchestrated by a compact, dependency-free state graph deliberately modelled on LangGraph’s programming interface.What distinguishes HypothesisForge from generic “research agent” templates is its scientific ambition: it is built to answer a measurable question—does the multi-agent revise-and-resubmit loop actually improve hypothesis quality across iterations? An integrated ablation harness systematically tests that claim by comparing configurations with and without the Critic, and across different judge personas, producing quantitative metrics and human-readable reports.

<img width="615" height="310" alt="Screenshot 2026-09-16 at 06 38 47" src="https://github.com/user-attachments/assets/6b32b105-839f-4422-a120-c405b7f365aa" />


**WHY THIS DOMAIN AND WHY GROUNDING MATTERS**

Scientific hypothesis generation is only as valuable as the constraints under which it operates. A free-form language model can invent plausible-sounding statements about almost any topic; HypothesisForge refuses that luxury. Every component is tightly coupled to a carefully engineered domain profile.

The DomainProfile class (hypothesisforge/config.py) encodes three critical layers of grounding:Scope Statement
An explicit declaration of in-scope and out-of-scope material classes. The Ideator is instructed to respect these boundaries at every proposal and revision step, preventing drift into chemically or electrochemically unrealistic territory.

**CANONICAL VOCABULARY**

A fixed set of material families (garnet oxides, sulfide argyrodites, lithium halides, polymer composites, and others) and performance metrics (ionic conductivity, critical current density, activation energy, interfacial stability, mechanical robustness, etc.). Every hypothesis must engage with these concepts; the system does not reward vague or unmeasurable claims.


**LOCAL LITERATURE CORPUS**


A curated collection of 48 original, synthetically generated but domain-realistic abstracts (data/corpus/materials_science_corpus.json). These records span the principal material families and common experimental angles: bulk conductivity, grain-boundary resistance, dopant chemistry, interfacial stability, dendrite suppression, processing effects, mechanical properties, air sensitivity, and electrochemical stability windows. The corpus is produced by build_corpus.py and is intentionally synthetic—ensuring the project ships with zero licensing ambiguity and fully reproducible outputs.

Because domain knowledge lives entirely in the DomainProfile and the associated corpus, the same agent and orchestration code can be retargeted to an entirely different scientific niche (cognitive psychology, climate modelling, catalysis, etc.) by swapping only those two assets. No agent logic or pipeline wiring needs to change.


QUICK START

HypothesisForge is designed for both offline reproducibility and live model evaluation.

# Environment setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Fully offline demonstration
# Uses a deterministic MockLLMClient — no API key required
python -m hypothesisforge.cli run --iterations 3 --seeds 4 --top-k 3

# Persist the complete structured result
python -m hypothesisforge.cli run --iterations 3 --out examples/my_run.json

# Live evaluation against the Anthropic API
export ANTHROPIC_API_KEY=sk-...
python -m hypothesisforge.cli run --live --iterations 3

# Full ablation study
python -m hypothesisforge.cli ablate \
  --out examples/ablation_report.md \
  --json-out examples/ablation_metrics.json


Sample outputs generated under the offline regime are checked into the repository (examples/sample_run_output.json and examples/sample_ablation_report.md) so that the shape and richness of the results can be inspected without executing any code.

**RUNNING THE TEST SUITE**

pip install -r requirements.txt
pytest -q

Forty-three carefully designed tests exercise every critical layer of the system:

- Schema validation for all inter-agent contracts  

- Embedding backends and corpus search correctness  

- Novelty-scoring mathematics  

- Sandboxed simulation tool (timeout enforcement, import blocking, metric parsing)  

- Graph engine isolation  

- Individual agent behavior  

- Full end-to-end pipeline execution  

- Ablation harness integrity

THE FOUR SPECIALISED AGENTS

1. Ideator (agents/ideator.py)

The Ideator is the creative engine of the system, yet creativity is deliberately constrained. It generates specific, mechanistic hypotheses that:

- Remain inside the domain’s declared scope  

- Reference only recognised material families  

- Engage measurable performance metrics  

- Explicitly avoid duplicating hypotheses already under active consideration


When the Critic returns a “revise” verdict, IdeatorAgent.revise() produces a new hypothesis linked to its predecessor via a parent_id. The revision incorporates the Critic’s concrete suggested_revision, closing the feedback loop.


2. Literature Scout (agents/literature_scout.py)

The Literature Scout is the system’s explicit tool-using agent. It never relies on the language model’s parametric memory for literature claims. Instead it:

- Invokes tools/corpus_search.py (CorpusIndex.search)—a deterministic TF-IDF / embedding similarity search over the local corpus.  

- Retrieves the most relevant records.  

- Prompts the LLM to synthesise a summary of existing knowledge and to articulate the precise gap between the current hypothesis and prior work.

This architecture guarantees that every literature statement is traceable to an actual retrieved document.

3. Critic (agents/critic.py)

The Critic performs a multi-axis evaluation of each hypothesis:

- Novelty:

Computed by tools/novelty_scorer.py. The scorer fuses an objective embedding-similarity signal with an LLM-as-judge conceptual/mechanistic score. This hybrid design mitigates the classic failure modes of each signal in isolation: pure embedding methods treat paraphrases of known work as “similar,” while pure LLM judges tend to be over-generous without grounding.  

- Feasibility:

An LLM judge assesses experimental practicality in light of the Literature Scout’s findings.  
Contradiction Detection, The same judge identifies logical or empirical conflicts with retrieved literature.

Crucially, the final overall_score is recomputed programmatically from configurable weights in RunConfig (novelty_weight, feasibility_weight, contradiction_penalty_weight). The system never trusts the LLM’s own arithmetic. This design choice makes scoring policy fully transparent and tuneable for ablation studies.


4. Synthesiser (agents/synthesizer.py)

For every hypothesis that survives critique, the Synthesiser delivers two concrete artifacts:

- A detailed experimental protocol specifying steps, independent and dependent variables, and controls.  

- A short, self-contained toy numerical simulation (for example, a simple Arrhenius conductivity model).

The simulation is executed inside the restricted environment provided by tools/simulation.py. Successfully executing simulations receive a modest ranking bonus; any key=value metrics printed to stdout are parsed and attached to the final result. Surviving hypotheses are then ranked by their composite score.

**MULTI AGENT ORCHESTRATION**

At the centre of HypothesisForge lies a minimal, dependency-free graph engine (orchestration/graph.py) that deliberately mirrors LangGraph’s programming model:

- Named nodes that transform state (state → state)  

- Explicit edges (add_edge)  

- Conditional edges driven by a router function (add_conditional_edges)  

- A reserved END sentinel  

- Compilation and invocation (.compile().invoke(state))

A max_steps guard prevents runaway loops should a router be misconfigured. No external orchestration package is required to run the system.


**HYPOTHESISFORGE PIPELINE** (orchestration/pipeline.py)

wires the four agents into the following control flow:

<img width="729" height="57" alt="Screenshot 2026-09-16 at 06 40 41" src="https://github.com/user-attachments/assets/5671e56f-3c5f-4cec-87f2-f6b1689818aa" />


The router (_route_after_critic) continues the loop while iteration budget remains and either:Hypotheses are queued for revision, or Fewer hypotheses have been accepted than the target top_k_final.

Revised hypotheses are re-proposed by the Ideator (linked via parent_id) and re-evaluated by Scout → Critic, forming a closed scientific feedback cycle.


**SEAMLESS MIGRATION TO REAL LANGRAPH**

Because the local graph API surface matches LangGraph’s, replacing the implementation is a mechanical change:

from langgraph.graph import StateGraph, END

Adapting PipelineState to a TypedDict is the only additional step; the remainder of pipeline.py continues to function unchanged.

NOVELTY SCORING IN DETAIL

The hybrid novelty function (tools/novelty_scorer.compute_novelty()) is defined as:


embedding_component = 10 × (1 − max_similarity^0.7)     # scaled to [0, 10]
combined_score = (embedding_component × embedding_weight
                  + llm_judge_score × judge_weight)
                 / (embedding_weight + judge_weight)


SIMULATION ENVIRONMENT

Toy simulations authored by the Synthesiser are executed by tools/simulation.run_toy_simulation() under strict sandbox constraints:

- A curated allowlist of builtins  

- An import allowlist limited to math, statistics, random, itertools, functools, and numpy (if installed)  

- Wall-clock timeout enforced by a daemon worker thread  

- Stdout-only capture (no filesystem or network access is exposed)

Printed lines of the form key=value are automatically parsed into structured metrics attached to the SimulationResult. The environment is intentionally narrow: it functions as a rapid order-of-magnitude sanity check, not a general-purpose code-execution sandbox, and is not intended for untrusted external code.


hypothesisforge/

├── config.py              Domain profiles and RunConfig (all tunable knobs)

├── schemas.py             Pydantic contracts exchanged between agents

├── llm_client.py          AnthropicLLMClient (live) + MockLLMClient (offline/deterministic)

├── prompts.py             Per-agent system and user prompt templates

├── agents/                Ideator · Literature Scout · Critic · Synthesizer

├── tools/                 Embeddings · corpus search · novelty scorer · sandboxed simulation

├── orchestration/         graph.py (LangGraph-style engine) · state.py · pipeline.py

├── evaluation/            metrics.py · ablation.py

└── cli.py                 `run` and `ablate` entry points

data/corpus/               48-record synthetic materials-science literature corpus

build_corpus.py            Corpus generation script (reference and regeneration)

tests/                     43 tests spanning schemas, tools, agents, orchestration, ablation

examples/                  Sample offline run output and ablation report




KNOWN LIMITATIONS AND DESIGN TRADE-OFFS

- The offline MockLLMClient produces plausible, schema-valid, but templated text. It faithfully demonstrates pipeline mechanics—agent hand-offs, scoring mathematics, the iteration loop, and ablation comparisons—yet hypothesis content quality should be assessed using the live mode (--live) against a real language model.  

- The local literature corpus is synthetic by design. For research-grade deployments, the retrieval backend of tools/corpus_search.py can be replaced by a real literature API (Semantic Scholar, arXiv, etc.) while preserving the identical CorpusIndex.search() interface.  

- Toy simulations are intentionally simple order-of-magnitude sanity checks (e.g., bare Arrhenius models). They are not substitutes for density-functional theory, molecular dynamics, or laboratory experimentation.


HypothesisForge is more than a multi-agent demo. It is an experimental apparatus for studying whether structured critique and iterative revision can elevate the quality of machine-generated scientific hypotheses in a domain where precision, novelty, and feasibility truly matter.













