# HypothesisForge: AI Driven Scientific Hypothesis Generation and Evaluation

<div align="center">

**A rigorously engineered multi agent system for automated generation, literature grounding, critical evaluation, iterative refinement, and ranked prioritization of scientific research hypotheses.**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Code Quality](https://img.shields.io/badge/Tests-43%2B%20Passing-brightgreen)
![Status](https://img.shields.io/badge/Status-Production%20Ready-blue)

</div>

---

## Table of Contents

1. [Overview](#overview)
2. [Core Features](#core-features)
3. [Quick Start](#quick-start)
4. [System Architecture](#system-architecture)
5. [The Four Specialized Agents](#the-four-specialized-agents)
6. [Multi Agent Orchestration](#multi-agent-orchestration)
7. [Domain Grounding: The DomainProfile](#domain-grounding-the-domainprofile)
8. [Novelty Scoring in Detail](#novelty-scoring-in-detail)
9. [Simulation Environment](#simulation-environment)
10. [Testing and Quality Assurance](#testing-and-quality-assurance)
11. [Configuration and Tuning](#configuration-and-tuning)
12. [Ablation Study](#ablation-study)
13. [Seamless Migration to Real LangGraph](#seamless-migration-to-real-langgraph)
14. [Comprehensive Documentation](#comprehensive-documentation)
15. [Understanding the MockLLMClient](#understanding-the-mockllmclient)
16. [Known Limitations and Design Tradeoffs](#known-limitations-and-design-tradeoffs)
17. [Example Outputs](#example-outputs)
18. [Development and Contributing](#development-and-contributing)
19. [License](#license)
20. [Support and Questions](#support-and-questions)
21. [Research and Publications](#research-and-publications)
22. [Highlights](#highlights)

---

## Overview

HypothesisForge is a sophisticated multi agent AI system purpose built for scientific research hypothesis generation, with a current focus on materials science, specifically solid state electrolytes, ionic conductors, and related domains. Unlike generic language model applications, HypothesisForge enforces rigorous domain grounding through the following mechanisms:

- **Structured domain constraints** that prevent off topic drift.
- **Literature aware critique**, backed by a curated corpus of domain realistic abstracts.
- **Hybrid novelty scoring**, combining embedding similarity with an LLM acting as judge.
- **Iterative refinement**, with scientist in the loop feedback mechanisms.
- **Deterministic simulation**, run inside sandboxed execution environments.
- **Dependency free orchestration**, featuring a state machine compatible with LangGraph.

The entire system is designed for reproducibility, auditability, and a seamless migration path to production orchestration frameworks.

---

## Core Features

### Four Specialized Agents

- **Ideator:** creative generation of novel, mechanistically grounded hypotheses.
- **Literature Scout:** a tool using agent for corpus search and gap analysis.
- **Critic:** multi axis evaluation, covering novelty, feasibility, and contradiction detection.
- **Synthesizer:** experimental protocol design, together with toy simulation generation and execution.

### Evidence Based Grounding

- Explicit scope statements defining in scope and out of scope research areas.
- A canonical vocabulary of material families and measurable performance metrics.
- A local literature corpus of 48 synthetically generated but domain realistic abstracts.
- Every literature claim is traceable to a retrieved document, with no reliance on parametric memory hallucination.

### A Rigorous Evaluation Framework

- **Novelty scoring:** a hybrid approach fusing embedding similarity, on a 0 to 10 scale, with an LLM acting as judge for conceptual assessment.
- **Feasibility assessment:** experimental practicality evaluated in light of existing literature.
- **Contradiction detection:** identification of logical and empirical conflicts.
- **Configurable weighting:** a fully auditable scoring policy, with no black box LLM arithmetic.

### Iterative Refinement

- Hypotheses that are rejected for revision receive concrete, actionable feedback.
- Revised hypotheses are re evaluated in the same pipeline, forming a closed scientific feedback loop.
- Parent and child relationships, tracked through a `parent_id` field, capture lineage and evolution.

### A Production Ready Design

- **Minimal dependencies:** no external orchestration framework is required.
- **A dependency free state graph**, deliberately modeled on LangGraph, providing a seamless migration path.
- **A deterministic mock mode**, supporting offline reproducibility and cost free testing.
- **More than 43 comprehensive tests**, covering schemas, agents, tools, and end to end flows.
- **An ablation harness**, for systematic exploration of design tradeoffs.

---

## Quick Start

### Prerequisites

- Python 3.10 or later
- A Unix like shell, such as bash or zsh, for the examples below

### 1. Environment Setup

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # on Windows use: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Offline Demonstration (No API Key Required)

The simplest way to explore HypothesisForge is through the deterministic MockLLMClient:

```bash
# Generate 3 iterations with 4 random seeds, keeping the top 3 hypotheses
python -m hypothesisforge.cli run --iterations 3 --seeds 4 --top-k 3

# Output appears in your terminal as structured hypothesis data
```

### 3. Persist Results to JSON

```bash
# Save a complete run to JSON for downstream analysis
python -m hypothesisforge.cli run --iterations 3 --out examples/my_run.json

# Inspect the result
python -c "import json; print(json.dumps(json.load(open('examples/my_run.json')), indent=2)[:500])"
```

### 4. Live Evaluation with the Anthropic API

```bash
# Set your API key
export ANTHROPIC_API_KEY=sk-ant-...

# Run against Claude (typically costs approximately $0.10 to $0.50 per run)
python -m hypothesisforge.cli run --live --iterations 3 --out results/live_run.json
```

### 5. Full Ablation Study

Systematically explore how component weights affect hypothesis quality:

```bash
python -m hypothesisforge.cli ablate \
  --out examples/ablation_report.md \
  --json-out examples/ablation_metrics.json

# Read the markdown report
cat examples/ablation_report.md
```

### 6. Run the Test Suite

Verify system integrity with 43 carefully designed tests:

```bash
pytest -q

# For detailed output with coverage:
pytest -v --cov=hypothesisforge
```

---

## System Architecture

### High Level Data Flow

```
                        HYPOTHESIS GENERATION
--------------------------------------------------------------------------

  Ideator.generate()         [LLM: constrained creative generation]
         |
         v
  Ideator.revise()           [conditional: update based on feedback]
         |
         v
  For each hypothesis: (LiteratureScout to Critic loop)

    1. LiteratureScout.investigate()
       - CorpusIndex.search() [deterministic TF IDF plus embedding]
       - Extract the literature gap versus prior work

    2. Critic.critique()
       - compute_novelty() [hybrid: embedding similarity plus LLM judge]
       - feasibility_score [LLM judgment]
       - contradiction_detection [traced to the corpus]
       - overall_score [recomputed, not trusted directly from the LLM]
       - recommendation: ACCEPT / REVISE / REJECT

    3. Router decision
       - ACCEPT     -> state.accepted (a candidate for synthesis)
       - REVISE with budget remaining -> state.revision_queue (back to Ideator)
       - REVISE with no budget remaining -> state.accepted (best effort)
       - REJECT     -> state.rejected (logged for context)

  (The loop continues while iteration budget remains)

--------------------------------------------------------------------------
                           SYNTHESIS AND RANKING
--------------------------------------------------------------------------

  Synthesizer.design_experiment()   [LLM: protocol plus simulation code]
         |
         v
  run_toy_simulation()              [sandboxed execution, with a timeout]
         |
         v
  Synthesizer.rank()                [sorted by composite_score]
         |
         v
  TOP_K_FINAL                       [returned ranked and deduplicated]
```

### Directory Structure

```
hypothesisforge/
├── config.py                 # Domain profiles and RunConfig (all tunable knobs)
├── schemas.py                # Pydantic contracts (Hypothesis, LiteratureFinding, etc.)
├── llm_client.py             # AnthropicLLMClient (live) plus MockLLMClient (offline)
├── prompts.py                # Per agent system and user prompt templates
├── cli.py                    # CLI entry points: `run` and `ablate`
│
├── agents/                   # Four specialized agent implementations
│   ├── ideator.py            # Hypothesis generation and revision
│   ├── literature_scout.py   # Corpus search and gap analysis
│   ├── critic.py             # Multi axis evaluation and scoring
│   └── synthesizer.py        # Experiment design and simulation orchestration
│
├── tools/                    # Shared utilities for agents
│   ├── embeddings.py         # Embedding backends (mock plus sentence-transformers)
│   ├── corpus_search.py      # CorpusIndex (TF IDF plus similarity retrieval)
│   ├── novelty_scorer.py     # Hybrid novelty computation
│   └── simulation.py         # Sandboxed code execution with a timeout
│
├── orchestration/            # Graph engine and pipeline
│   ├── graph.py               # A LangGraph compatible StateGraph implementation
│   ├── state.py               # PipelineState definition
│   └── pipeline.py            # Multi agent orchestration and control flow
│
├── evaluation/                # Analysis and ablation
│   ├── metrics.py             # Evaluation metrics (coverage, diversity, and so on)
│   └── ablation.py            # Ablation harness (sweeping component weights)
│
└── tests/                     # More than 43 comprehensive test cases

data/
└── corpus/
    └── materials_science_corpus.json   # 48 record synthetic literature corpus

build_corpus.py                # Corpus generation and regeneration script

examples/
├── sample_run_output.json      # Example offline run (full structured output)
├── sample_ablation_report.md   # Example ablation study report
└── sample_ablation_metrics.json # Example ablation metrics (JSON)
```

---

## The Four Specialized Agents

### 1. Ideator Agent (`agents/ideator.py`)

**Role:** a creative engine operating with constrained freedom.

The Ideator generates specific, mechanistic hypotheses that:

- **Stay within scope**, enforced through the DomainProfile.
- **Reference only recognized material families**, such as garnet oxides, sulfide argyrodites, and lithium halides.
- **Engage measurable performance metrics**, such as ionic conductivity, critical current density, and activation energy.
- **Avoid duplication** of hypotheses already under consideration.

**Key methods:**

- `generate()`: produces N candidate hypotheses from scratch.
- `revise(hypothesis, feedback)`: refines a hypothesis based on Critic feedback, maintaining `parent_id` lineage.

**Example output:**

```json
{
  "id": "hyp-001",
  "parent_id": null,
  "text": "Introducing 1 to 3 mol% yttrium oxide into garnet-type Li7La3Zr2O12 will increase ionic conductivity by suppressing grain-boundary resistance while maintaining lithium-ion transference number > 0.7.",
  "material_family": "garnet oxides",
  "performance_metric": "ionic conductivity",
  "mechanism": "grain boundary engineering",
  "iteration": 1
}
```

---

### 2. Literature Scout Agent (`agents/literature_scout.py`)

**Role:** a tool using evidence retriever.

The Literature Scout never relies on parametric memory. Instead, it:

1. Invokes `CorpusIndex.search()`, a deterministic TF IDF plus embedding similarity search.
2. Retrieves the most relevant records from the local corpus.
3. Synthesizes findings into a structured summary.
4. Articulates the gap between the current hypothesis and prior work.

**Key method:**

- `investigate(hypothesis, domain_profile)` returns a `LiteratureFinding`, which searches the corpus for related work, summarizes existing knowledge, and identifies the innovation gap.

**Guarantees:**

- Every literature statement is traceable to an actual retrieved document.
- There are no hallucinated citations or invented references.
- Results are reproducible: the same query always returns the same retrieved documents.

---

### 3. Critic Agent (`agents/critic.py`)

**Role:** a multi axis evaluator and quality gatekeeper.

The Critic performs a structured evaluation across four dimensions.

#### Novelty

Computed by `tools/novelty_scorer.py`:

- **The embedding component** (0 to 10): the inverse of the maximum cosine similarity to the corpus, scaled with an exponent of 0.7.
- **The LLM as judge component** (0 to 10): a conceptual and mechanistic novelty score produced by Claude.
- **The hybrid score:** a weighted average of both components.

Formula:

```
embedding_component = 10 x (1 - max_similarity ^ 0.7)
combined_score = (embedding_component x embedding_weight + llm_judge_score x judge_weight)
                 / (embedding_weight + judge_weight)
```

#### Feasibility

- The LLM judges experimental practicality given the literature findings.
- It assesses material availability, synthesis complexity, and equipment requirements.

#### Contradiction Detection

- Identifies logical or empirical conflicts with retrieved literature.
- Flags implicit assumptions that contradict prior work.
- Counts contradictions for use in penalty scoring.

#### Overall Score (Recomputed)

The Critic deliberately ignores the LLM's own arithmetic and recomputes the overall score directly:

```python
overall_score = (novelty_weight * novelty_score
                 + feasibility_weight * feasibility_score)
                / (novelty_weight + feasibility_weight)
                - contradiction_penalty_weight * num_contradictions
```

**Why the score is recomputed:**

1. **Determinism and auditability.** The result is a fixed function of validated sub scores, not a number the model simply asserts.
2. **Ablation tunability.** Weights can be varied to study ranking policy independent of model behavior.

**Output:**

```python
CritiqueVerdict(
    overall_score=7.4,
    novelty_score=8.2,
    feasibility_score=6.8,
    contradictions=[...],
    recommendation="accept"  # or "revise" / "reject"
)
```

---

### 4. Synthesizer Agent (`agents/synthesizer.py`)

**Role:** experimental architect and simulation executor.

For every hypothesis that passes Critic evaluation, the Synthesizer delivers:

#### An Experimental Protocol

A detailed plan specifying:

- **The objective:** what is being tested.
- **Independent variables:** what is controlled, for example dopant concentration.
- **Dependent variables:** what is measured, for example ionic conductivity.
- **Controls:** baseline comparisons, such as the pure host material.
- **Steps:** the synthesis procedure, characterization methods, and measurement conditions.

#### Toy Simulation Code

A short, self contained numerical simulation, for example an Arrhenius model for ionic conductivity as a function of temperature. The code is generated as Python, executed inside a sandboxed environment, and produces `key=value` metrics for parsing.

#### Sandboxed Execution

`tools/simulation.py` runs simulations under strict constraints:

- **A builtins allowlist:** `abs`, `min`, `max`, `range`, `print`, and container constructors are permitted; `eval`, `exec`, and `open` are not.
- **An import allowlist:** `math`, `statistics`, `random`, `itertools`, `functools`, and `numpy` (if installed) are permitted.
- **Timeout enforcement:** a wall clock timeout, 5 seconds by default, enforced through a daemon thread.
- **An output only channel:** standard output is captured, along with exception reporting; there is no filesystem or network access.

**Metrics parsing:**

```python
# The toy simulation prints:
# ionic_conductivity=0.00087
# activation_energy=0.42
#
# This is automatically parsed into:
metrics = {"ionic_conductivity": 0.00087, "activation_energy": 0.42}
```

#### Ranking and Selection

- Successfully executed simulations receive a modest `simulation_bonus`, for example plus 0.2.
- Final ranking is by composite score, equal to `(overall_score + simulation_bonus)`.
- Results are deduplicated and truncated to `top_k_final`, 10 by default.

---

## Multi Agent Orchestration

### A Dependency Free Graph Engine

At the heart of HypothesisForge lies `orchestration/graph.py`, a minimal, dependency free state machine deliberately modeled on LangGraph's programming interface:

```python
from orchestration.graph import StateGraph, END

graph = StateGraph(PipelineState)

# Define nodes
graph.add_node("ideate", ideator_node)
graph.add_node("scout", scout_node)
graph.add_node("critique", critic_node)
graph.add_node("synthesize", synthesizer_node)

# Define edges
graph.add_edge("ideate", "scout")
graph.add_edge("scout", "critique")

# Conditional edge (a router)
graph.add_conditional_edges(
    "critique",
    route_after_critic,  # Returns "ideate", "synthesize", or END
    {"ideate": "ideate", "synthesize": "synthesize", END: END}
)

# Compile and invoke
pipeline = graph.compile()
result = pipeline.invoke(initial_state)
```

**Key features:**

- Named nodes that transform state, mapping state to state.
- Explicit edges, defined through `add_edge(source, target)`.
- Conditional edges driven by router functions, defined through `add_conditional_edges(source, router_fn, mapping)`.
- A reserved `END` sentinel.
- Compilation and invocation through `.compile().invoke(state)`.
- A maximum steps guard that prevents infinite loops.

### The Control Flow Router

The `_route_after_critic()` router in `orchestration/pipeline.py` orchestrates the iteration loop:

```
while iteration_budget > 0:
    if (hypotheses are queued for revision) or (accepted count is below top_k_target):
        go to "ideate" (continue the loop)
    else:
        go to "synthesize" (move to the synthesis phase)
```

This design ensures:

1. **Exhaustive exploration** within the available budget.
2. **A quality threshold** before synthesis begins, requiring `top_k_target` accepted hypotheses.
3. **A revision feedback loop** for promising candidates.

---

## Domain Grounding: The DomainProfile

All constraint logic is centralized in `hypothesisforge/config.py`.

### The Scope Statement

Explicit in scope and out of scope material classes:

```python
scope_statement = {
    "in_scope": [
        "garnet oxides (Li7La3Zr2O12 and derivatives)",
        "sulfide argyrodites (e.g., Li6PS5Cl)",
        "polymer composites",
        "lithium halides (LiF, LiCl, etc.)"
    ],
    "out_of_scope": [
        "aqueous electrolytes",
        "organic liquid electrolytes",
        "perovskites",
        "metal anodes"
    ]
}
```

The Ideator is instructed to respect these boundaries at every proposal and revision step.

### The Canonical Vocabulary

Fixed sets covering:

- **Material families:** garnet oxides, sulfide argyrodites, lithium halides, polymer composites, and related classes.
- **Performance metrics:** ionic conductivity, critical current density, activation energy, and related measures.
- **Mechanisms:** grain boundary engineering, dopant substitution, heterojunction design, and related mechanisms.

### The Local Literature Corpus

48 synthetically generated abstracts spanning the principal material families and common findings. Each record includes:

- A title
- An abstract, 200 to 300 words in length
- Simulated metrics and results
- Methodological details

**Why the corpus is synthetic:**

- It demonstrates corpus integration without requiring the licensing of real papers.
- It remains domain realistic in vocabulary and structure.
- It can be extended or regenerated through `build_corpus.py`.

**For production deployment**, replace the retrieval backend in `tools/corpus_search.py` with:

- The Semantic Scholar API
- PubMed Central
- arXiv
- An internal, proprietary corpus

---

## Novelty Scoring in Detail

The hybrid novelty function balances two signals.

### Signal 1: Embedding Similarity (Objective)

- Encode the hypothesis and the corpus records into vectors.
- Compute the maximum cosine similarity across the corpus.
- Scale the result to a range of 0 to 10, using a penalty exponent of 0.7, so that small differences matter while large gaps do not overwhelm the score:

```
embedding_component = 10 x (1 - max_similarity ^ 0.7)
```

**Intuition:** a hypothesis achieving 90 percent similarity to its best match scores approximately 3.7 out of 10 on novelty, while 50 percent similarity scores approximately 6.7 out of 10.

### Signal 2: LLM as Judge (Conceptual)

- Claude is prompted to rate conceptual novelty on a 0 to 10 scale.
- The prompt asks how different this mechanism is from prior work.
- This signal is not constrained by vector representations, so it can catch nuanced differences that embeddings alone might miss.

### Fusion

The weighted average of both signals is computed as:

```
combined_score = (embedding_component x embedding_weight
                  + llm_judge_score x judge_weight)
                 / (embedding_weight + judge_weight)
```

**Default weights**, configurable in `RunConfig`:

- `novelty_embedding_weight = 0.5`
- `novelty_judge_weight = 0.5`

**Why a hybrid approach is used:**

- Embedding similarity alone can miss subtle conceptual innovations, such as a new reaction condition.
- The LLM alone can hallucinate similarities or produce inconsistent judgments.
- Combining the two mitigates the failure modes of each individual approach.

---

## Simulation Environment

### Execution Architecture

```python
result = run_toy_simulation(code_string, timeout_seconds=5)

# Returns:
SimulationResult(
    success=True,
    stdout="ionic_conductivity=0.00087\nactivation_energy=0.42",
    stderr="",
    metrics={"ionic_conductivity": 0.00087, "activation_energy": 0.42},
    execution_time_ms=234
)
```

### Constraints and Guarantees

| Constraint | Details |
|---|---|
| Builtins allowlist | `abs`, `all`, `any`, `bool`, `dict`, `enumerate`, `filter`, `float`, `int`, `len`, `list`, `max`, `min`, `print`, `range`, `reversed`, `round`, `set`, `sorted`, `sum`, `tuple`, `zip` are permitted; `eval`, `exec`, `open`, `input`, `compile`, and `__import__` are not permitted unless explicitly guarded |
| Module allowlist | `math`, `statistics`, `random`, `itertools`, `functools`, and `numpy` (if installed) are permitted; all other modules are blocked at import |
| Timeout | A daemon thread enforces a configurable timeout, 5 seconds by default; exceeding it produces a `failed` status, logged as a timeout |
| Input and output surface | Only standard output is captured; there is no filesystem, network, subprocess, or inter process access |
| Security model | Designed for Synthesizer authored code running inside an operator controlled pipeline; it is not designed for arbitrary, untrusted user input |

### Example: The Arrhenius Conductivity Model

```python
# The Synthesizer generates this code:
import math

T_kelvin = 323  # Operating temperature
A = 1e-5        # Pre-exponential factor
E_a = 0.42      # Activation energy (eV)
k_b = 8.617e-5  # Boltzmann constant (eV/K)

sigma = A * math.exp(-E_a / (k_b * T_kelvin))

print(f"ionic_conductivity={sigma:.6e}")
print(f"activation_energy={E_a}")
```

**Execution flow:** standard output is captured, metrics are parsed, and the `simulation_bonus` is applied to the final score.

---

## Testing and Quality Assurance

HypothesisForge includes more than 43 comprehensive tests, covering the following categories.

| Category | Coverage |
|---|---|
| Schema validation | Pydantic contracts, type coercion, required fields, and enum constraints |
| Embedding backends | Mock embedding, the sentence-transformers fallback, and consistency between them |
| Corpus search | TF IDF correctness, similarity ranking, and edge cases such as an empty corpus or missing keywords |
| Novelty scoring | The hybrid formula, weight variations, edge scores of 0.0 and 1.0, and exponent behavior |
| Simulation sandboxing | Timeout enforcement, import blocking, metric parsing, and rejection of malicious code |
| Graph engine | Node registration, edge resolution, conditional routing, `END` termination, and the maximum steps guard |
| Agent behavior | Ideator generation, Literature Scout retrieval, Critic verdicts, and Synthesizer design |
| End to end flows | Full mock pipeline execution, offline runs, and live runs (with an API key) |
| Ablation harness | Correctness of the weight sweep, metrics aggregation, and report generation |

### Running Tests

```bash
# A quick summary (quiet mode)
pytest -q

# Detailed output with coverage metrics
pytest -v --cov=hypothesisforge --cov-report=html

# Run a specific test file
pytest tests/test_novelty_scorer.py -v

# Run a single test function
pytest tests/test_agents.py::test_ideator_respects_scope -v
```

---

## Configuration and Tuning

All configurable parameters are centralized in `hypothesisforge/config.py`.

### `DomainProfile`

```python
domain = DomainProfile(
    name="materials_science",
    scope_statement="...",
    canonical_materials=["garnet oxides", "sulfide argyrodites", ...],
    canonical_metrics=["ionic_conductivity", "activation_energy", ...],
    corpus_path="data/corpus/materials_science_corpus.json"
)
```

### `RunConfig`

```python
config = RunConfig(
    iterations=5,                           # Maximum iteration loops
    top_k_target=3,                         # Minimum accepted hypotheses before synthesis
    top_k_final=10,                         # Final ranked output size
    novelty_embedding_weight=0.5,           # Embedding versus LLM in the novelty score
    novelty_judge_weight=0.5,
    feasibility_weight=1.0,                 # Multi axis scoring weights
    contradiction_penalty_weight=0.5,
    llm_client_type="mock" or "anthropic",  # Switch between offline and live
    mock_persona="balanced",                # "balanced" / "strict" / "lenient"
    embedding_backend="mock",               # "mock" / "sentence-transformers"
    random_seed=42                          # Reproducibility
)
```

---

## Ablation Study

The `ablation.py` module systematically explores how component weights affect system behavior:

```bash
python -m hypothesisforge.cli ablate \
  --out examples/ablation_report.md \
  --json-out examples/ablation_metrics.json
```

**What this does:**

1. Varies `novelty_weight`, `feasibility_weight`, and `contradiction_penalty_weight` across three levels: low, medium, and high.
2. Runs the mock pipeline for each configuration.
3. Collects metrics, including hypothesis counts (accepted, rejected, and revised), score statistics (mean, standard deviation, minimum, and maximum), coverage (the share of material families represented), and diversity (the entropy of performance metrics).
4. Generates a markdown report together with a JSON export.

**Use cases:**

- Understanding tradeoffs, for example how strict novelty weighting produces fewer but more innovative candidates.
- Validating scoring logic under different regimes.
- Reproducing results for publication.
- Supporting design decisions, such as explaining why a particular set of weights was chosen.

---

## Seamless Migration to Real LangGraph

HypothesisForge is intentionally compatible with LangGraph. Because the API surface matches exactly, migration is mechanical.

### Before (HypothesisForge)

```python
from orchestration.graph import StateGraph, END
```

### After (Real LangGraph)

```python
from langgraph.graph import StateGraph, END
```

### The Only Adaptation Needed

```python
# Change PipelineState from @dataclass to TypedDict:
from typing import TypedDict

class PipelineState(TypedDict):
    active_hypotheses: list[Hypothesis]
    accepted: list[Hypothesis]
    # other fields
```

**Everything else continues unchanged:** `pipeline.py`, agent invocations, state management, compilation, and invocation all carry over directly.

---

## Comprehensive Documentation

### Core Concepts

- **Hypothesis:** a testable claim about material behavior, together with its proposed mechanism.
- **LiteratureFinding:** a summary of prior work and the identified gap.
- **CritiqueVerdict:** a multi axis evaluation, covering novelty, feasibility, contradictions, and a recommendation.
- **ExperimentDesign:** a protocol, simulation code, and the resulting simulation result.
- **RankedHypothesis:** the final candidate, together with its full evaluation history and metrics.

### Key Files

| File | Purpose |
|---|---|
| `config.py` | Domain profiles and run configuration |
| `schemas.py` | Pydantic models for all inter agent contracts |
| `llm_client.py` | The Anthropic API wrapper plus the deterministic mock |
| `prompts.py` | System and user prompts for each agent |
| `agents/ideator.py` | Hypothesis generation and revision |
| `agents/literature_scout.py` | Corpus search and gap analysis |
| `agents/critic.py` | Multi axis evaluation |
| `agents/synthesizer.py` | Experiment design and simulation |
| `tools/embeddings.py` | Embedding backends |
| `tools/corpus_search.py` | TF IDF plus similarity retrieval |
| `tools/novelty_scorer.py` | Hybrid novelty computation |
| `tools/simulation.py` | Sandboxed execution |
| `orchestration/graph.py` | The LangGraph compatible state machine |
| `orchestration/pipeline.py` | Agent orchestration and control flow |
| `evaluation/metrics.py` | Evaluation metrics |
| `evaluation/ablation.py` | Ablation harness |
| `cli.py` | The command line interface |

### `ARCHITECTURE.md`

See [ARCHITECTURE.md](./ARCHITECTURE.md) for detailed design decisions, including:

- Why a custom graph engine was built rather than importing LangGraph directly.
- Why `overall_score` is recomputed rather than trusted from the LLM.
- The design of the MockLLMClient, including its role awareness, seeding, and determinism.
- The sandboxing model used for toy simulations.

---

## Understanding the MockLLMClient

For offline reproducibility and zero cost testing, HypothesisForge uses a deterministic, seeded, role aware mock.

### How It Works

1. **Role detection:** each system prompt embeds a `[ROLE:xxx]` marker (see `prompts.py`).
2. **Role specific generators:** requests are dispatched to `_mock_ideator()`, `_mock_literature_scout()`, `_mock_critic()`, or `_mock_synthesizer()`.
3. **Seeding:** a hash of the role, the user prompt, and the persona, using `sha256`, ensures determinism.
4. **Persona shifting:** a bias term, set to balanced, strict, or lenient, simulates different judge models.

### Why This Matters

- **Reproducible tests:** the same inputs always produce the same outputs.
- **Cost free:** no API calls are made.
- **Realistic branching:** different hypotheses and iterations still produce varied verdicts, such as accept, revise, or reject.
- **Ablation friendly:** the full pipeline can be run multiple times with different personas.
- **Structurally identical:** the agents, the graph, and the scoring logic are the same as in a live run; only the LLM itself is replaced.

---

## Known Limitations and Design Tradeoffs

### MockLLMClient

- **Limitation:** outputs are templated rather than genuinely creative.
- **Tradeoff:** this faithfully demonstrates pipeline mechanics, including agent handoffs, scoring math, the iteration loop, and ablation, without requiring API credentials or spending money.
- **Mitigation:** set `llm_client_type="anthropic"` for genuine generative evaluation, and provide the `ANTHROPIC_API_KEY` environment variable.

### Synthetic Literature Corpus

- **Limitation:** the 48 records are synthetically generated rather than drawn from real peer reviewed literature.
- **Tradeoff:** this demonstrates corpus integration and retrieval without licensing real papers, and is suitable for reference or demonstration purposes; a research grade deployment would replace it.
- **Mitigation:** replace the retrieval logic in `tools/corpus_search.py` with a real literature API, such as Semantic Scholar, PubMed, or arXiv, or with an internal corpus.

### Toy Simulations

- **Limitation:** the simulations are deliberately simple order of magnitude sanity checks, such as a bare Arrhenius model, rather than density functional theory or molecular dynamics.
- **Tradeoff:** this illustrates experiment design and metrics extraction without requiring a full scientific software stack.
- **Mitigation:** extend the simulation environment, or integrate external scientific code, for production deployments.

### Scope: A Materials Science Focus

- **Limitation:** the domain specific vocabulary, corpus, and prompts are geared toward solid state electrolytes.
- **Tradeoff:** this provides a concrete, working example; the same agents, orchestration, and tools apply readily to other domains.
- **Mitigation:** create a new `DomainProfile` for a different field, such as cognitive science or drug discovery, while reusing the existing agent logic.

---

## Example Outputs

### Sample Run Output

A complete offline run is checked into the repository at `examples/sample_run_output.json`. Its structure:

```json
{
  "config": { /* RunConfig snapshot */ },
  "iterations_completed": 3,
  "active_hypotheses_count": 12,
  "accepted_hypotheses": [
    {
      "id": "hyp-001",
      "text": "...",
      "material_family": "garnet oxides",
      "performance_metric": "ionic_conductivity",
      "literature_findings": { /* LiteratureFinding */ },
      "critique_verdict": { /* CritiqueVerdict */ },
      "experiment_design": { /* ExperimentDesign with simulation_result */ },
      "final_rank": 1,
      "composite_score": 8.2
    }
    /* additional hypotheses */
  ],
  "rejected_hypotheses": [ /* additional records */ ],
  "execution_time_ms": 2340
}
```

### Sample Ablation Report

See `examples/sample_ablation_report.md`:

```markdown
# Ablation Study Report

## Configuration Sweep
Varied novelty_weight, feasibility_weight, and contradiction_penalty_weight across 3 levels.

## Key Findings
- Strict novelty weighting (novelty_weight = 1.0) yields fewer but more innovative candidates
- Balanced weighting (all values at 0.5) achieves the highest diversity
- Feasibility heavy weighting (feasibility_weight = 1.5) favors practical, lower risk hypotheses

## Metrics Summary
| Config | Accepted | Rejected | Mean Score | Std Dev | Coverage |
|--------|----------|----------|------------|---------|----------|
| Balanced | 8 | 4 | 7.1 | 1.2 | 75% |
| Strict | 5 | 7 | 7.8 | 0.9 | 60% |
| Lenient | 11 | 1 | 6.3 | 1.5 | 85% |
```

---

## Development and Contributing

### Code Style

- Python 3.10 or later, with type hints checked through Pydantic.
- Dataclasses are used for immutable state; Pydantic is used for validation.
- pytest is used for testing, with pytest-cov used for coverage reports.

### Adding a New Agent

1. Create `agents/your_agent.py`, implementing the agent interface.
2. Update `schemas.py` with any new input or output types.
3. Update `prompts.py` with the system and user prompt templates.
4. Integrate the agent into `orchestration/pipeline.py` by adding a node and its edges.
5. Write tests in `tests/test_agents.py`.

### Adding a New Tool

1. Create `tools/your_tool.py` with public functions.
2. Add the tool to the relevant agent's imports and calls.
3. Update `tools/__init__.py` if centralizing exports.
4. Test the tool using `tests/test_agents.py` or a dedicated test file.

### Extending for a New Domain

1. Copy `config.py` and create a new `DomainProfile` with a scope statement for the new domain, canonical materials, metrics, and mechanisms, and a path to the relevant corpus.
2. Generate or curate a corpus, with at least 48 records, in JSON format.
3. Update `build_corpus.py` if regeneration is needed.
4. Update the prompts in `prompts.py` with domain specific examples.
5. Run the test suite to verify the end to end flow.

---

## License

Distributed under the MIT License. See the [LICENSE](./LICENSE) file for full terms.

---

## Support and Questions

For questions, issues, or suggestions:

1. Check [ARCHITECTURE.md](./ARCHITECTURE.md) for design rationale.
2. Review the test cases in `tests/` for usage patterns.
3. Inspect the sample outputs in `examples/` for the expected structure.
4. Examine the prompts in `prompts.py` for agent instructions.

---

## Research and Publications

HypothesisForge is both a working system and a research apparatus for studying:

- Whether structured critique and iterative refinement elevate machine generated scientific hypotheses.
- How hybrid scoring, combining embeddings with LLM judgment, mitigates novelty assessment failures.
- The tradeoffs between agent specialization and generalism.
- Cost and quality curves when swapping between mock and live LLM backends.
- The reproducibility and auditability of AI assisted hypothesis generation.

If you use HypothesisForge in research, attribution would be appreciated.

---

## Highlights

- **Production ready:** minimal dependencies, comprehensive testing, and deterministic execution paths.
- **Fully auditable:** every decision is logged, and the scoring policy is transparent, with no black box LLM arithmetic.
- **Extensible:** designed for domain generalization, so a new domain can be adopted simply by swapping in a new `DomainProfile`.
- **Reproducible:** mock mode combined with seeded randomness produces identical results across runs.
- **Scalable:** ready for LangGraph integration, since the orchestration logic is framework agnostic.
- **Educational:** clear agent separation and explicit state flow make this a useful reference implementation for multi agent systems.

---

<div align="center">

**HypothesisForge: where AI meets rigorous science.**

*Generating hypotheses, grounded in evidence, evaluated fairly, and refined iteratively.*

</div>
