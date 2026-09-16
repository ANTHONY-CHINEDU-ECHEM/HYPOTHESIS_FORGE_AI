# HypothesisForge: AI-Driven Scientific Hypothesis Generation & Evaluation

<div align="center">

**A rigorously engineered multi-agent system for automated generation, literature grounding, critical evaluation, iterative refinement, and ranked prioritization of scientific research hypotheses**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Code Quality](https://img.shields.io/badge/Tests-43%2B%20Passing-brightgreen)
![Status](https://img.shields.io/badge/Status-Production%20Ready-blue)

[Features](#-core-features) • [Quick Start](#-quick-start) • [Architecture](#-system-architecture) • [Agents](#-the-four-specialized-agents) • [Documentation](#-comprehensive-documentation)

</div>

---

## 🎯 Overview

HypothesisForge is a sophisticated multi-agent AI system purpose-built for scientific research hypothesis generation with a focus on **materials science** (solid-state electrolytes, ionic conductors, and related domains). Unlike generic language model applications, HypothesisForge enforces rigorous domain grounding through:

- **Structured domain constraints** preventing off-topic drift
- **Literature-aware critique** backed by a curated corpus of domain-realistic abstracts
- **Hybrid novelty scoring** combining embedding similarity with LLM-as-judge evaluation
- **Iterative refinement** with scientist-in-the-loop feedback mechanisms
- **Deterministic simulation** with sandboxed execution environments
- **Dependency-free orchestration** featuring a LangGraph-compatible state machine

The entire system is designed for **reproducibility**, **auditability**, and **seamless migration** to production orchestration frameworks.

---

## ✨ Core Features

### 🤖 **Four Specialized Agents**
- **Ideator** — Creative generation of novel, mechanistically grounded hypotheses
- **Literature Scout** — Tool-using agent for corpus search and gap analysis
- **Critic** — Multi-axis evaluation (novelty, feasibility, contradiction detection)
- **Synthesizer** — Experimental protocol design + toy simulation generation & execution

### 📚 **Evidence-Based Grounding**
- Explicit **Scope Statements** defining in/out-of-scope research areas
- **Canonical Vocabulary** of material families and measurable performance metrics
- **Local Literature Corpus** of 48 synthetically generated but domain-realistic abstracts
- All literature claims are **traceable** to retrieved documents (no parametric memory hallucination)

### 🎓 **Rigorous Evaluation Framework**
- **Novelty Scoring**: Hybrid approach fusing embedding similarity (0–10) with LLM-as-judge conceptual assessment
- **Feasibility Assessment**: Experimental practicality evaluation in light of existing literature
- **Contradiction Detection**: Logical and empirical conflict identification
- **Configurable Weighting**: Fully auditable scoring policy (no black-box LLM arithmetic)

### 🔄 **Iterative Refinement**
- Hypotheses rejected for revision receive **concrete, actionable feedback**
- Revised hypotheses are **re-evaluated in the same pipeline**, forming a closed scientific feedback loop
- Parent-child relationships (**parent_id**) track lineage and evolution

### ⚙️ **Production-Ready Design**
- **Minimal dependencies** — no external orchestration framework required
- **Dependency-free state graph** deliberately modeled on LangGraph (seamless migration path)
- **Deterministic mock mode** for offline reproducibility and cost-free testing
- **43+ comprehensive tests** covering schemas, agents, tools, and end-to-end flows
- **Ablation harness** for systematic exploration of design trade-offs

---

## 🚀 Quick Start

### Prerequisites
- **Python 3.10+**
- A Unix-like shell (bash, zsh) for the examples below

### 1. Environment Setup

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Offline Demonstration (No API Key Required)

The simplest way to explore HypothesisForge is using the **deterministic MockLLMClient**:

```bash
# Generate 3 iterations with 4 random seeds, keeping top-3 hypotheses
python -m hypothesisforge.cli run --iterations 3 --seeds 4 --top-k 3

# Output appears in your terminal with structured hypothesis data
```

### 3. Persist Results to JSON

```bash
# Save complete run to JSON for downstream analysis
python -m hypothesisforge.cli run --iterations 3 --out examples/my_run.json

# Inspect the result
python -c "import json; print(json.dumps(json.load(open('examples/my_run.json')), indent=2)[:500])"
```

### 4. Live Evaluation with Anthropic API

```bash
# Set your API key
export ANTHROPIC_API_KEY=sk-ant-...

# Run against Claude (costs ~$0.10–$0.50 per run)
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

## 📋 System Architecture

### High-Level Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                        HYPOTHESIS_GENERATION                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Ideator.generate()         [LLM: constrained creative generation]   │
│         │                                                             │
│         ▼                                                             │
│  Ideator.revise()           [conditional: update based on feedback]  │
│         │                                                             │
│         ▼                                                             │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ For each hypothesis: (LiteratureScout → Critic loop)        │   │
│  │                                                              │   │
│  │ 1. LiteratureScout.investigate()                            │   │
│  │    - CorpusIndex.search() [deterministic TF-IDF + embed]   │   │
│  │    - Extract literature gap vs. prior work                 │   │
│  │                                                              │   │
│  │ 2. Critic.critique()                                        │   │
│  │    - compute_novelty() [hybrid: embed_sim + LLM_judge]     │   │
│  │    - feasibility_score [LLM judgment]                       │   │
│  │    - contradiction_detection [trace to corpus]             │   │
│  │    - overall_score [recomputed, not trusted from LLM]      │   │
│  │    - recommendation: ACCEPT / REVISE / REJECT              │   │
│  │                                                              │   │
│  │ 3. Router Decision                                          │   │
│  │    - ACCEPT → state.accepted (candidate for synthesis)     │   │
│  │    - REVISE + budget → state.revision_queue (→ Ideator)   │   │
│  │    - REVISE + no budget → state.accepted (best effort)     │   │
│  │    - REJECT → state.rejected (logged for context)          │   │
│  │                                                              │   │
│  └──────────────────────────────────────────────────────────────┘   │
│         (Loop continues while iteration budget remains)              │
│                                                                       │
├─────────────────────────────────────────────────────────────────────┤
│                           SYNTHESIS & RANKING                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Synthesizer.design_experiment()   [LLM: protocol + simulation code] │
│         │                                                             │
│         ▼                                                             │
│  run_toy_simulation()              [sandboxed execution, timeout]    │
│         │                                                             │
│         ▼                                                             │
│  Synthesizer.rank()                [sort by composite_score]        │
│         │                                                             │
│         ▼                                                             │
│  TOP_K_FINAL                       [return ranked, deduplicated]    │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

### Directory Structure

```
hypothesisforge/
├── config.py                 # Domain profiles & RunConfig (all tunable knobs)
├── schemas.py                # Pydantic contracts (Hypothesis, LiteratureFinding, etc.)
├── llm_client.py             # AnthropicLLMClient (live) + MockLLMClient (offline)
├── prompts.py                # Per-agent system & user prompt templates
├── cli.py                    # CLI entry points: `run` and `ablate`
│
├── agents/                   # Four specialized agent implementations
│   ├── ideator.py            # Hypothesis generation & revision
│   ├── literature_scout.py   # Corpus search & gap analysis
│   ├── critic.py             # Multi-axis evaluation & scoring
│   └── synthesizer.py        # Experiment design & simulation orchestration
│
├── tools/                    # Shared utilities for agents
│   ├── embeddings.py         # Embedding backends (mock + sentence-transformers)
│   ├── corpus_search.py      # CorpusIndex (TF-IDF + similarity retrieval)
│   ├── novelty_scorer.py     # Hybrid novelty computation
│   └── simulation.py         # Sandboxed code execution with timeout
│
├── orchestration/            # Graph engine & pipeline
│   ├── graph.py              # LangGraph-compatible StateGraph implementation
│   ├── state.py              # PipelineState definition
│   └── pipeline.py           # Multi-agent orchestration & control flow
│
├── evaluation/               # Analysis & ablation
│   ├── metrics.py            # Evaluation metrics (coverage, diversity, etc.)
│   └── ablation.py           # Ablation harness (sweeping component weights)
│
└── tests/                    # 43+ comprehensive test cases

data/
└── corpus/
    └── materials_science_corpus.json   # 48-record synthetic literature corpus

build_corpus.py                # Corpus generation & regeneration script

examples/
├── sample_run_output.json     # Example offline run (full structured output)
├── sample_ablation_report.md  # Example ablation study report
└── sample_ablation_metrics.json # Example ablation metrics (JSON)
```

---

## 🧠 The Four Specialized Agents

### 1. **Ideator Agent** (`agents/ideator.py`)

**Role**: Creative engine with constrained freedom

The Ideator generates specific, mechanistic hypotheses that:
- **Stay within scope** (enforced via DomainProfile)
- **Reference only recognized material families** (garnet oxides, sulfide argyrodites, lithium halides, etc.)
- **Engage measurable performance metrics** (ionic conductivity, critical current density, activation energy, etc.)
- **Avoid duplication** of hypotheses already under consideration

**Key Methods**:
- `generate()` — Produce N candidate hypotheses from scratch
- `revise(hypothesis, feedback)` — Refine a hypothesis based on Critic feedback, maintaining parent_id lineage

**Example Output**:
```json
{
  "id": "hyp-001",
  "parent_id": null,
  "text": "Introducing 1–3 mol% yttrium oxide into garnet-type Li7La3Zr2O12 will increase ionic conductivity by suppressing grain-boundary resistance while maintaining lithium-ion transference number > 0.7.",
  "material_family": "garnet oxides",
  "performance_metric": "ionic conductivity",
  "mechanism": "grain boundary engineering",
  "iteration": 1
}
```

---

### 2. **Literature Scout Agent** (`agents/literature_scout.py`)

**Role**: Tool-using evidence retriever

The Literature Scout **never relies on parametric memory**. Instead, it:
1. **Invokes `CorpusIndex.search()`** — Deterministic TF-IDF + embedding similarity search
2. **Retrieves most relevant records** from the local corpus
3. **Synthesizes findings** into a structured summary
4. **Articulates the gap** between the current hypothesis and prior work

**Key Methods**:
- `investigate(hypothesis, domain_profile)` → `LiteratureFinding`
  - Searches corpus for related work
  - Summarizes existing knowledge
  - Identifies innovation gap

**Guarantees**:
✅ Every literature statement is traceable to an actual retrieved document  
✅ No hallucinated citations or made-up references  
✅ Reproducible results (same query → same retrieved documents)

---

### 3. **Critic Agent** (`agents/critic.py`)

**Role**: Multi-axis evaluator and quality gatekeeper

The Critic performs a structured evaluation across four dimensions:

#### **Novelty**
Computed by `tools/novelty_scorer.py`:
- **Embedding Component** (0–10): Inverse of max cosine similarity to corpus, scaled with exponent 0.7
- **LLM-as-Judge Component** (0–10): Conceptual/mechanistic novelty score from Claude
- **Hybrid Score**: Weighted average of both components

Formula:
```
embedding_component = 10 × (1 − max_similarity^0.7)
combined_score = (embedding_component × embedding_weight + llm_judge_score × judge_weight)
                 / (embedding_weight + judge_weight)
```

#### **Feasibility**
- LLM judges experimental practicality given literature findings
- Assesses availability of materials, synthesis complexity, equipment requirements

#### **Contradiction Detection**
- Identifies logical or empirical conflicts with retrieved literature
- Flags implicit assumptions that contradict prior work
- Counts contradictions for penalty scoring

#### **Overall Score (Recomputed)**
The Critic **ignores the LLM's arithmetic** and recomputes:

```python
overall_score = (novelty_weight × novelty_score
                 + feasibility_weight × feasibility_score)
                / (novelty_weight + feasibility_weight)
                - contradiction_penalty_weight × num_contradictions
```

**Why recompute?**
1. **Determinism & Auditability**: Fixed function of validated sub-scores
2. **Ablation Tunability**: Vary weights to study ranking policy independent of model behavior

**Output**:
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

### 4. **Synthesizer Agent** (`agents/synthesizer.py`)

**Role**: Experimental architect and simulation executor

For every hypothesis that **passes Critic evaluation**, the Synthesizer delivers:

#### **Experimental Protocol**
A detailed plan specifying:
- **Objective**: What you're testing
- **Independent Variables**: What you control (e.g., dopant concentration)
- **Dependent Variables**: What you measure (e.g., ionic conductivity)
- **Controls**: Baseline comparisons (pure host material)
- **Steps**: Synthesis procedure, characterization methods, measurement conditions

#### **Toy Simulation Code**
A short, self-contained numerical simulation:
- Example: Arrhenius model for ionic conductivity as a function of temperature
- Printed as Python code (executed in sandboxed environment)
- Outputs `key=value` metrics for parsing

#### **Sandboxed Execution**
`tools/simulation.py` runs simulations with strict constraints:
- **Builtins Allowlist**: `abs`, `min`, `max`, `range`, `print`, container constructors (no `eval`, `exec`, `open`)
- **Import Allowlist**: `math`, `statistics`, `random`, `itertools`, `functools`, `numpy` (if installed)
- **Timeout Enforcement**: Wall-clock timeout (default 5s) via daemon thread
- **Output-Only Channel**: stdout capture + exception reporting (no filesystem/network access)

**Metrics Parsing**:
```python
# Toy simulation prints:
# ionic_conductivity=0.00087
# activation_energy=0.42
# 
# → Automatically parsed into:
metrics = {"ionic_conductivity": 0.00087, "activation_energy": 0.42}
```

#### **Ranking & Selection**
- Successfully executed simulations receive a modest `simulation_bonus` (e.g., +0.2)
- Final ranking by composite score: `(overall_score + simulation_bonus)`
- Deduplicate and truncate to `top_k_final` (default 10)

---

## 🔗 Multi-Agent Orchestration

### Dependency-Free Graph Engine

At the heart of HypothesisForge lies `orchestration/graph.py`, a minimal, dependency-free state machine **deliberately modeled on LangGraph's programming interface**:

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

# Conditional edge (router)
graph.add_conditional_edges(
    "critique",
    route_after_critic,  # Returns "ideate", "synthesize", or END
    {"ideate": "ideate", "synthesize": "synthesize", END: END}
)

# Compile and invoke
pipeline = graph.compile()
result = pipeline.invoke(initial_state)
```

**Key Features**:
- Named nodes that transform state: `state → state`
- Explicit edges: `add_edge(source, target)`
- Conditional edges driven by router functions: `add_conditional_edges(source, router_fn, mapping)`
- Reserved END sentinel
- Compilation and invocation: `.compile().invoke(state)`
- Max-steps guard prevents infinite loops

### Control Flow Router

The `_route_after_critic()` router in `orchestration/pipeline.py` orchestrates the iteration loop:

```
while iteration_budget > 0:
    if (hypotheses queued for revision) OR (accepted < top_k_target):
        → "ideate" (continue loop)
    else:
        → "synthesize" (move to synthesis phase)
```

This ensures:
1. **Exhaustive exploration** within budget
2. **Quality threshold** before synthesis (reach `top_k_target` accepted hypotheses)
3. **Revision feedback loop** for promising candidates

---

## 📊 Domain Grounding: The DomainProfile

All constraint logic is centralized in `hypothesisforge/config.py`:

### **Scope Statement**
Explicit in/out-of-scope material classes:

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

The Ideator is **instructed to respect these boundaries** at every proposal and revision step.

### **Canonical Vocabulary**
Fixed sets of:
- **Material families**: Garnet oxides, sulfide argyrodites, lithium halides, polymer composites, etc.
- **Performance metrics**: Ionic conductivity, critical current density, activation energy, etc.
- **Mechanisms**: Grain boundary engineering, dopant substitution, heterojunction design, etc.

### **Local Literature Corpus**
48 synthetically generated abstracts spanning principal material families and common findings. Each record includes:
- Title
- Abstract (200–300 words)
- Simulated metrics and results
- Methodological details

**Why synthetic?**
- Demonstrates corpus integration without licensing real papers
- Domain-realistic in vocabulary and structure
- Can be extended or regenerated via `build_corpus.py`

**For production**:
Replace `tools/corpus_search.py` retrieval backend with:
- Semantic Scholar API
- PubMed Central
- arXiv
- Internal proprietary corpus

---

## 🧪 Novelty Scoring in Detail

The hybrid novelty function balances two signals:

### **Signal 1: Embedding Similarity (Objective)**
- Encode hypothesis and corpus records into vectors
- Compute max cosine similarity across corpus
- Scale to [0, 10] with penalty exponent 0.7 (small differences matter; large gaps don't magnify too much):

```
embedding_component = 10 × (1 − max_similarity^0.7)
```

**Intuition**: A hypothesis achieving 90% similarity to the best match scores ~3.7/10 novelty; 50% similarity scores ~6.7/10.

### **Signal 2: LLM-as-Judge (Conceptual)**
- Prompt Claude to rate conceptual novelty (0–10)
- Asks: "How different is this mechanism from prior work?"
- Not constrained by vectors; catches nuanced differences

### **Fusion**
Compute weighted average of both signals:

```
combined_score = (embedding_component × embedding_weight 
                  + llm_judge_score × judge_weight)
                 / (embedding_weight + judge_weight)
```

**Default weights** (configurable in `RunConfig`):
- `novelty_embedding_weight = 0.5`
- `novelty_judge_weight = 0.5`

**Why hybrid?**
- **Embedding alone** can miss subtle conceptual innovations (e.g., new reaction condition)
- **LLM alone** can hallucinate similarities or be inconsistent
- **Together**: Mitigates failure modes of each approach

---

## 🔬 Simulation Environment

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

### Constraints & Guarantees

| Constraint | Details |
|-----------|---------|
| **Builtins Allowlist** | `abs`, `all`, `any`, `bool`, `dict`, `enumerate`, `filter`, `float`, `int`, `len`, `list`, `max`, `min`, `print`, `range`, `reversed`, `round`, `set`, `sorted`, `sum`, `tuple`, `zip` — No `eval`, `exec`, `open`, `input`, `compile`, `__import__` (unless guarded) |
| **Module Allowlist** | `math`, `statistics`, `random`, `itertools`, `functools`, `numpy` (if installed). All others blocked at import. |
| **Timeout** | Daemon thread with configurable timeout (default 5s). Exceeded time → status `failed`, logged as timeout. |
| **I/O Surface** | Only stdout capture. No filesystem, network, subprocess, or inter-process access. |
| **Security Model** | Designed for Synthesizer-authored code in operator-controlled pipeline, **not** for arbitrary untrusted user input. |

### Example: Arrhenius Conductivity Model

```python
# Synthesizer generates this code:
import math

T_kelvin = 323  # Operating temperature
A = 1e-5        # Pre-exponential factor
E_a = 0.42      # Activation energy (eV)
k_b = 8.617e-5  # Boltzmann constant (eV/K)

sigma = A * math.exp(-E_a / (k_b * T_kelvin))

print(f"ionic_conductivity={sigma:.6e}")
print(f"activation_energy={E_a}")
```

**Execution** → stdout captured → metrics parsed → `simulation_bonus` applied to final score.

---

## 🧪 Testing & Quality Assurance

HypothesisForge includes **43+ comprehensive tests** covering:

### Test Categories

| Category | Coverage |
|----------|----------|
| **Schema Validation** | Pydantic contracts, type coercion, required fields, enum constraints |
| **Embedding Backends** | Mock embedding, sentence-transformers fallback, consistency |
| **Corpus Search** | TF-IDF correctness, similarity ranking, edge cases (empty corpus, missing keywords) |
| **Novelty Scoring** | Hybrid formula, weight variations, edge scores (0.0, 1.0), exponent behavior |
| **Simulation Sandboxing** | Timeout enforcement, import blocking, metric parsing, malicious code rejection |
| **Graph Engine** | Node registration, edge resolution, conditional routing, END termination, max-steps guard |
| **Agent Behavior** | Ideator generation, Literature Scout retrieval, Critic verdicts, Synthesizer design |
| **End-to-End Flows** | Full mock pipeline execution, offline runs, live runs (with API key) |
| **Ablation Harness** | Weight sweep correctness, metrics aggregation, report generation |

### Running Tests

```bash
# Quick summary (quiet mode)
pytest -q

# Detailed output with coverage metrics
pytest -v --cov=hypothesisforge --cov-report=html

# Run specific test file
pytest tests/test_novelty_scorer.py -v

# Run a single test function
pytest tests/test_agents.py::test_ideator_respects_scope -v
```

---

## 🎛️ Configuration & Tuning

All configurable parameters are centralized in `hypothesisforge/config.py`:

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
    iterations=5,                           # Max iteration loops
    top_k_target=3,                         # Minimum accepted hypotheses before synthesis
    top_k_final=10,                         # Final ranked output size
    novelty_embedding_weight=0.5,           # Embedding vs. LLM in novelty score
    novelty_judge_weight=0.5,
    feasibility_weight=1.0,                 # Multi-axis scoring weights
    contradiction_penalty_weight=0.5,
    llm_client_type="mock" or "anthropic",  # Switch between offline/live
    mock_persona="balanced",                # "balanced" / "strict" / "lenient"
    embedding_backend="mock",               # "mock" / "sentence-transformers"
    random_seed=42                          # Reproducibility
)
```

---

## 📈 Ablation Study

The `ablation.py` module systematically explores how component weights affect system behavior:

```bash
python -m hypothesisforge.cli ablate \
  --out examples/ablation_report.md \
  --json-out examples/ablation_metrics.json
```

**What it does**:
1. Varies `novelty_weight`, `feasibility_weight`, `contradiction_penalty_weight` across 3 levels (low, medium, high)
2. Runs mock pipeline for each configuration
3. Collects metrics:
   - **Hypothesis count** (accepted, rejected, revised)
   - **Score statistics** (mean, std, min, max)
   - **Coverage** (% of material families represented)
   - **Diversity** (entropy of performance metrics)
4. Generates markdown report + JSON export

**Use cases**:
- Understand trade-offs (e.g., strict novelty→fewer but more innovative candidates)
- Validate scoring logic under different regimes
- Reproduce results for publication
- Support design decisions ("Why did we choose these weights?")

---

## 🌉 Seamless Migration to Real LangGraph

HypothesisForge is **intentionally compatible** with LangGraph. Because the API surface matches exactly, migration is mechanical:

### Before (HypothesisForge)
```python
from orchestration.graph import StateGraph, END
```

### After (Real LangGraph)
```python
from langgraph.graph import StateGraph, END
```

### Only adaptation needed:
```python
# Change PipelineState from @dataclass to TypedDict:
from typing import TypedDict

class PipelineState(TypedDict):
    active_hypotheses: list[Hypothesis]
    accepted: list[Hypothesis]
    # ... other fields
```

**Everything else continues unchanged**: pipeline.py, agent invocations, state management, compilation, invocation.

---

## 📚 Comprehensive Documentation

### Core Concepts
- **Hypothesis**: A testable claim about material behavior with mechanisms
- **LiteratureFinding**: Summary of prior work and identified gap
- **CritiqueVerdict**: Multi-axis evaluation (novelty, feasibility, contradictions, recommendation)
- **ExperimentDesign**: Protocol + simulation code + simulation result
- **RankedHypothesis**: Final candidate with all evaluation history and metrics

### Key Files

| File | Purpose |
|------|---------|
| `config.py` | Domain profiles & run configuration |
| `schemas.py` | Pydantic models for all inter-agent contracts |
| `llm_client.py` | Anthropic API wrapper + deterministic mock |
| `prompts.py` | System & user prompts for each agent |
| `agents/ideator.py` | Hypothesis generation & revision |
| `agents/literature_scout.py` | Corpus search & gap analysis |
| `agents/critic.py` | Multi-axis evaluation |
| `agents/synthesizer.py` | Experiment design & simulation |
| `tools/embeddings.py` | Embedding backends |
| `tools/corpus_search.py` | TF-IDF + similarity retrieval |
| `tools/novelty_scorer.py` | Hybrid novelty computation |
| `tools/simulation.py` | Sandboxed execution |
| `orchestration/graph.py` | LangGraph-compatible state machine |
| `orchestration/pipeline.py` | Agent orchestration & control flow |
| `evaluation/metrics.py` | Evaluation metrics |
| `evaluation/ablation.py` | Ablation harness |
| `cli.py` | Command-line interface |

### ARCHITECTURE.md
See [ARCHITECTURE.md](./ARCHITECTURE.md) for detailed design decisions, including:
- Why a custom graph engine vs. importing LangGraph
- Why overall_score is recomputed (not trusted from LLM)
- MockLLMClient design (role-aware, seeded, deterministic)
- Sandboxing model for toy simulations

---

## 🎓 Understanding the MockLLMClient

For offline reproducibility and zero-cost testing, HypothesisForge uses a **deterministic, seeded, role-aware mock**:

### How It Works

1. **Role Detection**: Each system prompt embeds a `[ROLE:xxx]` marker (see `prompts.py`)
2. **Role-Specific Generators**: Dispatch to `_mock_ideator()`, `_mock_literature_scout()`, `_mock_critic()`, `_mock_synthesizer()`
3. **Seeding**: `sha256(role, user_prompt, persona)` ensures determinism
4. **Persona Shifting**: Bias term (balanced/strict/lenient) simulates different judge models

### Why This Matters

✅ **Reproducible tests**: Same inputs → same outputs  
✅ **Cost-free**: No API calls  
✅ **Realistic branching**: Different hypotheses & iterations still produce varied verdicts (accept/revise/reject)  
✅ **Ablation-friendly**: Run full pipeline multiple times with different personas  
✅ **Structurally identical**: Agents, graph, scoring are the same as live runs; only LLM replaced

---

## 🔐 Known Limitations & Design Trade-Offs

### MockLLMClient
- **Limitation**: Outputs are templated, not genuinely creative
- **Trade-off**: Faithfully demonstrates pipeline mechanics (agent hand-offs, scoring math, iteration loop, ablation) without requiring API credentials or spending money
- **Mitigation**: Set `llm_client_type="anthropic"` for true generative evaluation; include `ANTHROPIC_API_KEY` environment variable

### Synthetic Literature Corpus
- **Limitation**: 48 records are synthetically generated, not real peer-reviewed literature
- **Trade-off**: Demonstrates corpus integration and retrieval without licensing real papers; suitable for reference/demo; research-grade deployments replace it
- **Mitigation**: Replace `tools/corpus_search.py` retrieval with real literature API (Semantic Scholar, PubMed, arXiv, internal corpus)

### Toy Simulations
- **Limitation**: Deliberately simple order-of-magnitude sanity checks (bare Arrhenius models, no DFT/MD)
- **Trade-off**: Illustrate experiment design without requiring scientific software stacks; demonstrate metrics extraction
- **Mitigation**: Extend simulation environment or integrate external scientific code for production deployments

### Scope: Materials Science Focus
- **Limitation**: Domain-specific vocabulary, corpus, and prompts geared to solid-state electrolytes
- **Trade-off**: Provides concrete working example; same agents/orchestration/tools apply to other domains
- **Mitigation**: Create new `DomainProfile` for cognitive science, drug discovery, etc.; reuse agent logic

---

## 📊 Example Outputs

### Sample Run Output
A complete offline run is checked in at `examples/sample_run_output.json`. Structure:

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
    },
    /* ... more hypotheses ... */
  ],
  "rejected_hypotheses": [ /* ... */ ],
  "execution_time_ms": 2340
}
```

### Sample Ablation Report
See `examples/sample_ablation_report.md`:

```markdown
# Ablation Study Report

## Configuration Sweep
Varied novelty_weight, feasibility_weight, contradiction_penalty_weight across 3 levels.

## Key Findings
- **Strict novelty weighting** (novelty_weight=1.0) yields fewer but more innovative candidates
- **Balanced weighting** (all 0.5) achieves highest diversity
- **Feasibility-heavy** (feasibility_weight=1.5) favors practical, lower-risk hypotheses

## Metrics Summary
| Config | Accepted | Rejected | Mean Score | Std Dev | Coverage |
|--------|----------|----------|------------|---------|----------|
| Balanced | 8 | 4 | 7.1 | 1.2 | 75% |
| Strict | 5 | 7 | 7.8 | 0.9 | 60% |
| Lenient | 11 | 1 | 6.3 | 1.5 | 85% |
```

---

## 🛠️ Development & Contributing

### Code Style
- **Python 3.10+** with type hints (checked via Pydantic)
- **Dataclasses** for immutable state; **Pydantic** for validation
- **pytest** for testing; **pytest-cov** for coverage reports

### Adding a New Agent

1. Create `agents/your_agent.py` implementing the agent interface
2. Update `schemas.py` with any new input/output types
3. Update `prompts.py` with system/user prompt templates
4. Integrate into `orchestration/pipeline.py` by adding a node and edges
5. Write tests in `tests/test_agents.py`

### Adding a New Tool

1. Create `tools/your_tool.py` with public functions
2. Add to agent imports and calls
3. Update `tools/__init__.py` if centralizing exports
4. Test with `tests/test_agents.py` or dedicated test file

### Extending for a New Domain

1. Copy `config.py`; create new `DomainProfile` with:
   - Scope statement for your domain
   - Canonical materials / metrics / mechanisms
   - Path to your corpus
2. Generate or curate a corpus (48+ records) in JSON format
3. Update `build_corpus.py` if regeneration is needed
4. Update prompts in `prompts.py` with domain-specific examples
5. Run tests to verify end-to-end flow

---

## 📝 License

MIT License — see [LICENSE](./LICENSE) file for full text.

---

## 🤝 Support & Questions

For questions, issues, or suggestions:
1. Check [ARCHITECTURE.md](./ARCHITECTURE.md) for design rationale
2. Review test cases in `tests/` for usage patterns
3. Inspect sample outputs in `examples/` for expected structure
4. Examine prompts in `prompts.py` for agent instructions

---

## 🎯 Research & Publications

HypothesisForge is both a **working system** and a **research apparatus** for studying:

- Whether structured critique and iterative refinement elevate machine-generated scientific hypotheses
- How hybrid scoring (embedding + LLM judgment) mitigates novelty assessment failures
- Trade-offs between agent specialization vs. generalism
- Cost-quality curves when swapping mock vs. live LLM backends
- Reproducibility and auditability of AI-assisted hypothesis generation

If you use HypothesisForge in research, we'd appreciate attribution!

---

## 🌟 Highlights

✨ **Production-Ready**: Minimal dependencies, comprehensive testing, deterministic execution paths  
✨ **Fully Auditable**: Every decision is logged; scoring policy is transparent (no black-box LLM arithmetic)  
✨ **Extensible**: Designed for domain generalization (swap DomainProfile → new domain)  
✨ **Reproducible**: Mock mode + seeded RNG = identical results across runs  
✨ **Scalable**: LangGraph integration ready; orchestration logic is framework-agnostic  
✨ **Educational**: Clear agent separation, explicit state flow, reference implementation for multi-agent systems  

---

<div align="center">

**HypothesisForge: Where AI meets rigorous science.**

*Generating hypotheses, grounded in evidence, evaluated fairly, refined iteratively.*

![Made with ❤️ for scientific discovery](https://img.shields.io/badge/Made%20with%20%E2%9D%A4%EF%B8%8F%20for%20scientific%20discovery-blue)

</div>
