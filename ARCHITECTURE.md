# Architecture Notes

## Data flow for a single iteration

```
PipelineState.active_hypotheses
        │
        ▼
IdeatorAgent.generate() / .revise()      LLM call (role=ideator)
        │  -> list[Hypothesis]
        ▼
LiteratureScoutAgent.investigate()       CorpusIndex.search() [tool, no LLM]
        │                                 + LLM call (role=literature_scout)
        │  -> LiteratureFinding
        ▼
CriticAgent.critique()                   CorpusIndex.similarity_stats() [tool]
        │                                 + LLM call (role=critic)
        │                                 + compute_novelty() [fuses tool + LLM]
        │  -> CritiqueVerdict (recommendation: accept/revise/reject)
        ▼
   route: accept -> state.accepted
          revise + budget left -> state.revision_queue (fed back to Ideator)
          revise + no budget -> state.accepted (best effort, annotated)
          reject -> state.rejected (feedback logged for future Ideator prompts)
```

After the graph reaches `synthesize`:

```
for hyp in state.accepted:
    SynthesizerAgent.design_experiment(hyp, critique)
        -> ExperimentDesign (protocol + toy_simulation_code)
        -> run_toy_simulation(toy_simulation_code)  [sandboxed tool]
        -> ExperimentDesign.simulation_result

SynthesizerAgent.rank(survivors) -> list[RankedHypothesis], sorted, truncated to top_k_final
```

## Why a custom graph engine instead of importing `langgraph`

Three practical reasons, stated plainly:

1. **Zero install-time risk.** This project should run out of the box with
   `pip install -r requirements.txt`. A pinned `langgraph` dependency adds
   version-skew risk (LangGraph's API has changed across releases) for a
   demo/reference project.
2. **Readability.** `PipelineState` is a plain dataclass, not a
   `TypedDict` with reducer annotations — easier to read top-to-bottom for
   anyone auditing agent hand-offs.
3. **The programming model is what matters for the skill being
   demonstrated.** `graph.py`'s `StateGraph`/`add_node`/`add_edge`/
   `add_conditional_edges`/`END`/`.compile().invoke()` API intentionally
   mirrors LangGraph's real surface, so the orchestration *pattern* being
   exercised is the same one you'd use with the real package — swapping the
   import is a small, mechanical change (see README "Swapping in real
   LangGraph").

## Why overall_score is recomputed, not trusted from the LLM

The Critic prompt asks the LLM to also emit an `overall_score`, but
`CriticAgent._recompute_overall()` **ignores that number** and instead
computes:

```
overall = (novelty_weight * novelty_score + feasibility_weight * feasibility_score)
          / (novelty_weight + feasibility_weight)
          - contradiction_penalty_weight * len(contradictions)
```

using weights from `RunConfig`. This has two benefits:

1. **Determinism / auditability**: the scoring policy is a fixed function
   of already-validated sub-scores, not an LLM arithmetic claim that could
   be wrong or inconsistent between calls.
2. **Ablation tunability**: the ablation harness (or any caller) can vary
   `novelty_weight` / `feasibility_weight` / `contradiction_penalty_weight`
   and see how the *ranking policy* — not just the underlying judgments —
   changes final outcomes, independent of model behavior.

## MockLLMClient design

`MockLLMClient` is not a random stub — it is a deterministic, seeded,
role-aware synthetic response generator:

- Each system prompt embeds a `[ROLE:xxx]` marker (see `prompts.py`); the
  mock client dispatches on this marker to a role-specific generator
  (`_mock_ideator`, `_mock_literature_scout`, `_mock_critic`,
  `_mock_synthesizer`).
- Responses are seeded off `sha256(role, user_prompt, persona)`, so the
  same inputs always produce the same outputs (reproducible tests and
  demos) while different inputs (different hypothesis text, different
  iteration) still vary outputs enough to exercise realistic pipeline
  branching (accept vs. revise vs. reject).
- `persona` (`"balanced"|"strict"|"lenient"`) shifts a bias term applied to
  feasibility/novelty scores, which is how the ablation study simulates
  "different judge models" without requiring multiple live API keys or
  model subscriptions to demonstrate the harness.

This means the entire pipeline — including the ablation study — is
runnable, testable, and demoable with **zero API cost and zero network
access**, while remaining structurally identical to a live run (same
agents, same graph, same scoring code) when `AnthropicLLMClient` is
substituted in.

## Sandboxing model for toy simulations

`tools/simulation.run_toy_simulation()` is deliberately conservative:

- **Builtins allowlist**: only a curated set of safe builtins
  (`abs`, `min`, `max`, `range`, `print`, numeric/container constructors,
  etc.) is exposed via `__builtins__` in the exec namespace — no `open`,
  `eval`, `exec`, `input`, `compile`, or `__import__` beyond the guarded
  wrapper below.
- **Import allowlist**: a custom `__import__` implementation rejects any
  module not in `{math, statistics, random, itertools, functools}` (plus a
  pre-bound `np` reference if numpy is installed) — this blocks `os`,
  `sys`, `subprocess`, `socket`, etc. even if code tries to import them
  directly.
- **Timeout**: execution runs in a daemon thread; `thread.join(timeout=...)`
  bounds wall-clock time. If the thread is still alive after the timeout,
  the result is reported as failed/timed-out; the (daemon) thread is
  abandoned rather than force-killed, since Python has no safe primitive
  for killing a thread mid-execution — this is an accepted limitation for
  a toy-model sandbox, not a general-purpose untrusted-code executor.
- **Output-only surface**: the only channel back to the caller is
  captured stdout (parsed for `key=value` lines) plus any exception
  message — there is no filesystem or network access exposed through the
  allowed builtins/modules for code to exfiltrate data through.

This tool is scoped to execute **Synthesizer-authored** snippets in a
pipeline the operator controls, not to safely sandbox arbitrary
user-submitted code from an untrusted party.
