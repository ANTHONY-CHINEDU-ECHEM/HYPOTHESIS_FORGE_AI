"""
LLM client abstraction.

Two implementations share one interface:

- AnthropicLLMClient: calls the real Anthropic Messages API (requires
  ANTHROPIC_API_KEY). Used for live runs.
- MockLLMClient: deterministic, offline, template-driven responses. Used for
  tests, CI, and demos so the whole pipeline (including ablations) is
  runnable and reproducible with zero API cost / network access.

Agents depend only on the `LLMClient` protocol, never on a concrete class,
so swapping judge models for the ablation study is a one-line change.
"""

from __future__ import annotations

import hashlib
import json
import re
from abc import ABC, abstractmethod
from typing import Any, Optional

from .config import ANTHROPIC_API_KEY_ENV, has_live_api_key


class LLMError(RuntimeError):
    pass


class LLMClient(ABC):
    """Common interface every agent programs against."""

    model_name: str = "unknown"

    @abstractmethod
    def complete_json(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 1500,
        temperature: float = 0.7,
    ) -> dict[str, Any]:
        """Return a parsed JSON object. Raises LLMError on unrecoverable failure."""

    @abstractmethod
    def complete_text(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 800,
        temperature: float = 0.7,
    ) -> str:
        """Return free-text completion."""


def _extract_json_block(text: str) -> dict[str, Any]:
    """Best-effort extraction of a JSON object from an LLM response."""
    text = text.strip()
    # Strip markdown code fences if present.
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    # Fall back to grabbing the first {...} span.
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(candidate[start : end + 1])
        except json.JSONDecodeError as exc:
            raise LLMError(f"Could not parse JSON from LLM output: {exc}\n---\n{text}") from exc
    raise LLMError(f"No JSON object found in LLM output:\n{text}")


class AnthropicLLMClient(LLMClient):
    """Thin wrapper around the Anthropic Python SDK."""

    def __init__(self, model: str = "claude-sonnet-4-6", api_key: Optional[str] = None):
        if not (api_key or has_live_api_key()):
            raise LLMError(
                f"No API key found. Set the {ANTHROPIC_API_KEY_ENV} environment "
                "variable, or use MockLLMClient for offline runs."
            )
        try:
            import anthropic  # local import: optional dependency
        except ImportError as exc:  # pragma: no cover - exercised only without the package
            raise LLMError(
                "The 'anthropic' package is required for live runs. "
                "Install it with `pip install anthropic`."
            ) from exc
        self._client = anthropic.Anthropic(api_key=api_key)
        self.model_name = model

    def _call(self, system: str, user: str, max_tokens: int, temperature: float) -> str:
        try:
            response = self._client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
        except Exception as exc:  # pragma: no cover - network path
            raise LLMError(f"Anthropic API call failed: {exc}") from exc
        parts = [block.text for block in response.content if getattr(block, "type", "") == "text"]
        return "\n".join(parts)

    def complete_json(self, system, user, *, max_tokens=1500, temperature=0.7):
        system_with_format = (
            system
            + "\n\nRespond with ONLY a single valid JSON object. No prose, no markdown "
            "fences, no commentary before or after the JSON."
        )
        raw = self._call(system_with_format, user, max_tokens, temperature)
        return _extract_json_block(raw)

    def complete_text(self, system, user, *, max_tokens=800, temperature=0.7):
        return self._call(system, user, max_tokens, temperature)


class MockLLMClient(LLMClient):
    """
    Deterministic offline stand-in for the Anthropic API.

    Rather than calling out to a model, this inspects the *system prompt's
    role tag* (each agent embeds a `[ROLE:xxx]` marker — see prompts.py) and
    the content of the user prompt to synthesize plausible, schema-valid,
    domain-aware output. Responses are seeded off a hash of the input so the
    same inputs always produce the same outputs (reproducible tests/demo),
    while different inputs (e.g. iteration count, hypothesis text) still
    vary the output enough to exercise the pipeline realistically.

    `persona` lets the ablation study simulate "different judge models":
    - "strict": harsher scores, more contradictions flagged
    - "lenient": more generous scores
    - "balanced": default
    """

    def __init__(self, model: str = "mock-llm-v1", persona: str = "balanced"):
        self.model_name = model
        self.persona = persona

    # -- helpers -----------------------------------------------------
    def _seed(self, *parts: str) -> int:
        h = hashlib.sha256("||".join(parts).encode("utf-8")).hexdigest()
        return int(h[:8], 16)

    def _persona_bias(self) -> float:
        return {"strict": -1.5, "lenient": 1.5, "balanced": 0.0}.get(self.persona, 0.0)

    def _role_of(self, system: str) -> str:
        m = re.search(r"\[ROLE:(\w+)\]", system)
        return m.group(1) if m else "generic"

    # -- main entrypoints ---------------------------------------------
    def complete_json(self, system, user, *, max_tokens=1500, temperature=0.7):
        role = self._role_of(system)
        seed = self._seed(role, user, self.persona)
        rng_bucket = seed % 1000 / 1000.0  # deterministic pseudo-randomness in [0,1)

        if role == "ideator":
            return self._mock_ideator(user, seed)
        if role == "literature_scout":
            return self._mock_literature_scout(user, seed)
        if role == "critic":
            return self._mock_critic(user, seed, rng_bucket)
        if role == "synthesizer":
            return self._mock_synthesizer(user, seed)
        raise LLMError(f"MockLLMClient: no handler for role '{role}'")

    def complete_text(self, system, user, *, max_tokens=800, temperature=0.7):
        role = self._role_of(system)
        seed = self._seed(role, user, self.persona)
        return f"[mock free-text response for role={role}, seed={seed}]"

    # -- role-specific synthetic generators ----------------------------
    def _mock_ideator(self, user: str, seed: int) -> dict[str, Any]:
        n_match = re.search(r"Generate (\d+)", user)
        n = int(n_match.group(1)) if n_match else 3
        families = [
            "garnet-type oxides (e.g., LLZO-family)",
            "sulfide argyrodites and thio-LISICON phases",
            "lithium halides and halide-doped analogues",
            "polymer-in-ceramic and ceramic-in-polymer composites",
            "NASICON-type oxides",
            "perovskite-type oxides",
            "amorphous / glassy Li-ion conductors",
        ]
        mechanisms = [
            "aliovalent doping to introduce Li vacancies and lower migration barriers",
            "engineered grain-boundary chemistry via sintering-aid segregation",
            "gradient composite architecture to decouple bulk conduction from mechanical toughness",
            "interfacial buffer layer formed by in-situ reaction with Li metal",
            "anion-site substitution to widen the electrochemical stability window",
            "texture/orientation control during processing to favor fast-conduction planes",
        ]
        metrics = [
            "room-temperature ionic conductivity (S/cm)",
            "critical current density before short-circuit (mA/cm^2)",
            "interfacial resistance (ohm*cm^2)",
            "activation energy for Li-ion migration (eV)",
        ]
        hyps = []
        for i in range(n):
            local_seed = seed + i * 97
            fam = families[local_seed % len(families)]
            mech = mechanisms[(local_seed // 7) % len(mechanisms)]
            metric = metrics[(local_seed // 13) % len(metrics)]
            hyps.append(
                {
                    "statement": (
                        f"Substituting a low-concentration dopant into {fam.split(' (')[0]} "
                        f"lattice sites will improve {metric.split(' (')[0]} by promoting "
                        f"{mech.split(' to ')[0] if ' to ' in mech else mech}."
                    ),
                    "mechanism": mech,
                    "material_family": fam,
                    "target_metric": metric,
                    "rationale": (
                        "Analogous substitution strategies have shifted defect "
                        "concentrations and lowered migration barriers in structurally "
                        "related Li-ion conductors, suggesting a transferable design "
                        "principle for this family."
                    ),
                }
            )
        return {"hypotheses": hyps}

    def _mock_literature_scout(self, user: str, seed: int) -> dict[str, Any]:
        return {
            "scout_summary": (
                "Prior work in this material family has largely focused on bulk "
                "conductivity optimization via cation doping, with comparatively "
                "less attention paid to the specific interfacial or grain-boundary "
                "mechanism proposed here. Several studies report similar dopant "
                "strategies applied to adjacent compositions."
            ),
            "closest_prior_work_gap": (
                "Existing studies establish the general dopant-conductivity trend "
                "but do not isolate the specific mechanism-metric pairing proposed "
                "here, and none report critical-current-density data under the "
                "proposed processing route."
            ),
        }

    def _mock_critic(self, user: str, seed: int, rng_bucket: float) -> dict[str, Any]:
        bias = self._persona_bias()
        feasibility = max(0.0, min(10.0, 5.5 + bias + (rng_bucket - 0.5) * 4))
        judge_novelty = max(0.0, min(10.0, 5.0 + bias + (rng_bucket - 0.3) * 5))
        contradictions = []
        flaws = []
        if rng_bucket < 0.25 and self.persona != "lenient":
            contradictions.append(
                "Predicted conductivity gain is difficult to reconcile with the "
                "known trade-off between dopant concentration and grain-boundary "
                "resistance reported for structurally similar systems."
            )
        if rng_bucket < 0.4:
            flaws.append(
                "Rationale does not address expected mechanical/phase-purity "
                "side effects of the proposed substitution."
            )
        if self.persona == "strict":
            flaws.append(
                "No quantitative bound is given for the expected magnitude of improvement."
            )
        overall = max(0.0, min(10.0, 0.5 * feasibility + 0.5 * judge_novelty - 0.7 * len(contradictions)))
        if overall < 4.0:
            rec = "reject"
        elif overall < 7.0:
            rec = "revise"
        else:
            rec = "accept"
        return {
            "llm_judge_novelty_score": round(judge_novelty, 2),
            "llm_judge_novelty_rationale": (
                "Judged relative to the retrieved literature summary: the mechanism "
                "is a recognizable extension of known strategies rather than an "
                "entirely new principle, but the specific metric target is "
                "under-explored in the retrieved records."
            ),
            "feasibility_score": round(feasibility, 2),
            "feasibility_rationale": (
                "Processing route implied by the mechanism is achievable with "
                "standard solid-state synthesis techniques, though achieving the "
                "targeted improvement magnitude is uncertain without further "
                "compositional tuning."
            ),
            "contradictions": contradictions,
            "identified_flaws": flaws,
            "suggested_revision": (
                None
                if rec == "accept"
                else (
                    "Narrow the claim to a specific dopant concentration range and "
                    "explicitly address the grain-boundary resistance trade-off."
                )
            ),
            "overall_score": round(overall, 2),
            "recommendation": rec,
        }

    def _mock_synthesizer(self, user: str, seed: int) -> dict[str, Any]:
        return {
            "protocol_steps": [
                "Synthesize target composition via solid-state reaction or "
                "co-precipitation, per the proposed dopant stoichiometry.",
                "Sinter pressed pellets under controlled atmosphere; vary "
                "sintering temperature/time as the primary process variable.",
                "Characterize phase purity via XRD and microstructure via SEM.",
                "Measure ionic conductivity via electrochemical impedance "
                "spectroscopy (EIS) across a temperature series to extract "
                "activation energy.",
                "Assemble symmetric Li|electrolyte|Li cells; measure critical "
                "current density via stepped current cycling.",
            ],
            "independent_variables": ["dopant concentration", "sintering temperature"],
            "dependent_variables": [
                "room-temperature ionic conductivity",
                "activation energy",
                "critical current density",
            ],
            "controls": ["undoped baseline composition", "fixed sintering atmosphere"],
            "toy_simulation_code": (
                "import math\n"
                "# Toy Arrhenius model: sigma(T) = sigma0 * exp(-Ea / (k_B * T))\n"
                "k_B = 8.617333262e-5  # eV/K\n"
                "sigma0 = 1.0e4  # S/cm, prefactor (order-of-magnitude placeholder)\n"
                "Ea_baseline = 0.35  # eV, undoped baseline\n"
                "Ea_doped = 0.28  # eV, hypothesized reduction from doping\n"
                "T = 298.15  # K, room temperature\n"
                "sigma_baseline = sigma0 * math.exp(-Ea_baseline / (k_B * T))\n"
                "sigma_doped = sigma0 * math.exp(-Ea_doped / (k_B * T))\n"
                "improvement_factor = sigma_doped / sigma_baseline\n"
                "print(f'baseline_conductivity_S_cm={sigma_baseline:.6e}')\n"
                "print(f'doped_conductivity_S_cm={sigma_doped:.6e}')\n"
                "print(f'improvement_factor={improvement_factor:.3f}')\n"
            ),
            "predicted_outcome": (
                "A reduction in activation energy of ~0.05-0.10 eV from doping "
                "should yield a room-temperature conductivity improvement on the "
                "order of 5-15x, per the toy Arrhenius model, before accounting "
                "for grain-boundary resistance penalties."
            ),
        }
