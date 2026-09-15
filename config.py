"""
Central configuration for HypothesisForge.

Everything that makes this system "domain-grounded" rather than a generic
research-agent template lives here: the domain description, the constraint
vocabulary the Ideator must respect, and the scoring weights the Critic /
Synthesizer use. Swapping DOMAIN_PROFILE lets you retarget the whole system
at a different narrow domain without touching agent logic.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = PROJECT_ROOT / "data" / "corpus"
DEFAULT_CORPUS_PATH = CORPUS_DIR / "materials_science_corpus.json"


@dataclass(frozen=True)
class DomainProfile:
    """Grounds the whole pipeline in a specific, narrow scientific domain."""

    name: str
    short_description: str
    scope_statement: str
    known_material_families: tuple[str, ...]
    key_failure_modes: tuple[str, ...]
    key_metrics: tuple[str, ...]
    corpus_path: Path


MATERIALS_SCIENCE_SOLID_ELECTROLYTES = DomainProfile(
    name="solid_state_electrolytes",
    short_description=(
        "Solid-state electrolyte materials for lithium-metal batteries"
    ),
    scope_statement=(
        "Hypotheses must concern ion-conducting solid materials (ceramic, "
        "sulfide, halide, polymer, or composite) intended to replace liquid "
        "electrolytes in lithium-metal batteries. In scope: bulk ionic "
        "conductivity, grain-boundary transport, electrode/electrolyte "
        "interfacial stability, dendrite suppression, doping/substitution "
        "chemistry, processing routes (sintering, cold pressing, thin-film "
        "deposition), and mechanical properties relevant to cell assembly. "
        "Out of scope: liquid or gel electrolytes, non-lithium chemistries, "
        "and device-level engineering unrelated to the electrolyte material "
        "itself."
    ),
    known_material_families=(
        "garnet-type oxides (e.g., LLZO-family)",
        "NASICON-type oxides",
        "perovskite-type oxides",
        "sulfide argyrodites and thio-LISICON phases",
        "lithium halides and halide-doped analogues",
        "polymer-in-ceramic and ceramic-in-polymer composites",
        "single-ion conducting polymer electrolytes",
        "amorphous / glassy Li-ion conductors",
    ),
    key_failure_modes=(
        "lithium dendrite penetration through grain boundaries or pores",
        "poor electrode/electrolyte interfacial contact (voiding)",
        "chemical/electrochemical instability against Li metal",
        "low room-temperature ionic conductivity vs. liquid electrolytes",
        "high grain-boundary resistance relative to bulk resistance",
        "air/moisture sensitivity (especially sulfides)",
        "mechanical brittleness limiting cell fabrication",
        "interdiffusion / interphase formation at cathode interface",
    ),
    key_metrics=(
        "room-temperature ionic conductivity (S/cm)",
        "activation energy for Li-ion migration (eV)",
        "critical current density before short-circuit (mA/cm^2)",
        "electrochemical stability window (V vs Li/Li+)",
        "interfacial resistance (ohm*cm^2)",
        "relative density / porosity (%)",
    ),
    corpus_path=DEFAULT_CORPUS_PATH,
)

# Registry so the CLI can select domains by name; add more DomainProfiles
# here to retarget HypothesisForge at cognitive psychology, climate
# modeling, etc., without changing any agent code.
DOMAIN_REGISTRY: dict[str, DomainProfile] = {
    MATERIALS_SCIENCE_SOLID_ELECTROLYTES.name: MATERIALS_SCIENCE_SOLID_ELECTROLYTES,
}

DEFAULT_DOMAIN = MATERIALS_SCIENCE_SOLID_ELECTROLYTES.name


@dataclass
class RunConfig:
    """Parameters controlling a single pipeline run."""

    domain: str = DEFAULT_DOMAIN
    n_seed_hypotheses: int = 4
    max_iterations: int = 3
    top_k_final: int = 3
    novelty_weight: float = 0.35
    feasibility_weight: float = 0.35
    contradiction_penalty_weight: float = 0.30
    critic_reject_threshold: float = 4.0  # overall_score below this -> reject
    critic_revise_threshold: float = 7.0  # below this (and >= reject) -> revise
    embedding_backend: str = "tfidf"  # "tfidf" | "sentence-transformers"
    llm_model: str = field(
        default_factory=lambda: os.environ.get(
            "HYPOTHESISFORGE_MODEL", "claude-sonnet-4-6"
        )
    )
    judge_persona: str = "balanced"  # "balanced" | "strict" | "lenient"
    use_critic: bool = True  # ablation switch
    random_seed: int = 7

    def domain_profile(self) -> DomainProfile:
        return DOMAIN_REGISTRY[self.domain]


ANTHROPIC_API_KEY_ENV = "ANTHROPIC_API_KEY"


def has_live_api_key() -> bool:
    return bool(os.environ.get(ANTHROPIC_API_KEY_ENV))
