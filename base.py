"""Shared base class for all agents."""

from __future__ import annotations

from ..config import DomainProfile, RunConfig
from ..llm_client import LLMClient


class Agent:
    name: str = "agent"

    def __init__(self, llm: LLMClient, config: RunConfig):
        self.llm = llm
        self.config = config

    @property
    def domain(self) -> DomainProfile:
        return self.config.domain_profile()

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<{self.__class__.__name__} model={self.llm.model_name}>"
