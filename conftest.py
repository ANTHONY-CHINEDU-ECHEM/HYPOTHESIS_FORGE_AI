import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hypothesisforge.config import RunConfig
from hypothesisforge.llm_client import MockLLMClient
from hypothesisforge.tools.corpus_search import CorpusIndex


@pytest.fixture(scope="session")
def run_config() -> RunConfig:
    return RunConfig(n_seed_hypotheses=3, max_iterations=2, top_k_final=2)


@pytest.fixture(scope="session")
def mock_llm() -> MockLLMClient:
    return MockLLMClient()


@pytest.fixture(scope="session")
def corpus(run_config) -> CorpusIndex:
    return CorpusIndex(run_config.domain_profile().corpus_path, embedding_backend="tfidf")
