"""Shared pytest fixtures for DisasterResponseEnv tests."""
from __future__ import annotations

import os
import sys

import pytest

# Allow imports from repo root (models.py lives there)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


ENV_BASE_URL: str = os.getenv("ENV_BASE_URL", "http://localhost:7860").rstrip("/")


@pytest.fixture(scope="session")
def base_url() -> str:
    return ENV_BASE_URL


@pytest.fixture
def env():
    """Fresh DisasterResponseEnvironment instance (not via HTTP)."""
    from server.environment import DisasterResponseEnvironment
    return DisasterResponseEnvironment()
