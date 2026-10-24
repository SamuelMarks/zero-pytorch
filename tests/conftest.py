"""Pytest configuration and shared fixtures for zero_torch tests."""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator

import pytest

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../ml-switcheroo-compiler/src")
    ),
)
from ml_switcheroo_compiler.core.config import EagerMode


@pytest.fixture(autouse=True)
def switcheroo_config() -> Iterator[None]:
    """Configures `EagerMode` by default and resets the tracing state after each test.

    Yields:
        None during test execution.
    """
    with EagerMode():
        try:
            yield
        finally:
            from zero_torch.tracing import _tracer

            if _tracer.is_tracing:
                _tracer.stop_tracing()
