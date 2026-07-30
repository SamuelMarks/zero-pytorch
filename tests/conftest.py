import os
import sys

import pytest

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../ml-switcheroo-compiler/src")
    ),
)
from ml_switcheroo_compiler.core.config import EagerMode

try:
    from ml_switcheroo_compiler.core.errors import (
        ShapeMismatchError,
        UnimplementedMathError,
    )
except ImportError:
    UnimplementedMathError = Exception
    ShapeMismatchError = Exception


@pytest.fixture(autouse=True)
def switcheroo_config():
    """Tests for switcheroo_config."""
    # Unified pytest configuration that imports switcheroo config contexts
    with EagerMode():
        try:
            yield
        finally:
            from zero_torch.tracing import _tracer

            if _tracer.is_tracing:
                _tracer.stop_tracing()
