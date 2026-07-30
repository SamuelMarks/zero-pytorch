import numpy as np

from zero_torch.tensor import Tensor
from zero_torch.tracing import _tracer

try:
    from ml_switcheroo_compiler.core.errors import (
        ShapeMismatchError,
        UnimplementedMathError,
    )
except ImportError:
    UnimplementedMathError = Exception
    ShapeMismatchError = Exception


def test_tensor_t():
    """Tests for test_tensor_t."""
    t = Tensor(np.ones((2, 3)))
    t2 = t.T
    assert t2.shape == (3, 2)

    t1d = Tensor(np.ones((2,)))
    assert t1d.T.shape == (2,)


def test_tensor_t_traced():
    """Tests for test_tensor_t_traced."""
    t = Tensor(np.ones((2, 3)))
    _tracer.start_tracing()

    t_traced = t + t
    t3 = t_traced.T
    assert t3.shape == (3, 2)

    t4 = t.T
    assert t4.shape == (3, 2)

    _tracer.stop_tracing()
