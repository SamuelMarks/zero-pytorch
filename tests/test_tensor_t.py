"""Tests for `Tensor.T` transpose property in eager and tracing modes."""

from __future__ import annotations

import zero_torch as torch
from zero_torch.tracing import _tracer


def test_tensor_t() -> None:
    """Verifies `Tensor.T` on 2D and 1D tensors in eager mode."""
    t = torch.ones((2, 3))
    t2 = t.T
    assert t2.shape == (3, 2)

    t1d = torch.ones((2,))
    assert t1d.T.shape == (2,)


def test_tensor_t_traced() -> None:
    """Verifies `Tensor.T` on traced tensors."""
    t = torch.ones((2, 3))
    _tracer.start_tracing()

    t_traced = t + t
    t3 = t_traced.T
    assert t3.shape == (3, 2)

    t4 = t.T
    assert t4.shape == (3, 2)

    _tracer.stop_tracing()
