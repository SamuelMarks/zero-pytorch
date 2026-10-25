"""Tests for `Tensor` view, reshape, contiguous, and squeeze semantics."""

from __future__ import annotations

import zero_torch as torch
from zero_torch.tracing import _tracer


def test_view() -> None:
    """Verifies `Tensor.view` with variadic args and shape tuple in eager and tracing modes."""
    t = torch.ones((2, 3))
    t2 = t.view(3, 2)
    assert t2.shape == (3, 2)
    t3 = t.view((6,))
    assert t3.shape == (6,)

    _tracer.start_tracing()
    t4 = t.view(3, 2)
    assert t4.shape == (3, 2)
    _tracer.stop_tracing()


def test_reshape() -> None:
    """Verifies `Tensor.reshape` in eager and tracing modes."""
    t = torch.ones((2, 3))
    t2 = t.reshape(3, 2)
    assert t2.shape == (3, 2)


def test_contiguous() -> None:
    """Verifies `Tensor.contiguous` returns a tensor with matching shape."""
    t = torch.ones((2, 3))
    t2 = t.contiguous()
    assert t2.shape == (2, 3)


def test_squeeze() -> None:
    """Verifies `Tensor.squeeze` with and without explicit dimension in eager and tracing modes."""
    t = torch.ones((1, 2, 1, 3))
    t2 = t.squeeze()
    assert t2.shape == (2, 3)

    t3 = t.squeeze(0)
    assert t3.shape == (2, 1, 3)

    _tracer.start_tracing()
    t4 = t.squeeze()
    assert t4.shape == (2, 3)
    t5 = t.squeeze(2)
    assert t5.shape == (1, 2, 3)
    _tracer.stop_tracing()


def test_view_traced_with_id() -> None:
    """Verifies `reshape` and `squeeze` on an intermediate traced tensor."""
    t = torch.ones((2, 3))
    _tracer.start_tracing()
    t2 = t + t
    _ = t2.reshape(6)
    _ = t2.squeeze()
    _tracer.stop_tracing()
