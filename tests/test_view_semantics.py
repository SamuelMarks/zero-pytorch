"""Tests for View semantics."""

import numpy as np
from zero_torch import Tensor
from ml_switcheroo.tracing import _tracer


def test_view():
    t = Tensor(np.ones((2, 3)))
    t2 = t.view(6)
    assert t2.shape == (6,)

    t3 = t.view((3, 2))
    assert t3.shape == (3, 2)

    _tracer.start_tracing()
    t4 = t.view(6)
    assert t4.shape == (6,)
    _tracer.stop_tracing()


def test_reshape():
    t = Tensor(np.ones((2, 3)))
    t2 = t.reshape(6)
    assert t2.shape == (6,)


def test_contiguous():
    t = Tensor(np.ones((2, 3)))
    assert t.contiguous() is t


def test_squeeze():
    t = Tensor(np.ones((1, 2, 1, 3)))
    t2 = t.squeeze()
    assert t2.shape == (2, 3)

    t3 = t.squeeze(0)
    assert t3.shape == (2, 1, 3)

    _tracer.start_tracing()
    t4 = t.squeeze()
    assert t4.shape == (2, 3)
    t5 = t.squeeze(0)
    assert t5.shape == (2, 1, 3)
    _tracer.stop_tracing()


def test_view_traced_with_id():
    t = Tensor(np.ones((2, 3)))
    _tracer.start_tracing()
    t2 = t + t
    _ = t2.reshape(6)
    _ = t2.squeeze()
    _tracer.stop_tracing()
