"""Tests for `zero_torch` arithmetic operations and tensor properties."""

from __future__ import annotations

import numpy as np
from ml_switcheroo_compiler.core.config import config

import zero_torch
from zero_torch import Tensor
from zero_torch.tracing import _tracer


def test_torch_add() -> None:
    """Verifies `Tensor.__add__` in eager and tracing modes."""
    t1 = Tensor([1, 2])
    t2 = Tensor([3, 4])
    t3 = t1 + t2
    np.testing.assert_allclose(t3.tolist(), [4, 6])

    _tracer.start_tracing()
    t4 = t1 + t2
    assert not t4.requires_grad
    _tracer.stop_tracing()


def test_torch_sub() -> None:
    """Verifies `Tensor.__sub__`."""
    t1 = Tensor([3, 4])
    t2 = Tensor([1, 2])
    t3 = t1 - t2
    np.testing.assert_allclose(t3.tolist(), [2, 2])


def test_torch_mul() -> None:
    """Verifies `Tensor.__mul__`."""
    t1 = Tensor([1, 2])
    t2 = Tensor([3, 4])
    t3 = t1 * t2
    np.testing.assert_allclose(t3.tolist(), [3, 8])


def test_torch_div() -> None:
    """Verifies `Tensor.__truediv__`."""
    t1 = Tensor([6, 8])
    t2 = Tensor([2, 2])
    t3 = t1 / t2
    np.testing.assert_allclose(t3.tolist(), [3, 4])


def test_torch_matmul() -> None:
    """Verifies `Tensor.__matmul__`."""
    t1 = Tensor([[1, 2]])
    t2 = Tensor([[3], [4]])
    t3 = t1 @ t2
    np.testing.assert_allclose(t3.tolist(), [[11]])


def test_torch_math_tracing() -> None:
    """Verifies arithmetic operators in graph tracing mode (`eager_mode = False`)."""
    prev = config.eager_mode
    config.eager_mode = False
    try:
        _tracer.start_tracing()
        t1 = Tensor([[1, 2]])
        t2 = Tensor([[3], [4]])
        _ = t1 - t1
        _ = t1 * t1
        _ = t1 / t1
        _ = t1 @ t2
        _tracer.stop_tracing()
    finally:
        config.eager_mode = prev


def test_tensor_copy_init() -> None:
    """Verifies initializing a `Tensor` from another `Tensor` shares underlying data."""
    t1 = Tensor([1])
    t2 = Tensor(t1)
    assert t1._data is t2._data


def test_dtype_and_contiguous() -> None:
    """Verifies `Tensor.dtype` and `Tensor.contiguous()`."""
    t = Tensor(np.array([[1, 2], [3, 4]], dtype=np.float32))
    assert t.dtype == zero_torch.float32
    t2 = Tensor(np.array([[1, 2], [3, 4]], dtype=np.float32).T)
    assert t2.contiguous().shape == (2, 2)


def test_tensor_none_ops() -> None:
    """Verifies `contiguous()` and `squeeze()` on `Tensor(None)`."""
    t = Tensor(None)
    assert t.contiguous() is t
    assert t.squeeze() is t
