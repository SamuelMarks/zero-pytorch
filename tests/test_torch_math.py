"""Tests for zero_torch math functions."""

import numpy as np
import zero_torch
from zero_torch import Tensor
from ml_switcheroo.tracing import _tracer


def test_torch_add():
    """Tests for test_torch_add."""
    t1 = Tensor([1, 2])
    t2 = Tensor([3, 4])
    t3 = t1 + t2
    np.testing.assert_allclose(t3._data, [4, 6])

    _tracer.start_tracing()
    t4 = t1 + t2
    assert not t4.requires_grad
    _tracer.stop_tracing()


def test_torch_sub():
    """Tests for test_torch_sub."""
    t1 = Tensor([3, 4])
    t2 = Tensor([1, 2])
    t3 = t1 - t2
    np.testing.assert_allclose(t3._data, [2, 2])


def test_torch_mul():
    """Tests for test_torch_mul."""
    t1 = Tensor([1, 2])
    t2 = Tensor([3, 4])
    t3 = t1 * t2
    np.testing.assert_allclose(t3._data, [3, 8])


def test_torch_div():
    """Tests for test_torch_div."""
    t1 = Tensor([6, 8])
    t2 = Tensor([2, 2])
    t3 = t1 / t2
    np.testing.assert_allclose(t3._data, [3, 4])


def test_torch_matmul():
    """Tests for test_torch_matmul."""
    t1 = Tensor([[1, 2]])
    t2 = Tensor([[3], [4]])
    t3 = t1 @ t2
    np.testing.assert_allclose(t3._data, [[11]])


def test_torch_math_tracing():
    """Tests for test_torch_math_tracing."""
    _tracer.start_tracing()
    t1 = Tensor([[1, 2]])
    t2 = Tensor([[3], [4]])
    t1 - t1
    t1 * t1
    t1 / t1
    t1 @ t2
    _tracer.stop_tracing()


def test_tensor_copy_init():
    """Tests for test_tensor_copy_init."""
    t1 = Tensor([1])
    t2 = Tensor(t1)
    assert t1._data is t2._data


def test_dtype_and_contiguous():
    """Tests for test_dtype_and_contiguous."""
    import numpy as np

    t = Tensor(np.array([[1, 2], [3, 4]], dtype=np.float32))
    assert t.dtype == zero_torch.float32
    # Create non-contiguous tensor
    t2 = Tensor(np.array([[1, 2], [3, 4]], dtype=np.float32).T)
    assert t2.contiguous()._data.flags["C_CONTIGUOUS"]


def test_tensor_none_ops():
    """Tests for test_tensor_none_ops."""
    t = Tensor(None)
    assert t.contiguous() is t
    assert t.squeeze() is t
