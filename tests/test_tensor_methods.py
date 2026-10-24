"""Tests for `Tensor` methods (`item`, `tolist`, `numpy`, `repr`, `str`)."""

from __future__ import annotations

import pytest
from ml_switcheroo_compiler.core.tensor import (
    Tensor as _SwitcherooTensor,
)
from ml_switcheroo_compiler.core.tensor import (
    TensorConfig as _TensorConfig,
)

from zero_torch.tensor import Tensor


def test_tensor_item() -> None:
    """Verifies `Tensor.item()` on single-element tensors and error on multi-element tensors."""
    t1 = Tensor(42.0)
    assert t1.item() == 42.0

    t2 = Tensor([5])
    assert t2.item() == 5

    t3 = Tensor([[True]])
    assert t3.item() is True

    t4 = Tensor([1, 2])
    with pytest.raises(ValueError):
        t4.item()


def test_tensor_tolist() -> None:
    """Verifies `Tensor.tolist()` across 0D, 1D, 2D, and 3D tensors."""
    t1 = Tensor([[1, 2], [3, 4]])
    assert t1.tolist() == [[1, 2], [3, 4]]

    t2 = Tensor([])
    assert t2.tolist() == []

    t3 = Tensor(5)
    assert t3.tolist() == 5

    t4 = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    assert t4.tolist() == [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]


def test_tensor_numpy() -> None:
    """Verifies `Tensor.numpy()` returns a NumPy array matching the tensor data."""
    t1 = Tensor([1.0, 2.0, 3.0])
    arr = t1.numpy()
    assert list(arr) == [1.0, 2.0, 3.0]


def test_tensor_repr_str() -> None:
    """Verifies `str(tensor)` and `repr(tensor)` formatting."""
    t1 = Tensor([1.23456, 2.0])
    s = str(t1)
    assert "tensor" in s
    assert "1.234" in s or "1.23456" in s

    t2 = Tensor([[1, 2], [3, 4]])
    r = repr(t2)
    assert "tensor" in r
    assert "[" in r

    t3 = Tensor(5.0)
    assert "tensor" in str(t3)

    t4 = Tensor([[[1], [2]], [[3], [4]]])
    assert "tensor" in str(t4)

    t5 = Tensor([1, [2], [3, 4]])
    assert "tensor" in repr(t5)


def test_tensor_pure_python_paths() -> None:
    """Verifies `Tensor.tolist()` and `Tensor.item()` when backed by pure Python list data."""
    t_data = _SwitcherooTensor(
        data=[1, 2, 3, 4], config=_TensorConfig(shape=(2, 2), dtype=None, device="cpu")
    )
    t = Tensor(t_data)
    assert t.tolist() == [[1, 2], [3, 4]]
    assert "tensor" in repr(t)

    t_data2 = _SwitcherooTensor(
        data=[42.0], config=_TensorConfig(shape=(1,), dtype=None, device="cpu")
    )
    t2 = Tensor(t_data2)
    assert t2.item() == 42.0

    t_data3 = _SwitcherooTensor(
        data=[1.0, 2.0], config=_TensorConfig(shape=(2,), dtype=None, device="cpu")
    )
    t3 = Tensor(t_data3)
    with pytest.raises(ValueError):
        t3.item()


def test_tensor_threshold() -> None:
    """Verifies `repr(tensor)` truncates tensors with more than 1000 elements."""
    t_data = [i for i in range(1005)]
    t = Tensor(t_data)
    assert "..." in repr(t)
