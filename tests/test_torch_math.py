"""Tests for `zero_torch` arithmetic operations and tensor properties."""

from __future__ import annotations

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import (
    Tensor as SwitcherooTensor,
)
from ml_switcheroo_compiler.core.tensor import (
    TensorConfig,
)

import zero_torch
from tests.conftest import assert_allclose
from zero_torch import Tensor
from zero_torch._types import NestedList, Scalar
from zero_torch.tracing import _tracer


class _NonContiguousMockArray:
    """Pure-Python mock array implementing `_ContiguousArrayProtocol` and `_ArrayDataProtocol`."""

    def __init__(self, contiguous: bool = False) -> None:
        """Initializes `_NonContiguousMockArray`.

        Args:
            contiguous: Whether the mock array reports `C_CONTIGUOUS = True`.
        """
        self._contiguous = contiguous

    @property
    def shape(self) -> tuple[int, ...]:
        """Returns the shape tuple of the array.

        Returns:
            Shape tuple `(2, 2)`.
        """
        return (2, 2)

    @property
    def dtype(self) -> str:
        """Returns the dtype string of the array.

        Returns:
            String `'float32'`.
        """
        return "float32"

    def tolist(self) -> NestedList | Scalar:
        """Returns the nested Python list representation.

        Returns:
            Nested 2x2 list of floats.
        """
        return [[1.0, 2.0], [3.0, 4.0]]

    @property
    def flags(self) -> dict[str, bool]:
        """Returns the memory layout flags dictionary.

        Returns:
            Dictionary mapping `'C_CONTIGUOUS'` to `self._contiguous`.
        """
        return {"C_CONTIGUOUS": self._contiguous}

    def copy(self, order: str = "C") -> _NonContiguousMockArray:
        """Returns a C-contiguous copy of this mock array.

        Args:
            order: Memory layout order.

        Returns:
            New contiguous `_NonContiguousMockArray`.
        """
        _ = order
        return _NonContiguousMockArray(contiguous=True)


def test_torch_add() -> None:
    """Verifies `Tensor.__add__` in eager and tracing modes."""
    t1 = Tensor([1, 2])
    t2 = Tensor([3, 4])
    t3 = t1 + t2
    assert_allclose(t3.tolist(), [4, 6])

    _tracer.start_tracing()
    t4 = t1 + t2
    assert not t4.requires_grad
    _tracer.stop_tracing()


def test_torch_sub() -> None:
    """Verifies `Tensor.__sub__`."""
    t1 = Tensor([3, 4])
    t2 = Tensor([1, 2])
    t3 = t1 - t2
    assert_allclose(t3.tolist(), [2, 2])


def test_torch_mul() -> None:
    """Verifies `Tensor.__mul__`."""
    t1 = Tensor([1, 2])
    t2 = Tensor([3, 4])
    t3 = t1 * t2
    assert_allclose(t3.tolist(), [3, 8])


def test_torch_div() -> None:
    """Verifies `Tensor.__truediv__`."""
    t1 = Tensor([6, 8])
    t2 = Tensor([2, 2])
    t3 = t1 / t2
    assert_allclose(t3.tolist(), [3, 4])


def test_torch_matmul() -> None:
    """Verifies `Tensor.__matmul__`."""
    t1 = Tensor([[1, 2]])
    t2 = Tensor([[3], [4]])
    t3 = t1 @ t2
    assert_allclose(t3.tolist(), [[11]])


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
    """Verifies `Tensor.dtype` and `Tensor.contiguous()` for contiguous and non-contiguous buffers."""
    t = Tensor([[1.0, 2.0], [3.0, 4.0]], dtype=zero_torch.float32)
    assert t.dtype == zero_torch.float32
    assert t.contiguous().shape == (2, 2)

    non_contig_st = SwitcherooTensor(
        data=_NonContiguousMockArray(contiguous=False),
        config=TensorConfig(shape=(2, 2), dtype=DType.Float32, device="cpu"),
    )
    t2 = Tensor(non_contig_st)
    assert t2.contiguous().shape == (2, 2)


def test_tensor_none_ops() -> None:
    """Verifies `contiguous()` and `squeeze()` on `Tensor(None)`."""
    t = Tensor(None)
    assert t.shape == (0,)
    assert t.contiguous().shape == (0,)
    assert t.squeeze().shape == (0,)
