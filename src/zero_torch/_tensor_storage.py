"""Internal storage, protocol, and nested sequence helpers for `zero_torch.Tensor`."""

from __future__ import annotations

import functools
import operator
from collections.abc import Iterator
from typing import Protocol, runtime_checkable

from ml_switcheroo_compiler.core.dtype import DType

from zero_torch._types import (
    NestedList,
    OpArg,
    Scalar,
    SliceIndexType,
)


@runtime_checkable
class _ArrayDataProtocol(Protocol):
    """Protocol for backend array payloads with shape, dtype, and tolist()."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Returns the shape tuple of the array.

        Returns:
            Tuple of dimension sizes.
        """

    @property
    def dtype(self) -> str | DType:
        """Returns the data type of the array.

        Returns:
            Data type identifier.
        """

    def tolist(self) -> NestedList | Scalar:
        """Converts the array to a Python list or scalar.

        Returns:
            Python list or scalar representation.
        """


@runtime_checkable
class _NumpyArrayProtocol(_ArrayDataProtocol, Protocol):
    """Protocol for array objects returned by `Tensor.numpy()`."""

    def __iter__(self) -> Iterator[Scalar | _NumpyArrayProtocol]:
        """Iterates over the first dimension of the array.

        Returns:
            Iterator over elements or sub-arrays.
        """

    def __getitem__(
        self, index: int | SliceIndexType | tuple[int | SliceIndexType, ...]
    ) -> _NumpyArrayProtocol:
        """Indexes or slices the array.

        Args:
            index: Integer, slice, or tuple of indices.

        Returns:
            Indexed sub-array.
        """

    def __lt__(self, other: Scalar | _NumpyArrayProtocol) -> _NumpyArrayProtocol:
        """Element-wise less-than comparison.

        Args:
            other: Scalar or array to compare against.

        Returns:
            Boolean comparison array.
        """

    def all(self) -> bool:
        """Returns True if all elements evaluate to True.

        Returns:
            Boolean reduction result.
        """


@runtime_checkable
class _ContiguousArrayProtocol(Protocol):
    """Protocol for backend arrays supporting C-contiguous layout checks."""

    @property
    def flags(self) -> dict[str, bool]:
        """Returns the memory layout flags mapping.

        Returns:
            Dictionary of memory layout flags.
        """

    def copy(self, order: str = "C") -> _ArrayDataProtocol:
        """Returns a contiguous copy of the array.

        Args:
            order: Memory layout order ('C' or 'F').

        Returns:
            Copied array object.
        """


def _infer_shape_and_dtype(
    v: OpArg | _ArrayDataProtocol,
) -> tuple[tuple[int, ...], str]:
    """Infers the shape tuple and dtype string from an input value.

    Args:
        v: Scalar, nested sequence, or backend array object.

    Returns:
        A tuple of `(shape, dtype_string)`.
    """
    if isinstance(v, bool):
        return (), "bool"
    if isinstance(v, int):
        return (), "int64"
    if isinstance(v, (float, complex)):
        return (), "float32"
    if isinstance(v, (list, tuple)):
        if len(v) == 0:
            return (0,), "float32"
        inner_shape, inner_dt = _infer_shape_and_dtype(v[0])
        return (len(v),) + inner_shape, inner_dt
    if isinstance(v, _ArrayDataProtocol):
        return tuple(int(s) for s in v.shape), str(v.dtype)
    return (), "float32"


def _flatten_nested(lst: OpArg) -> list[Scalar]:
    """Flattens a nested list or tuple of scalars into a flat 1D list.

    Args:
        lst: Nested list, tuple, or scalar element.

    Returns:
        Flat list of scalar values.
    """
    if isinstance(lst, (list, tuple)):
        return functools.reduce(operator.iadd, (_flatten_nested(i) for i in lst), [])
    if isinstance(lst, (int, float, bool, complex)):
        return [lst]
    return [0.0]


def _nested_from_flat(
    flat_data: list[Scalar] | tuple[Scalar, ...], shape: tuple[int, ...]
) -> NestedList:
    """Reconstructs a nested list from a flat sequence and target shape.

    Args:
        flat_data: Flat sequence of scalar elements.
        shape: Target multi-dimensional shape tuple.

    Returns:
        Nested list matching `shape`.
    """
    if len(shape) <= 1:
        return list(flat_data)
    size = shape[0]
    chunk_size = 1
    for s in shape[1:]:
        chunk_size *= s
    result: NestedList = []
    for i in range(size):
        sub = _nested_from_flat(
            flat_data[i * chunk_size : (i + 1) * chunk_size], shape[1:]
        )
        result.append(sub)
    return result


def _compute_contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Computes row-major C-contiguous strides for a given `shape`.

    Args:
        shape: Tensor dimension sizes tuple.

    Returns:
        Tuple of row-major strides for each dimension.
    """
    if not shape:
        return ()
    strides = [1] * len(shape)
    running = 1
    for idx in range(len(shape) - 1, -1, -1):
        strides[idx] = running
        running *= max(1, shape[idx])
    return tuple(strides)
