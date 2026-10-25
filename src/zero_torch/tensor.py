"""Tensor API."""

from __future__ import annotations

import functools
import math
import operator
import uuid
from collections.abc import Iterator, Sequence
from typing import Protocol, runtime_checkable

from ml_switcheroo_compiler import ops
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import (
    Tensor as _SwitcherooTensor,
)
from ml_switcheroo_compiler.core.tensor import (
    TensorConfig as _TensorConfig,
)
from ml_switcheroo_compiler.ir.core import IRNode as LogicalNode

import zero_torch
from zero_torch._types import (
    DTypeLike,
    NestedList,
    NestedSequence,
    OpArg,
    Scalar,
    SliceIndexType,
    TensorLike,
)
from zero_torch.tracing import ProxyTensor, _tracer


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


def _to_tensor(
    x: OpArg | _ArrayDataProtocol, dtype: DTypeLike = None
) -> _SwitcherooTensor:
    """Converts the input to an `ml_switcheroo_compiler` Tensor.

    Args:
        x: Input data (Tensor, _SwitcherooTensor, ProxyTensor, array, or scalar/sequence).
        dtype: Optional desired data type of the tensor.

    Returns:
        The converted `_SwitcherooTensor`.
    """
    if isinstance(x, Tensor):
        x = x._tensor
    if isinstance(x, _SwitcherooTensor):
        if _tracer.is_tracing and not isinstance(x.data, ProxyTensor):
            out_id = str(uuid.uuid4())
            val = x.data.tolist() if isinstance(x.data, _ArrayDataProtocol) else x.data
            node = LogicalNode(
                id=out_id,
                op_type="Constant",
                attributes={"value": val},
                shape_metadata=x.shape,
            )
            _tracer.add_node(node)
            pt = ProxyTensor(id=out_id, shape=x.shape, dtype=x.dtype.value)
            return _SwitcherooTensor(
                data=pt,
                config=_TensorConfig(shape=x.shape, dtype=x.dtype, device=x.device),
            )
        return x
    if isinstance(x, ProxyTensor):
        dt = config.default_float_dtype
        try:
            if x.dtype:
                dt = DType(x.dtype)
        except (
            RuntimeError,
            ValueError,
            TypeError,
            AttributeError,
            KeyError,
            IndexError,
            ImportError,
            NotImplementedError,
        ):
            dt = config.default_float_dtype
        return _SwitcherooTensor(
            data=x,
            config=_TensorConfig(
                shape=x.shape,
                dtype=dt,
                device=config.default_device,
            ),
        )

    shape, dt_str = _infer_shape_and_dtype(x)

    dt = config.default_float_dtype
    if "float64" in dt_str:
        dt = DType.float64
    elif "float16" in dt_str:
        dt = DType.float16
    elif "float32" in dt_str:
        dt = DType.float32
    elif "uint64" in dt_str:
        dt = DType.uint64
    elif "int64" in dt_str:
        dt = DType.int64
    elif "uint32" in dt_str:
        dt = DType.uint32
    elif "int32" in dt_str:
        dt = DType.int32
    elif "uint16" in dt_str:
        dt = DType.uint16
    elif "int16" in dt_str:
        dt = DType.int16
    elif "uint8" in dt_str:
        dt = DType.uint8
    elif "int8" in dt_str:
        dt = DType.int8
    elif "complex128" in dt_str:
        dt = DType.complex128
    elif "complex64" in dt_str:
        dt = DType.complex64
    elif "bool" in dt_str:
        dt = DType.bool

    if dtype is not None:
        try:
            dt = DType(dtype)
        except (
            RuntimeError,
            ValueError,
            TypeError,
            AttributeError,
            KeyError,
            IndexError,
            ImportError,
            NotImplementedError,
        ):
            dt = config.default_float_dtype

    if not config.eager_mode and _tracer.is_tracing:
        out_id = str(uuid.uuid4())
        node = LogicalNode(
            id=out_id,
            op_type="Constant",
            attributes={"value": x},
            shape_metadata=shape,
        )
        _tracer.add_node(node)
        pt = ProxyTensor(id=out_id, shape=shape, dtype=dt.value)
        return _SwitcherooTensor(
            data=pt,
            config=_TensorConfig(shape=shape, dtype=dt, device=config.default_device),
        )

    if config.eager_mode and not isinstance(x, (_ArrayDataProtocol, ProxyTensor)):
        try:
            return ops.array(x, dtype=dt)
        except (
            RuntimeError,
            ValueError,
            TypeError,
            AttributeError,
            KeyError,
            IndexError,
            ImportError,
            NotImplementedError,
        ):
            pass

    payload: OpArg | _ArrayDataProtocol = (
        _flatten_nested(x) if isinstance(x, (list, tuple)) else x
    )

    return _SwitcherooTensor(
        data=payload,
        config=_TensorConfig(shape=shape, dtype=dt, device=config.default_device),
    )


def _wrap(
    x: Tensor
    | _SwitcherooTensor
    | ProxyTensor
    | Scalar
    | NestedSequence
    | tuple[_SwitcherooTensor | Tensor, ...]
    | list[_SwitcherooTensor | Tensor]
    | None,
) -> Tensor:
    """Wraps an `ml_switcheroo_compiler` Tensor or scalar into `zero_torch.Tensor`.

    Args:
        x: The tensor or scalar value to wrap.

    Returns:
        Wrapped `Tensor`.
    """
    if isinstance(x, Tensor):
        return x
    if isinstance(x, (tuple, list)):
        return _wrap(x[0]) if len(x) > 0 else Tensor(None)
    return Tensor(x)


def _wrap_tuple(
    x: tuple[_SwitcherooTensor | Tensor, ...]
    | list[_SwitcherooTensor | Tensor]
    | _SwitcherooTensor
    | Tensor
    | None,
) -> tuple[Tensor, ...]:
    """Wraps a tuple or sequence of `ml_switcheroo_compiler` Tensors into a tuple of `zero_torch.Tensor`s.

    Args:
        x: Tuple, list, or single tensor returned by a multi-output operation.

    Returns:
        Tuple of wrapped `Tensor` instances.
    """
    if isinstance(x, (tuple, list)):
        return tuple(_wrap(i) for i in x)
    return (_wrap(x),)


def _wrap_list(
    x: Sequence[_SwitcherooTensor | Tensor] | _SwitcherooTensor | Tensor | None,
) -> list[Tensor]:
    """Wraps a sequence of `ml_switcheroo_compiler` Tensors into a list of `zero_torch.Tensor`s.

    Args:
        x: Sequence or single tensor returned by a list-producing operation.

    Returns:
        List of wrapped `Tensor` instances.
    """
    if isinstance(x, (tuple, list)):
        return [_wrap(i) for i in x]
    return [_wrap(x)]


class Tensor:
    """A multi-dimensional matrix containing elements of a single data type."""

    _tensor: _SwitcherooTensor

    def __init__(
        self,
        data: OpArg | _ArrayDataProtocol = None,
        requires_grad: bool = False,
        dtype: DTypeLike = None,
    ) -> None:
        """Initializes a Tensor.

        Args:
            data: Initial data (list, tuple, backend array protocol, scalar, or another Tensor).
            requires_grad: Whether autograd should record operations on this tensor.
            dtype: Optional desired data type of the returned tensor.
        """
        if isinstance(data, Tensor):
            self._tensor = data._tensor
        elif data is None:
            self._tensor = _to_tensor([], dtype=dtype)
        else:
            self._tensor = _to_tensor(data, dtype=dtype)
        self.requires_grad: bool = requires_grad
        self.grad: Tensor | None = None
        from zero_torch.autograd.function import Function

        self.grad_fn: Function | None = None
        self._retains_grad: bool = False

    def digamma(self) -> Tensor:
        """Computes the logarithmic derivative of the gamma function on `self`.

        Returns:
            Resulting tensor after applying digamma.
        """
        return zero_torch.digamma(self)

    def erfc(self) -> Tensor:
        """Computes the complementary error function of `self`.

        Returns:
            Resulting tensor after applying erfc.
        """
        return zero_torch.erfc(self)

    def erfinv(self) -> Tensor:
        """Computes the inverse error function of `self`.

        Returns:
            Resulting tensor after applying erfinv.
        """
        return zero_torch.erfinv(self)

    def lgamma(self) -> Tensor:
        """Computes the natural logarithm of the absolute value of the gamma function on `self`.

        Returns:
            Resulting tensor after applying lgamma.
        """
        return zero_torch.lgamma(self)

    @property
    def is_leaf(self) -> bool:
        """Returns True if this Tensor is a leaf node in the autograd graph.

        Returns:
            True if this tensor is a leaf node, False otherwise.
        """
        return self.grad_fn is None

    def retain_grad(self) -> None:
        """Enables `.grad` attribute retention for non-leaf Tensors."""
        if not self.requires_grad:
            raise RuntimeError(
                "can't retain_grad on Tensor that has requires_grad=False"
            )
        self._retains_grad = True

    @property
    def data(self) -> ProxyTensor | _ArrayDataProtocol | list[Scalar] | Scalar | None:
        """Gets the underlying tensor data payload or proxy.

        Returns:
            The underlying data object or proxy.
        """
        return self._tensor.data

    @property
    def _data(self) -> ProxyTensor | _ArrayDataProtocol | list[Scalar] | Scalar | None:
        """Gets the underlying private tensor data payload or proxy.

        Returns:
            The underlying private data object or proxy.
        """
        return self._tensor.data

    @property
    def shape(self) -> tuple[int, ...]:
        """Gets the shape of the tensor.

        Returns:
            Tuple of dimension sizes.
        """
        return tuple(int(s or 0) for s in self._tensor.shape)

    @property
    def dtype(self) -> DType:
        """Gets the data type of the tensor elements.

        Returns:
            The tensor DType.
        """
        return self._tensor.dtype

    def view(self, *shape: int | tuple[int, ...] | list[int] | None) -> Tensor:
        """Returns a new tensor with the same data but a different shape.

        Args:
            *shape: Desired shape dimensions or a single shape tuple/list.

        Returns:
            Reshaped tensor.
        """
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            clean_shape = tuple(int(item) for item in shape[0] if item is not None)
        else:
            clean_shape = tuple(int(item) for item in shape if isinstance(item, int))
        return zero_torch.reshape(self, clean_shape)

    def reshape(self, *shape: int | tuple[int, ...] | list[int] | None) -> Tensor:
        """Returns a tensor with the same data and number of elements, but with the specified shape.

        Args:
            *shape: Desired shape dimensions or a single shape tuple/list.

        Returns:
            Reshaped tensor.
        """
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            clean_shape = tuple(int(item) for item in shape[0] if item is not None)
        else:
            clean_shape = tuple(int(item) for item in shape if isinstance(item, int))
        return zero_torch.reshape(self, clean_shape)

    def detach(self) -> Tensor:
        """Returns a new Tensor, detached from the current autograd graph.

        Returns:
            Detached Tensor sharing the underlying storage.
        """
        out = Tensor(self, requires_grad=False)
        out.grad_fn = None
        return out

    def item(self) -> Scalar:
        """Returns the value of a single-element tensor as a standard Python number.

        Returns:
            The scalar Python value.
        """
        arr = self.tolist()
        while isinstance(arr, list):
            if len(arr) != 1:
                raise ValueError(
                    "only one element tensors can be converted to Python scalars"
                )
            arr = arr[0]
        return arr

    def __bool__(self) -> bool:
        """Returns the truth value of the tensor.

        Returns:
            Boolean value of the tensor.
        """
        return bool(self.item()) if self.numel() == 1 else self.numel() > 0

    def __repr__(self) -> str:
        """Returns the canonical string representation of the tensor.

        Returns:
            Formatted string representation of the tensor.
        """
        if self.numel() > 1000:
            return "tensor([...])"
        return f"tensor({self.tolist()})"

    def __str__(self) -> str:
        """Returns the human-readable string representation of the tensor.

        Returns:
            Formatted string representation of the tensor.
        """
        return self.__repr__()

    def tolist(self) -> NestedList | Scalar:
        """Returns the tensor data as a (possibly nested) Python list or scalar.

        Returns:
            Nested Python list or scalar containing the tensor's elements.
        """
        raw = self._tensor.data
        if isinstance(raw, _ArrayDataProtocol):
            return raw.tolist()
        if isinstance(raw, (list, tuple)):
            return _nested_from_flat(raw, self.shape)
        if isinstance(raw, (int, float, bool, complex)):
            return [raw]
        return []

    def numpy(self) -> _NumpyArrayProtocol:
        """Returns the tensor as a NumPy ndarray.

        Returns:
            A NumPy ndarray sharing or copying the tensor's data.
        """
        res = self._tensor.numpy()
        assert isinstance(res, _NumpyArrayProtocol)
        return res

    def backward(
        self,
        gradient: Tensor | None = None,
        retain_graph: bool | None = None,
        create_graph: bool = False,
    ) -> None:
        """Computes the gradient of current tensor w.r.t. graph leaves.

        Args:
            gradient: Gradient w.r.t. this tensor (required for non-scalar tensors).
            retain_graph: If False, the graph used to compute the grads will be freed.
            create_graph: If True, graph of the derivative will be constructed.
        """
        if self.grad_fn is None and not self.requires_grad:
            raise RuntimeError(
                "element 0 of tensors does not require grad and does not have a grad_fn"
            )

        from .autograd.engine import _backward

        if gradient is None:
            if self.numel() != 1:
                raise RuntimeError(
                    "grad can be implicitly created only for scalar outputs"
                )
            gradient = zero_torch.ones_like(self)

        _backward(self, gradient, retain_graph, create_graph)

    @property
    def device(self) -> str:
        """Gets the device on which the tensor is allocated.

        Returns:
            Device string (e.g. `'cpu'`).
        """
        return "cpu"

    def contiguous(self) -> Tensor:
        """Returns a contiguous in-memory tensor containing the same data.

        Returns:
            Contiguous tensor.
        """
        arr = self._tensor.data
        if isinstance(arr, _ContiguousArrayProtocol) and not arr.flags["C_CONTIGUOUS"]:
            return _wrap(arr.copy(order="C"))

        return _wrap(self._tensor.contiguous())

    def squeeze(self, dim: int | tuple[int, ...] | None = None) -> Tensor:
        """Returns a tensor with all (or specified) dimensions of size 1 removed.

        Args:
            dim: Optional dimension(s) to squeeze.

        Returns:
            Squeezed tensor.
        """
        if dim is None:
            return zero_torch.squeeze(self)
        return zero_torch.squeeze(self, dim)

    def unsqueeze(self, dim: int) -> Tensor:
        """Returns a new tensor with a dimension of size one inserted at `dim`.

        Args:
            dim: Index at which to insert the singleton dimension.

        Returns:
            Unsqueezed tensor.
        """
        if self.requires_grad:
            from .autograd.math_ops import UnsqueezeBackward

            return UnsqueezeBackward.apply(self, dim)
        return zero_torch.unsqueeze(self, dim)

    @property
    def T(self) -> Tensor:
        """Returns the tensor with its dimensions reversed.

        Returns:
            Transposed tensor.
        """
        dims = tuple(reversed(range(len(self.shape))))
        return (
            zero_torch.transpose(self, dims[0], dims[1])
            if len(dims) == 2
            else zero_torch.permute(self, dims)
        )

    def __add__(self, other: TensorLike) -> Tensor:
        """Adds `other` to this tensor element-wise.

        Args:
            other: Tensor or scalar to add.

        Returns:
            Element-wise sum tensor.
        """
        from .autograd.math_ops import AddBackward

        return AddBackward.apply(self, other)

    def __radd__(self, other: TensorLike) -> Tensor:
        """Adds this tensor to `other` element-wise (reverse addition).

        Args:
            other: Tensor or scalar to add.

        Returns:
            Element-wise sum tensor.
        """
        from .autograd.math_ops import AddBackward

        return AddBackward.apply(other, self)

    def __sub__(self, other: TensorLike) -> Tensor:
        """Subtracts `other` from this tensor element-wise.

        Args:
            other: Tensor or scalar to subtract.

        Returns:
            Element-wise difference tensor.
        """
        from .autograd.math_ops import SubBackward

        return SubBackward.apply(self, other)

    def __rsub__(self, other: TensorLike) -> Tensor:
        """Subtracts this tensor from `other` element-wise (reverse subtraction).

        Args:
            other: Tensor or scalar from which to subtract.

        Returns:
            Element-wise difference tensor.
        """
        from .autograd.math_ops import SubBackward

        return SubBackward.apply(other, self)

    def __mul__(self, other: TensorLike) -> Tensor:
        """Multiplies this tensor by `other` element-wise.

        Args:
            other: Tensor or scalar multiplier.

        Returns:
            Element-wise product tensor.
        """
        from .autograd.math_ops import MulBackward

        return MulBackward.apply(self, other)

    def __rmul__(self, other: TensorLike) -> Tensor:
        """Multiplies `other` by this tensor element-wise (reverse multiplication).

        Args:
            other: Tensor or scalar multiplier.

        Returns:
            Element-wise product tensor.
        """
        from .autograd.math_ops import MulBackward

        return MulBackward.apply(other, self)

    def __truediv__(self, other: TensorLike) -> Tensor:
        """Divides this tensor by `other` element-wise.

        Args:
            other: Tensor or scalar divisor.

        Returns:
            Element-wise quotient tensor.
        """
        from .autograd.math_ops import DivBackward

        return DivBackward.apply(self, other)

    def __rtruediv__(self, other: TensorLike) -> Tensor:
        """Divides `other` by this tensor element-wise (reverse division).

        Args:
            other: Tensor or scalar dividend.

        Returns:
            Element-wise quotient tensor.
        """
        from .autograd.math_ops import DivBackward

        return DivBackward.apply(other, self)

    def __matmul__(self, other: TensorLike) -> Tensor:
        """Performs matrix multiplication of this tensor and `other`.

        Args:
            other: Right-hand tensor operand.

        Returns:
            Matrix product tensor.
        """
        from .autograd.math_ops import MatmulBackward

        return MatmulBackward.apply(self, other)

    def __rmatmul__(self, other: TensorLike) -> Tensor:
        """Performs matrix multiplication of `other` and this tensor (reverse matmul).

        Args:
            other: Left-hand tensor operand.

        Returns:
            Matrix product tensor.
        """
        from .autograd.math_ops import MatmulBackward

        return MatmulBackward.apply(other, self)

    def __pow__(self, other: TensorLike) -> Tensor:
        """Raises this tensor to the power of `other` element-wise.

        Args:
            other: Exponent tensor or scalar.

        Returns:
            Exponentiated tensor.
        """
        from .autograd.math_ops import PowBackward

        return PowBackward.apply(self, other)

    def __neg__(self) -> Tensor:
        """Returns a new tensor with negated elements.

        Returns:
            Negated tensor.
        """
        from .autograd.math_ops import NegBackward

        return NegBackward.apply(self)

    def __lt__(self, other: TensorLike) -> Tensor:
        """Computes element-wise `self < other`.

        Args:
            other: Tensor or scalar to compare against.

        Returns:
            Boolean comparison tensor.
        """
        return _wrap(ops.less(_to_tensor(self), _to_tensor(other)))

    def __gt__(self, other: TensorLike) -> Tensor:
        """Computes element-wise `self > other`.

        Args:
            other: Tensor or scalar to compare against.

        Returns:
            Boolean comparison tensor.
        """
        return _wrap(ops.greater(_to_tensor(self), _to_tensor(other)))

    def __le__(self, other: TensorLike) -> Tensor:
        """Computes element-wise `self <= other`.

        Args:
            other: Tensor or scalar to compare against.

        Returns:
            Boolean comparison tensor.
        """
        return _wrap(ops.less_equal(_to_tensor(self), _to_tensor(other)))

    def __ge__(self, other: TensorLike) -> Tensor:
        """Computes element-wise `self >= other`.

        Args:
            other: Tensor or scalar to compare against.

        Returns:
            Boolean comparison tensor.
        """
        return _wrap(ops.greater_equal(_to_tensor(self), _to_tensor(other)))

    def __eq__(self, other: OpArg) -> Tensor:
        """Computes element-wise `self == other` or returns a False tensor for non-tensor objects.

        Args:
            other: Tensor, scalar, or value to compare against.

        Returns:
            Boolean equality tensor.
        """
        if other is None or isinstance(other, str):
            return Tensor(False, dtype="bool")
        return _wrap(ops.equal(_to_tensor(self), _to_tensor(other)))

    def __ne__(self, other: OpArg) -> Tensor:
        """Computes element-wise `self != other` or returns a True tensor for non-tensor objects.

        Args:
            other: Tensor, scalar, or value to compare against.

        Returns:
            Boolean inequality tensor.
        """
        if other is None or isinstance(other, str):
            return Tensor(True, dtype="bool")
        return _wrap(ops.not_equal(_to_tensor(self), _to_tensor(other)))

    def __len__(self) -> int:
        """Returns the size of the first dimension of the tensor.

        Returns:
            Size of dimension 0, or 0 for 0D tensors.
        """
        return self.shape[0] if len(self.shape) > 0 else 0

    def dim(self) -> int:
        """Returns the number of dimensions of the tensor.

        Returns:
            Number of dimensions (`len(self.shape)`).
        """
        return len(self.shape)

    def transpose(self, dim0: int, dim1: int) -> Tensor:
        """Returns a tensor that is a transposed version of `self` with `dim0` and `dim1` swapped.

        Args:
            dim0: First dimension to be transposed.
            dim1: Second dimension to be transposed.

        Returns:
            Transposed tensor.
        """
        return zero_torch.transpose(self, dim0, dim1)

    def chunk(self, chunks: int, dim: int = 0) -> list[Tensor]:
        """Splits a tensor into a specific number of chunks along `dim`.

        Args:
            chunks: Number of chunks to return.
            dim: Dimension along which to split the tensor.

        Returns:
            List of chunked tensors.
        """
        if chunks <= 0:
            raise ValueError(f"chunks must be > 0, got {chunks}")
        shape = self.shape
        if not shape:
            return [self]
        norm_dim = dim if dim >= 0 else len(shape) + dim
        dim_size = shape[norm_dim]
        if dim_size == 0 or dim_size % chunks == 0:
            return list(zero_torch.split(self, chunks, axis=dim))
        chunk_size = math.ceil(dim_size / chunks)
        indices = list(range(chunk_size, dim_size, chunk_size))
        return list(zero_torch.split(self, indices, axis=dim))

    def sum(
        self,
        dim: int | tuple[int, ...] | None = None,
        keepdim: bool = False,
    ) -> Tensor:
        """Returns the sum of all elements in the tensor or along `dim`.

        Args:
            dim: Optional dimension or tuple of dimensions to reduce.
            keepdim: Whether the output tensor has `dim` retained or not.

        Returns:
            Reduced sum tensor.
        """
        from zero_torch.autograd.math_ops import SumBackward

        if dim is None:
            return SumBackward.apply(self)
        return SumBackward.apply(self, axis=dim, keepdims=keepdim)

    def mean(
        self,
        dim: int | tuple[int, ...] | None = None,
        keepdim: bool = False,
    ) -> Tensor:
        """Returns the mean of all elements in the tensor or along `dim`.

        Args:
            dim: Optional dimension or tuple of dimensions to reduce.
            keepdim: Whether the output tensor has `dim` retained or not.

        Returns:
            Reduced mean tensor.
        """
        if dim is None:
            return zero_torch.mean(self)
        return zero_torch.mean(self, axis=dim, keepdims=keepdim)

    def copy_(self, src: TensorLike) -> Tensor:
        """Copies the elements from `src` into `self` tensor in-place and returns `self`.

        Args:
            src: Source tensor or scalar/sequence to copy from.

        Returns:
            The modified `self` tensor.
        """
        src_t = src if isinstance(src, Tensor) else Tensor(src, dtype=self.dtype)
        if self.shape and src_t.shape != self.shape:
            src_t = zero_torch.broadcast_to(src_t, self.shape)
        self._tensor = _to_tensor(src_t, dtype=self.dtype)
        return self

    def __getitem__(
        self,
        index: int
        | SliceIndexType
        | Tensor
        | tuple[int | SliceIndexType | Tensor | None, ...],
    ) -> Tensor:
        """Indexes or slices the tensor.

        Args:
            index: Integer, slice, tensor, or tuple of indices.

        Returns:
            Indexed or sliced tensor.
        """
        if len(self.shape) == 0:
            raise IndexError("invalid index of a 0-dim tensor")
        if not isinstance(self._tensor.data, list):
            if isinstance(index, Tensor):
                return _wrap(self._tensor[index._tensor])
            if isinstance(index, tuple):
                norm_tuple = tuple(
                    k._tensor if isinstance(k, Tensor) else k for k in index
                )
                return _wrap(self._tensor[norm_tuple])
            return _wrap(self._tensor[index])
        if isinstance(index, (int, slice)):
            return Tensor(self._tensor.data[index], dtype=self.dtype)
        return self

    def numel(self) -> int:
        """Returns the total number of elements in the tensor.

        Returns:
            Total element count.
        """
        shape = self.shape
        if not shape:
            return 1
        return functools.reduce(operator.mul, shape, 1)
