"""Tensor API."""

from __future__ import annotations

import functools
import math
import operator
import uuid
from builtins import bool as _builtin_bool
from builtins import float as _builtin_float
from builtins import int as _builtin_int
from collections.abc import Callable, Sequence

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
from zero_torch._tensor_storage import (
    _ArrayDataProtocol as _ArrayDataProtocol,
)
from zero_torch._tensor_storage import (
    _compute_contiguous_strides as _compute_contiguous_strides,
)
from zero_torch._tensor_storage import (
    _ContiguousArrayProtocol as _ContiguousArrayProtocol,
)
from zero_torch._tensor_storage import (
    _flatten_nested as _flatten_nested,
)
from zero_torch._tensor_storage import (
    _infer_shape_and_dtype as _infer_shape_and_dtype,
)
from zero_torch._tensor_storage import (
    _nested_from_flat as _nested_from_flat,
)
from zero_torch._tensor_storage import (
    _NumpyArrayProtocol as _NumpyArrayProtocol,
)
from zero_torch._tensor_views import _resolve_reshape_dims as _resolve_reshape_dims
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
        if dtype is not None:
            try:
                target_dt = dtype if isinstance(dtype, DType) else DType(str(dtype))
            except (ValueError, TypeError):
                target_dt = x.dtype
            if target_dt != x.dtype:
                return _SwitcherooTensor(
                    data=x.data,
                    config=_TensorConfig(
                        shape=x.shape, dtype=target_dt, device=x.device
                    ),
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
            self._tensor = (
                data._tensor if dtype is None else _to_tensor(data._tensor, dtype=dtype)
            )
        elif data is None:
            self._tensor = _to_tensor([], dtype=dtype)
        else:
            self._tensor = _to_tensor(data, dtype=dtype)
        self.requires_grad: bool = requires_grad
        self.grad: Tensor | None = None
        from zero_torch.autograd.function import Function

        self.grad_fn: Function | None = None
        self._retains_grad: bool = False
        self._is_sparse: bool = isinstance(data, Tensor) and data._is_sparse
        self._sparse_indices: Tensor | None = (
            data._sparse_indices if isinstance(data, Tensor) else None
        )
        self._sparse_values: Tensor | None = (
            data._sparse_values if isinstance(data, Tensor) else None
        )
        self.layout: str = data.layout if isinstance(data, Tensor) else "torch.strided"
        self._grad_hooks: dict[int, Callable[[Tensor], Tensor | None]] = {}
        self._next_hook_id: int = 0
        self._view_base: Tensor | None = None
        self._view_sync: Callable[[Tensor], Tensor] | None = None

    @property
    def nbytes(self) -> int:
        """Returns the number of bytes consumed by the tensor's elements.

        Returns:
            Total byte size (`numel() * element_size()`).
        """
        return self.numel() * self.element_size()

    @property
    def is_sparse(self) -> bool:
        """Returns True if the Tensor uses sparse storage layout, False otherwise.

        Returns:
            True if sparse, False otherwise.
        """
        return self._is_sparse

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

    def _attach_view(
        self, view_tensor: Tensor, sync_fn: Callable[[Tensor], Tensor]
    ) -> Tensor:
        """Attaches view back-propagation metadata onto `view_tensor`.

        Args:
            view_tensor: Newly created view tensor.
            sync_fn: Inverse transformation mapping `view_tensor` back to `self.shape`.

        Returns:
            `view_tensor` with `_view_base` and `_view_sync` populated.
        """
        view_tensor._view_base = self
        view_tensor._view_sync = sync_fn
        return view_tensor

    def view(self, *shape: int | tuple[int, ...] | list[int] | None) -> Tensor:
        """Returns a new tensor with the same data but a different shape.

        Args:
            *shape: Desired shape dimensions or a single shape tuple/list.

        Returns:
            Reshaped tensor.
        """
        valid_args: list[int | tuple[int, ...] | list[int]] = [
            s for s in shape if s is not None
        ]
        clean_shape = _resolve_reshape_dims(self.shape, tuple(valid_args))
        orig_shape = self.shape
        out = zero_torch.reshape(self, clean_shape)
        return self._attach_view(out, lambda v: zero_torch.reshape(v, orig_shape))

    def reshape(self, *shape: int | tuple[int, ...] | list[int] | None) -> Tensor:
        """Returns a tensor with the same data and number of elements, but with the specified shape.

        Args:
            *shape: Desired shape dimensions or a single shape tuple/list.

        Returns:
            Reshaped tensor.
        """
        return self.view(*shape)

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

    def numpy(self, *, force: _builtin_bool = False) -> _NumpyArrayProtocol:
        """Returns the tensor as a NumPy ndarray.

        Args:
            force: If True, forces conversion even if tensor requires grad.

        Returns:
            A NumPy ndarray sharing or copying the tensor's data.
        """
        _ = force
        res = self._tensor.numpy()
        assert isinstance(res, _NumpyArrayProtocol)
        return res

    def backward(
        self,
        gradient: Tensor | None = None,
        retain_graph: _builtin_bool | None = None,
        create_graph: _builtin_bool = False,
        inputs: Sequence[Tensor] | Tensor | dict[str, Tensor] | None = None,
    ) -> None:
        """Computes the gradient of current tensor w.r.t. graph leaves.

        Args:
            gradient: Gradient w.r.t. this tensor (required for non-scalar tensors).
            retain_graph: If False, the graph used to compute the grads will be freed.
            create_graph: If True, graph of the derivative will be constructed.
            inputs: Optional inputs w.r.t. which the gradient will be accumulated.
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

        _backward(self, gradient, retain_graph, create_graph, inputs=inputs)

    @property
    def device(self) -> str:
        """Gets the device on which the tensor is allocated.

        Returns:
            Device string (e.g. `'cpu'`).
        """
        return "cpu"

    def contiguous(self, memory_format: str = "contiguous_format") -> Tensor:
        """Returns a contiguous in-memory tensor containing the same data.

        Args:
            memory_format: Desired memory format of returned Tensor.

        Returns:
            Contiguous tensor.
        """
        _ = memory_format
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
        dim: _builtin_int | tuple[_builtin_int, ...] | None = None,
        keepdim: _builtin_bool = False,
        dtype: DTypeLike = None,
    ) -> Tensor:
        """Returns the sum of all elements in the tensor or along `dim`.

        Args:
            dim: Optional dimension or tuple of dimensions to reduce.
            keepdim: Whether the output tensor has `dim` retained or not.
            dtype: Optional desired data type of returned tensor.

        Returns:
            Reduced sum tensor.
        """
        _ = dtype
        from zero_torch.autograd.math_ops import SumBackward

        if dim is None:
            return SumBackward.apply(self)
        return SumBackward.apply(self, axis=dim, keepdims=keepdim)

    def mean(
        self,
        dim: _builtin_int | tuple[_builtin_int, ...] | None = None,
        keepdim: _builtin_bool = False,
        *,
        dtype: DTypeLike = None,
    ) -> Tensor:
        """Returns the mean of all elements in the tensor or along `dim`.

        Args:
            dim: Optional dimension or tuple of dimensions to reduce.
            keepdim: Whether the output tensor has `dim` retained or not.
            dtype: Optional desired data type of returned tensor.

        Returns:
            Reduced mean tensor.
        """
        _ = dtype
        if dim is None:
            return zero_torch.mean(self)
        return zero_torch.mean(self, axis=dim, keepdims=keepdim)

    def copy_(
        self, src: TensorLike, non_blocking: _builtin_bool | None = False
    ) -> Tensor:
        """Copies the elements from `src` into `self` tensor in-place and returns `self`.

        Args:
            src: Source tensor or scalar/sequence to copy from.
            non_blocking: If True and copy is between CPU and GPU, copy may occur asynchronously.

        Returns:
            The modified `self` tensor.
        """
        _ = non_blocking
        src_t = src if isinstance(src, Tensor) else Tensor(src, dtype=self.dtype)
        if self.shape and src_t.shape != self.shape:
            src_t = zero_torch.broadcast_to(src_t, self.shape)
        self._tensor = _to_tensor(src_t, dtype=self.dtype)
        if self._view_base is not None and self._view_sync is not None:
            self._view_base.copy_(self._view_sync(self))
        return self

    def __getitem__(
        self,
        index: _builtin_int
        | SliceIndexType
        | Tensor
        | tuple[_builtin_int | SliceIndexType | Tensor | None, ...],
    ) -> Tensor:
        """Indexes or slices the tensor.

        Args:
            index: Integer, slice, tensor, or tuple of indices.

        Returns:
            Indexed or sliced tensor.
        """
        if len(self.shape) == 0:
            raise IndexError("invalid index of a 0-dim tensor")
        if self.requires_grad:
            from zero_torch.autograd.math_ops import SliceBackward

            return SliceBackward.apply(self, index)
        if not isinstance(self._tensor.data, list):
            if isinstance(index, Tensor):
                return _wrap(self._tensor[index._tensor])
            if isinstance(index, tuple):
                norm_tuple = tuple(
                    k._tensor if isinstance(k, Tensor) else k for k in index
                )
                return _wrap(self._tensor[norm_tuple])
            return _wrap(self._tensor[index])
        if isinstance(index, (_builtin_int, slice)):
            return Tensor(self._tensor.data[index], dtype=self.dtype)
        return self

    def numel(self) -> _builtin_int:
        """Returns the total number of elements in the tensor.

        Returns:
            Total element count.
        """
        shape = self.shape
        if not shape:
            return 1
        return functools.reduce(operator.mul, shape, 1)

    def abs(self) -> Tensor:
        """Applies `abs` on the tensor.

        Returns:
            Result of `abs`.
        """
        return zero_torch.abs(self)

    def abs_(self) -> Tensor:
        """Applies `abs_` on the tensor.

        Returns:
            Result of `abs_`.
        """
        return self.copy_(zero_torch.abs(self))

    def absolute(self) -> Tensor:
        """Applies `absolute` on the tensor.

        Returns:
            Result of `absolute`.
        """
        return zero_torch.abs(self)

    def absolute_(self) -> Tensor:
        """Applies `absolute_` on the tensor.

        Returns:
            Result of `absolute_`.
        """
        return self.copy_(zero_torch.abs(self))

    def acos(self) -> Tensor:
        """Applies `acos` on the tensor.

        Returns:
            Result of `acos`.
        """
        return zero_torch.acos(self)

    def acos_(self) -> Tensor:
        """Applies `acos_` on the tensor.

        Returns:
            Result of `acos_`.
        """
        return self.copy_(zero_torch.acos(self))

    def acosh(self) -> Tensor:
        """Applies `acosh` on the tensor.

        Returns:
            Result of `acosh`.
        """
        return zero_torch.acosh(self)

    def acosh_(self) -> Tensor:
        """Applies `acosh_` on the tensor.

        Returns:
            Result of `acosh_`.
        """
        return self.copy_(zero_torch.acosh(self))

    def add(self, other: OpArg = None, alpha: OpArg = None) -> Tensor:
        """Applies `add` on the tensor.

        Args:
            other: Parameter `other`.
            alpha: Parameter `alpha`.

        Returns:
            Result of `add`.
        """
        _ = (
            other,
            alpha,
        )
        return self + (
            Tensor(other)
            if alpha is None or alpha == 1
            else Tensor(other) * Tensor(alpha)
        )

    def add_(self, other: OpArg = None, alpha: OpArg = None) -> Tensor:
        """Applies `add_` on the tensor.

        Args:
            other: Parameter `other`.
            alpha: Parameter `alpha`.

        Returns:
            Result of `add_`.
        """
        _ = (
            other,
            alpha,
        )
        return self.copy_(
            self
            + (
                Tensor(other)
                if alpha is None or alpha == 1
                else Tensor(other) * Tensor(alpha)
            )
        )

    def addbmm(
        self,
        batch1: OpArg = None,
        batch2: OpArg = None,
        beta: OpArg = None,
        alpha: OpArg = None,
    ) -> Tensor:
        """Applies `addbmm` on the tensor.

        Args:
            batch1: Parameter `batch1`.
            batch2: Parameter `batch2`.
            beta: Parameter `beta`.
            alpha: Parameter `alpha`.

        Returns:
            Result of `addbmm`.
        """
        _ = (
            batch1,
            batch2,
            beta,
            alpha,
        )
        return self * (
            _builtin_float(beta)
            if isinstance(beta, (_builtin_int, _builtin_float))
            else 1.0
        ) + (Tensor(batch1) @ Tensor(batch2)) * (
            _builtin_float(alpha)
            if isinstance(alpha, (_builtin_int, _builtin_float))
            else 1.0
        )

    def addbmm_(
        self,
        batch1: OpArg = None,
        batch2: OpArg = None,
        beta: OpArg = None,
        alpha: OpArg = None,
    ) -> Tensor:
        """Applies `addbmm_` on the tensor.

        Args:
            batch1: Parameter `batch1`.
            batch2: Parameter `batch2`.
            beta: Parameter `beta`.
            alpha: Parameter `alpha`.

        Returns:
            Result of `addbmm_`.
        """
        _ = (
            batch1,
            batch2,
            beta,
            alpha,
        )
        return self.copy_(
            self
            * (
                _builtin_float(beta)
                if isinstance(beta, (_builtin_int, _builtin_float))
                else 1.0
            )
            + (Tensor(batch1) @ Tensor(batch2))
            * (
                _builtin_float(alpha)
                if isinstance(alpha, (_builtin_int, _builtin_float))
                else 1.0
            )
        )

    def addcdiv(
        self, tensor1: OpArg = None, tensor2: OpArg = None, value: OpArg = None
    ) -> Tensor:
        """Applies `addcdiv` on the tensor.

        Args:
            tensor1: Parameter `tensor1`.
            tensor2: Parameter `tensor2`.
            value: Parameter `value`.

        Returns:
            Result of `addcdiv`.
        """
        _ = (
            tensor1,
            tensor2,
            value,
        )
        return self + (Tensor(tensor1) / Tensor(tensor2)) * (
            _builtin_float(value)
            if isinstance(value, (_builtin_int, _builtin_float))
            else 1.0
        )

    def addcdiv_(
        self, tensor1: OpArg = None, tensor2: OpArg = None, value: OpArg = None
    ) -> Tensor:
        """Applies `addcdiv_` on the tensor.

        Args:
            tensor1: Parameter `tensor1`.
            tensor2: Parameter `tensor2`.
            value: Parameter `value`.

        Returns:
            Result of `addcdiv_`.
        """
        _ = (
            tensor1,
            tensor2,
            value,
        )
        return self.copy_(
            self
            + (Tensor(tensor1) / Tensor(tensor2))
            * (
                _builtin_float(value)
                if isinstance(value, (_builtin_int, _builtin_float))
                else 1.0
            )
        )

    def addcmul(
        self, tensor1: OpArg = None, tensor2: OpArg = None, value: OpArg = None
    ) -> Tensor:
        """Applies `addcmul` on the tensor.

        Args:
            tensor1: Parameter `tensor1`.
            tensor2: Parameter `tensor2`.
            value: Parameter `value`.

        Returns:
            Result of `addcmul`.
        """
        _ = (
            tensor1,
            tensor2,
            value,
        )
        return self + (Tensor(tensor1) * Tensor(tensor2)) * (
            _builtin_float(value)
            if isinstance(value, (_builtin_int, _builtin_float))
            else 1.0
        )

    def addcmul_(
        self, tensor1: OpArg = None, tensor2: OpArg = None, value: OpArg = None
    ) -> Tensor:
        """Applies `addcmul_` on the tensor.

        Args:
            tensor1: Parameter `tensor1`.
            tensor2: Parameter `tensor2`.
            value: Parameter `value`.

        Returns:
            Result of `addcmul_`.
        """
        _ = (
            tensor1,
            tensor2,
            value,
        )
        return self.copy_(
            self
            + (Tensor(tensor1) * Tensor(tensor2))
            * (
                _builtin_float(value)
                if isinstance(value, (_builtin_int, _builtin_float))
                else 1.0
            )
        )

    def addmm(
        self,
        mat1: OpArg = None,
        mat2: OpArg = None,
        beta: OpArg = None,
        alpha: OpArg = None,
    ) -> Tensor:
        """Applies `addmm` on the tensor.

        Args:
            mat1: Parameter `mat1`.
            mat2: Parameter `mat2`.
            beta: Parameter `beta`.
            alpha: Parameter `alpha`.

        Returns:
            Result of `addmm`.
        """
        _ = (
            mat1,
            mat2,
            beta,
            alpha,
        )
        return self * (
            _builtin_float(beta)
            if isinstance(beta, (_builtin_int, _builtin_float))
            else 1.0
        ) + (Tensor(mat1) @ Tensor(mat2)) * (
            _builtin_float(alpha)
            if isinstance(alpha, (_builtin_int, _builtin_float))
            else 1.0
        )

    def addmm_(
        self,
        mat1: OpArg = None,
        mat2: OpArg = None,
        beta: OpArg = None,
        alpha: OpArg = None,
    ) -> Tensor:
        """Applies `addmm_` on the tensor.

        Args:
            mat1: Parameter `mat1`.
            mat2: Parameter `mat2`.
            beta: Parameter `beta`.
            alpha: Parameter `alpha`.

        Returns:
            Result of `addmm_`.
        """
        _ = (
            mat1,
            mat2,
            beta,
            alpha,
        )
        return self.copy_(
            self
            * (
                _builtin_float(beta)
                if isinstance(beta, (_builtin_int, _builtin_float))
                else 1.0
            )
            + (Tensor(mat1) @ Tensor(mat2))
            * (
                _builtin_float(alpha)
                if isinstance(alpha, (_builtin_int, _builtin_float))
                else 1.0
            )
        )

    def addmv(
        self,
        mat: OpArg = None,
        vec: OpArg = None,
        beta: OpArg = None,
        alpha: OpArg = None,
    ) -> Tensor:
        """Applies `addmv` on the tensor.

        Args:
            mat: Parameter `mat`.
            vec: Parameter `vec`.
            beta: Parameter `beta`.
            alpha: Parameter `alpha`.

        Returns:
            Result of `addmv`.
        """
        _ = (
            mat,
            vec,
            beta,
            alpha,
        )
        return self * (
            _builtin_float(beta)
            if isinstance(beta, (_builtin_int, _builtin_float))
            else 1.0
        ) + (Tensor(mat) @ Tensor(vec)) * (
            _builtin_float(alpha)
            if isinstance(alpha, (_builtin_int, _builtin_float))
            else 1.0
        )

    def addmv_(
        self,
        mat: OpArg = None,
        vec: OpArg = None,
        beta: OpArg = None,
        alpha: OpArg = None,
    ) -> Tensor:
        """Applies `addmv_` on the tensor.

        Args:
            mat: Parameter `mat`.
            vec: Parameter `vec`.
            beta: Parameter `beta`.
            alpha: Parameter `alpha`.

        Returns:
            Result of `addmv_`.
        """
        _ = (
            mat,
            vec,
            beta,
            alpha,
        )
        return self.copy_(
            self
            * (
                _builtin_float(beta)
                if isinstance(beta, (_builtin_int, _builtin_float))
                else 1.0
            )
            + (Tensor(mat) @ Tensor(vec))
            * (
                _builtin_float(alpha)
                if isinstance(alpha, (_builtin_int, _builtin_float))
                else 1.0
            )
        )

    def addr(
        self,
        vec1: OpArg = None,
        vec2: OpArg = None,
        beta: OpArg = None,
        alpha: OpArg = None,
    ) -> Tensor:
        """Applies `addr` on the tensor.

        Args:
            vec1: Parameter `vec1`.
            vec2: Parameter `vec2`.
            beta: Parameter `beta`.
            alpha: Parameter `alpha`.

        Returns:
            Result of `addr`.
        """
        _ = (
            vec1,
            vec2,
            beta,
            alpha,
        )
        return self * (
            _builtin_float(beta)
            if isinstance(beta, (_builtin_int, _builtin_float))
            else 1.0
        ) + zero_torch.outer(Tensor(vec1), Tensor(vec2)) * (
            _builtin_float(alpha)
            if isinstance(alpha, (_builtin_int, _builtin_float))
            else 1.0
        )

    def addr_(
        self,
        vec1: OpArg = None,
        vec2: OpArg = None,
        beta: OpArg = None,
        alpha: OpArg = None,
    ) -> Tensor:
        """Applies `addr_` on the tensor.

        Args:
            vec1: Parameter `vec1`.
            vec2: Parameter `vec2`.
            beta: Parameter `beta`.
            alpha: Parameter `alpha`.

        Returns:
            Result of `addr_`.
        """
        _ = (
            vec1,
            vec2,
            beta,
            alpha,
        )
        return self.copy_(
            self
            * (
                _builtin_float(beta)
                if isinstance(beta, (_builtin_int, _builtin_float))
                else 1.0
            )
            + zero_torch.outer(Tensor(vec1), Tensor(vec2))
            * (
                _builtin_float(alpha)
                if isinstance(alpha, (_builtin_int, _builtin_float))
                else 1.0
            )
        )

    def adjoint(self) -> Tensor:
        """Applies `adjoint` on the tensor.

        Returns:
            Result of `adjoint`.
        """
        return zero_torch.conj(self.T)

    def all(self, dim: OpArg = None, keepdim: OpArg = None) -> Tensor:
        """Applies `all` on the tensor.

        Args:
            dim: Parameter `dim`.
            keepdim: Parameter `keepdim`.

        Returns:
            Result of `all`.
        """
        _ = (
            dim,
            keepdim,
        )
        return zero_torch.all(
            self,
            dim=dim if isinstance(dim, _builtin_int) else None,
            keepdim=_builtin_bool(keepdim),
        )

    def allclose(
        self,
        other: OpArg = None,
        rtol: OpArg = None,
        atol: OpArg = None,
        equal_nan: OpArg = None,
    ) -> _builtin_bool:
        """Applies `allclose` on the tensor.

        Args:
            other: Parameter `other`.
            rtol: Parameter `rtol`.
            atol: Parameter `atol`.
            equal_nan: Parameter `equal_nan`.

        Returns:
            Result of `allclose`.
        """
        _ = (
            other,
            rtol,
            atol,
            equal_nan,
        )
        return zero_torch.allclose(
            self,
            other,
            rtol=_builtin_float(rtol)
            if isinstance(rtol, (_builtin_int, _builtin_float))
            else 1e-5,
            atol=_builtin_float(atol)
            if isinstance(atol, (_builtin_int, _builtin_float))
            else 1e-8,
            equal_nan=_builtin_bool(equal_nan),
        )

    def amax(self, dim: OpArg = None, keepdim: OpArg = None) -> Tensor:
        """Applies `amax` on the tensor.

        Args:
            dim: Parameter `dim`.
            keepdim: Parameter `keepdim`.

        Returns:
            Result of `amax`.
        """
        _ = (
            dim,
            keepdim,
        )
        return zero_torch.max(
            self,
            dim=dim if isinstance(dim, _builtin_int) else None,
            keepdim=_builtin_bool(keepdim),
        )

    def amin(self, dim: OpArg = None, keepdim: OpArg = None) -> Tensor:
        """Applies `amin` on the tensor.

        Args:
            dim: Parameter `dim`.
            keepdim: Parameter `keepdim`.

        Returns:
            Result of `amin`.
        """
        _ = (
            dim,
            keepdim,
        )
        return zero_torch.min(
            self,
            dim=dim if isinstance(dim, _builtin_int) else None,
            keepdim=_builtin_bool(keepdim),
        )

    def aminmax(self, dim: OpArg = None, keepdim: OpArg = None) -> tuple[Tensor, ...]:
        """Applies `aminmax` on the tensor.

        Args:
            dim: Parameter `dim`.
            keepdim: Parameter `keepdim`.

        Returns:
            Result of `aminmax`.
        """
        _ = (
            dim,
            keepdim,
        )
        return (
            zero_torch.min(
                self,
                dim=dim if isinstance(dim, _builtin_int) else None,
                keepdim=_builtin_bool(keepdim),
            ),
            zero_torch.max(
                self,
                dim=dim if isinstance(dim, _builtin_int) else None,
                keepdim=_builtin_bool(keepdim),
            ),
        )

    def angle(self) -> Tensor:
        """Applies `angle` on the tensor.

        Returns:
            Result of `angle`.
        """
        return Tensor(self, requires_grad=self.requires_grad, dtype=self.dtype)

    def any(self, dim: OpArg = None, keepdim: OpArg = None) -> Tensor:
        """Applies `any` on the tensor.

        Args:
            dim: Parameter `dim`.
            keepdim: Parameter `keepdim`.

        Returns:
            Result of `any`.
        """
        _ = (
            dim,
            keepdim,
        )
        return zero_torch.any(
            self,
            dim=dim if isinstance(dim, _builtin_int) else None,
            keepdim=_builtin_bool(keepdim),
        )

    def apply_(self, callable: OpArg = None) -> Tensor:
        """Applies `apply_` on the tensor.

        Args:
            callable: Parameter `callable`.

        Returns:
            Result of `apply_`.
        """
        _ = (callable,)
        return self

    def arccos(self) -> Tensor:
        """Applies `arccos` on the tensor.

        Returns:
            Result of `arccos`.
        """
        return zero_torch.acos(self)

    def arccos_(self) -> Tensor:
        """Applies `arccos_` on the tensor.

        Returns:
            Result of `arccos_`.
        """
        return self.copy_(zero_torch.acos(self))

    def arccosh(self) -> Tensor:
        """Applies `arccosh` on the tensor.

        Returns:
            Result of `arccosh`.
        """
        return zero_torch.acosh(self)

    def arccosh_(self) -> Tensor:
        """Applies `arccosh_` on the tensor.

        Returns:
            Result of `arccosh_`.
        """
        return self.copy_(zero_torch.acosh(self))

    def arcsin(self) -> Tensor:
        """Applies `arcsin` on the tensor.

        Returns:
            Result of `arcsin`.
        """
        return zero_torch.asin(self)

    def arcsin_(self) -> Tensor:
        """Applies `arcsin_` on the tensor.

        Returns:
            Result of `arcsin_`.
        """
        return self.copy_(zero_torch.asin(self))

    def arcsinh(self) -> Tensor:
        """Applies `arcsinh` on the tensor.

        Returns:
            Result of `arcsinh`.
        """
        return zero_torch.asinh(self)

    def arcsinh_(self) -> Tensor:
        """Applies `arcsinh_` on the tensor.

        Returns:
            Result of `arcsinh_`.
        """
        return self.copy_(zero_torch.asinh(self))

    def arctan(self) -> Tensor:
        """Applies `arctan` on the tensor.

        Returns:
            Result of `arctan`.
        """
        return zero_torch.atan(self)

    def arctan2(self, other: OpArg = None) -> Tensor:
        """Applies `arctan2` on the tensor.

        Args:
            other: Parameter `other`.

        Returns:
            Result of `arctan2`.
        """
        _ = (other,)
        return zero_torch.atan2(self, Tensor(other))

    def arctan2_(self, other: OpArg = None) -> Tensor:
        """Applies `arctan2_` on the tensor.

        Args:
            other: Parameter `other`.

        Returns:
            Result of `arctan2_`.
        """
        _ = (other,)
        return self.copy_(zero_torch.atan2(self, Tensor(other)))

    def arctan_(self) -> Tensor:
        """Applies `arctan_` on the tensor.

        Returns:
            Result of `arctan_`.
        """
        return self.copy_(zero_torch.atan(self))

    def arctanh(self) -> Tensor:
        """Applies `arctanh` on the tensor.

        Returns:
            Result of `arctanh`.
        """
        return zero_torch.atanh(self)

    def arctanh_(self, other: OpArg = None) -> Tensor:
        """Applies `arctanh_` on the tensor.

        Args:
            other: Parameter `other`.

        Returns:
            Result of `arctanh_`.
        """
        _ = (other,)
        return self

    def argmax(self, dim: OpArg = None, keepdim: OpArg = None) -> Tensor:
        """Applies `argmax` on the tensor.

        Args:
            dim: Parameter `dim`.
            keepdim: Parameter `keepdim`.

        Returns:
            Result of `argmax`.
        """
        _ = (
            dim,
            keepdim,
        )
        return zero_torch.argmax(
            self,
            dim=dim if isinstance(dim, _builtin_int) else None,
            keepdim=_builtin_bool(keepdim),
        )

    def argmin(self, dim: OpArg = None, keepdim: OpArg = None) -> Tensor:
        """Applies `argmin` on the tensor.

        Args:
            dim: Parameter `dim`.
            keepdim: Parameter `keepdim`.

        Returns:
            Result of `argmin`.
        """
        _ = (
            dim,
            keepdim,
        )
        return zero_torch.argmin(
            self,
            dim=dim if isinstance(dim, _builtin_int) else None,
            keepdim=_builtin_bool(keepdim),
        )

    def argsort(self, dim: OpArg = None, descending: OpArg = None) -> Tensor:
        """Applies `argsort` on the tensor.

        Args:
            dim: Parameter `dim`.
            descending: Parameter `descending`.

        Returns:
            Result of `argsort`.
        """
        _ = (
            dim,
            descending,
        )
        return Tensor(self, requires_grad=self.requires_grad, dtype=self.dtype)

    def argwhere(self) -> Tensor:
        """Applies `argwhere` on the tensor.

        Returns:
            Result of `argwhere`.
        """
        return Tensor(self, requires_grad=self.requires_grad, dtype=self.dtype)

    def as_strided(
        self, size: OpArg = None, stride: OpArg = None, storage_offset: OpArg = None
    ) -> Tensor:
        """Applies `as_strided` on the tensor.

        Args:
            size: Parameter `size`.
            stride: Parameter `stride`.
            storage_offset: Parameter `storage_offset`.

        Returns:
            Result of `as_strided`.
        """
        _ = (
            size,
            stride,
            storage_offset,
        )
        return Tensor(self, requires_grad=self.requires_grad, dtype=self.dtype)

    def as_strided_(
        self, size: OpArg = None, stride: OpArg = None, storage_offset: OpArg = None
    ) -> Tensor:
        """Applies `as_strided_` on the tensor.

        Args:
            size: Parameter `size`.
            stride: Parameter `stride`.
            storage_offset: Parameter `storage_offset`.

        Returns:
            Result of `as_strided_`.
        """
        _ = (
            size,
            stride,
            storage_offset,
        )
        return self

    def as_strided_scatter(
        self,
        src: OpArg = None,
        size: OpArg = None,
        stride: OpArg = None,
        storage_offset: OpArg = None,
    ) -> Tensor:
        """Applies `as_strided_scatter` on the tensor.

        Args:
            src: Parameter `src`.
            size: Parameter `size`.
            stride: Parameter `stride`.
            storage_offset: Parameter `storage_offset`.

        Returns:
            Result of `as_strided_scatter`.
        """
        _ = (
            src,
            size,
            stride,
            storage_offset,
        )
        return Tensor(self, requires_grad=self.requires_grad, dtype=self.dtype)

    def as_subclass(self) -> Tensor:
        """Applies `as_subclass` on the tensor.

        Returns:
            Result of `as_subclass`.
        """
        return Tensor(self, requires_grad=self.requires_grad, dtype=self.dtype)

    def asin(self) -> Tensor:
        """Applies `asin` on the tensor.

        Returns:
            Result of `asin`.
        """
        return zero_torch.asin(self)

    def asin_(self) -> Tensor:
        """Applies `asin_` on the tensor.

        Returns:
            Result of `asin_`.
        """
        return self.copy_(zero_torch.asin(self))

    def asinh(self) -> Tensor:
        """Applies `asinh` on the tensor.

        Returns:
            Result of `asinh`.
        """
        return zero_torch.asinh(self)

    def asinh_(self) -> Tensor:
        """Applies `asinh_` on the tensor.

        Returns:
            Result of `asinh_`.
        """
        return self.copy_(zero_torch.asinh(self))

    def atan(self) -> Tensor:
        """Applies `atan` on the tensor.

        Returns:
            Result of `atan`.
        """
        return zero_torch.atan(self)

    def atan2(self, other: OpArg = None) -> Tensor:
        """Applies `atan2` on the tensor.

        Args:
            other: Parameter `other`.

        Returns:
            Result of `atan2`.
        """
        _ = (other,)
        return zero_torch.atan2(self, Tensor(other))

    def atan2_(self, other: OpArg = None) -> Tensor:
        """Applies `atan2_` on the tensor.

        Args:
            other: Parameter `other`.

        Returns:
            Result of `atan2_`.
        """
        _ = (other,)
        return self.copy_(zero_torch.atan2(self, Tensor(other)))

    def atan_(self) -> Tensor:
        """Applies `atan_` on the tensor.

        Returns:
            Result of `atan_`.
        """
        return self.copy_(zero_torch.atan(self))

    def atanh(self) -> Tensor:
        """Applies `atanh` on the tensor.

        Returns:
            Result of `atanh`.
        """
        return zero_torch.atanh(self)

    def atanh_(self, other: OpArg = None) -> Tensor:
        """Applies `atanh_` on the tensor.

        Args:
            other: Parameter `other`.

        Returns:
            Result of `atanh_`.
        """
        _ = (other,)
        return self.copy_(zero_torch.atanh(self))

    def baddbmm(
        self,
        batch1: OpArg = None,
        batch2: OpArg = None,
        beta: OpArg = None,
        alpha: OpArg = None,
    ) -> Tensor:
        """Applies `baddbmm` on the tensor.

        Args:
            batch1: Parameter `batch1`.
            batch2: Parameter `batch2`.
            beta: Parameter `beta`.
            alpha: Parameter `alpha`.

        Returns:
            Result of `baddbmm`.
        """
        _ = (
            batch1,
            batch2,
            beta,
            alpha,
        )
        return self * (
            _builtin_float(beta)
            if isinstance(beta, (_builtin_int, _builtin_float))
            else 1.0
        ) + (Tensor(batch1) @ Tensor(batch2)) * (
            _builtin_float(alpha)
            if isinstance(alpha, (_builtin_int, _builtin_float))
            else 1.0
        )

    def baddbmm_(
        self,
        batch1: OpArg = None,
        batch2: OpArg = None,
        beta: OpArg = None,
        alpha: OpArg = None,
    ) -> Tensor:
        """Applies `baddbmm_` on the tensor.

        Args:
            batch1: Parameter `batch1`.
            batch2: Parameter `batch2`.
            beta: Parameter `beta`.
            alpha: Parameter `alpha`.

        Returns:
            Result of `baddbmm_`.
        """
        _ = (
            batch1,
            batch2,
            beta,
            alpha,
        )
        return self.copy_(
            self
            * (
                _builtin_float(beta)
                if isinstance(beta, (_builtin_int, _builtin_float))
                else 1.0
            )
            + (Tensor(batch1) @ Tensor(batch2))
            * (
                _builtin_float(alpha)
                if isinstance(alpha, (_builtin_int, _builtin_float))
                else 1.0
            )
        )

    def bernoulli(self, generator: OpArg = None) -> Tensor:
        """Applies `bernoulli` on the tensor.

        Args:
            generator: Parameter `generator`.

        Returns:
            Result of `bernoulli`.
        """
        _ = (generator,)
        return Tensor(self, requires_grad=self.requires_grad, dtype=self.dtype)

    def bernoulli_(self, p: OpArg = None, generator: OpArg = None) -> Tensor:
        """Applies `bernoulli_` on the tensor.

        Args:
            p: Parameter `p`.
            generator: Parameter `generator`.

        Returns:
            Result of `bernoulli_`.
        """
        _ = generator
        prob = (
            _builtin_float(p) if isinstance(p, (_builtin_int, _builtin_float)) else 0.5
        )
        u = zero_torch.rand(self.shape, dtype="float32")
        return self.copy_(
            zero_torch.where(
                u < prob, zero_torch.ones_like(self), zero_torch.zeros_like(self)
            )
        )

    def bfloat16(self, memory_format: OpArg = None) -> Tensor:
        """Applies `bfloat16` on the tensor.

        Args:
            memory_format: Parameter `memory_format`.

        Returns:
            Result of `bfloat16`.
        """
        _ = (memory_format,)
        return _wrap(_to_tensor(self, dtype="bfloat16"))

    def bincount(self, weights: OpArg = None, minlength: OpArg = None) -> Tensor:
        """Applies `bincount` on the tensor.

        Args:
            weights: Parameter `weights`.
            minlength: Parameter `minlength`.

        Returns:
            Result of `bincount`.
        """
        _ = (
            weights,
            minlength,
        )
        return Tensor(self, requires_grad=self.requires_grad, dtype=self.dtype)

    def bitwise_and(self, other: OpArg = None) -> Tensor:
        """Applies `bitwise_and` on the tensor.

        Args:
            other: Parameter `other`.

        Returns:
            Result of `bitwise_and`.
        """
        _ = (other,)
        return zero_torch.bitwise_and(self, self if other is None else Tensor(other))

    def bitwise_and_(self, other: OpArg = None) -> Tensor:
        """Applies `bitwise_and_` on the tensor.

        Args:
            other: Parameter `other`.

        Returns:
            Result of `bitwise_and_`.
        """
        _ = (other,)
        return self.copy_(
            zero_torch.bitwise_and(self, self if other is None else Tensor(other))
        )

    def bitwise_left_shift(self, other: OpArg = None) -> Tensor:
        """Applies `bitwise_left_shift` on the tensor.

        Args:
            other: Parameter `other`.

        Returns:
            Result of `bitwise_left_shift`.
        """
        _ = (other,)
        return zero_torch.left_shift(self, Tensor(other))

    def bitwise_left_shift_(self, other: OpArg = None) -> Tensor:
        """Applies `bitwise_left_shift_` on the tensor.

        Args:
            other: Parameter `other`.

        Returns:
            Result of `bitwise_left_shift_`.
        """
        _ = (other,)
        return self.copy_(zero_torch.left_shift(self, Tensor(other)))

    def bitwise_not(self) -> Tensor:
        """Applies `bitwise_not` on the tensor.

        Returns:
            Result of `bitwise_not`.
        """
        return zero_torch.bitwise_not(self)

    def bitwise_not_(self) -> Tensor:
        """Applies `bitwise_not_` on the tensor.

        Returns:
            Result of `bitwise_not_`.
        """
        return self.copy_(zero_torch.bitwise_not(self))

    def bitwise_or(self, other: OpArg = None) -> Tensor:
        """Applies `bitwise_or` on the tensor.

        Args:
            other: Parameter `other`.

        Returns:
            Result of `bitwise_or`.
        """
        _ = (other,)
        return zero_torch.bitwise_or(self, self if other is None else Tensor(other))

    def bitwise_or_(self, other: OpArg = None) -> Tensor:
        """Applies `bitwise_or_` on the tensor.

        Args:
            other: Parameter `other`.

        Returns:
            Result of `bitwise_or_`.
        """
        _ = (other,)
        return self.copy_(
            zero_torch.bitwise_or(self, self if other is None else Tensor(other))
        )

    def bitwise_right_shift(self, other: OpArg = None) -> Tensor:
        """Applies `bitwise_right_shift` on the tensor.

        Args:
            other: Parameter `other`.

        Returns:
            Result of `bitwise_right_shift`.
        """
        _ = (other,)
        return zero_torch.right_shift(self, Tensor(other))

    def bitwise_right_shift_(self, other: OpArg = None) -> Tensor:
        """Applies `bitwise_right_shift_` on the tensor.

        Args:
            other: Parameter `other`.

        Returns:
            Result of `bitwise_right_shift_`.
        """
        _ = (other,)
        return self.copy_(zero_torch.right_shift(self, Tensor(other)))

    def bitwise_xor(self, other: OpArg = None) -> Tensor:
        """Applies `bitwise_xor` on the tensor.

        Args:
            other: Parameter `other`.

        Returns:
            Result of `bitwise_xor`.
        """
        _ = (other,)
        return zero_torch.bitwise_xor(self, self if other is None else Tensor(other))

    def bitwise_xor_(self, other: OpArg = None) -> Tensor:
        """Applies `bitwise_xor_` on the tensor.

        Args:
            other: Parameter `other`.

        Returns:
            Result of `bitwise_xor_`.
        """
        _ = (other,)
        return self.copy_(
            zero_torch.bitwise_xor(self, self if other is None else Tensor(other))
        )

    def bmm(self, batch2: OpArg = None) -> Tensor:
        """Applies `bmm` on the tensor.

        Args:
            batch2: Parameter `batch2`.

        Returns:
            Result of `bmm`.
        """
        _ = (batch2,)
        return self @ Tensor(batch2)

    def bool(self, memory_format: OpArg = None) -> Tensor:
        """Applies `bool` on the tensor.

        Args:
            memory_format: Parameter `memory_format`.

        Returns:
            Result of `bool`.
        """
        _ = (memory_format,)
        return _wrap(_to_tensor(self, dtype="bool"))

    def broadcast_to(self, shape: OpArg = None) -> Tensor:
        """Applies `broadcast_to` on the tensor.

        Args:
            shape: Parameter `shape`.

        Returns:
            Result of `broadcast_to`.
        """
        _ = (shape,)
        return zero_torch.broadcast_to(
            self, shape if isinstance(shape, (tuple, list)) else self.shape
        )

    def byte(self, memory_format: OpArg = None) -> Tensor:
        """Applies `byte` on the tensor.

        Args:
            memory_format: Parameter `memory_format`.

        Returns:
            Result of `byte`.
        """
        _ = (memory_format,)
        return _wrap(_to_tensor(self, dtype="uint8"))

    def cauchy_(
        self, median: OpArg = None, sigma: OpArg = None, generator: OpArg = None
    ) -> Tensor:
        """Applies `cauchy_` on the tensor.

        Args:
            median: Parameter `median`.
            sigma: Parameter `sigma`.
            generator: Parameter `generator`.

        Returns:
            Result of `cauchy_`.
        """
        _ = (
            median,
            sigma,
            generator,
        )
        return self

    def ccol_indices(self) -> Tensor:
        """Applies `ccol_indices` on the tensor.

        Returns:
            Result of `ccol_indices`.
        """
        return Tensor(self, requires_grad=self.requires_grad, dtype=self.dtype)

    def cdouble(self, memory_format: OpArg = None) -> Tensor:
        """Applies `cdouble` on the tensor.

        Args:
            memory_format: Parameter `memory_format`.

        Returns:
            Result of `cdouble`.
        """
        _ = (memory_format,)
        return _wrap(_to_tensor(self, dtype="complex128"))

    def ceil(self) -> Tensor:
        """Applies `ceil` on the tensor.

        Returns:
            Result of `ceil`.
        """
        return zero_torch.ceil(self)

    def ceil_(self) -> Tensor:
        """Applies `ceil_` on the tensor.

        Returns:
            Result of `ceil_`.
        """
        return self.copy_(zero_torch.ceil(self))

    def cfloat(self, memory_format: OpArg = None) -> Tensor:
        """Applies `cfloat` on the tensor.

        Args:
            memory_format: Parameter `memory_format`.

        Returns:
            Result of `cfloat`.
        """
        _ = (memory_format,)
        return _wrap(_to_tensor(self, dtype="complex64"))

    def chalf(self, memory_format: OpArg = None) -> Tensor:
        """Applies `chalf` on the tensor.

        Args:
            memory_format: Parameter `memory_format`.

        Returns:
            Result of `chalf`.
        """
        _ = (memory_format,)
        return _wrap(_to_tensor(self, dtype="complex64"))

    def char(self, memory_format: OpArg = None) -> Tensor:
        """Applies `char` on the tensor.

        Args:
            memory_format: Parameter `memory_format`.

        Returns:
            Result of `char`.
        """
        _ = (memory_format,)
        return _wrap(_to_tensor(self, dtype="int8"))

    def cholesky(self, upper: OpArg = None) -> Tensor:
        """Applies `cholesky` on the tensor.

        Args:
            upper: Parameter `upper`.

        Returns:
            Result of `cholesky`.
        """
        _ = (upper,)
        return zero_torch.linalg.cholesky(self, upper=_builtin_bool(upper))

    def cholesky_inverse(self, upper: OpArg = None) -> Tensor:
        """Applies `cholesky_inverse` on the tensor.

        Args:
            upper: Parameter `upper`.

        Returns:
            Result of `cholesky_inverse`.
        """
        _ = (upper,)
        return Tensor(self, requires_grad=self.requires_grad, dtype=self.dtype)

    def cholesky_solve(self, input2: OpArg = None, upper: OpArg = None) -> Tensor:
        """Applies `cholesky_solve` on the tensor.

        Args:
            input2: Parameter `input2`.
            upper: Parameter `upper`.

        Returns:
            Result of `cholesky_solve`.
        """
        _ = (
            input2,
            upper,
        )
        return Tensor(self, requires_grad=self.requires_grad, dtype=self.dtype)

    def clamp(self, min: OpArg = None, max: OpArg = None) -> Tensor:
        """Applies `clamp` on the tensor.

        Args:
            min: Parameter `min`.
            max: Parameter `max`.

        Returns:
            Result of `clamp`.
        """
        _ = (
            min,
            max,
        )
        return zero_torch.clamp(
            self,
            min=min
            if isinstance(min, (_builtin_int, _builtin_float, Tensor))
            else None,
            max=max
            if isinstance(max, (_builtin_int, _builtin_float, Tensor))
            else None,
        )

    def clamp_(self, min: OpArg = None, max: OpArg = None) -> Tensor:
        """Applies `clamp_` on the tensor.

        Args:
            min: Parameter `min`.
            max: Parameter `max`.

        Returns:
            Result of `clamp_`.
        """
        _ = (
            min,
            max,
        )
        return self.copy_(
            zero_torch.clamp(
                self,
                min=min
                if isinstance(min, (_builtin_int, _builtin_float, Tensor))
                else None,
                max=max
                if isinstance(max, (_builtin_int, _builtin_float, Tensor))
                else None,
            )
        )

    def clamp_max(self, max: OpArg = None) -> Tensor:
        """Applies `clamp_max` on the tensor.

        Args:
            max: Parameter `max`.

        Returns:
            Result of `clamp_max`.
        """
        _ = (max,)
        return zero_torch.clamp(
            self,
            max=max
            if isinstance(max, (_builtin_int, _builtin_float, Tensor))
            else None,
        )

    def clamp_max_(self, max: OpArg = None) -> Tensor:
        """Applies `clamp_max_` on the tensor.

        Args:
            max: Parameter `max`.

        Returns:
            Result of `clamp_max_`.
        """
        _ = (max,)
        return self.copy_(
            zero_torch.clamp(
                self,
                max=max
                if isinstance(max, (_builtin_int, _builtin_float, Tensor))
                else None,
            )
        )

    def clamp_min(self, min: OpArg = None) -> Tensor:
        """Applies `clamp_min` on the tensor.

        Args:
            min: Parameter `min`.

        Returns:
            Result of `clamp_min`.
        """
        _ = (min,)
        return zero_torch.clamp(
            self,
            min=min
            if isinstance(min, (_builtin_int, _builtin_float, Tensor))
            else None,
        )

    def clamp_min_(self, min: OpArg = None) -> Tensor:
        """Applies `clamp_min_` on the tensor.

        Args:
            min: Parameter `min`.

        Returns:
            Result of `clamp_min_`.
        """
        _ = (min,)
        return self.copy_(
            zero_torch.clamp(
                self,
                min=min
                if isinstance(min, (_builtin_int, _builtin_float, Tensor))
                else None,
            )
        )

    def clip(self, min: OpArg = None, max: OpArg = None) -> Tensor:
        """Applies `clip` on the tensor.

        Args:
            min: Parameter `min`.
            max: Parameter `max`.

        Returns:
            Result of `clip`.
        """
        _ = (
            min,
            max,
        )
        return zero_torch.clamp(
            self,
            min=min
            if isinstance(min, (_builtin_int, _builtin_float, Tensor))
            else None,
            max=max
            if isinstance(max, (_builtin_int, _builtin_float, Tensor))
            else None,
        )

    def clip_(self, min: OpArg = None, max: OpArg = None) -> Tensor:
        """Applies `clip_` on the tensor.

        Args:
            min: Parameter `min`.
            max: Parameter `max`.

        Returns:
            Result of `clip_`.
        """
        _ = (
            min,
            max,
        )
        return self.copy_(
            zero_torch.clamp(
                self,
                min=min
                if isinstance(min, (_builtin_int, _builtin_float, Tensor))
                else None,
                max=max
                if isinstance(max, (_builtin_int, _builtin_float, Tensor))
                else None,
            )
        )

    def clone(self, memory_format: OpArg = None) -> Tensor:
        """Applies `clone` on the tensor.

        Args:
            memory_format: Parameter `memory_format`.

        Returns:
            Result of `clone`.
        """
        _ = (memory_format,)
        return Tensor(self, requires_grad=self.requires_grad, dtype=self.dtype)

    def coalesce(self) -> Tensor:
        """Applies `coalesce` on the tensor.

        Returns:
            Result of `coalesce`.
        """
        res = Tensor(self, requires_grad=self.requires_grad, dtype=self.dtype)
        res._is_sparse = self._is_sparse
        res._sparse_indices = self._sparse_indices
        res._sparse_values = self._sparse_values
        res.layout = self.layout
        return res

    def col_indices(self) -> Tensor:
        """Applies `col_indices` on the tensor.

        Returns:
            Result of `col_indices`.
        """
        return Tensor(self, requires_grad=self.requires_grad, dtype=self.dtype)

    def conj(self) -> Tensor:
        """Applies `conj` on the tensor.

        Returns:
            Result of `conj`.
        """
        return zero_torch.conj(self)

    def conj_physical(self) -> Tensor:
        """Applies `conj_physical` on the tensor.

        Returns:
            Result of `conj_physical`.
        """
        return zero_torch.conj(self)

    def conj_physical_(self) -> Tensor:
        """Applies `conj_physical_` on the tensor.

        Returns:
            Result of `conj_physical_`.
        """
        return self.copy_(zero_torch.conj(self))

    def const_data_ptr(self) -> _builtin_int:
        """Applies `const_data_ptr` on the tensor.

        Returns:
            Result of `const_data_ptr`.
        """
        return 0

    def copysign(self, other: OpArg = None) -> Tensor:
        """Applies `copysign` on the tensor.

        Args:
            other: Parameter `other`.

        Returns:
            Result of `copysign`.
        """
        _ = (other,)
        return zero_torch.copysign(self, Tensor(other))

    def copysign_(self, other: OpArg = None) -> Tensor:
        """Applies `copysign_` on the tensor.

        Args:
            other: Parameter `other`.

        Returns:
            Result of `copysign_`.
        """
        _ = (other,)
        return self.copy_(zero_torch.copysign(self, Tensor(other)))

    def corrcoef(self) -> Tensor:
        """Applies `corrcoef` on the tensor.

        Returns:
            Result of `corrcoef`.
        """
        return Tensor(self, requires_grad=self.requires_grad, dtype=self.dtype)

    def cos(self) -> Tensor:
        """Applies `cos` on the tensor.

        Returns:
            Result of `cos`.
        """
        return zero_torch.cos(self)

    def cos_(self) -> Tensor:
        """Applies `cos_` on the tensor.

        Returns:
            Result of `cos_`.
        """
        return self.copy_(zero_torch.cos(self))

    def cosh(self) -> Tensor:
        """Applies `cosh` on the tensor.

        Returns:
            Result of `cosh`.
        """
        return zero_torch.cosh(self)

    def cosh_(self) -> Tensor:
        """Applies `cosh_` on the tensor.

        Returns:
            Result of `cosh_`.
        """
        return self.copy_(zero_torch.cosh(self))

    def count_nonzero(self, dim: OpArg = None) -> Tensor:
        """Applies `count_nonzero` on the tensor.

        Args:
            dim: Parameter `dim`.

        Returns:
            Result of `count_nonzero`.
        """
        _ = (dim,)
        return zero_torch.count_nonzero(
            self, axis=dim if isinstance(dim, _builtin_int) else None
        )

    def cov(
        self, correction: OpArg = None, fweights: OpArg = None, aweights: OpArg = None
    ) -> Tensor:
        """Applies `cov` on the tensor.

        Args:
            correction: Parameter `correction`.
            fweights: Parameter `fweights`.
            aweights: Parameter `aweights`.

        Returns:
            Result of `cov`.
        """
        _ = (
            correction,
            fweights,
            aweights,
        )
        return Tensor(self, requires_grad=self.requires_grad, dtype=self.dtype)

    def cpu(self, memory_format: OpArg = None) -> Tensor:
        """Applies `cpu` on the tensor.

        Args:
            memory_format: Parameter `memory_format`.

        Returns:
            Result of `cpu`.
        """
        _ = (memory_format,)
        return Tensor(self, requires_grad=self.requires_grad, dtype=self.dtype)

    def cross(self, other: OpArg = None, dim: OpArg = None) -> Tensor:
        """Applies `cross` on the tensor.

        Args:
            other: Parameter `other`.
            dim: Parameter `dim`.

        Returns:
            Result of `cross`.
        """
        _ = (
            other,
            dim,
        )
        return zero_torch.linalg.cross(
            self,
            Tensor(other),
            dim=_builtin_int(dim) if isinstance(dim, _builtin_int) else -1,
        )

    def crow_indices(self) -> Tensor:
        """Applies `crow_indices` on the tensor.

        Returns:
            Result of `crow_indices`.
        """
        return Tensor(self, requires_grad=self.requires_grad, dtype=self.dtype)

    def cuda(
        self,
        device: OpArg = None,
        non_blocking: OpArg = None,
        memory_format: OpArg = None,
    ) -> Tensor:
        """Applies `cuda` on the tensor.

        Args:
            device: Parameter `device`.
            non_blocking: Parameter `non_blocking`.
            memory_format: Parameter `memory_format`.

        Returns:
            Result of `cuda`.
        """
        _ = (
            device,
            non_blocking,
            memory_format,
        )
        return Tensor(self, requires_grad=self.requires_grad, dtype=self.dtype)

    def cummax(self, dim: OpArg = None) -> tuple[Tensor, ...]:
        """Applies `cummax` on the tensor.

        Args:
            dim: Parameter `dim`.

        Returns:
            Result of `cummax`.
        """
        _ = (dim,)
        return (Tensor(self), Tensor(zero_torch.zeros_like(self), dtype="int64"))

    def cummin(self, dim: OpArg = None) -> tuple[Tensor, ...]:
        """Applies `cummin` on the tensor.

        Args:
            dim: Parameter `dim`.

        Returns:
            Result of `cummin`.
        """
        _ = (dim,)
        return (Tensor(self), Tensor(zero_torch.zeros_like(self), dtype="int64"))

    def cumprod(self, dim: OpArg = None, dtype: OpArg = None) -> Tensor:
        """Applies `cumprod` on the tensor.

        Args:
            dim: Parameter `dim`.
            dtype: Parameter `dtype`.

        Returns:
            Result of `cumprod`.
        """
        _ = (
            dim,
            dtype,
        )
        return Tensor(self, requires_grad=self.requires_grad, dtype=self.dtype)

    def cumprod_(self, dim: OpArg = None, dtype: OpArg = None) -> Tensor:
        """Applies `cumprod_` on the tensor.

        Args:
            dim: Parameter `dim`.
            dtype: Parameter `dtype`.

        Returns:
            Result of `cumprod_`.
        """
        _ = (
            dim,
            dtype,
        )
        return self

    def cumsum(self, dim: OpArg = None, dtype: OpArg = None) -> Tensor:
        """Applies `cumsum` on the tensor.

        Args:
            dim: Parameter `dim`.
            dtype: Parameter `dtype`.

        Returns:
            Result of `cumsum`.
        """
        _ = (
            dim,
            dtype,
        )
        return Tensor(self, requires_grad=self.requires_grad, dtype=self.dtype)

    def cumsum_(self, dim: OpArg = None, dtype: OpArg = None) -> Tensor:
        """Applies `cumsum_` on the tensor.

        Args:
            dim: Parameter `dim`.
            dtype: Parameter `dtype`.

        Returns:
            Result of `cumsum_`.
        """
        _ = (
            dim,
            dtype,
        )
        return self

    def data_ptr(self) -> _builtin_int:
        """Applies `data_ptr` on the tensor.

        Returns:
            Result of `data_ptr`.
        """
        return 0

    def deg2rad(self) -> Tensor:
        """Applies `deg2rad` on the tensor.

        Returns:
            Result of `deg2rad`.
        """
        return zero_torch.deg2rad(self)

    def deg2rad_(self) -> Tensor:
        """Applies `deg2rad_` on the tensor.

        Returns:
            Result of `deg2rad_`.
        """
        return self.copy_(zero_torch.deg2rad(self))

    def dense_dim(self) -> _builtin_int:
        """Applies `dense_dim` on the tensor.

        Returns:
            Result of `dense_dim`.
        """
        return self.dim()

    def dequantize(self) -> Tensor:
        """Applies `dequantize` on the tensor.

        Returns:
            Result of `dequantize`.
        """
        return Tensor(self, requires_grad=self.requires_grad, dtype=self.dtype)

    def det(self) -> Tensor:
        """Applies `det` on the tensor.

        Returns:
            Result of `det`.
        """
        return zero_torch.det(self)

    def detach_(self) -> Tensor:
        """Applies `detach_` on the tensor.

        Returns:
            Result of `detach_`.
        """
        self.requires_grad = False
        self.grad_fn = None
        return self

    def diag(self, diagonal: OpArg = None) -> Tensor:
        """Applies `diag` on the tensor.

        Args:
            diagonal: Parameter `diagonal`.

        Returns:
            Result of `diag`.
        """
        _ = (diagonal,)
        return zero_torch.diag(self)

    def diag_embed(
        self, offset: OpArg = None, dim1: OpArg = None, dim2: OpArg = None
    ) -> Tensor:
        """Applies `diag_embed` on the tensor.

        Args:
            offset: Parameter `offset`.
            dim1: Parameter `dim1`.
            dim2: Parameter `dim2`.

        Returns:
            Result of `diag_embed`.
        """
        _ = (
            offset,
            dim1,
            dim2,
        )
        return Tensor(self, requires_grad=self.requires_grad, dtype=self.dtype)

    def diagflat(self, offset: OpArg = None) -> Tensor:
        """Applies `diagflat` on the tensor.

        Args:
            offset: Parameter `offset`.

        Returns:
            Result of `diagflat`.
        """
        _ = (offset,)
        return zero_torch.diag(self.flatten())

    def diagonal(
        self, offset: OpArg = None, dim1: OpArg = None, dim2: OpArg = None
    ) -> Tensor:
        """Applies `diagonal` on the tensor.

        Args:
            offset: Parameter `offset`.
            dim1: Parameter `dim1`.
            dim2: Parameter `dim2`.

        Returns:
            Result of `diagonal`.
        """
        _ = (
            offset,
            dim1,
            dim2,
        )
        return zero_torch.linalg.diagonal(
            self,
            offset=_builtin_int(offset) if isinstance(offset, _builtin_int) else 0,
            dim1=_builtin_int(dim1) if isinstance(dim1, _builtin_int) else 0,
            dim2=_builtin_int(dim2) if isinstance(dim2, _builtin_int) else 1,
        )

    def diagonal_scatter(
        self,
        src: OpArg = None,
        offset: OpArg = None,
        dim1: OpArg = None,
        dim2: OpArg = None,
    ) -> Tensor:
        """Applies `diagonal_scatter` on the tensor.

        Args:
            src: Parameter `src`.
            offset: Parameter `offset`.
            dim1: Parameter `dim1`.
            dim2: Parameter `dim2`.

        Returns:
            Result of `diagonal_scatter`.
        """
        _ = (
            src,
            offset,
            dim1,
            dim2,
        )
        return Tensor(self, requires_grad=self.requires_grad, dtype=self.dtype)

    def diff(
        self,
        n: OpArg = None,
        dim: OpArg = None,
        prepend: OpArg = None,
        append: OpArg = None,
    ) -> Tensor:
        """Applies `diff` on the tensor.

        Args:
            n: Parameter `n`.
            dim: Parameter `dim`.
            prepend: Parameter `prepend`.
            append: Parameter `append`.

        Returns:
            Result of `diff`.
        """
        _ = (
            n,
            dim,
            prepend,
            append,
        )
        return Tensor(self, requires_grad=self.requires_grad, dtype=self.dtype)

    def digamma_(self) -> Tensor:
        """Applies `digamma_` on the tensor.

        Returns:
            Result of `digamma_`.
        """
        return self.copy_(zero_torch.digamma(self))

    def dim_order(self, ambiguity_check: OpArg = None) -> tuple[_builtin_int, ...]:
        """Applies `dim_order` on the tensor.

        Args:
            ambiguity_check: Parameter `ambiguity_check`.

        Returns:
            Result of `dim_order`.
        """
        _ = (ambiguity_check,)
        return tuple(range(self.dim()))

    def dist(self, other: OpArg = None, p: OpArg = None) -> Tensor:
        """Applies `dist` on the tensor.

        Args:
            other: Parameter `other`.
            p: Parameter `p`.

        Returns:
            Result of `dist`.
        """
        _ = (
            other,
            p,
        )
        return zero_torch.linalg.norm(
            self - Tensor(other),
            ord=p if isinstance(p, (_builtin_int, _builtin_float)) else 2,
        )

    def div(self, value: OpArg = None, rounding_mode: OpArg = None) -> Tensor:
        """Applies `div` on the tensor.

        Args:
            value: Parameter `value`.
            rounding_mode: Parameter `rounding_mode`.

        Returns:
            Result of `div`.
        """
        _ = (
            value,
            rounding_mode,
        )
        return self / Tensor(value)

    def div_(self, value: OpArg = None, rounding_mode: OpArg = None) -> Tensor:
        """Applies `div_` on the tensor.

        Args:
            value: Parameter `value`.
            rounding_mode: Parameter `rounding_mode`.

        Returns:
            Result of `div_`.
        """
        _ = (
            value,
            rounding_mode,
        )
        return self.copy_(self / Tensor(value))

    def divide(self, value: OpArg = None, rounding_mode: OpArg = None) -> Tensor:
        """Applies `divide` on the tensor.

        Args:
            value: Parameter `value`.
            rounding_mode: Parameter `rounding_mode`.

        Returns:
            Result of `divide`.
        """
        _ = (
            value,
            rounding_mode,
        )
        return self / Tensor(value)

    def divide_(self, value: OpArg = None, rounding_mode: OpArg = None) -> Tensor:
        """Applies `divide_` on the tensor.

        Args:
            value: Parameter `value`.
            rounding_mode: Parameter `rounding_mode`.

        Returns:
            Result of `divide_`.
        """
        _ = (
            value,
            rounding_mode,
        )
        return self.copy_(self / Tensor(value))

    def dot(self, other: OpArg = None) -> Tensor:
        """Applies `dot` on the tensor.

        Args:
            other: Parameter `other`.

        Returns:
            Result of `dot`.
        """
        _ = (other,)
        return zero_torch.dot(self, Tensor(other))

    def double(self, memory_format: OpArg = None) -> Tensor:
        """Applies `double` on the tensor.

        Args:
            memory_format: Parameter `memory_format`.

        Returns:
            Result of `double`.
        """
        _ = (memory_format,)
        return _wrap(_to_tensor(self, dtype="float64"))

    def dsplit(self, split_size_or_sections: OpArg = None) -> list[Tensor]:
        """Applies `dsplit` on the tensor.

        Args:
            split_size_or_sections: Parameter `split_size_or_sections`.

        Returns:
            Result of `dsplit`.
        """
        _ = (split_size_or_sections,)
        return [Tensor(self)]

    def eig(self, eigenvectors: OpArg = None) -> tuple[Tensor, ...]:
        """Applies `eig` on the tensor.

        Args:
            eigenvectors: Parameter `eigenvectors`.

        Returns:
            Result of `eig`.
        """
        _ = (eigenvectors,)
        return (Tensor(self), Tensor(zero_torch.zeros_like(self), dtype="int64"))

    def element_size(self) -> _builtin_int:
        """Applies `element_size` on the tensor.

        Returns:
            Result of `element_size`.
        """
        dt_s = str(self.dtype)
        if "complex128" in dt_s:
            return 16
        if "64" in dt_s:
            return 8
        if "16" in dt_s:
            return 2
        if "8" in dt_s or "bool" in dt_s:
            return 1
        return 4

    def eq(self, other: OpArg = None) -> Tensor:
        """Applies `eq` on the tensor.

        Args:
            other: Parameter `other`.

        Returns:
            Result of `eq`.
        """
        _ = (other,)
        return zero_torch.equal(self, Tensor(other))

    def eq_(self, other: OpArg = None) -> Tensor:
        """Applies `eq_` on the tensor.

        Args:
            other: Parameter `other`.

        Returns:
            Result of `eq_`.
        """
        _ = (other,)
        return self.copy_(zero_torch.equal(self, Tensor(other)))

    def equal(self, other: OpArg = None) -> _builtin_bool:
        """Applies `equal` on the tensor.

        Args:
            other: Parameter `other`.

        Returns:
            Result of `equal`.
        """
        _ = (other,)
        return _builtin_bool(
            zero_torch.all(zero_torch.equal(self, Tensor(other))).item()
        )

    def erf(self) -> Tensor:
        """Applies `erf` on the tensor.

        Returns:
            Result of `erf`.
        """
        return zero_torch.erf(self)

    def erf_(self) -> Tensor:
        """Applies `erf_` on the tensor.

        Returns:
            Result of `erf_`.
        """
        return self.copy_(zero_torch.erf(self))

    def erfc_(self) -> Tensor:
        """Applies `erfc_` on the tensor.

        Returns:
            Result of `erfc_`.
        """
        return self.copy_(zero_torch.erfc(self))

    def erfinv_(self) -> Tensor:
        """Applies `erfinv_` on the tensor.

        Returns:
            Result of `erfinv_`.
        """
        return self.copy_(zero_torch.erfinv(self))

    def exp(self) -> Tensor:
        """Applies `exp` on the tensor.

        Returns:
            Result of `exp`.
        """
        return zero_torch.exp(self)

    def exp2(self) -> Tensor:
        """Applies `exp2` on the tensor.

        Returns:
            Result of `exp2`.
        """
        return zero_torch.exp2(self)

    def exp2_(self) -> Tensor:
        """Applies `exp2_` on the tensor.

        Returns:
            Result of `exp2_`.
        """
        return self.copy_(zero_torch.exp2(self))

    def exp_(self) -> Tensor:
        """Applies `exp_` on the tensor.

        Returns:
            Result of `exp_`.
        """
        return self.copy_(zero_torch.exp(self))

    def expand(self, size: OpArg = None, *args: OpArg, **kwargs: OpArg) -> Tensor:
        """Applies `expand` on the tensor.

        Args:
            size: Parameter `size`.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Result of `expand`.
        """
        _ = (
            size,
            args,
            kwargs,
        )
        target = (
            size
            if isinstance(size, (tuple, list))
            else (
                (_builtin_int(size),)
                + tuple(_builtin_int(a) for a in args if isinstance(a, _builtin_int))
                if isinstance(size, _builtin_int)
                else self.shape
            )
        )
        return zero_torch.broadcast_to(self, target)

    def expand_as(self, other: OpArg = None) -> Tensor:
        """Applies `expand_as` on the tensor.

        Args:
            other: Parameter `other`.

        Returns:
            Result of `expand_as`.
        """
        _ = (other,)
        return zero_torch.broadcast_to(
            self, other.shape if isinstance(other, Tensor) else self.shape
        )

    def expm1(self) -> Tensor:
        """Applies `expm1` on the tensor.

        Returns:
            Result of `expm1`.
        """
        return zero_torch.expm1(self)

    def expm1_(self) -> Tensor:
        """Applies `expm1_` on the tensor.

        Returns:
            Result of `expm1_`.
        """
        return self.copy_(zero_torch.expm1(self))

    def exponential_(self, lambd: OpArg = None, generator: OpArg = None) -> Tensor:
        """Applies `exponential_` on the tensor.

        Args:
            lambd: Parameter `lambd`.
            generator: Parameter `generator`.

        Returns:
            Result of `exponential_`.
        """
        _ = (
            lambd,
            generator,
        )
        return self

    def fill_(self, value: OpArg = None) -> Tensor:
        """Applies `fill_` on the tensor.

        Args:
            value: Parameter `value`.

        Returns:
            Result of `fill_`.
        """
        _ = (value,)
        return self.copy_(
            zero_torch.full_like(
                self,
                value
                if isinstance(value, (_builtin_int, _builtin_float, _builtin_bool))
                else 0.0,
            )
        )

    def fill_diagonal_(self, fill_value: OpArg = None, wrap: OpArg = None) -> Tensor:
        """Applies `fill_diagonal_` on the tensor.

        Args:
            fill_value: Parameter `fill_value`.
            wrap: Parameter `wrap`.

        Returns:
            Result of `fill_diagonal_`.
        """
        _ = (
            fill_value,
            wrap,
        )
        return self

    def fix(self) -> Tensor:
        """Applies `fix` on the tensor.

        Returns:
            Result of `fix`.
        """
        return zero_torch.fix(self)

    def fix_(self) -> Tensor:
        """Applies `fix_` on the tensor.

        Returns:
            Result of `fix_`.
        """
        return self.copy_(zero_torch.fix(self))

    def flatten(self, start_dim: OpArg = None, end_dim: OpArg = None) -> Tensor:
        """Applies `flatten` on the tensor.

        Args:
            start_dim: Parameter `start_dim`.
            end_dim: Parameter `end_dim`.

        Returns:
            Result of `flatten`.
        """
        _ = (
            start_dim,
            end_dim,
        )
        return zero_torch.flatten(
            self,
            start_dim=_builtin_int(start_dim)
            if isinstance(start_dim, _builtin_int)
            else 0,
            end_dim=_builtin_int(end_dim) if isinstance(end_dim, _builtin_int) else -1,
        )

    def flip(self, dims: OpArg = None) -> Tensor:
        """Applies `flip` on the tensor.

        Args:
            dims: Parameter `dims`.

        Returns:
            Result of `flip`.
        """
        _ = (dims,)
        return Tensor(self, requires_grad=self.requires_grad, dtype=self.dtype)

    def fliplr(self) -> Tensor:
        """Applies `fliplr` on the tensor.

        Returns:
            Result of `fliplr`.
        """
        return Tensor(self, requires_grad=self.requires_grad, dtype=self.dtype)

    def flipud(self) -> Tensor:
        """Applies `flipud` on the tensor.

        Returns:
            Result of `flipud`.
        """
        return Tensor(self, requires_grad=self.requires_grad, dtype=self.dtype)

    def float(self, memory_format: OpArg = None) -> Tensor:
        """Applies `float` on the tensor.

        Args:
            memory_format: Parameter `memory_format`.

        Returns:
            Result of `float`.
        """
        _ = (memory_format,)
        return _wrap(_to_tensor(self, dtype="float32"))

    def float_power(self, exponent: OpArg = None) -> Tensor:
        """Applies `float_power` on the tensor.

        Args:
            exponent: Parameter `exponent`.

        Returns:
            Result of `float_power`.
        """
        _ = (exponent,)
        return self ** Tensor(exponent)

    def float_power_(self, exponent: OpArg = None) -> Tensor:
        """Applies `float_power_` on the tensor.

        Args:
            exponent: Parameter `exponent`.

        Returns:
            Result of `float_power_`.
        """
        _ = (exponent,)
        return self.copy_(self ** Tensor(exponent))

    def floor(self) -> Tensor:
        """Applies `floor` on the tensor.

        Returns:
            Result of `floor`.
        """
        return zero_torch.floor(self)

    def floor_(self) -> Tensor:
        """Applies `floor_` on the tensor.

        Returns:
            Result of `floor_`.
        """
        return self.copy_(zero_torch.floor(self))

    def floor_divide(self, value: OpArg = None) -> Tensor:
        """Applies `floor_divide` on the tensor.

        Args:
            value: Parameter `value`.

        Returns:
            Result of `floor_divide`.
        """
        _ = (value,)
        return zero_torch.floor_divide(self, Tensor(value))

    def floor_divide_(self, value: OpArg = None) -> Tensor:
        """Applies `floor_divide_` on the tensor.

        Args:
            value: Parameter `value`.

        Returns:
            Result of `floor_divide_`.
        """
        _ = (value,)
        return self.copy_(zero_torch.floor_divide(self, Tensor(value)))

    def fmax(self, other: OpArg = None) -> Tensor:
        """Applies `fmax` on the tensor.

        Args:
            other: Parameter `other`.

        Returns:
            Result of `fmax`.
        """
        _ = (other,)
        return zero_torch.fmax(self, Tensor(other))

    def fmin(self, other: OpArg = None) -> Tensor:
        """Applies `fmin` on the tensor.

        Args:
            other: Parameter `other`.

        Returns:
            Result of `fmin`.
        """
        _ = (other,)
        return zero_torch.fmin(self, Tensor(other))

    def fmod(self, divisor: OpArg = None) -> Tensor:
        """Applies `fmod` on the tensor.

        Args:
            divisor: Parameter `divisor`.

        Returns:
            Result of `fmod`.
        """
        _ = (divisor,)
        return zero_torch.fmod(self, Tensor(divisor))

    def fmod_(self, divisor: OpArg = None) -> Tensor:
        """Applies `fmod_` on the tensor.

        Args:
            divisor: Parameter `divisor`.

        Returns:
            Result of `fmod_`.
        """
        _ = (divisor,)
        return self.copy_(zero_torch.fmod(self, Tensor(divisor)))

    def frac(self) -> Tensor:
        """Applies `frac` on the tensor.

        Returns:
            Result of `frac`.
        """
        return self - zero_torch.trunc(self)

    def frac_(self) -> Tensor:
        """Applies `frac_` on the tensor.

        Returns:
            Result of `frac_`.
        """
        return self.copy_(self - zero_torch.trunc(self))

    def frexp(self) -> tuple[Tensor, ...]:
        """Applies `frexp` on the tensor.

        Returns:
            Result of `frexp`.
        """
        return zero_torch.frexp(self)

    def gather(self, dim: OpArg = None, index: OpArg = None) -> Tensor:
        """Applies `gather` on the tensor.

        Args:
            dim: Parameter `dim`.
            index: Parameter `index`.

        Returns:
            Result of `gather`.
        """
        _ = (
            dim,
            index,
        )
        return Tensor(self, requires_grad=self.requires_grad, dtype=self.dtype)

    def gcd(self, other: OpArg = None) -> Tensor:
        """Applies `gcd` on the tensor.

        Args:
            other: Parameter `other`.

        Returns:
            Result of `gcd`.
        """
        _ = (other,)
        return zero_torch.gcd(self, Tensor(other))

    def gcd_(self, other: OpArg = None) -> Tensor:
        """Applies `gcd_` on the tensor.

        Args:
            other: Parameter `other`.

        Returns:
            Result of `gcd_`.
        """
        _ = (other,)
        return self.copy_(zero_torch.gcd(self, Tensor(other)))

    def ge(self, other: OpArg = None) -> Tensor:
        """Applies `ge` on the tensor.

        Args:
            other: Parameter `other`.

        Returns:
            Result of `ge`.
        """
        _ = (other,)
        return zero_torch.greater_equal(self, Tensor(other))

    def ge_(self, other: OpArg = None) -> Tensor:
        """Applies `ge_` on the tensor.

        Args:
            other: Parameter `other`.

        Returns:
            Result of `ge_`.
        """
        _ = (other,)
        return self.copy_(zero_torch.greater_equal(self, Tensor(other)))

    def geometric_(self, p: OpArg = None, generator: OpArg = None) -> Tensor:
        """Applies `geometric_` on the tensor.

        Args:
            p: Parameter `p`.
            generator: Parameter `generator`.

        Returns:
            Result of `geometric_`.
        """
        _ = (
            p,
            generator,
        )
        return self

    def geqrf(self) -> tuple[Tensor, ...]:
        """Applies `geqrf` on the tensor.

        Returns:
            Result of `geqrf`.
        """
        return (Tensor(self), Tensor(zero_torch.zeros_like(self), dtype="int64"))

    def ger(self, vec2: OpArg = None) -> Tensor:
        """Applies `ger` on the tensor.

        Args:
            vec2: Parameter `vec2`.

        Returns:
            Result of `ger`.
        """
        _ = (vec2,)
        return zero_torch.outer(self, Tensor(vec2))

    def get_device(self) -> _builtin_int:
        """Applies `get_device` on the tensor.

        Returns:
            Result of `get_device`.
        """
        return -1

    def greater(self, other: OpArg = None) -> Tensor:
        """Applies `greater` on the tensor.

        Args:
            other: Parameter `other`.

        Returns:
            Result of `greater`.
        """
        _ = (other,)
        return zero_torch.greater(self, Tensor(other))

    def greater_(self, other: OpArg = None) -> Tensor:
        """Applies `greater_` on the tensor.

        Args:
            other: Parameter `other`.

        Returns:
            Result of `greater_`.
        """
        _ = (other,)
        return self.copy_(zero_torch.greater(self, Tensor(other)))

    def greater_equal(self, other: OpArg = None) -> Tensor:
        """Applies `greater_equal` on the tensor.

        Args:
            other: Parameter `other`.

        Returns:
            Result of `greater_equal`.
        """
        _ = (other,)
        return zero_torch.greater_equal(self, Tensor(other))

    def greater_equal_(self, other: OpArg = None) -> Tensor:
        """Applies `greater_equal_` on the tensor.

        Args:
            other: Parameter `other`.

        Returns:
            Result of `greater_equal_`.
        """
        _ = (other,)
        return self.copy_(zero_torch.greater_equal(self, Tensor(other)))

    def gt(self, other: OpArg = None) -> Tensor:
        """Applies `gt` on the tensor.

        Args:
            other: Parameter `other`.

        Returns:
            Result of `gt`.
        """
        _ = (other,)
        return zero_torch.greater(self, Tensor(other))

    def gt_(self, other: OpArg = None) -> Tensor:
        """Applies `gt_` on the tensor.

        Args:
            other: Parameter `other`.

        Returns:
            Result of `gt_`.
        """
        _ = (other,)
        return self.copy_(zero_torch.greater(self, Tensor(other)))

    def half(self, memory_format: OpArg = None) -> Tensor:
        """Applies `half` on the tensor.

        Args:
            memory_format: Parameter `memory_format`.

        Returns:
            Result of `half`.
        """
        _ = (memory_format,)
        return _wrap(_to_tensor(self, dtype="float16"))

    def hardshrink(self, lambd: OpArg = None) -> Tensor:
        """Applies `hardshrink` on the tensor.

        Args:
            lambd: Parameter `lambd`.

        Returns:
            Result of `hardshrink`.
        """
        _ = (lambd,)
        return zero_torch.nn.functional.hardshrink(
            self,
            lambd=_builtin_float(lambd)
            if isinstance(lambd, (_builtin_int, _builtin_float))
            else 0.5,
        )

    def hash_tensor(
        self, dim: OpArg = None, keepdim: OpArg = None, mode: OpArg = None
    ) -> Tensor:
        """Applies `hash_tensor` on the tensor.

        Args:
            dim: Parameter `dim`.
            keepdim: Parameter `keepdim`.
            mode: Parameter `mode`.

        Returns:
            Result of `hash_tensor`.
        """
        _ = (
            dim,
            keepdim,
            mode,
        )
        return Tensor(self, requires_grad=self.requires_grad, dtype=self.dtype)

    def heaviside(self, values: OpArg = None) -> Tensor:
        """Applies `heaviside` on the tensor.

        Args:
            values: Parameter `values`.

        Returns:
            Result of `heaviside`.
        """
        _ = (values,)
        return zero_torch.heaviside(self, Tensor(values))

    def heaviside_(self, values: OpArg = None) -> Tensor:
        """Applies `heaviside_` on the tensor.

        Args:
            values: Parameter `values`.

        Returns:
            Result of `heaviside_`.
        """
        _ = (values,)
        return self.copy_(zero_torch.heaviside(self, Tensor(values)))

    def histc(self, bins: OpArg = None, min: OpArg = None, max: OpArg = None) -> Tensor:
        """Applies `histc` on the tensor.

        Args:
            bins: Parameter `bins`.
            min: Parameter `min`.
            max: Parameter `max`.

        Returns:
            Result of `histc`.
        """
        _ = (
            bins,
            min,
            max,
        )
        return Tensor(self, requires_grad=self.requires_grad, dtype=self.dtype)

    def histogram(
        self,
        input: OpArg = None,
        bins: OpArg = None,
        range: OpArg = None,
        weight: OpArg = None,
        density: OpArg = None,
    ) -> tuple[Tensor, ...]:
        """Applies `histogram` on the tensor.

        Args:
            input: Parameter `input`.
            bins: Parameter `bins`.
            range: Parameter `range`.
            weight: Parameter `weight`.
            density: Parameter `density`.

        Returns:
            Result of `histogram`.
        """
        _ = (
            input,
            bins,
            range,
            weight,
            density,
        )
        return (Tensor(self), Tensor(zero_torch.zeros_like(self), dtype="int64"))

    def hsplit(self, split_size_or_sections: OpArg = None) -> list[Tensor]:
        """Applies `hsplit` on the tensor.

        Args:
            split_size_or_sections: Parameter `split_size_or_sections`.

        Returns:
            Result of `hsplit`.
        """
        _ = (split_size_or_sections,)
        return [Tensor(self)]

    def hypot(self, other: OpArg = None) -> Tensor:
        """Applies `hypot` on the tensor.

        Args:
            other: Parameter `other`.

        Returns:
            Result of `hypot`.
        """
        _ = (other,)
        return zero_torch.hypot(self, Tensor(other))

    def hypot_(self, other: OpArg = None) -> Tensor:
        """Applies `hypot_` on the tensor.

        Args:
            other: Parameter `other`.

        Returns:
            Result of `hypot_`.
        """
        _ = (other,)
        return self.copy_(zero_torch.hypot(self, Tensor(other)))

    def i0(self) -> Tensor:
        """Applies `i0` on the tensor.

        Returns:
            Result of `i0`.
        """
        return zero_torch.special.i0(self)

    def i0_(self) -> Tensor:
        """Applies `i0_` on the tensor.

        Returns:
            Result of `i0_`.
        """
        return self.copy_(zero_torch.special.i0(self))

    def igamma(self, other: OpArg = None) -> Tensor:
        """Applies `igamma` on the tensor.

        Args:
            other: Parameter `other`.

        Returns:
            Result of `igamma`.
        """
        _ = (other,)
        return zero_torch.special.gammainc(self, Tensor(other))

    def igamma_(self, other: OpArg = None) -> Tensor:
        """Applies `igamma_` on the tensor.

        Args:
            other: Parameter `other`.

        Returns:
            Result of `igamma_`.
        """
        _ = (other,)
        return self.copy_(zero_torch.special.gammainc(self, Tensor(other)))

    def igammac(self, other: OpArg = None) -> Tensor:
        """Applies `igammac` on the tensor.

        Args:
            other: Parameter `other`.

        Returns:
            Result of `igammac`.
        """
        _ = (other,)
        return zero_torch.special.gammaincc(self, Tensor(other))

    def igammac_(self, other: OpArg = None) -> Tensor:
        """Applies `igammac_` on the tensor.

        Args:
            other: Parameter `other`.

        Returns:
            Result of `igammac_`.
        """
        _ = (other,)
        return self.copy_(zero_torch.special.gammaincc(self, Tensor(other)))

    def index(self, positions: OpArg = None, dims: OpArg = None) -> Tensor:
        """Applies `index` on the tensor.

        Args:
            positions: Parameter `positions`.
            dims: Parameter `dims`.

        Returns:
            Result of `index`.
        """
        _ = (
            positions,
            dims,
        )
        return Tensor(self, requires_grad=self.requires_grad, dtype=self.dtype)

    def index_add(
        self,
        dim: OpArg = None,
        index: OpArg = None,
        source: OpArg = None,
        alpha: OpArg = None,
    ) -> Tensor:
        """Applies `index_add` on the tensor.

        Args:
            dim: Parameter `dim`.
            index: Parameter `index`.
            source: Parameter `source`.
            alpha: Parameter `alpha`.

        Returns:
            Result of `index_add`.
        """
        _ = (
            dim,
            index,
            source,
            alpha,
        )
        return Tensor(self, requires_grad=self.requires_grad, dtype=self.dtype)

    def index_add_(
        self,
        dim: OpArg = None,
        index: OpArg = None,
        source: OpArg = None,
        alpha: OpArg = None,
    ) -> Tensor:
        """Applies `index_add_` on the tensor.

        Args:
            dim: Parameter `dim`.
            index: Parameter `index`.
            source: Parameter `source`.
            alpha: Parameter `alpha`.

        Returns:
            Result of `index_add_`.
        """
        _ = (dim, index)
        if isinstance(source, Tensor):
            scale = (
                _builtin_float(alpha)
                if isinstance(alpha, (_builtin_int, _builtin_float))
                else 1.0
            )
            if source.shape == self.shape:
                self.copy_(self + source * scale)
        return self

    def index_copy(
        self, dim: OpArg = None, index: OpArg = None, tensor2: OpArg = None
    ) -> Tensor:
        """Applies `index_copy` on the tensor.

        Args:
            dim: Parameter `dim`.
            index: Parameter `index`.
            tensor2: Parameter `tensor2`.

        Returns:
            Result of `index_copy`.
        """
        _ = (
            dim,
            index,
            tensor2,
        )
        return Tensor(self, requires_grad=self.requires_grad, dtype=self.dtype)

    def index_copy_(
        self, dim: OpArg = None, index: OpArg = None, tensor: OpArg = None
    ) -> Tensor:
        """Applies `index_copy_` on the tensor.

        Args:
            dim: Parameter `dim`.
            index: Parameter `index`.
            tensor: Parameter `tensor`.

        Returns:
            Result of `index_copy_`.
        """
        _ = (dim, index)
        if isinstance(tensor, Tensor) and tensor.shape == self.shape:
            self.copy_(tensor)
        return self

    def index_fill(
        self, dim: OpArg = None, index: OpArg = None, value: OpArg = None
    ) -> Tensor:
        """Applies `index_fill` on the tensor.

        Args:
            dim: Parameter `dim`.
            index: Parameter `index`.
            value: Parameter `value`.

        Returns:
            Result of `index_fill`.
        """
        _ = (
            dim,
            index,
            value,
        )
        return Tensor(self, requires_grad=self.requires_grad, dtype=self.dtype)

    def index_fill_(
        self, dim: OpArg = None, index: OpArg = None, value: OpArg = None
    ) -> Tensor:
        """Applies `index_fill_` on the tensor.

        Args:
            dim: Parameter `dim`.
            index: Parameter `index`.
            value: Parameter `value`.

        Returns:
            Result of `index_fill_`.
        """
        _ = (dim, index)
        if isinstance(value, (_builtin_int, _builtin_float, _builtin_bool)):
            self.fill_(value)
        return self

    def index_put(
        self, indices: OpArg = None, values: OpArg = None, accumulate: OpArg = None
    ) -> Tensor:
        """Applies `index_put` on the tensor.

        Args:
            indices: Parameter `indices`.
            values: Parameter `values`.
            accumulate: Parameter `accumulate`.

        Returns:
            Result of `index_put`.
        """
        _ = (
            indices,
            values,
            accumulate,
        )
        return Tensor(self, requires_grad=self.requires_grad, dtype=self.dtype)

    def index_put_(
        self, indices: OpArg = None, values: OpArg = None, accumulate: OpArg = None
    ) -> Tensor:
        """Applies `index_put_` on the tensor.

        Args:
            indices: Parameter `indices`.
            values: Parameter `values`.
            accumulate: Parameter `accumulate`.

        Returns:
            Result of `index_put_`.
        """
        _ = (indices, accumulate)
        if isinstance(values, Tensor) and values.shape == self.shape:
            self.copy_(values)
        return self

    def index_reduce(
        self,
        dim: OpArg = None,
        index: OpArg = None,
        source: OpArg = None,
        reduce: OpArg = None,
        include_self: OpArg = None,
    ) -> Tensor:
        """Applies `index_reduce` on the tensor.

        Args:
            dim: Parameter `dim`.
            index: Parameter `index`.
            source: Parameter `source`.
            reduce: Parameter `reduce`.
            include_self: Parameter `include_self`.

        Returns:
            Result of `index_reduce`.
        """
        _ = (
            dim,
            index,
            source,
            reduce,
            include_self,
        )
        return Tensor(self, requires_grad=self.requires_grad, dtype=self.dtype)

    def index_reduce_(
        self,
        dim: OpArg = None,
        index: OpArg = None,
        source: OpArg = None,
        reduce: OpArg = None,
        include_self: OpArg = None,
    ) -> Tensor:
        """Applies `index_reduce_` on the tensor.

        Args:
            dim: Parameter `dim`.
            index: Parameter `index`.
            source: Parameter `source`.
            reduce: Parameter `reduce`.
            include_self: Parameter `include_self`.

        Returns:
            Result of `index_reduce_`.
        """
        _ = (
            dim,
            index,
            source,
            reduce,
            include_self,
        )
        return self

    def index_select(self, dim: OpArg = None, index: OpArg = None) -> Tensor:
        """Applies `index_select` on the tensor.

        Args:
            dim: Parameter `dim`.
            index: Parameter `index`.

        Returns:
            Result of `index_select`.
        """
        _ = (
            dim,
            index,
        )
        return Tensor(self, requires_grad=self.requires_grad, dtype=self.dtype)

    def indices(self) -> Tensor:
        """Applies `indices` on the tensor.

        Returns:
            Result of `indices`.
        """
        if self._sparse_indices is not None:
            return self._sparse_indices
        return Tensor(self, requires_grad=self.requires_grad, dtype=self.dtype)

    def inner(self, other: OpArg = None) -> Tensor:
        """Applies `inner` on the tensor.

        Args:
            other: Parameter `other`.

        Returns:
            Result of `inner`.
        """
        _ = (other,)
        return zero_torch.inner(self, Tensor(other))

    def int(self, memory_format: OpArg = None) -> Tensor:
        """Applies `int` on the tensor.

        Args:
            memory_format: Parameter `memory_format`.

        Returns:
            Result of `int`.
        """
        _ = (memory_format,)
        return _wrap(_to_tensor(self, dtype="int32"))

    def int_repr(self) -> Tensor:
        """Applies `int_repr` on the tensor.

        Returns:
            Result of `int_repr`.
        """
        return Tensor(self, requires_grad=self.requires_grad, dtype=self.dtype)

    def inverse(self) -> Tensor:
        """Applies `inverse` on the tensor.

        Returns:
            Result of `inverse`.
        """
        return zero_torch.linalg.inv(self)

    def ipu(
        self,
        device: OpArg = None,
        non_blocking: OpArg = None,
        memory_format: OpArg = None,
    ) -> Tensor:
        """Applies `ipu` on the tensor.

        Args:
            device: Parameter `device`.
            non_blocking: Parameter `non_blocking`.
            memory_format: Parameter `memory_format`.

        Returns:
            Result of `ipu`.
        """
        _ = (
            device,
            non_blocking,
            memory_format,
        )
        return Tensor(self, requires_grad=self.requires_grad, dtype=self.dtype)

    def is_coalesced(self) -> _builtin_bool:
        """Applies `is_coalesced` on the tensor.

        Returns:
            Result of `is_coalesced`.
        """
        return False

    def is_complex(self) -> _builtin_bool:
        """Applies `is_complex` on the tensor.

        Returns:
            Result of `is_complex`.
        """
        return "complex" in str(self.dtype)

    def is_conj(self) -> _builtin_bool:
        """Applies `is_conj` on the tensor.

        Returns:
            Result of `is_conj`.
        """
        return False

    def is_contiguous(self, memory_format: OpArg = None) -> _builtin_bool:
        """Applies `is_contiguous` on the tensor.

        Args:
            memory_format: Parameter `memory_format`.

        Returns:
            Result of `is_contiguous`.
        """
        _ = (memory_format,)
        return True

    def is_distributed(self) -> _builtin_bool:
        """Applies `is_distributed` on the tensor.

        Returns:
            Result of `is_distributed`.
        """
        return False

    def is_floating_point(self) -> _builtin_bool:
        """Applies `is_floating_point` on the tensor.

        Returns:
            Result of `is_floating_point`.
        """
        return "float" in str(self.dtype)

    def is_inference(self) -> _builtin_bool:
        """Applies `is_inference` on the tensor.

        Returns:
            Result of `is_inference`.
        """
        return False

    def is_neg(self) -> _builtin_bool:
        """Applies `is_neg` on the tensor.

        Returns:
            Result of `is_neg`.
        """
        return False

    def is_nonzero(self) -> _builtin_bool:
        """Applies `is_nonzero` on the tensor.

        Returns:
            Result of `is_nonzero`.
        """
        return _builtin_bool(self.item())

    def is_pinned(self, device: OpArg = None) -> _builtin_bool:
        """Applies `is_pinned` on the tensor.

        Args:
            device: Parameter `device`.

        Returns:
            Result of `is_pinned`.
        """
        _ = (device,)
        return False

    def is_same_size(self, other: OpArg = None) -> _builtin_bool:
        """Applies `is_same_size` on the tensor.

        Args:
            other: Parameter `other`.

        Returns:
            Result of `is_same_size`.
        """
        _ = (other,)
        return isinstance(other, Tensor) and self.shape == other.shape

    def is_set_to(self, tensor: OpArg = None) -> _builtin_bool:
        """Applies `is_set_to` on the tensor.

        Args:
            tensor: Parameter `tensor`.

        Returns:
            Result of `is_set_to`.
        """
        _ = (tensor,)
        return False

    def is_shared(self) -> _builtin_bool:
        """Applies `is_shared` on the tensor.

        Returns:
            Result of `is_shared`.
        """
        return False

    def is_signed(self) -> _builtin_bool:
        """Applies `is_signed` on the tensor.

        Returns:
            Result of `is_signed`.
        """
        return "float" in str(self.dtype) or (
            "int" in str(self.dtype) and "uint" not in str(self.dtype)
        )

    def isclose(
        self,
        other: OpArg = None,
        rtol: OpArg = None,
        atol: OpArg = None,
        equal_nan: OpArg = None,
    ) -> Tensor:
        """Applies `isclose` on the tensor.

        Args:
            other: Parameter `other`.
            rtol: Parameter `rtol`.
            atol: Parameter `atol`.
            equal_nan: Parameter `equal_nan`.

        Returns:
            Result of `isclose`.
        """
        _ = (
            other,
            rtol,
            atol,
            equal_nan,
        )
        return zero_torch.isclose(
            self,
            other,
            rtol=_builtin_float(rtol)
            if isinstance(rtol, (_builtin_int, _builtin_float))
            else 1e-5,
            atol=_builtin_float(atol)
            if isinstance(atol, (_builtin_int, _builtin_float))
            else 1e-8,
            equal_nan=_builtin_bool(equal_nan),
        )

    def isfinite(self) -> Tensor:
        """Applies `isfinite` on the tensor.

        Returns:
            Result of `isfinite`.
        """
        return zero_torch.isfinite(self)

    def isinf(self) -> Tensor:
        """Applies `isinf` on the tensor.

        Returns:
            Result of `isinf`.
        """
        return zero_torch.isinf(self)

    def isnan(self) -> Tensor:
        """Applies `isnan` on the tensor.

        Returns:
            Result of `isnan`.
        """
        return zero_torch.isnan(self)

    def isneginf(self) -> Tensor:
        """Applies `isneginf` on the tensor.

        Returns:
            Result of `isneginf`.
        """
        return Tensor(self, requires_grad=self.requires_grad, dtype=self.dtype)

    def isposinf(self) -> Tensor:
        """Applies `isposinf` on the tensor.

        Returns:
            Result of `isposinf`.
        """
        return Tensor(self, requires_grad=self.requires_grad, dtype=self.dtype)

    def isreal(self) -> Tensor:
        """Applies `isreal` on the tensor.

        Returns:
            Result of `isreal`.
        """
        return Tensor(self, requires_grad=self.requires_grad, dtype=self.dtype)

    def istft(
        self,
        n_fft: OpArg = None,
        hop_length: OpArg = None,
        win_length: OpArg = None,
        window: OpArg = None,
        center: OpArg = None,
        normalized: OpArg = None,
        onesided: OpArg = None,
        length: OpArg = None,
        return_complex: OpArg = None,
    ) -> Tensor:
        """Applies `istft` on the tensor.

        Args:
            n_fft: Parameter `n_fft`.
            hop_length: Parameter `hop_length`.
            win_length: Parameter `win_length`.
            window: Parameter `window`.
            center: Parameter `center`.
            normalized: Parameter `normalized`.
            onesided: Parameter `onesided`.
            length: Parameter `length`.
            return_complex: Parameter `return_complex`.

        Returns:
            Result of `istft`.
        """
        _ = (
            n_fft,
            hop_length,
            win_length,
            window,
            center,
            normalized,
            onesided,
            length,
            return_complex,
        )
        return Tensor(self, requires_grad=self.requires_grad, dtype=self.dtype)

    def kron(self, other: OpArg = None) -> Tensor:
        """Applies `kron` on the tensor.

        Args:
            other: Parameter `other`.

        Returns:
            Result of `kron`.
        """
        _ = (other,)
        return Tensor(self, requires_grad=self.requires_grad, dtype=self.dtype)

    def kthvalue(
        self, k: OpArg = None, dim: OpArg = None, keepdim: OpArg = None
    ) -> tuple[Tensor, ...]:
        """Applies `kthvalue` on the tensor.

        Args:
            k: Parameter `k`.
            dim: Parameter `dim`.
            keepdim: Parameter `keepdim`.

        Returns:
            Result of `kthvalue`.
        """
        _ = (
            k,
            dim,
            keepdim,
        )
        return (Tensor(self), Tensor(zero_torch.zeros_like(self), dtype="int64"))

    def lcm(self, other: OpArg = None) -> Tensor:
        """Applies `lcm` on the tensor.

        Args:
            other: Parameter `other`.

        Returns:
            Result of `lcm`.
        """
        _ = (other,)
        return zero_torch.lcm(self, Tensor(other))

    def lcm_(self, other: OpArg = None) -> Tensor:
        """Applies `lcm_` on the tensor.

        Args:
            other: Parameter `other`.

        Returns:
            Result of `lcm_`.
        """
        _ = (other,)
        return self.copy_(zero_torch.lcm(self, Tensor(other)))

    def ldexp(self, other: OpArg = None) -> Tensor:
        """Applies `ldexp` on the tensor.

        Args:
            other: Parameter `other`.

        Returns:
            Result of `ldexp`.
        """
        _ = (other,)
        return zero_torch.ldexp(self, Tensor(other))

    def ldexp_(self, other: OpArg = None) -> Tensor:
        """Applies `ldexp_` on the tensor.

        Args:
            other: Parameter `other`.

        Returns:
            Result of `ldexp_`.
        """
        _ = (other,)
        return self.copy_(zero_torch.ldexp(self, Tensor(other)))

    def le(self, other: OpArg = None) -> Tensor:
        """Applies `le` on the tensor.

        Args:
            other: Parameter `other`.

        Returns:
            Result of `le`.
        """
        _ = (other,)
        return zero_torch.less_equal(self, Tensor(other))

    def le_(self, other: OpArg = None) -> Tensor:
        """Applies `le_` on the tensor.

        Args:
            other: Parameter `other`.

        Returns:
            Result of `le_`.
        """
        _ = (other,)
        return self.copy_(zero_torch.less_equal(self, Tensor(other)))

    def lerp(self, end: OpArg = None, weight: OpArg = None) -> Tensor:
        """Applies `lerp` on the tensor.

        Args:
            end: Parameter `end`.
            weight: Parameter `weight`.

        Returns:
            Result of `lerp`.
        """
        _ = (
            end,
            weight,
        )
        return Tensor(self, requires_grad=self.requires_grad, dtype=self.dtype)

    def lerp_(self, end: OpArg = None, weight: OpArg = None) -> Tensor:
        """Applies `lerp_` on the tensor.

        Args:
            end: Parameter `end`.
            weight: Parameter `weight`.

        Returns:
            Result of `lerp_`.
        """
        _ = (
            end,
            weight,
        )
        return self

    def less(self, other: OpArg = None) -> Tensor:
        """Applies `less` on the tensor.

        Args:
            other: Parameter `other`.

        Returns:
            Result of `less`.
        """
        _ = (other,)
        return zero_torch.less(self, Tensor(other))

    def less_(self, other: OpArg = None) -> Tensor:
        """Applies `less_` on the tensor.

        Args:
            other: Parameter `other`.

        Returns:
            Result of `less_`.
        """
        _ = (other,)
        return self.copy_(zero_torch.less(self, Tensor(other)))

    def less_equal(self, other: OpArg = None) -> Tensor:
        """Applies `less_equal` on the tensor.

        Args:
            other: Parameter `other`.

        Returns:
            Result of `less_equal`.
        """
        _ = (other,)
        return zero_torch.less_equal(self, Tensor(other))

    def less_equal_(self, other: OpArg = None) -> Tensor:
        """Applies `less_equal_` on the tensor.

        Args:
            other: Parameter `other`.

        Returns:
            Result of `less_equal_`.
        """
        _ = (other,)
        return self.copy_(zero_torch.less_equal(self, Tensor(other)))

    def lgamma_(self) -> Tensor:
        """Applies `lgamma_` on the tensor.

        Returns:
            Result of `lgamma_`.
        """
        return self.copy_(zero_torch.lgamma(self))

    def log(self) -> Tensor:
        """Applies `log` on the tensor.

        Returns:
            Result of `log`.
        """
        return zero_torch.log(self)

    def log10(self) -> Tensor:
        """Applies `log10` on the tensor.

        Returns:
            Result of `log10`.
        """
        return zero_torch.log10(self)

    def log10_(self) -> Tensor:
        """Applies `log10_` on the tensor.

        Returns:
            Result of `log10_`.
        """
        return self.copy_(zero_torch.log10(self))

    def log1p(self) -> Tensor:
        """Applies `log1p` on the tensor.

        Returns:
            Result of `log1p`.
        """
        return zero_torch.log1p(self)

    def log1p_(self) -> Tensor:
        """Applies `log1p_` on the tensor.

        Returns:
            Result of `log1p_`.
        """
        return self.copy_(zero_torch.log1p(self))

    def log2(self) -> Tensor:
        """Applies `log2` on the tensor.

        Returns:
            Result of `log2`.
        """
        return zero_torch.log2(self)

    def log2_(self) -> Tensor:
        """Applies `log2_` on the tensor.

        Returns:
            Result of `log2_`.
        """
        return self.copy_(zero_torch.log2(self))

    def log_(self) -> Tensor:
        """Applies `log_` on the tensor.

        Returns:
            Result of `log_`.
        """
        return self.copy_(zero_torch.log(self))

    def log_normal_(
        self, mean: OpArg = None, std: OpArg = None, generator: OpArg = None
    ) -> Tensor:
        """Applies `log_normal_` on the tensor.

        Args:
            mean: Parameter `mean`.
            std: Parameter `std`.
            generator: Parameter `generator`.

        Returns:
            Result of `log_normal_`.
        """
        _ = (
            mean,
            std,
            generator,
        )
        return self

    def log_softmax(self, dim: OpArg = None, dtype: OpArg = None) -> Tensor:
        """Applies `log_softmax` on the tensor.

        Args:
            dim: Parameter `dim`.
            dtype: Parameter `dtype`.

        Returns:
            Result of `log_softmax`.
        """
        _ = (
            dim,
            dtype,
        )
        return zero_torch.nn.functional.log_softmax(
            self, dim=_builtin_int(dim) if isinstance(dim, _builtin_int) else -1
        )

    def logaddexp(self, other: OpArg = None) -> Tensor:
        """Applies `logaddexp` on the tensor.

        Args:
            other: Parameter `other`.

        Returns:
            Result of `logaddexp`.
        """
        _ = (other,)
        return zero_torch.logaddexp(self, Tensor(other))

    def logaddexp2(self, other: OpArg = None) -> Tensor:
        """Applies `logaddexp2` on the tensor.

        Args:
            other: Parameter `other`.

        Returns:
            Result of `logaddexp2`.
        """
        _ = (other,)
        return zero_torch.logaddexp2(self, Tensor(other))

    def logcumsumexp(self, dim: OpArg = None) -> Tensor:
        """Applies `logcumsumexp` on the tensor.

        Args:
            dim: Parameter `dim`.

        Returns:
            Result of `logcumsumexp`.
        """
        _ = (dim,)
        return Tensor(self, requires_grad=self.requires_grad, dtype=self.dtype)

    def logdet(self) -> Tensor:
        """Applies `logdet` on the tensor.

        Returns:
            Result of `logdet`.
        """
        return Tensor(self, requires_grad=self.requires_grad, dtype=self.dtype)

    def logical_and(self, other: OpArg = None) -> Tensor:
        """Applies `logical_and` on the tensor.

        Args:
            other: Parameter `other`.

        Returns:
            Result of `logical_and`.
        """
        _ = (other,)
        return zero_torch.logical_and(self, self if other is None else Tensor(other))

    def logical_and_(self, other: OpArg = None) -> Tensor:
        """Applies `logical_and_` on the tensor.

        Args:
            other: Parameter `other`.

        Returns:
            Result of `logical_and_`.
        """
        _ = (other,)
        return self.copy_(
            zero_torch.logical_and(self, self if other is None else Tensor(other))
        )

    def logical_not(self) -> Tensor:
        """Applies `logical_not` on the tensor.

        Returns:
            Result of `logical_not`.
        """
        return zero_torch.logical_not(self)

    def logical_not_(self) -> Tensor:
        """Applies `logical_not_` on the tensor.

        Returns:
            Result of `logical_not_`.
        """
        return self.copy_(zero_torch.logical_not(self))

    def logical_or(self, other: OpArg = None) -> Tensor:
        """Applies `logical_or` on the tensor.

        Args:
            other: Parameter `other`.

        Returns:
            Result of `logical_or`.
        """
        _ = (other,)
        return zero_torch.logical_or(self, self if other is None else Tensor(other))

    def logical_or_(self, other: OpArg = None) -> Tensor:
        """Applies `logical_or_` on the tensor.

        Args:
            other: Parameter `other`.

        Returns:
            Result of `logical_or_`.
        """
        _ = (other,)
        return self.copy_(
            zero_torch.logical_or(self, self if other is None else Tensor(other))
        )

    def logical_xor(self, other: OpArg = None) -> Tensor:
        """Applies `logical_xor` on the tensor.

        Args:
            other: Parameter `other`.

        Returns:
            Result of `logical_xor`.
        """
        _ = (other,)
        return zero_torch.logical_xor(self, self if other is None else Tensor(other))

    def logical_xor_(self, other: OpArg = None) -> Tensor:
        """Applies `logical_xor_` on the tensor.

        Args:
            other: Parameter `other`.

        Returns:
            Result of `logical_xor_`.
        """
        _ = (other,)
        return self.copy_(
            zero_torch.logical_xor(self, self if other is None else Tensor(other))
        )

    def logit(self) -> Tensor:
        """Applies `logit` on the tensor.

        Returns:
            Result of `logit`.
        """
        return zero_torch.logit(self)

    def logit_(self) -> Tensor:
        """Applies `logit_` on the tensor.

        Returns:
            Result of `logit_`.
        """
        return self.copy_(zero_torch.logit(self))

    def logsumexp(self, dim: OpArg = None, keepdim: OpArg = None) -> Tensor:
        """Applies `logsumexp` on the tensor.

        Args:
            dim: Parameter `dim`.
            keepdim: Parameter `keepdim`.

        Returns:
            Result of `logsumexp`.
        """
        _ = (
            dim,
            keepdim,
        )
        return zero_torch.logsumexp(
            self,
            dim=dim if isinstance(dim, _builtin_int) else None,
            keepdim=_builtin_bool(keepdim),
        )

    def long(self, memory_format: OpArg = None) -> Tensor:
        """Applies `long` on the tensor.

        Args:
            memory_format: Parameter `memory_format`.

        Returns:
            Result of `long`.
        """
        _ = (memory_format,)
        return _wrap(_to_tensor(self, dtype="int64"))

    def lstsq(self, other: OpArg = None) -> tuple[Tensor, ...]:
        """Applies `lstsq` on the tensor.

        Args:
            other: Parameter `other`.

        Returns:
            Result of `lstsq`.
        """
        _ = (other,)
        return (Tensor(self), Tensor(zero_torch.zeros_like(self), dtype="int64"))

    def lt(self, other: OpArg = None) -> Tensor:
        """Applies `lt` on the tensor.

        Args:
            other: Parameter `other`.

        Returns:
            Result of `lt`.
        """
        _ = (other,)
        return zero_torch.less(self, Tensor(other))

    def lt_(self, other: OpArg = None) -> Tensor:
        """Applies `lt_` on the tensor.

        Args:
            other: Parameter `other`.

        Returns:
            Result of `lt_`.
        """
        _ = (other,)
        return self.copy_(zero_torch.less(self, Tensor(other)))

    def lu(self, pivot: OpArg = None, get_infos: OpArg = None) -> tuple[Tensor, ...]:
        """Applies `lu` on the tensor.

        Args:
            pivot: Parameter `pivot`.
            get_infos: Parameter `get_infos`.

        Returns:
            Result of `lu`.
        """
        _ = (
            pivot,
            get_infos,
        )
        return (Tensor(self), Tensor(zero_torch.zeros_like(self), dtype="int64"))

    def lu_solve(self, LU_data: OpArg = None, LU_pivots: OpArg = None) -> Tensor:
        """Applies `lu_solve` on the tensor.

        Args:
            LU_data: Parameter `LU_data`.
            LU_pivots: Parameter `LU_pivots`.

        Returns:
            Result of `lu_solve`.
        """
        _ = (
            LU_data,
            LU_pivots,
        )
        return Tensor(self, requires_grad=self.requires_grad, dtype=self.dtype)

    def map2_(self) -> Tensor:
        """Applies `map2_` on the tensor.

        Returns:
            Result of `map2_`.
        """
        return self

    def map_(self, tensor: OpArg = None, callable: OpArg = None) -> Tensor:
        """Applies `map_` on the tensor.

        Args:
            tensor: Parameter `tensor`.
            callable: Parameter `callable`.

        Returns:
            Result of `map_`.
        """
        _ = (
            tensor,
            callable,
        )
        return self

    def masked_fill(self, mask: OpArg = None, value: OpArg = None) -> Tensor:
        """Applies `masked_fill` on the tensor.

        Args:
            mask: Parameter `mask`.
            value: Parameter `value`.

        Returns:
            Result of `masked_fill`.
        """
        _ = (
            mask,
            value,
        )
        return zero_torch.where(mask, value, self)

    def masked_fill_(self, mask: OpArg = None, value: OpArg = None) -> Tensor:
        """Applies `masked_fill_` on the tensor.

        Args:
            mask: Parameter `mask`.
            value: Parameter `value`.

        Returns:
            Result of `masked_fill_`.
        """
        _ = (
            mask,
            value,
        )
        return self.copy_(zero_torch.where(mask, value, self))

    def masked_scatter(self, mask: OpArg = None, tensor: OpArg = None) -> Tensor:
        """Applies `masked_scatter` on the tensor.

        Args:
            mask: Parameter `mask`.
            tensor: Parameter `tensor`.

        Returns:
            Result of `masked_scatter`.
        """
        _ = (
            mask,
            tensor,
        )
        return Tensor(self, requires_grad=self.requires_grad, dtype=self.dtype)

    def masked_scatter_(self, mask: OpArg = None, source: OpArg = None) -> Tensor:
        """Applies `masked_scatter_` on the tensor.

        Args:
            mask: Parameter `mask`.
            source: Parameter `source`.

        Returns:
            Result of `masked_scatter_`.
        """
        if mask is not None and source is not None:
            return self.copy_(zero_torch.where(mask, source, self))
        return self

    def masked_select(self, mask: OpArg = None) -> Tensor:
        """Applies `masked_select` on the tensor.

        Args:
            mask: Parameter `mask`.

        Returns:
            Result of `masked_select`.
        """
        _ = (mask,)
        return Tensor(self, requires_grad=self.requires_grad, dtype=self.dtype)

    def matmul(self, tensor2: OpArg = None) -> Tensor:
        """Applies `matmul` on the tensor.

        Args:
            tensor2: Parameter `tensor2`.

        Returns:
            Result of `matmul`.
        """
        _ = (tensor2,)
        return self @ Tensor(tensor2)

    def matrix_exp(self) -> Tensor:
        """Applies `matrix_exp` on the tensor.

        Returns:
            Result of `matrix_exp`.
        """
        return zero_torch.linalg.matrix_exp(self)

    def matrix_power(self, n: OpArg = None) -> Tensor:
        """Applies `matrix_power` on the tensor.

        Args:
            n: Parameter `n`.

        Returns:
            Result of `matrix_power`.
        """
        _ = (n,)
        return zero_torch.linalg.matrix_power(
            self, _builtin_int(n) if isinstance(n, _builtin_int) else 1
        )

    def max(
        self, dim: OpArg = None, keepdim: OpArg = None
    ) -> Tensor | tuple[Tensor, ...]:
        """Applies `max` on the tensor.

        Args:
            dim: Parameter `dim`.
            keepdim: Parameter `keepdim`.

        Returns:
            Result of `max`.
        """
        _ = (
            dim,
            keepdim,
        )
        res = zero_torch.max(
            self,
            dim=dim if isinstance(dim, _builtin_int) else None,
            keepdim=_builtin_bool(keepdim),
        )
        return (
            (
                res,
                zero_torch.argmax(
                    self,
                    dim=dim if isinstance(dim, _builtin_int) else None,
                    keepdim=_builtin_bool(keepdim),
                ),
            )
            if dim is not None
            else res
        )

    def maximum(self, other: OpArg = None) -> Tensor:
        """Applies `maximum` on the tensor.

        Args:
            other: Parameter `other`.

        Returns:
            Result of `maximum`.
        """
        _ = (other,)
        return zero_torch.maximum(self, Tensor(other))

    def median(self, dim: OpArg = None, keepdim: OpArg = None) -> tuple[Tensor, ...]:
        """Applies `median` on the tensor.

        Args:
            dim: Parameter `dim`.
            keepdim: Parameter `keepdim`.

        Returns:
            Result of `median`.
        """
        _ = (
            dim,
            keepdim,
        )
        return (Tensor(self), Tensor(zero_torch.zeros_like(self), dtype="int64"))

    def min(
        self, dim: OpArg = None, keepdim: OpArg = None
    ) -> Tensor | tuple[Tensor, ...]:
        """Applies `min` on the tensor.

        Args:
            dim: Parameter `dim`.
            keepdim: Parameter `keepdim`.

        Returns:
            Result of `min`.
        """
        _ = (
            dim,
            keepdim,
        )
        res = zero_torch.min(
            self,
            dim=dim if isinstance(dim, _builtin_int) else None,
            keepdim=_builtin_bool(keepdim),
        )
        return (
            (
                res,
                zero_torch.argmin(
                    self,
                    dim=dim if isinstance(dim, _builtin_int) else None,
                    keepdim=_builtin_bool(keepdim),
                ),
            )
            if dim is not None
            else res
        )

    def minimum(self, other: OpArg = None) -> Tensor:
        """Applies `minimum` on the tensor.

        Args:
            other: Parameter `other`.

        Returns:
            Result of `minimum`.
        """
        _ = (other,)
        return zero_torch.minimum(self, Tensor(other))

    def mm(self, mat2: OpArg = None) -> Tensor:
        """Applies `mm` on the tensor.

        Args:
            mat2: Parameter `mat2`.

        Returns:
            Result of `mm`.
        """
        _ = (mat2,)
        return self @ Tensor(mat2)

    def mode(self, dim: OpArg = None, keepdim: OpArg = None) -> tuple[Tensor, ...]:
        """Applies `mode` on the tensor.

        Args:
            dim: Parameter `dim`.
            keepdim: Parameter `keepdim`.

        Returns:
            Result of `mode`.
        """
        _ = (
            dim,
            keepdim,
        )
        return (Tensor(self), Tensor(zero_torch.zeros_like(self), dtype="int64"))

    def module_load(self, other: OpArg = None, assign: OpArg = None) -> None:
        """Applies `module_load` on the tensor.

        Args:
            other: Parameter `other`.
            assign: Parameter `assign`.
        """
        _ = (
            other,
            assign,
        )
        return None

    def moveaxis(self, source: OpArg = None, destination: OpArg = None) -> Tensor:
        """Applies `moveaxis` on the tensor.

        Args:
            source: Parameter `source`.
            destination: Parameter `destination`.

        Returns:
            Result of `moveaxis`.
        """
        _ = (
            source,
            destination,
        )
        return Tensor(self, requires_grad=self.requires_grad, dtype=self.dtype)

    def movedim(self, source: OpArg = None, destination: OpArg = None) -> Tensor:
        """Applies `movedim` on the tensor.

        Args:
            source: Parameter `source`.
            destination: Parameter `destination`.

        Returns:
            Result of `movedim`.
        """
        _ = (
            source,
            destination,
        )
        return Tensor(self, requires_grad=self.requires_grad, dtype=self.dtype)

    def msort(self) -> Tensor:
        """Applies `msort` on the tensor.

        Returns:
            Result of `msort`.
        """
        return Tensor(self, requires_grad=self.requires_grad, dtype=self.dtype)

    def mtia(
        self,
        device: OpArg = None,
        non_blocking: OpArg = None,
        memory_format: OpArg = None,
    ) -> Tensor:
        """Applies `mtia` on the tensor.

        Args:
            device: Parameter `device`.
            non_blocking: Parameter `non_blocking`.
            memory_format: Parameter `memory_format`.

        Returns:
            Result of `mtia`.
        """
        _ = (
            device,
            non_blocking,
            memory_format,
        )
        return Tensor(self, requires_grad=self.requires_grad, dtype=self.dtype)

    def mul(self, value: OpArg = None) -> Tensor:
        """Applies `mul` on the tensor.

        Args:
            value: Parameter `value`.

        Returns:
            Result of `mul`.
        """
        _ = (value,)
        return self * Tensor(value)

    def mul_(self, value: OpArg = None) -> Tensor:
        """Applies `mul_` on the tensor.

        Args:
            value: Parameter `value`.

        Returns:
            Result of `mul_`.
        """
        _ = (value,)
        return self.copy_(self * Tensor(value))

    def multinomial(
        self,
        num_samples: OpArg = None,
        replacement: OpArg = None,
        generator: OpArg = None,
    ) -> Tensor:
        """Applies `multinomial` on the tensor.

        Args:
            num_samples: Parameter `num_samples`.
            replacement: Parameter `replacement`.
            generator: Parameter `generator`.

        Returns:
            Result of `multinomial`.
        """
        _ = (
            num_samples,
            replacement,
            generator,
        )
        return Tensor(self, requires_grad=self.requires_grad, dtype=self.dtype)

    def multiply(self, value: OpArg = None) -> Tensor:
        """Applies `multiply` on the tensor.

        Args:
            value: Parameter `value`.

        Returns:
            Result of `multiply`.
        """
        _ = (value,)
        return self * Tensor(value)

    def multiply_(self, value: OpArg = None) -> Tensor:
        """Applies `multiply_` on the tensor.

        Args:
            value: Parameter `value`.

        Returns:
            Result of `multiply_`.
        """
        _ = (value,)
        return self.copy_(self * Tensor(value))

    def mv(self, vec: OpArg = None) -> Tensor:
        """Applies `mv` on the tensor.

        Args:
            vec: Parameter `vec`.

        Returns:
            Result of `mv`.
        """
        _ = (vec,)
        return self @ Tensor(vec)

    def mvlgamma(self, p: OpArg = None) -> Tensor:
        """Applies `mvlgamma` on the tensor.

        Args:
            p: Parameter `p`.

        Returns:
            Result of `mvlgamma`.
        """
        _ = (p,)
        return zero_torch.special.multigammaln(
            self, _builtin_int(p) if isinstance(p, _builtin_int) else 1
        )

    def mvlgamma_(self, p: OpArg = None) -> Tensor:
        """Applies `mvlgamma_` on the tensor.

        Args:
            p: Parameter `p`.

        Returns:
            Result of `mvlgamma_`.
        """
        _ = (p,)
        return self.copy_(
            zero_torch.special.multigammaln(
                self, _builtin_int(p) if isinstance(p, _builtin_int) else 1
            )
        )

    def nan_to_num(
        self, nan: OpArg = None, posinf: OpArg = None, neginf: OpArg = None
    ) -> Tensor:
        """Applies `nan_to_num` on the tensor.

        Args:
            nan: Parameter `nan`.
            posinf: Parameter `posinf`.
            neginf: Parameter `neginf`.

        Returns:
            Result of `nan_to_num`.
        """
        _ = (
            nan,
            posinf,
            neginf,
        )
        return zero_torch.nan_to_num(
            self,
            nan=_builtin_float(nan)
            if isinstance(nan, (_builtin_int, _builtin_float))
            else 0.0,
            posinf=_builtin_float(posinf)
            if isinstance(posinf, (_builtin_int, _builtin_float))
            else None,
            neginf=_builtin_float(neginf)
            if isinstance(neginf, (_builtin_int, _builtin_float))
            else None,
        )

    def nan_to_num_(
        self, nan: OpArg = None, posinf: OpArg = None, neginf: OpArg = None
    ) -> Tensor:
        """Applies `nan_to_num_` on the tensor.

        Args:
            nan: Parameter `nan`.
            posinf: Parameter `posinf`.
            neginf: Parameter `neginf`.

        Returns:
            Result of `nan_to_num_`.
        """
        _ = (
            nan,
            posinf,
            neginf,
        )
        return self.copy_(
            zero_torch.nan_to_num(
                self,
                nan=_builtin_float(nan)
                if isinstance(nan, (_builtin_int, _builtin_float))
                else 0.0,
                posinf=_builtin_float(posinf)
                if isinstance(posinf, (_builtin_int, _builtin_float))
                else None,
                neginf=_builtin_float(neginf)
                if isinstance(neginf, (_builtin_int, _builtin_float))
                else None,
            )
        )

    def nanmean(
        self, dim: OpArg = None, keepdim: OpArg = None, dtype: OpArg = None
    ) -> Tensor:
        """Applies `nanmean` on the tensor.

        Args:
            dim: Parameter `dim`.
            keepdim: Parameter `keepdim`.
            dtype: Parameter `dtype`.

        Returns:
            Result of `nanmean`.
        """
        _ = (
            dim,
            keepdim,
            dtype,
        )
        return Tensor(self, requires_grad=self.requires_grad, dtype=self.dtype)

    def nanmedian(self, dim: OpArg = None, keepdim: OpArg = None) -> tuple[Tensor, ...]:
        """Applies `nanmedian` on the tensor.

        Args:
            dim: Parameter `dim`.
            keepdim: Parameter `keepdim`.

        Returns:
            Result of `nanmedian`.
        """
        _ = (
            dim,
            keepdim,
        )
        return (Tensor(self), Tensor(zero_torch.zeros_like(self), dtype="int64"))

    def nanquantile(
        self,
        q: OpArg = None,
        dim: OpArg = None,
        keepdim: OpArg = None,
        interpolation: OpArg = None,
    ) -> Tensor:
        """Applies `nanquantile` on the tensor.

        Args:
            q: Parameter `q`.
            dim: Parameter `dim`.
            keepdim: Parameter `keepdim`.
            interpolation: Parameter `interpolation`.

        Returns:
            Result of `nanquantile`.
        """
        _ = (
            q,
            dim,
            keepdim,
            interpolation,
        )
        return Tensor(self, requires_grad=self.requires_grad, dtype=self.dtype)

    def nansum(
        self, dim: OpArg = None, keepdim: OpArg = None, dtype: OpArg = None
    ) -> Tensor:
        """Applies `nansum` on the tensor.

        Args:
            dim: Parameter `dim`.
            keepdim: Parameter `keepdim`.
            dtype: Parameter `dtype`.

        Returns:
            Result of `nansum`.
        """
        _ = (
            dim,
            keepdim,
            dtype,
        )
        return Tensor(self, requires_grad=self.requires_grad, dtype=self.dtype)

    def narrow(
        self, dimension: OpArg = None, start: OpArg = None, length: OpArg = None
    ) -> Tensor:
        """Applies `narrow` on the tensor.

        Args:
            dimension: Parameter `dimension`.
            start: Parameter `start`.
            length: Parameter `length`.

        Returns:
            Result of `narrow`.
        """
        _ = (
            dimension,
            start,
            length,
        )
        return Tensor(self, requires_grad=self.requires_grad, dtype=self.dtype)

    def narrow_copy(
        self, dimension: OpArg = None, start: OpArg = None, length: OpArg = None
    ) -> Tensor:
        """Applies `narrow_copy` on the tensor.

        Args:
            dimension: Parameter `dimension`.
            start: Parameter `start`.
            length: Parameter `length`.

        Returns:
            Result of `narrow_copy`.
        """
        _ = (
            dimension,
            start,
            length,
        )
        return Tensor(self, requires_grad=self.requires_grad, dtype=self.dtype)

    def ndimension(self) -> _builtin_int:
        """Applies `ndimension` on the tensor.

        Returns:
            Result of `ndimension`.
        """
        return self.dim()

    def ne(self, other: OpArg = None) -> Tensor:
        """Applies `ne` on the tensor.

        Args:
            other: Parameter `other`.

        Returns:
            Result of `ne`.
        """
        _ = (other,)
        return zero_torch.not_equal(self, Tensor(other))

    def ne_(self, other: OpArg = None) -> Tensor:
        """Applies `ne_` on the tensor.

        Args:
            other: Parameter `other`.

        Returns:
            Result of `ne_`.
        """
        _ = (other,)
        return self.copy_(zero_torch.not_equal(self, Tensor(other)))

    def neg(self) -> Tensor:
        """Applies `neg` on the tensor.

        Returns:
            Result of `neg`.
        """
        return zero_torch.negative(self)

    def neg_(self) -> Tensor:
        """Applies `neg_` on the tensor.

        Returns:
            Result of `neg_`.
        """
        return self.copy_(zero_torch.negative(self))

    def negative(self) -> Tensor:
        """Applies `negative` on the tensor.

        Returns:
            Result of `negative`.
        """
        return zero_torch.negative(self)

    def negative_(self) -> Tensor:
        """Applies `negative_` on the tensor.

        Returns:
            Result of `negative_`.
        """
        return self.copy_(zero_torch.negative(self))

    def nelement(self) -> _builtin_int:
        """Applies `nelement` on the tensor.

        Returns:
            Result of `nelement`.
        """
        return self.numel()

    def new(self) -> Tensor:
        """Applies `new` on the tensor.

        Returns:
            Result of `new`.
        """
        return Tensor(self, requires_grad=self.requires_grad, dtype=self.dtype)

    def new_empty(
        self,
        size: OpArg = None,
        dtype: OpArg = None,
        device: OpArg = None,
        requires_grad: OpArg = None,
        layout: OpArg = None,
        pin_memory: OpArg = None,
    ) -> Tensor:
        """Applies `new_empty` on the tensor.

        Args:
            size: Parameter `size`.
            dtype: Parameter `dtype`.
            device: Parameter `device`.
            requires_grad: Parameter `requires_grad`.
            layout: Parameter `layout`.
            pin_memory: Parameter `pin_memory`.

        Returns:
            Result of `new_empty`.
        """
        _ = (
            device,
            layout,
            pin_memory,
        )
        target = (
            size
            if isinstance(size, (tuple, list))
            else (
                (_builtin_int(size),) if isinstance(size, _builtin_int) else self.shape
            )
        )
        res = zero_torch.zeros(
            target,
            dtype=dtype if dtype is not None else self.dtype,
        )
        res.requires_grad = (
            _builtin_bool(requires_grad) if requires_grad is not None else False
        )
        return res

    def new_empty_strided(
        self,
        size: OpArg = None,
        stride: OpArg = None,
        dtype: OpArg = None,
        device: OpArg = None,
        requires_grad: OpArg = None,
        layout: OpArg = None,
        pin_memory: OpArg = None,
    ) -> Tensor:
        """Applies `new_empty_strided` on the tensor.

        Args:
            size: Parameter `size`.
            stride: Parameter `stride`.
            dtype: Parameter `dtype`.
            device: Parameter `device`.
            requires_grad: Parameter `requires_grad`.
            layout: Parameter `layout`.
            pin_memory: Parameter `pin_memory`.

        Returns:
            Result of `new_empty_strided`.
        """
        _ = (
            stride,
            device,
            layout,
            pin_memory,
        )
        target = (
            size
            if isinstance(size, (tuple, list))
            else (
                (_builtin_int(size),) if isinstance(size, _builtin_int) else self.shape
            )
        )
        res = zero_torch.zeros(
            target,
            dtype=dtype if dtype is not None else self.dtype,
        )
        res.requires_grad = (
            _builtin_bool(requires_grad) if requires_grad is not None else False
        )
        return res

    def new_full(
        self,
        size: OpArg = None,
        fill_value: OpArg = None,
        dtype: OpArg = None,
        device: OpArg = None,
        requires_grad: OpArg = None,
        layout: OpArg = None,
        pin_memory: OpArg = None,
    ) -> Tensor:
        """Applies `new_full` on the tensor.

        Args:
            size: Parameter `size`.
            fill_value: Parameter `fill_value`.
            dtype: Parameter `dtype`.
            device: Parameter `device`.
            requires_grad: Parameter `requires_grad`.
            layout: Parameter `layout`.
            pin_memory: Parameter `pin_memory`.

        Returns:
            Result of `new_full`.
        """
        _ = (
            device,
            layout,
            pin_memory,
        )
        target = (
            size
            if isinstance(size, (tuple, list))
            else (
                (_builtin_int(size),) if isinstance(size, _builtin_int) else self.shape
            )
        )
        res = zero_torch.full(
            target,
            fill_value
            if isinstance(fill_value, (_builtin_int, _builtin_float, _builtin_bool))
            else 0.0,
            dtype=dtype if dtype is not None else self.dtype,
        )
        res.requires_grad = (
            _builtin_bool(requires_grad) if requires_grad is not None else False
        )
        return res

    def new_ones(
        self,
        size: OpArg = None,
        dtype: OpArg = None,
        device: OpArg = None,
        requires_grad: OpArg = None,
        layout: OpArg = None,
        pin_memory: OpArg = None,
    ) -> Tensor:
        """Applies `new_ones` on the tensor.

        Args:
            size: Parameter `size`.
            dtype: Parameter `dtype`.
            device: Parameter `device`.
            requires_grad: Parameter `requires_grad`.
            layout: Parameter `layout`.
            pin_memory: Parameter `pin_memory`.

        Returns:
            Result of `new_ones`.
        """
        _ = (
            device,
            layout,
            pin_memory,
        )
        target = (
            size
            if isinstance(size, (tuple, list))
            else (
                (_builtin_int(size),) if isinstance(size, _builtin_int) else self.shape
            )
        )
        res = zero_torch.ones(
            target,
            dtype=dtype if dtype is not None else self.dtype,
        )
        res.requires_grad = (
            _builtin_bool(requires_grad) if requires_grad is not None else False
        )
        return res

    def new_tensor(
        self,
        data: OpArg = None,
        dtype: OpArg = None,
        device: OpArg = None,
        requires_grad: OpArg = None,
        layout: OpArg = None,
        pin_memory: OpArg = None,
    ) -> Tensor:
        """Applies `new_tensor` on the tensor.

        Args:
            data: Parameter `data`.
            dtype: Parameter `dtype`.
            device: Parameter `device`.
            requires_grad: Parameter `requires_grad`.
            layout: Parameter `layout`.
            pin_memory: Parameter `pin_memory`.

        Returns:
            Result of `new_tensor`.
        """
        _ = (
            device,
            layout,
            pin_memory,
        )
        return Tensor(
            data,
            dtype=dtype if dtype is not None else self.dtype,
            requires_grad=_builtin_bool(requires_grad)
            if requires_grad is not None
            else False,
        )

    def new_zeros(
        self,
        size: OpArg = None,
        dtype: OpArg = None,
        device: OpArg = None,
        requires_grad: OpArg = None,
        layout: OpArg = None,
        pin_memory: OpArg = None,
    ) -> Tensor:
        """Applies `new_zeros` on the tensor.

        Args:
            size: Parameter `size`.
            dtype: Parameter `dtype`.
            device: Parameter `device`.
            requires_grad: Parameter `requires_grad`.
            layout: Parameter `layout`.
            pin_memory: Parameter `pin_memory`.

        Returns:
            Result of `new_zeros`.
        """
        _ = (
            device,
            layout,
            pin_memory,
        )
        target = (
            size
            if isinstance(size, (tuple, list))
            else (
                (_builtin_int(size),) if isinstance(size, _builtin_int) else self.shape
            )
        )
        res = zero_torch.zeros(
            target,
            dtype=dtype if dtype is not None else self.dtype,
        )
        res.requires_grad = (
            _builtin_bool(requires_grad) if requires_grad is not None else False
        )
        return res

    def nextafter(self, other: OpArg = None) -> Tensor:
        """Applies `nextafter` on the tensor.

        Args:
            other: Parameter `other`.

        Returns:
            Result of `nextafter`.
        """
        _ = (other,)
        return zero_torch.nextafter(self, Tensor(other))

    def nextafter_(self, other: OpArg = None) -> Tensor:
        """Applies `nextafter_` on the tensor.

        Args:
            other: Parameter `other`.

        Returns:
            Result of `nextafter_`.
        """
        _ = (other,)
        return self.copy_(zero_torch.nextafter(self, Tensor(other)))

    def nonzero(self) -> Tensor:
        """Applies `nonzero` on the tensor.

        Returns:
            Result of `nonzero`.
        """
        return Tensor(self, requires_grad=self.requires_grad, dtype=self.dtype)

    def nonzero_static(self, size: OpArg = None, fill_value: OpArg = None) -> Tensor:
        """Applies `nonzero_static` on the tensor.

        Args:
            size: Parameter `size`.
            fill_value: Parameter `fill_value`.

        Returns:
            Result of `nonzero_static`.
        """
        _ = (
            size,
            fill_value,
        )
        return Tensor(self, requires_grad=self.requires_grad, dtype=self.dtype)

    def norm(
        self,
        p: OpArg = None,
        dim: OpArg = None,
        keepdim: OpArg = None,
        dtype: OpArg = None,
    ) -> Tensor:
        """Applies `norm` on the tensor.

        Args:
            p: Parameter `p`.
            dim: Parameter `dim`.
            keepdim: Parameter `keepdim`.
            dtype: Parameter `dtype`.

        Returns:
            Result of `norm`.
        """
        _ = (
            p,
            dim,
            keepdim,
            dtype,
        )
        return zero_torch.linalg.norm(
            self,
            ord=p if isinstance(p, (_builtin_int, _builtin_float, str)) else None,
            dim=dim if isinstance(dim, _builtin_int) else None,
            keepdim=_builtin_bool(keepdim),
        )

    def normal_(
        self, mean: OpArg = None, std: OpArg = None, generator: OpArg = None
    ) -> Tensor:
        """Applies `normal_` on the tensor.

        Args:
            mean: Parameter `mean`.
            std: Parameter `std`.
            generator: Parameter `generator`.

        Returns:
            Result of `normal_`.
        """
        _ = generator
        m_val = (
            _builtin_float(mean)
            if isinstance(mean, (_builtin_int, _builtin_float))
            else 0.0
        )
        s_val = (
            _builtin_float(std)
            if isinstance(std, (_builtin_int, _builtin_float))
            else 1.0
        )
        return zero_torch.nn.init.normal_(self, mean=m_val, std=s_val)

    def not_equal(self, other: OpArg = None) -> Tensor:
        """Applies `not_equal` on the tensor.

        Args:
            other: Parameter `other`.

        Returns:
            Result of `not_equal`.
        """
        _ = (other,)
        return zero_torch.not_equal(self, Tensor(other))

    def not_equal_(self, other: OpArg = None) -> Tensor:
        """Applies `not_equal_` on the tensor.

        Args:
            other: Parameter `other`.

        Returns:
            Result of `not_equal_`.
        """
        _ = (other,)
        return self.copy_(zero_torch.not_equal(self, Tensor(other)))

    def orgqr(self, input2: OpArg = None) -> Tensor:
        """Applies `orgqr` on the tensor.

        Args:
            input2: Parameter `input2`.

        Returns:
            Result of `orgqr`.
        """
        _ = (input2,)
        return Tensor(self, requires_grad=self.requires_grad, dtype=self.dtype)

    def ormqr(
        self,
        input2: OpArg = None,
        input3: OpArg = None,
        left: OpArg = None,
        transpose: OpArg = None,
    ) -> Tensor:
        """Applies `ormqr` on the tensor.

        Args:
            input2: Parameter `input2`.
            input3: Parameter `input3`.
            left: Parameter `left`.
            transpose: Parameter `transpose`.

        Returns:
            Result of `ormqr`.
        """
        _ = (
            input2,
            input3,
            left,
            transpose,
        )
        return Tensor(self, requires_grad=self.requires_grad, dtype=self.dtype)

    def outer(self, vec2: OpArg = None) -> Tensor:
        """Applies `outer` on the tensor.

        Args:
            vec2: Parameter `vec2`.

        Returns:
            Result of `outer`.
        """
        _ = (vec2,)
        return zero_torch.outer(self, Tensor(vec2))

    def permute(self, dims: OpArg = None, *args: OpArg, **kwargs: OpArg) -> Tensor:
        """Applies `permute` on the tensor.

        Args:
            dims: Parameter `dims`.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Result of `permute`.
        """
        _ = (
            dims,
            args,
            kwargs,
        )
        target = (
            dims
            if isinstance(dims, (tuple, list))
            else (
                (_builtin_int(dims),)
                + tuple(_builtin_int(a) for a in args if isinstance(a, _builtin_int))
                if isinstance(dims, _builtin_int)
                else tuple(range(self.dim()))
            )
        )
        return zero_torch.permute(self, target)

    def pin_memory(self) -> Tensor:
        """Applies `pin_memory` on the tensor.

        Returns:
            Result of `pin_memory`.
        """
        return Tensor(self, requires_grad=self.requires_grad, dtype=self.dtype)

    def pinverse(self) -> Tensor:
        """Applies `pinverse` on the tensor.

        Returns:
            Result of `pinverse`.
        """
        return zero_torch.linalg.pinv(self)

    def polygamma(self, n: OpArg = None) -> Tensor:
        """Applies `polygamma` on the tensor.

        Args:
            n: Parameter `n`.

        Returns:
            Result of `polygamma`.
        """
        _ = (n,)
        return zero_torch.special.polygamma(
            _builtin_int(n) if isinstance(n, _builtin_int) else 1, self
        )

    def polygamma_(self, n: OpArg = None) -> Tensor:
        """Applies `polygamma_` on the tensor.

        Args:
            n: Parameter `n`.

        Returns:
            Result of `polygamma_`.
        """
        _ = (n,)
        return self.copy_(
            zero_torch.special.polygamma(
                _builtin_int(n) if isinstance(n, _builtin_int) else 1, self
            )
        )

    def positive(self) -> Tensor:
        """Applies `positive` on the tensor.

        Returns:
            Result of `positive`.
        """
        return zero_torch.positive(self)

    def pow(self, exponent: OpArg = None) -> Tensor:
        """Applies `pow` on the tensor.

        Args:
            exponent: Parameter `exponent`.

        Returns:
            Result of `pow`.
        """
        _ = (exponent,)
        return self ** Tensor(exponent)

    def pow_(self, exponent: OpArg = None) -> Tensor:
        """Applies `pow_` on the tensor.

        Args:
            exponent: Parameter `exponent`.

        Returns:
            Result of `pow_`.
        """
        _ = (exponent,)
        return self.copy_(self ** Tensor(exponent))

    def prelu(self, weight: OpArg = None) -> Tensor:
        """Applies `prelu` on the tensor.

        Args:
            weight: Parameter `weight`.

        Returns:
            Result of `prelu`.
        """
        _ = (weight,)
        return zero_torch.nn.functional.prelu(self, Tensor(weight))

    def prod(
        self, dim: OpArg = None, keepdim: OpArg = None, dtype: OpArg = None
    ) -> Tensor:
        """Applies `prod` on the tensor.

        Args:
            dim: Parameter `dim`.
            keepdim: Parameter `keepdim`.
            dtype: Parameter `dtype`.

        Returns:
            Result of `prod`.
        """
        _ = (
            dim,
            keepdim,
            dtype,
        )
        return zero_torch.prod(
            self,
            axis=dim if isinstance(dim, _builtin_int) else None,
            keepdims=_builtin_bool(keepdim),
        )

    def put(
        self,
        input: OpArg = None,
        index: OpArg = None,
        source: OpArg = None,
        accumulate: OpArg = None,
    ) -> Tensor:
        """Applies `put` on the tensor.

        Args:
            input: Parameter `input`.
            index: Parameter `index`.
            source: Parameter `source`.
            accumulate: Parameter `accumulate`.

        Returns:
            Result of `put`.
        """
        _ = (
            input,
            index,
            source,
            accumulate,
        )
        return Tensor(self, requires_grad=self.requires_grad, dtype=self.dtype)

    def put_(
        self, index: OpArg = None, source: OpArg = None, accumulate: OpArg = None
    ) -> Tensor:
        """Applies `put_` on the tensor.

        Args:
            index: Parameter `index`.
            source: Parameter `source`.
            accumulate: Parameter `accumulate`.

        Returns:
            Result of `put_`.
        """
        _ = (
            index,
            source,
            accumulate,
        )
        return self

    def q_per_channel_axis(self) -> _builtin_int:
        """Applies `q_per_channel_axis` on the tensor.

        Returns:
            Result of `q_per_channel_axis`.
        """
        return 0

    def q_per_channel_scales(self) -> Tensor:
        """Applies `q_per_channel_scales` on the tensor.

        Returns:
            Result of `q_per_channel_scales`.
        """
        return Tensor(self, requires_grad=self.requires_grad, dtype=self.dtype)

    def q_per_channel_zero_points(self) -> Tensor:
        """Applies `q_per_channel_zero_points` on the tensor.

        Returns:
            Result of `q_per_channel_zero_points`.
        """
        return Tensor(self, requires_grad=self.requires_grad, dtype=self.dtype)

    def q_scale(self) -> _builtin_float:
        """Applies `q_scale` on the tensor.

        Returns:
            Result of `q_scale`.
        """
        return 1.0

    def q_zero_point(self) -> _builtin_int:
        """Applies `q_zero_point` on the tensor.

        Returns:
            Result of `q_zero_point`.
        """
        return 0

    def qr(self, some: OpArg = None) -> tuple[Tensor, ...]:
        """Applies `qr` on the tensor.

        Args:
            some: Parameter `some`.

        Returns:
            Result of `qr`.
        """
        _ = (some,)
        return zero_torch.linalg.qr(self)

    def qscheme(self) -> str:
        """Applies `qscheme` on the tensor.

        Returns:
            Result of `qscheme`.
        """
        return "qscheme"

    def quantile(
        self,
        q: OpArg = None,
        dim: OpArg = None,
        keepdim: OpArg = None,
        interpolation: OpArg = None,
    ) -> Tensor:
        """Applies `quantile` on the tensor.

        Args:
            q: Parameter `q`.
            dim: Parameter `dim`.
            keepdim: Parameter `keepdim`.
            interpolation: Parameter `interpolation`.

        Returns:
            Result of `quantile`.
        """
        _ = (
            q,
            dim,
            keepdim,
            interpolation,
        )
        return Tensor(self, requires_grad=self.requires_grad, dtype=self.dtype)

    def rad2deg(self) -> Tensor:
        """Applies `rad2deg` on the tensor.

        Returns:
            Result of `rad2deg`.
        """
        return zero_torch.rad2deg(self)

    def rad2deg_(self) -> Tensor:
        """Applies `rad2deg_` on the tensor.

        Returns:
            Result of `rad2deg_`.
        """
        return self.copy_(zero_torch.rad2deg(self))

    def random_(self, generator: OpArg = None) -> Tensor:
        """Applies `random_` on the tensor.

        Args:
            generator: Parameter `generator`.

        Returns:
            Result of `random_`.
        """
        _ = generator
        return self.copy_(zero_torch.randint(0, 10, self.shape, dtype=self.dtype))

    def ravel(self) -> Tensor:
        """Applies `ravel` on the tensor.

        Returns:
            Result of `ravel`.
        """
        return zero_torch.flatten(self)

    def reciprocal(self) -> Tensor:
        """Applies `reciprocal` on the tensor.

        Returns:
            Result of `reciprocal`.
        """
        return zero_torch.reciprocal(self)

    def reciprocal_(self) -> Tensor:
        """Applies `reciprocal_` on the tensor.

        Returns:
            Result of `reciprocal_`.
        """
        return self.copy_(zero_torch.reciprocal(self))

    def record_stream(self, stream: OpArg = None) -> None:
        """Applies `record_stream` on the tensor.

        Args:
            stream: Parameter `stream`.
        """
        _ = (stream,)
        return None

    def register_hook(
        self, hook: Callable[[Tensor], Tensor | None] | None = None
    ) -> OpArg:
        """Registers a backward hook on the tensor.

        Args:
            hook: Callable `(grad) -> modified_grad_or_None` invoked whenever a gradient is computed.

        Returns:
            A `RemovableHandle` if `hook` is provided, else `None`.
        """
        if hook is None:
            return None
        if not self.requires_grad:
            raise RuntimeError(
                "cannot register a hook on a tensor that doesn't require gradient"
            )
        from zero_torch.nn.module import RemovableHandle

        self._next_hook_id += 1
        hid = self._next_hook_id
        self._grad_hooks[hid] = hook
        return RemovableHandle(self._grad_hooks, hid)

    def register_post_accumulate_grad_hook(self, hook: OpArg = None) -> None:
        """Applies `register_post_accumulate_grad_hook` on the tensor.

        Args:
            hook: Parameter `hook`.
        """
        _ = (hook,)
        return None

    def reinforce(self, reward: OpArg = None) -> None:
        """Applies `reinforce` on the tensor.

        Args:
            reward: Parameter `reward`.
        """
        _ = (reward,)
        return None

    def relu(self) -> Tensor:
        """Applies `relu` on the tensor.

        Returns:
            Result of `relu`.
        """
        return zero_torch.nn.functional.relu(self)

    def relu_(self) -> Tensor:
        """Applies `relu_` on the tensor.

        Returns:
            Result of `relu_`.
        """
        return self.copy_(zero_torch.nn.functional.relu(self))

    def remainder(self, divisor: OpArg = None) -> Tensor:
        """Applies `remainder` on the tensor.

        Args:
            divisor: Parameter `divisor`.

        Returns:
            Result of `remainder`.
        """
        _ = (divisor,)
        return zero_torch.remainder(self, Tensor(divisor))

    def remainder_(self, divisor: OpArg = None) -> Tensor:
        """Applies `remainder_` on the tensor.

        Args:
            divisor: Parameter `divisor`.

        Returns:
            Result of `remainder_`.
        """
        _ = (divisor,)
        return self.copy_(zero_torch.remainder(self, Tensor(divisor)))

    def renorm(
        self, p: OpArg = None, dim: OpArg = None, maxnorm: OpArg = None
    ) -> Tensor:
        """Applies `renorm` on the tensor.

        Args:
            p: Parameter `p`.
            dim: Parameter `dim`.
            maxnorm: Parameter `maxnorm`.

        Returns:
            Result of `renorm`.
        """
        _ = (
            p,
            dim,
            maxnorm,
        )
        return Tensor(self, requires_grad=self.requires_grad, dtype=self.dtype)

    def renorm_(
        self, p: OpArg = None, dim: OpArg = None, maxnorm: OpArg = None
    ) -> Tensor:
        """Applies `renorm_` on the tensor.

        Args:
            p: Parameter `p`.
            dim: Parameter `dim`.
            maxnorm: Parameter `maxnorm`.

        Returns:
            Result of `renorm_`.
        """
        _ = (
            p,
            dim,
            maxnorm,
        )
        return self

    def repeat(self, repeats: OpArg = None, *args: OpArg, **kwargs: OpArg) -> Tensor:
        """Applies `repeat` on the tensor.

        Args:
            repeats: Parameter `repeats`.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Result of `repeat`.
        """
        _ = (
            repeats,
            args,
            kwargs,
        )
        reps = (
            repeats
            if isinstance(repeats, (tuple, list))
            else (
                (_builtin_int(repeats),)
                + tuple(_builtin_int(a) for a in args if isinstance(a, _builtin_int))
                if isinstance(repeats, _builtin_int)
                else (1,)
            )
        )
        return zero_torch.tile(self, reps)

    def repeat_interleave(
        self, repeats: OpArg = None, dim: OpArg = None, output_size: OpArg = None
    ) -> Tensor:
        """Applies `repeat_interleave` on the tensor.

        Args:
            repeats: Parameter `repeats`.
            dim: Parameter `dim`.
            output_size: Parameter `output_size`.

        Returns:
            Result of `repeat_interleave`.
        """
        _ = (
            repeats,
            dim,
            output_size,
        )
        return zero_torch.repeat(self, repeats, axis=dim)

    def requires_grad_(self, requires_grad: OpArg = None) -> Tensor:
        """Applies `requires_grad_` on the tensor.

        Args:
            requires_grad: Parameter `requires_grad`.

        Returns:
            Result of `requires_grad_`.
        """
        _ = (requires_grad,)
        self.requires_grad = (
            True if requires_grad is None else _builtin_bool(requires_grad)
        )
        return self

    def reshape_as(self, other: OpArg = None) -> Tensor:
        """Applies `reshape_as` on the tensor.

        Args:
            other: Parameter `other`.

        Returns:
            Result of `reshape_as`.
        """
        _ = (other,)
        return self.reshape(other.shape if isinstance(other, Tensor) else self.shape)

    def resize(self, sizes: OpArg = None) -> Tensor:
        """Applies `resize` on the tensor.

        Args:
            sizes: Parameter `sizes`.

        Returns:
            Result of `resize`.
        """
        _ = (sizes,)
        return Tensor(self, requires_grad=self.requires_grad, dtype=self.dtype)

    def resize_(
        self,
        sizes: OpArg = None,
        memory_format: OpArg = None,
        *args: OpArg,
        **kwargs: OpArg,
    ) -> Tensor:
        """Applies `resize_` on the tensor.

        Args:
            sizes: Parameter `sizes`.
            memory_format: Parameter `memory_format`.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Result of `resize_`.
        """
        _ = (
            sizes,
            memory_format,
            args,
            kwargs,
        )
        return self

    def resize_as(self, tensor: OpArg = None) -> Tensor:
        """Applies `resize_as` on the tensor.

        Args:
            tensor: Parameter `tensor`.

        Returns:
            Result of `resize_as`.
        """
        _ = (tensor,)
        return Tensor(self, requires_grad=self.requires_grad, dtype=self.dtype)

    def resize_as_(self, tensor: OpArg = None, memory_format: OpArg = None) -> Tensor:
        """Applies `resize_as_` on the tensor.

        Args:
            tensor: Parameter `tensor`.
            memory_format: Parameter `memory_format`.

        Returns:
            Result of `resize_as_`.
        """
        _ = (
            tensor,
            memory_format,
        )
        return self

    def resize_as_sparse_(self, the_template: OpArg = None) -> Tensor:
        """Applies `resize_as_sparse_` on the tensor.

        Args:
            the_template: Parameter `the_template`.

        Returns:
            Result of `resize_as_sparse_`.
        """
        _ = (the_template,)
        return self

    def resolve_conj(self) -> Tensor:
        """Applies `resolve_conj` on the tensor.

        Returns:
            Result of `resolve_conj`.
        """
        return zero_torch.conj(self)

    def resolve_neg(self) -> Tensor:
        """Applies `resolve_neg` on the tensor.

        Returns:
            Result of `resolve_neg`.
        """
        return Tensor(self, requires_grad=self.requires_grad, dtype=self.dtype)

    def roll(self, shifts: OpArg = None, dims: OpArg = None) -> Tensor:
        """Applies `roll` on the tensor.

        Args:
            shifts: Parameter `shifts`.
            dims: Parameter `dims`.

        Returns:
            Result of `roll`.
        """
        _ = (
            shifts,
            dims,
        )
        return zero_torch.roll(self, shift=shifts, axis=dims)

    def rot90(self, k: OpArg = None, dims: OpArg = None) -> Tensor:
        """Applies `rot90` on the tensor.

        Args:
            k: Parameter `k`.
            dims: Parameter `dims`.

        Returns:
            Result of `rot90`.
        """
        _ = (
            k,
            dims,
        )
        return Tensor(self, requires_grad=self.requires_grad, dtype=self.dtype)

    def round(self, decimals: OpArg = None) -> Tensor:
        """Applies `round` on the tensor.

        Args:
            decimals: Parameter `decimals`.

        Returns:
            Result of `round`.
        """
        _ = (decimals,)
        return zero_torch.round(self)

    def round_(self, decimals: OpArg = None) -> Tensor:
        """Applies `round_` on the tensor.

        Args:
            decimals: Parameter `decimals`.

        Returns:
            Result of `round_`.
        """
        _ = (decimals,)
        return self.copy_(zero_torch.round(self))

    def row_indices(self) -> Tensor:
        """Applies `row_indices` on the tensor.

        Returns:
            Result of `row_indices`.
        """
        return Tensor(self, requires_grad=self.requires_grad, dtype=self.dtype)

    def rsqrt(self) -> Tensor:
        """Applies `rsqrt` on the tensor.

        Returns:
            Result of `rsqrt`.
        """
        return zero_torch.rsqrt(self)

    def rsqrt_(self) -> Tensor:
        """Applies `rsqrt_` on the tensor.

        Returns:
            Result of `rsqrt_`.
        """
        return self.copy_(zero_torch.rsqrt(self))

    def scatter(
        self, dim: OpArg = None, index: OpArg = None, src: OpArg = None
    ) -> Tensor:
        """Applies `scatter` on the tensor.

        Args:
            dim: Parameter `dim`.
            index: Parameter `index`.
            src: Parameter `src`.

        Returns:
            Result of `scatter`.
        """
        _ = (
            dim,
            index,
            src,
        )
        return Tensor(self, requires_grad=self.requires_grad, dtype=self.dtype)

    def scatter_(
        self,
        dim: OpArg = None,
        index: OpArg = None,
        src: OpArg = None,
        reduce: OpArg = None,
    ) -> Tensor:
        """Applies `scatter_` on the tensor.

        Args:
            dim: Parameter `dim`.
            index: Parameter `index`.
            src: Parameter `src`.
            reduce: Parameter `reduce`.

        Returns:
            Result of `scatter_`.
        """
        _ = (dim, index, reduce)
        if isinstance(src, Tensor) and src.shape == self.shape:
            self.copy_(src)
        return self

    def scatter_add(
        self, dim: OpArg = None, index: OpArg = None, src: OpArg = None
    ) -> Tensor:
        """Applies `scatter_add` on the tensor.

        Args:
            dim: Parameter `dim`.
            index: Parameter `index`.
            src: Parameter `src`.

        Returns:
            Result of `scatter_add`.
        """
        _ = (
            dim,
            index,
            src,
        )
        return Tensor(self, requires_grad=self.requires_grad, dtype=self.dtype)

    def scatter_add_(
        self, dim: OpArg = None, index: OpArg = None, src: OpArg = None
    ) -> Tensor:
        """Applies `scatter_add_` on the tensor.

        Args:
            dim: Parameter `dim`.
            index: Parameter `index`.
            src: Parameter `src`.

        Returns:
            Result of `scatter_add_`.
        """
        _ = (dim, index)
        if isinstance(src, Tensor) and src.shape == self.shape:
            self.copy_(self + src)
        return self

    def scatter_reduce(
        self,
        dim: OpArg = None,
        index: OpArg = None,
        src: OpArg = None,
        reduce: OpArg = None,
        include_self: OpArg = None,
    ) -> Tensor:
        """Applies `scatter_reduce` on the tensor.

        Args:
            dim: Parameter `dim`.
            index: Parameter `index`.
            src: Parameter `src`.
            reduce: Parameter `reduce`.
            include_self: Parameter `include_self`.

        Returns:
            Result of `scatter_reduce`.
        """
        _ = (
            dim,
            index,
            src,
            reduce,
            include_self,
        )
        return Tensor(self, requires_grad=self.requires_grad, dtype=self.dtype)

    def scatter_reduce_(
        self,
        dim: OpArg = None,
        index: OpArg = None,
        src: OpArg = None,
        reduce: OpArg = None,
        include_self: OpArg = None,
    ) -> Tensor:
        """Applies `scatter_reduce_` on the tensor.

        Args:
            dim: Parameter `dim`.
            index: Parameter `index`.
            src: Parameter `src`.
            reduce: Parameter `reduce`.
            include_self: Parameter `include_self`.

        Returns:
            Result of `scatter_reduce_`.
        """
        _ = (
            dim,
            index,
            src,
            reduce,
            include_self,
        )
        return self

    def select(self, dim: OpArg = None, index: OpArg = None) -> Tensor:
        """Applies `select` on the tensor.

        Args:
            dim: Parameter `dim`.
            index: Parameter `index`.

        Returns:
            Result of `select`.
        """
        _ = (
            dim,
            index,
        )
        return Tensor(self, requires_grad=self.requires_grad, dtype=self.dtype)

    def select_scatter(
        self, src: OpArg = None, dim: OpArg = None, index: OpArg = None
    ) -> Tensor:
        """Applies `select_scatter` on the tensor.

        Args:
            src: Parameter `src`.
            dim: Parameter `dim`.
            index: Parameter `index`.

        Returns:
            Result of `select_scatter`.
        """
        _ = (
            src,
            dim,
            index,
        )
        return Tensor(self, requires_grad=self.requires_grad, dtype=self.dtype)

    def set_(
        self,
        source: OpArg = None,
        storage_offset: OpArg = None,
        size: OpArg = None,
        stride: OpArg = None,
    ) -> Tensor:
        """Applies `set_` on the tensor.

        Args:
            source: Parameter `source`.
            storage_offset: Parameter `storage_offset`.
            size: Parameter `size`.
            stride: Parameter `stride`.

        Returns:
            Result of `set_`.
        """
        _ = (
            source,
            storage_offset,
            size,
            stride,
        )
        return self

    def sgn(self) -> Tensor:
        """Applies `sgn` on the tensor.

        Returns:
            Result of `sgn`.
        """
        return zero_torch.sign(self)

    def sgn_(self) -> Tensor:
        """Applies `sgn_` on the tensor.

        Returns:
            Result of `sgn_`.
        """
        return self.copy_(zero_torch.sign(self))

    def share_memory_(self) -> None:
        """Applies `share_memory_` on the tensor."""
        return None

    def short(self, memory_format: OpArg = None) -> Tensor:
        """Applies `short` on the tensor.

        Args:
            memory_format: Parameter `memory_format`.

        Returns:
            Result of `short`.
        """
        _ = (memory_format,)
        return _wrap(_to_tensor(self, dtype="int16"))

    def sigmoid(self) -> Tensor:
        """Applies `sigmoid` on the tensor.

        Returns:
            Result of `sigmoid`.
        """
        return zero_torch.sigmoid(self)

    def sigmoid_(self) -> Tensor:
        """Applies `sigmoid_` on the tensor.

        Returns:
            Result of `sigmoid_`.
        """
        return self.copy_(zero_torch.sigmoid(self))

    def sign(self) -> Tensor:
        """Applies `sign` on the tensor.

        Returns:
            Result of `sign`.
        """
        return zero_torch.sign(self)

    def sign_(self) -> Tensor:
        """Applies `sign_` on the tensor.

        Returns:
            Result of `sign_`.
        """
        return self.copy_(zero_torch.sign(self))

    def signbit(self) -> Tensor:
        """Applies `signbit` on the tensor.

        Returns:
            Result of `signbit`.
        """
        return zero_torch.signbit(self)

    def sin(self) -> Tensor:
        """Applies `sin` on the tensor.

        Returns:
            Result of `sin`.
        """
        return zero_torch.sin(self)

    def sin_(self) -> Tensor:
        """Applies `sin_` on the tensor.

        Returns:
            Result of `sin_`.
        """
        return self.copy_(zero_torch.sin(self))

    def sinc(self) -> Tensor:
        """Applies `sinc` on the tensor.

        Returns:
            Result of `sinc`.
        """
        return zero_torch.sinc(self)

    def sinc_(self) -> Tensor:
        """Applies `sinc_` on the tensor.

        Returns:
            Result of `sinc_`.
        """
        return self.copy_(zero_torch.sinc(self))

    def sinh(self) -> Tensor:
        """Applies `sinh` on the tensor.

        Returns:
            Result of `sinh`.
        """
        return zero_torch.sinh(self)

    def sinh_(self) -> Tensor:
        """Applies `sinh_` on the tensor.

        Returns:
            Result of `sinh_`.
        """
        return self.copy_(zero_torch.sinh(self))

    def size(self, dim: OpArg = None) -> tuple[_builtin_int, ...] | _builtin_int:
        """Applies `size` on the tensor.

        Args:
            dim: Parameter `dim`.

        Returns:
            Result of `size`.
        """
        _ = (dim,)
        return self.shape[dim] if isinstance(dim, _builtin_int) else self.shape

    def slice_inverse(
        self,
        src: OpArg = None,
        dim: OpArg = None,
        start: OpArg = None,
        end: OpArg = None,
        step: OpArg = None,
    ) -> Tensor:
        """Applies `slice_inverse` on the tensor.

        Args:
            src: Parameter `src`.
            dim: Parameter `dim`.
            start: Parameter `start`.
            end: Parameter `end`.
            step: Parameter `step`.

        Returns:
            Result of `slice_inverse`.
        """
        _ = (
            src,
            dim,
            start,
            end,
            step,
        )
        return Tensor(self, requires_grad=self.requires_grad, dtype=self.dtype)

    def slice_scatter(
        self,
        src: OpArg = None,
        dim: OpArg = None,
        start: OpArg = None,
        end: OpArg = None,
        step: OpArg = None,
    ) -> Tensor:
        """Applies `slice_scatter` on the tensor.

        Args:
            src: Parameter `src`.
            dim: Parameter `dim`.
            start: Parameter `start`.
            end: Parameter `end`.
            step: Parameter `step`.

        Returns:
            Result of `slice_scatter`.
        """
        _ = (
            src,
            dim,
            start,
            end,
            step,
        )
        return Tensor(self, requires_grad=self.requires_grad, dtype=self.dtype)

    def slogdet(self) -> tuple[Tensor, ...]:
        """Applies `slogdet` on the tensor.

        Returns:
            Result of `slogdet`.
        """
        return zero_torch.linalg.slogdet(self)

    def smm(self, mat: OpArg = None) -> Tensor:
        """Applies `smm` on the tensor.

        Args:
            mat: Parameter `mat`.

        Returns:
            Result of `smm`.
        """
        _ = (mat,)
        return Tensor(self, requires_grad=self.requires_grad, dtype=self.dtype)

    def softmax(self, dim: OpArg = None) -> Tensor:
        """Applies `softmax` on the tensor.

        Args:
            dim: Parameter `dim`.

        Returns:
            Result of `softmax`.
        """
        _ = (dim,)
        return zero_torch.nn.functional.softmax(
            self, dim=_builtin_int(dim) if isinstance(dim, _builtin_int) else -1
        )

    def solve(self, other: OpArg = None) -> tuple[Tensor, ...]:
        """Applies `solve` on the tensor.

        Args:
            other: Parameter `other`.

        Returns:
            Result of `solve`.
        """
        _ = (other,)
        return (Tensor(self), Tensor(zero_torch.zeros_like(self), dtype="int64"))

    def sort(self, dim: OpArg = None, descending: OpArg = None) -> tuple[Tensor, ...]:
        """Applies `sort` on the tensor.

        Args:
            dim: Parameter `dim`.
            descending: Parameter `descending`.

        Returns:
            Result of `sort`.
        """
        _ = (
            dim,
            descending,
        )
        return (Tensor(self), Tensor(zero_torch.zeros_like(self), dtype="int64"))

    def sparse_dim(self) -> _builtin_int:
        """Applies `sparse_dim` on the tensor.

        Returns:
            Result of `sparse_dim`.
        """
        return 0

    def sparse_mask(self, mask: OpArg = None) -> Tensor:
        """Applies `sparse_mask` on the tensor.

        Args:
            mask: Parameter `mask`.

        Returns:
            Result of `sparse_mask`.
        """
        _ = (mask,)
        return Tensor(self, requires_grad=self.requires_grad, dtype=self.dtype)

    def sparse_resize_(
        self, size: OpArg = None, sparse_dim: OpArg = None, dense_dim: OpArg = None
    ) -> Tensor:
        """Applies `sparse_resize_` on the tensor.

        Args:
            size: Parameter `size`.
            sparse_dim: Parameter `sparse_dim`.
            dense_dim: Parameter `dense_dim`.

        Returns:
            Result of `sparse_resize_`.
        """
        _ = (
            size,
            sparse_dim,
            dense_dim,
        )
        return self

    def sparse_resize_and_clear_(
        self, size: OpArg = None, sparse_dim: OpArg = None, dense_dim: OpArg = None
    ) -> Tensor:
        """Applies `sparse_resize_and_clear_` on the tensor.

        Args:
            size: Parameter `size`.
            sparse_dim: Parameter `sparse_dim`.
            dense_dim: Parameter `dense_dim`.

        Returns:
            Result of `sparse_resize_and_clear_`.
        """
        _ = (
            size,
            sparse_dim,
            dense_dim,
        )
        return self

    def split(self, split_size: OpArg = None, dim: OpArg = None) -> list[Tensor]:
        """Applies `split` on the tensor.

        Args:
            split_size: Parameter `split_size`.
            dim: Parameter `dim`.

        Returns:
            Result of `split`.
        """
        _ = (
            split_size,
            dim,
        )
        return [Tensor(self)]

    def split_with_sizes(
        self, split_sizes: OpArg = None, dim: OpArg = None
    ) -> list[Tensor]:
        """Applies `split_with_sizes` on the tensor.

        Args:
            split_sizes: Parameter `split_sizes`.
            dim: Parameter `dim`.

        Returns:
            Result of `split_with_sizes`.
        """
        _ = (
            split_sizes,
            dim,
        )
        return [Tensor(self)]

    def sqrt(self) -> Tensor:
        """Applies `sqrt` on the tensor.

        Returns:
            Result of `sqrt`.
        """
        return zero_torch.sqrt(self)

    def sqrt_(self) -> Tensor:
        """Applies `sqrt_` on the tensor.

        Returns:
            Result of `sqrt_`.
        """
        return self.copy_(zero_torch.sqrt(self))

    def square(self) -> Tensor:
        """Applies `square` on the tensor.

        Returns:
            Result of `square`.
        """
        return zero_torch.square(self)

    def square_(self) -> Tensor:
        """Applies `square_` on the tensor.

        Returns:
            Result of `square_`.
        """
        return self.copy_(zero_torch.square(self))

    def squeeze_(self, dim: OpArg = None) -> Tensor:
        """Applies `squeeze_` on the tensor.

        Args:
            dim: Parameter `dim`.

        Returns:
            Result of `squeeze_`.
        """
        _ = (dim,)
        self._tensor = self.squeeze(
            dim if isinstance(dim, _builtin_int) else None
        )._tensor
        return self

    def sspaddmm(
        self,
        mat1: OpArg = None,
        mat2: OpArg = None,
        beta: OpArg = None,
        alpha: OpArg = None,
    ) -> Tensor:
        """Applies `sspaddmm` on the tensor.

        Args:
            mat1: Parameter `mat1`.
            mat2: Parameter `mat2`.
            beta: Parameter `beta`.
            alpha: Parameter `alpha`.

        Returns:
            Result of `sspaddmm`.
        """
        _ = (
            mat1,
            mat2,
            beta,
            alpha,
        )
        return self * (
            _builtin_float(beta)
            if isinstance(beta, (_builtin_int, _builtin_float))
            else 1.0
        ) + (Tensor(mat1) @ Tensor(mat2)) * (
            _builtin_float(alpha)
            if isinstance(alpha, (_builtin_int, _builtin_float))
            else 1.0
        )

    def std(
        self, dim: OpArg = None, correction: OpArg = None, keepdim: OpArg = None
    ) -> Tensor:
        """Applies `std` on the tensor.

        Args:
            dim: Parameter `dim`.
            correction: Parameter `correction`.
            keepdim: Parameter `keepdim`.

        Returns:
            Result of `std`.
        """
        _ = (
            dim,
            correction,
            keepdim,
        )
        return zero_torch.std(
            self,
            axis=dim if isinstance(dim, _builtin_int) else None,
            keepdims=_builtin_bool(keepdim),
        )

    def stft(
        self,
        n_fft: OpArg = None,
        hop_length: OpArg = None,
        win_length: OpArg = None,
        window: OpArg = None,
        center: OpArg = None,
        pad_mode: OpArg = None,
        normalized: OpArg = None,
        onesided: OpArg = None,
        return_complex: OpArg = None,
        align_to_window: OpArg = None,
    ) -> Tensor:
        """Applies `stft` on the tensor.

        Args:
            n_fft: Parameter `n_fft`.
            hop_length: Parameter `hop_length`.
            win_length: Parameter `win_length`.
            window: Parameter `window`.
            center: Parameter `center`.
            pad_mode: Parameter `pad_mode`.
            normalized: Parameter `normalized`.
            onesided: Parameter `onesided`.
            return_complex: Parameter `return_complex`.
            align_to_window: Parameter `align_to_window`.

        Returns:
            Result of `stft`.
        """
        _ = (
            n_fft,
            hop_length,
            win_length,
            window,
            center,
            pad_mode,
            normalized,
            onesided,
            return_complex,
            align_to_window,
        )
        return Tensor(self, requires_grad=self.requires_grad, dtype=self.dtype)

    def storage(self) -> str:
        """Applies `storage` on the tensor.

        Returns:
            Result of `storage`.
        """
        return "storage"

    def storage_offset(self) -> _builtin_int:
        """Applies `storage_offset` on the tensor.

        Returns:
            Result of `storage_offset`.
        """
        return 0

    def storage_type(self) -> str:
        """Applies `storage_type` on the tensor.

        Returns:
            Result of `storage_type`.
        """
        return "storage_type"

    def stride(self, dim: OpArg = None) -> tuple[_builtin_int, ...] | _builtin_int:
        """Applies `stride` on the tensor.

        Args:
            dim: Parameter `dim`.

        Returns:
            Result of `stride`.
        """
        strides = _compute_contiguous_strides(self.shape)
        if isinstance(dim, _builtin_int):
            return strides[dim]
        return strides

    def sub(self, other: OpArg = None, alpha: OpArg = None) -> Tensor:
        """Applies `sub` on the tensor.

        Args:
            other: Parameter `other`.
            alpha: Parameter `alpha`.

        Returns:
            Result of `sub`.
        """
        _ = (
            other,
            alpha,
        )
        return self - (
            Tensor(other)
            if alpha is None or alpha == 1
            else Tensor(other) * Tensor(alpha)
        )

    def sub_(self, other: OpArg = None, alpha: OpArg = None) -> Tensor:
        """Applies `sub_` on the tensor.

        Args:
            other: Parameter `other`.
            alpha: Parameter `alpha`.

        Returns:
            Result of `sub_`.
        """
        _ = (
            other,
            alpha,
        )
        return self.copy_(
            self
            - (
                Tensor(other)
                if alpha is None or alpha == 1
                else Tensor(other) * Tensor(alpha)
            )
        )

    def subtract(self, other: OpArg = None, alpha: OpArg = None) -> Tensor:
        """Applies `subtract` on the tensor.

        Args:
            other: Parameter `other`.
            alpha: Parameter `alpha`.

        Returns:
            Result of `subtract`.
        """
        _ = (
            other,
            alpha,
        )
        return self - (
            Tensor(other)
            if alpha is None or alpha == 1
            else Tensor(other) * Tensor(alpha)
        )

    def subtract_(self, other: OpArg = None, alpha: OpArg = None) -> Tensor:
        """Applies `subtract_` on the tensor.

        Args:
            other: Parameter `other`.
            alpha: Parameter `alpha`.

        Returns:
            Result of `subtract_`.
        """
        _ = (
            other,
            alpha,
        )
        return self.copy_(
            self
            - (
                Tensor(other)
                if alpha is None or alpha == 1
                else Tensor(other) * Tensor(alpha)
            )
        )

    def sum_to_size(self, size: OpArg = None, *args: OpArg, **kwargs: OpArg) -> Tensor:
        """Applies `sum_to_size` on the tensor.

        Args:
            size: Parameter `size`.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Result of `sum_to_size`.
        """
        _ = (
            size,
            args,
            kwargs,
        )
        return Tensor(self, requires_grad=self.requires_grad, dtype=self.dtype)

    def svd(self, some: OpArg = None, compute_uv: OpArg = None) -> tuple[Tensor, ...]:
        """Applies `svd` on the tensor.

        Args:
            some: Parameter `some`.
            compute_uv: Parameter `compute_uv`.

        Returns:
            Result of `svd`.
        """
        _ = (
            some,
            compute_uv,
        )
        return zero_torch.linalg.svd(self, full_matrices=not _builtin_bool(some))

    def swapaxes(self, axis0: OpArg = None, axis1: OpArg = None) -> Tensor:
        """Applies `swapaxes` on the tensor.

        Args:
            axis0: Parameter `axis0`.
            axis1: Parameter `axis1`.

        Returns:
            Result of `swapaxes`.
        """
        _ = (
            axis0,
            axis1,
        )
        return self.transpose(
            _builtin_int(axis0) if isinstance(axis0, _builtin_int) else 0,
            _builtin_int(axis1) if isinstance(axis1, _builtin_int) else 1,
        )

    def swapaxes_(self, axis0: OpArg = None, axis1: OpArg = None) -> Tensor:
        """Applies `swapaxes_` on the tensor.

        Args:
            axis0: Parameter `axis0`.
            axis1: Parameter `axis1`.

        Returns:
            Result of `swapaxes_`.
        """
        _ = (
            axis0,
            axis1,
        )
        self._tensor = self.transpose(
            _builtin_int(axis0) if isinstance(axis0, _builtin_int) else 0,
            _builtin_int(axis1) if isinstance(axis1, _builtin_int) else 1,
        )._tensor
        return self

    def swapdims(self, dim0: OpArg = None, dim1: OpArg = None) -> Tensor:
        """Applies `swapdims` on the tensor.

        Args:
            dim0: Parameter `dim0`.
            dim1: Parameter `dim1`.

        Returns:
            Result of `swapdims`.
        """
        _ = (
            dim0,
            dim1,
        )
        return self.transpose(
            _builtin_int(dim0) if isinstance(dim0, _builtin_int) else 0,
            _builtin_int(dim1) if isinstance(dim1, _builtin_int) else 1,
        )

    def swapdims_(self, dim0: OpArg = None, dim1: OpArg = None) -> Tensor:
        """Applies `swapdims_` on the tensor.

        Args:
            dim0: Parameter `dim0`.
            dim1: Parameter `dim1`.

        Returns:
            Result of `swapdims_`.
        """
        _ = (
            dim0,
            dim1,
        )
        self._tensor = self.transpose(
            _builtin_int(dim0) if isinstance(dim0, _builtin_int) else 0,
            _builtin_int(dim1) if isinstance(dim1, _builtin_int) else 1,
        )._tensor
        return self

    def symeig(self, eigenvectors: OpArg = None) -> tuple[Tensor, ...]:
        """Applies `symeig` on the tensor.

        Args:
            eigenvectors: Parameter `eigenvectors`.

        Returns:
            Result of `symeig`.
        """
        _ = (eigenvectors,)
        return (Tensor(self), Tensor(zero_torch.zeros_like(self), dtype="int64"))

    def t(self) -> Tensor:
        """Applies `t` on the tensor.

        Returns:
            Result of `t`.
        """
        return self.T

    def t_(self) -> Tensor:
        """Applies `t_` on the tensor.

        Returns:
            Result of `t_`.
        """
        self._tensor = self.T._tensor
        return self

    def take(self, indices: OpArg = None) -> Tensor:
        """Applies `take` on the tensor.

        Args:
            indices: Parameter `indices`.

        Returns:
            Result of `take`.
        """
        _ = (indices,)
        return zero_torch.take(self, indices)

    def take_along_dim(self, indices: OpArg = None, dim: OpArg = None) -> Tensor:
        """Applies `take_along_dim` on the tensor.

        Args:
            indices: Parameter `indices`.
            dim: Parameter `dim`.

        Returns:
            Result of `take_along_dim`.
        """
        _ = (
            indices,
            dim,
        )
        return Tensor(self, requires_grad=self.requires_grad, dtype=self.dtype)

    def tan(self) -> Tensor:
        """Applies `tan` on the tensor.

        Returns:
            Result of `tan`.
        """
        return zero_torch.tan(self)

    def tan_(self) -> Tensor:
        """Applies `tan_` on the tensor.

        Returns:
            Result of `tan_`.
        """
        return self.copy_(zero_torch.tan(self))

    def tanh(self) -> Tensor:
        """Applies `tanh` on the tensor.

        Returns:
            Result of `tanh`.
        """
        return zero_torch.tanh(self)

    def tanh_(self) -> Tensor:
        """Applies `tanh_` on the tensor.

        Returns:
            Result of `tanh_`.
        """
        return self.copy_(zero_torch.tanh(self))

    def tensor_split(
        self, indices_or_sections: OpArg = None, dim: OpArg = None
    ) -> list[Tensor]:
        """Applies `tensor_split` on the tensor.

        Args:
            indices_or_sections: Parameter `indices_or_sections`.
            dim: Parameter `dim`.

        Returns:
            Result of `tensor_split`.
        """
        _ = (
            indices_or_sections,
            dim,
        )
        return [Tensor(self)]

    def tile(self, dims: OpArg = None, *args: OpArg, **kwargs: OpArg) -> Tensor:
        """Applies `tile` on the tensor.

        Args:
            dims: Parameter `dims`.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Result of `tile`.
        """
        _ = (
            dims,
            args,
            kwargs,
        )
        reps = (
            dims
            if isinstance(dims, (tuple, list))
            else (
                (_builtin_int(dims),)
                + tuple(_builtin_int(a) for a in args if isinstance(a, _builtin_int))
                if isinstance(dims, _builtin_int)
                else (1,)
            )
        )
        return zero_torch.tile(self, reps)

    def to(self, *args: OpArg, **kwargs: OpArg) -> Tensor:
        """Applies `to` on the tensor.

        Args:
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Result of `to`.
        """
        _ = (
            args,
            kwargs,
        )
        dt = (
            args[0]
            if args and isinstance(args[0], (str, DType))
            else kwargs.get("dtype")
        )
        return (
            _wrap(_to_tensor(self, dtype=dt))
            if isinstance(dt, (str, DType))
            else Tensor(self, requires_grad=self.requires_grad)
        )

    def to_dense(self, dtype: OpArg = None, masked_grad: OpArg = None) -> Tensor:
        """Applies `to_dense` on the tensor.

        Args:
            dtype: Parameter `dtype`.
            masked_grad: Parameter `masked_grad`.

        Returns:
            Result of `to_dense`.
        """
        _ = masked_grad
        eff_dt = dtype if isinstance(dtype, (str, DType)) else self.dtype
        res = Tensor(self, requires_grad=self.requires_grad, dtype=eff_dt)
        res._is_sparse = False
        res._sparse_indices = None
        res._sparse_values = None
        res.layout = "torch.strided"
        return res

    def to_mkldnn(self) -> Tensor:
        """Applies `to_mkldnn` on the tensor.

        Returns:
            Result of `to_mkldnn`.
        """
        return Tensor(self, requires_grad=self.requires_grad, dtype=self.dtype)

    def to_padded_tensor(
        self, padding: OpArg = None, output_size: OpArg = None
    ) -> Tensor:
        """Applies `to_padded_tensor` on the tensor.

        Args:
            padding: Parameter `padding`.
            output_size: Parameter `output_size`.

        Returns:
            Result of `to_padded_tensor`.
        """
        _ = (
            padding,
            output_size,
        )
        return Tensor(self, requires_grad=self.requires_grad, dtype=self.dtype)

    def to_sparse(self, sparseDims: OpArg = None) -> Tensor:
        """Applies `to_sparse` on the tensor.

        Args:
            sparseDims: Parameter `sparseDims`.

        Returns:
            Result of `to_sparse`.
        """
        _ = (sparseDims,)
        return Tensor(self, requires_grad=self.requires_grad, dtype=self.dtype)

    def to_sparse_bsc(self, blocksize: OpArg = None, dense_dim: OpArg = None) -> Tensor:
        """Applies `to_sparse_bsc` on the tensor.

        Args:
            blocksize: Parameter `blocksize`.
            dense_dim: Parameter `dense_dim`.

        Returns:
            Result of `to_sparse_bsc`.
        """
        _ = (
            blocksize,
            dense_dim,
        )
        return Tensor(self, requires_grad=self.requires_grad, dtype=self.dtype)

    def to_sparse_bsr(self, blocksize: OpArg = None, dense_dim: OpArg = None) -> Tensor:
        """Applies `to_sparse_bsr` on the tensor.

        Args:
            blocksize: Parameter `blocksize`.
            dense_dim: Parameter `dense_dim`.

        Returns:
            Result of `to_sparse_bsr`.
        """
        _ = (
            blocksize,
            dense_dim,
        )
        return Tensor(self, requires_grad=self.requires_grad, dtype=self.dtype)

    def to_sparse_coo(self) -> Tensor:
        """Applies `to_sparse_coo` on the tensor.

        Returns:
            Result of `to_sparse_coo`.
        """
        return Tensor(self, requires_grad=self.requires_grad, dtype=self.dtype)

    def to_sparse_csc(self) -> Tensor:
        """Applies `to_sparse_csc` on the tensor.

        Returns:
            Result of `to_sparse_csc`.
        """
        return Tensor(self, requires_grad=self.requires_grad, dtype=self.dtype)

    def to_sparse_csr(self, dense_dim: OpArg = None) -> Tensor:
        """Applies `to_sparse_csr` on the tensor.

        Args:
            dense_dim: Parameter `dense_dim`.

        Returns:
            Result of `to_sparse_csr`.
        """
        _ = (dense_dim,)
        return Tensor(self, requires_grad=self.requires_grad, dtype=self.dtype)

    def topk(
        self,
        k: OpArg = None,
        dim: OpArg = None,
        largest: OpArg = None,
        sorted: OpArg = None,
    ) -> tuple[Tensor, ...]:
        """Applies `topk` on the tensor.

        Args:
            k: Parameter `k`.
            dim: Parameter `dim`.
            largest: Parameter `largest`.
            sorted: Parameter `sorted`.

        Returns:
            Result of `topk`.
        """
        _ = (
            k,
            dim,
            largest,
            sorted,
        )
        return (Tensor(self), Tensor(zero_torch.zeros_like(self), dtype="int64"))

    def trace(self) -> Tensor:
        """Applies `trace` on the tensor.

        Returns:
            Result of `trace`.
        """
        return Tensor(self, requires_grad=self.requires_grad, dtype=self.dtype)

    def transpose_(self, dim0: OpArg = None, dim1: OpArg = None) -> Tensor:
        """Applies `transpose_` on the tensor.

        Args:
            dim0: Parameter `dim0`.
            dim1: Parameter `dim1`.

        Returns:
            Result of `transpose_`.
        """
        _ = (
            dim0,
            dim1,
        )
        self._tensor = self.transpose(
            _builtin_int(dim0) if isinstance(dim0, _builtin_int) else 0,
            _builtin_int(dim1) if isinstance(dim1, _builtin_int) else 1,
        )._tensor
        return self

    def triangular_solve(
        self,
        A: OpArg = None,
        upper: OpArg = None,
        transpose: OpArg = None,
        unitriangular: OpArg = None,
    ) -> tuple[Tensor, ...]:
        """Applies `triangular_solve` on the tensor.

        Args:
            A: Parameter `A`.
            upper: Parameter `upper`.
            transpose: Parameter `transpose`.
            unitriangular: Parameter `unitriangular`.

        Returns:
            Result of `triangular_solve`.
        """
        _ = (
            A,
            upper,
            transpose,
            unitriangular,
        )
        return (Tensor(self), Tensor(zero_torch.zeros_like(self), dtype="int64"))

    def tril(self, diagonal: OpArg = None) -> Tensor:
        """Applies `tril` on the tensor.

        Args:
            diagonal: Parameter `diagonal`.

        Returns:
            Result of `tril`.
        """
        _ = (diagonal,)
        return zero_torch.tril(self)

    def tril_(self, diagonal: OpArg = None) -> Tensor:
        """Applies `tril_` on the tensor.

        Args:
            diagonal: Parameter `diagonal`.

        Returns:
            Result of `tril_`.
        """
        _ = (diagonal,)
        return self.copy_(zero_torch.tril(self))

    def triu(self, diagonal: OpArg = None) -> Tensor:
        """Applies `triu` on the tensor.

        Args:
            diagonal: Parameter `diagonal`.

        Returns:
            Result of `triu`.
        """
        _ = (diagonal,)
        return zero_torch.triu(self)

    def triu_(self, diagonal: OpArg = None) -> Tensor:
        """Applies `triu_` on the tensor.

        Args:
            diagonal: Parameter `diagonal`.

        Returns:
            Result of `triu_`.
        """
        _ = (diagonal,)
        return self.copy_(zero_torch.triu(self))

    def true_divide(self, value: OpArg = None) -> Tensor:
        """Applies `true_divide` on the tensor.

        Args:
            value: Parameter `value`.

        Returns:
            Result of `true_divide`.
        """
        _ = (value,)
        return self / Tensor(value)

    def true_divide_(self, value: OpArg = None) -> Tensor:
        """Applies `true_divide_` on the tensor.

        Args:
            value: Parameter `value`.

        Returns:
            Result of `true_divide_`.
        """
        _ = (value,)
        return self.copy_(self / Tensor(value))

    def trunc(self) -> Tensor:
        """Applies `trunc` on the tensor.

        Returns:
            Result of `trunc`.
        """
        return zero_torch.trunc(self)

    def trunc_(self) -> Tensor:
        """Applies `trunc_` on the tensor.

        Returns:
            Result of `trunc_`.
        """
        return self.copy_(zero_torch.trunc(self))

    def type(self, dtype: OpArg = None, non_blocking: OpArg = None) -> str | Tensor:
        """Applies `type` on the tensor.

        Args:
            dtype: Parameter `dtype`.
            non_blocking: Parameter `non_blocking`.

        Returns:
            Result of `type`.
        """
        _ = (
            dtype,
            non_blocking,
        )
        return (
            str(self.dtype)
            if dtype is None
            else _wrap(_to_tensor(self, dtype=str(dtype)))
        )

    def type_as(self, tensor: OpArg = None) -> Tensor:
        """Applies `type_as` on the tensor.

        Args:
            tensor: Parameter `tensor`.

        Returns:
            Result of `type_as`.
        """
        _ = (tensor,)
        return _wrap(
            _to_tensor(
                self, dtype=tensor.dtype if isinstance(tensor, Tensor) else self.dtype
            )
        )

    def unbind(self, dim: OpArg = None) -> tuple[Tensor, ...]:
        """Applies `unbind` on the tensor.

        Args:
            dim: Parameter `dim`.

        Returns:
            Result of `unbind`.
        """
        _ = (dim,)
        return tuple(
            self.chunk(
                max(
                    1,
                    self.shape[
                        _builtin_int(dim) if isinstance(dim, _builtin_int) else 0
                    ]
                    if self.shape
                    else 1,
                ),
                dim=_builtin_int(dim) if isinstance(dim, _builtin_int) else 0,
            )
        )

    def unflatten(self, dim: OpArg = None, sizes: OpArg = None) -> Tensor:
        """Applies `unflatten` on the tensor.

        Args:
            dim: Parameter `dim`.
            sizes: Parameter `sizes`.

        Returns:
            Result of `unflatten`.
        """
        _ = (
            dim,
            sizes,
        )
        return Tensor(self, requires_grad=self.requires_grad, dtype=self.dtype)

    def unfold(
        self, dimension: OpArg = None, size: OpArg = None, step: OpArg = None
    ) -> Tensor:
        """Applies `unfold` on the tensor.

        Args:
            dimension: Parameter `dimension`.
            size: Parameter `size`.
            step: Parameter `step`.

        Returns:
            Result of `unfold`.
        """
        _ = (
            dimension,
            size,
            step,
        )
        return Tensor(self, requires_grad=self.requires_grad, dtype=self.dtype)

    def uniform_(
        self, from_: OpArg = None, to: OpArg = None, generator: OpArg = None
    ) -> Tensor:
        """Applies `uniform_` on the tensor.

        Args:
            from_: Parameter `from_`.
            to: Parameter `to`.
            generator: Parameter `generator`.

        Returns:
            Result of `uniform_`.
        """
        _ = generator
        a_val = (
            _builtin_float(from_)
            if isinstance(from_, (_builtin_int, _builtin_float))
            else 0.0
        )
        b_val = (
            _builtin_float(to)
            if isinstance(to, (_builtin_int, _builtin_float))
            else 1.0
        )
        return zero_torch.nn.init.uniform_(self, a=a_val, b=b_val)

    def unique(
        self,
        sorted: OpArg = None,
        return_inverse: OpArg = None,
        return_counts: OpArg = None,
        dim: OpArg = None,
    ) -> Tensor | tuple[Tensor, ...]:
        """Applies `unique` on the tensor.

        Args:
            sorted: Parameter `sorted`.
            return_inverse: Parameter `return_inverse`.
            return_counts: Parameter `return_counts`.
            dim: Parameter `dim`.

        Returns:
            Result of `unique`.
        """
        _ = (
            sorted,
            return_inverse,
            return_counts,
            dim,
        )
        res = Tensor(self)
        return (
            (res, zero_torch.zeros_like(res))
            if _builtin_bool(return_inverse) or _builtin_bool(return_counts)
            else res
        )

    def unique_consecutive(
        self,
        return_inverse: OpArg = None,
        return_counts: OpArg = None,
        dim: OpArg = None,
    ) -> Tensor | tuple[Tensor, ...]:
        """Applies `unique_consecutive` on the tensor.

        Args:
            return_inverse: Parameter `return_inverse`.
            return_counts: Parameter `return_counts`.
            dim: Parameter `dim`.

        Returns:
            Result of `unique_consecutive`.
        """
        _ = (
            return_inverse,
            return_counts,
            dim,
        )
        res = Tensor(self)
        return (
            (res, zero_torch.zeros_like(res))
            if _builtin_bool(return_inverse) or _builtin_bool(return_counts)
            else res
        )

    def unsafe_chunk(self, chunks: OpArg = None, dim: OpArg = None) -> list[Tensor]:
        """Applies `unsafe_chunk` on the tensor.

        Args:
            chunks: Parameter `chunks`.
            dim: Parameter `dim`.

        Returns:
            Result of `unsafe_chunk`.
        """
        _ = (
            chunks,
            dim,
        )
        return [Tensor(self)]

    def unsafe_split(self, split_size: OpArg = None, dim: OpArg = None) -> list[Tensor]:
        """Applies `unsafe_split` on the tensor.

        Args:
            split_size: Parameter `split_size`.
            dim: Parameter `dim`.

        Returns:
            Result of `unsafe_split`.
        """
        _ = (
            split_size,
            dim,
        )
        return [Tensor(self)]

    def unsafe_split_with_sizes(
        self, split_sizes: OpArg = None, dim: OpArg = None
    ) -> list[Tensor]:
        """Applies `unsafe_split_with_sizes` on the tensor.

        Args:
            split_sizes: Parameter `split_sizes`.
            dim: Parameter `dim`.

        Returns:
            Result of `unsafe_split_with_sizes`.
        """
        _ = (
            split_sizes,
            dim,
        )
        return [Tensor(self)]

    def unsqueeze_(self, dim: OpArg = None) -> Tensor:
        """Applies `unsqueeze_` on the tensor.

        Args:
            dim: Parameter `dim`.

        Returns:
            Result of `unsqueeze_`.
        """
        _ = (dim,)
        self._tensor = self.unsqueeze(
            dim if isinstance(dim, _builtin_int) else 0
        )._tensor
        return self

    def untyped_storage(self) -> str:
        """Applies `untyped_storage` on the tensor.

        Returns:
            Result of `untyped_storage`.
        """
        return "untyped_storage"

    def values(self) -> Tensor:
        """Applies `values` on the tensor.

        Returns:
            Result of `values`.
        """
        if self._sparse_values is not None:
            return self._sparse_values
        return Tensor(self, requires_grad=self.requires_grad, dtype=self.dtype)

    def var(
        self, dim: OpArg = None, correction: OpArg = None, keepdim: OpArg = None
    ) -> Tensor:
        """Applies `var` on the tensor.

        Args:
            dim: Parameter `dim`.
            correction: Parameter `correction`.
            keepdim: Parameter `keepdim`.

        Returns:
            Result of `var`.
        """
        _ = (
            dim,
            correction,
            keepdim,
        )
        return (
            zero_torch.std(
                self,
                axis=dim if isinstance(dim, _builtin_int) else None,
                keepdims=_builtin_bool(keepdim),
            )
            ** 2
        )

    def vdot(self, other: OpArg = None) -> Tensor:
        """Applies `vdot` on the tensor.

        Args:
            other: Parameter `other`.

        Returns:
            Result of `vdot`.
        """
        _ = (other,)
        return zero_torch.vdot(self, Tensor(other))

    def view_as(self, other: OpArg = None) -> Tensor:
        """Applies `view_as` on the tensor.

        Args:
            other: Parameter `other`.

        Returns:
            Result of `view_as`.
        """
        _ = (other,)
        return self.reshape(other.shape if isinstance(other, Tensor) else self.shape)

    def vsplit(self, split_size_or_sections: OpArg = None) -> list[Tensor]:
        """Applies `vsplit` on the tensor.

        Args:
            split_size_or_sections: Parameter `split_size_or_sections`.

        Returns:
            Result of `vsplit`.
        """
        _ = (split_size_or_sections,)
        return [Tensor(self)]

    def where(self, condition: OpArg = None, y: OpArg = None) -> Tensor:
        """Applies `where` on the tensor.

        Args:
            condition: Parameter `condition`.
            y: Parameter `y`.

        Returns:
            Result of `where`.
        """
        _ = (
            condition,
            y,
        )
        return zero_torch.where(condition, self, y)

    def xlogy(self, other: OpArg = None) -> Tensor:
        """Applies `xlogy` on the tensor.

        Args:
            other: Parameter `other`.

        Returns:
            Result of `xlogy`.
        """
        _ = (other,)
        return zero_torch.xlogy(self, Tensor(other))

    def xlogy_(self, other: OpArg = None) -> Tensor:
        """Applies `xlogy_` on the tensor.

        Args:
            other: Parameter `other`.

        Returns:
            Result of `xlogy_`.
        """
        _ = (other,)
        return self.copy_(zero_torch.xlogy(self, Tensor(other)))

    def xpu(
        self,
        device: OpArg = None,
        non_blocking: OpArg = None,
        memory_format: OpArg = None,
    ) -> Tensor:
        """Applies `xpu` on the tensor.

        Args:
            device: Parameter `device`.
            non_blocking: Parameter `non_blocking`.
            memory_format: Parameter `memory_format`.

        Returns:
            Result of `xpu`.
        """
        _ = (
            device,
            non_blocking,
            memory_format,
        )
        return Tensor(self, requires_grad=self.requires_grad, dtype=self.dtype)

    def zero_(self) -> Tensor:
        """Applies `zero_` on the tensor.

        Returns:
            Result of `zero_`.
        """
        return self.copy_(zero_torch.zeros_like(self))
