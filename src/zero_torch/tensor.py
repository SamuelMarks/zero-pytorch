"""Tensor API."""

from __future__ import annotations

import functools
import operator
from typing import Any

from ml_switcheroo_compiler import ops
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import (
    Tensor as _SwitcherooTensor,
)
from ml_switcheroo_compiler.core.tensor import (
    TensorConfig as _TensorConfig,
)
from ml_switcheroo_compiler.ir.core import IRNode as LogicalNode

import zero_torch
from zero_torch.tracing import ProxyTensor, _tracer


def _to_tensor(x: Any, dtype: Any | None = None) -> _SwitcherooTensor:
    """Converts the input to an ml_switcheroo Tensor.

    Args:
        x (Any): The input data to convert. Can be a Tensor, _SwitcherooTensor, ProxyTensor, or array-like.
        dtype (Optional[Any], optional): The desired data type of the tensor. Defaults to None.

    Returns:
        _SwitcherooTensor: The converted tensor.
    """
    if isinstance(x, Tensor):
        x = x._tensor  # type: ignore[has-type]
    if isinstance(x, _SwitcherooTensor):
        if _tracer.is_tracing and not hasattr(x.data, "id"):
            # lift eager tensor as constant
            import uuid

            out_id = str(uuid.uuid4())
            val = x.data
            if hasattr(val, "tolist"):
                val = val.tolist()
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
        from ml_switcheroo_compiler.core.dtype import DType

        # Determine dtype roughly or default to Float32
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
            _pass = True
        return _SwitcherooTensor(
            data=x,
            config=_TensorConfig(
                shape=x.shape,
                dtype=dt,
                device=config.default_device,
            ),
        )

    # Otherwise it's an array-like
    def _get_shape_and_dt(v: Any) -> tuple[tuple[int, ...], str]:
        if isinstance(v, (int, float, bool)):
            if isinstance(v, bool):
                return (), "bool"
            if isinstance(v, int):
                return (), "int64"
            return (), "float32"
        if isinstance(v, (list, tuple)):
            if len(v) == 0:
                return (0,), "float32"
            inner_shape, inner_dt = _get_shape_and_dt(v[0])
            return (len(v),) + inner_shape, inner_dt
        if hasattr(v, "shape"):
            dt_str = str(getattr(v, "dtype", "float32"))
            return tuple(int(s) for s in v.shape), dt_str
        return (), "float32"

    shape, dt_str = _get_shape_and_dt(x)

    from ml_switcheroo_compiler.core.dtype import DType

    dt = config.default_float_dtype
    if "float" in dt_str or "int" in dt_str or "bool" in dt_str:
        if "float64" in dt_str:
            dt = DType.Float64
        elif "float32" in dt_str:
            dt = DType.Float32
        elif "int64" in dt_str:
            dt = DType.Int64
        elif "int32" in dt_str:
            dt = DType.Int32
        elif "bool" in dt_str:
            dt = DType.Bool

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
            _pass = True

    if not config.eager_mode and _tracer.is_tracing:
        import uuid

        from ml_switcheroo_compiler.ir.core import IRNode

        out_id = str(uuid.uuid4())
        val = x
        node = IRNode(
            id=out_id,
            op_type="Constant",
            attributes={"value": val},
            shape_metadata=shape,
        )
        _tracer.add_node(node)
        pt = ProxyTensor(id=out_id, shape=shape, dtype=dt.value)
        return _SwitcherooTensor(
            data=pt,
            config=_TensorConfig(shape=shape, dtype=dt, device=config.default_device),
        )

    if config.eager_mode and not hasattr(x, "shape") and not isinstance(x, ProxyTensor):
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
            _pass = True

    # Eager Mode Data Ingestion: flattening nested lists for continuous memory layout simulation
    if isinstance(x, (list, tuple)):

        def flatten(lst):
            if isinstance(lst, (list, tuple)):
                return functools.reduce(operator.iadd, (flatten(i) for i in lst), [])
            return [lst]

        x = flatten(x)

    t = _SwitcherooTensor(
        data=x,
        config=_TensorConfig(shape=shape, dtype=dt, device=config.default_device),
    )
    return t


def _wrap(x: Any) -> Any:
    """Wraps an ml_switcheroo Tensor into a zero_torch Tensor.

    Args:
        x (Any): The tensor or value to wrap.

    Returns:
        Tensor | tuple | list: The wrapped zero_torch Tensor(s).
    """
    if isinstance(x, Tensor):
        return x
    if isinstance(x, tuple):
        return tuple(_wrap(i) for i in x)
    if isinstance(x, list):
        return [_wrap(i) for i in x]
    return Tensor(x)


class Tensor:
    def digamma(self, *args, **kwargs):
        import zero_torch

        return zero_torch.digamma(self, *args, **kwargs)

    def erfc(self, *args, **kwargs):
        import zero_torch

        return zero_torch.erfc(self, *args, **kwargs)

    def erfinv(self, *args, **kwargs):
        import zero_torch

        return zero_torch.erfinv(self, *args, **kwargs)

    def lgamma(self, *args, **kwargs):
        import zero_torch

        return zero_torch.lgamma(self, *args, **kwargs)

    """A multi-dimensional matrix containing elements of a single data type."""

    _tensor: Any

    def __init__(self, data: Any, requires_grad: bool = False, dtype=None):
        """Initializes a Tensor.

        Args:
            data (Any): The initial data for the tensor. Can be a list, tuple, NumPy array, scalar, or another Tensor.
            requires_grad (bool, optional): If autograd should record operations on the returned tensor. Defaults to False.
            dtype (Optional[Any], optional): The desired data type of returned tensor. Defaults to None.
        """
        if isinstance(data, Tensor):
            self._tensor = data._tensor
        else:
            if data is None:
                self._tensor = None
            else:
                self._tensor = _to_tensor(data, dtype=dtype)
        self.requires_grad = requires_grad
        self.grad = None
        self.grad_fn = None
        self._retains_grad = False

    @property
    def is_leaf(self) -> bool:
        """Returns True if this Tensor is a leaf node.

        Returns:
            bool: True if it is a leaf.
        """
        return self.grad_fn is None

    def retain_grad(self) -> None:
        """Enables .grad attribute for non-leaf Tensors."""
        if not self.requires_grad:
            raise RuntimeError(
                "can't retain_grad on Tensor that has requires_grad=False"
            )
        self._retains_grad = True  # pragma: no cover

    @property
    def data(self) -> Any:
        """Gets the underlying tensor data.

        Returns:
            Any: The underlying data object or proxy.
        """
        return self._tensor.data if self._tensor is not None else None

    @property
    def _data(self) -> Any:
        """Gets the underlying private tensor data.

        Returns:
            Any: The underlying private data object or proxy.
        """
        return self._tensor.data if self._tensor is not None else None

    @property
    def shape(self) -> tuple:
        """Gets the shape of the tensor.

        Returns:
            tuple: The dimensions of the tensor.
        """
        return self._tensor.shape if self._tensor is not None else ()

    @property
    def dtype(self) -> Any:
        """Gets the data type of the tensor elements.

        Returns:
            Any: The data type of the tensor.
        """
        return self._tensor.dtype if self._tensor is not None else None

    def view(self, *shape) -> Tensor:
        """Returns a new tensor with the same data but of a different shape.

        Args:
            *shape: The desired shape of the returned tensor.

        Returns:
            Tensor: A new tensor with the specified shape.
        """
        if self._tensor is None:
            return self
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            s = shape[0]
            shape = tuple(item for item in s if item is not None)
        else:
            shape = tuple(item for item in shape if item is not None)
        return zero_torch.reshape(self, shape)

    def reshape(self, *shape) -> Tensor:
        """Returns a tensor with the same data and number of elements as input, but with the specified shape.

        Args:
            *shape: The desired shape of the returned tensor.

        Returns:
            Tensor: A new tensor with the specified shape.
        """
        if self._tensor is None:
            return self
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            s = shape[0]
            shape = tuple(item for item in s if item is not None)
        else:
            shape = tuple(item for item in shape if item is not None)
        return zero_torch.reshape(self, shape)

    def detach(self) -> Tensor:
        """Returns a new Tensor, detached from the current graph."""
        if self._tensor is None:
            return self
        res = self._tensor.detach()
        return _wrap(res)

    def item(self):
        """Returns the value of this tensor as a standard Python number."""
        arr = self.tolist()
        while isinstance(arr, list):
            if len(arr) != 1:
                raise ValueError(
                    "only one element tensors can be converted to Python scalars"
                )
            arr = arr[0]
        return arr

    def __repr__(self) -> str:
        if self.numel() > 1000:
            return "tensor([...])"
        return f"tensor({self.tolist()})"

    def __str__(self) -> str:
        return self.__repr__()

    def tolist(self) -> list:
        """Returns the tensor as a nested list.

        Returns:
            list: A nested list containing the tensor's data.
        """
        if self._tensor is None:
            return []  # pragma: no cover

        if hasattr(self._tensor.data, "tolist"):
            return self._tensor.data.tolist()
        data = self._tensor.data
        if isinstance(data, (list, tuple)):
            if len(self.shape) <= 1:
                return list(data)

            # Need to reshape flat list to nested list based on self.shape
            def _to_nested(flat_data, shape):
                if len(shape) == 0:
                    return flat_data[0]  # pragma: no cover
                if len(shape) == 1:
                    return list(flat_data)

                size = shape[0]
                chunk_size = 1
                for s in shape[1:]:
                    chunk_size *= s

                result = []
                for i in range(size):
                    result.append(
                        _to_nested(
                            flat_data[i * chunk_size : (i + 1) * chunk_size], shape[1:]
                        )
                    )
                return result

            return _to_nested(data, self.shape)

        return [data]  # pragma: no cover

    def numpy(self) -> Any:
        """Returns the tensor as a NumPy array.

        Returns:
            Any: A NumPy ndarray containing the tensor's data.

        Raises:
            RuntimeError: If NumPy is not installed.
        """
        if hasattr(self._tensor, "numpy"):
            try:
                return self._tensor.numpy()
            except Exception as e:  # pragma: no cover
                if type(e).__name__ == "ImportError":  # pragma: no cover
                    raise RuntimeError(
                        "NumPy is not installed. `Tensor.numpy()` requires NumPy."
                    ) from e  # pragma: no cover
                raise  # pragma: no cover

        raise RuntimeError("NumPy is not installed.")  # pragma: no cover

    def backward(self, gradient=None, retain_graph=None, create_graph=False) -> None:
        """Computes the gradient of current tensor w.r.t. graph leaves.

        Args:
            gradient (Tensor, optional): The gradient of the tensor. Defaults to None.
            retain_graph (bool, optional): Whether to retain the graph. Defaults to None.
            create_graph (bool, optional): Whether to create the graph of the derivative. Defaults to False.
        """
        if self.grad_fn is None and not self.requires_grad:
            raise RuntimeError(
                "element 0 of tensors does not require grad and does not have a grad_fn"
            )

        import zero_torch as torch

        from .autograd.engine import _backward

        if gradient is None:
            if self.numel() != 1:
                raise RuntimeError(  # pragma: no cover
                    "grad can be implicitly created only for scalar outputs"
                )
            gradient = torch.ones_like(self)

        _backward(self, gradient, retain_graph, create_graph)

    @property
    def device(self) -> str:
        """Gets the device on which the tensor is allocated.

        Returns:
            str: The device string (e.g., 'cpu').
        """
        return "cpu"

    def contiguous(self) -> Tensor:
        """Returns a contiguous tensor.

        Returns:
            Tensor: A new contiguous tensor.
        """
        if self._tensor is None:
            return self

        arr = self._tensor.data
        if hasattr(arr, "flags") and not arr.flags["C_CONTIGUOUS"]:
            return _wrap(arr.copy(order="C"))

        return _wrap(self._tensor.contiguous())

    def squeeze(self, dim=None) -> Tensor:
        """Returns a tensor with all specified dimensions of input of size 1 removed.

        Args:
            dim (Optional[int], optional): If given, the input will be squeezed only in this dimension. Defaults to None.

        Returns:
            Tensor: The squeezed tensor.
        """
        if self._tensor is None:
            return self
        if dim is None:
            return zero_torch.squeeze(self)
        return zero_torch.squeeze(self, dim)

    def unsqueeze(self, dim) -> Tensor:
        """Returns a new tensor with a dimension of size one inserted at the specified position.

        Args:
            dim (int): The index at which to insert the singleton dimension.

        Returns:
            Tensor: The unsqueezed tensor.
        """
        if self._tensor is None:
            return self
        import zero_torch

        return zero_torch.unsqueeze(self, dim)

    @property
    def T(self) -> Tensor:
        """Return the transpose of the tensor.

        Returns:
            Tensor: The transposed tensor.
        """
        if self._tensor is None:
            return self
        dims = tuple(reversed(range(len(self.shape))))
        return (
            zero_torch.transpose(self, dims[0], dims[1])
            if len(dims) == 2
            else zero_torch.permute(self, dims)
        )

    def __add__(self, other) -> Tensor:
        """Adds this tensor with another tensor or scalar.

        Args:
            other (Any): The other tensor or scalar to add.

        Returns:
            Tensor: The sum of the two inputs.
        """
        from .autograd.math_ops import AddBackward

        return AddBackward.apply(self, other)

    def __radd__(self, other) -> Tensor:
        """Adds another tensor or scalar with this tensor (reverse addition).

        Args:
            other (Any): The other tensor or scalar to add.

        Returns:
            Tensor: The sum of the two inputs.
        """
        from .autograd.math_ops import AddBackward

        return AddBackward.apply(other, self)

    def __sub__(self, other) -> Tensor:
        """Subtracts another tensor or scalar from this tensor.

        Args:
            other (Any): The other tensor or scalar to subtract.

        Returns:
            Tensor: The difference between the two inputs.
        """
        from .autograd.math_ops import SubBackward

        return SubBackward.apply(self, other)

    def __rsub__(self, other) -> Tensor:
        """Subtracts this tensor from another tensor or scalar (reverse subtraction).

        Args:
            other (Any): The other tensor or scalar from which to subtract.

        Returns:
            Tensor: The difference between the two inputs.
        """
        from .autograd.math_ops import SubBackward

        return SubBackward.apply(other, self)

    def __mul__(self, other) -> Tensor:
        """Multiplies this tensor with another tensor or scalar element-wise.

        Args:
            other (Any): The other tensor or scalar to multiply.

        Returns:
            Tensor: The product of the two inputs.
        """
        from .autograd.math_ops import MulBackward

        return MulBackward.apply(self, other)

    def __rmul__(self, other) -> Tensor:
        """Multiplies another tensor or scalar with this tensor element-wise (reverse multiplication).

        Args:
            other (Any): The other tensor or scalar to multiply.

        Returns:
            Tensor: The product of the two inputs.
        """
        from .autograd.math_ops import MulBackward

        return MulBackward.apply(other, self)

    def __truediv__(self, other) -> Tensor:
        """Divides this tensor by another tensor or scalar element-wise.

        Args:
            other (Any): The divisor tensor or scalar.

        Returns:
            Tensor: The quotient of the two inputs.
        """
        from .autograd.math_ops import DivBackward

        return DivBackward.apply(self, other)

    def __rtruediv__(self, other) -> Tensor:
        """Divides another tensor or scalar by this tensor element-wise (reverse division).

        Args:
            other (Any): The dividend tensor or scalar.

        Returns:
            Tensor: The quotient of the two inputs.
        """
        from .autograd.math_ops import DivBackward

        return DivBackward.apply(other, self)

    def __matmul__(self, other) -> Tensor:
        """Performs matrix multiplication of this tensor and another tensor.

        Args:
            other (Any): The other tensor to multiply.

        Returns:
            Tensor: The result of matrix multiplication.
        """
        from .autograd.math_ops import MatmulBackward

        return MatmulBackward.apply(self, other)

    def __rmatmul__(self, other) -> Tensor:
        """Performs matrix multiplication of another tensor and this tensor (reverse matmul).

        Args:
            other (Any): The other tensor to multiply.

        Returns:
            Tensor: The result of matrix multiplication.
        """
        from .autograd.math_ops import MatmulBackward

        return MatmulBackward.apply(other, self)

    def __pow__(self, other) -> Tensor:
        """Raises this tensor to the power of another tensor or scalar.

        Args:
            other (Any): The exponent tensor or scalar.

        Returns:
            Tensor: The result of the exponentiation.
        """
        from .autograd.math_ops import PowBackward

        return PowBackward.apply(self, other)

    def __neg__(self) -> Tensor:
        """Returns a new tensor with the negative of the elements of this tensor.

        Returns:
            Tensor: The negated tensor.
        """
        from .autograd.math_ops import NegBackward

        return NegBackward.apply(self)

    def __lt__(self, other) -> Tensor:
        return _wrap(ops.less(_to_tensor(self), _to_tensor(other)))

    def __gt__(self, other) -> Tensor:
        return _wrap(ops.greater(_to_tensor(self), _to_tensor(other)))

    def __le__(self, other) -> Tensor:
        return _wrap(ops.less_equal(_to_tensor(self), _to_tensor(other)))

    def __ge__(self, other) -> Tensor:
        return _wrap(ops.greater_equal(_to_tensor(self), _to_tensor(other)))

    def __eq__(self, other) -> Tensor:  # type: ignore[override]
        return _wrap(ops.equal(_to_tensor(self), _to_tensor(other)))

    def __ne__(self, other) -> Tensor:  # type: ignore[override]
        return _wrap(ops.not_equal(_to_tensor(self), _to_tensor(other)))

    def __len__(self) -> int:
        """Returns the length of the tensor's first dimension.

        Returns:
            int: The size of the first dimension, or 0 if it has no dimensions.
        """
        return self.shape[0] if len(self.shape) > 0 else 0

    def numel(self) -> int:
        """Returns the total number of elements in the tensor.

        Returns:
            int: The number of elements.
        """
        import functools
        import operator

        shape = self.shape
        if not shape:
            return 1
        return functools.reduce(operator.mul, shape, 1)
