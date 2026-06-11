"""Tensor API."""

from typing import Any, Optional, Union

import ml_switcheroo
import numpy as np
import ml_switcheroo.ops as ops
from ml_switcheroo.core.config import config
from ml_switcheroo.tracing import _tracer, ProxyTensor
from ml_switcheroo.ir.core import LogicalNode
import uuid


def _to_tensor(x: Any, dtype: Optional[Any] = None) -> ml_switcheroo.Tensor:
    """Converts the input to an ml_switcheroo Tensor.

    Args:
        x (Any): The input data to convert. Can be a Tensor, ml_switcheroo.Tensor, ProxyTensor, or array-like.
        dtype (Optional[Any], optional): The desired data type of the tensor. Defaults to None.

    Returns:
        ml_switcheroo.Tensor: The converted tensor.
    """
    if isinstance(x, Tensor):
        x = x._tensor
    if isinstance(x, ml_switcheroo.Tensor):
        if _tracer.is_tracing and not hasattr(x.data, "id"):
            # lift eager tensor as constant
            out_id = str(uuid.uuid4())
            node = LogicalNode(
                id=out_id,
                op_type="Constant",
                attributes={"value": np.array(x.data).tolist()},
                shape_metadata=x.shape,
            )
            _tracer.add_node(node)
            pt = ProxyTensor(id=out_id, shape=x.shape, dtype=x.dtype.value)
            return ml_switcheroo.Tensor(
                data=pt, shape=x.shape, dtype=x.dtype, device=x.device
            )
        return x
    if isinstance(x, ProxyTensor):
        from ml_switcheroo.core.dtype import DType

        # Determine dtype roughly or default to Float32
        dt = config.default_float_dtype
        try:
            if x.dtype:
                dt = DType(x.dtype)
        except Exception:
            pass
        return ml_switcheroo.Tensor(
            data=x,
            shape=x.shape,
            dtype=dt,
            device=config.default_device,
        )

    # Otherwise it's an array-like
    arr = np.array(x)
    if dtype is not None:
        arr = arr.astype(dtype)

    from ml_switcheroo.core.dtype import DType

    dt_str = str(arr.dtype)
    dt = config.default_float_dtype
    try:
        if "float" in dt_str or "int" in dt_str or "bool" in dt_str:
            if dt_str == "float64":
                dt = DType.Float64
            elif dt_str == "float32":
                dt = DType.Float32
            elif dt_str == "int64":
                dt = DType.Int64
            elif dt_str == "int32":
                dt = DType.Int32
            elif dt_str == "bool":
                dt = DType.Bool
            else:
                dt = DType(dt_str)
    except Exception:
        dt = config.default_float_dtype

    return ml_switcheroo.Tensor(
        data=arr, shape=arr.shape, dtype=dt, device=config.default_device
    )


def _wrap(x: Any) -> "Tensor":
    """Wraps an ml_switcheroo Tensor into a zero_torch Tensor.

    Args:
        x (Any): The tensor or value to wrap.

    Returns:
        Tensor: The wrapped zero_torch Tensor.
    """
    if isinstance(x, Tensor):
        return x
    return Tensor(x)


class Tensor:
    """A multi-dimensional matrix containing elements of a single data type."""

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

    def view(self, *shape) -> "Tensor":
        """Returns a new tensor with the same data but of a different shape.

        Args:
            *shape: The desired shape of the returned tensor.

        Returns:
            Tensor: A new tensor with the specified shape.
        """
        if self._tensor is None:
            return self
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        shape = tuple(s for s in shape if s is not None)
        return _wrap(ops.reshape(_to_tensor(self), shape=shape))

    def reshape(self, *shape) -> "Tensor":
        """Returns a tensor with the same data and number of elements as input, but with the specified shape.

        Args:
            *shape: The desired shape of the returned tensor.

        Returns:
            Tensor: A new tensor with the specified shape.
        """
        if self._tensor is None:
            return self
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        shape = tuple(s for s in shape if s is not None)
        return _wrap(ops.reshape(_to_tensor(self), shape=shape))

    def numpy(self) -> Any:
        """Returns the tensor as a NumPy array.

        Returns:
            Any: A NumPy ndarray containing the tensor's data.
        """
        return np.array(self._tensor.data)

    def backward(self, *args, **kwargs) -> None:
        """Computes the gradient of current tensor w.r.t. graph leaves.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Raises:
            NotImplementedError: Because this needs to be implemented by autograd system.
        """
        raise NotImplementedError

    @property
    def device(self) -> str:
        """Gets the device on which the tensor is allocated.

        Returns:
            str: The device string (e.g., 'cpu').
        """
        return "cpu"

    def contiguous(self) -> "Tensor":
        """Returns a contiguous in memory tensor containing the same data as self tensor.

        Returns:
            Tensor: A contiguous tensor.
        """
        if self._tensor is None:
            return self
        if config.eager_mode:
            arr = self._tensor.data
            if hasattr(arr, "flags") and not arr.flags["C_CONTIGUOUS"]:
                return _wrap(np.ascontiguousarray(arr))
        return self

    def squeeze(self, dim=None) -> "Tensor":
        """Returns a tensor with all specified dimensions of input of size 1 removed.

        Args:
            dim (Optional[int], optional): If given, the input will be squeezed only in this dimension. Defaults to None.

        Returns:
            Tensor: The squeezed tensor.
        """
        if self._tensor is None:
            return self
        if dim is None:
            return _wrap(ops.squeeze(_to_tensor(self)))
        return _wrap(ops.squeeze(_to_tensor(self), dim=dim))

    def unsqueeze(self, dim) -> "Tensor":
        """Returns a new tensor with a dimension of size one inserted at the specified position.

        Args:
            dim (int): The index at which to insert the singleton dimension.

        Returns:
            Tensor: The unsqueezed tensor.
        """
        if self._tensor is None:
            return self
        return _wrap(ops.unsqueeze(_to_tensor(self), dim=dim))

    @property
    def T(self) -> "Tensor":
        """Return the transpose of the tensor.

        Returns:
            Tensor: The transposed tensor.
        """
        if self._tensor is None:
            return self
        dims = tuple(reversed(range(len(self.shape))))
        return _wrap(ops.permute(_to_tensor(self), dims=dims))

    def __add__(self, other) -> "Tensor":
        """Adds this tensor with another tensor or scalar.

        Args:
            other (Any): The other tensor or scalar to add.

        Returns:
            Tensor: The sum of the two inputs.
        """
        return _wrap(ops.add(_to_tensor(self), _to_tensor(other)))

    def __radd__(self, other) -> "Tensor":
        """Adds another tensor or scalar with this tensor (reverse addition).

        Args:
            other (Any): The other tensor or scalar to add.

        Returns:
            Tensor: The sum of the two inputs.
        """
        return _wrap(ops.add(_to_tensor(other), _to_tensor(self)))

    def __sub__(self, other) -> "Tensor":
        """Subtracts another tensor or scalar from this tensor.

        Args:
            other (Any): The other tensor or scalar to subtract.

        Returns:
            Tensor: The difference between the two inputs.
        """
        return _wrap(ops.subtract(_to_tensor(self), _to_tensor(other)))

    def __rsub__(self, other) -> "Tensor":
        """Subtracts this tensor from another tensor or scalar (reverse subtraction).

        Args:
            other (Any): The other tensor or scalar from which to subtract.

        Returns:
            Tensor: The difference between the two inputs.
        """
        return _wrap(ops.subtract(_to_tensor(other), _to_tensor(self)))

    def __mul__(self, other) -> "Tensor":
        """Multiplies this tensor with another tensor or scalar element-wise.

        Args:
            other (Any): The other tensor or scalar to multiply.

        Returns:
            Tensor: The product of the two inputs.
        """
        return _wrap(ops.multiply(_to_tensor(self), _to_tensor(other)))

    def __rmul__(self, other) -> "Tensor":
        """Multiplies another tensor or scalar with this tensor element-wise (reverse multiplication).

        Args:
            other (Any): The other tensor or scalar to multiply.

        Returns:
            Tensor: The product of the two inputs.
        """
        return _wrap(ops.multiply(_to_tensor(other), _to_tensor(self)))

    def __truediv__(self, other) -> "Tensor":
        """Divides this tensor by another tensor or scalar element-wise.

        Args:
            other (Any): The divisor tensor or scalar.

        Returns:
            Tensor: The quotient of the two inputs.
        """
        return _wrap(ops.divide(_to_tensor(self), _to_tensor(other)))

    def __rtruediv__(self, other) -> "Tensor":
        """Divides another tensor or scalar by this tensor element-wise (reverse division).

        Args:
            other (Any): The dividend tensor or scalar.

        Returns:
            Tensor: The quotient of the two inputs.
        """
        return _wrap(ops.divide(_to_tensor(other), _to_tensor(self)))

    def __matmul__(self, other) -> "Tensor":
        """Performs matrix multiplication of this tensor and another tensor.

        Args:
            other (Any): The other tensor to multiply.

        Returns:
            Tensor: The result of matrix multiplication.
        """
        return _wrap(ops.matmul(_to_tensor(self), _to_tensor(other)))

    def __rmatmul__(self, other) -> "Tensor":
        """Performs matrix multiplication of another tensor and this tensor (reverse matmul).

        Args:
            other (Any): The other tensor to multiply.

        Returns:
            Tensor: The result of matrix multiplication.
        """
        return _wrap(ops.matmul(_to_tensor(other), _to_tensor(self)))

    def __pow__(self, other) -> "Tensor":
        """Raises this tensor to the power of another tensor or scalar.

        Args:
            other (Any): The exponent tensor or scalar.

        Returns:
            Tensor: The result of the exponentiation.
        """
        return _wrap(ops.power(_to_tensor(self), _to_tensor(other)))

    def __neg__(self) -> "Tensor":
        """Returns a new tensor with the negative of the elements of this tensor.

        Returns:
            Tensor: The negated tensor.
        """
        return _wrap(ops.negative(_to_tensor(self)))

    def __len__(self) -> int:
        """Returns the length of the tensor's first dimension.

        Returns:
            int: The size of the first dimension, or 0 if it has no dimensions.
        """
        return self.shape[0] if len(self.shape) > 0 else 0

    def size(self, dim=None) -> Any:
        """Returns the size of the tensor or the size of a specific dimension.

        Args:
            dim (Optional[int], optional): If specified, returns the size of that dimension. Defaults to None.

        Returns:
            Any: The shape tuple or the size of the specified dimension.
        """
        if dim is None:
            return self.shape
        return self.shape[dim]

    def item(self) -> Union[int, float, bool]:
        """Returns the value of this tensor as a standard Python number. This only works for tensors with one element.

        Returns:
            int | float | bool: The scalar value of the tensor.
        """
        return np.array(self._tensor.data).item()
