"""Tensor API."""

from typing import Any, Optional

import ml_switcheroo
import ml_switcheroo.core.tensor_utils
import ml_switcheroo.ops as ops
from ml_switcheroo.core.config import config
from ml_switcheroo.tracing import _tracer, ProxyTensor
from ml_switcheroo_ir import LogicalNode
import uuid


def _to_tensor(x: Any, dtype: Optional[Any] = None) -> ml_switcheroo.Tensor:
    """Function."""
    if isinstance(x, Tensor):
        x = x._tensor
    if isinstance(x, ml_switcheroo.Tensor):
        if _tracer.is_tracing and not hasattr(x.data, "id"):
            # lift eager tensor as constant
            out_id = str(uuid.uuid4())
            node = LogicalNode(
                id=out_id,
                op_type="Constant",
                attributes={
                    "value": ml_switcheroo.core.tensor_utils.to_array(x.data).tolist()
                },
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
    arr = ml_switcheroo.core.tensor_utils.to_array(x)
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
    """Function."""
    if isinstance(x, Tensor):
        return x
    return Tensor(x)


class Tensor:
    """Class."""

    def __init__(self, data: Any, requires_grad: bool = False, dtype=None):
        """Function."""
        if isinstance(data, Tensor):
            self._tensor = data._tensor
        else:
            if data is None:
                self._tensor = None
            else:
                self._tensor = _to_tensor(data, dtype=dtype)
        self.requires_grad = requires_grad

    @property
    def data(self):
        """Function."""
        return self._tensor.data if self._tensor is not None else None

    @property
    def _data(self):
        """Function."""
        return self._tensor.data if self._tensor is not None else None

    @property
    def shape(self):
        """Function."""
        return self._tensor.shape if self._tensor is not None else ()

    @property
    def dtype(self):
        # Maps DType enum to numpy dtype for tests
        """Function."""
        return (
            ml_switcheroo.core.tensor_utils.to_numpy_dtype(self._tensor.dtype.value)
            if self._tensor is not None
            else None
        )

    def view(self, *shape) -> "Tensor":
        """Function."""
        if self._tensor is None:
            return self
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        shape = tuple(s for s in shape if s is not None)
        return _wrap(ops.reshape(_to_tensor(self), shape=shape))

    def reshape(self, *shape) -> "Tensor":
        """Function."""
        if self._tensor is None:
            return self
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        shape = tuple(s for s in shape if s is not None)
        return _wrap(ops.reshape(_to_tensor(self), shape=shape))

    def numpy(self):
        """Function."""
        return ml_switcheroo.core.tensor_utils.to_array(self._tensor.data)

    def backward(self, *args, **kwargs):
        """Function."""
        raise NotImplementedError

    @property
    def device(self):
        """Function."""
        return "cpu"

    def contiguous(self) -> "Tensor":
        """Function."""
        if self._tensor is None:
            return self
        if config.eager_mode:
            arr = self._tensor.data
            if hasattr(arr, "flags") and not arr.flags["C_CONTIGUOUS"]:
                return _wrap(ml_switcheroo.core.tensor_utils.ascontiguousarray(arr))
        return self

    def squeeze(self, dim=None) -> "Tensor":
        """Function."""
        if self._tensor is None:
            return self
        if dim is None:
            return _wrap(ops.squeeze(_to_tensor(self)))
        return _wrap(ops.squeeze(_to_tensor(self), dim=dim))

    def unsqueeze(self, dim) -> "Tensor":
        """Function."""
        if self._tensor is None:
            return self
        return _wrap(ops.unsqueeze(_to_tensor(self), dim=dim))

    @property
    def T(self) -> "Tensor":
        """Return the transpose of the tensor."""
        if self._tensor is None:
            return self
        dims = tuple(reversed(range(len(self.shape))))
        return _wrap(ops.permute(_to_tensor(self), dims=dims))

    def __add__(self, other):
        """Function."""
        return _wrap(ops.add(_to_tensor(self), _to_tensor(other)))

    def __radd__(self, other):
        """Function."""
        return _wrap(ops.add(_to_tensor(other), _to_tensor(self)))

    def __sub__(self, other):
        """Function."""
        return _wrap(ops.subtract(_to_tensor(self), _to_tensor(other)))

    def __rsub__(self, other):
        """Function."""
        return _wrap(ops.subtract(_to_tensor(other), _to_tensor(self)))

    def __mul__(self, other):
        """Function."""
        return _wrap(ops.multiply(_to_tensor(self), _to_tensor(other)))

    def __rmul__(self, other):
        """Function."""
        return _wrap(ops.multiply(_to_tensor(other), _to_tensor(self)))

    def __truediv__(self, other):
        """Function."""
        return _wrap(ops.divide(_to_tensor(self), _to_tensor(other)))

    def __rtruediv__(self, other):
        """Function."""
        return _wrap(ops.divide(_to_tensor(other), _to_tensor(self)))

    def __matmul__(self, other):
        """Function."""
        return _wrap(ops.matmul(_to_tensor(self), _to_tensor(other)))

    def __rmatmul__(self, other):
        """Function."""
        return _wrap(ops.matmul(_to_tensor(other), _to_tensor(self)))

    def __pow__(self, other):
        """Function."""
        return _wrap(ops.power(_to_tensor(self), _to_tensor(other)))

    def __neg__(self):
        """Function."""
        return _wrap(ops.negative(_to_tensor(self)))

    def __len__(self):
        """Function."""
        return self.shape[0] if len(self.shape) > 0 else 0

    def size(self, dim=None):
        """Function."""
        if dim is None:
            return self.shape
        return self.shape[dim]

    def item(self):
        """Function."""
        return ml_switcheroo.core.tensor_utils.to_array(self._tensor.data).item()
