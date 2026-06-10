"""Tensor API."""

from typing import Any
import numpy as np

try:
    from ml_switcheroo.tracing import _tracer, ProxyTensor
except ImportError:
    _tracer = None
    ProxyTensor = type("ProxyTensor", (), {})
try:
    from ml_switcheroo_ir import LogicalNode
except ImportError:
    LogicalNode = type("LogicalNode", (), {})


class Tensor:
    def __init__(self, data: Any, requires_grad: bool = False, dtype=None):
        if isinstance(data, Tensor):
            self._data = data._data
        else:
            self._data = np.array(data, dtype=dtype) if data is not None else None
        self.requires_grad = requires_grad

    @property
    def shape(self):
        return self._data.shape if self._data is not None else ()

    @property
    def dtype(self):
        return self._data.dtype if self._data is not None else None

    def view(self, *shape) -> "Tensor":
        return Tensor(self._data.reshape(*shape)) if self._data is not None else self

    def reshape(self, *shape) -> "Tensor":
        return Tensor(self._data.reshape(*shape)) if self._data is not None else self

    def contiguous(self) -> "Tensor":
        if self._data is None:
            return self
        if self._data.flags["C_CONTIGUOUS"]:
            return self
        return Tensor(np.ascontiguousarray(self._data))

    def squeeze(self, dim=None) -> "Tensor":
        if self._data is None:
            return self
        if dim is None:
            return Tensor(np.squeeze(self._data))
        else:
            return Tensor(np.squeeze(self._data, axis=dim))

    @property
    def T(self) -> "Tensor":
        """Return the transpose of the tensor."""
        return Tensor(self._data.T) if self._data is not None else self

    def backward(self):
        pass

    def _op(self, other: Any, op_type: str, eager_fn) -> "Tensor":
        other_data = other._data if isinstance(other, Tensor) else other
        res = eager_fn(self._data, other_data) if self._data is not None else None
        return Tensor(res)

    def __add__(self, other):
        return self._op(other, "add", lambda a, b: a + b)

    def __sub__(self, other):
        return self._op(other, "sub", lambda a, b: a - b)

    def __mul__(self, other):
        return self._op(other, "mul", lambda a, b: a * b)

    def __truediv__(self, other):
        return self._op(other, "div", lambda a, b: a / b)

    def __matmul__(self, other):
        return self._op(other, "matmul", lambda a, b: a @ b)
