"""Autograd module."""

from .engine import _backward as _backward
from .function import Context as Context
from .function import Function as Function
from .grad_mode import (
    is_grad_enabled as is_grad_enabled,
)
from .grad_mode import (
    no_grad as no_grad,
)
from .grad_mode import (
    set_grad_enabled as set_grad_enabled,
)


def backward(t) -> None:
    """Computes the sum of gradients of given tensors w.r.t. graph leaves.

    Args:
        t (Tensor): Tensors of which the derivative will be computed.
    """
    t.backward()
