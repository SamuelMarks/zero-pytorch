"""Autograd module."""

from __future__ import annotations

from zero_torch.tensor import Tensor

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


def backward(t: Tensor) -> None:
    """Computes the sum of gradients of given tensors w.r.t. graph leaves.

    Args:
        t: Tensor of which the derivative will be computed.
    """
    t.backward()
