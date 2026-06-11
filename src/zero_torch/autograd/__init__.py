"""Autograd module."""

from .grad_mode import no_grad, set_grad_enabled, is_grad_enabled


def backward(t) -> None:
    """Computes the sum of gradients of given tensors w.r.t. graph leaves.

    Args:
        t (Tensor): Tensors of which the derivative will be computed.
    """
    pass
