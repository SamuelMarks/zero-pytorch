"""Internal reduction and statistical helpers for `zero_torch.Tensor`."""

from __future__ import annotations

from collections.abc import Sequence

import zero_torch
from zero_torch.tensor import Tensor


def _tensor_aminmax(
    tensor: Tensor,
    dim: int | None = None,
    keepdim: bool = False,
) -> tuple[Tensor, Tensor]:
    """Computes both minimum and maximum of `tensor` along `dim`.

    Args:
        tensor: Input tensor.
        dim: Optional dimension along which to reduce.
        keepdim: Whether the output tensors have `dim` retained.

    Returns:
        Tuple of `(min_tensor, max_tensor)`.
    """
    return (
        zero_torch.amin(tensor, dim=dim, keepdim=keepdim),
        zero_torch.amax(tensor, dim=dim, keepdim=keepdim),
    )


def _tensor_std_var(
    tensor: Tensor,
    dim: int | Sequence[int] | None = None,
    correction: int = 1,
    keepdim: bool = False,
    take_sqrt: bool = True,
) -> Tensor:
    """Computes sample/population standard deviation or variance of `tensor`.

    Args:
        tensor: Input tensor.
        dim: Optional dimension or sequence of dimensions to reduce.
        correction: Bessel's correction (`ddof`).
        keepdim: Whether the output tensor has `dim` retained.
        take_sqrt: If True, returns standard deviation; otherwise variance.

    Returns:
        Reduced standard deviation or variance tensor.
    """
    if take_sqrt:
        return zero_torch.std(tensor, dim=dim, correction=correction, keepdim=keepdim)
    return zero_torch.var(tensor, dim=dim, correction=correction, keepdim=keepdim)
