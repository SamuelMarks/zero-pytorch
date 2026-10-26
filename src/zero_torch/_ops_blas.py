"""Domain module for top-level BLAS / matrix-vector linear algebra operations."""

from __future__ import annotations

import zero_torch
from zero_torch.tensor import Tensor


def _addmv_impl(
    input: Tensor,
    mat: Tensor,
    vec: Tensor,
    beta: float = 1.0,
    alpha: float = 1.0,
    out: Tensor | None = None,
) -> Tensor:
    """Computes `beta * input + alpha * (mat @ vec)`.

    Args:
        input: 1D vector to be added.
        mat: 2D matrix tensor.
        vec: 1D vector tensor.
        beta: Scaling factor for `input`.
        alpha: Scaling factor for `mat @ vec`.
        out: Optional output tensor.

    Returns:
        1D output tensor.
    """
    mv_prod = zero_torch.matmul(mat, vec)
    res = input * beta + mv_prod * alpha if beta != 0.0 else mv_prod * alpha
    if out is not None:
        return out.copy_(res)
    return res


def _addr_impl(
    input: Tensor,
    vec1: Tensor,
    vec2: Tensor,
    beta: float = 1.0,
    alpha: float = 1.0,
    out: Tensor | None = None,
) -> Tensor:
    """Computes outer-product update `beta * input + alpha * (vec1 (x) vec2)`.

    Args:
        input: 2D matrix to be added.
        vec1: First 1D vector.
        vec2: Second 1D vector.
        beta: Scaling factor for `input`.
        alpha: Scaling factor for outer product.
        out: Optional output tensor.

    Returns:
        2D output matrix tensor.
    """
    outer = zero_torch.unsqueeze(vec1, -1) * zero_torch.unsqueeze(vec2, 0)
    res = input * beta + outer * alpha if beta != 0.0 else outer * alpha
    if out is not None:
        return out.copy_(res)
    return res


def _baddbmm_impl(
    input: Tensor,
    batch1: Tensor,
    batch2: Tensor,
    beta: float = 1.0,
    alpha: float = 1.0,
    out: Tensor | None = None,
) -> Tensor:
    """Computes batched matrix-matrix product with addition `beta * input + alpha * (batch1 @ batch2)`.

    Args:
        input: Input tensor to be added.
        batch1: First 3D batch of matrices.
        batch2: Second 3D batch of matrices.
        beta: Scaling factor for `input`.
        alpha: Scaling factor for `batch1 @ batch2`.
        out: Optional output tensor.

    Returns:
        3D output batch tensor.
    """
    bmm_prod = zero_torch.matmul(batch1, batch2)
    res = input * beta + bmm_prod * alpha if beta != 0.0 else bmm_prod * alpha
    if out is not None:
        return out.copy_(res)
    return res


def _addbmm_impl(
    input: Tensor,
    batch1: Tensor,
    batch2: Tensor,
    beta: float = 1.0,
    alpha: float = 1.0,
    out: Tensor | None = None,
) -> Tensor:
    """Computes `beta * input + alpha * sum_i(batch1[i] @ batch2[i])`.

    Args:
        input: 2D matrix to be added.
        batch1: First 3D batch of matrices `(B, N, M)`.
        batch2: Second 3D batch of matrices `(B, M, P)`.
        beta: Scaling factor for `input`.
        alpha: Scaling factor for summed batch products.
        out: Optional output tensor.

    Returns:
        2D output matrix tensor `(N, P)`.
    """
    bmm_sum = zero_torch.sum(zero_torch.matmul(batch1, batch2), axis=0)
    res = input * beta + bmm_sum * alpha if beta != 0.0 else bmm_sum * alpha
    if out is not None:
        return out.copy_(res)
    return res
