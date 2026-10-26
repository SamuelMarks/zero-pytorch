"""Domain module for top-level mathematical and distance operations."""

from __future__ import annotations

import math

import zero_torch
from zero_torch.tensor import Tensor, _flatten_nested, _nested_from_flat


def _frac_impl(input: Tensor, out: Tensor | None = None) -> Tensor:
    """Computes the fractional part of each element in `input`: `input - trunc(input)`.

    Args:
        input: Input tensor.
        out: Optional output tensor.

    Returns:
        Fractional part tensor.
    """
    res = input - zero_torch.trunc(input)
    if out is not None:
        return out.copy_(res)
    return res


def _cdist_impl(x1: Tensor, x2: Tensor, p: float = 2.0) -> Tensor:
    r"""Computes batched $L_p$ distance between each pair of row vectors in `x1` and `x2`.

    Args:
        x1: First tensor of shape `(..., P, M)`.
        x2: Second tensor of shape `(..., R, M)`.
        p: Norm degree $p \ge 0$.

    Returns:
        Distance tensor of shape `(..., P, R)`.
    """
    if len(x1.shape) >= 2 and len(x2.shape) >= 2 and x1.shape[-1] == x2.shape[-1]:
        diff = zero_torch.abs(
            zero_torch.unsqueeze(x1, -2) - zero_torch.unsqueeze(x2, -3)
        )
        if p == 0.0:
            return zero_torch.sum(
                zero_torch.where(
                    diff != 0.0,
                    zero_torch.ones_like(diff),
                    zero_torch.zeros_like(diff),
                ),
                axis=-1,
            )
        if math.isinf(p):
            return zero_torch.max(diff, axis=-1)
        return zero_torch.power(
            zero_torch.sum(zero_torch.power(diff, p), axis=-1), 1.0 / p
        )
    return x1


def _nan_replace_flat(
    input: Tensor,
    nan_val: float = 0.0,
    posinf_val: float = 1e38,
    neginf_val: float = -1e38,
) -> Tensor:
    """Replaces `NaN`, `+inf`, and `-inf` elements in `input`.

    Args:
        input: Input tensor.
        nan_val: Value to substitute for `NaN`.
        posinf_val: Value to substitute for `+inf`.
        neginf_val: Value to substitute for `-inf`.

    Returns:
        Cleaned tensor.
    """
    flat = _flatten_nested(input.tolist())
    cleaned: list[int | float | bool | complex] = []
    for v in flat:
        fv = float(v) if isinstance(v, (int, float)) else 0.0
        if math.isnan(fv):
            cleaned.append(nan_val)
        elif math.isinf(fv):
            cleaned.append(posinf_val if fv > 0 else neginf_val)
        else:
            cleaned.append(fv)
    if not input.shape:
        return Tensor(cleaned[0] if cleaned else 0.0, dtype=input.dtype)
    return Tensor(_nested_from_flat(cleaned, input.shape), dtype=input.dtype)
