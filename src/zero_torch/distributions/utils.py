"""Utilities for probability distributions."""

from __future__ import annotations

from collections.abc import Sequence

import zero_torch as torch
from zero_torch.tensor import Tensor, _flatten_nested


def _broadcast_shape(shape1: Sequence[int], shape2: Sequence[int]) -> tuple[int, ...]:
    """Broadcasts two tensor shapes following standard PyTorch broadcasting rules.

    Args:
        shape1: First shape sequence.
        shape2: Second shape sequence.

    Returns:
        Broadcasted shape tuple.
    """
    if not shape1:
        return tuple(shape2)
    if not shape2:
        return tuple(shape1)
    res: list[int] = []
    max_len = max(len(shape1), len(shape2))
    s1 = (1,) * (max_len - len(shape1)) + tuple(shape1)
    s2 = (1,) * (max_len - len(shape2)) + tuple(shape2)
    for d1, d2 in zip(s1, s2):
        if d1 != d2 and d1 != 1 and d2 != 1:
            raise RuntimeError(f"Shapes {shape1} and {shape2} are not broadcastable")
        res.append(max(d1, d2))
    return tuple(res)


def _check_positive(value: Tensor, name: str) -> None:
    """Validates that all elements of `value` are strictly positive (`> 0`).

    Args:
        value: Parameter tensor to validate.
        name: Parameter name for error reporting.
    """
    flat = _flatten_nested(value.tolist())
    if any(isinstance(v, (int, float)) and float(v) <= 0.0 for v in flat):
        raise ValueError(f"Expected parameter {name} to be positive")


def _check_non_negative(value: Tensor, name: str) -> None:
    """Validates that all elements of `value` are non-negative (`>= 0`).

    Args:
        value: Parameter tensor to validate.
        name: Parameter name for error reporting.
    """
    flat = _flatten_nested(value.tolist())
    if any(isinstance(v, (int, float)) and float(v) < 0.0 for v in flat):
        raise ValueError(f"Expected parameter {name} to be non-negative")


def _check_unit_interval(value: Tensor, name: str) -> None:
    """Validates that all elements of `value` lie in `[0, 1]`.

    Args:
        value: Parameter tensor to validate.
        name: Parameter name for error reporting.
    """
    flat = _flatten_nested(value.tolist())
    if any(
        isinstance(v, (int, float)) and (float(v) < 0.0 or float(v) > 1.0) for v in flat
    ):
        raise ValueError(f"Expected parameter {name} in [0, 1]")


def _check_less_than(low: Tensor, high: Tensor) -> None:
    """Validates that `low < high` element-wise after broadcasting.

    Args:
        low: Lower bound tensor.
        high: Upper bound tensor.
    """
    b_shape = _broadcast_shape(low.shape, high.shape)
    b_low = torch.broadcast_to(low, b_shape) if b_shape else low
    b_high = torch.broadcast_to(high, b_shape) if b_shape else high
    flat_low = _flatten_nested(b_low.tolist())
    flat_high = _flatten_nested(b_high.tolist())
    if any(
        isinstance(lo, (int, float))
        and isinstance(hi, (int, float))
        and float(lo) >= float(hi)
        for lo, hi in zip(flat_low, flat_high)
    ):
        raise ValueError("Uniform is not defined when low >= high")
