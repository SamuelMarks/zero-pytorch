"""Utilities for probability distributions."""

from __future__ import annotations

from collections.abc import Sequence


def _broadcast_shape(shape1: Sequence[int], shape2: Sequence[int]) -> tuple[int, ...]:
    """Broadcasts two tensor shapes following standard NumPy/PyTorch broadcasting rules.

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
