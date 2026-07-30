"""Utilities for distributions."""


def _broadcast_shape(shape1, shape2):
    """Broadcasts two shapes."""
    if not shape1:
        return shape2
    if not shape2:
        return shape1
    res = []
    max_len = max(len(shape1), len(shape2))
    s1 = (1,) * (max_len - len(shape1)) + tuple(shape1)
    s2 = (1,) * (max_len - len(shape2)) + tuple(shape2)
    for d1, d2 in zip(s1, s2):
        if d1 != d2 and d1 != 1 and d2 != 1:
            raise RuntimeError(f"Shapes {shape1} and {shape2} are not broadcastable")
        res.append(max(d1, d2))
    return tuple(res)
