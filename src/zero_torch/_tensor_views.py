"""Internal view, stride, and shape manipulation helpers for `zero_torch.Tensor`."""

from __future__ import annotations

import functools
import operator
from collections.abc import Sequence

from zero_torch._types import ShapeLike


def _resolve_reshape_dims(
    orig_shape: tuple[int, ...],
    shape_args: tuple[ShapeLike, ...],
) -> tuple[int, ...]:
    """Normalizes target shape arguments and resolves at most one `-1` inferred dimension.

    Args:
        orig_shape: Current shape tuple of the tensor.
        shape_args: Positional shape arguments passed to `view` or `reshape`.

    Returns:
        Resolved positive integer shape tuple.
    """
    if len(shape_args) == 1 and isinstance(shape_args[0], (list, tuple)):
        raw_dims: Sequence[int] = shape_args[0]
    else:
        raw_dims = [int(s) for s in shape_args if isinstance(s, int)]

    clean_shape = list(raw_dims)
    if -1 in clean_shape:
        neg_idx = clean_shape.index(-1)
        known_prod = 1
        for i, s in enumerate(clean_shape):
            if i != neg_idx:
                known_prod *= s
        total_elements = functools.reduce(operator.mul, orig_shape, 1)
        clean_shape[neg_idx] = total_elements // max(1, known_prod)
    return tuple(clean_shape)


def _movedim_perm(
    ndim: int,
    source: int | Sequence[int],
    destination: int | Sequence[int],
) -> tuple[int, ...]:
    """Computes the axis permutation tuple for `movedim(input, source, destination)`.

    Args:
        ndim: Number of dimensions of the input tensor.
        source: Original positions of the dimensions to move.
        destination: Destination positions for each of the original dimensions.

    Returns:
        Permutation tuple of length `ndim`.
    """
    src_list = [source] if isinstance(source, int) else list(source)
    dst_list = [destination] if isinstance(destination, int) else list(destination)
    norm_src = [s if s >= 0 else ndim + s for s in src_list]
    norm_dst = [d if d >= 0 else ndim + d for d in dst_list]
    order = [i for i in range(ndim) if i not in norm_src]
    for dst_ax, src_ax in sorted(zip(norm_dst, norm_src)):
        order.insert(dst_ax, src_ax)
    return tuple(order)
