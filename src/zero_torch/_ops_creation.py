"""Domain module for top-level tensor creation and coordinate index operations."""

from __future__ import annotations

from collections.abc import Sequence

from zero_torch._types import DTypeLike, OpArg
from zero_torch.tensor import Tensor, _flatten_nested


def _tril_indices_impl(
    row: int,
    col: int,
    offset: int = 0,
    dtype: DTypeLike = "int64",
) -> Tensor:
    """Computes lower-triangular coordinate indices of a `(row, col)` matrix as a `(2, N)` tensor.

    Args:
        row: Number of rows in the 2D matrix.
        col: Number of columns in the 2D matrix.
        offset: Diagonal offset from the main diagonal.
        dtype: Desired integer data type.

    Returns:
        `(2, N)` tensor of `(row_indices, col_indices)`.
    """
    r_idx: list[int] = []
    c_idx: list[int] = []
    for r in range(max(0, row)):
        for c in range(max(0, col)):
            if c - r <= offset:
                r_idx.append(r)
                c_idx.append(c)
    return Tensor([r_idx, c_idx], dtype=dtype)


def _triu_indices_impl(
    row: int,
    col: int,
    offset: int = 0,
    dtype: DTypeLike = "int64",
) -> Tensor:
    """Computes upper-triangular coordinate indices of a `(row, col)` matrix as a `(2, N)` tensor.

    Args:
        row: Number of rows in the 2D matrix.
        col: Number of columns in the 2D matrix.
        offset: Diagonal offset from the main diagonal.
        dtype: Desired integer data type.

    Returns:
        `(2, N)` tensor of `(row_indices, col_indices)`.
    """
    r_idx: list[int] = []
    c_idx: list[int] = []
    for r in range(max(0, row)):
        for c in range(max(0, col)):
            if c - r >= offset:
                r_idx.append(r)
                c_idx.append(c)
    return Tensor([r_idx, c_idx], dtype=dtype)


def _cartesian_prod_impl(tensors: Sequence[Tensor]) -> Tensor:
    """Computes the Cartesian product of a sequence of 1D tensors.

    Args:
        tensors: Sequence of 1D input tensors.

    Returns:
        1D tensor if `len(tensors) == 1`, else 2D tensor of all combinations.
    """
    if len(tensors) == 1:
        return tensors[0]
    lists = [
        [
            float(x) if isinstance(x, (int, float)) else 0.0
            for x in _flatten_nested(t.tolist())
        ]
        for t in tensors
    ]
    pools: list[list[float]] = [[]]
    for lst in lists:
        pools = [x + [y] for x in pools for y in lst]
    return Tensor(pools, dtype=tensors[0].dtype)


def _combinations_impl(
    input: Tensor, r: int = 2, with_replacement: bool = False
) -> Tensor:
    """Computes combinations of length `r` from a 1D tensor `input`.

    Args:
        input: 1D input tensor.
        r: Number of elements per combination.
        with_replacement: Whether to allow repeated elements.

    Returns:
        Tensor of combinations of shape `(num_combinations, r)`.
    """
    import itertools

    vals = [
        float(x) if isinstance(x, (int, float)) else 0.0
        for x in _flatten_nested(input.tolist())
    ]
    combos = (
        list(itertools.combinations_with_replacement(vals, r))
        if with_replacement
        else list(itertools.combinations(vals, r))
    )
    return Tensor([list(c) for c in combos], dtype=input.dtype)


def _broadcast_shapes_impl(*shapes: OpArg) -> tuple[int, ...]:
    """Computes the broadcasted shape across multiple shape tuples, raising RuntimeError if incompatible.

    Args:
        *shapes: Shape tuples or integers.

    Returns:
        Broadcasted shape tuple.
    """
    norm_shapes: list[tuple[int, ...]] = []
    for s in shapes:
        if isinstance(s, (list, tuple)):
            norm_shapes.append(tuple(int(x) for x in s if isinstance(x, int)))
        else:
            assert isinstance(s, int)
            norm_shapes.append((s,))
    if not norm_shapes:
        return (1,)
    max_ndim = max(len(s) for s in norm_shapes)
    result = [1] * max_ndim
    for s in norm_shapes:
        padded = (1,) * (max_ndim - len(s)) + s
        for i in range(max_ndim):
            if result[i] == 1:
                result[i] = padded[i]
            elif padded[i] not in (1, result[i]):
                raise RuntimeError(
                    f"Shape mismatch: objects cannot be broadcast to a single shape: {norm_shapes}"
                )
    return tuple(result)
