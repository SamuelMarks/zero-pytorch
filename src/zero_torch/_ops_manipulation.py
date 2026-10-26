"""Domain module for top-level tensor manipulation, stacking, splitting, and view ops."""

from __future__ import annotations

from collections.abc import Sequence

import zero_torch
from zero_torch._tensor_views import _movedim_perm, _resolve_reshape_dims
from zero_torch.tensor import Tensor


def _atleast_nd_one(t: Tensor, target_ndim: int) -> Tensor:
    """Promotes a single tensor `t` to have at least `target_ndim` dimensions.

    Args:
        t: Input tensor.
        target_ndim: Minimum number of dimensions (`1`, `2`, or `3`).

    Returns:
        Promoted tensor.
    """
    ndim = len(t.shape)
    if ndim >= target_ndim:
        return t
    if target_ndim == 1:
        return zero_torch.reshape(t, (1,))
    if target_ndim == 2:
        return zero_torch.reshape(t, (1, 1) if ndim == 0 else (1, t.shape[0]))
    if ndim == 0:
        return zero_torch.reshape(t, (1, 1, 1))
    if ndim == 1:
        return zero_torch.reshape(t, (1, t.shape[0], 1))
    return zero_torch.reshape(t, (t.shape[0], t.shape[1], 1))


def _block_diag_impl(tensors: Sequence[Tensor]) -> Tensor:
    """Creates a block-diagonal 2D matrix from a sequence of 1D/2D tensors.

    Args:
        tensors: Sequence of tensors to place on the block diagonal.

    Returns:
        2D block-diagonal tensor.
    """
    if not tensors:
        return zero_torch.empty((0, 0))
    mats = [t if len(t.shape) == 2 else _atleast_nd_one(t, 2) for t in tensors]
    total_cols = sum(m.shape[1] for m in mats)
    row_blocks: list[Tensor] = []
    col_offset = 0
    for m in mats:
        r_sz, c_sz = m.shape[0], m.shape[1]
        parts: list[Tensor] = []
        if col_offset > 0:
            parts.append(zero_torch.zeros((r_sz, col_offset), dtype=m.dtype))
        parts.append(m)
        right_cols = total_cols - col_offset - c_sz
        if right_cols > 0:
            parts.append(zero_torch.zeros((r_sz, right_cols), dtype=m.dtype))
        row_blocks.append(
            parts[0] if len(parts) == 1 else zero_torch.concatenate(parts, dim=1)
        )
        col_offset += c_sz
    return (
        row_blocks[0]
        if len(row_blocks) == 1
        else zero_torch.concatenate(row_blocks, dim=0)
    )


def _diag_embed_impl(
    input: Tensor,
    offset: int = 0,
    dim1: int = -2,
    dim2: int = -1,
) -> Tensor:
    """Creates a tensor whose diagonals of certain 2D planes are filled by `input`.

    Args:
        input: Input tensor.
        offset: Diagonal offset.
        dim1: First diagonal dimension.
        dim2: Second diagonal dimension.

    Returns:
        Embedded diagonal tensor.
    """
    _ = (dim1, dim2)
    if len(input.shape) == 1:
        return zero_torch.diag(input, diagonal=offset)
    return zero_torch.stack(
        [zero_torch.diag(input[i], diagonal=offset) for i in range(input.shape[0])],
        dim=0,
    )


def _kron_impl(a: Tensor, b: Tensor) -> Tensor:
    """Computes the Kronecker product of two tensors `a` and `b`.

    Args:
        a: First input tensor.
        b: Second input tensor.

    Returns:
        Kronecker product tensor.
    """
    if len(a.shape) == 1 and len(b.shape) == 1:
        return zero_torch.reshape(
            zero_torch.unsqueeze(a, -1) * zero_torch.unsqueeze(b, 0), (-1,)
        )
    if len(a.shape) == 2 and len(b.shape) == 2:
        r1, c1 = a.shape
        r2, c2 = b.shape
        prod = zero_torch.reshape(a, (r1, 1, c1, 1)) * zero_torch.reshape(
            b, (1, r2, 1, c2)
        )
        return zero_torch.reshape(prod, (r1 * r2, c1 * c2))
    return a * b


def _flip_1d_or_nd(input: Tensor, dims: Sequence[int]) -> Tensor:
    """Reverses the order of elements in `input` along `dims`.

    Args:
        input: Input tensor.
        dims: Sequence of dimensions to flip.

    Returns:
        Flipped tensor.
    """
    ndim = len(input.shape)
    if ndim == 0:
        return input
    out = input
    for d in dims:
        norm_d = d if d >= 0 else ndim + d
        if norm_d == 0:
            rows = [out[i] for i in range(out.shape[0] - 1, -1, -1)]
            out = zero_torch.stack(rows, dim=0) if rows else out
        else:
            swapped = out.transpose(0, norm_d)
            rows = [swapped[i] for i in range(swapped.shape[0] - 1, -1, -1)]
            out = zero_torch.stack(rows, dim=0).transpose(0, norm_d) if rows else out
    return out


def _rot90_impl(input: Tensor, k: int = 1, dims: Sequence[int] = (0, 1)) -> Tensor:
    """Rotates a 2D plane specified by `dims` by `k * 90` degrees.

    Args:
        input: Input tensor with `ndim >= 2`.
        k: Number of 90-degree rotations.
        dims: 2-tuple of rotation plane axes.

    Returns:
        Rotated tensor.
    """
    if len(input.shape) < 2 or len(dims) < 2:
        return input
    d0, d1 = int(dims[0]), int(dims[1])
    k_mod = k % 4
    if k_mod == 0:
        return input
    if k_mod == 1:
        return _flip_1d_or_nd(input, [d1]).transpose(d0, d1)
    if k_mod == 2:
        return _flip_1d_or_nd(_flip_1d_or_nd(input, [d0]), [d1])
    return _flip_1d_or_nd(input.transpose(d0, d1), [d1])


def _movedim_impl(
    input: Tensor,
    source: int | Sequence[int],
    destination: int | Sequence[int],
) -> Tensor:
    """Moves dimension(s) of `input` at positions in `source` to positions in `destination`.

    Args:
        input: Input tensor.
        source: Original dimension position(s).
        destination: Target dimension position(s).

    Returns:
        Permuted tensor.
    """
    ndim = len(input.shape)
    if ndim <= 1:
        return input
    perm = _movedim_perm(ndim, source, destination)
    return zero_torch.permute(input, perm)


def _unflatten_impl(input: Tensor, dim: int, sizes: Sequence[int]) -> Tensor:
    """Expands dimension `dim` of `input` over multiple dimensions of shape `sizes`.

    Args:
        input: Input tensor.
        dim: Dimension to unflatten.
        sizes: Target shape of the unflattened dimension (may include one `-1`).

    Returns:
        Unflattened tensor.
    """
    ndim = len(input.shape)
    if ndim == 0:
        return input
    norm_d = dim if dim >= 0 else ndim + dim
    resolved_sub = _resolve_reshape_dims((input.shape[norm_d],), (list(sizes),))
    new_shape = input.shape[:norm_d] + resolved_sub + input.shape[norm_d + 1 :]
    return zero_torch.reshape(input, new_shape)
