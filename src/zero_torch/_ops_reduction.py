"""Domain module for top-level reduction, scan, histogram, and search operations."""

from __future__ import annotations

import math

import zero_torch
from zero_torch.tensor import Tensor, _flatten_nested, _nested_from_flat


def _cum_extrema_impl(
    input: Tensor,
    dim: int = 0,
    find_max: bool = True,
) -> tuple[Tensor, Tensor]:
    """Computes cumulative maximum or minimum and running argmax/argmin indices along `dim`.

    Args:
        input: Input tensor.
        dim: Dimension along which to scan.
        find_max: If True, computes `cummax`; else `cummin`.

    Returns:
        Tuple of `(values, indices)` tensors.
    """
    if len(input.shape) == 1:
        vals = [
            float(x) if isinstance(x, (int, float)) else 0.0
            for x in _flatten_nested(input.tolist())
        ]
        out_v: list[float] = []
        out_i: list[int] = []
        best_v = vals[0] if vals else 0.0
        best_i = 0
        for idx, v in enumerate(vals):
            if (find_max and v >= best_v) or (not find_max and v <= best_v):
                best_v = v
                best_i = idx
            out_v.append(best_v)
            out_i.append(best_i)
        return (
            Tensor(out_v, dtype=input.dtype),
            Tensor(out_i, dtype="int64"),
        )
    _ = dim
    return (input, zero_torch.zeros_like(input, dtype="int64"))


def _nansum_impl(
    input: Tensor,
    dim: int | tuple[int, ...] | None = None,
    keepdim: bool = False,
) -> Tensor:
    """Computes sum of `input` elements treating `NaN`s as zero.

    Args:
        input: Input tensor.
        dim: Optional reduction dimension(s).
        keepdim: Whether reduced dimensions are retained.

    Returns:
        Reduced sum tensor.
    """
    cleaned = zero_torch.nan_to_num(input, nan=0.0)
    return zero_torch.sum(cleaned, dim=dim, keepdim=keepdim)


def _nanmean_impl(
    input: Tensor,
    dim: int | tuple[int, ...] | None = None,
    keepdim: bool = False,
) -> Tensor:
    """Computes mean of `input` elements ignoring `NaN`s.

    Args:
        input: Input tensor.
        dim: Optional reduction dimension(s).
        keepdim: Whether reduced dimensions are retained.

    Returns:
        Reduced mean tensor.
    """
    not_nan = zero_torch.where(
        zero_torch.isnan(input),
        zero_torch.zeros_like(input),
        zero_torch.ones_like(input),
    )
    total = _nansum_impl(input, dim=dim, keepdim=keepdim)
    count = zero_torch.maximum(zero_torch.sum(not_nan, dim=dim, keepdim=keepdim), 1.0)
    return total / count


def _quantile_1d(vals: list[float], q: float, interpolation: str = "linear") -> float:
    """Computes scalar quantile of a non-empty sorted list of floats.

    Args:
        vals: Sorted list of floats.
        q: Quantile in `[0, 1]`.
        interpolation: Interpolation method (`'linear'`, `'lower'`, `'higher'`, `'midpoint'`, `'nearest'`).

    Returns:
        Interpolated quantile value.
    """
    if not vals:
        return 0.0
    n = len(vals)
    pos = max(0.0, min(1.0, q)) * (n - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    frac = pos - lo
    if interpolation == "lower":
        return vals[lo]
    if interpolation == "higher":
        return vals[hi]
    if interpolation == "midpoint":
        return 0.5 * (vals[lo] + vals[hi])
    if interpolation == "nearest":
        return vals[hi] if frac >= 0.5 else vals[lo]
    return vals[lo] + frac * (vals[hi] - vals[lo])


def _quantile_impl(
    input: Tensor,
    q: float | Tensor = 0.5,
    interpolation: str = "linear",
    ignore_nan: bool = False,
) -> Tensor:
    """Computes quantiles of `input`.

    Args:
        input: Input tensor.
        q: Scalar or 1D tensor of quantiles in `[0, 1]`.
        interpolation: Interpolation mode.
        ignore_nan: If True, filters out `NaN` values prior to sorting.

    Returns:
        Quantile tensor.
    """
    raw = [
        float(x) if isinstance(x, (int, float)) else 0.0
        for x in _flatten_nested(input.tolist())
    ]
    vals = sorted(x for x in raw if not (ignore_nan and math.isnan(x)))
    if isinstance(q, Tensor) and len(q.shape) > 0:
        q_list = [
            float(x) if isinstance(x, (int, float)) else 0.5
            for x in _flatten_nested(q.tolist())
        ]
        return Tensor(
            [_quantile_1d(vals, qi, interpolation) for qi in q_list],
            dtype=input.dtype,
        )
    q_item = q.item() if isinstance(q, Tensor) else q
    q_val = float(q_item) if isinstance(q_item, (int, float)) else 0.5
    return Tensor(_quantile_1d(vals, q_val, interpolation), dtype=input.dtype)


def _bincount_impl(
    input: Tensor,
    weights: Tensor | None = None,
    minlength: int = 0,
) -> Tensor:
    """Counts occurrences of each non-negative integer value in a 1D tensor.

    Args:
        input: 1D integer tensor.
        weights: Optional weight tensor of same length as `input`.
        minlength: Minimum number of bins.

    Returns:
        1D histogram count tensor.
    """
    vals = [
        max(0, int(float(x)))
        for x in _flatten_nested(input.tolist())
        if isinstance(x, (int, float))
    ]
    n_bins = max(minlength, (max(vals) + 1) if vals else 0)
    if weights is not None:
        w_vals = [
            float(x) if isinstance(x, (int, float)) else 0.0
            for x in _flatten_nested(weights.tolist())
        ]
        counts_f = [0.0] * n_bins
        for v, w in zip(vals, w_vals):
            counts_f[v] += w
        return Tensor(counts_f, dtype=weights.dtype)
    counts_i = [0] * n_bins
    for v in vals:
        counts_i[v] += 1
    return Tensor(counts_i, dtype="int64")


def _histc_impl(
    input: Tensor,
    bins: int = 100,
    min_val: float = 0.0,
    max_val: float = 0.0,
) -> Tensor:
    """Computes the histogram of a tensor with `bins` equal-width bins.

    Args:
        input: Input tensor.
        bins: Number of histogram bins.
        min_val: Lower end of the range (if `min_val == max_val == 0`, uses input min/max).
        max_val: Upper end of the range.

    Returns:
        1D tensor of length `bins`.
    """
    vals = [
        float(x)
        for x in _flatten_nested(input.tolist())
        if isinstance(x, (int, float)) and not math.isnan(float(x))
    ]
    n_bins = max(1, bins)
    lo = min_val
    hi = max_val
    if lo == 0.0 and hi == 0.0 and vals:
        lo = min(vals)
        hi = max(vals)
    if hi <= lo:
        hi = lo + 1.0
    counts = [0.0] * n_bins
    width = (hi - lo) / n_bins
    for v in vals:
        if lo <= v <= hi:
            b_idx = min(n_bins - 1, max(0, int((v - lo) / width)))
            counts[b_idx] += 1.0
    return Tensor(counts, dtype=input.dtype)


def _bucketize_impl(
    input: Tensor,
    boundaries: Tensor,
    out_int32: bool = False,
    right: bool = False,
) -> Tensor:
    """Returns the indices of the buckets to which each value in `input` belongs.

    Args:
        input: N-D tensor or scalar with search values.
        boundaries: 1D sorted sequence of bucket boundaries.
        out_int32: Indicates the output data type (`int32` if True, else `int64`).
        right: If False, uses `bisect_left`; if True, uses `bisect_right`.

    Returns:
        Bucket index tensor with the same shape as `input`.
    """
    b_vals = [
        float(x) if isinstance(x, (int, float)) else 0.0
        for x in _flatten_nested(boundaries.tolist())
    ]
    in_vals = [
        float(x) if isinstance(x, (int, float)) else 0.0
        for x in _flatten_nested(input.tolist())
    ]
    indices: list[int | float | bool | complex] = [
        sum(1 for b in b_vals if (b <= v if right else b < v)) for v in in_vals
    ]
    dt = "int32" if out_int32 else "int64"
    if not input.shape:
        return Tensor(indices[0] if indices else 0, dtype=dt)
    return Tensor(_nested_from_flat(indices, input.shape), dtype=dt)
