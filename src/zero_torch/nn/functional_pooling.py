"""Functional pooling, unpooling, fold, and unfold operations."""

from __future__ import annotations

import functools
import itertools
import math
import operator
from collections.abc import Sequence

from ml_switcheroo_compiler.core import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.ops.reductions.frontend_utils import _emit_reduction_node

import zero_torch
from zero_torch._types import (
    OptSize1d,
    OptSize2d,
    OptSize3d,
    Size1d,
    Size2d,
    Size3d,
)
from zero_torch.tensor import Tensor, _flatten_nested, _to_tensor, _wrap


def _format_padding(
    padding: int | tuple[int, ...], ndims: int
) -> tuple[tuple[int, int], ...]:
    """Formats an integer or tuple padding into `((pad_before, pad_after), ...)` pairs.

    Args:
        padding: Integer or tuple of per-dimension padding values.
        ndims: Number of spatial dimensions.

    Returns:
        Tuple of `(pad_before, pad_after)` pairs of length `ndims`.
    """
    if isinstance(padding, int):
        return tuple((padding, padding) for _ in range(ndims))
    return tuple((int(p), int(p)) for p in padding)


def _normalize_tuple(val: int | Sequence[int], ndims: int) -> tuple[int, ...]:
    """Normalizes an integer or sequence of integers into an `ndims`-tuple.

    Args:
        val: Integer or sequence of integers.
        ndims: Target tuple length.

    Returns:
        Tuple of `ndims` integers.
    """
    if isinstance(val, int):
        return (val,) * ndims
    return tuple(int(v) for v in val)


def _resolve_opt_size(
    output_size: int | Sequence[int | None] | None,
    ndims: int,
    in_spatial: tuple[int, ...],
) -> tuple[int, ...]:
    """Resolves an optional adaptive output size into a concrete `ndims`-tuple of ints.

    Args:
        output_size: Target output size (int, tuple with optional Nones, or None).
        ndims: Number of spatial dimensions.
        in_spatial: Input spatial dimensions tuple.

    Returns:
        Resolved output spatial shape of length `ndims`.
    """
    if output_size is None:
        return in_spatial
    if isinstance(output_size, int):
        return (output_size,) * ndims
    return tuple(
        in_spatial[i] if s is None else int(s) for i, s in enumerate(output_size)
    )


def _pool_out_dim(in_size: int, k: int, s: int, p: int, d: int, ceil_mode: bool) -> int:
    """Computes the output spatial size along a single pooled dimension.

    Args:
        in_size: Input dimension size.
        k: Kernel size.
        s: Stride size.
        p: Symmetric padding size.
        d: Dilation factor.
        ceil_mode: Whether to use ceiling division for output size.

    Returns:
        Output dimension size.
    """
    k_eff = d * (k - 1) + 1
    if ceil_mode:
        out = (in_size + 2 * p - k_eff + s - 1) // s + 1
        out -= int((out - 1) * s >= in_size + p)
        return max(1, out)
    return max(1, (in_size + 2 * p - k_eff) // s + 1)


def _build_sliding_windows(
    in_spatial: tuple[int, ...],
    k: tuple[int, ...],
    s: tuple[int, ...],
    p: tuple[int, ...],
    d: tuple[int, ...],
    ceil_mode: bool,
) -> tuple[list[list[list[int]]], list[list[int]], tuple[int, ...]]:
    """Constructs valid input coordinate lists and window spans for each spatial dimension.

    Args:
        in_spatial: Input spatial dimensions.
        k: Kernel sizes per spatial dimension.
        s: Strides per spatial dimension.
        p: Paddings per spatial dimension.
        d: Dilations per spatial dimension.
        ceil_mode: Whether to use ceiling division for output shape.

    Returns:
        Tuple of `(valid_coords_per_dim, window_spans_per_dim, out_spatial_shape)`.
    """
    ndims = len(in_spatial)
    valid_coords: list[list[list[int]]] = []
    window_spans: list[list[int]] = []
    out_spatial: list[int] = []

    for m in range(ndims):
        in_len = in_spatial[m]
        km, sm, pm, dm = k[m], s[m], p[m], d[m]
        out_len = _pool_out_dim(in_len, km, sm, pm, dm, ceil_mode)
        out_spatial.append(out_len)
        dim_coords: list[list[int]] = []
        dim_spans: list[int] = []
        for o_idx in range(out_len):
            start = o_idx * sm - pm
            coords = [start + r * dm for r in range(km) if 0 <= start + r * dm < in_len]
            dim_coords.append(coords)
            end_padded = min(start + km, in_len + pm)
            start_padded = max(start, -pm)
            dim_spans.append(max(1, end_padded - start_padded))
        valid_coords.append(dim_coords)
        window_spans.append(dim_spans)

    return valid_coords, window_spans, tuple(out_spatial)


def _build_adaptive_windows(
    in_spatial: tuple[int, ...], out_spatial: tuple[int, ...]
) -> list[list[list[int]]]:
    """Constructs dynamic start/end window coordinate lists for adaptive pooling.

    Args:
        in_spatial: Input spatial dimensions.
        out_spatial: Target output spatial dimensions.

    Returns:
        Per-dimension list of window coordinate lists.
    """
    valid_coords: list[list[list[int]]] = []
    for in_len, out_len in zip(in_spatial, out_spatial):
        dim_coords: list[list[int]] = []
        for i in range(out_len):
            start = (i * in_len) // out_len
            end = ((i + 1) * in_len + out_len - 1) // out_len
            dim_coords.append(list(range(start, max(start + 1, end))))
        valid_coords.append(dim_coords)
    return valid_coords


def _spatial_strides(in_spatial: tuple[int, ...]) -> list[int]:
    """Computes row-major flattened spatial strides for `in_spatial`.

    Args:
        in_spatial: Spatial shape tuple.

    Returns:
        List of row-major strides of length `len(in_spatial)`.
    """
    ndims = len(in_spatial)
    strides = [1] * ndims
    for m in range(ndims - 2, -1, -1):
        strides[m] = strides[m + 1] * in_spatial[m + 1]
    return strides


def _to_float_list(t: Tensor) -> list[float]:
    """Flattens tensor `t` into a 1D Python list of `float` values.

    Args:
        t: Input tensor.

    Returns:
        Flattened list of floats.
    """
    return [
        float(v.real) if isinstance(v, complex) else float(v)
        for v in _flatten_nested(t.tolist())
    ]


def _extract_flat_channels(
    input: Tensor, batch_channels: int, spatial_numel: int
) -> list[list[float]]:
    """Extracts a 2D `[batch_channels][spatial_numel]` float matrix from `input`.

    Args:
        input: Input tensor.
        batch_channels: Product of leading non-spatial dimensions.
        spatial_numel: Product of spatial dimensions.

    Returns:
        Nested list of shape `[batch_channels][spatial_numel]`.
    """
    flat = _to_float_list(input)
    return [
        flat[b * spatial_numel : (b + 1) * spatial_numel] for b in range(batch_channels)
    ]


def _adaptive_avg_pool_axis(x: Tensor, out_len: int, axis: int) -> Tensor:
    """Applies 1D adaptive average pooling along a single `axis` using tensor primitives.

    Args:
        x: Input tensor.
        out_len: Target length along `axis`.
        axis: Negative or non-negative dimension index.

    Returns:
        Pooled tensor along `axis`.
    """
    ndim = len(x.shape)
    norm_axis = axis if axis >= 0 else ndim + axis
    in_len = x.shape[norm_axis]
    if out_len == in_len:
        return x
    prefix = tuple(slice(None) for _ in range(norm_axis))
    pieces: list[Tensor] = []
    for i in range(out_len):
        start = (i * in_len) // out_len
        end = ((i + 1) * in_len + out_len - 1) // out_len
        slc = x[prefix + (slice(start, max(start + 1, end)),)]
        pieces.append(zero_torch.mean(slc, axis=norm_axis, keepdims=True))
    return zero_torch.concatenate(pieces, dim=norm_axis)


def _adaptive_avg_pool_nd(
    input: Tensor, output_size: int | Sequence[int | None] | None, ndims: int
) -> Tensor:
    """Applies `ndims`-D adaptive average pooling over the last `ndims` dimensions.

    Args:
        input: Input tensor.
        output_size: Target output spatial size.
        ndims: Number of spatial dimensions (1, 2, or 3).

    Returns:
        Pooled tensor.
    """
    if len(input.shape) < ndims:
        return input
    in_spatial = input.shape[-ndims:]
    target = _resolve_opt_size(output_size, ndims, in_spatial)
    if not config.eager_mode:
        out_shape = input.shape[:-ndims] + target
        return _wrap(
            _emit_reduction_node(
                f"AdaptiveAvgPool{ndims}D",
                [_to_tensor(input)],
                {"output_size": target},
                out_shape,
                input.dtype,
            )
        )
    out = input
    for i, out_len in enumerate(target):
        out = _adaptive_avg_pool_axis(out, out_len, axis=-ndims + i)
    return out


def _max_pool_from_coord_windows(
    input: Tensor,
    valid_coords: Sequence[Sequence[Sequence[int]]],
    out_spatial: tuple[int, ...],
    return_indices: bool,
    op_name: str,
) -> Tensor | tuple[Tensor, Tensor]:
    """Executes max pooling (with optional argmax indices) over per-dimension coordinate windows.

    Args:
        input: Input tensor.
        valid_coords: Per-dimension list of window coordinate lists.
        out_spatial: Output spatial dimensions tuple.
        return_indices: Whether to return flattened spatial argmax indices.
        op_name: Operator name for tracing mode.

    Returns:
        Pooled tensor, or `(output, indices)` when `return_indices=True`.
    """
    ndims = len(out_spatial)
    lead_shape = input.shape[:-ndims]
    in_spatial = input.shape[-ndims:]
    out_shape = lead_shape + out_spatial

    if not config.eager_mode:
        pooled_node = _wrap(
            _emit_reduction_node(
                op_name,
                [_to_tensor(input)],
                {"output_size": out_spatial},
                out_shape,
                input.dtype,
            )
        )
        if return_indices:
            idx_node = _wrap(
                _emit_reduction_node(
                    f"{op_name}_Indices",
                    [_to_tensor(input)],
                    {"output_size": out_spatial},
                    out_shape,
                    DType.int64,
                )
            )
            return pooled_node, idx_node
        return pooled_node

    batch_channels = functools.reduce(operator.mul, lead_shape, 1)
    spatial_numel = functools.reduce(operator.mul, in_spatial, 1)
    strides_in = _spatial_strides(in_spatial)
    channels = _extract_flat_channels(input, batch_channels, spatial_numel)

    flat_windows: list[list[int]] = []
    for win_coords in itertools.product(*valid_coords):
        flat_idxs = [
            sum(c_m * strides_in[m] for m, c_m in enumerate(pt))
            for pt in itertools.product(*win_coords)
        ]
        flat_windows.append(flat_idxs)

    out_vals: list[float] = []
    out_idxs: list[int] = []
    for row in channels:
        for f_win in flat_windows:
            best_idx = f_win[0]
            best_val = row[best_idx]
            for fi in f_win[1:]:
                val = row[fi]
                if val > best_val:
                    best_val = val
                    best_idx = fi
            out_vals.append(best_val)
            out_idxs.append(best_idx)

    out_tensor = zero_torch.reshape(Tensor(out_vals, dtype=input.dtype), out_shape)
    if return_indices:
        idx_tensor = zero_torch.reshape(Tensor(out_idxs, dtype=DType.Int64), out_shape)
        return out_tensor, idx_tensor
    return out_tensor


def _avg_pool_nd(
    input: Tensor,
    kernel_size: int | Sequence[int],
    stride: int | Sequence[int] | None,
    padding: int | Sequence[int],
    ceil_mode: bool,
    count_include_pad: bool,
    divisor_override: int | None,
    ndims: int,
) -> Tensor:
    """Applies `ndims`-D average pooling over the last `ndims` spatial dimensions.

    Args:
        input: Input tensor.
        kernel_size: Pooling window size.
        stride: Pooling window stride (defaults to `kernel_size`).
        padding: Implicit zero padding on both sides.
        ceil_mode: Whether to use ceiling division for output shape.
        count_include_pad: Whether to include zero-padding in the averaging divisor.
        divisor_override: Optional explicit divisor overriding window size.
        ndims: Number of spatial dimensions (1, 2, or 3).

    Returns:
        Pooled tensor.
    """
    k = _normalize_tuple(kernel_size, ndims)
    s = k if stride is None else _normalize_tuple(stride, ndims)
    p = _normalize_tuple(padding, ndims)
    d = (1,) * ndims
    lead_shape = input.shape[:-ndims]
    in_spatial = input.shape[-ndims:]
    valid_coords, window_spans, out_spatial = _build_sliding_windows(
        in_spatial, k, s, p, d, ceil_mode
    )
    out_shape = lead_shape + out_spatial

    if not config.eager_mode:
        return _wrap(
            _emit_reduction_node(
                f"AvgPool{ndims}D",
                [_to_tensor(input)],
                {"kernel_size": k, "stride": s, "padding": p},
                out_shape,
                input.dtype,
            )
        )

    batch_channels = functools.reduce(operator.mul, lead_shape, 1)
    spatial_numel = functools.reduce(operator.mul, in_spatial, 1)
    strides_in = _spatial_strides(in_spatial)
    channels = _extract_flat_channels(input, batch_channels, spatial_numel)

    window_specs: list[tuple[list[int], float]] = []
    out_ranges: list[list[int]] = [list(range(o)) for o in out_spatial]
    for o_tuple in itertools.product(*out_ranges):
        win_coords = [valid_coords[m][o_tuple[m]] for m in range(ndims)]
        flat_idxs = [
            sum(c_m * strides_in[m] for m, c_m in enumerate(pt))
            for pt in itertools.product(*win_coords)
        ]
        if divisor_override is not None:
            denom = float(divisor_override)
        elif count_include_pad:
            span_prod = 1
            for m in range(ndims):
                span_prod *= window_spans[m][o_tuple[m]]
            denom = float(span_prod)
        else:
            denom = float(max(1, len(flat_idxs)))
        window_specs.append((flat_idxs, denom))

    out_vals: list[float] = []
    for row in channels:
        for flat_idxs, denom in window_specs:
            total = sum(row[fi] for fi in flat_idxs)
            out_vals.append(total / denom)

    return zero_torch.reshape(Tensor(out_vals, dtype=input.dtype), out_shape)


def _lp_pool_nd(
    input: Tensor,
    norm_type: float,
    kernel_size: int | Sequence[int],
    stride: int | Sequence[int] | None,
    ceil_mode: bool,
    ndims: int,
) -> Tensor:
    """Applies `ndims`-D power-average (`LP`) pooling over the last `ndims` spatial dimensions.

    Args:
        input: Input tensor.
        norm_type: Exponent `p` for LP-pooling.
        kernel_size: Size of the pooling window.
        stride: Stride of the pooling window (defaults to `kernel_size`).
        ceil_mode: Whether to use ceiling division for output shape.
        ndims: Number of spatial dimensions (1, 2, or 3).

    Returns:
        Pooled tensor.
    """
    p_val = float(norm_type)
    powered = zero_torch.abs(input) if p_val == 1.0 else zero_torch.abs(input) ** p_val
    summed = _avg_pool_nd(
        powered,
        kernel_size=kernel_size,
        stride=stride,
        padding=0,
        ceil_mode=ceil_mode,
        count_include_pad=True,
        divisor_override=1,
        ndims=ndims,
    )
    if p_val == 1.0:
        return summed
    return summed ** (1.0 / p_val)


def _fractional_max_pool_nd(
    input: Tensor,
    kernel_size: int | Sequence[int],
    output_size: int | Sequence[int] | None,
    output_ratio: float | Sequence[float] | None,
    return_indices: bool,
    _random_samples: Tensor | None,
    ndims: int,
) -> Tensor | tuple[Tensor, Tensor]:
    """Applies `ndims`-D fractional max pooling over the last `ndims` spatial dimensions.

    Args:
        input: Input tensor of shape `(*lead, *spatial)`.
        kernel_size: Pooling window size.
        output_size: Target output spatial size.
        output_ratio: Target output spatial ratio in `(0, 1)`.
        return_indices: Whether to return argmax indices alongside pooled outputs.
        _random_samples: Optional pre-generated random samples in `[0, 1)`.
        ndims: Number of spatial dimensions (2 or 3).

    Returns:
        Pooled tensor, or `(output, indices)` if `return_indices=True`.
    """
    if len(input.shape) < ndims:
        return input
    k = _normalize_tuple(kernel_size, ndims)
    lead_shape = input.shape[:-ndims]
    in_spatial = input.shape[-ndims:]

    if output_size is not None:
        out_spatial = _normalize_tuple(output_size, ndims)
    elif output_ratio is not None:
        ratios = (
            (float(output_ratio),) * ndims
            if isinstance(output_ratio, (int, float))
            else tuple(float(r) for r in output_ratio)
        )
        out_spatial = tuple(
            max(1, math.floor(in_spatial[m] * ratios[m])) for m in range(ndims)
        )
    else:
        raise ValueError("Either output_size or output_ratio must be specified")

    out_shape = lead_shape + out_spatial
    if not config.eager_mode:
        pooled_node = _wrap(
            _emit_reduction_node(
                f"FractionalMaxPool{ndims}D",
                [_to_tensor(input)],
                {"kernel_size": k, "output_size": out_spatial},
                out_shape,
                input.dtype,
            )
        )
        if return_indices:
            idx_node = _wrap(
                _emit_reduction_node(
                    f"FractionalMaxPool{ndims}D_Indices",
                    [_to_tensor(input)],
                    {"kernel_size": k, "output_size": out_spatial},
                    out_shape,
                    DType.int64,
                )
            )
            return pooled_node, idx_node
        return pooled_node

    batch_channels = functools.reduce(operator.mul, lead_shape, 1)
    spatial_numel = functools.reduce(operator.mul, in_spatial, 1)
    strides_in = _spatial_strides(in_spatial)
    channels = _extract_flat_channels(input, batch_channels, spatial_numel)

    if _random_samples is not None:
        raw_samples = _to_float_list(_random_samples)
    else:
        rand_t = zero_torch.rand((batch_channels, ndims))
        raw_samples = _to_float_list(rand_t)

    out_vals: list[float] = []
    out_idxs: list[int] = []
    for b, row in enumerate(channels):
        b_coords: list[list[list[int]]] = []
        for m in range(ndims):
            u_idx = (b * ndims + m) % max(1, len(raw_samples))
            u = raw_samples[u_idx] if raw_samples else 0.5
            in_len = in_spatial[m]
            out_len = out_spatial[m]
            km = min(k[m], in_len)
            alpha = (in_len - km) / float(out_len - 1) if out_len > 1 else 0.0
            dim_wins: list[list[int]] = []
            for i in range(out_len):
                start = (
                    int((i + u) * alpha) - int(u * alpha)
                    if i < out_len - 1
                    else in_len - km
                )
                start = max(0, min(start, in_len - km))
                dim_wins.append(list(range(start, start + km)))
            b_coords.append(dim_wins)

        for win_coords in itertools.product(*b_coords):
            flat_idxs = [
                sum(c_m * strides_in[m] for m, c_m in enumerate(pt))
                for pt in itertools.product(*win_coords)
            ]
            best_idx = flat_idxs[0]
            best_val = row[best_idx]
            for fi in flat_idxs[1:]:
                val = row[fi]
                if val > best_val:
                    best_val = val
                    best_idx = fi
            out_vals.append(best_val)
            out_idxs.append(best_idx)

    out_tensor = zero_torch.reshape(Tensor(out_vals, dtype=input.dtype), out_shape)
    if return_indices:
        idx_tensor = zero_torch.reshape(Tensor(out_idxs, dtype=DType.Int64), out_shape)
        return out_tensor, idx_tensor
    return out_tensor


def _max_unpool_nd(
    input: Tensor,
    indices: Tensor,
    kernel_size: int | Sequence[int],
    stride: int | Sequence[int] | None,
    padding: int | Sequence[int],
    output_size: Sequence[int] | None,
    ndims: int,
) -> Tensor:
    """Computes the partial inverse of `MaxPool{ndims}d` by scattering values at `indices`.

    Args:
        input: Pooled input tensor.
        indices: Flattened spatial indices produced by `MaxPool{ndims}d`.
        kernel_size: Size of the max pooling window.
        stride: Stride of the max pooling window (defaults to `kernel_size`).
        padding: Padding that was added to the input.
        output_size: Optional target output size (`(*lead, *spatial)` or `(*spatial,)`).
        ndims: Number of spatial dimensions (1, 2, or 3).

    Returns:
        Unpooled tensor.
    """
    k = _normalize_tuple(kernel_size, ndims)
    s = k if stride is None else _normalize_tuple(stride, ndims)
    p = _normalize_tuple(padding, ndims)
    lead_shape = input.shape[:-ndims]
    in_spatial = input.shape[-ndims:]

    if output_size is not None:
        if len(output_size) == ndims:
            out_spatial = tuple(int(x) for x in output_size)
        elif len(output_size) == len(input.shape):
            out_spatial = tuple(int(x) for x in output_size[-ndims:])
        else:
            raise ValueError(
                f"output_size must have {ndims} or {len(input.shape)} elements, got {len(output_size)}"
            )
    else:
        out_spatial = tuple(
            (in_spatial[m] - 1) * s[m] - 2 * p[m] + k[m] for m in range(ndims)
        )

    out_shape = lead_shape + out_spatial
    if not config.eager_mode:
        return _wrap(
            _emit_reduction_node(
                f"MaxUnpool{ndims}D",
                [_to_tensor(input), _to_tensor(indices)],
                {
                    "kernel_size": k,
                    "stride": s,
                    "padding": p,
                    "output_size": out_spatial,
                },
                out_shape,
                input.dtype,
            )
        )

    batch_channels = functools.reduce(operator.mul, lead_shape, 1)
    in_numel = functools.reduce(operator.mul, in_spatial, 1)
    out_numel = functools.reduce(operator.mul, out_spatial, 1)

    val_channels = _extract_flat_channels(input, batch_channels, in_numel)
    idx_channels = _extract_flat_channels(indices, batch_channels, in_numel)

    out_flat: list[float] = []
    for v_row, i_row in zip(val_channels, idx_channels):
        buf = [0.0] * out_numel
        for val, raw_idx in zip(v_row, i_row):
            buf[int(raw_idx)] = val
        out_flat.extend(buf)

    return zero_torch.reshape(Tensor(out_flat, dtype=input.dtype), out_shape)


def adaptive_avg_pool1d(input: Tensor, output_size: OptSize1d) -> Tensor:
    """Applies a 1D adaptive average pooling over an input signal.

    Args:
        input: Input tensor of shape `(N, C, L)` or `(C, L)`.
        output_size: Target output length `L_out`.

    Returns:
        Pooled tensor.
    """
    return _adaptive_avg_pool_nd(input, output_size, 1)


def adaptive_avg_pool2d(input: Tensor, output_size: OptSize2d) -> Tensor:
    """Applies a 2D adaptive average pooling over an input signal.

    Args:
        input: Input tensor of shape `(N, C, H, W)` or `(C, H, W)`.
        output_size: Target output spatial size `(H_out, W_out)` or `H_out`.

    Returns:
        Pooled tensor.
    """
    return _adaptive_avg_pool_nd(input, output_size, 2)


def adaptive_avg_pool3d(input: Tensor, output_size: OptSize3d) -> Tensor:
    """Applies a 3D adaptive average pooling over an input signal.

    Args:
        input: Input tensor of shape `(N, C, D, H, W)` or `(C, D, H, W)`.
        output_size: Target output volumetric size `(D_out, H_out, W_out)` or `D_out`.

    Returns:
        Pooled tensor.
    """
    return _adaptive_avg_pool_nd(input, output_size, 3)


def adaptive_max_pool1d(
    input: Tensor, output_size: OptSize1d, return_indices: bool = False
) -> Tensor | tuple[Tensor, Tensor]:
    """Applies a 1D adaptive max pooling over an input signal.

    Args:
        input: Input tensor of shape `(N, C, L)` or `(C, L)`.
        output_size: Target output length `L_out`.
        return_indices: Whether to return argmax indices alongside pooled outputs.

    Returns:
        Pooled tensor, or `(output, indices)` if `return_indices=True`.
    """
    in_spatial = input.shape[-1:]
    out_spatial = _resolve_opt_size(output_size, 1, in_spatial)
    valid_coords = _build_adaptive_windows(in_spatial, out_spatial)
    return _max_pool_from_coord_windows(
        input, valid_coords, out_spatial, return_indices, "AdaptiveMaxPool1D"
    )


def adaptive_max_pool2d(
    input: Tensor, output_size: OptSize2d, return_indices: bool = False
) -> Tensor | tuple[Tensor, Tensor]:
    """Applies a 2D adaptive max pooling over an input signal.

    Args:
        input: Input tensor of shape `(N, C, H, W)` or `(C, H, W)`.
        output_size: Target output spatial size `(H_out, W_out)` or `H_out`.
        return_indices: Whether to return argmax indices alongside pooled outputs.

    Returns:
        Pooled tensor, or `(output, indices)` if `return_indices=True`.
    """
    in_spatial = input.shape[-2:]
    out_spatial = _resolve_opt_size(output_size, 2, in_spatial)
    valid_coords = _build_adaptive_windows(in_spatial, out_spatial)
    return _max_pool_from_coord_windows(
        input, valid_coords, out_spatial, return_indices, "AdaptiveMaxPool2D"
    )


def adaptive_max_pool3d(
    input: Tensor, output_size: OptSize3d, return_indices: bool = False
) -> Tensor | tuple[Tensor, Tensor]:
    """Applies a 3D adaptive max pooling over an input signal.

    Args:
        input: Input tensor of shape `(N, C, D, H, W)` or `(C, D, H, W)`.
        output_size: Target output volumetric size `(D_out, H_out, W_out)` or `D_out`.
        return_indices: Whether to return argmax indices alongside pooled outputs.

    Returns:
        Pooled tensor, or `(output, indices)` if `return_indices=True`.
    """
    in_spatial = input.shape[-3:]
    out_spatial = _resolve_opt_size(output_size, 3, in_spatial)
    valid_coords = _build_adaptive_windows(in_spatial, out_spatial)
    return _max_pool_from_coord_windows(
        input, valid_coords, out_spatial, return_indices, "AdaptiveMaxPool3D"
    )


def fractional_max_pool2d(
    input: Tensor,
    kernel_size: Size2d,
    output_size: Size2d | None = None,
    output_ratio: float | tuple[float, float] | None = None,
    return_indices: bool = False,
    _random_samples: Tensor | None = None,
) -> Tensor | tuple[Tensor, Tensor]:
    """Applies a 2D fractional max pooling over an input signal.

    Args:
        input: Input tensor of shape `(N, C, H, W)` or `(C, H, W)`.
        kernel_size: Size of the pooling window.
        output_size: Target output spatial size `(H_out, W_out)`.
        output_ratio: Target output spatial ratio in `(0, 1)`.
        return_indices: Whether to return argmax indices alongside pooled outputs.
        _random_samples: Optional pre-generated random step samples of shape `(N, C, 2)`.

    Returns:
        Pooled tensor, or `(output, indices)` if `return_indices=True`.
    """
    return _fractional_max_pool_nd(
        input,
        kernel_size,
        output_size,
        output_ratio,
        return_indices,
        _random_samples,
        2,
    )


def fractional_max_pool3d(
    input: Tensor,
    kernel_size: Size3d,
    output_size: Size3d | None = None,
    output_ratio: float | tuple[float, float, float] | None = None,
    return_indices: bool = False,
    _random_samples: Tensor | None = None,
) -> Tensor | tuple[Tensor, Tensor]:
    """Applies a 3D fractional max pooling over an input signal.

    Args:
        input: Input tensor of shape `(N, C, D, H, W)` or `(C, D, H, W)`.
        kernel_size: Size of the pooling window.
        output_size: Target output volumetric size `(D_out, H_out, W_out)`.
        output_ratio: Target output volumetric ratio in `(0, 1)`.
        return_indices: Whether to return argmax indices alongside pooled outputs.
        _random_samples: Optional pre-generated random step samples of shape `(N, C, 3)`.

    Returns:
        Pooled tensor, or `(output, indices)` if `return_indices=True`.
    """
    return _fractional_max_pool_nd(
        input,
        kernel_size,
        output_size,
        output_ratio,
        return_indices,
        _random_samples,
        3,
    )


def avg_pool1d(
    input: Tensor,
    kernel_size: Size1d,
    stride: Size1d | None = None,
    padding: Size1d = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    divisor_override: int | None = None,
) -> Tensor:
    """Applies a 1D average pooling over an input signal.

    Args:
        input: Input tensor of shape `(N, C, L)` or `(C, L)`.
        kernel_size: Size of the pooling window.
        stride: Stride of the pooling window.
        padding: Implicit zero padding added on both sides.
        ceil_mode: When True, uses ceil instead of floor to compute output shape.
        count_include_pad: When True, includes zero-padding in the averaging calculation.
        divisor_override: If specified, used as divisor instead of `kernel_size`.

    Returns:
        Pooled tensor.
    """
    return _avg_pool_nd(
        input,
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override,
        1,
    )


def avg_pool2d(
    input: Tensor,
    kernel_size: Size2d,
    stride: Size2d | None = None,
    padding: Size2d = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    divisor_override: int | None = None,
) -> Tensor:
    """Applies a 2D average pooling over an input signal.

    Args:
        input: Input tensor of shape `(N, C, H, W)` or `(C, H, W)`.
        kernel_size: Size of the pooling window.
        stride: Stride of the pooling window.
        padding: Implicit zero padding added on both sides.
        ceil_mode: When True, uses ceil instead of floor to compute output shape.
        count_include_pad: When True, includes zero-padding in the averaging calculation.
        divisor_override: If specified, used as divisor instead of `kernel_size`.

    Returns:
        Pooled tensor.
    """
    return _avg_pool_nd(
        input,
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override,
        2,
    )


def avg_pool3d(
    input: Tensor,
    kernel_size: Size3d,
    stride: Size3d | None = None,
    padding: Size3d = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    divisor_override: int | None = None,
) -> Tensor:
    """Applies a 3D average pooling over an input signal.

    Args:
        input: Input tensor of shape `(N, C, D, H, W)` or `(C, D, H, W)`.
        kernel_size: Size of the pooling window.
        stride: Stride of the pooling window.
        padding: Implicit zero padding added on both sides.
        ceil_mode: When True, uses ceil instead of floor to compute output shape.
        count_include_pad: When True, includes zero-padding in the averaging calculation.
        divisor_override: If specified, used as divisor instead of `kernel_size`.

    Returns:
        Pooled tensor.
    """
    return _avg_pool_nd(
        input,
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override,
        3,
    )


def max_pool1d(
    input: Tensor,
    kernel_size: Size1d,
    stride: Size1d | None = None,
    padding: Size1d = 0,
    dilation: Size1d = 1,
    ceil_mode: bool = False,
    return_indices: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Applies a 1D max pooling over an input signal.

    Args:
        input: Input tensor of shape `(N, C, L)` or `(C, L)`.
        kernel_size: Size of the pooling window.
        stride: Stride of the pooling window.
        padding: Implicit negative infinity padding added on both sides.
        dilation: Parameter controlling stride of elements in the window.
        ceil_mode: When True, uses ceil instead of floor to compute output shape.
        return_indices: If True, returns argmax indices alongside pooled outputs.

    Returns:
        Pooled tensor, or `(output, indices)` if `return_indices=True`.
    """
    k = _normalize_tuple(kernel_size, 1)
    s = k if stride is None else _normalize_tuple(stride, 1)
    p = _normalize_tuple(padding, 1)
    d = _normalize_tuple(dilation, 1)
    valid_coords, _, out_spatial = _build_sliding_windows(
        input.shape[-1:], k, s, p, d, ceil_mode
    )
    return _max_pool_from_coord_windows(
        input, valid_coords, out_spatial, return_indices, "MaxPool1D"
    )


def max_pool2d(
    input: Tensor,
    kernel_size: Size2d,
    stride: Size2d | None = None,
    padding: Size2d = 0,
    dilation: Size2d = 1,
    ceil_mode: bool = False,
    return_indices: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Applies a 2D max pooling over an input signal.

    Args:
        input: Input tensor of shape `(N, C, H, W)` or `(C, H, W)`.
        kernel_size: Size of the pooling window.
        stride: Stride of the pooling window.
        padding: Implicit negative infinity padding added on both sides.
        dilation: Parameter controlling stride of elements in the window.
        ceil_mode: When True, uses ceil instead of floor to compute output shape.
        return_indices: If True, returns argmax indices alongside pooled outputs.

    Returns:
        Pooled tensor, or `(output, indices)` if `return_indices=True`.
    """
    k = _normalize_tuple(kernel_size, 2)
    s = k if stride is None else _normalize_tuple(stride, 2)
    p = _normalize_tuple(padding, 2)
    d = _normalize_tuple(dilation, 2)
    valid_coords, _, out_spatial = _build_sliding_windows(
        input.shape[-2:], k, s, p, d, ceil_mode
    )
    return _max_pool_from_coord_windows(
        input, valid_coords, out_spatial, return_indices, "MaxPool2D"
    )


def max_pool3d(
    input: Tensor,
    kernel_size: Size3d,
    stride: Size3d | None = None,
    padding: Size3d = 0,
    dilation: Size3d = 1,
    ceil_mode: bool = False,
    return_indices: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Applies a 3D max pooling over an input signal.

    Args:
        input: Input tensor of shape `(N, C, D, H, W)` or `(C, D, H, W)`.
        kernel_size: Size of the pooling window.
        stride: Stride of the pooling window.
        padding: Implicit negative infinity padding added on both sides.
        dilation: Parameter controlling stride of elements in the window.
        ceil_mode: When True, uses ceil instead of floor to compute output shape.
        return_indices: If True, returns argmax indices alongside pooled outputs.

    Returns:
        Pooled tensor, or `(output, indices)` if `return_indices=True`.
    """
    k = _normalize_tuple(kernel_size, 3)
    s = k if stride is None else _normalize_tuple(stride, 3)
    p = _normalize_tuple(padding, 3)
    d = _normalize_tuple(dilation, 3)
    valid_coords, _, out_spatial = _build_sliding_windows(
        input.shape[-3:], k, s, p, d, ceil_mode
    )
    return _max_pool_from_coord_windows(
        input, valid_coords, out_spatial, return_indices, "MaxPool3D"
    )


def max_unpool1d(
    input: Tensor,
    indices: Tensor,
    kernel_size: Size1d,
    stride: Size1d | None = None,
    padding: Size1d = 0,
    output_size: Sequence[int] | None = None,
) -> Tensor:
    """Computes a partial inverse of `MaxPool1d`.

    Args:
        input: Pooled input tensor to invert.
        indices: Indices given out by `MaxPool1d`.
        kernel_size: Size of the max pooling window.
        stride: Stride of the max pooling window.
        padding: Padding that was added to the input.
        output_size: Optional targeted output size.

    Returns:
        Unpooled tensor.
    """
    return _max_unpool_nd(input, indices, kernel_size, stride, padding, output_size, 1)


def max_unpool2d(
    input: Tensor,
    indices: Tensor,
    kernel_size: Size2d,
    stride: Size2d | None = None,
    padding: Size2d = 0,
    output_size: Sequence[int] | None = None,
) -> Tensor:
    """Computes a partial inverse of `MaxPool2d`.

    Args:
        input: Pooled input tensor to invert.
        indices: Indices given out by `MaxPool2d`.
        kernel_size: Size of the max pooling window.
        stride: Stride of the max pooling window.
        padding: Padding that was added to the input.
        output_size: Optional targeted output size.

    Returns:
        Unpooled tensor.
    """
    return _max_unpool_nd(input, indices, kernel_size, stride, padding, output_size, 2)


def max_unpool3d(
    input: Tensor,
    indices: Tensor,
    kernel_size: Size3d,
    stride: Size3d | None = None,
    padding: Size3d = 0,
    output_size: Sequence[int] | None = None,
) -> Tensor:
    """Computes a partial inverse of `MaxPool3d`.

    Args:
        input: Pooled input tensor to invert.
        indices: Indices given out by `MaxPool3d`.
        kernel_size: Size of the max pooling window.
        stride: Stride of the max pooling window.
        padding: Padding that was added to the input.
        output_size: Optional targeted output size.

    Returns:
        Unpooled tensor.
    """
    return _max_unpool_nd(input, indices, kernel_size, stride, padding, output_size, 3)


def lp_pool1d(
    input: Tensor,
    norm_type: float,
    kernel_size: Size1d,
    stride: Size1d | None = None,
    ceil_mode: bool = False,
) -> Tensor:
    """Applies a 1D power-average pooling over an input signal.

    Args:
        input: Input tensor.
        norm_type: Exponent `p` for LP-pooling.
        kernel_size: Size of the pooling window.
        stride: Stride of the pooling window.
        ceil_mode: When True, uses ceil instead of floor to compute output shape.

    Returns:
        Pooled tensor.
    """
    return _lp_pool_nd(input, norm_type, kernel_size, stride, ceil_mode, 1)


def lp_pool2d(
    input: Tensor,
    norm_type: float,
    kernel_size: Size2d,
    stride: Size2d | None = None,
    ceil_mode: bool = False,
) -> Tensor:
    """Applies a 2D power-average pooling over an input signal.

    Args:
        input: Input tensor.
        norm_type: Exponent `p` for LP-pooling.
        kernel_size: Size of the pooling window.
        stride: Stride of the pooling window.
        ceil_mode: When True, uses ceil instead of floor to compute output shape.

    Returns:
        Pooled tensor.
    """
    return _lp_pool_nd(input, norm_type, kernel_size, stride, ceil_mode, 2)


def lp_pool3d(
    input: Tensor,
    norm_type: float,
    kernel_size: Size3d,
    stride: Size3d | None = None,
    ceil_mode: bool = False,
) -> Tensor:
    """Applies a 3D power-average pooling over an input signal.

    Args:
        input: Input tensor.
        norm_type: Exponent `p` for LP-pooling.
        kernel_size: Size of the pooling window.
        stride: Stride of the pooling window.
        ceil_mode: When True, uses ceil instead of floor to compute output shape.

    Returns:
        Pooled tensor.
    """
    return _lp_pool_nd(input, norm_type, kernel_size, stride, ceil_mode, 3)


def unfold(
    input: Tensor,
    kernel_size: Size2d,
    dilation: Size2d = 1,
    padding: Size2d = 0,
    stride: Size2d = 1,
) -> Tensor:
    """Extracts sliding local blocks (`im2col`) from a batched 4D or unbatched 3D input tensor.

    Args:
        input: Input tensor of shape `(N, C, H, W)` or `(C, H, W)`.
        kernel_size: Size of the sliding blocks.
        dilation: Parameter controlling stride of elements within the neighborhood.
        padding: Implicit zero padding added on both sides of input.
        stride: Stride of the sliding blocks in the input spatial dimensions.

    Returns:
        Unfolded tensor of shape `(N, C * k_h * k_w, L)` or `(C * k_h * k_w, L)`.
    """
    kh, kw = _normalize_tuple(kernel_size, 2)
    dh, dw = _normalize_tuple(dilation, 2)
    ph, pw = _normalize_tuple(padding, 2)
    sh, sw = _normalize_tuple(stride, 2)

    h_in, w_in = input.shape[-2], input.shape[-1]
    c_in = input.shape[-3]
    batch_shape = input.shape[:-3]
    h_out = _pool_out_dim(h_in, kh, sh, ph, dh, False)
    w_out = _pool_out_dim(w_in, kw, sw, pw, dw, False)
    num_blocks = h_out * w_out
    out_shape = batch_shape + (c_in * kh * kw, num_blocks)

    if not config.eager_mode:
        return _wrap(
            _emit_reduction_node(
                "Unfold",
                [_to_tensor(input)],
                {
                    "kernel_size": (kh, kw),
                    "dilation": (dh, dw),
                    "padding": (ph, pw),
                    "stride": (sh, sw),
                },
                out_shape,
                input.dtype,
            )
        )

    batch_size = functools.reduce(operator.mul, batch_shape, 1)
    planes = _extract_flat_channels(input, batch_size * c_in, h_in * w_in)

    out_vals: list[float] = []
    for n in range(batch_size):
        for c in range(c_in):
            plane = planes[n * c_in + c]
            for i in range(kh):
                for j in range(kw):
                    for ho in range(h_out):
                        h_coord = ho * sh - ph + i * dh
                        for wo in range(w_out):
                            w_coord = wo * sw - pw + j * dw
                            if 0 <= h_coord < h_in and 0 <= w_coord < w_in:
                                out_vals.append(plane[h_coord * w_in + w_coord])
                            else:
                                out_vals.append(0.0)

    return zero_torch.reshape(Tensor(out_vals, dtype=input.dtype), out_shape)


def fold(
    input: Tensor,
    output_size: Size2d,
    kernel_size: Size2d,
    dilation: Size2d = 1,
    padding: Size2d = 0,
    stride: Size2d = 1,
) -> Tensor:
    """Combines an array of sliding local blocks (`col2im`) into a large containing tensor.

    Args:
        input: Input tensor of shape `(N, C * k_h * k_w, L)` or `(C * k_h * k_w, L)`.
        output_size: Spatial shape `(H, W)` of the output tensor.
        kernel_size: Size of the sliding blocks.
        dilation: Parameter controlling stride of elements within the neighborhood.
        padding: Implicit zero padding added on both sides of input.
        stride: Stride of the sliding blocks in the input spatial dimensions.

    Returns:
        Folded tensor of shape `(N, C, H, W)` or `(C, H, W)`.
    """
    h_out_img, w_out_img = _normalize_tuple(output_size, 2)
    kh, kw = _normalize_tuple(kernel_size, 2)
    dh, dw = _normalize_tuple(dilation, 2)
    ph, pw = _normalize_tuple(padding, 2)
    sh, sw = _normalize_tuple(stride, 2)

    batch_shape = input.shape[:-2]
    c_kh_kw = input.shape[-2]
    num_blocks = input.shape[-1]
    c_out = c_kh_kw // (kh * kw)
    out_shape = batch_shape + (c_out, h_out_img, w_out_img)

    if not config.eager_mode:
        return _wrap(
            _emit_reduction_node(
                "Fold",
                [_to_tensor(input)],
                {
                    "output_size": (h_out_img, w_out_img),
                    "kernel_size": (kh, kw),
                    "dilation": (dh, dw),
                    "padding": (ph, pw),
                    "stride": (sh, sw),
                },
                out_shape,
                input.dtype,
            )
        )

    h_blocks = _pool_out_dim(h_out_img, kh, sh, ph, dh, False)
    w_blocks = _pool_out_dim(w_out_img, kw, sw, pw, dw, False)
    batch_size = functools.reduce(operator.mul, batch_shape, 1)
    rows = _extract_flat_channels(input, batch_size * c_kh_kw, num_blocks)

    img_numel = h_out_img * w_out_img
    out_vals: list[float] = []
    for n in range(batch_size):
        for c in range(c_out):
            plane = [0.0] * img_numel
            for i in range(kh):
                for j in range(kw):
                    row = rows[(n * c_out + c) * (kh * kw) + i * kw + j]
                    for ho in range(h_blocks):
                        h_coord = ho * sh - ph + i * dh
                        if 0 <= h_coord < h_out_img:
                            for wo in range(w_blocks):
                                w_coord = wo * sw - pw + j * dw
                                if 0 <= w_coord < w_out_img:
                                    plane[h_coord * w_out_img + w_coord] += row[
                                        ho * w_blocks + wo
                                    ]
            out_vals.extend(plane)

    return zero_torch.reshape(Tensor(out_vals, dtype=input.dtype), out_shape)
