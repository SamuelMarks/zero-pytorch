"""Functional pooling operations backed by ml-switcheroo-compiler."""

from __future__ import annotations

from ml_switcheroo_compiler import ops
from ml_switcheroo_compiler.core.errors import UnimplementedMathError
from ml_switcheroo_compiler.ops import abs as _op_abs
from ml_switcheroo_compiler.ops import power as _op_power
from ml_switcheroo_compiler.ops.nn.pooling import avg_pool, max_pool

from zero_torch._types import (
    OpArg,
    OptSize1d,
    OptSize2d,
    OptSize3d,
    Size1d,
    Size2d,
    Size3d,
)
from zero_torch.tensor import Tensor, _to_tensor, _wrap


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


def adaptive_avg_pool1d(input: Tensor, output_size: OptSize1d) -> Tensor:
    """Applies a 1D adaptive average pooling over an input signal.

    Args:
        input: Input tensor.
        output_size: Target output length `L_out`.

    Returns:
        Pooled tensor.
    """
    _ = (input, output_size)
    raise UnimplementedMathError()


def adaptive_avg_pool2d(input: Tensor, output_size: OptSize2d) -> Tensor:
    """Applies a 2D adaptive average pooling over an input signal.

    Args:
        input: Input tensor.
        output_size: Target output spatial size `(H_out, W_out)` or `H_out`.

    Returns:
        Pooled tensor.
    """
    resolved = (
        (output_size, output_size) if isinstance(output_size, int) else output_size
    )
    res = ops.adaptive_avg_pool2d(_to_tensor(input), output_size=resolved)
    return _wrap(res)


def adaptive_avg_pool3d(input: Tensor, output_size: OptSize3d) -> Tensor:
    """Applies a 3D adaptive average pooling over an input signal.

    Args:
        input: Input tensor.
        output_size: Target output volumetric size `(D_out, H_out, W_out)`.

    Returns:
        Pooled tensor.
    """
    _ = (input, output_size)
    raise UnimplementedMathError()


def adaptive_max_pool1d(
    input: Tensor, output_size: OptSize1d, return_indices: bool = False
) -> Tensor | tuple[Tensor, Tensor]:
    """Applies a 1D adaptive max pooling over an input signal.

    Args:
        input: Input tensor.
        output_size: Target output length `L_out`.
        return_indices: Whether to return argmax indices alongside pooled outputs.

    Returns:
        Pooled tensor, or `(output, indices)` if `return_indices=True`.
    """
    _ = (input, output_size, return_indices)
    raise UnimplementedMathError()


def adaptive_max_pool2d(
    input: Tensor, output_size: OptSize2d, return_indices: bool = False
) -> Tensor | tuple[Tensor, Tensor]:
    """Applies a 2D adaptive max pooling over an input signal.

    Args:
        input: Input tensor.
        output_size: Target output spatial size `(H_out, W_out)`.
        return_indices: Whether to return argmax indices alongside pooled outputs.

    Returns:
        Pooled tensor, or `(output, indices)` if `return_indices=True`.
    """
    resolved = (
        (output_size, output_size) if isinstance(output_size, int) else output_size
    )
    res = ops.adaptive_max_pool2d(_to_tensor(input), output_size=resolved)
    if return_indices:
        raise UnimplementedMathError()
    return _wrap(res)


def adaptive_max_pool3d(
    input: Tensor, output_size: OptSize3d, return_indices: bool = False
) -> Tensor | tuple[Tensor, Tensor]:
    """Applies a 3D adaptive max pooling over an input signal.

    Args:
        input: Input tensor.
        output_size: Target output volumetric size `(D_out, H_out, W_out)`.
        return_indices: Whether to return argmax indices alongside pooled outputs.

    Returns:
        Pooled tensor, or `(output, indices)` if `return_indices=True`.
    """
    _ = (input, output_size, return_indices)
    raise UnimplementedMathError()


def fractional_max_pool2d(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies a 2D fractional max pooling over an input signal.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Pooled tensor.
    """
    _ = (args, kwargs)
    raise UnimplementedMathError()


def fractional_max_pool3d(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies a 3D fractional max pooling over an input signal.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Pooled tensor.
    """
    _ = (args, kwargs)
    raise UnimplementedMathError()


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
        input: Input tensor.
        kernel_size: Size of the pooling window.
        stride: Stride of the pooling window.
        padding: Implicit zero padding added on both sides.
        ceil_mode: When True, uses ceil instead of floor to compute output shape.
        count_include_pad: When True, includes zero-padding in the averaging calculation.
        divisor_override: If specified, used as divisor instead of `kernel_size`.

    Returns:
        Pooled tensor.
    """
    _ = (ceil_mode, count_include_pad, divisor_override)
    k = (kernel_size,) if isinstance(kernel_size, int) else kernel_size
    s = k if stride is None else ((stride,) if isinstance(stride, int) else stride)
    pads = _format_padding(padding, 1)
    res = avg_pool(_to_tensor(input), window_shape=k, strides=s, padding=pads)
    return _wrap(res)


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
        input: Input tensor.
        kernel_size: Size of the pooling window.
        stride: Stride of the pooling window.
        padding: Implicit zero padding added on both sides.
        ceil_mode: When True, uses ceil instead of floor to compute output shape.
        count_include_pad: When True, includes zero-padding in the averaging calculation.
        divisor_override: If specified, used as divisor instead of `kernel_size`.

    Returns:
        Pooled tensor.
    """
    _ = (ceil_mode, count_include_pad, divisor_override)
    k = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    s = (
        k
        if stride is None
        else ((stride, stride) if isinstance(stride, int) else stride)
    )
    pads = _format_padding(padding, 2)
    res = avg_pool(_to_tensor(input), window_shape=k, strides=s, padding=pads)
    return _wrap(res)


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
        input: Input tensor.
        kernel_size: Size of the pooling window.
        stride: Stride of the pooling window.
        padding: Implicit zero padding added on both sides.
        ceil_mode: When True, uses ceil instead of floor to compute output shape.
        count_include_pad: When True, includes zero-padding in the averaging calculation.
        divisor_override: If specified, used as divisor instead of `kernel_size`.

    Returns:
        Pooled tensor.
    """
    _ = (ceil_mode, count_include_pad, divisor_override)
    k = (
        (kernel_size, kernel_size, kernel_size)
        if isinstance(kernel_size, int)
        else kernel_size
    )
    s = (
        k
        if stride is None
        else ((stride, stride, stride) if isinstance(stride, int) else stride)
    )
    pads = _format_padding(padding, 3)
    res = avg_pool(_to_tensor(input), window_shape=k, strides=s, padding=pads)
    return _wrap(res)


def max_pool1d(
    input: Tensor,
    kernel_size: Size1d,
    stride: Size1d | None = None,
    padding: Size1d = 0,
    dilation: Size1d = 1,
    ceil_mode: bool = False,
    return_indices: bool = False,
) -> Tensor:
    """Applies a 1D max pooling over an input signal.

    Args:
        input: Input tensor.
        kernel_size: Size of the pooling window.
        stride: Stride of the pooling window.
        padding: Implicit negative infinity padding added on both sides.
        dilation: Parameter controlling stride of elements in the window.
        ceil_mode: When True, uses ceil instead of floor to compute output shape.
        return_indices: If True, returns argmax indices alongside pooled outputs.

    Returns:
        Pooled tensor.
    """
    _ = (dilation, ceil_mode, return_indices)
    k = (kernel_size,) if isinstance(kernel_size, int) else kernel_size
    s = k if stride is None else ((stride,) if isinstance(stride, int) else stride)
    pads = _format_padding(padding, 1)
    res = max_pool(_to_tensor(input), window_shape=k, strides=s, padding=pads)
    return _wrap(res)


def max_pool2d(
    input: Tensor,
    kernel_size: Size2d,
    stride: Size2d | None = None,
    padding: Size2d = 0,
    dilation: Size2d = 1,
    ceil_mode: bool = False,
    return_indices: bool = False,
) -> Tensor:
    """Applies a 2D max pooling over an input signal.

    Args:
        input: Input tensor.
        kernel_size: Size of the pooling window.
        stride: Stride of the pooling window.
        padding: Implicit negative infinity padding added on both sides.
        dilation: Parameter controlling stride of elements in the window.
        ceil_mode: When True, uses ceil instead of floor to compute output shape.
        return_indices: If True, returns argmax indices alongside pooled outputs.

    Returns:
        Pooled tensor.
    """
    _ = (dilation, ceil_mode, return_indices)
    k = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    s = (
        k
        if stride is None
        else ((stride, stride) if isinstance(stride, int) else stride)
    )
    pads = _format_padding(padding, 2)
    res = max_pool(_to_tensor(input), window_shape=k, strides=s, padding=pads)
    return _wrap(res)


def max_pool3d(
    input: Tensor,
    kernel_size: Size3d,
    stride: Size3d | None = None,
    padding: Size3d = 0,
    dilation: Size3d = 1,
    ceil_mode: bool = False,
    return_indices: bool = False,
) -> Tensor:
    """Applies a 3D max pooling over an input signal.

    Args:
        input: Input tensor.
        kernel_size: Size of the pooling window.
        stride: Stride of the pooling window.
        padding: Implicit negative infinity padding added on both sides.
        dilation: Parameter controlling stride of elements in the window.
        ceil_mode: When True, uses ceil instead of floor to compute output shape.
        return_indices: If True, returns argmax indices alongside pooled outputs.

    Returns:
        Pooled tensor.
    """
    _ = (dilation, ceil_mode, return_indices)
    k = (
        (kernel_size, kernel_size, kernel_size)
        if isinstance(kernel_size, int)
        else kernel_size
    )
    s = (
        k
        if stride is None
        else ((stride, stride, stride) if isinstance(stride, int) else stride)
    )
    pads = _format_padding(padding, 3)
    res = max_pool(_to_tensor(input), window_shape=k, strides=s, padding=pads)
    return _wrap(res)


def max_unpool1d(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Computes a partial inverse of `MaxPool1d`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Unpooled tensor.
    """
    _ = (args, kwargs)
    raise UnimplementedMathError()


def max_unpool2d(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Computes a partial inverse of `MaxPool2d`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Unpooled tensor.
    """
    _ = (args, kwargs)
    raise UnimplementedMathError()


def max_unpool3d(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Computes a partial inverse of `MaxPool3d`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Unpooled tensor.
    """
    _ = (args, kwargs)
    raise UnimplementedMathError()


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
    _ = ceil_mode
    k = (kernel_size,) if isinstance(kernel_size, int) else kernel_size
    s = k if stride is None else ((stride,) if isinstance(stride, int) else stride)
    pads = _format_padding(0, 1)
    input_t = _to_tensor(input)
    if norm_type == 1.0:
        res = avg_pool(_op_abs(input_t), window_shape=k, strides=s, padding=pads) * k[0]
    else:
        out = (
            avg_pool(
                _op_power(_op_abs(input_t), norm_type),
                window_shape=k,
                strides=s,
                padding=pads,
            )
            * k[0]
        )
        res = _op_power(out, 1.0 / norm_type)
    return _wrap(res)


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
    _ = ceil_mode
    k = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    s = (
        k
        if stride is None
        else ((stride, stride) if isinstance(stride, int) else stride)
    )
    pads = _format_padding(0, 2)
    input_t = _to_tensor(input)
    factor = k[0] * k[1]
    if norm_type == 1.0:
        res = (
            avg_pool(_op_abs(input_t), window_shape=k, strides=s, padding=pads) * factor
        )
    else:
        out = (
            avg_pool(
                _op_power(_op_abs(input_t), norm_type),
                window_shape=k,
                strides=s,
                padding=pads,
            )
            * factor
        )
        res = _op_power(out, 1.0 / norm_type)
    return _wrap(res)


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
    _ = ceil_mode
    k = (
        (kernel_size, kernel_size, kernel_size)
        if isinstance(kernel_size, int)
        else kernel_size
    )
    s = (
        k
        if stride is None
        else ((stride, stride, stride) if isinstance(stride, int) else stride)
    )
    pads = _format_padding(0, 3)
    input_t = _to_tensor(input)
    factor = k[0] * k[1] * k[2]
    if norm_type == 1.0:
        res = (
            avg_pool(_op_abs(input_t), window_shape=k, strides=s, padding=pads) * factor
        )
    else:
        out = (
            avg_pool(
                _op_power(_op_abs(input_t), norm_type),
                window_shape=k,
                strides=s,
                padding=pads,
            )
            * factor
        )
        res = _op_power(out, 1.0 / norm_type)
    return _wrap(res)
