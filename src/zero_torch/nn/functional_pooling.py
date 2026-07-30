"""API Frontend backed by ml-switcheroo-compiler."""

from __future__ import annotations

from ml_switcheroo_compiler import ops
from ml_switcheroo_compiler.ops.nn.pooling import avg_pool, max_pool

from zero_torch.tensor import Tensor, _to_tensor, _wrap


def _format_padding(padding, ndims):
    if isinstance(padding, int):
        return tuple((padding, padding) for _ in range(ndims))
    if isinstance(padding, tuple) and len(padding) == ndims:  # pragma: no cover
        return tuple((p, p) for p in padding)  # pragma: no cover
    return padding  # pragma: no cover


def adaptive_avg_pool1d(input: Tensor, output_size: int | tuple):
    import ml_switcheroo_compiler.core.errors

    raise ml_switcheroo_compiler.core.errors.UnimplementedMathError()


def adaptive_avg_pool2d(input: Tensor, output_size: int | tuple):
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    res = ops.adaptive_avg_pool2d(_to_tensor(input), output_size=output_size)
    return _wrap(res)


def adaptive_avg_pool3d(input: Tensor, output_size: int | tuple):
    import ml_switcheroo_compiler.core.errors

    raise ml_switcheroo_compiler.core.errors.UnimplementedMathError()


def adaptive_max_pool1d(
    input: Tensor, output_size: int | tuple, return_indices: bool = False
):
    import ml_switcheroo_compiler.core.errors

    raise ml_switcheroo_compiler.core.errors.UnimplementedMathError()


def adaptive_max_pool2d(
    input: Tensor, output_size: int | tuple, return_indices: bool = False
):
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    res = ops.adaptive_max_pool2d(_to_tensor(input), output_size=output_size)
    if return_indices:
        import ml_switcheroo_compiler.core.errors

        raise ml_switcheroo_compiler.core.errors.UnimplementedMathError()
    return _wrap(res)


def adaptive_max_pool3d(
    input: Tensor, output_size: int | tuple, return_indices: bool = False
):
    import ml_switcheroo_compiler.core.errors

    raise ml_switcheroo_compiler.core.errors.UnimplementedMathError()


def fractional_max_pool2d(*args, **kwargs):
    import ml_switcheroo_compiler.core.errors

    raise ml_switcheroo_compiler.core.errors.UnimplementedMathError()


def avg_pool1d(
    input: Tensor,
    kernel_size: int | tuple,
    stride: int | tuple | None = None,
    padding: int | tuple = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    divisor_override: int | None = None,
):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size,)
    if stride is None:
        stride = kernel_size
    elif isinstance(stride, int):
        stride = (stride,)
    pads = _format_padding(padding, 1)
    res = avg_pool(
        _to_tensor(input), window_shape=kernel_size, strides=stride, padding=pads
    )
    return _wrap(res)


def fractional_max_pool3d(*args, **kwargs):
    import ml_switcheroo_compiler.core.errors

    raise ml_switcheroo_compiler.core.errors.UnimplementedMathError()


def avg_pool2d(
    input: Tensor,
    kernel_size: int | tuple,
    stride: int | tuple | None = None,
    padding: int | tuple = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    divisor_override: int | None = None,
):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if stride is None:
        stride = kernel_size
    elif isinstance(stride, int):
        stride = (stride, stride)  # pragma: no cover
    pads = _format_padding(padding, 2)
    res = avg_pool(
        _to_tensor(input), window_shape=kernel_size, strides=stride, padding=pads
    )
    return _wrap(res)


def avg_pool3d(
    input: Tensor,
    kernel_size: int | tuple,
    stride: int | tuple | None = None,
    padding: int | tuple = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    divisor_override: int | None = None,
):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size, kernel_size)
    if stride is None:
        stride = kernel_size
    elif isinstance(stride, int):
        stride = (stride, stride, stride)  # pragma: no cover
    pads = _format_padding(padding, 3)
    res = avg_pool(
        _to_tensor(input), window_shape=kernel_size, strides=stride, padding=pads
    )
    return _wrap(res)


def max_pool1d(
    input: Tensor,
    kernel_size: int | tuple,
    stride: int | tuple | None = None,
    padding: int | tuple = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    divisor_override: int | None = None,
):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size,)  # pragma: no cover
    if stride is None:
        stride = kernel_size
    elif isinstance(stride, int):  # pragma: no cover
        stride = (stride,)  # pragma: no cover
    pads = _format_padding(padding, 1)
    res = max_pool(
        _to_tensor(input), window_shape=kernel_size, strides=stride, padding=pads
    )
    return _wrap(res)  # pragma: no cover


def max_pool2d(
    input: Tensor,
    kernel_size: int | tuple,
    stride: int | tuple | None = None,
    padding: int | tuple = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    divisor_override: int | None = None,
):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)  # pragma: no cover
    if stride is None:
        stride = kernel_size
    elif isinstance(stride, int):  # pragma: no cover
        stride = (stride, stride)  # pragma: no cover
    pads = _format_padding(padding, 2)
    res = max_pool(
        _to_tensor(input), window_shape=kernel_size, strides=stride, padding=pads
    )
    return _wrap(res)  # pragma: no cover


def max_unpool1d(*args, **kwargs):
    import ml_switcheroo_compiler.core.errors

    raise ml_switcheroo_compiler.core.errors.UnimplementedMathError()


def max_pool3d(
    input: Tensor,
    kernel_size: int | tuple,
    stride: int | tuple | None = None,
    padding: int | tuple = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    divisor_override: int | None = None,
):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size, kernel_size)  # pragma: no cover
    if stride is None:
        stride = kernel_size
    elif isinstance(stride, int):  # pragma: no cover
        stride = (stride, stride, stride)  # pragma: no cover
    pads = _format_padding(padding, 3)
    res = max_pool(
        _to_tensor(input), window_shape=kernel_size, strides=stride, padding=pads
    )
    return _wrap(res)  # pragma: no cover


def max_unpool2d(*args, **kwargs):
    import ml_switcheroo_compiler.core.errors

    raise ml_switcheroo_compiler.core.errors.UnimplementedMathError()


def max_unpool3d(*args, **kwargs):
    import ml_switcheroo_compiler.core.errors

    raise ml_switcheroo_compiler.core.errors.UnimplementedMathError()


def lp_pool1d(
    input: Tensor,
    norm_type: float,
    kernel_size: int | tuple,
    stride: int | tuple | None = None,
    ceil_mode: bool = False,
):
    from ml_switcheroo_compiler.ops import abs, power

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size,)
    if stride is None:
        stride = kernel_size
    elif isinstance(stride, int):  # pragma: no cover
        stride = (stride,)  # pragma: no cover

    input_t = _to_tensor(input)
    if norm_type == 1:
        res = (  # pragma: no cover
            avg_pool(abs(input_t), window_shape=kernel_size, strides=stride, padding=0)
            * kernel_size[0]
        )
    else:
        out = (
            avg_pool(
                power(abs(input_t), norm_type),
                window_shape=kernel_size,
                strides=stride,
                padding=0,
            )
            * kernel_size[0]
        )
        res = power(out, 1.0 / norm_type)  # pragma: no cover
    return _wrap(res)  # pragma: no cover


def lp_pool2d(
    input: Tensor,
    norm_type: float,
    kernel_size: int | tuple,
    stride: int | tuple | None = None,
    ceil_mode: bool = False,
):
    from ml_switcheroo_compiler.ops import abs, power

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if stride is None:
        stride = kernel_size
    elif isinstance(stride, int):  # pragma: no cover
        stride = (stride, stride)  # pragma: no cover

    input_t = _to_tensor(input)
    factor = kernel_size[0] * kernel_size[1]
    if norm_type == 1:
        res = (  # pragma: no cover
            avg_pool(abs(input_t), window_shape=kernel_size, strides=stride, padding=0)
            * factor
        )
    else:
        out = (
            avg_pool(
                power(abs(input_t), norm_type),
                window_shape=kernel_size,
                strides=stride,
                padding=0,
            )
            * factor
        )
        res = power(out, 1.0 / norm_type)  # pragma: no cover
    return _wrap(res)  # pragma: no cover


def lp_pool3d(
    input: Tensor,
    norm_type: float,
    kernel_size: int | tuple,
    stride: int | tuple | None = None,
    ceil_mode: bool = False,
):
    from ml_switcheroo_compiler.ops import abs, power

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size, kernel_size)
    if stride is None:
        stride = kernel_size
    elif isinstance(stride, int):  # pragma: no cover
        stride = (stride, stride, stride)  # pragma: no cover

    input_t = _to_tensor(input)
    factor = kernel_size[0] * kernel_size[1] * kernel_size[2]
    if norm_type == 1:
        res = (  # pragma: no cover
            avg_pool(abs(input_t), window_shape=kernel_size, strides=stride, padding=0)
            * factor
        )
    else:
        out = (
            avg_pool(
                power(abs(input_t), norm_type),
                window_shape=kernel_size,
                strides=stride,
                padding=0,
            )
            * factor
        )
        res = power(out, 1.0 / norm_type)  # pragma: no cover
    return _wrap(res)  # pragma: no cover
