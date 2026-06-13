"API Frontend backed by ml-switcheroo-compiler."

from typing import Union, Optional
from zero_torch.tensor import Tensor
import ml_switcheroo_compiler.nn as _nn


def _get_nn_op(name):
    import ml_switcheroo_compiler.core.errors

    if not hasattr(_nn, name):
        raise ml_switcheroo_compiler.core.errors.UnimplementedMathError()
    return getattr(_nn, name)


def adaptive_avg_pool1d(input: Tensor, output_size: Union[int, tuple]):
    """Applies a pooling operation (adaptive_avg_pool1d).

    Args:
        input (Tensor): The input tensor.
        output_size (Union[int, tuple]): The output size.

    Returns:
        Tensor: The result of the adaptive_avg_pool1d operation.
    """
    input = Tensor(input) if not isinstance(input, Tensor) else input
    res = _get_nn_op("adaptive_avg_pool1d")(input._tensor, output_size=output_size)
    return Tensor(res) if res is not None else None


def adaptive_avg_pool2d(input: Tensor, output_size: Union[int, tuple]):
    """Applies a pooling operation (adaptive_avg_pool2d).

    Args:
        input (Tensor): The input tensor.
        output_size (Union[int, tuple]): The output size.

    Returns:
        Tensor: The result of the adaptive_avg_pool2d operation.
    """
    input = Tensor(input) if not isinstance(input, Tensor) else input
    res = _get_nn_op("adaptive_avg_pool2d")(input._tensor, output_size=output_size)
    return Tensor(res) if res is not None else None


def adaptive_avg_pool3d(input: Tensor, output_size: Union[int, tuple]):
    """Applies a pooling operation (adaptive_avg_pool3d).

    Args:
        input (Tensor): The input tensor.
        output_size (Union[int, tuple]): The output size.

    Returns:
        Tensor: The result of the adaptive_avg_pool3d operation.
    """
    input = Tensor(input) if not isinstance(input, Tensor) else input
    res = _get_nn_op("adaptive_avg_pool3d")(input._tensor, output_size=output_size)
    return Tensor(res) if res is not None else None


def adaptive_max_pool1d(
    input: Tensor, output_size: Union[int, tuple], return_indices: bool = False
):
    """Applies a pooling operation (adaptive_max_pool1d).

    Args:
        input (Tensor): The input tensor.
        output_size (Union[int, tuple]): The output size.
        return_indices (bool, optional): Whether to return the indices along with the outputs.

    Returns:
        Union[Tensor, tuple[Tensor, Tensor]]: The result of the adaptive_max_pool1d operation.
    """
    input = Tensor(input) if not isinstance(input, Tensor) else input
    res = _get_nn_op("adaptive_max_pool1d")(
        input._tensor, output_size=output_size, return_indices=return_indices
    )
    if return_indices:
        out0 = Tensor(res[0]) if res is not None and res[0] is not None else None
        out1 = Tensor(res[1]) if res is not None and res[1] is not None else None
        return out0, out1
    return Tensor(res) if res is not None else None


def adaptive_max_pool2d(
    input: Tensor, output_size: Union[int, tuple], return_indices: bool = False
):
    """Applies a pooling operation (adaptive_max_pool2d).

    Args:
        input (Tensor): The input tensor.
        output_size (Union[int, tuple]): The output size.
        return_indices (bool, optional): Whether to return the indices along with the outputs.

    Returns:
        Union[Tensor, tuple[Tensor, Tensor]]: The result of the adaptive_max_pool2d operation.
    """
    input = Tensor(input) if not isinstance(input, Tensor) else input
    res = _get_nn_op("adaptive_max_pool2d")(
        input._tensor, output_size=output_size, return_indices=return_indices
    )
    if return_indices:
        out0 = Tensor(res[0]) if res is not None and res[0] is not None else None
        out1 = Tensor(res[1]) if res is not None and res[1] is not None else None
        return out0, out1
    return Tensor(res) if res is not None else None


def adaptive_max_pool3d(
    input: Tensor, output_size: Union[int, tuple], return_indices: bool = False
):
    """Applies a pooling operation (adaptive_max_pool3d).

    Args:
        input (Tensor): The input tensor.
        output_size (Union[int, tuple]): The output size.
        return_indices (bool, optional): Whether to return the indices along with the outputs.

    Returns:
        Union[Tensor, tuple[Tensor, Tensor]]: The result of the adaptive_max_pool3d operation.
    """
    input = Tensor(input) if not isinstance(input, Tensor) else input
    res = _get_nn_op("adaptive_max_pool3d")(
        input._tensor, output_size=output_size, return_indices=return_indices
    )
    if return_indices:
        out0 = Tensor(res[0]) if res is not None and res[0] is not None else None
        out1 = Tensor(res[1]) if res is not None and res[1] is not None else None
        return out0, out1
    return Tensor(res) if res is not None else None


def fractional_max_pool2d(*args, **kwargs):
    """Applies a pooling operation (fractional_max_pool2d).

    Args:
        *args (Any): Variable length argument list.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        Any: The result of the fractional_max_pool2d operation.
    """
    import ml_switcheroo_compiler.core.errors

    raise ml_switcheroo_compiler.core.errors.UnimplementedMathError()


def avg_pool1d(
    input: Tensor,
    kernel_size: Union[int, tuple],
    stride: Optional[Union[int, tuple]] = None,
    padding: Union[int, tuple] = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    divisor_override: Optional[int] = None,
):
    """Applies a pooling operation (avg_pool1d).

    Args:
        input (Tensor): The input tensor.
        kernel_size: The size of the window to take a max over.
        stride: The stride of the window.
        padding: Implicit zero padding to be added on both sides.
        ceil_mode: When True, will use ceil instead of floor to compute the output shape.
        count_include_pad: When True, will include the zero-padding in the averaging calculation.
        divisor_override: If specified, it will be used as divisor.

    Returns:
        Tensor: The result of the avg_pool1d operation.
    """
    input = Tensor(input) if not isinstance(input, Tensor) else input
    # Ignore count_include_pad and divisor_override for ML Switcheroo basic support
    res = _get_nn_op("avg_pool1d")(
        input._tensor,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
    )
    return Tensor(res) if res is not None else None


def fractional_max_pool3d(*args, **kwargs):
    """Applies a pooling operation (fractional_max_pool3d).

    Args:
        *args (Any): Variable length argument list.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        Any: The result of the fractional_max_pool3d operation.
    """
    import ml_switcheroo_compiler.core.errors

    raise ml_switcheroo_compiler.core.errors.UnimplementedMathError()


def avg_pool2d(
    input: Tensor,
    kernel_size: Union[int, tuple],
    stride: Optional[Union[int, tuple]] = None,
    padding: Union[int, tuple] = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    divisor_override: Optional[int] = None,
):
    """Applies a pooling operation (avg_pool2d).

    Args:
        input (Tensor): The input tensor.
        kernel_size: The size of the window to take a max over.
        stride: The stride of the window.
        padding: Implicit zero padding to be added on both sides.
        ceil_mode: When True, will use ceil instead of floor to compute the output shape.
        count_include_pad: When True, will include the zero-padding in the averaging calculation.
        divisor_override: If specified, it will be used as divisor.

    Returns:
        Tensor: The result of the avg_pool2d operation.
    """
    input = Tensor(input) if not isinstance(input, Tensor) else input
    # Ignore count_include_pad and divisor_override for ML Switcheroo basic support
    res = _get_nn_op("avg_pool2d")(
        input._tensor,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
    )
    return Tensor(res) if res is not None else None


def lp_pool1d(*args, **kwargs):
    """Applies a pooling operation (lp_pool1d).

    Args:
        *args (Any): Variable length argument list.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        Any: The result of the lp_pool1d operation.
    """
    import ml_switcheroo_compiler.core.errors

    raise ml_switcheroo_compiler.core.errors.UnimplementedMathError()


def avg_pool3d(
    input: Tensor,
    kernel_size: Union[int, tuple],
    stride: Optional[Union[int, tuple]] = None,
    padding: Union[int, tuple] = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    divisor_override: Optional[int] = None,
):
    """Applies a pooling operation (avg_pool3d).

    Args:
        input (Tensor): The input tensor.
        kernel_size: The size of the window to take a max over.
        stride: The stride of the window.
        padding: Implicit zero padding to be added on both sides.
        ceil_mode: When True, will use ceil instead of floor to compute the output shape.
        count_include_pad: When True, will include the zero-padding in the averaging calculation.
        divisor_override: If specified, it will be used as divisor.

    Returns:
        Tensor: The result of the avg_pool3d operation.
    """
    input = Tensor(input) if not isinstance(input, Tensor) else input
    # Ignore count_include_pad and divisor_override for ML Switcheroo basic support
    res = _get_nn_op("avg_pool3d")(
        input._tensor,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
    )
    return Tensor(res) if res is not None else None


def lp_pool2d(*args, **kwargs):
    """Applies a pooling operation (lp_pool2d).

    Args:
        *args (Any): Variable length argument list.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        Any: The result of the lp_pool2d operation.
    """
    import ml_switcheroo_compiler.core.errors

    raise ml_switcheroo_compiler.core.errors.UnimplementedMathError()


def max_pool1d(*args, **kwargs):
    """Applies a pooling operation (max_pool1d).

    Args:
        *args (Any): Variable length argument list.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        Any: The result of the max_pool1d operation.
    """
    import ml_switcheroo_compiler.core.errors

    raise ml_switcheroo_compiler.core.errors.UnimplementedMathError()


def lp_pool3d(*args, **kwargs):
    """Applies a pooling operation (lp_pool3d).

    Args:
        *args (Any): Variable length argument list.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        Any: The result of the lp_pool3d operation.
    """
    import ml_switcheroo_compiler.core.errors

    raise ml_switcheroo_compiler.core.errors.UnimplementedMathError()


def max_pool2d(*args, **kwargs):
    """Applies a pooling operation (max_pool2d).

    Args:
        *args (Any): Variable length argument list.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        Any: The result of the max_pool2d operation.
    """
    import ml_switcheroo_compiler.core.errors

    raise ml_switcheroo_compiler.core.errors.UnimplementedMathError()


def max_unpool1d(*args, **kwargs):
    """Applies a pooling operation (max_unpool1d).

    Args:
        *args (Any): Variable length argument list.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        Any: The result of the max_unpool1d operation.
    """
    import ml_switcheroo_compiler.core.errors

    raise ml_switcheroo_compiler.core.errors.UnimplementedMathError()


def max_pool3d(*args, **kwargs):
    """Applies a pooling operation (max_pool3d).

    Args:
        *args (Any): Variable length argument list.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        Any: The result of the max_pool3d operation.
    """
    import ml_switcheroo_compiler.core.errors

    raise ml_switcheroo_compiler.core.errors.UnimplementedMathError()


def max_unpool2d(*args, **kwargs):
    """Applies a pooling operation (max_unpool2d).

    Args:
        *args (Any): Variable length argument list.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        Any: The result of the max_unpool2d operation.
    """
    import ml_switcheroo_compiler.core.errors

    raise ml_switcheroo_compiler.core.errors.UnimplementedMathError()


def max_unpool3d(*args, **kwargs):
    """Applies a pooling operation (max_unpool3d).

    Args:
        *args (Any): Variable length argument list.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        Any: The result of the max_unpool3d operation.
    """
    import ml_switcheroo_compiler.core.errors

    raise ml_switcheroo_compiler.core.errors.UnimplementedMathError()
