"""Initialization functions."""

import ml_switcheroo_compiler as ml_switcheroo
from typing import Any, Optional, Tuple
import math
from zero_torch.tensor import Tensor

__all__ = [
    "calculate_gain",
    "constant",
    "constant_",
    "dirac",
    "dirac_",
    "eye",
    "eye_",
    "kaiming_normal",
    "kaiming_normal_",
    "kaiming_uniform",
    "kaiming_uniform_",
    "normal",
    "normal_",
    "ones_",
    "orthogonal",
    "orthogonal_",
    "sparse",
    "sparse_",
    "trunc_normal_",
    "uniform",
    "uniform_",
    "xavier_normal",
    "xavier_normal_",
    "xavier_uniform",
    "xavier_uniform_",
    "zeros_",
]


def calculate_gain(nonlinearity: str, param: Optional[float] = None) -> float:
    """Return the recommended gain value for the given nonlinearity function.

    Args:
        nonlinearity (str): the non-linear function (`nn.functional` name).
        param (Optional[float], optional): optional parameter for the non-linear function. Defaults to None.

    Returns:
        float: the recommended gain.
    """
    pass


def constant_(tensor: Tensor, val: float) -> Tensor:
    """Fills the input Tensor with the value `val`.

    Args:
        tensor (Tensor): an n-dimensional torch.Tensor.
        val (float): the value to fill the tensor with.

    Returns:
        Tensor: the filled tensor.
    """
    pass


def constant(*args, **kwargs) -> Tensor:
    """Fills the input Tensor with a constant value.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: the filled tensor.
    """
    pass


def dirac_(tensor: Tensor, groups: int = 1) -> Tensor:
    """Fills the {3, 4, 5}-dimensional input Tensor with the Dirac delta function.

    Args:
        tensor (Tensor): a {3, 4, 5}-dimensional torch.Tensor.
        groups (int, optional): number of groups in the conv layer. Defaults to 1.

    Returns:
        Tensor: the filled tensor.
    """
    pass


def dirac(*args, **kwargs) -> Tensor:
    """Fills the input Tensor with the Dirac delta function.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: the filled tensor.
    """
    pass


def eye_(tensor: Tensor) -> Tensor:
    """Fills the 2-dimensional input Tensor with the identity matrix.

    Args:
        tensor (Tensor): a 2-dimensional torch.Tensor.

    Returns:
        Tensor: the filled tensor.
    """
    pass


def eye(*args, **kwargs) -> Tensor:
    """Fills the input Tensor with the identity matrix.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: the filled tensor.
    """
    pass


def normal_(
    tensor: Tensor, mean: float = 0.0, std: float = 1.0, generator: Optional[Any] = None
) -> Tensor:
    """Fills the input Tensor with values drawn from the normal distribution.

    Args:
        tensor (Tensor): an n-dimensional torch.Tensor.
        mean (float, optional): the mean of the normal distribution. Defaults to 0.0.
        std (float, optional): the standard deviation of the normal distribution. Defaults to 1.0.
        generator (Optional[Any], optional): pseudo-random number generator. Defaults to None.

    Returns:
        Tensor: the filled tensor.
    """
    pass


def normal(*args, **kwargs) -> Tensor:
    """Fills the input Tensor with values drawn from the normal distribution.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: the filled tensor.
    """
    pass


def ones_(tensor: Tensor) -> Tensor:
    """Fills the input Tensor with the scalar value 1.

    Args:
        tensor (Tensor): an n-dimensional torch.Tensor.

    Returns:
        Tensor: the filled tensor.
    """
    pass


def zeros_(tensor: Tensor) -> Tensor:
    """Fills the input Tensor with the scalar value 0.

    Args:
        tensor (Tensor): an n-dimensional torch.Tensor.

    Returns:
        Tensor: the filled tensor.
    """
    pass


def uniform_(
    tensor: Tensor, a: float = 0.0, b: float = 1.0, generator: Optional[Any] = None
) -> Tensor:
    """Fills the input Tensor with values drawn from the uniform distribution.

    Args:
        tensor (Tensor): an n-dimensional torch.Tensor.
        a (float, optional): the lower bound of the uniform distribution. Defaults to 0.0.
        b (float, optional): the upper bound of the uniform distribution. Defaults to 1.0.
        generator (Optional[Any], optional): pseudo-random number generator. Defaults to None.

    Returns:
        Tensor: the filled tensor.
    """
    pass


def uniform(*args, **kwargs) -> Tensor:
    """Fills the input Tensor with values drawn from the uniform distribution.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: the filled tensor.
    """
    pass


def trunc_normal_(
    tensor: Tensor,
    mean: float = 0.0,
    std: float = 1.0,
    a: float = -2.0,
    b: float = 2.0,
    generator: Optional[Any] = None,
) -> Tensor:
    """Fills the input Tensor with values drawn from a truncated normal distribution.

    Args:
        tensor (Tensor): an n-dimensional torch.Tensor.
        mean (float, optional): the mean of the normal distribution. Defaults to 0.0.
        std (float, optional): the standard deviation of the normal distribution. Defaults to 1.0.
        a (float, optional): the minimum cutoff value. Defaults to -2.0.
        b (float, optional): the maximum cutoff value. Defaults to 2.0.
        generator (Optional[Any], optional): pseudo-random number generator. Defaults to None.

    Returns:
        Tensor: the filled tensor.
    """
    pass


def _calculate_fan_in_and_fan_out(tensor: Tensor) -> Tuple[int, int]:
    """Calculates the fan_in and fan_out of a tensor.

    Args:
        tensor (Tensor): the input tensor.

    Returns:
        Tuple[int, int]: fan_in and fan_out values.
    """
    pass


def xavier_uniform_(
    tensor: Tensor, gain: float = 1.0, generator: Optional[Any] = None
) -> Tensor:
    """Fills the input Tensor with values according to the method described in Understanding the difficulty of training deep feedforward neural networks - Glorot, X. & Bengio, Y. (2010), using a uniform distribution.

    Args:
        tensor (Tensor): an n-dimensional torch.Tensor.
        gain (float, optional): an optional scaling factor. Defaults to 1.0.
        generator (Optional[Any], optional): pseudo-random number generator. Defaults to None.

    Returns:
        Tensor: the filled tensor.
    """
    pass


def xavier_uniform(*args, **kwargs) -> Tensor:
    """Fills the input Tensor with values according to Xavier uniform initialization.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: the filled tensor.
    """
    pass


def xavier_normal_(
    tensor: Tensor, gain: float = 1.0, generator: Optional[Any] = None
) -> Tensor:
    """Fills the input Tensor with values according to the method described in Understanding the difficulty of training deep feedforward neural networks - Glorot, X. & Bengio, Y. (2010), using a normal distribution.

    Args:
        tensor (Tensor): an n-dimensional torch.Tensor.
        gain (float, optional): an optional scaling factor. Defaults to 1.0.
        generator (Optional[Any], optional): pseudo-random number generator. Defaults to None.

    Returns:
        Tensor: the filled tensor.
    """
    pass


def xavier_normal(*args, **kwargs) -> Tensor:
    """Fills the input Tensor with values according to Xavier normal initialization.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: the filled tensor.
    """
    pass


def kaiming_uniform_(
    tensor: Tensor,
    a: float = 0,
    mode: str = "fan_in",
    nonlinearity: str = "leaky_relu",
    generator: Optional[Any] = None,
) -> Tensor:
    """Fills the input Tensor with values according to the method described in Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification - He, K. et al. (2015), using a uniform distribution.

    Args:
        tensor (Tensor): an n-dimensional torch.Tensor.
        a (float, optional): the negative slope of the rectifier used after this layer. Defaults to 0.
        mode (str, optional): either 'fan_in' or 'fan_out'. Defaults to "fan_in".
        nonlinearity (str, optional): the non-linear function. Defaults to "leaky_relu".
        generator (Optional[Any], optional): pseudo-random number generator. Defaults to None.

    Returns:
        Tensor: the filled tensor.
    """
    pass


def kaiming_uniform(*args, **kwargs) -> Tensor:
    """Fills the input Tensor with values according to Kaiming uniform initialization.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: the filled tensor.
    """
    pass


def kaiming_normal_(
    tensor: Tensor,
    a: float = 0,
    mode: str = "fan_in",
    nonlinearity: str = "leaky_relu",
    generator: Optional[Any] = None,
) -> Tensor:
    """Fills the input Tensor with values according to the method described in Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification - He, K. et al. (2015), using a normal distribution.

    Args:
        tensor (Tensor): an n-dimensional torch.Tensor.
        a (float, optional): the negative slope of the rectifier used after this layer. Defaults to 0.
        mode (str, optional): either 'fan_in' or 'fan_out'. Defaults to "fan_in".
        nonlinearity (str, optional): the non-linear function. Defaults to "leaky_relu".
        generator (Optional[Any], optional): pseudo-random number generator. Defaults to None.

    Returns:
        Tensor: the filled tensor.
    """
    pass


def kaiming_normal(*args, **kwargs) -> Tensor:
    """Fills the input Tensor with values according to Kaiming normal initialization.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: the filled tensor.
    """
    pass


def orthogonal_(
    tensor: Tensor, gain: float = 1, generator: Optional[Any] = None
) -> Tensor:
    """Fills the input Tensor with a (semi) orthogonal matrix, as described in Exact solutions to the nonlinear dynamics of learning in deep linear neural networks - Saxe, A. et al. (2013).

    Args:
        tensor (Tensor): an n-dimensional torch.Tensor.
        gain (float, optional): an optional scaling factor. Defaults to 1.
        generator (Optional[Any], optional): pseudo-random number generator. Defaults to None.

    Returns:
        Tensor: the filled tensor.
    """
    pass


def orthogonal(*args, **kwargs) -> Tensor:
    """Fills the input Tensor with a (semi) orthogonal matrix.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: the filled tensor.
    """
    pass


def sparse_(
    tensor: Tensor, sparsity: float, std: float = 0.01, generator: Optional[Any] = None
) -> Tensor:
    """Fills the 2D input Tensor as a sparse matrix, where the non-zero elements will be drawn from the normal distribution N(0, std).

    Args:
        tensor (Tensor): an n-dimensional torch.Tensor.
        sparsity (float): the fraction of elements in each column to be set to zero.
        std (float, optional): the standard deviation of the normal distribution. Defaults to 0.01.
        generator (Optional[Any], optional): pseudo-random number generator. Defaults to None.

    Returns:
        Tensor: the filled tensor.
    """
    pass


def sparse(*args, **kwargs) -> Tensor:
    """Fills the 2D input Tensor as a sparse matrix.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: the filled tensor.
    """
    pass
