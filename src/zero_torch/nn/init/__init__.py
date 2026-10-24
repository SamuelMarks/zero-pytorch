"""Parameter initialization functions for zero_torch.nn.init."""

from __future__ import annotations

import math
import random
from collections.abc import Callable

import zero_torch
from zero_torch._types import OpArg
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

_LINEAR_NONLINEARITIES: frozenset[str] = frozenset(
    {
        "linear",
        "conv1d",
        "conv2d",
        "conv3d",
        "conv_transpose1d",
        "conv_transpose2d",
        "conv_transpose3d",
        "sigmoid",
    }
)


def calculate_gain(nonlinearity: str, param: float | None = None) -> float:
    """Returns the recommended gain value for the given nonlinearity function.

    Args:
        nonlinearity: The non-linear function (`nn.functional` name).
        param: Optional parameter for the non-linear function (e.g. negative_slope for leaky_relu).

    Returns:
        Recommended gain scalar.
    """
    if nonlinearity in _LINEAR_NONLINEARITIES:
        return 1.0
    if nonlinearity == "tanh":
        return 5.0 / 3.0
    if nonlinearity == "relu":
        return math.sqrt(2.0)
    if nonlinearity == "leaky_relu":
        negative_slope = 0.01 if param is None else float(param)
        return math.sqrt(2.0 / (1.0 + negative_slope**2))
    if nonlinearity == "selu":
        return 3.0 / 4.0
    raise ValueError(f"Unsupported nonlinearity {nonlinearity}")


def constant_(tensor: Tensor, val: float) -> Tensor:
    """Fills the input Tensor with the value `val` in-place.

    Args:
        tensor: An n-dimensional `Tensor`.
        val: The value to fill the tensor with.

    Returns:
        The modified `tensor`.
    """
    new_t = zero_torch.full_like(tensor, val)
    tensor._tensor = new_t._tensor
    return tensor


def constant(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Deprecated alias for `constant_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Initialized tensor.
    """
    if (
        len(args) >= 2
        and isinstance(args[0], Tensor)
        and isinstance(args[1], (int, float))
    ):
        return constant_(args[0], float(args[1]))
    t = kwargs.get("tensor")
    v = kwargs.get("val", 0.0)
    if isinstance(t, Tensor) and isinstance(v, (int, float)):
        return constant_(t, float(v))
    return zero_torch.tensor(0)


def dirac_(tensor: Tensor, groups: int = 1) -> Tensor:
    """Fills the {3, 4, 5}-dimensional input `Tensor` with the Dirac delta function in-place.

    Args:
        tensor: A {3, 4, 5}-dimensional convolution weight `Tensor`.
        groups: Number of groups in the conv layer.

    Returns:
        The modified `tensor`.
    """
    shape = tensor.shape
    dimensions = len(shape)
    if dimensions not in (3, 4, 5):
        raise ValueError("Only tensors with 3, 4, or 5 dimensions are supported")
    out_ch = shape[0]
    in_ch = shape[1]
    if out_ch % groups != 0:
        raise ValueError("dim 0 must be divisible by groups")
    out_per_group = out_ch // groups
    min_dim = min(out_per_group, in_ch)
    spatial = shape[2:]
    centers = [s // 2 for s in spatial]

    if dimensions == 3:
        grid3 = [
            [[0.0 for _ in range(spatial[0])] for _ in range(in_ch)]
            for _ in range(out_ch)
        ]
        for g in range(groups):
            for d in range(min_dim):
                grid3[g * out_per_group + d][d][centers[0]] = 1.0
        tensor._tensor = Tensor(grid3, dtype=tensor.dtype)._tensor
    elif dimensions == 4:
        grid4 = [
            [
                [[0.0 for _ in range(spatial[1])] for _ in range(spatial[0])]
                for _ in range(in_ch)
            ]
            for _ in range(out_ch)
        ]
        for g in range(groups):
            for d in range(min_dim):
                grid4[g * out_per_group + d][d][centers[0]][centers[1]] = 1.0
        tensor._tensor = Tensor(grid4, dtype=tensor.dtype)._tensor
    else:
        grid5 = [
            [
                [
                    [[0.0 for _ in range(spatial[2])] for _ in range(spatial[1])]
                    for _ in range(spatial[0])
                ]
                for _ in range(in_ch)
            ]
            for _ in range(out_ch)
        ]
        for g in range(groups):
            for d in range(min_dim):
                grid5[g * out_per_group + d][d][centers[0]][centers[1]][centers[2]] = (
                    1.0
                )
        tensor._tensor = Tensor(grid5, dtype=tensor.dtype)._tensor
    return tensor


def dirac(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Deprecated alias for `dirac_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Initialized tensor.
    """
    t = args[0] if len(args) > 0 else kwargs.get("tensor")
    g_raw = args[1] if len(args) > 1 else kwargs.get("groups", 1)
    g = int(g_raw) if isinstance(g_raw, int) else 1
    if isinstance(t, Tensor):
        return dirac_(t, groups=g)
    return zero_torch.tensor(0)


def eye_(tensor: Tensor) -> Tensor:
    """Fills the 2-dimensional input `Tensor` with the identity matrix in-place.

    Args:
        tensor: A 2-dimensional `Tensor`.

    Returns:
        The modified `tensor`.
    """
    if len(tensor.shape) != 2:
        raise ValueError("Only tensors with 2 dimensions are supported")
    tensor._tensor = zero_torch.eye(*tensor.shape)._tensor
    return tensor


def eye(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Deprecated alias for `eye_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Initialized tensor.
    """
    t = args[0] if len(args) > 0 else kwargs.get("tensor")
    if isinstance(t, Tensor):
        return eye_(t)
    return zero_torch.tensor(0)


def normal_(
    tensor: Tensor,
    mean: float = 0.0,
    std: float = 1.0,
    generator: Callable[[], float] | None = None,
) -> Tensor:
    """Fills the input Tensor with values drawn from the normal distribution `N(mean, std^2)` in-place.

    Args:
        tensor: An n-dimensional `Tensor`.
        mean: Mean of the normal distribution.
        std: Standard deviation of the normal distribution.
        generator: Optional random number generator.

    Returns:
        The modified `tensor`.
    """
    _ = generator
    new_t = zero_torch.randn(tensor.shape) * std + mean
    tensor._tensor = new_t._tensor
    return tensor


def normal(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Deprecated alias for `normal_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Initialized tensor.
    """
    t = args[0] if len(args) > 0 else kwargs.get("tensor")
    mean_raw = args[1] if len(args) > 1 else kwargs.get("mean", 0.0)
    std_raw = args[2] if len(args) > 2 else kwargs.get("std", 1.0)
    mean = float(mean_raw) if isinstance(mean_raw, (int, float)) else 0.0
    std = float(std_raw) if isinstance(std_raw, (int, float)) else 1.0
    if isinstance(t, Tensor):
        return normal_(t, mean=mean, std=std)
    return zero_torch.tensor(0)


def ones_(tensor: Tensor) -> Tensor:
    """Fills the input Tensor with the scalar value `1` in-place.

    Args:
        tensor: An n-dimensional `Tensor`.

    Returns:
        The modified `tensor`.
    """
    tensor._tensor = zero_torch.ones_like(tensor)._tensor
    return tensor


def zeros_(tensor: Tensor) -> Tensor:
    """Fills the input Tensor with the scalar value `0` in-place.

    Args:
        tensor: An n-dimensional `Tensor`.

    Returns:
        The modified `tensor`.
    """
    tensor._tensor = zero_torch.zeros_like(tensor)._tensor
    return tensor


def uniform_(
    tensor: Tensor,
    a: float = 0.0,
    b: float = 1.0,
    generator: Callable[[], float] | None = None,
) -> Tensor:
    """Fills the input Tensor with values drawn from the continuous uniform distribution `U(a, b)` in-place.

    Args:
        tensor: An n-dimensional `Tensor`.
        a: Lower bound of the uniform distribution.
        b: Upper bound of the uniform distribution.
        generator: Optional random number generator.

    Returns:
        The modified `tensor`.
    """
    _ = generator
    new_t = zero_torch.rand(tensor.shape) * (b - a) + a
    tensor._tensor = new_t._tensor
    return tensor


def uniform(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Deprecated alias for `uniform_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Initialized tensor.
    """
    t = args[0] if len(args) > 0 else kwargs.get("tensor")
    a_raw = args[1] if len(args) > 1 else kwargs.get("a", 0.0)
    b_raw = args[2] if len(args) > 2 else kwargs.get("b", 1.0)
    a = float(a_raw) if isinstance(a_raw, (int, float)) else 0.0
    b = float(b_raw) if isinstance(b_raw, (int, float)) else 1.0
    if isinstance(t, Tensor):
        return uniform_(t, a=a, b=b)
    return zero_torch.tensor(0)


def trunc_normal_(
    tensor: Tensor,
    mean: float = 0.0,
    std: float = 1.0,
    a: float = -2.0,
    b: float = 2.0,
    generator: Callable[[], float] | None = None,
) -> Tensor:
    """Fills the input Tensor with values drawn from a truncated normal distribution in-place.

    Args:
        tensor: An n-dimensional `Tensor`.
        mean: Mean of the normal distribution.
        std: Standard deviation of the normal distribution.
        a: Minimum cutoff value.
        b: Maximum cutoff value.
        generator: Optional random number generator.

    Returns:
        The modified `tensor`.
    """
    _ = generator
    raw = zero_torch.randn(tensor.shape) * std + mean
    clipped = zero_torch.clip(raw, a, b)
    tensor._tensor = clipped._tensor
    return tensor


def _calculate_fan_in_and_fan_out(tensor: Tensor) -> tuple[int, int]:
    """Computes `fan_in` and `fan_out` for a weight tensor of at least 2 dimensions.

    Args:
        tensor: Weight tensor with `len(tensor.shape) >= 2`.

    Returns:
        Tuple of `(fan_in, fan_out)`.
    """
    dimensions = len(tensor.shape)
    if dimensions < 2:
        raise ValueError(
            "Fan in and fan out can not be computed for tensor with fewer than 2 dimensions"
        )
    num_input_fmaps = tensor.shape[1]
    num_output_fmaps = tensor.shape[0]
    receptive_field_size = 1
    for s in tensor.shape[2:]:
        receptive_field_size *= s
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size
    return fan_in, fan_out


def xavier_uniform_(
    tensor: Tensor,
    gain: float = 1.0,
    generator: Callable[[], float] | None = None,
) -> Tensor:
    """Fills the input `Tensor` with values using a Xavier uniform distribution in-place.

    Args:
        tensor: An n-dimensional `Tensor` (`n >= 2`).
        gain: Optional scaling factor.
        generator: Optional random number generator.

    Returns:
        The modified `tensor`.
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    a = math.sqrt(3.0) * std
    return uniform_(tensor, -a, a, generator)


def xavier_uniform(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Deprecated alias for `xavier_uniform_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Initialized tensor.
    """
    t = args[0] if len(args) > 0 else kwargs.get("tensor")
    gain_raw = args[1] if len(args) > 1 else kwargs.get("gain", 1.0)
    gain = float(gain_raw) if isinstance(gain_raw, (int, float)) else 1.0
    if isinstance(t, Tensor):
        return xavier_uniform_(t, gain=gain)
    return zero_torch.tensor(0)


def xavier_normal_(
    tensor: Tensor,
    gain: float = 1.0,
    generator: Callable[[], float] | None = None,
) -> Tensor:
    """Fills the input `Tensor` with values using a Xavier normal distribution in-place.

    Args:
        tensor: An n-dimensional `Tensor` (`n >= 2`).
        gain: Optional scaling factor.
        generator: Optional random number generator.

    Returns:
        The modified `tensor`.
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    return normal_(tensor, 0.0, std, generator)


def xavier_normal(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Deprecated alias for `xavier_normal_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Initialized tensor.
    """
    t = args[0] if len(args) > 0 else kwargs.get("tensor")
    gain_raw = args[1] if len(args) > 1 else kwargs.get("gain", 1.0)
    gain = float(gain_raw) if isinstance(gain_raw, (int, float)) else 1.0
    if isinstance(t, Tensor):
        return xavier_normal_(t, gain=gain)
    return zero_torch.tensor(0)


def kaiming_uniform_(
    tensor: Tensor,
    a: float = 0.0,
    mode: str = "fan_in",
    nonlinearity: str = "leaky_relu",
    generator: Callable[[], float] | None = None,
) -> Tensor:
    """Fills the input `Tensor` with values using a Kaiming uniform distribution in-place.

    Args:
        tensor: An n-dimensional `Tensor` (`n >= 2`).
        a: Negative slope of the rectifier used after this layer.
        mode: Either `'fan_in'` or `'fan_out'`.
        nonlinearity: Non-linear function (`nn.functional` name).
        generator: Optional random number generator.

    Returns:
        The modified `tensor`.
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    fan = fan_in if mode == "fan_in" else fan_out
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std
    return uniform_(tensor, -bound, bound, generator)


def kaiming_uniform(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Deprecated alias for `kaiming_uniform_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Initialized tensor.
    """
    t = args[0] if len(args) > 0 else kwargs.get("tensor")
    if isinstance(t, Tensor):
        return kaiming_uniform_(t)
    return zero_torch.tensor(0)


def kaiming_normal_(
    tensor: Tensor,
    a: float = 0.0,
    mode: str = "fan_in",
    nonlinearity: str = "leaky_relu",
    generator: Callable[[], float] | None = None,
) -> Tensor:
    """Fills the input `Tensor` with values using a Kaiming normal distribution in-place.

    Args:
        tensor: An n-dimensional `Tensor` (`n >= 2`).
        a: Negative slope of the rectifier used after this layer.
        mode: Either `'fan_in'` or `'fan_out'`.
        nonlinearity: Non-linear function (`nn.functional` name).
        generator: Optional random number generator.

    Returns:
        The modified `tensor`.
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    fan = fan_in if mode == "fan_in" else fan_out
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    return normal_(tensor, 0.0, std, generator)


def kaiming_normal(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Deprecated alias for `kaiming_normal_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Initialized tensor.
    """
    t = args[0] if len(args) > 0 else kwargs.get("tensor")
    if isinstance(t, Tensor):
        return kaiming_normal_(t)
    return zero_torch.tensor(0)


def orthogonal_(
    tensor: Tensor,
    gain: float = 1.0,
    generator: Callable[[], float] | None = None,
) -> Tensor:
    """Fills the input `Tensor` with a (semi) orthogonal matrix in-place.

    Args:
        tensor: An n-dimensional `Tensor` (`n >= 2`).
        gain: Multiplicative scaling factor.
        generator: Optional random number generator.

    Returns:
        The modified `tensor`.
    """
    _ = generator
    if len(tensor.shape) < 2:
        raise ValueError("Only tensors with 2 or more dimensions are supported")
    rows = tensor.shape[0]
    cols = tensor.numel() // rows
    flattened = zero_torch.randn((max(rows, cols), min(rows, cols)))
    q, r = zero_torch.linalg.qr(flattened)
    diag = zero_torch.diag(r)
    ph = zero_torch.sign(diag)
    q_orth = q * ph
    if rows < cols:
        q_orth = zero_torch.transpose(q_orth, 0, 1)
    scaled = zero_torch.reshape(q_orth, tensor.shape) * gain
    tensor._tensor = scaled._tensor
    return tensor


def orthogonal(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Deprecated alias for `orthogonal_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Initialized tensor.
    """
    t = args[0] if len(args) > 0 else kwargs.get("tensor")
    gain_raw = args[1] if len(args) > 1 else kwargs.get("gain", 1.0)
    gain = float(gain_raw) if isinstance(gain_raw, (int, float)) else 1.0
    if isinstance(t, Tensor):
        return orthogonal_(t, gain=gain)
    return zero_torch.tensor(0)


def sparse_(
    tensor: Tensor,
    sparsity: float,
    std: float = 0.01,
    generator: Callable[[], float] | None = None,
) -> Tensor:
    """Fills the 2D input `Tensor` as a sparse matrix in-place.

    Args:
        tensor: A 2-dimensional `Tensor`.
        sparsity: The fraction of elements in each column to be set to zero.
        std: Standard deviation of the normal distribution used to generate non-zero values.
        generator: Optional random number generator.

    Returns:
        The modified `tensor`.
    """
    if len(tensor.shape) != 2:
        raise ValueError("Only tensors with 2 dimensions are supported")
    rows, cols = tensor.shape
    num_zeros = math.ceil(sparsity * rows)
    normal_(tensor, mean=0.0, std=std, generator=generator)
    mask_grid = [[1.0 for _ in range(cols)] for _ in range(rows)]
    for col_idx in range(cols):
        row_indices = list(range(rows))
        random.shuffle(row_indices)
        for zero_row in row_indices[:num_zeros]:
            mask_grid[zero_row][col_idx] = 0.0
    mask = Tensor(mask_grid, dtype=tensor.dtype)
    res = tensor * mask
    tensor._tensor = res._tensor
    return tensor


def sparse(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Deprecated alias for `sparse_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Initialized tensor.
    """
    t = args[0] if len(args) > 0 else kwargs.get("tensor")
    sp_raw = args[1] if len(args) > 1 else kwargs.get("sparsity", 0.1)
    std_raw = args[2] if len(args) > 2 else kwargs.get("std", 0.01)
    sp = float(sp_raw) if isinstance(sp_raw, (int, float)) else 0.1
    std = float(std_raw) if isinstance(std_raw, (int, float)) else 0.01
    if isinstance(t, Tensor):
        return sparse_(t, sparsity=sp, std=std)
    return zero_torch.tensor(0)
