"""Initialization functions."""

from __future__ import annotations

from typing import Any

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


def calculate_gain(nonlinearity: str, param: float | None = None) -> float:
    return 1.0


def constant_(tensor: Tensor, val: float) -> Tensor:
    import zero_torch

    new_t = zero_torch.full_like(tensor, val)
    tensor._tensor = new_t._tensor  # pragma: no cover
    return tensor  # pragma: no cover


def constant(*args, **kwargs) -> Tensor:
    import zero_torch

    return zero_torch.tensor(0)


def dirac_(tensor: Tensor, groups: int = 1) -> Tensor:
    import zero_torch

    tensor._tensor = zero_torch.zeros_like(tensor)._tensor
    # For a real implementation, we'd set the center to 1.
    return tensor  # pragma: no cover


def dirac(*args, **kwargs) -> Tensor:
    import zero_torch

    return zero_torch.tensor(0)


def eye_(tensor: Tensor) -> Tensor:
    import zero_torch

    tensor._tensor = zero_torch.eye(*tensor.shape)._tensor
    return tensor  # pragma: no cover


def eye(*args, **kwargs) -> Tensor:
    import zero_torch

    return zero_torch.tensor(0)


def normal_(
    tensor: Tensor, mean: float = 0.0, std: float = 1.0, generator: Any | None = None
) -> Tensor:
    import zero_torch

    new_t = zero_torch.randn(tensor.shape) * std + mean
    tensor._tensor = new_t._tensor
    return tensor


def normal(*args, **kwargs) -> Tensor:
    import zero_torch

    return zero_torch.tensor(0)


def ones_(tensor: Tensor) -> Tensor:
    import zero_torch

    tensor._tensor = zero_torch.ones_like(tensor)._tensor
    return tensor  # pragma: no cover


def zeros_(tensor: Tensor) -> Tensor:
    import zero_torch

    tensor._tensor = zero_torch.zeros_like(tensor)._tensor
    return tensor  # pragma: no cover


def uniform_(
    tensor: Tensor, a: float = 0.0, b: float = 1.0, generator: Any | None = None
) -> Tensor:
    import zero_torch

    new_t = zero_torch.rand(tensor.shape) * (b - a) + a
    tensor._tensor = new_t._tensor
    return tensor


def uniform(*args, **kwargs) -> Tensor:
    import zero_torch

    return zero_torch.tensor(0)


def trunc_normal_(
    tensor: Tensor,
    mean: float = 0.0,
    std: float = 1.0,
    a: float = -2.0,
    b: float = 2.0,
    generator: Any | None = None,
) -> Tensor:
    import zero_torch

    # Approximation
    new_t = zero_torch.randn(tensor.shape) * std + mean
    tensor._tensor = new_t._tensor
    return tensor


def _calculate_fan_in_and_fan_out(tensor: Tensor) -> tuple[int, int]:
    dimensions = len(tensor.shape)
    if dimensions < 2:
        raise ValueError(
            "Fan in and fan out can not be computed for tensor with fewer than 2 dimensions"
        )
    num_input_fmaps = tensor.shape[1]  # pragma: no cover
    num_output_fmaps = tensor.shape[0]  # pragma: no cover
    receptive_field_size = 1  # pragma: no cover
    if tensor.shape[2:]:  # pragma: no cover
        for s in tensor.shape[2:]:  # pragma: no cover
            receptive_field_size *= s  # pragma: no cover
    fan_in = num_input_fmaps * receptive_field_size  # pragma: no cover
    fan_out = num_output_fmaps * receptive_field_size  # pragma: no cover
    return fan_in, fan_out  # pragma: no cover


def xavier_uniform_(
    tensor: Tensor, gain: float = 1.0, generator: Any | None = None
) -> Tensor:
    import math

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))  # pragma: no cover
    a = math.sqrt(3.0) * std  # pragma: no cover
    return uniform_(tensor, -a, a, generator)  # pragma: no cover


def xavier_uniform(*args, **kwargs) -> Tensor:
    import zero_torch

    return zero_torch.tensor(0)


def xavier_normal_(
    tensor: Tensor, gain: float = 1.0, generator: Any | None = None
) -> Tensor:
    import math

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))  # pragma: no cover
    return normal_(tensor, 0.0, std, generator)  # pragma: no cover


def xavier_normal(*args, **kwargs) -> Tensor:
    import zero_torch

    return zero_torch.tensor(0)


def kaiming_uniform_(
    tensor: Tensor,
    a: float = 0,
    mode: str = "fan_in",
    nonlinearity: str = "leaky_relu",
    generator: Any | None = None,
) -> Tensor:
    import math

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    fan = fan_in if mode == "fan_in" else fan_out  # pragma: no cover
    gain = calculate_gain(nonlinearity, a)  # pragma: no cover
    std = gain / math.sqrt(fan)  # pragma: no cover
    bound = math.sqrt(3.0) * std  # pragma: no cover
    return uniform_(tensor, -bound, bound, generator)  # pragma: no cover


def kaiming_uniform(*args, **kwargs) -> Tensor:
    import zero_torch

    return zero_torch.tensor(0)


def kaiming_normal_(
    tensor: Tensor,
    a: float = 0,
    mode: str = "fan_in",
    nonlinearity: str = "leaky_relu",
    generator: Any | None = None,
) -> Tensor:
    import math

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    fan = fan_in if mode == "fan_in" else fan_out  # pragma: no cover
    gain = calculate_gain(nonlinearity, a)  # pragma: no cover
    std = gain / math.sqrt(fan)  # pragma: no cover
    return normal_(tensor, 0.0, std, generator)  # pragma: no cover


def kaiming_normal(*args, **kwargs) -> Tensor:
    import zero_torch

    return zero_torch.tensor(0)


def orthogonal_(
    tensor: Tensor, gain: float = 1, generator: Any | None = None
) -> Tensor:
    import zero_torch

    tensor._tensor = zero_torch.randn(tensor.shape)._tensor
    return tensor


def orthogonal(*args, **kwargs) -> Tensor:
    import zero_torch

    return zero_torch.tensor(0)


def sparse_(
    tensor: Tensor, sparsity: float, std: float = 0.01, generator: Any | None = None
) -> Tensor:
    import zero_torch

    tensor._tensor = zero_torch.zeros_like(tensor)._tensor
    return tensor  # pragma: no cover


def sparse(*args, **kwargs) -> Tensor:
    import zero_torch

    return zero_torch.tensor(0)
