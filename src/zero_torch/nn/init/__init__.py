"""Module."""

import ml_switcheroo

"Initialization functions."
from typing import Any, Optional, Tuple
import math
import numpy as np
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
    """Function."""
    pass


def constant_(tensor: Tensor, val: float) -> Tensor:
    """Function."""
    pass


def constant(*args, **kwargs) -> Tensor:
    """Function."""
    pass


def dirac_(tensor: Tensor, groups: int = 1) -> Tensor:
    """Function."""
    pass


def dirac(*args, **kwargs) -> Tensor:
    """Function."""
    pass


def eye_(tensor: Tensor) -> Tensor:
    """Function."""
    pass


def eye(*args, **kwargs) -> Tensor:
    """Function."""
    pass


def normal_(
    tensor: Tensor, mean: float = 0.0, std: float = 1.0, generator: Optional[Any] = None
) -> Tensor:
    """Function."""
    pass


def normal(*args, **kwargs) -> Tensor:
    """Function."""
    pass


def ones_(tensor: Tensor) -> Tensor:
    """Function."""
    pass


def zeros_(tensor: Tensor) -> Tensor:
    """Function."""
    pass


def uniform_(
    tensor: Tensor, a: float = 0.0, b: float = 1.0, generator: Optional[Any] = None
) -> Tensor:
    """Function."""
    pass


def uniform(*args, **kwargs) -> Tensor:
    """Function."""
    pass


def trunc_normal_(
    tensor: Tensor,
    mean: float = 0.0,
    std: float = 1.0,
    a: float = -2.0,
    b: float = 2.0,
    generator: Optional[Any] = None,
) -> Tensor:
    """Function."""
    pass


def _calculate_fan_in_and_fan_out(tensor: Tensor) -> Tuple[int, int]:
    """Function."""
    pass


def xavier_uniform_(
    tensor: Tensor, gain: float = 1.0, generator: Optional[Any] = None
) -> Tensor:
    """Function."""
    pass


def xavier_uniform(*args, **kwargs) -> Tensor:
    """Function."""
    pass


def xavier_normal_(
    tensor: Tensor, gain: float = 1.0, generator: Optional[Any] = None
) -> Tensor:
    """Function."""
    pass


def xavier_normal(*args, **kwargs) -> Tensor:
    """Function."""
    pass


def kaiming_uniform_(
    tensor: Tensor,
    a: float = 0,
    mode: str = "fan_in",
    nonlinearity: str = "leaky_relu",
    generator: Optional[Any] = None,
) -> Tensor:
    """Function."""
    pass


def kaiming_uniform(*args, **kwargs) -> Tensor:
    """Function."""
    pass


def kaiming_normal_(
    tensor: Tensor,
    a: float = 0,
    mode: str = "fan_in",
    nonlinearity: str = "leaky_relu",
    generator: Optional[Any] = None,
) -> Tensor:
    """Function."""
    pass


def kaiming_normal(*args, **kwargs) -> Tensor:
    """Function."""
    pass


def orthogonal_(
    tensor: Tensor, gain: float = 1, generator: Optional[Any] = None
) -> Tensor:
    """Function."""
    pass


def orthogonal(*args, **kwargs) -> Tensor:
    """Function."""
    pass


def sparse_(
    tensor: Tensor, sparsity: float, std: float = 0.01, generator: Optional[Any] = None
) -> Tensor:
    """Function."""
    pass


def sparse(*args, **kwargs) -> Tensor:
    """Function."""
    pass
