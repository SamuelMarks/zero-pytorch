"""Type stubs for torch.nn.init."""

from __future__ import annotations

from typing import TypeAlias

import torch
from torch import Tensor

DeviceLike: TypeAlias = str | None
DTypeLike: TypeAlias = str | None

def calculate_gain(
    nonlinearity: str = ...,
    param: int | float | None = None,
) -> float: ...
def constant(*args: Tensor | float, **kwargs: Tensor | float) -> Tensor: ...
def constant_(tensor: Tensor = ..., val: float = ...) -> Tensor: ...
def dirac(*args: Tensor | int, **kwargs: Tensor | int) -> Tensor: ...
def dirac_(tensor: Tensor = ..., groups: int | None = 1) -> Tensor: ...
def eye(*args: Tensor, **kwargs: Tensor) -> Tensor: ...
def eye_(tensor: Tensor = ...) -> Tensor: ...
def kaiming_normal(
    *args: Tensor | float | str, **kwargs: Tensor | float | str
) -> Tensor: ...
def kaiming_normal_(
    tensor: Tensor = ...,
    a: float = 0.0,
    mode: str = "fan_in",
    nonlinearity: str = "leaky_relu",
    generator: torch.Generator | None = None,
) -> Tensor: ...
def kaiming_uniform(
    *args: Tensor | float | str, **kwargs: Tensor | float | str
) -> Tensor: ...
def kaiming_uniform_(
    tensor: Tensor = ...,
    a: float = 0.0,
    mode: str = "fan_in",
    nonlinearity: str = "leaky_relu",
    generator: torch.Generator | None = None,
) -> Tensor: ...
def normal(*args: Tensor | float, **kwargs: Tensor | float) -> Tensor: ...
def normal_(
    tensor: Tensor = ...,
    mean: float = 0.0,
    std: float = 1.0,
    generator: torch.Generator | None = None,
) -> Tensor: ...
def ones_(tensor: Tensor = ...) -> Tensor: ...
def orthogonal(*args: Tensor | float, **kwargs: Tensor | float) -> Tensor: ...
def orthogonal_(
    tensor: Tensor = ...,
    gain: float = 1.0,
    generator: torch.Generator | None = None,
) -> Tensor: ...
def sparse(*args: Tensor | float, **kwargs: Tensor | float) -> Tensor: ...
def sparse_(
    tensor: Tensor = ...,
    sparsity: float = ...,
    std: float = 0.01,
    generator: torch.Generator | None = None,
) -> Tensor: ...
def trunc_normal_(
    tensor: Tensor = ...,
    mean: float = 0.0,
    std: float = 1.0,
    a: float = -2.0,
    b: float = 2.0,
    generator: torch.Generator | None = None,
) -> Tensor: ...
def uniform(*args: Tensor | float, **kwargs: Tensor | float) -> Tensor: ...
def uniform_(
    tensor: Tensor = ...,
    a: float = 0.0,
    b: float = 1.0,
    generator: torch.Generator | None = None,
) -> Tensor: ...
def xavier_normal(*args: Tensor | float, **kwargs: Tensor | float) -> Tensor: ...
def xavier_normal_(
    tensor: Tensor = ...,
    gain: float = 1.0,
    generator: torch.Generator | None = None,
) -> Tensor: ...
def xavier_uniform(*args: Tensor | float, **kwargs: Tensor | float) -> Tensor: ...
def xavier_uniform_(
    tensor: Tensor = ...,
    gain: float = 1.0,
    generator: torch.Generator | None = None,
) -> Tensor: ...
def zeros_(tensor: Tensor = ...) -> Tensor: ...
