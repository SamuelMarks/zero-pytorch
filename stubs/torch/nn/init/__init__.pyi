from typing import Any, Optional, Iterable, TypeVar
import torch
from torch import Tensor

_R = Any
_FanMode = Any
_NonlinearityType = Any
_Optional = Optional
_empty = Any
ParamsT = Any
_size_1_t = Any
_size_2_t = Any
_size_3_t = Any
_size_any_t = Any
_size_2_opt_t = Any
_size_3_opt_t = Any
_ratio_2_t = Any
_ratio_3_t = Any
optional = Any
UninitializedParameter = Any
Parameter = Any
real = Any
NamedShape = Any
_ratio_any_t = Any
_size_any_opt_t = Any
_size = Any
iterable = Iterable
NoneType = type(None)
_T_co = TypeVar("_T_co", covariant=True)
_T = TypeVar("_T")
_collate_fn_t = Any
_worker_init_fn_t = Any
_SnapshotState = Any

def calculate_gain(
    nonlinearity: _NonlinearityType = ...,
    param: _Optional[int | float] = "```(None)```",
) -> float: ...
def constant(*args, **kwargs) -> ~_R: ...
def constant_(tensor: Tensor = ..., val: float = ...) -> torch.Tensor: ...
def dirac(*args, **kwargs) -> ~_R: ...
def dirac_(tensor: Tensor = ..., groups: int | None = 1) -> torch.Tensor: ...
def eye(*args, **kwargs) -> ~_R: ...
def eye_(tensor: Tensor = ...) -> torch.Tensor: ...
def kaiming_normal(*args, **kwargs) -> ~_R: ...
def kaiming_normal_(
    tensor: Tensor = ...,
    a: float = 0,
    mode: _FanMode = "fan_in",
    nonlinearity: _NonlinearityType = "leaky_relu",
    generator: _Optional[torch.Generator] = "```(None)```",
) -> torch.Tensor: ...
def kaiming_uniform(*args, **kwargs) -> ~_R: ...
def kaiming_uniform_(
    tensor: Tensor = ...,
    a: float = 0,
    mode: _FanMode = "fan_in",
    nonlinearity: _NonlinearityType = "leaky_relu",
    generator: _Optional[torch.Generator] = "```(None)```",
) -> torch.Tensor: ...
def normal(*args, **kwargs) -> ~_R: ...
def normal_(
    tensor: Tensor = ...,
    mean: float = 0.0,
    std: float = 1.0,
    generator: _Optional[torch.Generator] = "```(None)```",
) -> torch.Tensor: ...
def ones_(tensor: Tensor = ...) -> torch.Tensor: ...
def orthogonal(*args, **kwargs) -> ~_R: ...
def orthogonal_(
    tensor: Tensor = ...,
    gain: float = 1,
    generator: _Optional[torch.Generator] = "```(None)```",
) -> torch.Tensor: ...
def sparse(*args, **kwargs) -> ~_R: ...
def sparse_(
    tensor: Tensor = ...,
    sparsity: float = ...,
    std: float = 0.01,
    generator: _Optional[torch.Generator] = "```(None)```",
) -> torch.Tensor: ...
def trunc_normal_(
    tensor: Tensor = ...,
    mean: float = 0.0,
    std: float = 1.0,
    a: float = -2.0,
    b: float = 2.0,
    generator: _Optional[torch.Generator] = "```(None)```",
) -> torch.Tensor: ...
def uniform(*args, **kwargs) -> ~_R: ...
def uniform_(
    tensor: Tensor = ...,
    a: float = 0.0,
    b: float = 1.0,
    generator: _Optional[torch.Generator] = "```(None)```",
) -> torch.Tensor: ...
def xavier_normal(*args, **kwargs) -> ~_R: ...
def xavier_normal_(
    tensor: Tensor = ...,
    gain: float = 1.0,
    generator: _Optional[torch.Generator] = "```(None)```",
) -> torch.Tensor: ...
def xavier_uniform(*args, **kwargs) -> ~_R: ...
def xavier_uniform_(
    tensor: Tensor = ...,
    gain: float = 1.0,
    generator: _Optional[torch.Generator] = "```(None)```",
) -> torch.Tensor: ...
def zeros_(tensor: Tensor = ...) -> torch.Tensor: ...
