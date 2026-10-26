"""Centralized strict type aliases for zero_torch (zero Any, zero object)."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Literal, TypedDict, Union

from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor as _SwitcherooTensor
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor

if TYPE_CHECKING:
    from types import EllipsisType
    from typing import TypeAlias

    from zero_torch.nn.module import Module, Parameter
    from zero_torch.tensor import Tensor

SliceIndexType: TypeAlias = "slice[int | None, int | None, int | None]"
Scalar: TypeAlias = Union[int, float, bool, complex]
NestedSequence: TypeAlias = Sequence["Scalar | NestedSequence"]
NestedList: TypeAlias = list["Scalar | NestedList"]
TensorOrScalar: TypeAlias = "Tensor | _SwitcherooTensor | ProxyTensor | Scalar"
TensorLike: TypeAlias = (
    "Tensor | _SwitcherooTensor | ProxyTensor | Scalar | NestedSequence"
)
ShapeLike: TypeAlias = Union[int, tuple[int, ...], list[int]]
ShapeArg: TypeAlias = Union[int, tuple[int, ...], list[int], Sequence[int]]
DimArg: TypeAlias = Union[int, tuple[int, ...], list[int], Sequence[int], None]
PadArg: TypeAlias = Union[int, tuple[int, ...], list[int], Sequence[int], str]
DTypeLiteral: TypeAlias = Literal[
    "float32",
    "float64",
    "float16",
    "bfloat16",
    "int64",
    "int32",
    "int16",
    "int8",
    "uint8",
    "bool",
    "complex64",
    "complex128",
]
ReductionType: TypeAlias = Literal["none", "mean", "sum", "batchmean"]
PaddingMode: TypeAlias = Literal[
    "constant", "reflect", "replicate", "circular", "zeros"
]
ReductionMode: TypeAlias = str
NormOrd: TypeAlias = Union[int, float, str, None]
DTypeLike: TypeAlias = Union[DType, str, None]
DeviceLike: TypeAlias = Union[Device, str, None]

Size1d: TypeAlias = Union[int, tuple[int]]
Size2d: TypeAlias = Union[int, tuple[int, int]]
Size3d: TypeAlias = Union[int, tuple[int, int, int]]
SizeNd: TypeAlias = Union[int, tuple[int, ...], list[int]]

OptSize1d: TypeAlias = Union[int, tuple[Union[int, None]], None]
OptSize2d: TypeAlias = Union[int, tuple[Union[int, None], Union[int, None]], None]
OptSize3d: TypeAlias = Union[
    int, tuple[Union[int, None], Union[int, None], Union[int, None]], None
]
OptSizeNd: TypeAlias = Union[
    int, tuple[Union[int, None], ...], list[Union[int, None]], None
]

PaddingArg1d: TypeAlias = Union[int, tuple[int], str]
PaddingArg2d: TypeAlias = Union[int, tuple[int, int], str]
PaddingArg3d: TypeAlias = Union[int, tuple[int, int, int], str]
PaddingArgNd: TypeAlias = Union[int, tuple[int, ...], list[int], str]

OpArg: TypeAlias = "Tensor | _SwitcherooTensor | ProxyTensor | Scalar | NestedSequence | DType | Device | str | SliceIndexType | EllipsisType | None"
OpReturn: TypeAlias = "Tensor | tuple[Tensor, ...] | list[Tensor]"


class OptimizerStateEntry(TypedDict, total=False):
    """Strongly typed per-parameter optimizer state dictionary."""

    step: Tensor
    exp_avg: Tensor
    exp_avg_sq: Tensor
    max_exp_avg_sq: Tensor
    exp_inf: Tensor
    momentum_buffer: Tensor
    sum: Tensor
    square_avg: Tensor
    acc_delta: Tensor
    grad_avg: Tensor
    mu_product: Tensor
    eta: Tensor
    ax: Tensor
    prev: Tensor
    step_size: Tensor
    exp_avg_sq_row: Tensor
    exp_avg_sq_col: Tensor
    prev_param: Tensor
    prev_grad: Tensor


class ParamGroupDict(TypedDict, total=False):
    """Strongly typed optimizer parameter group configuration dictionary."""

    params: list[Parameter] | list[Tensor] | Iterable[Parameter] | Iterable[Tensor]
    lr: float
    initial_lr: float
    betas: tuple[float, float]
    eps: float
    weight_decay: float
    amsgrad: bool
    momentum: float
    dampening: float
    nesterov: bool
    rho: float
    lr_decay: float
    initial_accumulator_value: float
    lambd: float
    alpha: float
    t0: float
    max_iter: int | None
    max_eval: int | None
    tolerance_grad: float | None
    tolerance_change: float | None
    history_size: int | None
    line_search_fn: str | None
    momentum_decay: float
    decoupled_weight_decay: bool
    centered: bool
    etas: tuple[float, float]
    step_sizes: tuple[float, float]
    ns_coefficients: tuple[float, float, float]
    ns_steps: int
    adjust_lr_fn: str | None
    maximize: bool
    foreach: bool | None
    differentiable: bool
    capturable: bool
    fused: bool | None
    beta2_decay: float
    d: float


ParamGroupValue: TypeAlias = "list[Parameter] | list[Tensor] | Tensor | float | int | bool | str | tuple[float, float] | tuple[float, float, float] | tuple[int, int] | None"
ParamsArg: TypeAlias = "Iterable[Parameter] | Iterable[Tensor] | Iterable[ParamGroupDict] | Iterable[Parameter | Tensor | ParamGroupDict]"
LossClosure: TypeAlias = "Callable[[], Tensor | float] | None"

CollateValue: TypeAlias = "Tensor | Scalar | str | bytes | Mapping[str, CollateValue] | Sequence[CollateValue]"
HookCallable: TypeAlias = "Callable[[Tensor], Tensor | None]"
ModuleDictArg: TypeAlias = "Mapping[str, Module] | Iterable[tuple[str, Module]] | None"
ParamDictArg: TypeAlias = (
    "Mapping[str, Parameter] | Iterable[tuple[str, Parameter]] | None"
)

__all__ = [
    "CollateValue",
    "DTypeLike",
    "DTypeLiteral",
    "DeviceLike",
    "DimArg",
    "HookCallable",
    "LossClosure",
    "ModuleDictArg",
    "NestedList",
    "NestedSequence",
    "NormOrd",
    "OpArg",
    "OpReturn",
    "OptSize1d",
    "OptSize2d",
    "OptSize3d",
    "OptSizeNd",
    "OptimizerStateEntry",
    "PadArg",
    "PaddingArg1d",
    "PaddingArg2d",
    "PaddingArg3d",
    "PaddingArgNd",
    "PaddingMode",
    "ParamDictArg",
    "ParamGroupDict",
    "ParamGroupValue",
    "ParamsArg",
    "ReductionMode",
    "ReductionType",
    "Scalar",
    "ShapeArg",
    "ShapeLike",
    "Size1d",
    "Size2d",
    "Size3d",
    "SizeNd",
    "SliceIndexType",
    "TensorLike",
    "TensorOrScalar",
]
