"""Centralized strict type aliases for zero_torch (zero Any, zero object)."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Union

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
TensorLike: TypeAlias = (
    "Tensor | _SwitcherooTensor | ProxyTensor | Scalar | NestedSequence"
)
ShapeLike: TypeAlias = Union[int, tuple[int, ...], list[int]]
DimArg: TypeAlias = Union[int, tuple[int, ...], list[int], None]
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

ParamGroupValue: TypeAlias = "list[Parameter] | list[Tensor] | Tensor | float | int | bool | str | tuple[float, float] | tuple[float, float, float] | tuple[int, int] | None"
ParamGroupDict: TypeAlias = dict[str, ParamGroupValue]
ParamsArg: TypeAlias = (
    "Iterable[Parameter] | Iterable[Tensor] | Iterable[ParamGroupDict]"
)
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
    "DeviceLike",
    "DimArg",
    "HookCallable",
    "LossClosure",
    "ModuleDictArg",
    "NestedList",
    "NestedSequence",
    "OpArg",
    "OpReturn",
    "OptSize1d",
    "OptSize2d",
    "OptSize3d",
    "OptSizeNd",
    "PaddingArg1d",
    "PaddingArg2d",
    "PaddingArg3d",
    "PaddingArgNd",
    "ParamDictArg",
    "ParamGroupDict",
    "ParamGroupValue",
    "ParamsArg",
    "Scalar",
    "ShapeLike",
    "Size1d",
    "Size2d",
    "Size3d",
    "SizeNd",
    "SliceIndexType",
    "TensorLike",
]
