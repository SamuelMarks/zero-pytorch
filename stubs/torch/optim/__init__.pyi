"""Type stubs for torch.optim."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TypeAlias

from torch import Tensor

ParamsArg: TypeAlias = (
    Iterable[Tensor]
    | Iterable[dict[str, Tensor | float | int | bool | tuple[float, float] | None]]
    | Iterable[tuple[str, Tensor]]
)

class ASGD:
    def __init__(
        self,
        params: ParamsArg = ...,
        lr: float | Tensor = 0.01,
        lambd: float = 0.0001,
        alpha: float = 0.75,
        t0: float = 1000000.0,
        weight_decay: float = 0.0,
        foreach: bool | None = None,
        maximize: bool = False,
        differentiable: bool = False,
        capturable: bool = False,
    ) -> None: ...

class Adadelta:
    def __init__(
        self,
        params: ParamsArg = ...,
        lr: float | Tensor = 1.0,
        rho: float = 0.9,
        eps: float = 1e-06,
        weight_decay: float = 0.0,
        foreach: bool | None = None,
        capturable: bool = False,
        maximize: bool = False,
        differentiable: bool = False,
    ) -> None: ...

class Adafactor:
    def __init__(
        self,
        params: ParamsArg = ...,
        lr: float | Tensor = 0.01,
        beta2_decay: float = -0.8,
        eps: tuple[float | None, float] = (None, 0.001),
        d: float = 1.0,
        weight_decay: float = 0.0,
        foreach: bool | None = None,
        maximize: bool = False,
    ) -> None: ...

class Adagrad:
    def __init__(
        self,
        params: ParamsArg = ...,
        lr: float | Tensor = 0.01,
        lr_decay: float = 0.0,
        weight_decay: float = 0.0,
        initial_accumulator_value: float = 0.0,
        eps: float = 1e-10,
        foreach: bool | None = None,
        maximize: bool = False,
        differentiable: bool = False,
        fused: bool | None = None,
    ) -> None: ...

class Adam:
    def __init__(
        self,
        params: ParamsArg = ...,
        lr: float | Tensor = 0.001,
        betas: tuple[float | Tensor, float | Tensor] = (0.9, 0.999),
        eps: float = 1e-08,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
        foreach: bool | None = None,
        maximize: bool = False,
        capturable: bool = False,
        differentiable: bool = False,
        fused: bool | None = None,
        decoupled_weight_decay: bool = False,
    ) -> None: ...

class AdamW:
    def __init__(
        self,
        params: ParamsArg = ...,
        lr: float | Tensor = 0.001,
        betas: tuple[float | Tensor, float | Tensor] = (0.9, 0.999),
        eps: float = 1e-08,
        weight_decay: float = 0.01,
        amsgrad: bool = False,
        maximize: bool = False,
        foreach: bool | None = None,
        capturable: bool = False,
        differentiable: bool = False,
        fused: bool | None = None,
    ) -> None: ...

class Adamax:
    def __init__(
        self,
        params: ParamsArg = ...,
        lr: float | Tensor = 0.002,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-08,
        weight_decay: float = 0.0,
        foreach: bool | None = None,
        maximize: bool = False,
        differentiable: bool = False,
        capturable: bool = False,
    ) -> None: ...

class LBFGS:
    def __init__(
        self,
        params: ParamsArg = ...,
        lr: float | None = 1.0,
        max_iter: int | None = 20,
        max_eval: int | None = None,
        tolerance_grad: float | None = 1e-07,
        tolerance_change: float | None = 1e-09,
        history_size: int | None = 100,
        line_search_fn: str | None = None,
    ) -> None: ...

class NAdam:
    def __init__(
        self,
        params: ParamsArg = ...,
        lr: float | Tensor = 0.002,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-08,
        weight_decay: float = 0.0,
        momentum_decay: float = 0.004,
        decoupled_weight_decay: bool = False,
        foreach: bool | None = None,
        maximize: bool = False,
        capturable: bool = False,
        differentiable: bool = False,
    ) -> None: ...

class RAdam:
    def __init__(
        self,
        params: ParamsArg = ...,
        lr: float | Tensor = 0.001,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-08,
        weight_decay: float = 0.0,
        decoupled_weight_decay: bool = False,
        foreach: bool | None = None,
        maximize: bool = False,
        capturable: bool = False,
        differentiable: bool = False,
    ) -> None: ...

class RMSprop:
    def __init__(
        self,
        params: ParamsArg = ...,
        lr: float | Tensor = 0.01,
        alpha: float = 0.99,
        eps: float = 1e-08,
        weight_decay: float = 0.0,
        momentum: float = 0.0,
        centered: bool = False,
        capturable: bool = False,
        foreach: bool | None = None,
        maximize: bool = False,
        differentiable: bool = False,
    ) -> None: ...

class Rprop:
    def __init__(
        self,
        params: ParamsArg = ...,
        lr: float | Tensor = 0.01,
        etas: tuple[float, float] = (0.5, 1.2),
        step_sizes: tuple[float, float] = (1e-06, 50.0),
        capturable: bool = False,
        foreach: bool | None = None,
        maximize: bool = False,
        differentiable: bool = False,
    ) -> None: ...

class SGD:
    def __init__(
        self,
        params: ParamsArg = ...,
        lr: float | Tensor = 0.001,
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float | Tensor = 0.0,
        nesterov: bool = False,
        maximize: bool = False,
        foreach: bool | None = None,
        differentiable: bool = False,
        fused: bool | None = None,
    ) -> None: ...

class SparseAdam:
    def __init__(
        self,
        params: ParamsArg = ...,
        lr: float | Tensor = 0.001,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-08,
        maximize: bool = False,
    ) -> None: ...
