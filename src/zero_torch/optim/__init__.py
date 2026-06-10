import ml_switcheroo

"Optimizers."
from typing import Any, Iterable, Optional, Tuple, Union
from zero_torch.tensor import Tensor

__all__ = [
    "ASGD",
    "Adadelta",
    "Adafactor",
    "Adagrad",
    "Adam",
    "AdamW",
    "Adamax",
    "LBFGS",
    "NAdam",
    "RAdam",
    "RMSprop",
    "Rprop",
    "SGD",
    "SparseAdam",
]


class Optimizer:
    def __init__(self, params: Any, defaults: dict) -> None:
        pass

    def step(self, closure: Optional[Any] = None) -> Optional[Any]:
        pass

    def zero_grad(self, set_to_none: bool = False) -> None:
        pass


class ASGD(Optimizer):
    def __init__(
        self,
        params: Any,
        lr: Union[float, Tensor] = 0.01,
        lambd: float = 0.0001,
        alpha: float = 0.75,
        t0: float = 1000000.0,
        weight_decay: float = 0,
        foreach: Optional[bool] = None,
        maximize: bool = False,
        differentiable: bool = False,
        capturable: bool = False,
    ) -> None:
        pass


class Adadelta(Optimizer):
    def __init__(
        self,
        params: Any,
        lr: Union[float, Tensor] = 1.0,
        rho: float = 0.9,
        eps: float = 1e-06,
        weight_decay: float = 0,
        foreach: Optional[bool] = None,
        capturable: bool = False,
        maximize: bool = False,
        differentiable: bool = False,
    ) -> None:
        pass


class Adafactor(Optimizer):
    def __init__(
        self,
        params: Any,
        lr: Union[float, Tensor] = 0.01,
        beta2_decay: float = -0.8,
        eps: Tuple[Optional[float], float] = (None, 0.001),
        d: float = 1.0,
        weight_decay: float = 0.0,
        foreach: Optional[bool] = None,
        maximize: bool = False,
    ) -> None:
        pass


class Adagrad(Optimizer):
    def __init__(
        self,
        params: Any,
        lr: Union[float, Tensor] = 0.01,
        lr_decay: float = 0,
        weight_decay: float = 0,
        initial_accumulator_value: float = 0,
        eps: float = 1e-10,
        foreach: Optional[bool] = None,
        maximize: bool = False,
        differentiable: bool = False,
        fused: Optional[bool] = None,
    ) -> None:
        pass


class Adam(Optimizer):
    def __init__(
        self,
        params: Any,
        lr: Union[float, Tensor] = 0.001,
        betas: Tuple[Union[float, Tensor], Union[float, Tensor]] = (0.9, 0.999),
        eps: float = 1e-08,
        weight_decay: float = 0,
        amsgrad: bool = False,
        foreach: Optional[bool] = None,
        maximize: bool = False,
        capturable: bool = False,
        differentiable: bool = False,
        fused: Optional[bool] = None,
        decoupled_weight_decay: bool = False,
    ) -> None:
        pass


class AdamW(Optimizer):
    def __init__(
        self,
        params: Any,
        lr: Union[float, Tensor] = 0.001,
        betas: Tuple[Union[float, Tensor], Union[float, Tensor]] = (0.9, 0.999),
        eps: float = 1e-08,
        weight_decay: float = 0.01,
        amsgrad: bool = False,
        maximize: bool = False,
        foreach: Optional[bool] = None,
        capturable: bool = False,
        differentiable: bool = False,
        fused: Optional[bool] = None,
    ) -> None:
        pass


class Adamax(Optimizer):
    def __init__(
        self,
        params: Any,
        lr: Union[float, Tensor] = 0.002,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-08,
        weight_decay: float = 0,
        foreach: Optional[bool] = None,
        maximize: bool = False,
        differentiable: bool = False,
        capturable: bool = False,
    ) -> None:
        pass


class LBFGS(Optimizer):
    def __init__(
        self,
        params: Iterable,
        lr: Optional[float] = 1,
        max_iter: Optional[int] = 20,
        max_eval: Optional[int] = None,
        tolerance_grad: Optional[float] = 1e-07,
        tolerance_change: Optional[float] = 1e-09,
        history_size: Optional[int] = 100,
        line_search_fn: Optional[str] = None,
    ) -> None:
        pass


class NAdam(Optimizer):
    def __init__(
        self,
        params: Any,
        lr: Union[float, Tensor] = 0.002,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-08,
        weight_decay: float = 0,
        momentum_decay: float = 0.004,
        decoupled_weight_decay: bool = False,
        foreach: Optional[bool] = None,
        maximize: bool = False,
        capturable: bool = False,
        differentiable: bool = False,
    ) -> None:
        pass


class RAdam(Optimizer):
    def __init__(
        self,
        params: Any,
        lr: Union[float, Tensor] = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-08,
        weight_decay: float = 0,
        decoupled_weight_decay: bool = False,
        foreach: Optional[bool] = None,
        maximize: bool = False,
        capturable: bool = False,
        differentiable: bool = False,
    ) -> None:
        pass


class RMSprop(Optimizer):
    def __init__(
        self,
        params: Any,
        lr: Union[float, Tensor] = 0.01,
        alpha: float = 0.99,
        eps: float = 1e-08,
        weight_decay: float = 0,
        momentum: float = 0,
        centered: bool = False,
        capturable: bool = False,
        foreach: Optional[bool] = None,
        maximize: bool = False,
        differentiable: bool = False,
    ) -> None:
        pass


class Rprop(Optimizer):
    def __init__(
        self,
        params: Any,
        lr: Union[float, Tensor] = 0.01,
        etas: Tuple[float, float] = (0.5, 1.2),
        step_sizes: Tuple[float, float] = (1e-06, 50),
        capturable: bool = False,
        foreach: Optional[bool] = None,
        maximize: bool = False,
        differentiable: bool = False,
    ) -> None:
        pass


class SGD(Optimizer):
    def __init__(
        self,
        params: Any,
        lr: Union[float, Tensor] = 0.001,
        momentum: float = 0,
        dampening: float = 0,
        weight_decay: Union[float, Tensor] = 0,
        nesterov: bool = False,
        maximize: bool = False,
        foreach: Optional[bool] = None,
        differentiable: bool = False,
        fused: Optional[bool] = None,
    ) -> None:
        pass


class SparseAdam(Optimizer):
    def __init__(
        self,
        params: Any,
        lr: Union[float, Tensor] = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-08,
        maximize: bool = False,
    ) -> None:
        pass
