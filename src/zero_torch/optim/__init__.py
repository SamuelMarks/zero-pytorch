"""Optimizers module."""

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
    """Base class for all optimizers."""

    def __init__(self, params: Any, defaults: dict = None) -> None:
        """Initializes the Optimizer.

        Args:
            params (Any): An iterable of parameters to optimize or dicts defining parameter groups.
            defaults (dict, optional): A dict containing default values of optimization options. Defaults to None.
        """
        self.params = list(params)

    def step(self, closure: Optional[Any] = None) -> Optional[Any]:
        """Performs a single optimization step.

        Args:
            closure (Optional[Any], optional): A closure that reevaluates the model and returns the loss. Defaults to None.

        Returns:
            Optional[Any]: The loss if closure is provided, else None.
        """
        pass
        self.defaults = {}

    def zero_grad(self, set_to_none: bool = False) -> None:
        """Sets the gradients of all optimized parameters to zero.

        Args:
            set_to_none (bool, optional): Instead of setting to zero, set the grads to None. Defaults to False.
        """
        import zero_torch

        for p in self.params:
            if hasattr(p, "grad") and p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    p.grad = zero_torch.zeros_like(p.grad)


class ASGD(Optimizer):
    """Implements Averaged Stochastic Gradient Descent."""

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
        """Initializes ASGD optimizer.

        Args:
            params (Any): Iterable of parameters to optimize.
            lr (Union[float, Tensor], optional): Learning rate. Defaults to 0.01.
            lambd (float, optional): Decay term. Defaults to 0.0001.
            alpha (float, optional): Power for eta update. Defaults to 0.75.
            t0 (float, optional): Point at which to start averaging. Defaults to 1000000.0.
            weight_decay (float, optional): Weight decay (L2 penalty). Defaults to 0.
            foreach (Optional[bool], optional): Whether foreach implementation of optimizer is used. Defaults to None.
            maximize (bool, optional): Maximize the params based on the objective. Defaults to False.
            differentiable (bool, optional): Whether autograd should track the optimizer step. Defaults to False.
            capturable (bool, optional): Whether this instance is safe to capture in a CUDA graph. Defaults to False.
        """
        pass
        self.defaults = {}


class Adadelta(Optimizer):
    """Implements Adadelta algorithm."""

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
        """Initializes Adadelta optimizer.

        Args:
            params (Any): Iterable of parameters to optimize.
            lr (Union[float, Tensor], optional): Coefficient that scales delta before it is applied. Defaults to 1.0.
            rho (float, optional): Interpolation parameter. Defaults to 0.9.
            eps (float, optional): Term added to the denominator to improve numerical stability. Defaults to 1e-06.
            weight_decay (float, optional): Weight decay. Defaults to 0.
            foreach (Optional[bool], optional): Whether foreach implementation is used. Defaults to None.
            capturable (bool, optional): Safe to capture in CUDA graph. Defaults to False.
            maximize (bool, optional): Maximize the params. Defaults to False.
            differentiable (bool, optional): Track optimizer step in autograd. Defaults to False.
        """
        pass
        self.defaults = {}


class Adafactor(Optimizer):
    """Implements Adafactor algorithm."""

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
        """Initializes Adafactor optimizer.

        Args:
            params (Any): Iterable of parameters to optimize.
            lr (Union[float, Tensor], optional): Learning rate. Defaults to 0.01.
            beta2_decay (float, optional): Beta2 decay term. Defaults to -0.8.
            eps (Tuple[Optional[float], float], optional): Epsilon values for numerical stability. Defaults to (None, 0.001).
            d (float, optional): Scaling factor. Defaults to 1.0.
            weight_decay (float, optional): Weight decay. Defaults to 0.0.
            foreach (Optional[bool], optional): Use foreach implementation. Defaults to None.
            maximize (bool, optional): Maximize the objective. Defaults to False.
        """
        pass
        self.defaults = {}


class Adagrad(Optimizer):
    """Implements Adagrad algorithm."""

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
        """Initializes Adagrad optimizer.

        Args:
            params (Any): Iterable of parameters.
            lr (Union[float, Tensor], optional): Learning rate. Defaults to 0.01.
            lr_decay (float, optional): Learning rate decay. Defaults to 0.
            weight_decay (float, optional): Weight decay. Defaults to 0.
            initial_accumulator_value (float, optional): Initial value for accumulator. Defaults to 0.
            eps (float, optional): Epsilon term. Defaults to 1e-10.
            foreach (Optional[bool], optional): Use foreach. Defaults to None.
            maximize (bool, optional): Maximize objective. Defaults to False.
            differentiable (bool, optional): Differentiable step. Defaults to False.
            fused (Optional[bool], optional): Fused implementation. Defaults to None.
        """
        pass
        self.defaults = {}


class Adam(Optimizer):
    """Implements Adam algorithm."""

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
        """Initializes Adam optimizer.

        Args:
            params (Any): Iterable of parameters.
            lr (Union[float, Tensor], optional): Learning rate. Defaults to 0.001.
            betas (Tuple[Union[float, Tensor], Union[float, Tensor]], optional): Coefficients used for computing running averages. Defaults to (0.9, 0.999).
            eps (float, optional): Term for numerical stability. Defaults to 1e-08.
            weight_decay (float, optional): Weight decay. Defaults to 0.
            amsgrad (bool, optional): Use AMSGrad variant. Defaults to False.
            foreach (Optional[bool], optional): Use foreach. Defaults to None.
            maximize (bool, optional): Maximize objective. Defaults to False.
            capturable (bool, optional): Safe to capture in CUDA graph. Defaults to False.
            differentiable (bool, optional): Differentiable step. Defaults to False.
            fused (Optional[bool], optional): Fused implementation. Defaults to None.
            decoupled_weight_decay (bool, optional): Decoupled weight decay. Defaults to False.
        """
        pass
        self.defaults = {}


class AdamW(Optimizer):
    """Implements AdamW algorithm."""

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
        """Initializes AdamW optimizer.

        Args:
            params (Any): Iterable of parameters.
            lr (Union[float, Tensor], optional): Learning rate. Defaults to 0.001.
            betas (Tuple[Union[float, Tensor], Union[float, Tensor]], optional): Coefficients for running averages. Defaults to (0.9, 0.999).
            eps (float, optional): Term for numerical stability. Defaults to 1e-08.
            weight_decay (float, optional): Weight decay coefficient. Defaults to 0.01.
            amsgrad (bool, optional): Use AMSGrad variant. Defaults to False.
            maximize (bool, optional): Maximize objective. Defaults to False.
            foreach (Optional[bool], optional): Use foreach. Defaults to None.
            capturable (bool, optional): Safe to capture in CUDA graph. Defaults to False.
            differentiable (bool, optional): Differentiable step. Defaults to False.
            fused (Optional[bool], optional): Fused implementation. Defaults to None.
        """
        pass
        self.defaults = {}


class Adamax(Optimizer):
    """Implements Adamax algorithm (a variant of Adam based on infinity norm)."""

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
        """Initializes Adamax optimizer.

        Args:
            params (Any): Iterable of parameters.
            lr (Union[float, Tensor], optional): Learning rate. Defaults to 0.002.
            betas (Tuple[float, float], optional): Coefficients used for computing running averages. Defaults to (0.9, 0.999).
            eps (float, optional): Term for numerical stability. Defaults to 1e-08.
            weight_decay (float, optional): Weight decay. Defaults to 0.
            foreach (Optional[bool], optional): Use foreach. Defaults to None.
            maximize (bool, optional): Maximize objective. Defaults to False.
            differentiable (bool, optional): Differentiable step. Defaults to False.
            capturable (bool, optional): Safe to capture in CUDA graph. Defaults to False.
        """
        pass
        self.defaults = {}


class LBFGS(Optimizer):
    """Implements L-BFGS algorithm."""

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
        """Initializes L-BFGS optimizer.

        Args:
            params (Iterable): Iterable of parameters to optimize.
            lr (Optional[float], optional): Learning rate. Defaults to 1.
            max_iter (Optional[int], optional): Maximal number of iterations per optimization step. Defaults to 20.
            max_eval (Optional[int], optional): Maximal number of function evaluations per step. Defaults to None.
            tolerance_grad (Optional[float], optional): Termination tolerance on first order optimality. Defaults to 1e-07.
            tolerance_change (Optional[float], optional): Termination tolerance on function value/parameter changes. Defaults to 1e-09.
            history_size (Optional[int], optional): Update history size. Defaults to 100.
            line_search_fn (Optional[str], optional): Either 'strong_wolfe' or None. Defaults to None.
        """
        pass
        self.defaults = {}


class NAdam(Optimizer):
    """Implements NAdam algorithm."""

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
        """Initializes NAdam optimizer.

        Args:
            params (Any): Iterable of parameters.
            lr (Union[float, Tensor], optional): Learning rate. Defaults to 0.002.
            betas (Tuple[float, float], optional): Coefficients used for computing running averages. Defaults to (0.9, 0.999).
            eps (float, optional): Epsilon term. Defaults to 1e-08.
            weight_decay (float, optional): Weight decay. Defaults to 0.
            momentum_decay (float, optional): Momentum decay. Defaults to 0.004.
            decoupled_weight_decay (bool, optional): Decoupled weight decay. Defaults to False.
            foreach (Optional[bool], optional): Use foreach. Defaults to None.
            maximize (bool, optional): Maximize objective. Defaults to False.
            capturable (bool, optional): Capturable in CUDA graphs. Defaults to False.
            differentiable (bool, optional): Differentiable step. Defaults to False.
        """
        pass
        self.defaults = {}


class RAdam(Optimizer):
    """Implements RAdam algorithm."""

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
        """Initializes RAdam optimizer.

        Args:
            params (Any): Iterable of parameters.
            lr (Union[float, Tensor], optional): Learning rate. Defaults to 0.001.
            betas (Tuple[float, float], optional): Coefficients used for computing running averages. Defaults to (0.9, 0.999).
            eps (float, optional): Epsilon term. Defaults to 1e-08.
            weight_decay (float, optional): Weight decay. Defaults to 0.
            decoupled_weight_decay (bool, optional): Decoupled weight decay. Defaults to False.
            foreach (Optional[bool], optional): Use foreach. Defaults to None.
            maximize (bool, optional): Maximize objective. Defaults to False.
            capturable (bool, optional): Capturable in CUDA graphs. Defaults to False.
            differentiable (bool, optional): Differentiable step. Defaults to False.
        """
        pass
        self.defaults = {}


class RMSprop(Optimizer):
    """Implements RMSprop algorithm."""

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
        """Initializes RMSprop optimizer.

        Args:
            params (Any): Iterable of parameters.
            lr (Union[float, Tensor], optional): Learning rate. Defaults to 0.01.
            alpha (float, optional): Smoothing constant. Defaults to 0.99.
            eps (float, optional): Epsilon term. Defaults to 1e-08.
            weight_decay (float, optional): Weight decay. Defaults to 0.
            momentum (float, optional): Momentum factor. Defaults to 0.
            centered (bool, optional): Compute centered RMSprop. Defaults to False.
            capturable (bool, optional): Capturable in CUDA graphs. Defaults to False.
            foreach (Optional[bool], optional): Use foreach. Defaults to None.
            maximize (bool, optional): Maximize objective. Defaults to False.
            differentiable (bool, optional): Differentiable step. Defaults to False.
        """
        pass
        self.defaults = {}


class Rprop(Optimizer):
    """Implements the resilient backpropagation algorithm."""

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
        """Initializes Rprop optimizer.

        Args:
            params (Any): Iterable of parameters.
            lr (Union[float, Tensor], optional): Learning rate. Defaults to 0.01.
            etas (Tuple[float, float], optional): Pair of (etaminus, etaplus). Defaults to (0.5, 1.2).
            step_sizes (Tuple[float, float], optional): Pair of minimal and maximal allowed step sizes. Defaults to (1e-06, 50).
            capturable (bool, optional): Capturable in CUDA graphs. Defaults to False.
            foreach (Optional[bool], optional): Use foreach. Defaults to None.
            maximize (bool, optional): Maximize objective. Defaults to False.
            differentiable (bool, optional): Differentiable step. Defaults to False.
        """
        pass
        self.defaults = {}


class SGD(Optimizer):
    """Implements stochastic gradient descent (optionally with momentum)."""

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
        """Initializes SGD optimizer.

        Args:
            params (Any): Iterable of parameters.
            lr (Union[float, Tensor], optional): Learning rate. Defaults to 0.001.
            momentum (float, optional): Momentum factor. Defaults to 0.
            dampening (float, optional): Dampening for momentum. Defaults to 0.
            weight_decay (Union[float, Tensor], optional): Weight decay (L2 penalty). Defaults to 0.
            nesterov (bool, optional): Enables Nesterov momentum. Defaults to False.
            maximize (bool, optional): Maximize objective. Defaults to False.
            foreach (Optional[bool], optional): Use foreach. Defaults to None.
            differentiable (bool, optional): Differentiable step. Defaults to False.
            fused (Optional[bool], optional): Fused implementation. Defaults to None.
        """
        super().__init__(params)
        self.lr = lr

    def step(self, closure=None) -> Optional[Any]:
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss. Defaults to None.

        Returns:
            Optional[Any]: The loss if closure is provided, otherwise None.
        """
        for p in self.params:
            if hasattr(p, "grad") and p.grad is not None:
                # eager update
                new_data = p - p.grad * self.lr
                p._tensor = new_data._tensor


class SparseAdam(Optimizer):
    """Implements lazy version of Adam algorithm suitable for sparse tensors."""

    def __init__(
        self,
        params: Any,
        lr: Union[float, Tensor] = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-08,
        maximize: bool = False,
    ) -> None:
        """Initializes SparseAdam optimizer.

        Args:
            params (Any): Iterable of parameters.
            lr (Union[float, Tensor], optional): Learning rate. Defaults to 0.001.
            betas (Tuple[float, float], optional): Coefficients used for computing running averages. Defaults to (0.9, 0.999).
            eps (float, optional): Term for numerical stability. Defaults to 1e-08.
            maximize (bool, optional): Maximize objective. Defaults to False.
        """
        pass
        self.defaults = {}
