"""Optimizers module."""

from __future__ import annotations

import math

import ml_switcheroo_compiler.ops.optimizers.updates as _opt
from ml_switcheroo_compiler.core.tensor import Tensor as S_Tensor

import zero_torch
from zero_torch._types import (
    LossClosure,
    ParamGroupDict,
    ParamGroupValue,
    ParamsArg,
)
from zero_torch.tensor import Tensor

__all__ = [
    "ASGD",
    "LBFGS",
    "SGD",
    "Adadelta",
    "Adafactor",
    "Adagrad",
    "Adam",
    "AdamW",
    "Adamax",
    "Muon",
    "NAdam",
    "Optimizer",
    "RAdam",
    "RMSprop",
    "Rprop",
    "SparseAdam",
    "lr_scheduler",
]


def _float_opt(val: ParamGroupValue, default: float) -> float:
    """Extracts a float hyperparameter from a parameter group value.

    Args:
        val: Raw parameter group value.
        default: Fallback default float.

    Returns:
        Extracted float value.
    """
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, Tensor) and val.numel() == 1:
        item_val = val.item()
        return float(item_val) if isinstance(item_val, (int, float)) else default
    return default


def _step_count(state_tensor: Tensor) -> float:
    """Extracts a float step counter from a scalar state tensor.

    Args:
        state_tensor: Scalar step tensor.

    Returns:
        Incremented step count as a float.
    """
    raw = state_tensor.item()
    return (float(raw) if isinstance(raw, (int, float)) else 0.0) + 1.0


def _betas_opt(
    val: ParamGroupValue, default: tuple[float, float] = (0.9, 0.999)
) -> tuple[float, float]:
    """Extracts a `(beta1, beta2)` float tuple from a parameter group value.

    Args:
        val: Raw parameter group value.
        default: Fallback default tuple.

    Returns:
        Tuple of `(beta1, beta2)`.
    """
    if (
        isinstance(val, tuple)
        and len(val) == 2
        and isinstance(val[0], (int, float))
        and isinstance(val[1], (int, float))
    ):
        return float(val[0]), float(val[1])
    return default


class Optimizer:
    """Base class for all optimizers."""

    def __init__(
        self,
        params: ParamsArg,
        defaults: dict[str, ParamGroupValue] | None = None,
    ) -> None:
        """Initializes the Optimizer.

        Args:
            params: Iterable of parameters to optimize or dicts defining parameter groups.
            defaults: Dictionary containing default values of optimization options.
        """
        self.defaults: dict[str, ParamGroupValue] = defaults or {}
        param_list = list(params)
        if len(param_list) > 0 and isinstance(param_list[0], dict):
            self.param_groups: list[ParamGroupDict] = [
                dict(g) for g in param_list if isinstance(g, dict)
            ]
        else:
            tensor_params: list[Tensor] = [
                p for p in param_list if isinstance(p, Tensor)
            ]
            self.param_groups = [{"params": tensor_params}]

        for group in self.param_groups:
            for k, v in self.defaults.items():
                if k not in group:
                    group[k] = v

        self.state: dict[int, dict[str, Tensor]] = {}

    def step(self, closure: LossClosure = None) -> Tensor | float | None:
        """Performs a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss.

        Returns:
            The evaluated loss if `closure` is provided, otherwise None.
        """
        loss: Tensor | float | None = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = _float_opt(group.get("lr"), 0.01)
            wd = _float_opt(group.get("weight_decay"), 0.0)
            maximize = bool(group.get("maximize", False))
            group_params = group.get("params")
            if not isinstance(group_params, list):
                continue
            for p in group_params:
                if not isinstance(p, Tensor) or p.grad is None:
                    continue
                d_p = (-p.grad if maximize else p.grad) + p * wd
                updated = p - d_p * lr
                p._tensor = updated._tensor

        return loss

    def zero_grad(self, set_to_none: bool = False) -> None:
        """Sets the gradients of all optimized parameters to zero or None.

        Args:
            set_to_none: Instead of setting to zero, set the grads to None.
        """
        for group in self.param_groups:
            group_params = group.get("params")
            if isinstance(group_params, list):
                for p in group_params:
                    if isinstance(p, Tensor) and p.grad is not None:
                        if set_to_none:
                            p.grad = None
                        else:
                            p.grad = zero_torch.zeros_like(p.grad)


def _adam_family_step(
    opt: Optimizer,
    closure: LossClosure = None,
    decoupled_wd: bool = False,
) -> Tensor | float | None:
    """Executes an Adam/AdamW/NAdam/RAdam/SparseAdam moment update step.

    Args:
        opt: Optimizer instance.
        closure: Optional loss evaluation closure.
        decoupled_wd: Whether to apply decoupled AdamW-style weight decay.

    Returns:
        Evaluated loss if `closure` is provided, else None.
    """
    loss: Tensor | float | None = None
    if closure is not None:
        loss = closure()

    for group in opt.param_groups:
        lr = _float_opt(group.get("lr"), 0.001)
        beta1, beta2 = _betas_opt(group.get("betas"), (0.9, 0.999))
        eps = _float_opt(group.get("eps"), 1e-8)
        wd = _float_opt(group.get("weight_decay"), 0.0)
        amsgrad = bool(group.get("amsgrad", False))
        maximize = bool(group.get("maximize", False))

        group_params = group.get("params")
        if not isinstance(group_params, list):
            continue

        for p in group_params:
            if not isinstance(p, Tensor) or p.grad is None:
                continue
            grad = -p.grad if maximize else p.grad
            if wd != 0.0:
                if decoupled_wd:
                    p._tensor = (p * (1.0 - lr * wd))._tensor
                else:
                    grad = grad + p * wd

            p_state = opt.state.setdefault(id(p), {})
            if "step" not in p_state:
                p_state["step"] = zero_torch.tensor(0.0)
                p_state["exp_avg"] = zero_torch.zeros_like(p)
                p_state["exp_avg_sq"] = zero_torch.zeros_like(p)
                if amsgrad:
                    p_state["max_exp_avg_sq"] = zero_torch.zeros_like(p)

            step_val = _step_count(p_state["step"])
            p_state["step"] = zero_torch.tensor(step_val)
            exp_avg = p_state["exp_avg"] * beta1 + grad * (1.0 - beta1)
            exp_avg_sq = p_state["exp_avg_sq"] * beta2 + (grad * grad) * (1.0 - beta2)
            p_state["exp_avg"] = exp_avg
            p_state["exp_avg_sq"] = exp_avg_sq

            bias_correction1 = 1.0 - (beta1**step_val)
            bias_correction2 = 1.0 - (beta2**step_val)

            if amsgrad and "max_exp_avg_sq" in p_state:
                max_sq = zero_torch.maximum(p_state["max_exp_avg_sq"], exp_avg_sq)
                p_state["max_exp_avg_sq"] = max_sq
                denom = (zero_torch.sqrt(max_sq) / math.sqrt(bias_correction2)) + eps
            else:
                denom = (
                    zero_torch.sqrt(exp_avg_sq) / math.sqrt(bias_correction2)
                ) + eps

            step_size = lr / bias_correction1
            updated = p - (exp_avg / denom) * step_size
            p._tensor = updated._tensor

    return loss


class ASGD(Optimizer):
    """Implements Averaged Stochastic Gradient Descent."""

    def __init__(
        self,
        params: ParamsArg,
        lr: float | Tensor = 0.01,
        lambd: float = 0.0001,
        alpha: float = 0.75,
        t0: float = 1000000.0,
        weight_decay: float = 0.0,
        foreach: bool | None = None,
        maximize: bool = False,
        differentiable: bool = False,
        capturable: bool = False,
    ) -> None:
        """Initializes ASGD optimizer.

        Args:
            params: Iterable of parameters to optimize.
            lr: Learning rate.
            lambd: Decay term.
            alpha: Power for eta update.
            t0: Point at which to start averaging.
            weight_decay: Weight decay (L2 penalty).
            foreach: Whether foreach implementation of optimizer is used.
            maximize: Maximize the params based on the objective.
            differentiable: Whether autograd should track the optimizer step.
            capturable: Whether this instance is safe to capture in a CUDA graph.
        """
        _ = (foreach, differentiable, capturable)
        super().__init__(
            params,
            {
                "lr": lr,
                "lambd": lambd,
                "alpha": alpha,
                "t0": t0,
                "weight_decay": weight_decay,
                "maximize": maximize,
            },
        )


class Adadelta(Optimizer):
    """Implements Adadelta algorithm."""

    def __init__(
        self,
        params: ParamsArg,
        lr: float | Tensor = 1.0,
        rho: float = 0.9,
        eps: float = 1e-06,
        weight_decay: float = 0.0,
        foreach: bool | None = None,
        capturable: bool = False,
        maximize: bool = False,
        differentiable: bool = False,
    ) -> None:
        """Initializes Adadelta optimizer.

        Args:
            params: Iterable of parameters to optimize.
            lr: Coefficient that scales delta before it is applied.
            rho: Coefficient used for computing a running average of squared gradients.
            eps: Term added to the denominator to improve numerical stability.
            weight_decay: Weight decay (L2 penalty).
            foreach: Whether foreach implementation is used.
            capturable: Whether safe to capture in a CUDA graph.
            maximize: Maximize the params based on the objective.
            differentiable: Whether autograd should track the optimizer step.
        """
        _ = (foreach, capturable, differentiable)
        super().__init__(
            params,
            {
                "lr": lr,
                "rho": rho,
                "eps": eps,
                "weight_decay": weight_decay,
                "maximize": maximize,
            },
        )

    def step(self, closure: LossClosure = None) -> Tensor | float | None:
        """Performs a single Adadelta optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss.

        Returns:
            Evaluated loss if `closure` is provided, else None.
        """
        loss: Tensor | float | None = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            lr = _float_opt(group.get("lr"), 1.0)
            rho = _float_opt(group.get("rho"), 0.9)
            eps = _float_opt(group.get("eps"), 1e-6)
            wd = _float_opt(group.get("weight_decay"), 0.0)
            maximize = bool(group.get("maximize", False))
            group_params = group.get("params")
            if not isinstance(group_params, list):
                continue
            for p in group_params:
                if not isinstance(p, Tensor) or p.grad is None:
                    continue
                grad = (-p.grad if maximize else p.grad) + p * wd
                p_state = self.state.setdefault(id(p), {})
                if "square_avg" not in p_state:
                    p_state["square_avg"] = zero_torch.zeros_like(p)
                    p_state["acc_delta"] = zero_torch.zeros_like(p)
                sq_avg = p_state["square_avg"] * rho + (grad * grad) * (1.0 - rho)
                p_state["square_avg"] = sq_avg
                std = zero_torch.sqrt(sq_avg + eps)
                delta = (zero_torch.sqrt(p_state["acc_delta"] + eps) / std) * grad
                p_state["acc_delta"] = p_state["acc_delta"] * rho + (delta * delta) * (
                    1.0 - rho
                )
                p._tensor = (p - delta * lr)._tensor
        return loss


class Adafactor(Optimizer):
    """Implements Adafactor algorithm."""

    def __init__(
        self,
        params: ParamsArg,
        lr: float | Tensor = 0.01,
        beta2_decay: float = -0.8,
        eps: tuple[float | None, float] = (None, 0.001),
        d: float = 1.0,
        weight_decay: float = 0.0,
        foreach: bool | None = None,
        maximize: bool = False,
    ) -> None:
        """Initializes Adafactor optimizer.

        Args:
            params: Iterable of parameters to optimize.
            lr: Learning rate.
            beta2_decay: Beta2 decay rate.
            eps: Regularization constants for square gradient and parameter scale.
            d: Clipping threshold.
            weight_decay: Weight decay coefficient.
            foreach: Whether foreach implementation is used.
            maximize: Maximize the objective.
        """
        _ = (eps, foreach)
        super().__init__(
            params,
            {
                "lr": lr,
                "beta2_decay": beta2_decay,
                "d": d,
                "weight_decay": weight_decay,
                "maximize": maximize,
            },
        )


class Adagrad(Optimizer):
    """Implements Adagrad algorithm."""

    def __init__(
        self,
        params: ParamsArg,
        lr: float | Tensor = 0.01,
        lr_decay: float = 0.0,
        weight_decay: float = 0.0,
        initial_accumulator_value: float = 0.0,
        eps: float = 1e-10,
        foreach: bool | None = None,
        maximize: bool = False,
        differentiable: bool = False,
        fused: bool | None = None,
    ) -> None:
        """Initializes Adagrad optimizer.

        Args:
            params: Iterable of parameters to optimize.
            lr: Learning rate.
            lr_decay: Learning rate decay.
            weight_decay: Weight decay (L2 penalty).
            initial_accumulator_value: Initial value of the sum of squares of gradients.
            eps: Term added to the denominator to improve numerical stability.
            foreach: Whether foreach implementation is used.
            maximize: Maximize the objective.
            differentiable: Whether autograd should track the optimizer step.
            fused: Whether fused implementation is used.
        """
        _ = (foreach, differentiable, fused)
        super().__init__(
            params,
            {
                "lr": lr,
                "lr_decay": lr_decay,
                "weight_decay": weight_decay,
                "initial_accumulator_value": initial_accumulator_value,
                "eps": eps,
                "maximize": maximize,
            },
        )

    def step(self, closure: LossClosure = None) -> Tensor | float | None:
        """Performs a single Adagrad optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss.

        Returns:
            Evaluated loss if `closure` is provided, else None.
        """
        loss: Tensor | float | None = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            lr = _float_opt(group.get("lr"), 0.01)
            lr_decay = _float_opt(group.get("lr_decay"), 0.0)
            wd = _float_opt(group.get("weight_decay"), 0.0)
            init_acc = _float_opt(group.get("initial_accumulator_value"), 0.0)
            eps = _float_opt(group.get("eps"), 1e-10)
            maximize = bool(group.get("maximize", False))
            group_params = group.get("params")
            if not isinstance(group_params, list):
                continue
            for p in group_params:
                if not isinstance(p, Tensor) or p.grad is None:
                    continue
                grad = (-p.grad if maximize else p.grad) + p * wd
                p_state = self.state.setdefault(id(p), {})
                if "step" not in p_state:
                    p_state["step"] = zero_torch.tensor(0.0)
                    p_state["sum"] = zero_torch.full_like(p, init_acc)
                step_val = _step_count(p_state["step"])
                p_state["step"] = zero_torch.tensor(step_val)
                clr = lr / (1.0 + (step_val - 1.0) * lr_decay)
                p_state["sum"] = p_state["sum"] + grad * grad
                std = zero_torch.sqrt(p_state["sum"]) + eps
                p._tensor = (p - (grad / std) * clr)._tensor
        return loss


class Adam(Optimizer):
    """Implements Adam algorithm."""

    def __init__(
        self,
        params: ParamsArg,
        lr: float | Tensor = 0.001,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-08,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
        foreach: bool | None = None,
        maximize: bool = False,
        capturable: bool = False,
        differentiable: bool = False,
        fused: bool | None = None,
        decoupled_weight_decay: bool = False,
    ) -> None:
        """Initializes Adam optimizer.

        Args:
            params: Iterable of parameters to optimize.
            lr: Learning rate.
            betas: Coefficients used for computing running averages of gradient and its square.
            eps: Term added to the denominator to improve numerical stability.
            weight_decay: Weight decay (L2 penalty).
            amsgrad: Whether to use the AMSGrad variant.
            foreach: Whether foreach implementation is used.
            maximize: Maximize the objective.
            capturable: Whether safe to capture in a CUDA graph.
            differentiable: Whether autograd should track the optimizer step.
            fused: Whether fused implementation is used.
            decoupled_weight_decay: Whether to apply decoupled weight decay.
        """
        _ = (foreach, capturable, differentiable, fused)
        self.decoupled_weight_decay: bool = decoupled_weight_decay
        super().__init__(
            params,
            {
                "lr": lr,
                "betas": betas,
                "eps": eps,
                "weight_decay": weight_decay,
                "amsgrad": amsgrad,
                "maximize": maximize,
            },
        )

    def step(self, closure: LossClosure = None) -> Tensor | float | None:
        """Performs a single Adam optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss.

        Returns:
            Evaluated loss if `closure` is provided, else None.
        """
        return _adam_family_step(
            self, closure, decoupled_wd=self.decoupled_weight_decay
        )


class AdamW(Optimizer):
    """Implements AdamW algorithm."""

    def __init__(
        self,
        params: ParamsArg,
        lr: float | Tensor = 0.001,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-08,
        weight_decay: float = 0.01,
        amsgrad: bool = False,
        maximize: bool = False,
        foreach: bool | None = None,
        capturable: bool = False,
        differentiable: bool = False,
        fused: bool | None = None,
    ) -> None:
        """Initializes AdamW optimizer.

        Args:
            params: Iterable of parameters to optimize.
            lr: Learning rate.
            betas: Coefficients used for computing running averages of gradient and its square.
            eps: Term added to the denominator to improve numerical stability.
            weight_decay: Weight decay coefficient.
            amsgrad: Whether to use the AMSGrad variant.
            maximize: Maximize the objective.
            foreach: Whether foreach implementation is used.
            capturable: Whether safe to capture in a CUDA graph.
            differentiable: Whether autograd should track the optimizer step.
            fused: Whether fused implementation is used.
        """
        _ = (foreach, capturable, differentiable, fused)
        super().__init__(
            params,
            {
                "lr": lr,
                "betas": betas,
                "eps": eps,
                "weight_decay": weight_decay,
                "amsgrad": amsgrad,
                "maximize": maximize,
            },
        )

    def step(self, closure: LossClosure = None) -> Tensor | float | None:
        """Performs a single AdamW optimization step with decoupled weight decay.

        Args:
            closure: A closure that reevaluates the model and returns the loss.

        Returns:
            Evaluated loss if `closure` is provided, else None.
        """
        return _adam_family_step(self, closure, decoupled_wd=True)


class Adamax(Optimizer):
    """Implements Adamax algorithm (a variant of Adam based on infinity norm)."""

    def __init__(
        self,
        params: ParamsArg,
        lr: float | Tensor = 0.002,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-08,
        weight_decay: float = 0.0,
        foreach: bool | None = None,
        maximize: bool = False,
        differentiable: bool = False,
        capturable: bool = False,
    ) -> None:
        """Initializes Adamax optimizer.

        Args:
            params: Iterable of parameters to optimize.
            lr: Learning rate.
            betas: Coefficients used for computing running averages.
            eps: Term added to the denominator to improve numerical stability.
            weight_decay: Weight decay (L2 penalty).
            foreach: Whether foreach implementation is used.
            maximize: Maximize the objective.
            differentiable: Whether autograd should track the optimizer step.
            capturable: Whether safe to capture in a CUDA graph.
        """
        _ = (foreach, differentiable, capturable)
        super().__init__(
            params,
            {
                "lr": lr,
                "betas": betas,
                "eps": eps,
                "weight_decay": weight_decay,
                "maximize": maximize,
            },
        )

    def step(self, closure: LossClosure = None) -> Tensor | float | None:
        """Performs a single Adamax optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss.

        Returns:
            Evaluated loss if `closure` is provided, else None.
        """
        loss: Tensor | float | None = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            lr = _float_opt(group.get("lr"), 0.002)
            beta1, beta2 = _betas_opt(group.get("betas"), (0.9, 0.999))
            eps = _float_opt(group.get("eps"), 1e-8)
            wd = _float_opt(group.get("weight_decay"), 0.0)
            maximize = bool(group.get("maximize", False))
            group_params = group.get("params")
            if not isinstance(group_params, list):
                continue
            for p in group_params:
                if not isinstance(p, Tensor) or p.grad is None:
                    continue
                grad = (-p.grad if maximize else p.grad) + p * wd
                p_state = self.state.setdefault(id(p), {})
                if "step" not in p_state:
                    p_state["step"] = zero_torch.tensor(0.0)
                    p_state["exp_avg"] = zero_torch.zeros_like(p)
                    p_state["exp_inf"] = zero_torch.zeros_like(p)
                step_val = _step_count(p_state["step"])
                p_state["step"] = zero_torch.tensor(step_val)
                p_state["exp_avg"] = p_state["exp_avg"] * beta1 + grad * (1.0 - beta1)
                p_state["exp_inf"] = zero_torch.maximum(
                    p_state["exp_inf"] * beta2, zero_torch.abs(grad) + eps
                )
                clr = lr / (1.0 - (beta1**step_val))
                p._tensor = (
                    p - (p_state["exp_avg"] / p_state["exp_inf"]) * clr
                )._tensor
        return loss


class LBFGS(Optimizer):
    """Implements L-BFGS algorithm."""

    def __init__(
        self,
        params: ParamsArg,
        lr: float | None = 1.0,
        max_iter: int | None = 20,
        max_eval: int | None = None,
        tolerance_grad: float | None = 1e-07,
        tolerance_change: float | None = 1e-09,
        history_size: int | None = 100,
        line_search_fn: str | None = None,
    ) -> None:
        """Initializes L-BFGS optimizer.

        Args:
            params: Iterable of parameters to optimize.
            lr: Learning rate.
            max_iter: Maximal number of iterations per optimization step.
            max_eval: Maximal number of function evaluations per optimization step.
            tolerance_grad: Termination tolerance on first order optimality.
            tolerance_change: Termination tolerance on function value/parameter changes.
            history_size: Update history size.
            line_search_fn: Either `'strong_wolfe'` or None.
        """
        super().__init__(
            params,
            {
                "lr": lr,
                "max_iter": max_iter,
                "max_eval": max_eval,
                "tolerance_grad": tolerance_grad,
                "tolerance_change": tolerance_change,
                "history_size": history_size,
                "line_search_fn": line_search_fn,
            },
        )


class NAdam(Optimizer):
    """Implements NAdam algorithm."""

    def __init__(
        self,
        params: ParamsArg,
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
    ) -> None:
        """Initializes NAdam optimizer.

        Args:
            params: Iterable of parameters to optimize.
            lr: Learning rate.
            betas: Coefficients used for computing running averages.
            eps: Term added to the denominator to improve numerical stability.
            weight_decay: Weight decay (L2 penalty).
            momentum_decay: Momentum momentum_decay coefficient.
            decoupled_weight_decay: Whether to apply decoupled weight decay.
            foreach: Whether foreach implementation is used.
            maximize: Maximize the objective.
            capturable: Whether safe to capture in a CUDA graph.
            differentiable: Whether autograd should track the optimizer step.
        """
        _ = (foreach, capturable, differentiable)
        self.decoupled_weight_decay: bool = decoupled_weight_decay
        super().__init__(
            params,
            {
                "lr": lr,
                "betas": betas,
                "eps": eps,
                "weight_decay": weight_decay,
                "momentum_decay": momentum_decay,
                "maximize": maximize,
            },
        )

    def step(self, closure: LossClosure = None) -> Tensor | float | None:
        """Performs a single NAdam optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss.

        Returns:
            Evaluated loss if `closure` is provided, else None.
        """
        return _adam_family_step(
            self, closure, decoupled_wd=self.decoupled_weight_decay
        )


class RAdam(Optimizer):
    """Implements RAdam algorithm."""

    def __init__(
        self,
        params: ParamsArg,
        lr: float | Tensor = 0.001,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-08,
        weight_decay: float = 0.0,
        decoupled_weight_decay: bool = False,
        foreach: bool | None = None,
        maximize: bool = False,
        capturable: bool = False,
        differentiable: bool = False,
    ) -> None:
        """Initializes RAdam optimizer.

        Args:
            params: Iterable of parameters to optimize.
            lr: Learning rate.
            betas: Coefficients used for computing running averages.
            eps: Term added to the denominator to improve numerical stability.
            weight_decay: Weight decay (L2 penalty).
            decoupled_weight_decay: Whether to apply decoupled weight decay.
            foreach: Whether foreach implementation is used.
            maximize: Maximize the objective.
            capturable: Whether safe to capture in a CUDA graph.
            differentiable: Whether autograd should track the optimizer step.
        """
        _ = (foreach, capturable, differentiable)
        self.decoupled_weight_decay: bool = decoupled_weight_decay
        super().__init__(
            params,
            {
                "lr": lr,
                "betas": betas,
                "eps": eps,
                "weight_decay": weight_decay,
                "maximize": maximize,
            },
        )

    def step(self, closure: LossClosure = None) -> Tensor | float | None:
        """Performs a single RAdam optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss.

        Returns:
            Evaluated loss if `closure` is provided, else None.
        """
        return _adam_family_step(
            self, closure, decoupled_wd=self.decoupled_weight_decay
        )


class RMSprop(Optimizer):
    """Implements RMSprop algorithm."""

    def __init__(
        self,
        params: ParamsArg,
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
    ) -> None:
        """Initializes RMSprop optimizer.

        Args:
            params: Iterable of parameters to optimize.
            lr: Learning rate.
            alpha: Smoothing constant.
            eps: Term added to the denominator to improve numerical stability.
            weight_decay: Weight decay (L2 penalty).
            momentum: Momentum factor.
            centered: If True, compute the centered RMSProp.
            capturable: Whether safe to capture in a CUDA graph.
            foreach: Whether foreach implementation is used.
            maximize: Maximize the objective.
            differentiable: Whether autograd should track the optimizer step.
        """
        _ = (capturable, foreach, differentiable)
        super().__init__(
            params,
            {
                "lr": lr,
                "alpha": alpha,
                "eps": eps,
                "weight_decay": weight_decay,
                "momentum": momentum,
                "centered": centered,
                "maximize": maximize,
            },
        )

    def step(self, closure: LossClosure = None) -> Tensor | float | None:
        """Performs a single RMSprop optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss.

        Returns:
            Evaluated loss if `closure` is provided, else None.
        """
        loss: Tensor | float | None = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            lr = _float_opt(group.get("lr"), 0.01)
            alpha = _float_opt(group.get("alpha"), 0.99)
            eps = _float_opt(group.get("eps"), 1e-8)
            wd = _float_opt(group.get("weight_decay"), 0.0)
            mom = _float_opt(group.get("momentum"), 0.0)
            centered = bool(group.get("centered", False))
            maximize = bool(group.get("maximize", False))
            group_params = group.get("params")
            if not isinstance(group_params, list):
                continue
            for p in group_params:
                if not isinstance(p, Tensor) or p.grad is None:
                    continue
                grad = (-p.grad if maximize else p.grad) + p * wd
                p_state = self.state.setdefault(id(p), {})
                if "square_avg" not in p_state:
                    p_state["square_avg"] = zero_torch.zeros_like(p)
                    if mom > 0.0:
                        p_state["momentum_buffer"] = zero_torch.zeros_like(p)
                    if centered:
                        p_state["grad_avg"] = zero_torch.zeros_like(p)
                sq_avg = p_state["square_avg"] * alpha + (grad * grad) * (1.0 - alpha)
                p_state["square_avg"] = sq_avg
                if centered and "grad_avg" in p_state:
                    g_avg = p_state["grad_avg"] * alpha + grad * (1.0 - alpha)
                    p_state["grad_avg"] = g_avg
                    avg = (
                        zero_torch.sqrt(zero_torch.maximum(sq_avg - g_avg * g_avg, 0.0))
                        + eps
                    )
                else:
                    avg = zero_torch.sqrt(sq_avg) + eps
                if mom > 0.0 and "momentum_buffer" in p_state:
                    buf = p_state["momentum_buffer"] * mom + (grad / avg)
                    p_state["momentum_buffer"] = buf
                    p._tensor = (p - buf * lr)._tensor
                else:
                    p._tensor = (p - (grad / avg) * lr)._tensor
        return loss


class Rprop(Optimizer):
    """Implements the resilient backpropagation algorithm."""

    def __init__(
        self,
        params: ParamsArg,
        lr: float | Tensor = 0.01,
        etas: tuple[float, float] = (0.5, 1.2),
        step_sizes: tuple[float, float] = (1e-06, 50.0),
        capturable: bool = False,
        foreach: bool | None = None,
        maximize: bool = False,
        differentiable: bool = False,
    ) -> None:
        """Initializes Rprop optimizer.

        Args:
            params: Iterable of parameters to optimize.
            lr: Learning rate.
            etas: Pair of `(etaminus, etaplus)` multiplicative factors.
            step_sizes: Pair of minimal and maximal allowed step sizes.
            capturable: Whether safe to capture in a CUDA graph.
            foreach: Whether foreach implementation is used.
            maximize: Maximize the objective.
            differentiable: Whether autograd should track the optimizer step.
        """
        _ = (capturable, foreach, differentiable)
        super().__init__(
            params,
            {
                "lr": lr,
                "etas": etas,
                "step_sizes": step_sizes,
                "maximize": maximize,
            },
        )


class SGD(Optimizer):
    """Implements stochastic gradient descent (optionally with momentum)."""

    def __init__(
        self,
        params: ParamsArg,
        lr: float | Tensor = 0.001,
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float | Tensor = 0.0,
        nesterov: bool = False,
        maximize: bool = False,
        foreach: bool | None = None,
        differentiable: bool = False,
        fused: bool | None = None,
    ) -> None:
        """Initializes SGD optimizer.

        Args:
            params: Iterable of parameters to optimize.
            lr: Learning rate.
            momentum: Momentum factor.
            dampening: Dampening for momentum.
            weight_decay: Weight decay (L2 penalty).
            nesterov: Enables Nesterov momentum.
            maximize: Maximize the objective.
            foreach: Whether foreach implementation is used.
            differentiable: Whether autograd should track the optimizer step.
            fused: Whether fused implementation is used.
        """
        _ = (foreach, differentiable, fused)
        defaults: dict[str, ParamGroupValue] = {
            "lr": lr,
            "momentum": momentum,
            "dampening": dampening,
            "weight_decay": weight_decay,
            "nesterov": nesterov,
            "maximize": maximize,
        }
        super().__init__(params, defaults)

    def step(self, closure: LossClosure = None) -> Tensor | float | None:
        """Performs a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss.

        Returns:
            The evaluated loss if `closure` is provided, otherwise None.
        """
        loss: Tensor | float | None = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            wd_raw = group.get("weight_decay", 0.0)
            mom_raw = group.get("momentum", 0.0)
            damp_raw = group.get("dampening", 0.0)
            nest_raw = group.get("nesterov", False)
            max_raw = group.get("maximize", False)
            lr_raw = group.get("lr", 0.001)

            cfg = _opt.SGDConfig(
                lr=float(lr_raw) if isinstance(lr_raw, (int, float)) else 0.001,
                momentum=float(mom_raw) if isinstance(mom_raw, (int, float)) else 0.0,
                dampening=(
                    float(damp_raw) if isinstance(damp_raw, (int, float)) else 0.0
                ),
                weight_decay=(
                    float(wd_raw) if isinstance(wd_raw, (int, float)) else 0.0
                ),
                nesterov=bool(nest_raw),
            )

            group_params = group.get("params")
            if not isinstance(group_params, list):
                continue

            for p in group_params:
                if not isinstance(p, Tensor) or p.grad is None:
                    continue
                d_p = -p.grad if bool(max_raw) else p.grad

                param_state = self.state.setdefault(id(p), {})
                s_state: dict[str, S_Tensor] = {}
                if "momentum_buffer" in param_state:
                    mb_t = param_state["momentum_buffer"]._tensor
                    assert isinstance(mb_t, S_Tensor)
                    s_state["momentum_buffer"] = mb_t

                p_t, grad_t = p._tensor, d_p._tensor
                assert isinstance(p_t, S_Tensor)
                assert isinstance(grad_t, S_Tensor)

                new_p, new_s_state = _opt.sgd_update(p_t, grad_t, cfg, state=s_state)

                p._tensor = new_p
                if "momentum_buffer" in new_s_state:
                    param_state["momentum_buffer"] = Tensor(
                        new_s_state["momentum_buffer"]
                    )

        return loss


class SparseAdam(Optimizer):
    """Implements lazy version of Adam algorithm suitable for sparse tensors."""

    def __init__(
        self,
        params: ParamsArg,
        lr: float | Tensor = 0.001,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-08,
        maximize: bool = False,
    ) -> None:
        """Initializes SparseAdam optimizer.

        Args:
            params: Iterable of parameters to optimize.
            lr: Learning rate.
            betas: Coefficients used for computing running averages.
            eps: Term added to the denominator to improve numerical stability.
            maximize: Maximize the objective.
        """
        super().__init__(
            params,
            {"lr": lr, "betas": betas, "eps": eps, "maximize": maximize},
        )

    def step(self, closure: LossClosure = None) -> Tensor | float | None:
        """Performs a single SparseAdam optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss.

        Returns:
            Evaluated loss if `closure` is provided, else None.
        """
        return _adam_family_step(self, closure, decoupled_wd=False)


class Muon(Optimizer):
    """Implements Muon optimizer (MomentUm Orthogonalized by Newton-schulz)."""

    def __init__(
        self,
        params: ParamsArg,
        lr: float = 0.001,
        weight_decay: float = 0.1,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_coefficients: tuple[float, float, float] = (3.4445, -4.775, 2.0315),
        eps: float = 1e-07,
        ns_steps: int = 5,
        adjust_lr_fn: str | None = None,
    ) -> None:
        """Initializes Muon optimizer.

        Args:
            params: Iterable of parameters to optimize or dicts defining parameter groups.
            lr: Learning rate.
            weight_decay: Weight decay coefficient.
            momentum: Momentum factor.
            nesterov: Whether to enable Nesterov momentum.
            ns_coefficients: Coefficients for Newton-Schulz iteration.
            eps: Term added to the denominator to improve numerical stability.
            ns_steps: Number of Newton-Schulz iteration steps.
            adjust_lr_fn: Optional learning rate adjustment strategy name.
        """
        defaults: dict[str, ParamGroupValue] = {
            "lr": lr,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "nesterov": nesterov,
            "ns_coefficients": ns_coefficients,
            "eps": eps,
            "ns_steps": ns_steps,
            "adjust_lr_fn": adjust_lr_fn,
        }
        super().__init__(params, defaults)


from . import lr_scheduler as lr_scheduler
