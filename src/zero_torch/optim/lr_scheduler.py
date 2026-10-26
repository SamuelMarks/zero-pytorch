"""Learning rate schedulers for zero_torch.optim."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

from zero_torch.optim import Optimizer


class LRScheduler:
    """Base class for learning rate schedulers."""

    def __init__(self, optimizer: Optimizer, last_epoch: int = -1) -> None:
        """Initializes LRScheduler.

        Args:
            optimizer: Wrapped optimizer instance.
            last_epoch: The index of the last epoch.
        """
        self.optimizer: Optimizer = optimizer
        self.last_epoch: int = last_epoch
        self.base_lrs: list[float] = []
        for group in optimizer.param_groups:
            lr_val = group.get("lr", 0.001)
            self.base_lrs.append(
                float(lr_val) if isinstance(lr_val, (int, float)) else 0.001
            )
        self._last_lr: list[float] = list(self.base_lrs)
        self.step()

    def get_lr(self) -> list[float]:
        """Computes learning rate for the current step.

        Returns:
            List of learning rates for each parameter group.
        """
        return list(self.base_lrs)

    def get_last_lr(self) -> list[float]:
        """Returns last computed learning rate by current scheduler.

        Returns:
            List of last computed learning rates.
        """
        return list(self._last_lr)

    def step(self, epoch: int | None = None) -> None:
        """Advances the scheduler by one epoch or to a specified epoch.

        Args:
            epoch: Optional epoch index to set.
        """
        if epoch is None:
            self.last_epoch += 1
        else:
            self.last_epoch = epoch
        values = self.get_lr()
        for group, lr in zip(self.optimizer.param_groups, values):
            group["lr"] = lr
        self._last_lr = list(values)

    def state_dict(self) -> dict[str, int | list[float]]:
        """Returns the state of the scheduler as a dictionary.

        Returns:
            Dictionary containing scheduler state.
        """
        return {
            "last_epoch": self.last_epoch,
            "base_lrs": list(self.base_lrs),
            "_last_lr": list(self._last_lr),
        }

    def load_state_dict(self, state_dict: dict[str, int | list[float]]) -> None:
        """Loads the schedulers state and synchronizes `optimizer.param_groups` learning rates.

        Args:
            state_dict: Scheduler state dictionary returned from state_dict().
        """
        last_ep = state_dict.get("last_epoch", -1)
        if isinstance(last_ep, int):
            self.last_epoch = last_ep
        base = state_dict.get("base_lrs")
        if isinstance(base, list):
            self.base_lrs = [float(x) for x in base]
        last = state_dict.get("_last_lr")
        if isinstance(last, list):
            self._last_lr = [float(x) for x in last]
            for group, lr in zip(self.optimizer.param_groups, self._last_lr):
                group["lr"] = lr


_LRScheduler = LRScheduler


class LambdaLR(LRScheduler):
    """Sets the learning rate of each parameter group to the initial lr times a given function."""

    def __init__(
        self,
        optimizer: Optimizer,
        lr_lambda: Callable[[int], float] | Sequence[Callable[[int], float]],
        last_epoch: int = -1,
    ) -> None:
        """Initializes LambdaLR.

        Args:
            optimizer: Wrapped optimizer.
            lr_lambda: Function or list of functions computing a multiplicative factor given epoch.
            last_epoch: The index of last epoch.
        """
        if callable(lr_lambda):
            self.lr_lambdas: list[Callable[[int], float]] = [
                lr_lambda for _ in optimizer.param_groups
            ]
        else:
            self.lr_lambdas = list(lr_lambda)
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        """Computes the learning rates using the lambda functions.

        Returns:
            List of computed learning rates.
        """
        return [
            base_lr * float(lmbda(self.last_epoch))
            for lmbda, base_lr in zip(self.lr_lambdas, self.base_lrs)
        ]


class MultiplicativeLR(LRScheduler):
    """Multiply the learning rate of each parameter group by the factor given in the specified function."""

    def __init__(
        self,
        optimizer: Optimizer,
        lr_lambda: Callable[[int], float] | Sequence[Callable[[int], float]],
        last_epoch: int = -1,
    ) -> None:
        """Initializes MultiplicativeLR.

        Args:
            optimizer: Wrapped optimizer.
            lr_lambda: Function or list of functions computing a multiplicative factor given epoch.
            last_epoch: The index of last epoch.
        """
        if callable(lr_lambda):
            self.lr_lambdas: list[Callable[[int], float]] = [
                lr_lambda for _ in optimizer.param_groups
            ]
        else:
            self.lr_lambdas = list(lr_lambda)
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        """Computes the learning rates multiplicatively.

        Returns:
            List of computed learning rates.
        """
        if self.last_epoch > 0:
            out: list[float] = []
            for lmbda, group, base_lr in zip(
                self.lr_lambdas, self.optimizer.param_groups, self.base_lrs
            ):
                raw_lr = group.get("lr", base_lr)
                cur_lr = float(raw_lr) if isinstance(raw_lr, (int, float)) else base_lr
                out.append(cur_lr * float(lmbda(self.last_epoch)))
            return out
        return list(self.base_lrs)


class StepLR(LRScheduler):
    """Decays the learning rate of each parameter group by gamma every step_size epochs."""

    def __init__(
        self,
        optimizer: Optimizer,
        step_size: int,
        gamma: float = 0.1,
        last_epoch: int = -1,
    ) -> None:
        """Initializes StepLR.

        Args:
            optimizer: Wrapped optimizer.
            step_size: Period of learning rate decay.
            gamma: Multiplicative factor of learning rate decay.
            last_epoch: The index of last epoch.
        """
        self.step_size: int = step_size
        self.gamma: float = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        """Computes step-decayed learning rates.

        Returns:
            List of computed learning rates.
        """
        factor = self.gamma ** (self.last_epoch // self.step_size)
        return [base_lr * factor for base_lr in self.base_lrs]


class MultiStepLR(LRScheduler):
    """Decays the learning rate of each parameter group by gamma once the number of epoch reaches one of the milestones."""

    def __init__(
        self,
        optimizer: Optimizer,
        milestones: Sequence[int],
        gamma: float = 0.1,
        last_epoch: int = -1,
    ) -> None:
        """Initializes MultiStepLR.

        Args:
            optimizer: Wrapped optimizer.
            milestones: List of epoch indices.
            gamma: Multiplicative factor of learning rate decay.
            last_epoch: The index of last epoch.
        """
        self.milestones: list[int] = sorted(milestones)
        self.gamma: float = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        """Computes multi-step decayed learning rates.

        Returns:
            List of computed learning rates.
        """
        count = sum(1 for m in self.milestones if m <= self.last_epoch)
        factor = self.gamma**count
        return [base_lr * factor for base_lr in self.base_lrs]


class ConstantLR(LRScheduler):
    """Multiply the learning rate of each parameter group by a small constant factor until total_iters."""

    def __init__(
        self,
        optimizer: Optimizer,
        factor: float = 1.0 / 3.0,
        total_iters: int = 5,
        last_epoch: int = -1,
    ) -> None:
        """Initializes ConstantLR.

        Args:
            optimizer: Wrapped optimizer.
            factor: Multiplicative constant factor until total_iters.
            total_iters: Number of steps that the scheduler multiplies the learning rate by the factor.
            last_epoch: The index of last epoch.
        """
        self.factor: float = factor
        self.total_iters: int = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        """Computes constant-scaled learning rates.

        Returns:
            List of computed learning rates.
        """
        if self.last_epoch < self.total_iters:
            return [base_lr * self.factor for base_lr in self.base_lrs]
        return list(self.base_lrs)


class LinearLR(LRScheduler):
    """Decays the learning rate of each parameter group by linearly changing small multiplicative factor."""

    def __init__(
        self,
        optimizer: Optimizer,
        start_factor: float = 1.0 / 3.0,
        end_factor: float = 1.0,
        total_iters: int = 5,
        last_epoch: int = -1,
    ) -> None:
        """Initializes LinearLR.

        Args:
            optimizer: Wrapped optimizer.
            start_factor: Multiplicative factor in the first epoch.
            end_factor: Multiplicative factor at the end of linear changing process.
            total_iters: Number of iterations that multiplicative factor reaches end_factor.
            last_epoch: The index of last epoch.
        """
        self.start_factor: float = start_factor
        self.end_factor: float = end_factor
        self.total_iters: int = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        """Computes linearly interpolated learning rates.

        Returns:
            List of computed learning rates.
        """
        if self.last_epoch >= self.total_iters or self.total_iters <= 0:
            return [base_lr * self.end_factor for base_lr in self.base_lrs]
        alpha = float(self.last_epoch) / float(self.total_iters)
        factor = self.start_factor + alpha * (self.end_factor - self.start_factor)
        return [base_lr * factor for base_lr in self.base_lrs]


class ExponentialLR(LRScheduler):
    """Decays the learning rate of each parameter group by gamma every epoch."""

    def __init__(
        self,
        optimizer: Optimizer,
        gamma: float,
        last_epoch: int = -1,
    ) -> None:
        """Initializes ExponentialLR.

        Args:
            optimizer: Wrapped optimizer.
            gamma: Multiplicative factor of learning rate decay.
            last_epoch: The index of last epoch.
        """
        self.gamma: float = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        """Computes exponentially decayed learning rates.

        Returns:
            List of computed learning rates.
        """
        factor = self.gamma**self.last_epoch
        return [base_lr * factor for base_lr in self.base_lrs]


class PolynomialLR(LRScheduler):
    """Decays the learning rate of each parameter group using a polynomial function in the given total_iters."""

    def __init__(
        self,
        optimizer: Optimizer,
        total_iters: int = 5,
        power: float = 1.0,
        last_epoch: int = -1,
    ) -> None:
        """Initializes PolynomialLR.

        Args:
            optimizer: Wrapped optimizer.
            total_iters: Number of steps that the scheduler decays the learning rate.
            power: Power of the polynomial.
            last_epoch: The index of last epoch.
        """
        self.total_iters: int = total_iters
        self.power: float = power
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        """Computes polynomial decayed learning rates.

        Returns:
            List of computed learning rates.
        """
        if self.last_epoch >= self.total_iters or self.total_iters <= 0:
            return [0.0 for _ in self.base_lrs]
        factor = (1.0 - float(self.last_epoch) / float(self.total_iters)) ** self.power
        return [base_lr * factor for base_lr in self.base_lrs]


class CosineAnnealingLR(LRScheduler):
    """Set the learning rate of each parameter group using a cosine annealing schedule."""

    def __init__(
        self,
        optimizer: Optimizer,
        T_max: int,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        """Initializes CosineAnnealingLR.

        Args:
            optimizer: Wrapped optimizer.
            T_max: Maximum number of iterations.
            eta_min: Minimum learning rate.
            last_epoch: The index of last epoch.
        """
        self.T_max: int = T_max
        self.eta_min: float = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        """Computes cosine annealed learning rates.

        Returns:
            List of computed learning rates.
        """
        cos_val = math.cos(math.pi * float(self.last_epoch) / float(max(1, self.T_max)))
        return [
            self.eta_min + (base_lr - self.eta_min) * (1.0 + cos_val) / 2.0
            for base_lr in self.base_lrs
        ]


class CosineAnnealingWarmRestarts(LRScheduler):
    """Set the learning rate of each parameter group using a cosine annealing schedule with warm restarts."""

    def __init__(
        self,
        optimizer: Optimizer,
        T_0: int,
        T_mult: int = 1,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        """Initializes CosineAnnealingWarmRestarts.

        Args:
            optimizer: Wrapped optimizer.
            T_0: Number of iterations for the first restart.
            T_mult: A factor by which T_i increases after a restart.
            eta_min: Minimum learning rate.
            last_epoch: The index of last epoch.
        """
        self.T_0: int = T_0
        self.T_mult: int = T_mult
        self.eta_min: float = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        """Computes cosine annealed learning rates with warm restarts.

        Returns:
            List of computed learning rates.
        """
        t_cur = self.last_epoch % max(1, self.T_0)
        cos_val = math.cos(math.pi * float(t_cur) / float(max(1, self.T_0)))
        return [
            self.eta_min + (base_lr - self.eta_min) * (1.0 + cos_val) / 2.0
            for base_lr in self.base_lrs
        ]


class CyclicLR(LRScheduler):
    """Sets the learning rate of each parameter group according to cyclical learning rate policy."""

    def __init__(
        self,
        optimizer: Optimizer,
        base_lr: float | Sequence[float],
        max_lr: float | Sequence[float],
        step_size_up: int = 2000,
        step_size_down: int | None = None,
        mode: str = "triangular",
        gamma: float = 1.0,
        scale_fn: Callable[[int], float] | None = None,
        scale_mode: str = "cycle",
        cycle_momentum: bool = True,
        base_momentum: float | Sequence[float] = 0.8,
        max_momentum: float | Sequence[float] = 0.9,
        last_epoch: int = -1,
    ) -> None:
        """Initializes CyclicLR.

        Args:
            optimizer: Wrapped optimizer.
            base_lr: Initial learning rate which is the lower boundary in the cycle.
            max_lr: Upper learning rate boundaries in the cycle.
            step_size_up: Number of training iterations in the increasing half of a cycle.
            step_size_down: Number of training iterations in the decreasing half of a cycle.
            mode: One of 'triangular', 'triangular2', or 'exp_range'.
            gamma: Constant in 'exp_range' scaling function.
            scale_fn: Custom scaling policy function.
            scale_mode: 'cycle' or 'iterations'.
            cycle_momentum: If True, momentum is cycled inversely to learning rate.
            base_momentum: Lower momentum boundaries in the cycle.
            max_momentum: Upper momentum boundaries in the cycle.
            last_epoch: The index of the last batch.
        """
        self.base_lr_list: list[float] = (
            [float(base_lr)] * len(optimizer.param_groups)
            if isinstance(base_lr, (int, float))
            else [float(x) for x in base_lr]
        )
        self.max_lr_list: list[float] = (
            [float(max_lr)] * len(optimizer.param_groups)
            if isinstance(max_lr, (int, float))
            else [float(x) for x in max_lr]
        )
        self.step_size_up: int = step_size_up
        self.step_size_down: int = (
            step_size_down if step_size_down is not None else step_size_up
        )
        self.mode: str = mode
        self.gamma: float = gamma
        self.scale_fn: Callable[[int], float] | None = scale_fn
        self.scale_mode: str = scale_mode
        self.cycle_momentum: bool = cycle_momentum
        self.base_momentum: float | Sequence[float] = base_momentum
        self.max_momentum: float | Sequence[float] = max_momentum
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        """Computes cyclical learning rates.

        Returns:
            List of computed learning rates.
        """
        total_size = max(1, self.step_size_up + self.step_size_down)
        cycle = 1 + self.last_epoch // total_size
        x = float(self.last_epoch % total_size)
        if x <= float(self.step_size_up):
            scale_x = x / float(max(1, self.step_size_up))
        else:
            scale_x = (float(total_size) - x) / float(max(1, self.step_size_down))
        if self.scale_fn is not None:
            scale_factor = self.scale_fn(cycle)
        elif self.mode == "triangular2":
            scale_factor = 1.0 / (2.0 ** (cycle - 1))
        elif self.mode == "exp_range":
            scale_factor = self.gamma**self.last_epoch
        else:
            scale_factor = 1.0
        return [
            b_lr + (m_lr - b_lr) * max(0.0, scale_x) * scale_factor
            for b_lr, m_lr in zip(self.base_lr_list, self.max_lr_list)
        ]


class OneCycleLR(LRScheduler):
    """Sets the learning rate of each parameter group according to the 1cycle learning rate policy."""

    def __init__(
        self,
        optimizer: Optimizer,
        max_lr: float | Sequence[float],
        total_steps: int | None = None,
        epochs: int | None = None,
        steps_per_epoch: int | None = None,
        pct_start: float = 0.3,
        anneal_strategy: str = "cos",
        cycle_momentum: bool = True,
        base_momentum: float | Sequence[float] = 0.85,
        max_momentum: float | Sequence[float] = 0.95,
        div_factor: float = 25.0,
        final_div_factor: float = 10000.0,
        three_phase: bool = False,
        last_epoch: int = -1,
    ) -> None:
        """Initializes OneCycleLR.

        Args:
            optimizer: Wrapped optimizer.
            max_lr: Upper learning rate boundaries in the cycle.
            total_steps: The total number of steps in the cycle.
            epochs: The number of epochs to train for.
            steps_per_epoch: The number of steps per epoch to train for.
            pct_start: The percentage of the cycle spent increasing the learning rate.
            anneal_strategy: 'cos' or 'linear'.
            cycle_momentum: If True, momentum is cycled inversely to learning rate.
            base_momentum: Lower momentum boundaries in the cycle.
            max_momentum: Upper momentum boundaries in the cycle.
            div_factor: Determines the initial learning rate via initial_lr = max_lr / div_factor.
            final_div_factor: Determines the minimum learning rate via min_lr = initial_lr / final_div_factor.
            three_phase: If True, use a third phase of the schedule.
            last_epoch: The index of the last batch.
        """
        self.max_lr_list: list[float] = (
            [float(max_lr)] * len(optimizer.param_groups)
            if isinstance(max_lr, (int, float))
            else [float(x) for x in max_lr]
        )
        if total_steps is not None:
            self.total_steps: int = total_steps
        elif epochs is not None and steps_per_epoch is not None:
            self.total_steps = epochs * steps_per_epoch
        else:
            self.total_steps = 100
        self.pct_start: float = pct_start
        self.anneal_strategy: str = anneal_strategy
        self.cycle_momentum: bool = cycle_momentum
        self.base_momentum: float | Sequence[float] = base_momentum
        self.max_momentum: float | Sequence[float] = max_momentum
        self.div_factor: float = div_factor
        self.final_div_factor: float = final_div_factor
        self.three_phase: bool = three_phase
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        """Computes 1cycle learning rates.

        Returns:
            List of computed learning rates.
        """
        up_steps = max(1.0, float(self.total_steps) * self.pct_start)
        step = float(self.last_epoch)
        result: list[float] = []
        for m_lr in self.max_lr_list:
            init_lr = m_lr / self.div_factor
            min_lr = init_lr / self.final_div_factor
            if step <= up_steps:
                pct = step / up_steps
                lr = init_lr + pct * (m_lr - init_lr)
            else:
                down_steps = max(1.0, float(self.total_steps) - up_steps)
                pct = min(1.0, (step - up_steps) / down_steps)
                if self.anneal_strategy == "linear":
                    lr = m_lr + pct * (min_lr - m_lr)
                else:
                    cos_out = math.cos(math.pi * pct) + 1.0
                    lr = min_lr + (m_lr - min_lr) * cos_out / 2.0
            result.append(lr)
        return result


class ChainedScheduler(LRScheduler):
    """Chains a list of learning rate schedulers."""

    def __init__(
        self,
        schedulers: Sequence[LRScheduler],
        optimizer: Optimizer | None = None,
    ) -> None:
        """Initializes ChainedScheduler.

        Args:
            schedulers: Sequence of chained schedulers.
            optimizer: Optional wrapped optimizer.
        """
        self._schedulers: list[LRScheduler] = list(schedulers)
        opt = optimizer if optimizer is not None else self._schedulers[0].optimizer
        self.optimizer = opt
        self.last_epoch = -1
        self.base_lrs = list(self._schedulers[0].base_lrs)
        self._last_lr = list(self._schedulers[-1].get_last_lr())

    def step(self, epoch: int | None = None) -> None:
        """Steps all chained schedulers sequentially.

        Args:
            epoch: Unused optional epoch argument.
        """
        _ = epoch
        for scheduler in self._schedulers:
            scheduler.step()
        self._last_lr = list(self._schedulers[-1].get_last_lr())


class SequentialLR(LRScheduler):
    """Receives the list of schedulers that is expected to be called sequentially during optimization."""

    def __init__(
        self,
        optimizer: Optimizer,
        schedulers: Sequence[LRScheduler],
        milestones: Sequence[int],
        last_epoch: int = -1,
    ) -> None:
        """Initializes SequentialLR.

        Args:
            optimizer: Wrapped optimizer.
            schedulers: List of chained schedulers.
            milestones: List of integers that reflects milestone points.
            last_epoch: The index of last epoch.
        """
        self.optimizer = optimizer
        self._schedulers: list[LRScheduler] = list(schedulers)
        self._milestones: list[int] = list(milestones)
        self.last_epoch: int = last_epoch + 1
        self.base_lrs = list(self._schedulers[0].base_lrs)
        self._last_lr = list(self._schedulers[0].get_last_lr())

    def step(self, epoch: int | None = None) -> None:
        """Steps the active scheduler for the current milestone interval.

        Args:
            epoch: Unused optional epoch argument.
        """
        _ = epoch
        self.last_epoch += 1
        idx = sum(1 for m in self._milestones if m <= self.last_epoch)
        idx = min(idx, len(self._schedulers) - 1)
        active = self._schedulers[idx]
        active.step()
        self._last_lr = list(active.get_last_lr())


class ReduceLROnPlateau:
    """Reduce learning rate when a metric has stopped improving."""

    def __init__(
        self,
        optimizer: Optimizer,
        mode: str = "min",
        factor: float = 0.1,
        patience: int = 10,
        threshold: float = 1e-4,
        threshold_mode: str = "rel",
        cooldown: int = 0,
        min_lr: float | Sequence[float] = 0.0,
        eps: float = 1e-8,
    ) -> None:
        """Initializes ReduceLROnPlateau.

        Args:
            optimizer: Wrapped optimizer.
            mode: One of 'min', 'max'.
            factor: Factor by which the learning rate will be reduced.
            patience: Number of allowed epochs with no improvement after which learning rate will be reduced.
            threshold: Threshold for measuring the new optimum.
            threshold_mode: One of 'rel', 'abs'.
            cooldown: Number of epochs to wait before resuming normal operation after lr has been reduced.
            min_lr: A scalar or a list of scalars specifying lower bound on the learning rate.
            eps: Minimal decay applied to lr.
        """
        self.optimizer: Optimizer = optimizer
        self.mode: str = mode
        self.factor: float = factor
        self.patience: int = patience
        self.threshold: float = threshold
        self.threshold_mode: str = threshold_mode
        self.cooldown: int = cooldown
        self.min_lrs: list[float] = (
            [float(min_lr)] * len(optimizer.param_groups)
            if isinstance(min_lr, (int, float))
            else [float(x) for x in min_lr]
        )
        self.eps: float = eps
        self.best: float | None = None
        self.num_bad_epochs: int = 0
        self.cooldown_counter: int = 0
        self.last_epoch: int = 0
        self._last_lr: list[float] = []
        for g in optimizer.param_groups:
            lr_val = g.get("lr", 0.001)
            self._last_lr.append(
                float(lr_val) if isinstance(lr_val, (int, float)) else 0.001
            )

    def get_last_lr(self) -> list[float]:
        """Returns last computed learning rate by current scheduler.

        Returns:
            List of last computed learning rates.
        """
        return list(self._last_lr)

    def state_dict(self) -> dict[str, int | float | list[float] | None]:
        """Returns the state of the `ReduceLROnPlateau` scheduler as a dictionary.

        Returns:
            Dictionary containing plateau state attributes.
        """
        return {
            "best": self.best,
            "num_bad_epochs": self.num_bad_epochs,
            "cooldown_counter": self.cooldown_counter,
            "last_epoch": self.last_epoch,
            "_last_lr": list(self._last_lr),
        }

    def load_state_dict(
        self, state_dict: dict[str, int | float | list[float] | None]
    ) -> None:
        """Loads the `ReduceLROnPlateau` scheduler state.

        Args:
            state_dict: State dictionary returned from `state_dict()`.
        """
        best = state_dict.get("best")
        if best is None or isinstance(best, (int, float)):
            self.best = float(best) if isinstance(best, (int, float)) else None
        n_bad = state_dict.get("num_bad_epochs")
        if isinstance(n_bad, int):
            self.num_bad_epochs = n_bad
        cd = state_dict.get("cooldown_counter")
        if isinstance(cd, int):
            self.cooldown_counter = cd
        last_ep = state_dict.get("last_epoch")
        if isinstance(last_ep, int):
            self.last_epoch = last_ep
        last = state_dict.get("_last_lr")
        if isinstance(last, list):
            self._last_lr = [float(x) for x in last]
            for group, lr in zip(self.optimizer.param_groups, self._last_lr):
                group["lr"] = lr

    def step(self, metrics: float, epoch: int | None = None) -> None:
        """Evaluates metric improvement and reduces learning rate on plateau.

        Args:
            metrics: Quantity monitored to determine plateau.
            epoch: Optional epoch index.
        """
        current = float(metrics)
        self.last_epoch = self.last_epoch + 1 if epoch is None else epoch
        is_better = False
        if self.best is None:
            is_better = True
        elif self.mode == "min":
            rel_bound = (
                self.best * (1.0 - self.threshold)
                if self.threshold_mode == "rel"
                else self.best - self.threshold
            )
            is_better = current < rel_bound
        else:
            rel_bound = (
                self.best * (1.0 + self.threshold)
                if self.threshold_mode == "rel"
                else self.best + self.threshold
            )
            is_better = current > rel_bound

        if is_better:
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0

        if self.num_bad_epochs > self.patience:
            for i, group in enumerate(self.optimizer.param_groups):
                raw_old_lr = group.get("lr", 0.001)
                old_lr = (
                    float(raw_old_lr) if isinstance(raw_old_lr, (int, float)) else 0.001
                )
                new_lr = max(old_lr * self.factor, self.min_lrs[i])
                if old_lr - new_lr > self.eps:
                    group["lr"] = new_lr
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

        self._last_lr = []
        for g in self.optimizer.param_groups:
            lr_val = g.get("lr", 0.001)
            self._last_lr.append(
                float(lr_val) if isinstance(lr_val, (int, float)) else 0.001
            )


__all__ = [
    "ChainedScheduler",
    "ConstantLR",
    "CosineAnnealingLR",
    "CosineAnnealingWarmRestarts",
    "CyclicLR",
    "ExponentialLR",
    "LRScheduler",
    "LambdaLR",
    "LinearLR",
    "MultiStepLR",
    "MultiplicativeLR",
    "OneCycleLR",
    "PolynomialLR",
    "ReduceLROnPlateau",
    "SequentialLR",
    "StepLR",
    "_LRScheduler",
]
