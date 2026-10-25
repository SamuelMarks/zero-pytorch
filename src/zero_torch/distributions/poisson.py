"""Poisson distribution."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

import zero_torch as torch
from zero_torch.tensor import Tensor, _flatten_nested

from .distribution import Distribution
from .exp_family import ExponentialFamily
from .utils import _check_non_negative


def _sample_poisson_scalar(lam: float) -> float:
    """Draws a single exact Poisson sample using Knuth's algorithm for small `lam`.

    Args:
        lam: Non-negative rate parameter.

    Returns:
        Sampled non-negative integer count as a float.
    """
    if lam <= 0.0:
        return 0.0
    if lam < 30.0:
        limit = math.exp(-lam)
        k = 0
        p = 1.0
        while True:
            k += 1
            p *= random.random()
            if p <= limit:
                return float(k - 1)
    z = random.gauss(0.0, 1.0)
    return float(max(0, round(lam + math.sqrt(lam) * z)))


class Poisson(ExponentialFamily):
    """Poisson probability distribution."""

    has_rsample: bool = False

    def __init__(
        self,
        rate: Tensor,
        validate_args: bool | None = None,
    ) -> None:
        """Initializes the Poisson distribution.

        Args:
            rate: Rate parameter (`lambda`).
            validate_args: Whether to validate parameter constraints.
        """
        if validate_args:
            _check_non_negative(rate, "rate")
        self.rate: Tensor = rate
        batch_shape = rate.shape
        super().__init__(batch_shape, validate_args=validate_args)

    def expand(
        self, batch_shape: Sequence[int], _instance: Distribution | None = None
    ) -> Poisson:
        """Returns a new Poisson distribution with batch dimensions expanded to `batch_shape`.

        Args:
            batch_shape: Desired expanded batch shape.
            _instance: Unused optional instance parameter.

        Returns:
            Expanded Poisson distribution.
        """
        _ = _instance
        b_shape = tuple(batch_shape)
        return Poisson(torch.broadcast_to(self.rate, b_shape), validate_args=False)

    @property
    def _natural_params(self) -> tuple[Tensor, ...]:
        """Returns the natural parameter `(log(rate),)` of the Poisson distribution.

        Returns:
            Single-element tuple containing `log(self.rate)`.
        """
        return (torch.log(self.rate),)

    def _log_normalizer(self, *natural_params: Tensor) -> Tensor:
        """Computes the log normalizer `exp(eta)` for the Poisson distribution.

        Args:
            *natural_params: Natural parameter `(log(rate),)`.

        Returns:
            Log normalizer tensor.
        """
        return torch.exp(natural_params[0])

    @property
    def mean(self) -> Tensor:
        """Returns the mean (`rate`) of the Poisson distribution.

        Returns:
            Mean tensor.
        """
        return self.rate

    @property
    def variance(self) -> Tensor:
        """Returns the variance (`rate`) of the Poisson distribution.

        Returns:
            Variance tensor.
        """
        return self.rate

    def sample(self, sample_shape: Sequence[int] | None = None) -> Tensor:
        """Generates a `sample_shape` shaped sample of non-negative integer counts.

        Args:
            sample_shape: Shape of samples to draw.

        Returns:
            Sampled count tensor.
        """
        if sample_shape is None:
            sample_shape = ()
        shape = tuple(sample_shape) + self.batch_shape
        b_rate = torch.broadcast_to(self.rate, shape)
        flat_rates = _flatten_nested(b_rate.tolist())
        samples = [
            _sample_poisson_scalar(float(r) if isinstance(r, (int, float)) else 0.0)
            for r in flat_rates
        ]
        return torch.reshape(Tensor(samples, dtype=self.rate.dtype), shape)

    def log_prob(self, value: Tensor) -> Tensor:
        """Returns the log of the probability mass function evaluated at `value`.

        Args:
            value: Non-negative count tensor.

        Returns:
            Log probability mass tensor.
        """
        return (
            value * torch.log(self.rate)
            - self.rate
            - torch.special.gammaln(value + 1.0)
        )
