"""Gamma distribution."""

from __future__ import annotations

from collections.abc import Sequence

import zero_torch as torch
from zero_torch.tensor import Tensor

from .exp_family import ExponentialFamily
from .utils import _broadcast_shape


class Gamma(ExponentialFamily):
    """Gamma probability distribution."""

    has_rsample: bool = True

    def __init__(
        self,
        concentration: Tensor,
        rate: Tensor,
        validate_args: bool | None = None,
    ) -> None:
        """Initializes the Gamma distribution.

        Args:
            concentration: Shape parameter (`alpha` / `k`).
            rate: Inverse scale parameter (`beta`).
            validate_args: Whether to validate parameter constraints.
        """
        self.concentration: Tensor = concentration
        self.rate: Tensor = rate
        batch_shape = _broadcast_shape(concentration.shape, rate.shape)
        super().__init__(batch_shape, validate_args=validate_args)

    @property
    def mean(self) -> Tensor:
        """Returns the mean of the Gamma distribution (`concentration / rate`).

        Returns:
            Mean tensor.
        """
        return self.concentration / self.rate

    @property
    def variance(self) -> Tensor:
        """Returns the variance of the Gamma distribution (`concentration / rate^2`).

        Returns:
            Variance tensor.
        """
        return self.concentration / (self.rate**2)

    def rsample(self, sample_shape: Sequence[int] | None = None) -> Tensor:
        """Generates a `sample_shape` shaped reparameterized sample.

        Args:
            sample_shape: Shape of samples to draw.

        Returns:
            Positive sampled tensor.
        """
        if sample_shape is None:
            sample_shape = ()
        shape = tuple(sample_shape) + self.batch_shape
        n = torch.randn(*shape)
        k = self.concentration
        theta = 1.0 / self.rate
        res = k * theta * (1.0 - 1.0 / (9.0 * k) + n / torch.sqrt(9.0 * k)) ** 3
        return torch.maximum(res, torch.tensor(1e-6))

    def log_prob(self, value: Tensor) -> Tensor:
        """Returns the log of the probability density function evaluated at `value`.

        Args:
            value: Positive value tensor.

        Returns:
            Log probability density tensor.
        """
        return (
            self.concentration * torch.log(self.rate)
            + (self.concentration - 1) * torch.log(value)
            - self.rate * value
            - torch.special.gammaln(self.concentration)
        )
