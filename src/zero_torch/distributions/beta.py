"""Beta distribution."""

from __future__ import annotations

from collections.abc import Sequence

import zero_torch as torch
from zero_torch.tensor import Tensor

from .exp_family import ExponentialFamily
from .utils import _broadcast_shape


class Beta(ExponentialFamily):
    """Beta probability distribution."""

    has_rsample: bool = True

    def __init__(
        self,
        concentration1: Tensor,
        concentration0: Tensor,
        validate_args: bool | None = None,
    ) -> None:
        """Initializes the Beta distribution.

        Args:
            concentration1: First concentration parameter (`alpha`).
            concentration0: Second concentration parameter (`beta`).
            validate_args: Whether to validate parameter constraints.
        """
        self.concentration1: Tensor = concentration1
        self.concentration0: Tensor = concentration0
        batch_shape = _broadcast_shape(concentration1.shape, concentration0.shape)
        super().__init__(batch_shape, validate_args=validate_args)

    @property
    def mean(self) -> Tensor:
        """Returns the mean of the Beta distribution (`c1 / (c1 + c0)`).

        Returns:
            Mean tensor.
        """
        return self.concentration1 / (self.concentration1 + self.concentration0)

    @property
    def variance(self) -> Tensor:
        """Returns the variance of the Beta distribution.

        Returns:
            Variance tensor.
        """
        total = self.concentration1 + self.concentration0
        return (self.concentration1 * self.concentration0) / (
            (total**2) * (total + 1.0)
        )

    def rsample(self, sample_shape: Sequence[int] | None = None) -> Tensor:
        """Generates a `sample_shape` shaped reparameterized sample.

        Args:
            sample_shape: Shape of samples to draw.

        Returns:
            Sampled tensor in `(0, 1)`.
        """
        if sample_shape is None:
            sample_shape = ()
        from .gamma import Gamma

        g1 = Gamma(self.concentration1, torch.ones_like(self.concentration1)).rsample(
            sample_shape
        )
        g0 = Gamma(self.concentration0, torch.ones_like(self.concentration0)).rsample(
            sample_shape
        )
        return g1 / (g1 + g0)

    def log_prob(self, value: Tensor) -> Tensor:
        """Returns the log of the probability density function evaluated at `value`.

        Args:
            value: Value tensor in `(0, 1)`.

        Returns:
            Log probability density tensor.
        """
        log_b = (
            torch.special.gammaln(self.concentration1)
            + torch.special.gammaln(self.concentration0)
            - torch.special.gammaln(self.concentration1 + self.concentration0)
        )
        return (
            (self.concentration1 - 1) * torch.log(value)
            + (self.concentration0 - 1) * torch.log(1 - value)
            - log_b
        )
