"""Poisson distribution."""

from __future__ import annotations

from collections.abc import Sequence

import zero_torch as torch
from zero_torch.tensor import Tensor

from .exp_family import ExponentialFamily


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
        self.rate: Tensor = rate
        batch_shape = rate.shape
        super().__init__(batch_shape, validate_args=validate_args)

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
        n = torch.randn(*shape)
        res = torch.round(n * torch.sqrt(self.rate) + self.rate)
        return torch.maximum(res, torch.tensor(0.0))

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
