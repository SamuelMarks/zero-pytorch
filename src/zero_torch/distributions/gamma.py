"""Gamma distribution."""

from __future__ import annotations

from collections.abc import Sequence

import zero_torch as torch
from zero_torch.tensor import Tensor

from .distribution import Distribution
from .exp_family import ExponentialFamily
from .utils import _broadcast_shape, _check_positive


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
        if validate_args:
            _check_positive(concentration, "concentration")
            _check_positive(rate, "rate")
        self.concentration: Tensor = concentration
        self.rate: Tensor = rate
        batch_shape = _broadcast_shape(concentration.shape, rate.shape)
        super().__init__(batch_shape, validate_args=validate_args)

    def expand(
        self, batch_shape: Sequence[int], _instance: Distribution | None = None
    ) -> Gamma:
        """Returns a new Gamma distribution with batch dimensions expanded to `batch_shape`.

        Args:
            batch_shape: Desired expanded batch shape.
            _instance: Unused optional instance parameter.

        Returns:
            Expanded Gamma distribution.
        """
        _ = _instance
        b_shape = tuple(batch_shape)
        return Gamma(
            torch.broadcast_to(self.concentration, b_shape),
            torch.broadcast_to(self.rate, b_shape),
            validate_args=False,
        )

    @property
    def _natural_params(self) -> tuple[Tensor, ...]:
        """Returns the natural parameters `(concentration - 1, -rate)` of the Gamma distribution.

        Returns:
            Two-element tuple of natural parameter tensors.
        """
        return (self.concentration - 1.0, -self.rate)

    def _log_normalizer(self, *natural_params: Tensor) -> Tensor:
        """Computes the log normalizer function for given Gamma natural parameters.

        Args:
            *natural_params: Natural parameters `(concentration - 1, -rate)`.

        Returns:
            Log normalizer tensor.
        """
        x = natural_params[0]
        y = natural_params[1]
        c = x + 1.0
        return torch.special.gammaln(c) - c * torch.log(-y)

    def entropy(self) -> Tensor:
        """Returns the differential entropy of the Gamma distribution.

        Returns:
            Entropy tensor.
        """
        return (
            self.concentration
            - torch.log(self.rate)
            + torch.special.gammaln(self.concentration)
            + (1.0 - self.concentration) * torch.digamma(self.concentration)
        )

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
