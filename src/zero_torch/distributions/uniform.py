"""Uniform distribution."""

from __future__ import annotations

from collections.abc import Sequence

import zero_torch as torch
from zero_torch.tensor import Tensor

from .distribution import Distribution
from .utils import _broadcast_shape, _check_less_than


class Uniform(Distribution):
    """Continuous Uniform probability distribution."""

    has_rsample: bool = True

    def __init__(
        self,
        low: Tensor,
        high: Tensor,
        validate_args: bool | None = None,
    ) -> None:
        """Initializes the Uniform distribution.

        Args:
            low: Lower range (inclusive).
            high: Upper range (exclusive).
            validate_args: Whether to validate parameter constraints.
        """
        if validate_args:
            _check_less_than(low, high)
        self.low: Tensor = low
        self.high: Tensor = high
        batch_shape = _broadcast_shape(low.shape, high.shape)
        super().__init__(batch_shape, validate_args=validate_args)

    def expand(
        self, batch_shape: Sequence[int], _instance: Distribution | None = None
    ) -> Uniform:
        """Returns a new Uniform distribution with batch dimensions expanded to `batch_shape`.

        Args:
            batch_shape: Desired expanded batch shape.
            _instance: Unused optional instance parameter.

        Returns:
            Expanded Uniform distribution.
        """
        _ = _instance
        b_shape = tuple(batch_shape)
        return Uniform(
            torch.broadcast_to(self.low, b_shape),
            torch.broadcast_to(self.high, b_shape),
            validate_args=False,
        )

    @property
    def mean(self) -> Tensor:
        """Returns the mean (`(low + high) / 2`) of the Uniform distribution.

        Returns:
            Mean tensor.
        """
        return (self.low + self.high) / 2.0

    @property
    def variance(self) -> Tensor:
        """Returns the variance (`(high - low)^2 / 12`) of the Uniform distribution.

        Returns:
            Variance tensor.
        """
        return ((self.high - self.low) ** 2) / 12.0

    def entropy(self) -> Tensor:
        """Returns the differential entropy (`log(high - low)`) of the Uniform distribution.

        Returns:
            Entropy tensor.
        """
        return torch.log(self.high - self.low)

    def cdf(self, value: Tensor) -> Tensor:
        """Returns the cumulative distribution function evaluated at `value`.

        Args:
            value: Value tensor.

        Returns:
            Cumulative probability tensor in `[0, 1]`.
        """
        res = (value - self.low) / (self.high - self.low)
        return torch.clamp(res, 0.0, 1.0)

    def icdf(self, value: Tensor) -> Tensor:
        """Returns the inverse cumulative distribution function evaluated at `value`.

        Args:
            value: Probability quantile tensor in `[0, 1]`.

        Returns:
            Quantile value tensor in `[low, high]`.
        """
        return self.low + value * (self.high - self.low)

    def rsample(self, sample_shape: Sequence[int] | None = None) -> Tensor:
        """Generates a `sample_shape` shaped reparameterized sample.

        Args:
            sample_shape: Shape of samples to draw.

        Returns:
            Sampled tensor in `[low, high)`.
        """
        if sample_shape is None:
            sample_shape = ()
        shape = tuple(sample_shape) + self.batch_shape
        rand_tensor = torch.rand(*shape)
        return self.low + rand_tensor * (self.high - self.low)

    def log_prob(self, value: Tensor) -> Tensor:
        """Returns the log of the probability density function evaluated at `value`.

        Args:
            value: Value tensor at which to evaluate log probability.

        Returns:
            Log probability density tensor.
        """
        prob = torch.where(
            torch.logical_and(value >= self.low, value < self.high),
            torch.ones_like(value) / (self.high - self.low),
            torch.zeros_like(value),
        )
        return torch.log(prob)
