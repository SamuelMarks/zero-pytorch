"""Beta distribution."""

from __future__ import annotations

from collections.abc import Sequence

import zero_torch as torch
from zero_torch.tensor import Tensor

from .distribution import Distribution
from .exp_family import ExponentialFamily
from .utils import _broadcast_shape, _check_positive


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
        if validate_args:
            _check_positive(concentration1, "concentration1")
            _check_positive(concentration0, "concentration0")
        self.concentration1: Tensor = concentration1
        self.concentration0: Tensor = concentration0
        batch_shape = _broadcast_shape(concentration1.shape, concentration0.shape)
        super().__init__(batch_shape, validate_args=validate_args)

    def expand(
        self, batch_shape: Sequence[int], _instance: Distribution | None = None
    ) -> Beta:
        """Returns a new Beta distribution with batch dimensions expanded to `batch_shape`.

        Args:
            batch_shape: Desired expanded batch shape.
            _instance: Unused optional instance parameter.

        Returns:
            Expanded Beta distribution.
        """
        _ = _instance
        b_shape = tuple(batch_shape)
        return Beta(
            torch.broadcast_to(self.concentration1, b_shape),
            torch.broadcast_to(self.concentration0, b_shape),
            validate_args=False,
        )

    @property
    def _natural_params(self) -> tuple[Tensor, ...]:
        """Returns the natural parameters `(concentration1, concentration0)` of the Beta distribution.

        Returns:
            Two-element tuple of concentration tensors.
        """
        return (self.concentration1, self.concentration0)

    def _log_normalizer(self, *natural_params: Tensor) -> Tensor:
        """Computes the log Beta function normalizer for given concentration parameters.

        Args:
            *natural_params: Natural parameters `(c1, c0)`.

        Returns:
            Log normalizer tensor.
        """
        c1 = natural_params[0]
        c0 = natural_params[1]
        return (
            torch.special.gammaln(c1)
            + torch.special.gammaln(c0)
            - torch.special.gammaln(c1 + c0)
        )

    def entropy(self) -> Tensor:
        """Returns the differential entropy of the Beta distribution.

        Returns:
            Entropy tensor.
        """
        c1 = self.concentration1
        c0 = self.concentration0
        total = c1 + c0
        return (
            self._log_normalizer(c1, c0)
            - (c1 - 1.0) * torch.digamma(c1)
            - (c0 - 1.0) * torch.digamma(c0)
            + (total - 2.0) * torch.digamma(total)
        )

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
        log_b = self._log_normalizer(self.concentration1, self.concentration0)
        return (
            (self.concentration1 - 1) * torch.log(value)
            + (self.concentration0 - 1) * torch.log(1 - value)
            - log_b
        )
