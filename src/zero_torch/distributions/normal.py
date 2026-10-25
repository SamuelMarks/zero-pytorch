"""Normal distribution."""

from __future__ import annotations

import math
from collections.abc import Sequence

import zero_torch as torch
from zero_torch.tensor import Tensor

from .distribution import Distribution
from .exp_family import ExponentialFamily
from .utils import _broadcast_shape, _check_positive


class Normal(ExponentialFamily):
    """Normal (Gaussian) probability distribution."""

    has_rsample: bool = True

    def __init__(
        self,
        loc: Tensor,
        scale: Tensor,
        validate_args: bool | None = None,
    ) -> None:
        """Initializes the Normal distribution.

        Args:
            loc: Mean of the distribution.
            scale: Standard deviation of the distribution.
            validate_args: Whether to validate parameter constraints.
        """
        if validate_args:
            _check_positive(scale, "scale")
        self.loc: Tensor = loc
        self.scale: Tensor = scale
        batch_shape = _broadcast_shape(loc.shape, scale.shape)
        super().__init__(batch_shape, validate_args=validate_args)

    def expand(
        self, batch_shape: Sequence[int], _instance: Distribution | None = None
    ) -> Normal:
        """Returns a new Normal distribution with batch dimensions expanded to `batch_shape`.

        Args:
            batch_shape: Desired expanded batch shape.
            _instance: Unused optional instance parameter.

        Returns:
            Expanded Normal distribution.
        """
        _ = _instance
        b_shape = tuple(batch_shape)
        return Normal(
            torch.broadcast_to(self.loc, b_shape),
            torch.broadcast_to(self.scale, b_shape),
            validate_args=False,
        )

    @property
    def _natural_params(self) -> tuple[Tensor, ...]:
        """Returns the natural parameters `(loc / scale^2, -0.5 / scale^2)` of the Normal distribution.

        Returns:
            Two-element tuple of natural parameter tensors.
        """
        return (self.loc / (self.scale**2), -0.5 / (self.scale**2))

    def _log_normalizer(self, *natural_params: Tensor) -> Tensor:
        """Computes the log normalizer function for given Normal natural parameters.

        Args:
            *natural_params: Natural parameters `(eta_1, eta_2)`.

        Returns:
            Log normalizer tensor.
        """
        x = natural_params[0]
        y = natural_params[1]
        return -0.25 * (x**2) / y + 0.5 * torch.log(-math.pi / y)

    @property
    def mean(self) -> Tensor:
        """Returns the mean (`loc`) of the Normal distribution.

        Returns:
            Mean tensor.
        """
        return self.loc

    @property
    def variance(self) -> Tensor:
        """Returns the variance (`scale^2`) of the Normal distribution.

        Returns:
            Variance tensor.
        """
        return self.scale**2

    def entropy(self) -> Tensor:
        """Returns the differential entropy of the Normal distribution.

        Returns:
            Entropy tensor.
        """
        return 0.5 + 0.5 * math.log(2.0 * math.pi) + torch.log(self.scale)

    def cdf(self, value: Tensor) -> Tensor:
        """Returns the cumulative distribution function evaluated at `value`.

        Args:
            value: Value tensor.

        Returns:
            Cumulative probability tensor.
        """
        return 0.5 * (
            1.0 + torch.erf((value - self.loc) / (self.scale * math.sqrt(2.0)))
        )

    def icdf(self, value: Tensor) -> Tensor:
        """Returns the inverse cumulative distribution function evaluated at `value`.

        Args:
            value: Probability quantile tensor in `(0, 1)`.

        Returns:
            Quantile value tensor.
        """
        return self.loc + self.scale * torch.erfinv(2.0 * value - 1.0) * math.sqrt(2.0)

    def rsample(self, sample_shape: Sequence[int] | None = None) -> Tensor:
        """Generates a `sample_shape` shaped reparameterized sample.

        Args:
            sample_shape: Shape of samples to draw.

        Returns:
            Sampled tensor.
        """
        if sample_shape is None:
            sample_shape = ()
        shape = tuple(sample_shape) + self.batch_shape
        return torch.randn(*shape) * self.scale + self.loc

    def log_prob(self, value: Tensor) -> Tensor:
        """Returns the log of the probability density function evaluated at `value`.

        Args:
            value: Value tensor at which to evaluate log probability.

        Returns:
            Log probability density tensor.
        """
        var = self.scale**2
        log_scale = torch.log(self.scale)
        return (
            -((value - self.loc) ** 2) / (2 * var)
            - log_scale
            - math.log(math.sqrt(2 * math.pi))
        )
