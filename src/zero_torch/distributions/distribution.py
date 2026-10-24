"""Base class for probability distributions."""

from __future__ import annotations

from collections.abc import Sequence

import zero_torch as torch
from zero_torch.tensor import Tensor


class Distribution:
    """Base class for probability distributions."""

    has_rsample: bool = False
    has_enumerate_support: bool = False
    _validate_args: bool = False

    def __init__(
        self,
        batch_shape: Sequence[int] = (),
        event_shape: Sequence[int] = (),
        validate_args: bool | None = None,
    ) -> None:
        """Initializes the Distribution.

        Args:
            batch_shape: Shape over which parameters are batched.
            event_shape: Shape of a single sample (without batching).
            validate_args: Whether to validate distribution parameter constraints.
        """
        self._batch_shape: tuple[int, ...] = tuple(batch_shape)
        self._event_shape: tuple[int, ...] = tuple(event_shape)
        if validate_args is not None:
            self._validate_args = validate_args

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """Returns the shape over which parameters are batched.

        Returns:
            Batch shape tuple.
        """
        return self._batch_shape

    @property
    def event_shape(self) -> tuple[int, ...]:
        """Returns the shape of a single sample (without batching).

        Returns:
            Event shape tuple.
        """
        return self._event_shape

    @property
    def mean(self) -> Tensor:
        """Returns the mean of the distribution.

        Returns:
            Mean tensor.
        """
        raise NotImplementedError

    @property
    def variance(self) -> Tensor:
        """Returns the variance of the distribution.

        Returns:
            Variance tensor.
        """
        raise NotImplementedError

    @property
    def stddev(self) -> Tensor:
        """Returns the standard deviation of the distribution.

        Returns:
            Standard deviation tensor.
        """
        return torch.sqrt(self.variance)

    def sample(self, sample_shape: Sequence[int] | None = None) -> Tensor:
        """Generates a `sample_shape` shaped sample or batch of samples.

        Args:
            sample_shape: Shape of samples to draw.

        Returns:
            Sampled tensor.
        """
        if sample_shape is None:
            sample_shape = ()
        with torch.no_grad():
            return self.rsample(sample_shape)

    def rsample(self, sample_shape: Sequence[int] | None = None) -> Tensor:
        """Generates a `sample_shape` shaped reparameterized sample.

        Args:
            sample_shape: Shape of samples to draw.

        Returns:
            Reparameterized sample tensor.
        """
        _ = sample_shape
        raise NotImplementedError

    def expand(
        self, batch_shape: Sequence[int], _instance: Distribution | None = None
    ) -> Distribution:
        """Returns a new distribution instance with batch dimensions expanded to `batch_shape`.

        Args:
            batch_shape: The desired expanded batch size.
            _instance: New instance provided by subclasses that need to override `.expand`.

        Returns:
            Expanded distribution instance.
        """
        _ = (batch_shape, _instance)
        raise NotImplementedError

    def log_prob(self, value: Tensor) -> Tensor:
        """Returns the log of the probability density/mass function evaluated at `value`.

        Args:
            value: Value tensor at which to evaluate log probability.

        Returns:
            Log probability tensor.
        """
        _ = value
        raise NotImplementedError

    def cdf(self, value: Tensor) -> Tensor:
        """Returns the cumulative density/mass function evaluated at `value`.

        Args:
            value: Value tensor at which to evaluate the CDF.

        Returns:
            Cumulative probability tensor.
        """
        _ = value
        raise NotImplementedError

    def icdf(self, value: Tensor) -> Tensor:
        """Returns the inverse cumulative density/mass function evaluated at `value`.

        Args:
            value: Probability quantile tensor.

        Returns:
            Inverse CDF tensor.
        """
        _ = value
        raise NotImplementedError

    def enumerate_support(self, expand: bool = True) -> Tensor:
        """Returns tensor containing all values supported by a discrete distribution.

        Args:
            expand: Whether to expand support over `batch_shape`.

        Returns:
            Support tensor.
        """
        _ = expand
        raise NotImplementedError

    def entropy(self) -> Tensor:
        """Returns entropy of distribution, batched over `batch_shape`.

        Returns:
            Entropy tensor.
        """
        raise NotImplementedError
