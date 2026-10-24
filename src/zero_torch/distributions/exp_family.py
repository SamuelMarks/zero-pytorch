"""ExponentialFamily distribution base class."""

from __future__ import annotations

from zero_torch.tensor import Tensor

from .distribution import Distribution


class ExponentialFamily(Distribution):
    """Base class for probability distributions belonging to the exponential family."""

    @property
    def _natural_params(self) -> tuple[Tensor, ...]:
        """Returns the natural parameters of the exponential family distribution.

        Returns:
            Tuple of natural parameter tensors.
        """
        raise NotImplementedError

    def _log_normalizer(self, *natural_params: Tensor) -> Tensor:
        """Computes the log normalizer function for given natural parameters.

        Args:
            *natural_params: Natural parameter tensors.

        Returns:
            Log normalizer tensor.
        """
        _ = natural_params
        raise NotImplementedError

    @property
    def _mean_carrier_measure(self) -> float:
        """Returns the expected carrier measure value.

        Returns:
            Mean carrier measure scalar.
        """
        raise NotImplementedError

    def entropy(self) -> Tensor:
        """Computes the entropy for ExponentialFamily distributions.

        Returns:
            Entropy tensor.
        """
        raise NotImplementedError
