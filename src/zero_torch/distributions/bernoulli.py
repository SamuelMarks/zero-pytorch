"""Bernoulli distribution."""

from __future__ import annotations

from collections.abc import Sequence

from ml_switcheroo_compiler.ops.base import get_op

import zero_torch as torch
from zero_torch.tensor import Tensor, _wrap

from .distribution import Distribution
from .exp_family import ExponentialFamily
from .utils import _check_unit_interval


class Bernoulli(ExponentialFamily):
    """Bernoulli probability distribution."""

    has_rsample: bool = False
    has_enumerate_support: bool = True

    def __init__(
        self,
        probs: Tensor | None = None,
        logits: Tensor | None = None,
        validate_args: bool | None = None,
    ) -> None:
        """Initializes the Bernoulli distribution.

        Args:
            probs: Probability of sampling `1`.
            logits: Log-odds of sampling `1`.
            validate_args: Whether to validate parameter constraints.
        """
        if (probs is None) == (logits is None):
            raise ValueError(
                "Either `probs` or `logits` must be specified, but not both."
            )
        if probs is not None:
            if validate_args:
                _check_unit_interval(probs, "probs")
            self.probs: Tensor = probs
            eps = 1e-6
            clamped_probs = torch.clamp(probs, eps, 1.0 - eps)
            self.logits: Tensor = torch.log(clamped_probs) - torch.log(
                1.0 - clamped_probs
            )
            batch_shape = probs.shape
        else:
            assert logits is not None
            self.logits = logits
            self.probs = torch.sigmoid(logits)
            batch_shape = logits.shape
        super().__init__(batch_shape, validate_args=validate_args)

    def expand(
        self, batch_shape: Sequence[int], _instance: Distribution | None = None
    ) -> Bernoulli:
        """Returns a new Bernoulli distribution with batch dimensions expanded to `batch_shape`.

        Args:
            batch_shape: Desired expanded batch shape.
            _instance: Unused optional instance parameter.

        Returns:
            Expanded Bernoulli distribution.
        """
        _ = _instance
        b_shape = tuple(batch_shape)
        return Bernoulli(
            probs=torch.broadcast_to(self.probs, b_shape), validate_args=False
        )

    @property
    def _natural_params(self) -> tuple[Tensor, ...]:
        """Returns the natural parameters `(logits,)` of the Bernoulli distribution.

        Returns:
            Single-element tuple containing `self.logits`.
        """
        return (self.logits,)

    def _log_normalizer(self, *natural_params: Tensor) -> Tensor:
        """Computes the log normalizer `softplus(logits)` for the Bernoulli distribution.

        Args:
            *natural_params: Natural parameter tensor `(logits,)`.

        Returns:
            Log normalizer tensor.
        """
        return torch.nn.functional.softplus(natural_params[0])

    @property
    def mean(self) -> Tensor:
        """Returns the mean of the Bernoulli distribution (`probs`).

        Returns:
            Mean tensor.
        """
        return self.probs

    @property
    def variance(self) -> Tensor:
        """Returns the variance of the Bernoulli distribution (`probs * (1 - probs)`).

        Returns:
            Variance tensor.
        """
        return self.probs * (1.0 - self.probs)

    def entropy(self) -> Tensor:
        """Returns the binary entropy of the Bernoulli distribution.

        Returns:
            Entropy tensor.
        """
        return torch.nn.functional.binary_cross_entropy_with_logits(
            self.logits, self.probs, reduction="none"
        )

    def enumerate_support(self, expand: bool = True) -> Tensor:
        """Returns tensor containing all values `(0, 1)` supported by the Bernoulli distribution.

        Args:
            expand: Whether to expand support over `batch_shape`.

        Returns:
            Support tensor of shape `(2, *batch_shape)` if `expand=True`.
        """
        values = torch.reshape(
            torch.tensor([0.0, 1.0], dtype=self.probs.dtype),
            (2,) + (1,) * len(self.batch_shape),
        )
        if expand and self.batch_shape:
            return torch.broadcast_to(values, (2,) + self.batch_shape)
        return values

    def rsample(self, sample_shape: Sequence[int] | None = None) -> Tensor:
        """Raises NotImplementedError because Bernoulli is discrete.

        Args:
            sample_shape: Shape of samples to draw.

        Returns:
            Never returns.
        """
        _ = sample_shape
        raise NotImplementedError("Bernoulli does not support rsample")

    def sample(self, sample_shape: Sequence[int] | None = None) -> Tensor:
        """Generates a `sample_shape` shaped sample.

        Args:
            sample_shape: Shape of samples to draw.

        Returns:
            Sampled tensor of 0s and 1s.
        """
        with torch.no_grad():
            if sample_shape is None:
                sample_shape = ()
            shape = tuple(sample_shape) + self.batch_shape
            u = torch.rand(*shape)
            res = u < self.probs
            return _wrap(get_op("Cast")()(res._tensor, torch.float32))

    def log_prob(self, value: Tensor) -> Tensor:
        """Returns the log of the probability mass function evaluated at `value`.

        Args:
            value: Binary tensor of 0s and 1s.

        Returns:
            Log probability tensor.
        """
        logits = torch.broadcast_to(self.logits, value.shape)
        return value * logits - torch.nn.functional.softplus(logits)
