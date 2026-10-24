"""Bernoulli distribution."""

from __future__ import annotations

from collections.abc import Sequence

from ml_switcheroo_compiler.ops.base import get_op

import zero_torch as torch
from zero_torch.tensor import Tensor, _wrap

from .exp_family import ExponentialFamily


class Bernoulli(ExponentialFamily):
    """Bernoulli probability distribution."""

    has_rsample: bool = True
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

    def rsample(self, sample_shape: Sequence[int] | None = None) -> Tensor:
        """Generates a `sample_shape` shaped reparameterized sample.

        Args:
            sample_shape: Shape of samples to draw.

        Returns:
            Sampled tensor of 0s and 1s.
        """
        if sample_shape is None:
            sample_shape = ()
        shape = tuple(sample_shape) + self.batch_shape
        u = torch.rand(*shape)
        res = u < self.probs
        return _wrap(get_op("Cast")()(res._tensor, torch.float32))

    def sample(self, sample_shape: Sequence[int] | None = None) -> Tensor:
        """Generates a `sample_shape` shaped sample.

        Args:
            sample_shape: Shape of samples to draw.

        Returns:
            Sampled tensor of 0s and 1s.
        """
        with torch.no_grad():
            return self.rsample(sample_shape)

    def log_prob(self, value: Tensor) -> Tensor:
        """Returns the log of the probability mass function evaluated at `value`.

        Args:
            value: Binary tensor of 0s and 1s.

        Returns:
            Log probability tensor.
        """
        logits = torch.broadcast_to(self.logits, value.shape)
        return value * logits - torch.nn.functional.softplus(logits)
