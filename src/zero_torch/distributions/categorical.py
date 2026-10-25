"""Categorical distribution."""

from __future__ import annotations

from collections.abc import Sequence

from ml_switcheroo_compiler.ops.base import get_op

import zero_torch as torch
from zero_torch.tensor import Tensor, _wrap

from .distribution import Distribution
from .utils import _check_non_negative, _check_positive


class Categorical(Distribution):
    """Categorical probability distribution."""

    has_rsample: bool = False
    has_enumerate_support: bool = True

    def __init__(
        self,
        probs: Tensor | None = None,
        logits: Tensor | None = None,
        validate_args: bool | None = None,
    ) -> None:
        """Initializes the Categorical distribution.

        Args:
            probs: Event probabilities tensor.
            logits: Unnormalized log probabilities tensor.
            validate_args: Whether to validate parameter constraints.
        """
        if (probs is None) == (logits is None):
            raise ValueError(
                "Either `probs` or `logits` must be specified, but not both."
            )
        if probs is not None:
            if len(probs.shape) < 1:
                raise ValueError("`probs` parameter must be at least one-dimensional.")
            if validate_args:
                _check_non_negative(probs, "probs")
                _check_positive(torch.sum(probs, axis=-1), "probs sum")
            self.probs: Tensor = probs / torch.sum(probs, axis=-1, keepdims=True)
            eps = 1e-6
            clamped_probs = torch.clamp(self.probs, eps, 1.0 - eps)
            self.logits: Tensor = torch.log(clamped_probs)
            batch_shape = probs.shape[:-1]
            self._num_events: int = probs.shape[-1]
        else:
            assert logits is not None
            if len(logits.shape) < 1:
                raise ValueError("`logits` parameter must be at least one-dimensional.")
            self.logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
            self.probs = torch.softmax(self.logits, dim=-1)
            batch_shape = logits.shape[:-1]
            self._num_events = logits.shape[-1]
        super().__init__(batch_shape, validate_args=validate_args)

    def expand(
        self, batch_shape: Sequence[int], _instance: Distribution | None = None
    ) -> Categorical:
        """Returns a new Categorical distribution with batch dimensions expanded to `batch_shape`.

        Args:
            batch_shape: Desired expanded batch shape.
            _instance: Unused optional instance parameter.

        Returns:
            Expanded Categorical distribution.
        """
        _ = _instance
        b_shape = tuple(batch_shape)
        return Categorical(
            probs=torch.broadcast_to(self.probs, b_shape + (self._num_events,)),
            validate_args=False,
        )

    def enumerate_support(self, expand: bool = True) -> Tensor:
        """Returns tensor containing all category indices supported by the distribution.

        Args:
            expand: Whether to expand support over `batch_shape`.

        Returns:
            Support tensor of shape `(num_events, *batch_shape)` if `expand=True`.
        """
        values = torch.reshape(
            torch.arange(0, self._num_events, dtype=torch.int32),
            (self._num_events,) + (1,) * len(self.batch_shape),
        )
        if expand and self.batch_shape:
            return torch.broadcast_to(values, (self._num_events,) + self.batch_shape)
        return values

    @property
    def mean(self) -> Tensor:
        """Returns the expected category value.

        Returns:
            Mean tensor.
        """
        cats = torch.arange(0, self._num_events, dtype=self.probs.dtype)
        return torch.sum(self.probs * cats, axis=-1)

    @property
    def variance(self) -> Tensor:
        """Returns the variance of the Categorical distribution.

        Returns:
            Variance tensor.
        """
        cats = torch.arange(0, self._num_events, dtype=self.probs.dtype)
        m = torch.unsqueeze(self.mean, -1)
        return torch.sum(self.probs * ((cats - m) ** 2), axis=-1)

    def entropy(self) -> Tensor:
        """Returns the Shannon entropy of the Categorical distribution.

        Returns:
            Entropy tensor.
        """
        return -torch.sum(self.probs * self.logits, axis=-1)

    def sample(self, sample_shape: Sequence[int] | None = None) -> Tensor:
        """Generates a `sample_shape` shaped sample of category indices.

        Args:
            sample_shape: Shape of samples to draw.

        Returns:
            Sampled integer category indices tensor.
        """
        if sample_shape is None:
            sample_shape = ()
        shape = tuple(sample_shape) + self.logits.shape
        u = torch.rand(*shape)
        u = torch.clamp(u, 1e-6, 1.0 - 1e-6)
        gumbel = -torch.log(-torch.log(u))
        noisy_logits = self.logits + gumbel
        return torch.argmax(noisy_logits, axis=-1)

    def log_prob(self, value: Tensor) -> Tensor:
        """Returns the log of the probability mass function evaluated at `value`.

        Args:
            value: Category index tensor.

        Returns:
            Log probability tensor.
        """
        value_i32 = _wrap(get_op("Cast")()(value._tensor, torch.int32))
        logits = torch.broadcast_to(
            self.logits, value_i32.shape + (self.logits.shape[-1],)
        )
        return torch.gather(logits, -1, value_i32.unsqueeze(-1)).squeeze(-1)
