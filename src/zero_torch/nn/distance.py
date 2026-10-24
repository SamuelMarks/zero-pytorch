"""Distance modules."""

from __future__ import annotations

from ml_switcheroo_compiler.ops.binary import maximum, power
from ml_switcheroo_compiler.ops.reductions import sum as _reduce_sum
from ml_switcheroo_compiler.ops.unary import abs as _unary_abs
from ml_switcheroo_compiler.ops.unary import sqrt

from zero_torch.tensor import Tensor, _to_tensor, _wrap

from .module import Module


class CosineSimilarity(Module):
    """Returns cosine similarity between `x1` and `x2`, computed along `dim`."""

    def __init__(self, dim: int = 1, eps: float = 1e-08) -> None:
        """Initializes CosineSimilarity.

        Args:
            dim: Dimension where cosine similarity is computed.
            eps: Small value to avoid division by zero.
        """
        super().__init__()
        self.dim: int = dim
        self.eps: float = eps

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        """Computes cosine similarity between `x1` and `x2`.

        Args:
            x1: First input tensor.
            x2: Second input tensor.

        Returns:
            Cosine similarity tensor.
        """
        x1_t = _to_tensor(x1)
        x2_t = _to_tensor(x2)
        dot_product = _reduce_sum(x1_t * x2_t, axis=self.dim)
        norm_x1 = sqrt(maximum(_reduce_sum(x1_t * x1_t, axis=self.dim), self.eps))
        norm_x2 = sqrt(maximum(_reduce_sum(x2_t * x2_t, axis=self.dim), self.eps))
        return _wrap(dot_product / (norm_x1 * norm_x2))


class PairwiseDistance(Module):
    """Computes the pairwise distance between vectors using the `p`-norm."""

    def __init__(
        self, p: float = 2.0, eps: float = 1e-06, keepdim: bool = False
    ) -> None:
        """Initializes PairwiseDistance.

        Args:
            p: The norm degree.
            eps: Small value to avoid division by zero.
            keepdim: Whether or not to keep the vector dimension.
        """
        super().__init__()
        self.p: float = p
        self.eps: float = eps
        self.keepdim: bool = keepdim

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        """Computes pairwise distance between `x1` and `x2`.

        Args:
            x1: First input tensor.
            x2: Second input tensor.

        Returns:
            Pairwise distance tensor.
        """
        x1_t = _to_tensor(x1)
        x2_t = _to_tensor(x2)
        diff = _unary_abs(x1_t - x2_t)

        if self.p == 1.0:
            res = _reduce_sum(diff, axis=-1, keepdims=self.keepdim)
        elif self.p == 2.0:
            res = sqrt(
                _reduce_sum(diff * diff, axis=-1, keepdims=self.keepdim) + self.eps
            )
        else:
            res = power(
                _reduce_sum(power(diff, self.p), axis=-1, keepdims=self.keepdim)
                + self.eps,
                1.0 / self.p,
            )

        return _wrap(res)
