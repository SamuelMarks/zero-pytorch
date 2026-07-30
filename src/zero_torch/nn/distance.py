"""Distance modules."""

from ml_switcheroo_compiler.ops.binary import maximum, power
from ml_switcheroo_compiler.ops.reductions import sum
from ml_switcheroo_compiler.ops.unary import abs, sqrt

from .module import Module


class CosineSimilarity(Module):
    """Returns cosine similarity between x1 and x2."""

    def __init__(self, dim: int = 1, eps: float = 1e-08) -> None:
        """Initializes CosineSimilarity."""
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x1, x2):
        """Forward pass."""
        from zero_torch.tensor import _to_tensor, _wrap

        x1_t = _to_tensor(x1)
        x2_t = _to_tensor(x2)
        dot_product = sum(x1_t * x2_t, dim=self.dim)
        norm_x1 = sqrt(maximum(sum(x1_t * x1_t, dim=self.dim), self.eps))
        norm_x2 = sqrt(maximum(sum(x2_t * x2_t, dim=self.dim), self.eps))
        return _wrap(dot_product / (norm_x1 * norm_x2))


class PairwiseDistance(Module):
    """Computes the pairwise distance."""

    def __init__(
        self, p: float = 2.0, eps: float = 1e-06, keepdim: bool = False
    ) -> None:
        """Initializes PairwiseDistance."""
        super().__init__()
        self.p = p
        self.eps = eps
        self.keepdim = keepdim

    def forward(self, x1, x2):
        """Forward pass."""
        from zero_torch.tensor import _to_tensor, _wrap

        x1_t = _to_tensor(x1)
        x2_t = _to_tensor(x2)
        diff = abs(x1_t - x2_t)

        # Add small eps inside the power base to avoid NaN grads near zero
        eps = self.eps
        if self.p == 1.0:
            res = sum(diff, dim=-1, keepdims=self.keepdim)  # pragma: no cover
        elif self.p == 2.0:
            res = sqrt(sum(diff * diff, dim=-1, keepdims=self.keepdim) + eps)
        else:
            res = power(  # pragma: no cover
                sum(power(diff, self.p), dim=-1, keepdims=self.keepdim) + eps,
                1.0 / self.p,
            )

        return _wrap(res)
