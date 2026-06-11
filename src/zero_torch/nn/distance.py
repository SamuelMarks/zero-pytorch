"""Distance modules."""

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
        import ml_switcheroo.core.errors

        raise ml_switcheroo.core.errors.UnimplementedMathError


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
        import ml_switcheroo.core.errors

        raise ml_switcheroo.core.errors.UnimplementedMathError
