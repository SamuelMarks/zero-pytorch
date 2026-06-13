"""Sparse modules."""

from .module import Module


class Embedding(Module):
    """A simple lookup table that stores embeddings of a fixed dictionary and size."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx=None,
        max_norm=None,
        norm_type=2.0,
        scale_grad_by_freq=False,
        sparse=False,
        _weight=None,
        device=None,
        dtype=None,
    ) -> None:
        """Initializes Embedding."""
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse

    def forward(self, input):
        """Forward pass."""
        import ml_switcheroo_compiler.core.errors

        raise ml_switcheroo_compiler.core.errors.UnimplementedMathError


class EmbeddingBag(Module):
    """Computes sums or means of 'bags' of embeddings."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        max_norm=None,
        norm_type=2.0,
        scale_grad_by_freq=False,
        mode="mean",
        sparse=False,
        _weight=None,
        include_last_offset=False,
        padding_idx=None,
        device=None,
        dtype=None,
    ) -> None:
        """Initializes EmbeddingBag."""
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.mode = mode
        self.sparse = sparse
        self.include_last_offset = include_last_offset
        self.padding_idx = padding_idx

    def forward(self, input, offsets=None, per_sample_weights=None):
        """Forward pass."""
        import ml_switcheroo_compiler.core.errors

        raise ml_switcheroo_compiler.core.errors.UnimplementedMathError
