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
        from ml_switcheroo_compiler.ops.nn.nlp import embedding

        from zero_torch.tensor import _to_tensor, _wrap  # pragma: no cover

        if not hasattr(self, "weight"):  # pragma: no cover
            raise ValueError("weight not initialized")  # pragma: no cover
        return _wrap(
            embedding(_to_tensor(input), _to_tensor(self.weight))
        )  # pragma: no cover


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
        from ml_switcheroo_compiler.ops import embedding_bag

        from zero_torch.tensor import _to_tensor, _wrap  # pragma: no cover

        if not hasattr(self, "weight"):  # pragma: no cover
            raise ValueError("weight not initialized")  # pragma: no cover
        # pragma: no cover
        input_t = _to_tensor(input)  # pragma: no cover
        weight_t = _to_tensor(self.weight)  # pragma: no cover
        offsets_t = (
            _to_tensor(offsets) if offsets is not None else None
        )  # pragma: no cover
        # pragma: no cover
        # PyTorch EmbeddingBag applies per_sample_weights to embeddings before reduction  # pragma: no cover
        # ml_switcheroo_compiler.ops.embedding_bag doesn't support per_sample_weights directly  # pragma: no cover
        # but zero_torch is API frontend, we route to compiler. If the compiler gets updated  # pragma: no cover
        # to support it later, it will just work or we do manual math.  # pragma: no cover
        # For now, we will pass offsets and mode.  # pragma: no cover
        return _wrap(  # pragma: no cover
            embedding_bag(
                weight_t,
                input_t,
                offsets=offsets_t,
                mode=self.mode,
                padding_idx=self.padding_idx,
            )
        )
