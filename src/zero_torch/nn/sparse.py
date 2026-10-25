"""Sparse modules."""

from __future__ import annotations

import zero_torch
from zero_torch.tensor import Tensor

from .module import Module, Parameter


class Embedding(Module):
    """A simple lookup table that stores embeddings of a fixed dictionary and size."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int | None = None,
        max_norm: float | None = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        _weight: Tensor | None = None,
        device: str | None = None,
        dtype: str | None = None,
    ) -> None:
        """Initializes Embedding.

        Args:
            num_embeddings: Size of the dictionary of embeddings.
            embedding_dim: The size of each embedding vector.
            padding_idx: If specified, the entries at padding_idx do not contribute to the gradient.
            max_norm: If given, each embedding vector with norm larger than max_norm is renormalized.
            norm_type: The p of the p-norm to compute for the max_norm option.
            scale_grad_by_freq: If given, this will scale gradients by the inverse of frequency of the words.
            sparse: If True, gradient w.r.t. weight matrix will be a sparse tensor.
            _weight: Optional pre-initialized weight tensor.
            device: Desired device of the weight parameter.
            dtype: Desired data type of the weight parameter.
        """
        super().__init__()
        self.num_embeddings: int = num_embeddings
        self.embedding_dim: int = embedding_dim
        self.padding_idx: int | None = padding_idx
        self.max_norm: float | None = max_norm
        self.norm_type: float = norm_type
        self.scale_grad_by_freq: bool = scale_grad_by_freq
        self.sparse: bool = sparse
        self.device: str | None = device
        self.dtype: str | None = dtype
        if _weight is None:
            self.weight: Parameter = Parameter(
                zero_torch.randn(num_embeddings, embedding_dim)
            )
        else:
            self.weight = Parameter(_weight)
        self.register_parameter("weight", self.weight)

    def forward(self, input: Tensor) -> Tensor:
        """Looks up embedding vectors for the given input indices.

        Args:
            input: Tensor containing indices into the embedding matrix.

        Returns:
            Embedded tensor of shape (*input.shape, embedding_dim).
        """
        return zero_torch.nn.functional.embedding(
            input,
            self.weight,
            padding_idx=self.padding_idx,
            max_norm=self.max_norm,
            norm_type=self.norm_type,
            scale_grad_by_freq=self.scale_grad_by_freq,
            sparse=self.sparse,
        )


class EmbeddingBag(Module):
    """Computes sums, means, or maxes of bags of embeddings."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        max_norm: float | None = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        mode: str = "mean",
        sparse: bool = False,
        _weight: Tensor | None = None,
        include_last_offset: bool = False,
        padding_idx: int | None = None,
        device: str | None = None,
        dtype: str | None = None,
    ) -> None:
        """Initializes EmbeddingBag.

        Args:
            num_embeddings: Size of the dictionary of embeddings.
            embedding_dim: The size of each embedding vector.
            max_norm: If given, each embedding vector with norm larger than max_norm is renormalized.
            norm_type: The p of the p-norm to compute for the max_norm option.
            scale_grad_by_freq: If given, this will scale gradients by the inverse of frequency of the words.
            mode: Reduction mode: 'sum', 'mean', or 'max'.
            sparse: If True, gradient w.r.t. weight matrix will be a sparse tensor.
            _weight: Optional pre-initialized weight tensor.
            include_last_offset: If True, offsets has one additional element where the last element is the size of input.
            padding_idx: If specified, the entries at padding_idx do not contribute to the gradient.
            device: Desired device of the weight parameter.
            dtype: Desired data type of the weight parameter.
        """
        super().__init__()
        self.num_embeddings: int = num_embeddings
        self.embedding_dim: int = embedding_dim
        self.max_norm: float | None = max_norm
        self.norm_type: float = norm_type
        self.scale_grad_by_freq: bool = scale_grad_by_freq
        self.mode: str = mode
        self.sparse: bool = sparse
        self.include_last_offset: bool = include_last_offset
        self.padding_idx: int | None = padding_idx
        self.device: str | None = device
        self.dtype: str | None = dtype
        if _weight is None:
            self.weight: Parameter = Parameter(
                zero_torch.randn(num_embeddings, embedding_dim)
            )
        else:
            self.weight = Parameter(_weight)
        self.register_parameter("weight", self.weight)

    def forward(
        self,
        input: Tensor,
        offsets: Tensor | None = None,
        per_sample_weights: Tensor | None = None,
    ) -> Tensor:
        """Computes bag reductions over embeddings looked up by input indices.

        Args:
            input: Tensor containing bags of indices into the embedding matrix.
            offsets: Only used when input is 1D. Determines the starting index position of each bag.
            per_sample_weights: Tensor of float/double weights, or None to indicate all weights should be taken to be 1.

        Returns:
            Aggregated embedding tensor of shape (num_bags, embedding_dim).
        """
        _ = offsets
        embedded = zero_torch.nn.functional.embedding(input, self.weight)
        if per_sample_weights is not None:
            embedded = embedded * zero_torch.unsqueeze(per_sample_weights, -1)
        if len(embedded.shape) >= 3:
            if self.mode == "sum":
                return zero_torch.sum(embedded, axis=1)
            if self.mode == "max":
                return zero_torch.max(embedded, axis=1)
            return zero_torch.mean(embedded, axis=1)
        if self.mode == "sum":
            return zero_torch.unsqueeze(zero_torch.sum(embedded, axis=0), 0)
        if self.mode == "max":
            return zero_torch.unsqueeze(zero_torch.max(embedded, axis=0), 0)
        return zero_torch.unsqueeze(zero_torch.mean(embedded, axis=0), 0)
