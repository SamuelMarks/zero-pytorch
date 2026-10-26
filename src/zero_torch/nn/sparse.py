"""Sparse modules."""

from __future__ import annotations

import zero_torch
from zero_torch.nn.init import normal_
from zero_torch.tensor import Tensor, _flatten_nested

from .module import Module, Parameter


def _zero_padding_row(weight: Parameter, padding_idx: int | None) -> None:
    """Zeroes out row `padding_idx` of `weight` in-place when `padding_idx` is specified.

    Args:
        weight: 2D embedding weight parameter.
        padding_idx: Optional padding index.
    """
    if padding_idx is not None:
        rows = weight.tolist()
        assert isinstance(rows, list)
        norm_idx = padding_idx if padding_idx >= 0 else len(rows) + padding_idx
        rows[norm_idx] = [0.0] * weight.shape[1]
        weight.copy_(Tensor(rows, dtype=weight.dtype))


class Embedding(Module):
    """A simple lookup table that stores embeddings of a fixed dictionary and size."""

    __constants__: list[str] = [
        "num_embeddings",
        "embedding_dim",
        "padding_idx",
        "max_norm",
        "norm_type",
        "scale_grad_by_freq",
        "sparse",
    ]
    weight: Parameter | None = None
    _freeze: bool = False

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
        _freeze: bool = False,
        device: str | None = None,
        dtype: str | None = None,
        weight: Tensor | None = None,
        freeze: bool = False,
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
            _freeze: If True, the tensor does not get updated in the learning process.
            device: Desired device of the weight parameter.
            dtype: Desired data type of the weight parameter.
            weight: Optional alias for `_weight`.
            freeze: Optional alias for `_freeze`.
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
        self._freeze = _freeze or freeze
        eff_weight = _weight if _weight is not None else weight
        if eff_weight is None:
            self.weight = Parameter(
                zero_torch.empty((num_embeddings, embedding_dim)),
                requires_grad=not self._freeze,
            )
            self.reset_parameters()
        else:
            self.weight = Parameter(eff_weight, requires_grad=not self._freeze)
        self.register_parameter("weight", self.weight)

    def reset_parameters(self) -> None:
        """Initializes `self.weight` from $N(0, 1)$ and zeroes out `padding_idx` if set."""
        assert self.weight is not None
        normal_(self.weight)
        _zero_padding_row(self.weight, self.padding_idx)

    def forward(self, input: Tensor) -> Tensor:
        """Looks up embedding vectors for the given input indices.

        Args:
            input: Tensor containing indices into the embedding matrix.

        Returns:
            Embedded tensor of shape (*input.shape, embedding_dim).
        """
        assert self.weight is not None
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

    __constants__: list[str] = [
        "num_embeddings",
        "embedding_dim",
        "max_norm",
        "norm_type",
        "scale_grad_by_freq",
        "mode",
        "sparse",
        "include_last_offset",
        "padding_idx",
    ]
    weight: Parameter | None = None

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
        weight: Tensor | None = None,
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
            weight: Optional alias for `_weight`.
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
        eff_weight = _weight if _weight is not None else weight
        if eff_weight is None:
            self.weight = Parameter(zero_torch.empty((num_embeddings, embedding_dim)))
            self.reset_parameters()
        else:
            self.weight = Parameter(eff_weight)
        self.register_parameter("weight", self.weight)

    def reset_parameters(self) -> None:
        """Initializes `self.weight` from $N(0, 1)$ and zeroes out `padding_idx` if set."""
        assert self.weight is not None
        normal_(self.weight)
        _zero_padding_row(self.weight, self.padding_idx)

    def _reduce_bag(self, bag_emb: Tensor, axis: int) -> Tensor:
        """Reduces `bag_emb` along `axis` according to `self.mode`.

        Args:
            bag_emb: Looked-up embedding tensor.
            axis: Reduction axis.

        Returns:
            Reduced bag tensor.
        """
        if self.mode == "sum":
            return zero_torch.sum(bag_emb, axis=axis)
        if self.mode == "max":
            return zero_torch.max(bag_emb, axis=axis)
        return zero_torch.mean(bag_emb, axis=axis)

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
        assert self.weight is not None
        embedded = zero_torch.nn.functional.embedding(
            input,
            self.weight,
            padding_idx=self.padding_idx,
            max_norm=self.max_norm,
            norm_type=self.norm_type,
            scale_grad_by_freq=self.scale_grad_by_freq,
            sparse=self.sparse,
        )
        if per_sample_weights is not None:
            embedded = embedded * zero_torch.unsqueeze(per_sample_weights, -1)
        if len(embedded.shape) >= 3:
            return self._reduce_bag(embedded, axis=1)
        if offsets is not None and len(input.shape) == 1:
            off_vals = [
                int(float(x))
                for x in _flatten_nested(offsets.tolist())
                if isinstance(x, (int, float))
            ]
            bounds = (
                off_vals if self.include_last_offset else off_vals + [input.shape[0]]
            )
            bag_rows: list[Tensor] = []
            for b_i in range(len(bounds) - 1):
                start_i = bounds[b_i]
                end_i = bounds[b_i + 1]
                if end_i > start_i:
                    bag_slice = embedded[start_i:end_i]
                    bag_rows.append(self._reduce_bag(bag_slice, axis=0))
                else:
                    bag_rows.append(
                        zero_torch.zeros((self.embedding_dim,), dtype=embedded.dtype)
                    )
            return zero_torch.stack(bag_rows, dim=0)
        return zero_torch.unsqueeze(self._reduce_bag(embedded, axis=0), 0)
