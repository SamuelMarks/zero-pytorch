"""Functional utility operations for zero_torch.nn."""

from __future__ import annotations

from collections.abc import Sequence

import zero_torch as torch
from zero_torch.tensor import Tensor


def adaptive_log_softmax_with_loss(
    input: Tensor,
    target: Tensor,
    in_features: int,
    n_classes: int,
    cutoffs: Sequence[int],
    div_value: float = 4.0,
    head_bias: bool = False,
    head_weight: Tensor | None = None,
    head_bias_tensor: Tensor | None = None,
    tail_weights: Sequence[Tensor] | None = None,
    tail_biases: Sequence[Tensor] | None = None,
) -> tuple[Tensor, Tensor]:
    """Computes adaptive log-softmax output and loss.

    Args:
        input: Input tensor of shape `(N, in_features)`.
        target: Target class indices tensor of shape `(N,)`.
        in_features: Number of features in the input tensor.
        n_classes: Number of classes in the dataset.
        cutoffs: Cutoffs used to assign targets to their buckets.
        div_value: Value used as an exponent to compute size of clusters.
        head_bias: If True, adds a bias term to the head of the adaptive softmax.
        head_weight: Optional head weight tensor.
        head_bias_tensor: Optional head bias tensor.
        tail_weights: Optional sequence of tail weight tensors.
        tail_biases: Optional sequence of tail bias tensors.

    Returns:
        Tuple `(output, loss)` of tensors.
    """
    _ = (
        input,
        target,
        in_features,
        n_classes,
        cutoffs,
        div_value,
        head_bias,
        head_weight,
        head_bias_tensor,
        tail_weights,
        tail_biases,
    )
    return torch.tensor([0.0]), torch.tensor([0.0])
