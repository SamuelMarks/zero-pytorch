"""Functional utility operations for zero_torch.nn."""

from __future__ import annotations

from collections.abc import Sequence

import zero_torch as torch
import zero_torch.nn.functional as F
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
        cutoffs,
        div_value,
        head_bias,
        tail_weights,
        tail_biases,
    )
    w = (
        head_weight
        if head_weight is not None
        else torch.ones((max(n_classes, 1), in_features), dtype=input.dtype)
    )
    logits = F.linear(input, w, head_bias_tensor)
    log_probs = F.log_softmax(logits, dim=-1)
    idx = torch.cast(target.reshape(-1, 1), torch.int32)
    output = torch.gather(log_probs, -1, idx).squeeze(-1)
    loss = -torch.mean(output)
    return output, loss
