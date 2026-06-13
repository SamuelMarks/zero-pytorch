"API Frontend backed by ml-switcheroo-compiler."

from typing import Optional
from zero_torch.tensor import Tensor
import ml_switcheroo_compiler.nn as _nn


def _get_nn_op(name):
    import ml_switcheroo_compiler.core.errors

    if not hasattr(_nn, name):
        raise ml_switcheroo_compiler.core.errors.UnimplementedMathError()
    return getattr(_nn, name)


def adaptive_log_softmax_with_loss(
    input: Tensor,
    target: Tensor,
    in_features: int,
    n_classes: int,
    cutoffs: list[int],
    div_value: float = 4.0,
    head_bias: bool = False,
    head_weight: Optional[Tensor] = None,
    head_bias_tensor: Optional[Tensor] = None,
    tail_weights: Optional[list[Tensor]] = None,
    tail_biases: Optional[list[Tensor]] = None,
) -> tuple[Tensor, Tensor]:
    """adaptive_log_softmax_with_loss."""
    input = Tensor(input) if not isinstance(input, Tensor) else input
    target = Tensor(target) if not isinstance(target, Tensor) else target

    hw = head_weight._tensor if isinstance(head_weight, Tensor) else head_weight
    hb = (
        head_bias_tensor._tensor
        if isinstance(head_bias_tensor, Tensor)
        else head_bias_tensor
    )
    tw = (
        [t._tensor if isinstance(t, Tensor) else t for t in tail_weights]
        if tail_weights is not None
        else None
    )
    tb = (
        [t._tensor if isinstance(t, Tensor) else t for t in tail_biases]
        if tail_biases is not None
        else None
    )

    res = _get_nn_op("adaptive_log_softmax_with_loss")(
        input._tensor,
        target._tensor,
        in_features=in_features,
        n_classes=n_classes,
        cutoffs=cutoffs,
        div_value=div_value,
        head_bias=head_bias,
        head_weight=hw,
        head_bias_tensor=hb,
        tail_weights=tw,
        tail_biases=tb,
    )
    return Tensor(res[0]) if res[0] is not None else None, Tensor(res[1]) if res[
        1
    ] is not None else None
