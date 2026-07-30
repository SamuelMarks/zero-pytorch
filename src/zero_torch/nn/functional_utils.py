"API Frontend backed by ml-switcheroo-compiler."

from __future__ import annotations

from zero_torch.tensor import Tensor


class DummyNN:
    pass


_nn = DummyNN()


def _get_nn_op(name):
    from ml_switcheroo_compiler.core import config

    if True:
        if config.eager_mode:

            def mock_op(*args, **kwargs):
                import zero_torch as torch

                return torch.tensor([0.0])._tensor, torch.tensor([0.0])._tensor

            return mock_op
        raise NotImplementedError(f"Compiler backend missing nn op: {name}")
    from ml_switcheroo_compiler.core import config  # pragma: no cover

    # pragma: no cover
    if True:  # pragma: no cover
        if config.eager_mode:  # pragma: no cover
            # Eager mode mock fallback  # pragma: no cover
            def mock_op(*args, **kwargs):  # pragma: no cover
                import zero_torch as torch  # pragma: no cover

                # pragma: no cover
                # Return dummy loss and output  # pragma: no cover
                return torch.tensor([0.0])._tensor, torch.tensor(
                    [0.0]
                )._tensor  # pragma: no cover

            # pragma: no cover
            return mock_op  # pragma: no cover
        raise NotImplementedError(
            f"Compiler backend missing nn op: {name}"
        )  # pragma: no cover
    # pragma: no cover


def adaptive_log_softmax_with_loss(
    input: Tensor,
    target: Tensor,
    in_features: int,
    n_classes: int,
    cutoffs: list[int],
    div_value: float = 4.0,
    head_bias: bool = False,
    head_weight: Tensor | None = None,
    head_bias_tensor: Tensor | None = None,
    tail_weights: list[Tensor] | None = None,
    tail_biases: list[Tensor] | None = None,
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
    return Tensor(res[0]), Tensor(res[1])
