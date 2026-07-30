"API Frontend backed by ml-switcheroo-compiler."

import ml_switcheroo_compiler.ops.nn.dropout as _nn

from zero_torch.tensor import Tensor


def _get_nn_op(name):
    from ml_switcheroo_compiler.core import config

    if not hasattr(_nn, name):
        if config.eager_mode:
            # Eager mode mock fallback
            def mock_op(input, p=0.5, training=True, **kwargs):
                if not training or p == 0:
                    return input  # pragma: no cover
                # basic dropout using python random
                import random

                import zero_torch as torch

                def _dropout_nested(data, prob):
                    if isinstance(data, (list, tuple)):
                        return [
                            _dropout_nested(x, prob) for x in data
                        ]  # pragma: no cover
                    return data / (1 - prob) if random.random() > prob else 0.0

                arr = input.tolist() if hasattr(input, "tolist") else input
                res = _dropout_nested(arr, p)
                return torch.tensor(res, dtype=getattr(input, "dtype", None))._tensor

            return mock_op
        raise NotImplementedError(f"Compiler backend missing nn op: {name}")
    return getattr(_nn, name)


def alpha_dropout(
    input: Tensor, p: float = 0.5, training: bool = False, inplace: bool = False
) -> Tensor:
    """Applies alpha_dropout.

    Args:
        input (Tensor): The input tensor.
        p (float, optional): The dropout probability. Defaults to 0.5.
        training (bool, optional): Apply dropout if True. Defaults to False.
        inplace (bool, optional): If set to True, will do this operation in-place. Defaults to False.

    Returns:
        Tensor: The result.
    """
    input = Tensor(input) if not isinstance(input, Tensor) else input
    res = _get_nn_op("alpha_dropout")(input._tensor, p=p, training=training)
    if inplace:
        if res is not None:
            input._tensor = res
        return input
    return Tensor(res) if res is not None else input


def dropout(
    input: Tensor, p: float = 0.5, training: bool = True, inplace: bool = False
) -> Tensor:
    """Applies dropout."""
    input = Tensor(input) if not isinstance(input, Tensor) else input
    res = _get_nn_op("dropout")(input._tensor, p=p, training=training)
    if inplace:
        if res is not None:
            input._tensor = res
        return input
    return Tensor(res) if res is not None else input


def dropout1d(
    input: Tensor, p: float = 0.5, training: bool = True, inplace: bool = False
) -> Tensor:
    """Applies dropout1d."""
    input = Tensor(input) if not isinstance(input, Tensor) else input
    res = _get_nn_op("dropout1d")(input._tensor, p=p, training=training)
    if inplace:
        if res is not None:
            input._tensor = res
        return input
    return Tensor(res) if res is not None else input


def dropout2d(
    input: Tensor, p: float = 0.5, training: bool = True, inplace: bool = False
) -> Tensor:
    """Applies dropout2d."""
    input = Tensor(input) if not isinstance(input, Tensor) else input
    res = _get_nn_op("dropout2d")(input._tensor, p=p, training=training)
    if inplace:
        if res is not None:
            input._tensor = res
        return input
    return Tensor(res) if res is not None else input


def dropout3d(
    input: Tensor, p: float = 0.5, training: bool = True, inplace: bool = False
) -> Tensor:
    """Applies dropout3d."""
    input = Tensor(input) if not isinstance(input, Tensor) else input
    res = _get_nn_op("dropout3d")(input._tensor, p=p, training=training)
    if inplace:
        if res is not None:
            input._tensor = res
        return input
    return Tensor(res) if res is not None else input


def feature_alpha_dropout(
    input: Tensor, p: float = 0.5, training: bool = True, inplace: bool = False
) -> Tensor:
    """Applies feature_alpha_dropout."""
    input = Tensor(input) if not isinstance(input, Tensor) else input
    res = _get_nn_op("feature_alpha_dropout")(input._tensor, p=p, training=training)
    if inplace:
        if res is not None:
            input._tensor = res
        return input
    return Tensor(res) if res is not None else input
