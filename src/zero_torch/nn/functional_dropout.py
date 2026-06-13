"API Frontend backed by ml-switcheroo-compiler."

from zero_torch.tensor import Tensor
import ml_switcheroo_compiler.nn as _nn


def _get_nn_op(name):
    import ml_switcheroo_compiler.core.errors

    if not hasattr(_nn, name):
        raise ml_switcheroo_compiler.core.errors.UnimplementedMathError()
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
    return Tensor(res) if res is not None else None
