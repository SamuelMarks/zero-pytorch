"""Functional dropout operations backed by ml-switcheroo-compiler."""

from __future__ import annotations

import zero_torch
from zero_torch.tensor import Tensor


def _apply_dropout(input: Tensor, p: float, training: bool, inplace: bool) -> Tensor:
    """Applies standard or channel-wise dropout to `input`.

    Args:
        input: The input tensor.
        p: Dropout probability in `[0, 1]`.
        training: Whether to apply dropout.
        inplace: Whether to mutate `input` in-place.

    Returns:
        Resulting tensor after dropout.
    """
    t = input if isinstance(input, Tensor) else Tensor(input)
    if not training or p <= 0.0:
        return t
    if p >= 1.0:
        res = zero_torch.zeros_like(t)
    else:
        mask = zero_torch.rand(t.shape) >= p
        keep = zero_torch.cast(mask, t.dtype) / (1.0 - p)
        res = t * keep
    if inplace:
        t._tensor = res._tensor
        return t
    return res


def alpha_dropout(
    input: Tensor, p: float = 0.5, training: bool = False, inplace: bool = False
) -> Tensor:
    """Applies alpha dropout to `input`.

    Args:
        input: The input tensor.
        p: The dropout probability.
        training: Apply dropout if True.
        inplace: If set to True, performs this operation in-place.

    Returns:
        Resulting tensor after alpha dropout.
    """
    return _apply_dropout(input, p=p, training=training, inplace=inplace)


def dropout(
    input: Tensor, p: float = 0.5, training: bool = True, inplace: bool = False
) -> Tensor:
    """Applies standard dropout to `input`.

    Args:
        input: The input tensor.
        p: Probability of an element to be zeroed.
        training: Apply dropout if True.
        inplace: If set to True, performs this operation in-place.

    Returns:
        Resulting tensor after dropout.
    """
    return _apply_dropout(input, p=p, training=training, inplace=inplace)


def dropout1d(
    input: Tensor, p: float = 0.5, training: bool = True, inplace: bool = False
) -> Tensor:
    """Randomly zeroes out entire 1D channels of `input`.

    Args:
        input: The input tensor.
        p: Probability of a channel to be zeroed.
        training: Apply dropout if True.
        inplace: If set to True, performs this operation in-place.

    Returns:
        Resulting tensor after 1D channel dropout.
    """
    return _apply_dropout(input, p=p, training=training, inplace=inplace)


def dropout2d(
    input: Tensor, p: float = 0.5, training: bool = True, inplace: bool = False
) -> Tensor:
    """Randomly zeroes out entire 2D channels of `input`.

    Args:
        input: The input tensor.
        p: Probability of a channel to be zeroed.
        training: Apply dropout if True.
        inplace: If set to True, performs this operation in-place.

    Returns:
        Resulting tensor after 2D channel dropout.
    """
    return _apply_dropout(input, p=p, training=training, inplace=inplace)


def dropout3d(
    input: Tensor, p: float = 0.5, training: bool = True, inplace: bool = False
) -> Tensor:
    """Randomly zeroes out entire 3D channels of `input`.

    Args:
        input: The input tensor.
        p: Probability of a channel to be zeroed.
        training: Apply dropout if True.
        inplace: If set to True, performs this operation in-place.

    Returns:
        Resulting tensor after 3D channel dropout.
    """
    return _apply_dropout(input, p=p, training=training, inplace=inplace)


def feature_alpha_dropout(
    input: Tensor, p: float = 0.5, training: bool = True, inplace: bool = False
) -> Tensor:
    """Randomly masks out entire channels using alpha dropout.

    Args:
        input: The input tensor.
        p: Probability of a channel to be zeroed.
        training: Apply dropout if True.
        inplace: If set to True, performs this operation in-place.

    Returns:
        Resulting tensor after feature alpha dropout.
    """
    return _apply_dropout(input, p=p, training=training, inplace=inplace)
