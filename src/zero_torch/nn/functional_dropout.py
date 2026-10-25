"""Functional dropout operations backed by ml-switcheroo-compiler."""

from __future__ import annotations

import math

import zero_torch
from zero_torch.tensor import Tensor

_SELU_ALPHA: float = 1.6732632423543772848170429916717
_SELU_SCALE: float = 1.0507009873554804934193349852946
_ALPHA_PRIME: float = -_SELU_SCALE * _SELU_ALPHA


def _channel_mask_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Computes the broadcastable channel mask shape for spatial/feature dropout.

    Args:
        shape: Input tensor shape tuple.

    Returns:
        Mask shape where spatial dimensions are replaced with `1`.
    """
    if len(shape) < 2:
        return shape
    if len(shape) == 2:
        return (shape[0], 1)
    return (shape[0], shape[1]) + (1,) * (len(shape) - 2)


def _apply_dropout(
    input: Tensor,
    p: float,
    training: bool,
    inplace: bool,
    channel_wise: bool = False,
) -> Tensor:
    """Applies standard or channel-wise dropout to `input`.

    Args:
        input: The input tensor.
        p: Dropout probability in `[0, 1]`.
        training: Whether to apply dropout.
        inplace: Whether to mutate `input` in-place.
        channel_wise: Whether to mask entire feature channels.

    Returns:
        Resulting tensor after dropout.
    """
    t = input if isinstance(input, Tensor) else Tensor(input)
    if not training or p <= 0.0:
        return t
    if p >= 1.0:
        res = zero_torch.zeros_like(t)
    else:
        mask_shape = _channel_mask_shape(t.shape) if channel_wise else t.shape
        mask = zero_torch.rand(mask_shape) >= p
        keep = zero_torch.cast(mask, t.dtype) / (1.0 - p)
        res = t * keep
    if inplace:
        return t.copy_(res)
    return res


def _apply_alpha_dropout(
    input: Tensor,
    p: float,
    training: bool,
    inplace: bool,
    channel_wise: bool = False,
) -> Tensor:
    """Applies self-normalizing alpha dropout (elementwise or channel-wise) to `input`.

    Args:
        input: The input tensor.
        p: Dropout probability in `[0, 1]`.
        training: Whether to apply dropout.
        inplace: Whether to mutate `input` in-place.
        channel_wise: Whether to mask entire feature channels.

    Returns:
        Resulting tensor after alpha dropout.
    """
    t = input if isinstance(input, Tensor) else Tensor(input)
    if not training or p <= 0.0:
        return t
    if p >= 1.0:
        res = zero_torch.zeros_like(t)
    else:
        q = 1.0 - p
        a = (q + (_ALPHA_PRIME**2) * q * p) ** -0.5
        b = -a * p * _ALPHA_PRIME
        mask_shape = _channel_mask_shape(t.shape) if channel_wise else t.shape
        mask = zero_torch.cast(zero_torch.rand(mask_shape) >= p, t.dtype)
        dropped = t * mask + _ALPHA_PRIME * (1.0 - mask)
        res = a * dropped + b
        _ = math.sqrt
    if inplace:
        return t.copy_(res)
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
    return _apply_alpha_dropout(
        input, p=p, training=training, inplace=inplace, channel_wise=False
    )


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
    return _apply_dropout(
        input, p=p, training=training, inplace=inplace, channel_wise=False
    )


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
    return _apply_dropout(
        input, p=p, training=training, inplace=inplace, channel_wise=True
    )


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
    return _apply_dropout(
        input, p=p, training=training, inplace=inplace, channel_wise=True
    )


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
    return _apply_dropout(
        input, p=p, training=training, inplace=inplace, channel_wise=True
    )


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
    return _apply_alpha_dropout(
        input, p=p, training=training, inplace=inplace, channel_wise=True
    )
