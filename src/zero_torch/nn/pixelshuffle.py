"""PixelShuffle modules."""

from __future__ import annotations

import zero_torch
from zero_torch.tensor import Tensor

from .module import Module


class PixelShuffle(Module):
    """Rearranges elements in a tensor of shape `(*, C * r^2, H, W)` to `(*, C, H * r, W * r)`."""

    __constants__: list[str] = ["upscale_factor"]

    def __init__(self, upscale_factor: int) -> None:
        """Initializes PixelShuffle.

        Args:
            upscale_factor: Factor to increase spatial resolution by.
        """
        super().__init__()
        self.upscale_factor: int = upscale_factor

    def forward(self, input: Tensor) -> Tensor:
        """Applies pixel shuffle to the input tensor.

        Args:
            input: Input tensor of shape (*, C * r^2, H, W).

        Returns:
            Output tensor of shape (*, C, H * r, W * r).
        """
        r = self.upscale_factor
        shape = input.shape
        if len(shape) < 3:
            raise ValueError("PixelShuffle expects input with at least 3 dimensions")
        batch_dims = shape[:-3]
        c_in, h, w = shape[-3], shape[-2], shape[-1]
        c_out = c_in // (r * r)
        reshaped = zero_torch.reshape(input, batch_dims + (c_out, r, r, h, w))
        n_batch = len(batch_dims)
        perm = tuple(range(n_batch)) + (
            n_batch,
            n_batch + 3,
            n_batch + 1,
            n_batch + 4,
            n_batch + 2,
        )
        permuted = zero_torch.permute(reshaped, perm)
        return zero_torch.reshape(permuted, batch_dims + (c_out, h * r, w * r))


class PixelUnshuffle(Module):
    """Reverses the PixelShuffle operation by rearranging `(*, C, H * r, W * r)` to `(*, C * r^2, H, W)`."""

    __constants__: list[str] = ["downscale_factor"]

    def __init__(self, downscale_factor: int) -> None:
        """Initializes PixelUnshuffle.

        Args:
            downscale_factor: Factor to decrease spatial resolution by.
        """
        super().__init__()
        self.downscale_factor: int = downscale_factor

    def forward(self, input: Tensor) -> Tensor:
        """Applies pixel unshuffle to the input tensor.

        Args:
            input: Input tensor of shape (*, C, H * r, W * r).

        Returns:
            Output tensor of shape (*, C * r^2, H, W).
        """
        r = self.downscale_factor
        shape = input.shape
        if len(shape) < 3:
            raise ValueError("PixelUnshuffle expects input with at least 3 dimensions")
        batch_dims = shape[:-3]
        c_in, h_in, w_in = shape[-3], shape[-2], shape[-1]
        h_out = h_in // r
        w_out = w_in // r
        c_out = c_in * r * r
        reshaped = zero_torch.reshape(input, batch_dims + (c_in, h_out, r, w_out, r))
        n_batch = len(batch_dims)
        perm = tuple(range(n_batch)) + (
            n_batch,
            n_batch + 2,
            n_batch + 4,
            n_batch + 1,
            n_batch + 3,
        )
        permuted = zero_torch.permute(reshaped, perm)
        return zero_torch.reshape(permuted, batch_dims + (c_out, h_out, w_out))
