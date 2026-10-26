"""Shuffle modules."""

from __future__ import annotations

import zero_torch
from zero_torch.tensor import Tensor

from .module import Module


class ChannelShuffle(Module):
    """Divides and rearranges the channels in a tensor into `groups` groups."""

    __constants__: list[str] = ["groups"]

    def __init__(self, groups: int) -> None:
        """Initializes ChannelShuffle.

        Args:
            groups: Number of groups to divide channels into.
        """
        super().__init__()
        self.groups: int = groups

    def forward(self, input: Tensor) -> Tensor:
        """Applies channel shuffle to `input`.

        Args:
            input: Input tensor of shape `(N, C, *)`.

        Returns:
            Channel-shuffled tensor of the same shape as `input`.
        """
        b, c, *dims = input.shape
        channels_per_group = c // self.groups
        view_shape = (b, self.groups, channels_per_group) + tuple(dims)
        reshaped = input.view(*view_shape)
        transposed = zero_torch.transpose(reshaped, 1, 2).contiguous()
        out_shape = (b, c) + tuple(dims)
        return transposed.view(*out_shape)
