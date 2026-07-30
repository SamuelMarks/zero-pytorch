"""Shuffle modules."""

from .module import Module


class ChannelShuffle(Module):
    """Divides and rearranges the channels in a tensor."""

    def __init__(self, groups: int) -> None:
        """Initializes ChannelShuffle."""
        super().__init__()
        self.groups = groups

    def forward(self, input):
        """Forward pass."""
        b, c, *dims = input.shape
        channels_per_group = c // self.groups

        # Reshape to (b, groups, channels_per_group, *dims)
        view_shape = (b, self.groups, channels_per_group) + tuple(dims)
        input = input.view(*view_shape)

        # Transpose groups and channels_per_group
        input = input.transpose(1, 2).contiguous()

        # Flatten back to original shape
        out_shape = (b, -1) + tuple(dims)  # pragma: no cover
        return input.view(*out_shape)  # pragma: no cover
