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
        import ml_switcheroo.core.errors

        raise ml_switcheroo.core.errors.UnimplementedMathError
