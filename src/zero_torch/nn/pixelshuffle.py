"""PixelShuffle modules."""

from .module import Module


class PixelShuffle(Module):
    """Rearrange elements in a tensor according to an upscaling factor."""

    def __init__(self, upscale_factor: int) -> None:
        super().__init__()
        self.upscale_factor = upscale_factor

    def forward(self, input):
        import ml_switcheroo.core.errors

        raise ml_switcheroo.core.errors.UnimplementedMathError


class PixelUnshuffle(Module):
    """Reverse the PixelShuffle operation."""

    def __init__(self, downscale_factor: int) -> None:
        super().__init__()
        self.downscale_factor = downscale_factor

    def forward(self, input):
        import ml_switcheroo.core.errors

        raise ml_switcheroo.core.errors.UnimplementedMathError
