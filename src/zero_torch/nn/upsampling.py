"""Upsampling modules."""

from .module import Module


class Upsample(Module):
    """Upsamples a given multi-channel 1D (temporal), 2D (spatial) or 3D (volumetric) data."""

    def __init__(
        self,
        size=None,
        scale_factor=None,
        mode="nearest",
        align_corners=None,
        recompute_scale_factor=None,
    ) -> None:
        super().__init__()

    def forward(self, input):
        import ml_switcheroo_compiler.core.errors

        raise ml_switcheroo_compiler.core.errors.UnimplementedMathError


class UpsamplingBilinear2d(Module):
    """Applies a 2D bilinear upsampling to an input signal."""

    def __init__(self, size=None, scale_factor=None) -> None:
        super().__init__()

    def forward(self, input):
        import ml_switcheroo_compiler.core.errors

        raise ml_switcheroo_compiler.core.errors.UnimplementedMathError


class UpsamplingNearest2d(Module):
    """Applies a 2D nearest neighbor upsampling to an input signal."""

    def __init__(self, size=None, scale_factor=None) -> None:
        super().__init__()

    def forward(self, input):
        import ml_switcheroo_compiler.core.errors

        raise ml_switcheroo_compiler.core.errors.UnimplementedMathError
