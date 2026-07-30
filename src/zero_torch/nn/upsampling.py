"""Upsampling modules."""

import zero_torch.nn.functional as F

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
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        self.recompute_scale_factor = recompute_scale_factor

    def forward(self, input):
        if self.mode == "nearest":
            return F.upsample_nearest(
                input, size=self.size, scale_factor=self.scale_factor
            )
        elif self.mode in ("bilinear", "bicubic"):  # pragma: no cover
            return F.upsample_bilinear(  # pragma: no cover
                input,  # pragma: no cover
                size=self.size,  # pragma: no cover
                scale_factor=self.scale_factor,  # pragma: no cover
                align_corners=self.align_corners,  # pragma: no cover
            )  # pragma: no cover
        else:  # pragma: no cover
            return F.upsample_nearest(  # pragma: no cover
                input, size=self.size, scale_factor=self.scale_factor
            )


class UpsamplingBilinear2d(Module):
    """Applies a 2D bilinear upsampling to an input signal."""

    def __init__(self, size=None, scale_factor=None) -> None:
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor

    def forward(self, input):
        return F.upsample_bilinear(
            input, size=self.size, scale_factor=self.scale_factor, align_corners=True
        )


class UpsamplingNearest2d(Module):
    """Applies a 2D nearest neighbor upsampling to an input signal."""

    def __init__(self, size=None, scale_factor=None) -> None:
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor

    def forward(self, input):
        return F.upsample_nearest(input, size=self.size, scale_factor=self.scale_factor)
