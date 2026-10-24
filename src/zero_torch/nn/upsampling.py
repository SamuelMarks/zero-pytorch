"""Upsampling modules."""

from __future__ import annotations

import zero_torch.nn.functional as F
from zero_torch._types import OptSizeNd
from zero_torch.tensor import Tensor

from .module import Module


class Upsample(Module):
    """Upsamples a given multi-channel 1D, 2D, or 3D data tensor."""

    def __init__(
        self,
        size: OptSizeNd = None,
        scale_factor: float | tuple[float, ...] | None = None,
        mode: str = "nearest",
        align_corners: bool | None = None,
        recompute_scale_factor: bool | None = None,
    ) -> None:
        """Initializes Upsample.

        Args:
            size: Output spatial sizes.
            scale_factor: Multiplier for spatial size.
            mode: Upsampling algorithm (`'nearest'`, `'linear'`, `'bilinear'`, `'bicubic'`, `'trilinear'`).
            align_corners: If True, corner pixels of input and output tensors are aligned.
            recompute_scale_factor: Whether to recompute the scale_factor for use in interpolation.
        """
        super().__init__()
        self.size: OptSizeNd = size
        self.scale_factor: float | tuple[float, ...] | None = scale_factor
        self.mode: str = mode
        self.align_corners: bool | None = align_corners
        self.recompute_scale_factor: bool | None = recompute_scale_factor

    def forward(self, input: Tensor) -> Tensor:
        """Upsamples `input` according to configured size/scale_factor and mode.

        Args:
            input: Input tensor.

        Returns:
            Upsampled tensor.
        """
        if self.mode in ("bilinear", "bicubic"):
            return F.upsample_bilinear(
                input,
                size=self.size,
                scale_factor=self.scale_factor,
                align_corners=bool(self.align_corners),
            )
        return F.upsample_nearest(input, size=self.size, scale_factor=self.scale_factor)


class UpsamplingBilinear2d(Module):
    """Applies a 2D bilinear upsampling to an input signal."""

    def __init__(
        self,
        size: OptSizeNd = None,
        scale_factor: float | tuple[float, ...] | None = None,
    ) -> None:
        """Initializes UpsamplingBilinear2d.

        Args:
            size: Output spatial sizes.
            scale_factor: Multiplier for spatial size.
        """
        super().__init__()
        self.size: OptSizeNd = size
        self.scale_factor: float | tuple[float, ...] | None = scale_factor

    def forward(self, input: Tensor) -> Tensor:
        """Applies 2D bilinear upsampling to `input`.

        Args:
            input: 4D input tensor `(N, C, H, W)`.

        Returns:
            Upsampled 4D tensor.
        """
        return F.upsample_bilinear(
            input, size=self.size, scale_factor=self.scale_factor, align_corners=True
        )


class UpsamplingNearest2d(Module):
    """Applies a 2D nearest-neighbor upsampling to an input signal."""

    def __init__(
        self,
        size: OptSizeNd = None,
        scale_factor: float | tuple[float, ...] | None = None,
    ) -> None:
        """Initializes UpsamplingNearest2d.

        Args:
            size: Output spatial sizes.
            scale_factor: Multiplier for spatial size.
        """
        super().__init__()
        self.size: OptSizeNd = size
        self.scale_factor: float | tuple[float, ...] | None = scale_factor

    def forward(self, input: Tensor) -> Tensor:
        """Applies 2D nearest-neighbor upsampling to `input`.

        Args:
            input: 4D input tensor `(N, C, H, W)`.

        Returns:
            Upsampled 4D tensor.
        """
        return F.upsample_nearest(input, size=self.size, scale_factor=self.scale_factor)
