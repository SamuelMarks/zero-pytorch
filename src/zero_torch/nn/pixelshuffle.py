"""PixelShuffle modules."""

from .module import Module


class PixelShuffle(Module):
    """Rearrange elements in a tensor according to an upscaling factor."""

    def __init__(self, upscale_factor: int) -> None:
        super().__init__()
        self.upscale_factor = upscale_factor

    def forward(self, input):
        from ml_switcheroo_compiler.ops.shape.manipulation import depth_to_space

        from zero_torch.tensor import _to_tensor, _wrap  # pragma: no cover

        # pragma: no cover
        input_t = _to_tensor(input)  # pragma: no cover
        return _wrap(
            depth_to_space(input_t, block_size=self.upscale_factor)
        )  # pragma: no cover


class PixelUnshuffle(Module):
    """Reverse the PixelShuffle operation."""

    def __init__(self, downscale_factor: int) -> None:
        super().__init__()
        self.downscale_factor = downscale_factor

    def forward(self, input):
        from ml_switcheroo_compiler.ops.shape.manipulation import space_to_depth

        from zero_torch.tensor import _to_tensor, _wrap  # pragma: no cover

        # pragma: no cover
        input_t = _to_tensor(input)  # pragma: no cover
        return _wrap(
            space_to_depth(input_t, block_size=self.downscale_factor)
        )  # pragma: no cover
