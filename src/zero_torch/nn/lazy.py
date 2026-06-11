"""Lazy modules."""

from .module import Module


class LazyBatchNorm1d(Module):
    """Applies LazyBatchNorm1d."""

    def __init__(
        self,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

    def forward(self, input):
        import ml_switcheroo.core.errors

        raise ml_switcheroo.core.errors.UnimplementedMathError


class LazyBatchNorm2d(Module):
    """Applies LazyBatchNorm2d."""

    def __init__(
        self,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

    def forward(self, input):
        import ml_switcheroo.core.errors

        raise ml_switcheroo.core.errors.UnimplementedMathError


class LazyBatchNorm3d(Module):
    """Applies LazyBatchNorm3d."""

    def __init__(
        self,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

    def forward(self, input):
        import ml_switcheroo.core.errors

        raise ml_switcheroo.core.errors.UnimplementedMathError


class LazyConv1d(Module):
    """Applies LazyConv1d."""

    def __init__(
        self,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.padding_mode = padding_mode

    def forward(self, input):
        import ml_switcheroo.core.errors

        raise ml_switcheroo.core.errors.UnimplementedMathError


class LazyConv2d(Module):
    """Applies LazyConv2d."""

    def __init__(
        self,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.padding_mode = padding_mode

    def forward(self, input):
        import ml_switcheroo.core.errors

        raise ml_switcheroo.core.errors.UnimplementedMathError


class LazyConv3d(Module):
    """Applies LazyConv3d."""

    def __init__(
        self,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.padding_mode = padding_mode

    def forward(self, input):
        import ml_switcheroo.core.errors

        raise ml_switcheroo.core.errors.UnimplementedMathError


class LazyConvTranspose1d(Module):
    """Applies LazyConvTranspose1d."""

    def __init__(
        self,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        groups: int = 1,
        bias: bool = True,
        dilation=1,
        padding_mode: str = "zeros",
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias
        self.dilation = dilation
        self.padding_mode = padding_mode

    def forward(self, input, output_size=None):
        import ml_switcheroo.core.errors

        raise ml_switcheroo.core.errors.UnimplementedMathError


class LazyConvTranspose2d(Module):
    """Applies LazyConvTranspose2d."""

    def __init__(
        self,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        groups: int = 1,
        bias: bool = True,
        dilation=1,
        padding_mode: str = "zeros",
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias
        self.dilation = dilation
        self.padding_mode = padding_mode

    def forward(self, input, output_size=None):
        import ml_switcheroo.core.errors

        raise ml_switcheroo.core.errors.UnimplementedMathError


class LazyConvTranspose3d(Module):
    """Applies LazyConvTranspose3d."""

    def __init__(
        self,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        groups: int = 1,
        bias: bool = True,
        dilation=1,
        padding_mode: str = "zeros",
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias
        self.dilation = dilation
        self.padding_mode = padding_mode

    def forward(self, input, output_size=None):
        import ml_switcheroo.core.errors

        raise ml_switcheroo.core.errors.UnimplementedMathError


class LazyInstanceNorm1d(Module):
    """Applies LazyInstanceNorm1d."""

    def __init__(
        self,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = False,
        track_running_stats: bool = False,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

    def forward(self, input):
        import ml_switcheroo.core.errors

        raise ml_switcheroo.core.errors.UnimplementedMathError


class LazyInstanceNorm2d(Module):
    """Applies LazyInstanceNorm2d."""

    def __init__(
        self,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = False,
        track_running_stats: bool = False,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

    def forward(self, input):
        import ml_switcheroo.core.errors

        raise ml_switcheroo.core.errors.UnimplementedMathError


class LazyInstanceNorm3d(Module):
    """Applies LazyInstanceNorm3d."""

    def __init__(
        self,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = False,
        track_running_stats: bool = False,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

    def forward(self, input):
        import ml_switcheroo.core.errors

        raise ml_switcheroo.core.errors.UnimplementedMathError


class LazyLinear(Module):
    """Applies LazyLinear."""

    def __init__(self, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.out_features = out_features
        self.bias = bias

    def forward(self, input):
        import ml_switcheroo.core.errors

        raise ml_switcheroo.core.errors.UnimplementedMathError
