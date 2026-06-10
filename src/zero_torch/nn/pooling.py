"Pooling layers."

from typing import Any
from .module import Module
from zero_torch.tensor import Tensor


class AdaptiveAvgPool1d(Module):
    """Applies a 1D adaptive average pooling over an input signal composed of several input planes."""

    def __init__(self, output_size: Any, *args: Any, **kwargs: Any) -> None:
        """Initialize the module."""
        pass

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass."""
        pass


class AdaptiveAvgPool2d(Module):
    """Applies a 2D adaptive average pooling over an input signal composed of several input planes."""

    def __init__(self, output_size: Any, *args: Any, **kwargs: Any) -> None:
        """Initialize the module."""
        pass

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass."""
        pass


class AdaptiveAvgPool3d(Module):
    """Applies a 3D adaptive average pooling over an input signal composed of several input planes."""

    def __init__(self, output_size: Any, *args: Any, **kwargs: Any) -> None:
        """Initialize the module."""
        pass

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass."""
        pass


class AdaptiveMaxPool1d(Module):
    """Applies a 1D adaptive max pooling over an input signal composed of several input planes."""

    def __init__(
        self, output_size: Any, return_indices: bool = False, *args: Any, **kwargs: Any
    ) -> None:
        """Initialize."""
        pass

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass."""
        pass


class AdaptiveMaxPool2d(Module):
    """Applies a 2D adaptive max pooling over an input signal composed of several input planes."""

    def __init__(
        self, output_size: Any, return_indices: bool = False, *args: Any, **kwargs: Any
    ) -> None:
        """Initialize."""
        pass

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass."""
        pass


class AdaptiveMaxPool3d(Module):
    """Applies a 3D adaptive max pooling over an input signal composed of several input planes."""

    def __init__(
        self, output_size: Any, return_indices: bool = False, *args: Any, **kwargs: Any
    ) -> None:
        """Initialize."""
        pass

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass."""
        pass


class AvgPool1d(Module):
    """Applies a 1D average pooling over an input signal composed of several input planes."""

    def __init__(
        self,
        kernel_size: Any,
        stride: Any = None,
        padding: Any = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize."""
        pass

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass."""
        pass


class AvgPool2d(Module):
    """Applies a 2D average pooling over an input signal composed of several input planes."""

    def __init__(
        self,
        kernel_size: Any,
        stride: Any = None,
        padding: Any = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        divisor_override: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize."""
        pass

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass."""
        pass


class AvgPool3d(Module):
    """Applies a 3D average pooling over an input signal composed of several input planes."""

    def __init__(
        self,
        kernel_size: Any,
        stride: Any = None,
        padding: Any = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        divisor_override: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize."""
        pass

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass."""
        pass


class MaxPool1d:
    pass


class FractionalMaxPool2d:
    pass


class MaxPool2d:
    pass


class FractionalMaxPool3d:
    pass


class MaxPool3d:
    pass


class LPPool1d:
    pass


class LPPool2d:
    pass


class LPPool3d:
    pass


class MaxUnpool1d:
    pass


class MaxUnpool2d:
    pass


class MaxUnpool3d:
    pass
