"Pooling layers."

from typing import Any
from .module import Module
from zero_torch.tensor import Tensor


class AdaptiveAvgPool1d(Module):
    """Applies a 1D adaptive average pooling over an input signal composed of several input planes."""

    def __init__(self, output_size: Any, *args: Any, **kwargs: Any) -> None:
        """Initialize the module.

        Args:
            output_size (Any): The target output size.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        pass

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass.

        Args:
            input (Tensor): The input tensor.

        Returns:
            Tensor: The pooled tensor.
        """
        pass


class AdaptiveAvgPool2d(Module):
    """Applies a 2D adaptive average pooling over an input signal composed of several input planes."""

    def __init__(self, output_size: Any, *args: Any, **kwargs: Any) -> None:
        """Initialize the module.

        Args:
            output_size (Any): The target output size.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        pass

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass.

        Args:
            input (Tensor): The input tensor.

        Returns:
            Tensor: The pooled tensor.
        """
        pass


class AdaptiveAvgPool3d(Module):
    """Applies a 3D adaptive average pooling over an input signal composed of several input planes."""

    def __init__(self, output_size: Any, *args: Any, **kwargs: Any) -> None:
        """Initialize the module.

        Args:
            output_size (Any): The target output size.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        pass

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass.

        Args:
            input (Tensor): The input tensor.

        Returns:
            Tensor: The pooled tensor.
        """
        pass


class AdaptiveMaxPool1d(Module):
    """Applies a 1D adaptive max pooling over an input signal composed of several input planes."""

    def __init__(
        self, output_size: Any, return_indices: bool = False, *args: Any, **kwargs: Any
    ) -> None:
        """Initialize the module.

        Args:
            output_size (Any): The target output size.
            return_indices (bool, optional): If True, will return the indices along with the outputs. Defaults to False.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        pass

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass.

        Args:
            input (Tensor): The input tensor.

        Returns:
            Tensor: The pooled tensor.
        """
        pass


class AdaptiveMaxPool2d(Module):
    """Applies a 2D adaptive max pooling over an input signal composed of several input planes."""

    def __init__(
        self, output_size: Any, return_indices: bool = False, *args: Any, **kwargs: Any
    ) -> None:
        """Initialize the module.

        Args:
            output_size (Any): The target output size.
            return_indices (bool, optional): If True, will return the indices along with the outputs. Defaults to False.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        pass

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass.

        Args:
            input (Tensor): The input tensor.

        Returns:
            Tensor: The pooled tensor.
        """
        pass


class AdaptiveMaxPool3d(Module):
    """Applies a 3D adaptive max pooling over an input signal composed of several input planes."""

    def __init__(
        self, output_size: Any, return_indices: bool = False, *args: Any, **kwargs: Any
    ) -> None:
        """Initialize the module.

        Args:
            output_size (Any): The target output size.
            return_indices (bool, optional): If True, will return the indices along with the outputs. Defaults to False.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        pass

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass.

        Args:
            input (Tensor): The input tensor.

        Returns:
            Tensor: The pooled tensor.
        """
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
        """Initialize the module.

        Args:
            kernel_size (Any): Size of the window to take an average over.
            stride (Any, optional): Stride of the window. Defaults to None.
            padding (Any, optional): Implicit zero padding to be added on both sides. Defaults to 0.
            ceil_mode (bool, optional): If True, will use ceil instead of floor to compute the output shape. Defaults to False.
            count_include_pad (bool, optional): If True, will include the zero-padding in the averaging calculation. Defaults to True.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        pass

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass.

        Args:
            input (Tensor): The input tensor.

        Returns:
            Tensor: The pooled tensor.
        """
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
        """Initialize the module.

        Args:
            kernel_size (Any): Size of the window to take an average over.
            stride (Any, optional): Stride of the window. Defaults to None.
            padding (Any, optional): Implicit zero padding to be added on both sides. Defaults to 0.
            ceil_mode (bool, optional): If True, will use ceil instead of floor. Defaults to False.
            count_include_pad (bool, optional): If True, will include the zero-padding in the averaging. Defaults to True.
            divisor_override (Any, optional): If specified, it will be used as divisor. Defaults to None.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        pass

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass.

        Args:
            input (Tensor): The input tensor.

        Returns:
            Tensor: The pooled tensor.
        """
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
        """Initialize the module.

        Args:
            kernel_size (Any): Size of the window to take an average over.
            stride (Any, optional): Stride of the window. Defaults to None.
            padding (Any, optional): Implicit zero padding to be added on both sides. Defaults to 0.
            ceil_mode (bool, optional): If True, will use ceil instead of floor. Defaults to False.
            count_include_pad (bool, optional): If True, will include the zero-padding in the averaging. Defaults to True.
            divisor_override (Any, optional): If specified, it will be used as divisor. Defaults to None.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        pass

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass.

        Args:
            input (Tensor): The input tensor.

        Returns:
            Tensor: The pooled tensor.
        """
        pass


class MaxPool1d(Module):
    """Applies a 1D max pooling over an input signal composed of several input planes."""

    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0) -> None:
        """Initialize MaxPool1d.

        Args:
            kernel_size (int): Size of the pooling window.
            stride (int, optional): Stride of the pooling window. Defaults to None.
            padding (int, optional): Implicit zero padding. Defaults to 0.
        """
        pass

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass.

        Args:
            input (Tensor): Input tensor.

        Returns:
            Tensor: Pooled tensor.
        """
        pass


class FractionalMaxPool2d(Module):
    """Applies a 2D fractional max pooling."""

    def __init__(self, kernel_size: int, output_size: int = None) -> None:
        """Initialize FractionalMaxPool2d.

        Args:
            kernel_size (int): Size of the pooling window.
            output_size (int, optional): Target output size. Defaults to None.
        """
        pass

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass.

        Args:
            input (Tensor): Input tensor.

        Returns:
            Tensor: Pooled tensor.
        """
        pass


class MaxPool2d(Module):
    """Applies a 2D max pooling."""

    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0) -> None:
        """Initialize MaxPool2d.

        Args:
            kernel_size (int): Size of the pooling window.
            stride (int, optional): Stride of the pooling window. Defaults to None.
            padding (int, optional): Implicit zero padding. Defaults to 0.
        """
        pass

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass.

        Args:
            input (Tensor): Input tensor.

        Returns:
            Tensor: Pooled tensor.
        """
        pass


class FractionalMaxPool3d(Module):
    """Applies a 3D fractional max pooling."""

    def __init__(self, kernel_size: int, output_size: int = None) -> None:
        """Initialize FractionalMaxPool3d.

        Args:
            kernel_size (int): Size of the pooling window.
            output_size (int, optional): Target output size. Defaults to None.
        """
        pass

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass.

        Args:
            input (Tensor): Input tensor.

        Returns:
            Tensor: Pooled tensor.
        """
        pass


class MaxPool3d(Module):
    """Applies a 3D max pooling."""

    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0) -> None:
        """Initialize MaxPool3d.

        Args:
            kernel_size (int): Size of the pooling window.
            stride (int, optional): Stride of the pooling window. Defaults to None.
            padding (int, optional): Implicit zero padding. Defaults to 0.
        """
        pass

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass.

        Args:
            input (Tensor): Input tensor.

        Returns:
            Tensor: Pooled tensor.
        """
        pass


class LPPool1d(Module):
    """Applies a 1D power-average pooling."""

    def __init__(self, norm_type: float, kernel_size: int, stride: int = None) -> None:
        """Initialize LPPool1d.

        Args:
            norm_type (float): Power for the pooling.
            kernel_size (int): Size of the pooling window.
            stride (int, optional): Stride of the pooling window. Defaults to None.
        """
        pass

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass.

        Args:
            input (Tensor): Input tensor.

        Returns:
            Tensor: Pooled tensor.
        """
        pass


class LPPool2d(Module):
    """Applies a 2D power-average pooling."""

    def __init__(self, norm_type: float, kernel_size: int, stride: int = None) -> None:
        """Initialize LPPool2d.

        Args:
            norm_type (float): Power for the pooling.
            kernel_size (int): Size of the pooling window.
            stride (int, optional): Stride of the pooling window. Defaults to None.
        """
        pass

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass.

        Args:
            input (Tensor): Input tensor.

        Returns:
            Tensor: Pooled tensor.
        """
        pass


class LPPool3d(Module):
    """Applies a 3D power-average pooling."""

    def __init__(self, norm_type: float, kernel_size: int, stride: int = None) -> None:
        """Initialize LPPool3d.

        Args:
            norm_type (float): Power for the pooling.
            kernel_size (int): Size of the pooling window.
            stride (int, optional): Stride of the pooling window. Defaults to None.
        """
        pass

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass.

        Args:
            input (Tensor): Input tensor.

        Returns:
            Tensor: Pooled tensor.
        """
        pass


class MaxUnpool1d(Module):
    """Computes a partial inverse of MaxPool1d."""

    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0) -> None:
        """Initialize MaxUnpool1d.

        Args:
            kernel_size (int): Size of the unpooling window.
            stride (int, optional): Stride of the window. Defaults to None.
            padding (int, optional): Implicit zero padding. Defaults to 0.
        """
        pass

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass.

        Args:
            input (Tensor): Input tensor.

        Returns:
            Tensor: Unpooled tensor.
        """
        pass


class MaxUnpool2d(Module):
    """Computes a partial inverse of MaxPool2d."""

    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0) -> None:
        """Initialize MaxUnpool2d.

        Args:
            kernel_size (int): Size of the unpooling window.
            stride (int, optional): Stride of the window. Defaults to None.
            padding (int, optional): Implicit zero padding. Defaults to 0.
        """
        pass

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass.

        Args:
            input (Tensor): Input tensor.

        Returns:
            Tensor: Unpooled tensor.
        """
        pass


class MaxUnpool3d(Module):
    """Computes a partial inverse of MaxPool3d."""

    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0) -> None:
        """Initialize MaxUnpool3d.

        Args:
            kernel_size (int): Size of the unpooling window.
            stride (int, optional): Stride of the window. Defaults to None.
            padding (int, optional): Implicit zero padding. Defaults to 0.
        """
        pass

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass.

        Args:
            input (Tensor): Input tensor.

        Returns:
            Tensor: Unpooled tensor.
        """
        pass
