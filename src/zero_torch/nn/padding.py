"""Padding modules."""

from __future__ import annotations

import zero_torch.nn.functional as F
from zero_torch._types import SizeNd
from zero_torch.tensor import Tensor

from .module import Module


def _norm_pad(padding: SizeNd, length: int) -> tuple[int, ...]:
    """Normalizes an integer or sequence padding into a tuple of `length` ints.

    Args:
        padding: Integer or sequence of padding sizes.
        length: Expected tuple length (`2 * ndims`).

    Returns:
        Normalized padding tuple.
    """
    if isinstance(padding, int):
        return (padding,) * length
    return tuple(int(p) for p in padding)


class CircularPad1d(Module):
    """Pads the input tensor using circular padding of the 1D input boundary."""

    def __init__(self, padding: SizeNd) -> None:
        """Initializes CircularPad1d.

        Args:
            padding: Size of the padding (`int` or 2-tuple).
        """
        super().__init__()
        self.padding: tuple[int, ...] = _norm_pad(padding, 2)

    def forward(self, input: Tensor) -> Tensor:
        """Applies 1D circular padding to `input`.

        Args:
            input: Input tensor.

        Returns:
            Padded tensor.
        """
        return F.pad(input, self.padding, mode="circular")


class CircularPad2d(Module):
    """Pads the input tensor using circular padding of the 2D input boundary."""

    def __init__(self, padding: SizeNd) -> None:
        """Initializes CircularPad2d.

        Args:
            padding: Size of the padding (`int` or 4-tuple).
        """
        super().__init__()
        self.padding: tuple[int, ...] = _norm_pad(padding, 4)

    def forward(self, input: Tensor) -> Tensor:
        """Applies 2D circular padding to `input`.

        Args:
            input: Input tensor.

        Returns:
            Padded tensor.
        """
        return F.pad(input, self.padding, mode="circular")


class CircularPad3d(Module):
    """Pads the input tensor using circular padding of the 3D input boundary."""

    def __init__(self, padding: SizeNd) -> None:
        """Initializes CircularPad3d.

        Args:
            padding: Size of the padding (`int` or 6-tuple).
        """
        super().__init__()
        self.padding: tuple[int, ...] = _norm_pad(padding, 6)

    def forward(self, input: Tensor) -> Tensor:
        """Applies 3D circular padding to `input`.

        Args:
            input: Input tensor.

        Returns:
            Padded tensor.
        """
        return F.pad(input, self.padding, mode="circular")


class ConstantPad1d(Module):
    """Pads the 1D input tensor boundaries with a constant value."""

    def __init__(self, padding: SizeNd, value: float) -> None:
        """Initializes ConstantPad1d.

        Args:
            padding: Size of the padding (`int` or 2-tuple).
            value: Fill value for constant padding.
        """
        super().__init__()
        self.padding: tuple[int, ...] = _norm_pad(padding, 2)
        self.value: float = value

    def forward(self, input: Tensor) -> Tensor:
        """Applies 1D constant padding to `input`.

        Args:
            input: Input tensor.

        Returns:
            Padded tensor.
        """
        return F.pad(input, self.padding, mode="constant", value=self.value)


class ConstantPad2d(Module):
    """Pads the 2D input tensor boundaries with a constant value."""

    __constants__: list[str] = ["padding", "value"]

    def __init__(self, padding: SizeNd, value: float) -> None:
        """Initializes ConstantPad2d.

        Args:
            padding: Size of the padding (`int` or 4-tuple).
            value: Fill value for constant padding.
        """
        super().__init__()
        self.padding: tuple[int, ...] = _norm_pad(padding, 4)
        self.value: float = value

    def forward(self, input: Tensor) -> Tensor:
        """Applies 2D constant padding to `input`.

        Args:
            input: Input tensor.

        Returns:
            Padded tensor.
        """
        return F.pad(input, self.padding, mode="constant", value=self.value)


class ConstantPad3d(Module):
    """Pads the 3D input tensor boundaries with a constant value."""

    def __init__(self, padding: SizeNd, value: float) -> None:
        """Initializes ConstantPad3d.

        Args:
            padding: Size of the padding (`int` or 6-tuple).
            value: Fill value for constant padding.
        """
        super().__init__()
        self.padding: tuple[int, ...] = _norm_pad(padding, 6)
        self.value: float = value

    def forward(self, input: Tensor) -> Tensor:
        """Applies 3D constant padding to `input`.

        Args:
            input: Input tensor.

        Returns:
            Padded tensor.
        """
        return F.pad(input, self.padding, mode="constant", value=self.value)


class ReflectionPad1d(Module):
    """Pads the input tensor using the reflection of the 1D input boundary."""

    def __init__(self, padding: SizeNd) -> None:
        """Initializes ReflectionPad1d.

        Args:
            padding: Size of the padding (`int` or 2-tuple).
        """
        super().__init__()
        self.padding: tuple[int, ...] = _norm_pad(padding, 2)

    def forward(self, input: Tensor) -> Tensor:
        """Applies 1D reflection padding to `input`.

        Args:
            input: Input tensor.

        Returns:
            Padded tensor.
        """
        return F.pad(input, self.padding, mode="reflect")


class ReflectionPad2d(Module):
    """Pads the input tensor using the reflection of the 2D input boundary."""

    def __init__(self, padding: SizeNd) -> None:
        """Initializes ReflectionPad2d.

        Args:
            padding: Size of the padding (`int` or 4-tuple).
        """
        super().__init__()
        self.padding: tuple[int, ...] = _norm_pad(padding, 4)

    def forward(self, input: Tensor) -> Tensor:
        """Applies 2D reflection padding to `input`.

        Args:
            input: Input tensor.

        Returns:
            Padded tensor.
        """
        return F.pad(input, self.padding, mode="reflect")


class ReflectionPad3d(Module):
    """Pads the input tensor using the reflection of the 3D input boundary."""

    def __init__(self, padding: SizeNd) -> None:
        """Initializes ReflectionPad3d.

        Args:
            padding: Size of the padding (`int` or 6-tuple).
        """
        super().__init__()
        self.padding: tuple[int, ...] = _norm_pad(padding, 6)

    def forward(self, input: Tensor) -> Tensor:
        """Applies 3D reflection padding to `input`.

        Args:
            input: Input tensor.

        Returns:
            Padded tensor.
        """
        return F.pad(input, self.padding, mode="reflect")


class ReplicationPad1d(Module):
    """Pads the input tensor using replication of the 1D input boundary."""

    def __init__(self, padding: SizeNd) -> None:
        """Initializes ReplicationPad1d.

        Args:
            padding: Size of the padding (`int` or 2-tuple).
        """
        super().__init__()
        self.padding: tuple[int, ...] = _norm_pad(padding, 2)

    def forward(self, input: Tensor) -> Tensor:
        """Applies 1D replication padding to `input`.

        Args:
            input: Input tensor.

        Returns:
            Padded tensor.
        """
        return F.pad(input, self.padding, mode="replicate")


class ReplicationPad2d(Module):
    """Pads the input tensor using replication of the 2D input boundary."""

    def __init__(self, padding: SizeNd) -> None:
        """Initializes ReplicationPad2d.

        Args:
            padding: Size of the padding (`int` or 4-tuple).
        """
        super().__init__()
        self.padding: tuple[int, ...] = _norm_pad(padding, 4)

    def forward(self, input: Tensor) -> Tensor:
        """Applies 2D replication padding to `input`.

        Args:
            input: Input tensor.

        Returns:
            Padded tensor.
        """
        return F.pad(input, self.padding, mode="replicate")


class ReplicationPad3d(Module):
    """Pads the input tensor using replication of the 3D input boundary."""

    def __init__(self, padding: SizeNd) -> None:
        """Initializes ReplicationPad3d.

        Args:
            padding: Size of the padding (`int` or 6-tuple).
        """
        super().__init__()
        self.padding: tuple[int, ...] = _norm_pad(padding, 6)

    def forward(self, input: Tensor) -> Tensor:
        """Applies 3D replication padding to `input`.

        Args:
            input: Input tensor.

        Returns:
            Padded tensor.
        """
        return F.pad(input, self.padding, mode="replicate")


class ZeroPad1d(Module):
    """Pads the 1D input tensor boundaries with zero."""

    def __init__(self, padding: SizeNd) -> None:
        """Initializes ZeroPad1d.

        Args:
            padding: Size of the padding (`int` or 2-tuple).
        """
        super().__init__()
        self.padding: tuple[int, ...] = _norm_pad(padding, 2)

    def forward(self, input: Tensor) -> Tensor:
        """Applies 1D zero padding to `input`.

        Args:
            input: Input tensor.

        Returns:
            Padded tensor.
        """
        return F.pad(input, self.padding, mode="constant", value=0.0)


class ZeroPad2d(Module):
    """Pads the 2D input tensor boundaries with zero."""

    def __init__(self, padding: SizeNd) -> None:
        """Initializes ZeroPad2d.

        Args:
            padding: Size of the padding (`int` or 4-tuple).
        """
        super().__init__()
        self.padding: tuple[int, ...] = _norm_pad(padding, 4)

    def forward(self, input: Tensor) -> Tensor:
        """Applies 2D zero padding to `input`.

        Args:
            input: Input tensor.

        Returns:
            Padded tensor.
        """
        return F.pad(input, self.padding, mode="constant", value=0.0)


class ZeroPad3d(Module):
    """Pads the 3D input tensor boundaries with zero."""

    def __init__(self, padding: SizeNd) -> None:
        """Initializes ZeroPad3d.

        Args:
            padding: Size of the padding (`int` or 6-tuple).
        """
        super().__init__()
        self.padding: tuple[int, ...] = _norm_pad(padding, 6)

    def forward(self, input: Tensor) -> Tensor:
        """Applies 3D zero padding to `input`.

        Args:
            input: Input tensor.

        Returns:
            Padded tensor.
        """
        return F.pad(input, self.padding, mode="constant", value=0.0)
