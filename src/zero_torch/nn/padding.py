"""Padding modules."""

import zero_torch.nn.functional as F

from .module import Module


class CircularPad1d(Module):
    """Pads the input tensor using circular padding of the input boundary."""

    def __init__(self, padding: tuple) -> None:
        """Initializes CircularPad1d."""
        super().__init__()
        self.padding = padding

    def forward(self, input):
        return F.pad(input, self.padding, mode="circular")


class CircularPad2d(Module):
    """Pads the input tensor using circular padding of the input boundary."""

    def __init__(self, padding: tuple) -> None:
        """Initializes CircularPad2d."""
        super().__init__()
        self.padding = padding

    def forward(self, input):
        return F.pad(input, self.padding, mode="circular")


class CircularPad3d(Module):
    """Pads the input tensor using circular padding of the input boundary."""

    def __init__(self, padding: tuple) -> None:
        """Initializes CircularPad3d."""
        super().__init__()
        self.padding = padding

    def forward(self, input):
        return F.pad(input, self.padding, mode="circular")


class ConstantPad1d(Module):
    """Pads the input tensor boundaries with a constant value."""

    def __init__(self, padding: tuple, value: float) -> None:
        """Initializes ConstantPad1d."""
        super().__init__()
        self.padding = padding
        self.value = value

    def forward(self, input):
        return F.pad(input, self.padding, mode="constant", value=self.value)


class ConstantPad2d(Module):
    """Pads the input tensor boundaries with a constant value."""

    def __init__(self, padding: tuple, value: float) -> None:
        """Initializes ConstantPad2d."""
        super().__init__()
        self.padding = padding
        self.value = value

    def forward(self, input):
        return F.pad(input, self.padding, mode="constant", value=self.value)


class ConstantPad3d(Module):
    """Pads the input tensor boundaries with a constant value."""

    def __init__(self, padding: tuple, value: float) -> None:
        """Initializes ConstantPad3d."""
        super().__init__()
        self.padding = padding
        self.value = value

    def forward(self, input):
        return F.pad(input, self.padding, mode="constant", value=self.value)


class ReflectionPad1d(Module):
    """Pads the input tensor using the reflection of the input boundary."""

    def __init__(self, padding) -> None:
        super().__init__()
        self.padding = padding

    def forward(self, input):
        return F.pad(input, self.padding, mode="reflect")


class ReflectionPad2d(Module):
    """Pads the input tensor using the reflection of the input boundary."""

    def __init__(self, padding) -> None:
        super().__init__()
        self.padding = padding

    def forward(self, input):
        return F.pad(input, self.padding, mode="reflect")


class ReflectionPad3d(Module):
    """Pads the input tensor using the reflection of the input boundary."""

    def __init__(self, padding) -> None:
        super().__init__()
        self.padding = padding

    def forward(self, input):
        return F.pad(input, self.padding, mode="reflect")


class ReplicationPad1d(Module):
    """Pads the input tensor using replication of the input boundary."""

    def __init__(self, padding) -> None:
        super().__init__()
        self.padding = padding

    def forward(self, input):
        return F.pad(input, self.padding, mode="replicate")


class ReplicationPad2d(Module):
    """Pads the input tensor using replication of the input boundary."""

    def __init__(self, padding) -> None:
        super().__init__()
        self.padding = padding

    def forward(self, input):
        return F.pad(input, self.padding, mode="replicate")


class ReplicationPad3d(Module):
    """Pads the input tensor using replication of the input boundary."""

    def __init__(self, padding) -> None:
        super().__init__()
        self.padding = padding

    def forward(self, input):
        return F.pad(input, self.padding, mode="replicate")


class ZeroPad1d(Module):
    """Pads the input tensor boundaries with zero."""

    def __init__(self, padding) -> None:
        super().__init__()
        self.padding = padding

    def forward(self, input):
        return F.pad(input, self.padding, mode="constant", value=0.0)


class ZeroPad2d(Module):
    """Pads the input tensor boundaries with zero."""

    def __init__(self, padding) -> None:
        super().__init__()
        self.padding = padding

    def forward(self, input):
        return F.pad(input, self.padding, mode="constant", value=0.0)


class ZeroPad3d(Module):
    """Pads the input tensor boundaries with zero."""

    def __init__(self, padding) -> None:
        super().__init__()
        self.padding = padding

    def forward(self, input):
        return F.pad(input, self.padding, mode="constant", value=0.0)
