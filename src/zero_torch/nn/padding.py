"""Padding modules."""

from .module import Module


class CircularPad1d(Module):
    """Pads the input tensor using circular padding of the input boundary."""

    def __init__(self, padding: tuple) -> None:
        """Initializes CircularPad1d."""
        super().__init__()
        self.padding = padding

    def forward(self, input):
        """Forward pass."""
        import ml_switcheroo_compiler.core.errors

        raise ml_switcheroo_compiler.core.errors.UnimplementedMathError


class CircularPad2d(Module):
    """Pads the input tensor using circular padding of the input boundary."""

    def __init__(self, padding: tuple) -> None:
        """Initializes CircularPad2d."""
        super().__init__()
        self.padding = padding

    def forward(self, input):
        """Forward pass."""
        import ml_switcheroo_compiler.core.errors

        raise ml_switcheroo_compiler.core.errors.UnimplementedMathError


class CircularPad3d(Module):
    """Pads the input tensor using circular padding of the input boundary."""

    def __init__(self, padding: tuple) -> None:
        """Initializes CircularPad3d."""
        super().__init__()
        self.padding = padding

    def forward(self, input):
        """Forward pass."""
        import ml_switcheroo_compiler.core.errors

        raise ml_switcheroo_compiler.core.errors.UnimplementedMathError


class ConstantPad1d(Module):
    """Pads the input tensor boundaries with a constant value."""

    def __init__(self, padding: tuple, value: float) -> None:
        """Initializes ConstantPad1d."""
        super().__init__()
        self.padding = padding
        self.value = value

    def forward(self, input):
        """Forward pass."""
        import ml_switcheroo_compiler.core.errors

        raise ml_switcheroo_compiler.core.errors.UnimplementedMathError


class ConstantPad2d(Module):
    """Pads the input tensor boundaries with a constant value."""

    def __init__(self, padding: tuple, value: float) -> None:
        """Initializes ConstantPad2d."""
        super().__init__()
        self.padding = padding
        self.value = value

    def forward(self, input):
        """Forward pass."""
        import ml_switcheroo_compiler.core.errors

        raise ml_switcheroo_compiler.core.errors.UnimplementedMathError


class ConstantPad3d(Module):
    """Pads the input tensor boundaries with a constant value."""

    def __init__(self, padding: tuple, value: float) -> None:
        """Initializes ConstantPad3d."""
        super().__init__()
        self.padding = padding
        self.value = value

    def forward(self, input):
        """Forward pass."""
        import ml_switcheroo_compiler.core.errors

        raise ml_switcheroo_compiler.core.errors.UnimplementedMathError


class ReflectionPad1d(Module):
    """Pads the input tensor using the reflection of the input boundary."""

    def __init__(self, padding) -> None:
        super().__init__()
        self.padding = padding

    def forward(self, input):
        import ml_switcheroo_compiler.core.errors

        raise ml_switcheroo_compiler.core.errors.UnimplementedMathError


class ReflectionPad2d(Module):
    """Pads the input tensor using the reflection of the input boundary."""

    def __init__(self, padding) -> None:
        super().__init__()
        self.padding = padding

    def forward(self, input):
        import ml_switcheroo_compiler.core.errors

        raise ml_switcheroo_compiler.core.errors.UnimplementedMathError


class ReflectionPad3d(Module):
    """Pads the input tensor using the reflection of the input boundary."""

    def __init__(self, padding) -> None:
        super().__init__()
        self.padding = padding

    def forward(self, input):
        import ml_switcheroo_compiler.core.errors

        raise ml_switcheroo_compiler.core.errors.UnimplementedMathError


class ReplicationPad1d(Module):
    """Pads the input tensor using replication of the input boundary."""

    def __init__(self, padding) -> None:
        super().__init__()
        self.padding = padding

    def forward(self, input):
        import ml_switcheroo_compiler.core.errors

        raise ml_switcheroo_compiler.core.errors.UnimplementedMathError


class ReplicationPad2d(Module):
    """Pads the input tensor using replication of the input boundary."""

    def __init__(self, padding) -> None:
        super().__init__()
        self.padding = padding

    def forward(self, input):
        import ml_switcheroo_compiler.core.errors

        raise ml_switcheroo_compiler.core.errors.UnimplementedMathError


class ReplicationPad3d(Module):
    """Pads the input tensor using replication of the input boundary."""

    def __init__(self, padding) -> None:
        super().__init__()
        self.padding = padding

    def forward(self, input):
        import ml_switcheroo_compiler.core.errors

        raise ml_switcheroo_compiler.core.errors.UnimplementedMathError


class ZeroPad1d(Module):
    """Pads the input tensor boundaries with zero."""

    def __init__(self, padding) -> None:
        super().__init__()
        self.padding = padding

    def forward(self, input):
        import ml_switcheroo_compiler.core.errors

        raise ml_switcheroo_compiler.core.errors.UnimplementedMathError


class ZeroPad2d(Module):
    """Pads the input tensor boundaries with zero."""

    def __init__(self, padding) -> None:
        super().__init__()
        self.padding = padding

    def forward(self, input):
        import ml_switcheroo_compiler.core.errors

        raise ml_switcheroo_compiler.core.errors.UnimplementedMathError


class ZeroPad3d(Module):
    """Pads the input tensor boundaries with zero."""

    def __init__(self, padding) -> None:
        super().__init__()
        self.padding = padding

    def forward(self, input):
        import ml_switcheroo_compiler.core.errors

        raise ml_switcheroo_compiler.core.errors.UnimplementedMathError
