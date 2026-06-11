"""Flatten modules."""

from .module import Module


class Flatten(Module):
    """Flattens a contiguous range of dims into a tensor."""

    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        """Initializes Flatten."""
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input):
        """Forward pass."""
        import ml_switcheroo.core.errors

        raise ml_switcheroo.core.errors.UnimplementedMathError


class Unflatten(Module):
    """Unflattens a tensor."""

    def __init__(self, dim, unflattened_size) -> None:
        super().__init__()
        self.dim = dim
        self.unflattened_size = unflattened_size

    def forward(self, input):
        import ml_switcheroo.core.errors

        raise ml_switcheroo.core.errors.UnimplementedMathError
