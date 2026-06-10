from typing import Any
from .module import Module


class DataParallel(Module):
    """Implements data parallelism at the module level."""

    def __init__(
        self,
        module: Any,
        device_ids: Any = None,
        output_device: Any = None,
        dim: int = 0,
        *args: Any,
        **kwargs: Any,
    ):
        """Initialize."""
        pass

    def forward(self, *inputs: Any, **kwargs: Any) -> Any:
        """Forward pass."""
        pass
