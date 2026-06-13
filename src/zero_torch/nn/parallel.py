"""Module."""

from typing import Any
from .module import Module


class DataParallel(Module):
    """Implements data parallelism at the module level.

    This container parallelizes the application of the given module by
    splitting the input across the specified devices.
    """

    def __init__(
        self,
        module: Any,
        device_ids: Any = None,
        output_device: Any = None,
        dim: int = 0,
        *args: Any,
        **kwargs: Any,
    ):
        """Initialize the DataParallel module.

        Args:
            module (Any): The module to be parallelized.
            device_ids (Any, optional): CUDA devices to place the replicas on. Defaults to None.
            output_device (Any, optional): Device location of output. Defaults to None.
            dim (int, optional): Tensors are scattered along this dimension. Defaults to 0.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()
        self.module = module
        self.device_ids = device_ids
        self.output_device = output_device
        self.dim = dim

    def forward(self, *inputs: Any, **kwargs: Any) -> Any:
        """Forward pass parallelized across multiple devices.

        Args:
            *inputs (Any): The inputs to the module.
            **kwargs (Any): Keyword arguments to the module.

        Returns:
            Any: The combined output of the module replicas.
        """
        import ml_switcheroo_compiler.core.errors

        raise ml_switcheroo_compiler.core.errors.UnimplementedMathError
