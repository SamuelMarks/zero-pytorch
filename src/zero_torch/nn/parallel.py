"""DataParallel module."""

from __future__ import annotations

from collections.abc import Sequence

from zero_torch._types import DeviceLike, OpArg
from zero_torch.tensor import Tensor

from .module import Module


class DataParallel(Module):
    """Implements data parallelism at the module level."""

    def __init__(
        self,
        module: Module,
        device_ids: Sequence[int | str] | None = None,
        output_device: DeviceLike = None,
        dim: int = 0,
        *args: OpArg,
        **kwargs: OpArg,
    ) -> None:
        """Initializes the DataParallel module.

        Args:
            module: Module to be parallelized.
            device_ids: Devices on which to replicate the module.
            output_device: Device location of the output.
            dim: Dimension along which tensors are scattered.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        _ = (args, kwargs)
        super().__init__()
        self.module: Module = module
        self.add_module("module", module)
        self.device_ids: Sequence[int | str] | None = device_ids
        self.output_device: DeviceLike = output_device
        self.dim: int = dim

    def forward(self, *inputs: OpArg, **kwargs: OpArg) -> Tensor:
        """Executes the forward pass of the wrapped module.

        Args:
            *inputs: Positional inputs to the wrapped module.
            **kwargs: Keyword arguments to the wrapped module.

        Returns:
            Output tensor from the wrapped module.
        """
        return self.module(*inputs, **kwargs)
