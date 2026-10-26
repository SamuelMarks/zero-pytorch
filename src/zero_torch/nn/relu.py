"""ReLU and Sequential modules."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import zero_torch.nn.functional as F
from zero_torch._types import OpArg
from zero_torch.tensor import Tensor

from .module import Module


class ReLU(Module):
    """Applies the rectified linear unit function element-wise: `ReLU(x) = max(0, x)`."""

    __constants__: list[str] = ["inplace"]

    def __init__(
        self,
        inplace: bool = False,
        __constants__: Sequence[str] | None = None,
        *args: OpArg,
        **kwargs: OpArg,
    ) -> None:
        """Initializes ReLU module.

        Args:
            inplace: If True, performs the operation in-place.
            __constants__: Optional constant attribute names.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        _ = (args, kwargs)
        super().__init__()
        self.__constants__: list[str] = (
            list(__constants__) if __constants__ is not None else ["inplace"]
        )
        self.inplace: bool = inplace

    def forward(self, input: Tensor) -> Tensor:
        """Applies ReLU element-wise to `input`.

        Args:
            input: Input tensor.

        Returns:
            Output tensor with ReLU applied.
        """
        return F.relu(input, inplace=self.inplace)


class Sequential(Module):
    """A sequential container of child modules executed in order."""

    def __init__(self, *args: Module | Mapping[str, Module]) -> None:
        """Initializes Sequential module.

        Args:
            *args: Modules (or a single mapping of named modules) to add in sequence.
        """
        super().__init__()
        if len(args) == 1 and isinstance(args[0], Mapping):
            for key, mod in args[0].items():
                self.add_module(key, mod)
        else:
            for idx, arg in enumerate(args):
                assert isinstance(arg, Module)
                self.add_module(str(idx), arg)

    def forward(self, input: Tensor) -> Tensor:
        """Executes all child modules in sequence on `input`.

        Args:
            input: Input tensor to the first module.

        Returns:
            Output tensor of the final module in the sequence.
        """
        out = input
        for child_mod in self._modules.values():
            out = child_mod(out)
        return out
