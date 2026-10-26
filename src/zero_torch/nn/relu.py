"""ReLU and Sequential modules."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import overload

import zero_torch.nn.functional as F
from zero_torch._types import OpArg, SliceIndexType
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

    def __len__(self) -> int:
        """Returns the number of modules in the `Sequential` container.

        Returns:
            Number of child modules.
        """
        return len(self._modules)

    def __iter__(self) -> Iterator[Module]:
        """Returns an iterator over the child modules in sequence.

        Returns:
            Iterator over child modules.
        """
        return iter(self._modules.values())

    @overload
    def __getitem__(self, idx: int) -> Module:
        """Retrieves a child module by integer index.

        Args:
            idx: Integer index.

        Returns:
            Child module at `idx`.
        """

    @overload
    def __getitem__(self, idx: SliceIndexType) -> Sequential:
        """Retrieves a sliced `Sequential` container.

        Args:
            idx: Slice object.

        Returns:
            New `Sequential` containing the sliced child modules.
        """

    def __getitem__(self, idx: int | SliceIndexType) -> Module | Sequential:
        """Retrieves the child module at `idx` or a sliced `Sequential`.

        Args:
            idx: Integer index or slice.

        Returns:
            Child `Module` or new `Sequential` slice.
        """
        items = list(self._modules.items())
        if isinstance(idx, slice):
            return Sequential(dict(items[idx]))
        return items[idx][1]

    def __setitem__(self, idx: int, module: Module) -> None:
        """Replaces the child module at integer index `idx`.

        Args:
            idx: Integer index to replace.
            module: New `Module` to store.
        """
        keys = list(self._modules.keys())
        self._modules[keys[idx]] = module

    def __delitem__(self, idx: int | SliceIndexType) -> None:
        """Deletes the child module(s) at `idx`.

        Args:
            idx: Integer index or slice to delete.
        """
        keys = list(self._modules.keys())
        if isinstance(idx, slice):
            for k in keys[idx]:
                del self._modules[k]
        else:
            del self._modules[keys[idx]]

    def append(self, module: Module) -> Sequential:
        """Appends a given module to the end of the `Sequential` container.

        Args:
            module: Module to append.

        Returns:
            The modified `Sequential` (`self`).
        """
        self.add_module(str(len(self._modules)), module)
        return self

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
