"ReLU and Sequential modules."

from typing import Any

from zero_torch.tensor import Tensor

from .module import Module


class ReLU(Module):
    """Applies the rectified linear unit function element-wise."""

    def __init__(
        self,
        inplace: bool = False,
        __constants__=None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize ReLU module.

        Args:
            inplace (bool, optional): If set to True, will do this operation in-place. Defaults to False.
            __constants__ (list, optional): List of constants. Defaults to ["inplace"].
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        if __constants__ is None:
            __constants__ = ["inplace"]
        super().__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass applying ReLU.

        Args:
            input (Tensor): The input tensor.

        Returns:
            Tensor: A tensor with the ReLU function applied element-wise.
        """
        import zero_torch.nn.functional as F

        # Eager mode zero_torch doesn't have inplace relu out of the box so just fallback to normal
        return F.relu(input, inplace=self.inplace)


class Sequential(Module):
    """A sequential container.

    Modules will be added to it in the order they are passed in the constructor.
    Alternatively, an OrderedDict of modules can be passed in.
    """

    def __init__(self, *args: Any) -> None:
        """Initialize Sequential module.

        Args:
            *args: Modules to add to the sequential container.
        """
        super().__init__()
        if len(args) == 1 and isinstance(args[0], dict):
            for key, module in args[0].items():  # pragma: no cover
                self.add_module(key, module)  # pragma: no cover
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)  # pragma: no cover

    def forward(self, input: Any) -> Any:
        """Forward pass through all modules in sequence.

        Args:
            input (Any): The input data.

        Returns:
            Any: The output of the final module in the sequence.
        """
        for module in self._modules.values():
            input = module(input)  # pragma: no cover
        return input
