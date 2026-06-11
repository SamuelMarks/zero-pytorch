"""Module."""

from typing import Any
from .module import Module
from zero_torch.tensor import Tensor


class Dropout(Module):
    """During training, randomly zeroes some of the elements of the input tensor with probability p."""

    def __init__(
        self, p: float = 0.5, inplace: bool = False, *args: Any, **kwargs: Any
    ) -> None:
        """Initialize the dropout layer.

        Args:
            p (float, optional): Probability of an element to be zeroed. Defaults to 0.5.
            inplace (bool, optional): If set to True, will do this operation in-place. Defaults to False.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass for dropout.

        Args:
            input (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor with elements randomly zeroed out.
        """
        import ml_switcheroo.core.errors

        raise ml_switcheroo.core.errors.UnimplementedMathError


class Dropout1d(Module):
    """Randomly zero out entire channels (a channel is a 1D feature map)."""

    def __init__(
        self, p: Any = 0.5, inplace: Any = False, *args: Any, **kwargs: Any
    ) -> None:
        """Initialize the 1D dropout layer.

        Args:
            p (Any, optional): Probability of a channel to be zeroed. Defaults to 0.5.
            inplace (Any, optional): If set to True, will do this operation in-place. Defaults to False.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass for 1D dropout.

        Args:
            input (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor with channels randomly zeroed out.
        """
        import ml_switcheroo.core.errors

        raise ml_switcheroo.core.errors.UnimplementedMathError


class Dropout2d(Module):
    """Randomly zero out entire channels (a channel is a 2D feature map)."""

    def __init__(
        self, p: Any = 0.5, inplace: Any = False, *args: Any, **kwargs: Any
    ) -> None:
        """Initialize the 2D dropout layer.

        Args:
            p (Any, optional): Probability of a channel to be zeroed. Defaults to 0.5.
            inplace (Any, optional): If set to True, will do this operation in-place. Defaults to False.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass for 2D dropout.

        Args:
            input (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor with channels randomly zeroed out.
        """
        import ml_switcheroo.core.errors

        raise ml_switcheroo.core.errors.UnimplementedMathError


class Dropout3d(Module):
    """Randomly zero out entire channels (a channel is a 3D feature map)."""

    def __init__(
        self, p: Any = 0.5, inplace: Any = False, *args: Any, **kwargs: Any
    ) -> None:
        """Initialize the 3D dropout layer.

        Args:
            p (Any, optional): Probability of a channel to be zeroed. Defaults to 0.5.
            inplace (Any, optional): If set to True, will do this operation in-place. Defaults to False.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass for 3D dropout.

        Args:
            input (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor with channels randomly zeroed out.
        """
        import ml_switcheroo.core.errors

        raise ml_switcheroo.core.errors.UnimplementedMathError


class AlphaDropout(Module):
    """Applies Alpha Dropout over the input. Alpha Dropout is a type of Dropout that maintains the self-normalizing property."""

    def __init__(
        self, p: float = 0.5, inplace: Any = False, *args: Any, **kwargs: Any
    ) -> None:
        """Initialize the alpha dropout layer.

        Args:
            p (float, optional): Probability of an element to be dropped. Defaults to 0.5.
            inplace (Any, optional): If set to True, will do this operation in-place. Defaults to False.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass for alpha dropout.

        Args:
            input (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor with alpha dropout applied.
        """
        from .functional_dropout import alpha_dropout

        return alpha_dropout(
            input, p=self.p, training=self.training, inplace=self.inplace
        )


class FeatureAlphaDropout(Module):
    """Randomly masks out entire channels (a channel is a feature map) with Alpha Dropout."""

    def __init__(
        self, p: Any = 0.5, inplace: Any = None, *args: Any, **kwargs: Any
    ) -> None:
        """Initialize the feature alpha dropout layer.

        Args:
            p (Any, optional): Probability of a channel to be zeroed. Defaults to 0.5.
            inplace (Any, optional): If set to True, will do this operation in-place. Defaults to None.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass for feature alpha dropout.

        Args:
            input (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor with feature alpha dropout applied.
        """
        import ml_switcheroo.core.errors

        raise ml_switcheroo.core.errors.UnimplementedMathError
