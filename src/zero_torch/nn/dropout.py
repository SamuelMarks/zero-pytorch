"""Dropout modules."""

from __future__ import annotations

from zero_torch._types import OpArg
from zero_torch.tensor import Tensor

from .functional_dropout import (
    alpha_dropout,
    dropout,
    dropout1d,
    dropout2d,
    dropout3d,
    feature_alpha_dropout,
)
from .module import Module


class Dropout(Module):
    """During training, randomly zeroes some of the elements of the input tensor with probability `p`."""

    def __init__(
        self, p: float = 0.5, inplace: bool = False, *args: OpArg, **kwargs: OpArg
    ) -> None:
        """Initializes the Dropout layer.

        Args:
            p: Probability of an element to be zeroed.
            inplace: If True, performs the operation in-place.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        _ = (args, kwargs)
        super().__init__()
        self.p: float = p
        self.inplace: bool = inplace

    def forward(self, input: Tensor) -> Tensor:
        """Applies dropout to `input`.

        Args:
            input: Input tensor.

        Returns:
            Output tensor with elements randomly zeroed during training.
        """
        return dropout(input, p=self.p, training=self.training, inplace=self.inplace)


class Dropout1d(Module):
    """Randomly zero out entire 1D channels."""

    def __init__(
        self, p: float = 0.5, inplace: bool = False, *args: OpArg, **kwargs: OpArg
    ) -> None:
        """Initializes the Dropout1d layer.

        Args:
            p: Probability of a channel to be zeroed.
            inplace: If True, performs the operation in-place.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        _ = (args, kwargs)
        super().__init__()
        self.p: float = p
        self.inplace: bool = inplace

    def forward(self, input: Tensor) -> Tensor:
        """Applies 1D channel dropout to `input`.

        Args:
            input: Input tensor.

        Returns:
            Output tensor with 1D channels randomly zeroed during training.
        """
        return dropout1d(input, p=self.p, training=self.training, inplace=self.inplace)


class Dropout2d(Module):
    """Randomly zero out entire 2D channels."""

    def __init__(
        self, p: float = 0.5, inplace: bool = False, *args: OpArg, **kwargs: OpArg
    ) -> None:
        """Initializes the Dropout2d layer.

        Args:
            p: Probability of a channel to be zeroed.
            inplace: If True, performs the operation in-place.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        _ = (args, kwargs)
        super().__init__()
        self.p: float = p
        self.inplace: bool = inplace

    def forward(self, input: Tensor) -> Tensor:
        """Applies 2D channel dropout to `input`.

        Args:
            input: Input tensor.

        Returns:
            Output tensor with 2D channels randomly zeroed during training.
        """
        return dropout2d(input, p=self.p, training=self.training, inplace=self.inplace)


class Dropout3d(Module):
    """Randomly zero out entire 3D channels."""

    def __init__(
        self, p: float = 0.5, inplace: bool = False, *args: OpArg, **kwargs: OpArg
    ) -> None:
        """Initializes the Dropout3d layer.

        Args:
            p: Probability of a channel to be zeroed.
            inplace: If True, performs the operation in-place.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        _ = (args, kwargs)
        super().__init__()
        self.p: float = p
        self.inplace: bool = inplace

    def forward(self, input: Tensor) -> Tensor:
        """Applies 3D channel dropout to `input`.

        Args:
            input: Input tensor.

        Returns:
            Output tensor with 3D channels randomly zeroed during training.
        """
        return dropout3d(input, p=self.p, training=self.training, inplace=self.inplace)


class AlphaDropout(Module):
    """Applies Alpha Dropout over the input, maintaining self-normalizing properties."""

    def __init__(
        self, p: float = 0.5, inplace: bool = False, *args: OpArg, **kwargs: OpArg
    ) -> None:
        """Initializes the AlphaDropout layer.

        Args:
            p: Probability of an element to be dropped.
            inplace: If True, performs the operation in-place.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        _ = (args, kwargs)
        super().__init__()
        self.p: float = p
        self.inplace: bool = inplace

    def forward(self, input: Tensor) -> Tensor:
        """Applies alpha dropout to `input`.

        Args:
            input: Input tensor.

        Returns:
            Output tensor with alpha dropout applied.
        """
        return alpha_dropout(
            input, p=self.p, training=self.training, inplace=self.inplace
        )


class FeatureAlphaDropout(Module):
    """Randomly masks out entire channels with Alpha Dropout."""

    def __init__(
        self, p: float = 0.5, inplace: bool = False, *args: OpArg, **kwargs: OpArg
    ) -> None:
        """Initializes the FeatureAlphaDropout layer.

        Args:
            p: Probability of a channel to be zeroed.
            inplace: If True, performs the operation in-place.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        _ = (args, kwargs)
        super().__init__()
        self.p: float = p
        self.inplace: bool = bool(inplace)

    def forward(self, input: Tensor) -> Tensor:
        """Applies feature alpha dropout to `input`.

        Args:
            input: Input tensor.

        Returns:
            Output tensor with feature alpha dropout applied.
        """
        return feature_alpha_dropout(
            input, p=self.p, training=self.training, inplace=self.inplace
        )
