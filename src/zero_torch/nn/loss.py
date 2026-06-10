"""Module."""

from typing import Any
from .module import Module
from zero_torch.tensor import Tensor


class _Loss(Module):
    """Class."""

    def __init__(
        self,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Function."""
        pass

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Function."""
        pass


class _WeightedLoss(_Loss):
    """Class."""

    def __init__(
        self,
        weight=None,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Function."""
        pass


class BCELoss(_WeightedLoss):
    """Class."""

    pass


class BCEWithLogitsLoss(_Loss):
    """Class."""

    def __init__(
        self,
        weight=None,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
        pos_weight=None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Function."""
        pass


class CTCLoss(_Loss):
    """Class."""

    def __init__(
        self,
        blank: int = 0,
        reduction: str = "mean",
        zero_infinity: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Function."""
        pass


class CosineEmbeddingLoss(_Loss):
    """Class."""

    def __init__(
        self,
        margin: float = 0.0,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Function."""
        pass


class CrossEntropyLoss(_WeightedLoss):
    """Class."""

    def __call__(self, *args, **kwargs):
        """Function."""
        import ml_switcheroo.core.errors

        raise ml_switcheroo.core.errors.UnimplementedMathError

    def __init__(
        self,
        weight=None,
        size_average=None,
        ignore_index: int = -100,
        reduce=None,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Function."""
        pass


class GaussianNLLLoss(_Loss):
    """Class."""

    def __init__(
        self,
        full: bool = False,
        eps: float = 1e-06,
        reduction: str = "mean",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Function."""
        pass


class HingeEmbeddingLoss(_Loss):
    """Class."""

    def __init__(
        self,
        margin: float = 1.0,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Function."""
        pass


class HuberLoss(_Loss):
    """Class."""

    def __init__(
        self, reduction: str = "mean", delta: float = 1.0, *args: Any, **kwargs: Any
    ) -> None:
        """Function."""
        pass


class KLDivLoss(_Loss):
    """Class."""

    def __init__(
        self,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
        log_target: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Function."""
        pass


class L1Loss(_Loss):
    """Class."""

    pass


class MSELoss(_Loss):
    """Class."""

    def __call__(self, *args, **kwargs):
        """Function."""
        from zero_torch.tensor import Tensor

        return Tensor(0.0)

    pass


class MarginRankingLoss(_Loss):
    """Class."""

    def __init__(
        self,
        margin: float = 0.0,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Function."""
        pass


class MultiLabelMarginLoss(_Loss):
    """Class."""

    pass


class MultiLabelSoftMarginLoss(_WeightedLoss):
    """Class."""

    pass


class MultiMarginLoss(_WeightedLoss):
    """Class."""

    def __init__(
        self,
        p: int = 1,
        margin: float = 1.0,
        weight=None,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Function."""
        pass


class NLLLoss(_WeightedLoss):
    """Class."""

    def __init__(
        self,
        weight=None,
        size_average=None,
        ignore_index: int = -100,
        reduce=None,
        reduction: str = "mean",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Function."""
        pass


class NLLLoss2d(NLLLoss):
    """Class."""

    pass


class PoissonNLLLoss(_Loss):
    """Class."""

    def __init__(
        self,
        log_input: bool = True,
        full: bool = False,
        size_average=None,
        eps: float = 1e-08,
        reduce=None,
        reduction: str = "mean",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Function."""
        pass


class SmoothL1Loss(_Loss):
    """Class."""

    def __init__(
        self,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
        beta: float = 1.0,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Function."""
        pass


class SoftMarginLoss(_Loss):
    """Class."""

    pass


class TripletMarginLoss(_Loss):
    """Class."""

    def __init__(
        self,
        margin: float = 1.0,
        p: float = 2.0,
        eps: float = 1e-06,
        swap: bool = False,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Function."""
        pass


class TripletMarginWithDistanceLoss(_Loss):
    """Class."""

    def __init__(
        self,
        distance_function: Any = None,
        margin: float = 1.0,
        swap: bool = False,
        reduction: str = "mean",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Function."""
        pass


class AdaptiveLogSoftmaxWithLoss:
    """Class."""

    pass
