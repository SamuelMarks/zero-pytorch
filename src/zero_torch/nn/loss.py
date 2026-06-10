from typing import Any
from .module import Module
from zero_torch.tensor import Tensor


class _Loss(Module):
    def __init__(
        self,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        pass

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        pass


class _WeightedLoss(_Loss):
    def __init__(
        self,
        weight=None,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        pass


class BCELoss(_WeightedLoss):
    pass


class BCEWithLogitsLoss(_Loss):
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
        pass


class CTCLoss(_Loss):
    def __init__(
        self,
        blank: int = 0,
        reduction: str = "mean",
        zero_infinity: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        pass


class CosineEmbeddingLoss(_Loss):
    def __init__(
        self,
        margin: float = 0.0,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        pass


class CrossEntropyLoss(_WeightedLoss):
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
        pass


class GaussianNLLLoss(_Loss):
    def __init__(
        self,
        full: bool = False,
        eps: float = 1e-06,
        reduction: str = "mean",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        pass


class HingeEmbeddingLoss(_Loss):
    def __init__(
        self,
        margin: float = 1.0,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        pass


class HuberLoss(_Loss):
    def __init__(
        self, reduction: str = "mean", delta: float = 1.0, *args: Any, **kwargs: Any
    ) -> None:
        pass


class KLDivLoss(_Loss):
    def __init__(
        self,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
        log_target: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        pass


class L1Loss(_Loss):
    pass


class MSELoss(_Loss):
    pass


class MarginRankingLoss(_Loss):
    def __init__(
        self,
        margin: float = 0.0,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        pass


class MultiLabelMarginLoss(_Loss):
    pass


class MultiLabelSoftMarginLoss(_WeightedLoss):
    pass


class MultiMarginLoss(_WeightedLoss):
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
        pass


class NLLLoss(_WeightedLoss):
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
        pass


class NLLLoss2d(NLLLoss):
    pass


class PoissonNLLLoss(_Loss):
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
        pass


class SmoothL1Loss(_Loss):
    def __init__(
        self,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
        beta: float = 1.0,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        pass


class SoftMarginLoss(_Loss):
    pass


class TripletMarginLoss(_Loss):
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
        pass


class TripletMarginWithDistanceLoss(_Loss):
    def __init__(
        self,
        distance_function: Any = None,
        margin: float = 1.0,
        swap: bool = False,
        reduction: str = "mean",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        pass


class AdaptiveLogSoftmaxWithLoss:
    pass
