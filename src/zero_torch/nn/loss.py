"""Loss module."""

from typing import Any

from zero_torch.tensor import Tensor

from .module import Module


class _Loss(Module):
    """Base class for all neural network loss functions."""

    def __init__(
        self,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initializes the _Loss module.

        Args:
            size_average (Any, optional): Deprecated (see reduction). Defaults to None.
            reduce (Any, optional): Deprecated (see reduction). Defaults to None.
            reduction (str, optional): Specifies the reduction to apply to the output. Defaults to "mean".
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Computes the loss.

        Args:
            input (Tensor): The input tensor.
            target (Tensor): The target tensor.

        Returns:
            Tensor: The computed loss.
        """
        return input  # pragma: no cover


class _WeightedLoss(_Loss):
    """Base class for loss functions that use weights."""

    def __init__(
        self,
        weight=None,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initializes the _WeightedLoss module.

        Args:
            weight (Tensor, optional): A manual rescaling weight given to each class. Defaults to None.
            size_average (Any, optional): Deprecated. Defaults to None.
            reduce (Any, optional): Deprecated. Defaults to None.
            reduction (str, optional): Specifies the reduction to apply to the output. Defaults to "mean".
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__()


class BCELoss(_WeightedLoss):
    """Creates a criterion that measures the Binary Cross Entropy between the target and the input probabilities."""

    def forward(self, input, target):
        """Forward pass.

        Args:
            input (Tensor): input.
            target (Tensor): target.

        Returns:
            Tensor: loss.
        """
        import zero_torch.nn.functional as F

        return F.binary_cross_entropy(input, target)

    """Creates a criterion that measures the Binary Cross Entropy between the target and the input probabilities."""


class BCEWithLogitsLoss(_Loss):
    """This loss combines a Sigmoid layer and the BCELoss in one single class."""

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
        """Initializes the BCEWithLogitsLoss module.

        Args:
            weight (Tensor, optional): A manual rescaling weight given to the loss of each batch element. Defaults to None.
            size_average (Any, optional): Deprecated. Defaults to None.
            reduce (Any, optional): Deprecated. Defaults to None.
            reduction (str, optional): Specifies the reduction to apply to the output. Defaults to "mean".
            pos_weight (Tensor, optional): A weight of positive examples. Defaults to None.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__()

    def forward(self, input, target):
        """Forward pass.

        Args:
            input (Tensor): input.
            target (Tensor): target.

        Returns:
            Tensor: loss.
        """
        import zero_torch.nn.functional as F

        return F.binary_cross_entropy_with_logits(input, target)


class CTCLoss(_Loss):
    """The Connectionist Temporal Classification loss."""

    def __init__(
        self,
        blank: int = 0,
        reduction: str = "mean",
        zero_infinity: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initializes the CTCLoss module.

        Args:
            blank (int, optional): Blank label. Defaults to 0.
            reduction (str, optional): Specifies the reduction to apply to the output. Defaults to "mean".
            zero_infinity (bool, optional): Whether to zero infinite losses and the associated gradients. Defaults to False.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__()

    def forward(self, input, target):
        """Forward pass.

        Args:
            input (Tensor): input.
            target (Tensor): target.

        Returns:
            Tensor: loss.
        """
        import zero_torch.nn.functional as F

        return F.ctc_loss(input, target)


class CosineEmbeddingLoss(_Loss):
    """Creates a criterion that measures the loss given input tensors and a target tensor with values 1 or -1."""

    def __init__(
        self,
        margin: float = 0.0,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initializes the CosineEmbeddingLoss module.

        Args:
            margin (float, optional): Should be a number from -1 to 1, 0 to 0.5 is recommended. Defaults to 0.0.
            size_average (Any, optional): Deprecated. Defaults to None.
            reduce (Any, optional): Deprecated. Defaults to None.
            reduction (str, optional): Specifies the reduction to apply to the output. Defaults to "mean".
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__()

    def forward(self, input, target):
        """Forward pass.

        Args:
            input (Tensor): input.
            target (Tensor): target.

        Returns:
            Tensor: loss.
        """
        import zero_torch.nn.functional as F

        return F.cosine_embedding_loss(input, target)


class CrossEntropyLoss(_WeightedLoss):
    """This criterion computes the cross entropy loss between input logits and target."""

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Computes the cross entropy loss.

        Args:
            input (Tensor): input tensor.
            target (Tensor): target tensor.

        Returns:
            Tensor: The computed loss.
        """
        import zero_torch.nn.functional as F

        return F.cross_entropy(input, target)

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
        """Initializes the CrossEntropyLoss module.

        Args:
            weight (Tensor, optional): A manual rescaling weight given to each class. Defaults to None.
            size_average (Any, optional): Deprecated. Defaults to None.
            ignore_index (int, optional): Specifies a target value that is ignored and does not contribute to the input gradient. Defaults to -100.
            reduce (Any, optional): Deprecated. Defaults to None.
            reduction (str, optional): Specifies the reduction to apply to the output. Defaults to "mean".
            label_smoothing (float, optional): A float in [0.0, 1.0]. Specifies the amount of smoothing. Defaults to 0.0.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__()


class GaussianNLLLoss(_Loss):
    """Gaussian negative log likelihood loss."""

    def __init__(
        self,
        full: bool = False,
        eps: float = 1e-06,
        reduction: str = "mean",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initializes the GaussianNLLLoss module.

        Args:
            full (bool, optional): Include the constant term in the loss calculation. Defaults to False.
            eps (float, optional): Value used to clamp the variance. Defaults to 1e-06.
            reduction (str, optional): Specifies the reduction to apply to the output. Defaults to "mean".
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__()

    def forward(self, input, target):
        """Forward pass.

        Args:
            input (Tensor): input.
            target (Tensor): target.

        Returns:
            Tensor: loss.
        """
        import zero_torch.nn.functional as F

        return F.gaussian_nll_loss(input, target)


class HingeEmbeddingLoss(_Loss):
    """Measures the loss given an input tensor and a labels tensor."""

    def __init__(
        self,
        margin: float = 1.0,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initializes the HingeEmbeddingLoss module.

        Args:
            margin (float, optional): Has a default value of 1. Defaults to 1.0.
            size_average (Any, optional): Deprecated. Defaults to None.
            reduce (Any, optional): Deprecated. Defaults to None.
            reduction (str, optional): Specifies the reduction to apply to the output. Defaults to "mean".
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__()

    def forward(self, input, target):
        """Forward pass.

        Args:
            input (Tensor): input.
            target (Tensor): target.

        Returns:
            Tensor: loss.
        """
        import zero_torch.nn.functional as F

        return F.hinge_embedding_loss(input, target)


class KLDivLoss(_Loss):
    """The Kullback-Leibler divergence loss measure."""

    def __init__(
        self,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
        log_target: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initializes the KLDivLoss module.

        Args:
            size_average (Any, optional): Deprecated. Defaults to None.
            reduce (Any, optional): Deprecated. Defaults to None.
            reduction (str, optional): Specifies the reduction to apply to the output. Defaults to "mean".
            log_target (bool, optional): Specifies whether target is passed in the log space. Defaults to False.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__()

    def forward(self, input, target):
        """Forward pass.

        Args:
            input (Tensor): input.
            target (Tensor): target.

        Returns:
            Tensor: loss.
        """
        import zero_torch.nn.functional as F

        return F.kl_div(input, target)


class L1Loss(_Loss):
    """Creates a criterion that measures the mean absolute error (MAE) between each element in the input x and target y."""

    def forward(self, input, target):
        """Forward pass.

        Args:
            input (Tensor): input.
            target (Tensor): target.

        Returns:
            Tensor: loss.
        """
        import zero_torch.nn.functional as F

        return F.l1_loss(input, target)

    """Creates a criterion that measures the mean absolute error (MAE) between each element in the input x and target y."""


class MSELoss(_Loss):
    """Creates a criterion that measures the mean squared error (squared L2 norm) between each element in the input x and target y."""

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Computes the MSE loss.

        Args:
            input (Tensor): input tensor.
            target (Tensor): target tensor.

        Returns:
            Tensor: The computed MSE loss.
        """
        import zero_torch.nn.functional as F

        return F.mse_loss(input, target)


class NLLLoss(_WeightedLoss):
    """The negative log likelihood loss."""

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
        """Initializes the NLLLoss module.

        Args:
            weight (Tensor, optional): A manual rescaling weight given to each class. Defaults to None.
            size_average (Any, optional): Deprecated. Defaults to None.
            ignore_index (int, optional): Specifies a target value that is ignored and does not contribute to the input gradient. Defaults to -100.
            reduce (Any, optional): Deprecated. Defaults to None.
            reduction (str, optional): Specifies the reduction to apply to the output. Defaults to "mean".
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__()

    def forward(self, input, target):
        """Forward pass.

        Args:
            input (Tensor): input.
            target (Tensor): target.

        Returns:
            Tensor: loss.
        """
        import zero_torch.nn.functional as F

        return F.nll_loss(input, target)


class NLLLoss2d(NLLLoss):
    """The negative log likelihood loss for 2D images."""

    def forward(self, input, target):
        """Forward pass.

        Args:
            input (Tensor): input.
            target (Tensor): target.

        Returns:
            Tensor: loss.
        """
        import zero_torch.nn.functional as F

        return F.nll_loss(input, target)

    """The negative log likelihood loss for 2D images."""


class PoissonNLLLoss(_Loss):
    """Negative log likelihood loss with Poisson distribution of target."""

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
        """Initializes the PoissonNLLLoss module.

        Args:
            log_input (bool, optional): if True the loss is computed as exp(input) - target * input. Defaults to True.
            full (bool, optional): whether to compute full loss. Defaults to False.
            size_average (Any, optional): Deprecated. Defaults to None.
            eps (float, optional): Small value to avoid evaluation of log(0). Defaults to 1e-08.
            reduce (Any, optional): Deprecated. Defaults to None.
            reduction (str, optional): Specifies the reduction to apply to the output. Defaults to "mean".
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__()

    def forward(self, input, target):
        """Forward pass.

        Args:
            input (Tensor): input.
            target (Tensor): target.

        Returns:
            Tensor: loss.
        """
        import zero_torch.nn.functional as F

        return F.poisson_nll_loss(input, target)


class SmoothL1Loss(_Loss):
    """Creates a criterion that uses a squared term if the absolute element-wise error falls below beta and an L1 term otherwise."""

    def __init__(
        self,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
        beta: float = 1.0,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initializes the SmoothL1Loss module.

        Args:
            size_average (Any, optional): Deprecated. Defaults to None.
            reduce (Any, optional): Deprecated. Defaults to None.
            reduction (str, optional): Specifies the reduction to apply to the output. Defaults to "mean".
            beta (float, optional): Specifies the threshold at which to change between L1 and L2 loss. Defaults to 1.0.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__()

    def forward(self, input, target):
        """Forward pass.

        Args:
            input (Tensor): input.
            target (Tensor): target.

        Returns:
            Tensor: loss.
        """
        import zero_torch.nn.functional as F

        return F.smooth_l1_loss(input, target)


class SoftMarginLoss(_Loss):
    """Creates a criterion that optimizes a two-class classification logistic loss."""

    def forward(self, input, target):
        """Forward pass.

        Args:
            input (Tensor): input.
            target (Tensor): target.

        Returns:
            Tensor: loss.
        """
        import zero_torch.nn.functional as F

        return F.soft_margin_loss(input, target)

    """Creates a criterion that optimizes a two-class classification logistic loss."""


class TripletMarginLoss(_Loss):
    """Creates a criterion that measures the triplet loss given an input tensors x1, x2, x3."""

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
        """Initializes the TripletMarginLoss module.

        Args:
            margin (float, optional): Default value is 1. Defaults to 1.0.
            p (float, optional): The norm degree for pairwise distance. Defaults to 2.0.
            eps (float, optional): Small value to avoid numerical issues. Defaults to 1e-06.
            swap (bool, optional): The distance swap is described in detail in the paper Learning shallow convolutional feature descriptors with triplet losses. Defaults to False.
            size_average (Any, optional): Deprecated. Defaults to None.
            reduce (Any, optional): Deprecated. Defaults to None.
            reduction (str, optional): Specifies the reduction to apply to the output. Defaults to "mean".
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__()

    def forward(self, anchor, positive, negative):
        """Forward pass.

        Args:
            anchor (Tensor): anchor.
            positive (Tensor): positive.
            negative (Tensor): negative.

        Returns:
            Tensor: loss.
        """
        import zero_torch.nn.functional as F

        return F.triplet_margin_loss(anchor, positive, negative)


class TripletMarginWithDistanceLoss(_Loss):
    """Creates a criterion that measures the triplet loss given input tensors and a custom distance function."""

    def __init__(
        self,
        distance_function: Any = None,
        margin: float = 1.0,
        swap: bool = False,
        reduction: str = "mean",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initializes the TripletMarginWithDistanceLoss module.

        Args:
            distance_function (Any, optional): A nonnegative, real-valued function that quantifies the closeness of two tensors. Defaults to None.
            margin (float, optional): Default value is 1. Defaults to 1.0.
            swap (bool, optional): Distance swap as described in Learning shallow convolutional feature descriptors. Defaults to False.
            reduction (str, optional): Specifies the reduction to apply to the output. Defaults to "mean".
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__()

    def forward(self, anchor, positive, negative):
        """Forward pass.

        Args:
            anchor (Tensor): anchor.
            positive (Tensor): positive.
            negative (Tensor): negative.

        Returns:
            Tensor: loss.
        """
        import zero_torch.nn.functional as F

        return F.triplet_margin_with_distance_loss(anchor, positive, negative)


class AdaptiveLogSoftmaxWithLoss(Module):
    """AdaptiveLogSoftmaxWithLoss."""

    def __init__(
        self,
        in_features: int,
        n_classes: int,
        cutoffs: list[int],
        div_value: float = 4.0,
        head_bias: bool = False,
        head: Any = None,
        tail: Any = None,
        device: Any = None,
        dtype: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize."""
        super().__init__()
        self.in_features = in_features
        self.n_classes = n_classes
        self.cutoffs = cutoffs
        self.div_value = div_value
        self.head_bias = head_bias

    def forward(self, input: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
        """Forward."""
        from .functional_utils import adaptive_log_softmax_with_loss

        return adaptive_log_softmax_with_loss(
            input,
            target,
            self.in_features,
            self.n_classes,
            self.cutoffs,
            self.div_value,
            self.head_bias,
        )


class HuberLoss(_Loss):
    """Applies HuberLoss."""

    def __init__(self, reduction: str = "mean", delta: float = 1.0) -> None:
        super().__init__(reduction=reduction)
        self.delta = delta

    def forward(self, input, target):
        """Forward pass.

        Args:
            input (Tensor): input.
            target (Tensor): target.

        Returns:
            Tensor: loss.
        """
        import zero_torch.nn.functional as F

        return F.huber_loss(input, target)


class MarginRankingLoss(_Loss):
    """Applies MarginRankingLoss."""

    def __init__(
        self,
        margin: float = 0.0,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
    ) -> None:
        super().__init__(size_average, reduce, reduction)
        self.margin = margin

    def forward(self, input1, input2, target):
        """Forward pass.

        Args:
            input1 (Tensor): input1.
            input2 (Tensor): input2.
            target (Tensor): target.

        Returns:
            Tensor: loss.
        """
        import zero_torch.nn.functional as F

        return F.margin_ranking_loss(input1, input2, target)


class MultiLabelMarginLoss(_Loss):
    """Applies MultiLabelMarginLoss."""

    def __init__(self, size_average=None, reduce=None, reduction: str = "mean") -> None:
        super().__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        """Forward pass.

        Args:
            input (Tensor): input.
            target (Tensor): target.

        Returns:
            Tensor: loss.
        """
        import zero_torch.nn.functional as F

        return F.multi_label_margin_loss(input, target)


class MultiLabelSoftMarginLoss(_WeightedLoss):
    """Applies MultiLabelSoftMarginLoss."""

    def __init__(
        self, weight=None, size_average=None, reduce=None, reduction: str = "mean"
    ) -> None:
        super().__init__(weight, size_average, reduce, reduction)

    def forward(self, input, target):
        """Forward pass.

        Args:
            input (Tensor): input.
            target (Tensor): target.

        Returns:
            Tensor: loss.
        """
        import zero_torch.nn.functional as F

        return F.multi_label_soft_margin_loss(input, target)


class MultiMarginLoss(_WeightedLoss):
    """Applies MultiMarginLoss."""

    def __init__(
        self,
        p: int = 1,
        margin: float = 1.0,
        weight=None,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
    ) -> None:
        super().__init__(weight, size_average, reduce, reduction)
        self.p = p
        self.margin = margin

    def forward(self, input, target):
        """Forward pass.

        Args:
            input (Tensor): input.
            target (Tensor): target.

        Returns:
            Tensor: loss.
        """
        import zero_torch.nn.functional as F

        return F.multi_margin_loss(input, target)
