"""Loss modules."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import zero_torch.nn.functional as F
from zero_torch._types import DeviceLike, DTypeLike
from zero_torch.tensor import Tensor

from .functional_utils import adaptive_log_softmax_with_loss
from .linear import Linear
from .module import Module, ModuleList
from .relu import Sequential


class _Loss(Module):
    """Base class for all neural network loss functions."""

    __constants__: list[str] = ["reduction"]

    def __init__(
        self,
        size_average: bool | None = None,
        reduce: bool | None = None,
        reduction: str = "mean",
    ) -> None:
        """Initializes the `_Loss` module.

        Args:
            size_average: Deprecated flag (see `reduction`).
            reduce: Deprecated flag (see `reduction`).
            reduction: Specifies the reduction to apply (`'none'`, `'mean'`, `'sum'`).
        """
        _ = (size_average, reduce)
        super().__init__()
        self.reduction: str = reduction


class _WeightedLoss(_Loss):
    """Base class for loss functions that use class weights."""

    def __init__(
        self,
        weight: Tensor | None = None,
        size_average: bool | None = None,
        reduce: bool | None = None,
        reduction: str = "mean",
    ) -> None:
        """Initializes the `_WeightedLoss` module.

        Args:
            weight: Manual rescaling weight given to each class.
            size_average: Deprecated flag.
            reduce: Deprecated flag.
            reduction: Specifies the reduction to apply (`'none'`, `'mean'`, `'sum'`).
        """
        super().__init__(size_average, reduce, reduction)
        self.weight: Tensor | None = weight


class BCELoss(_WeightedLoss):
    """Creates a criterion that measures the Binary Cross Entropy between target and input probabilities."""

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Computes binary cross-entropy loss.

        Args:
            input: Predicted probabilities tensor.
            target: Binary target tensor.

        Returns:
            Computed BCE loss tensor.
        """
        return F.binary_cross_entropy(
            input, target, weight=self.weight, reduction=self.reduction
        )


class BCEWithLogitsLoss(_Loss):
    """Combines a `Sigmoid` layer and the `BCELoss` in one single class."""

    def __init__(
        self,
        weight: Tensor | None = None,
        size_average: bool | None = None,
        reduce: bool | None = None,
        reduction: str = "mean",
        pos_weight: Tensor | None = None,
    ) -> None:
        """Initializes the BCEWithLogitsLoss module.

        Args:
            weight: Manual rescaling weight given to the loss of each batch element.
            size_average: Deprecated flag.
            reduce: Deprecated flag.
            reduction: Specifies the reduction to apply (`'none'`, `'mean'`, `'sum'`).
            pos_weight: Weight of positive examples.
        """
        super().__init__(size_average, reduce, reduction)
        self.weight: Tensor | None = weight
        self.pos_weight: Tensor | None = pos_weight

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Computes binary cross-entropy with logits loss.

        Args:
            input: Unnormalized logits tensor.
            target: Binary target tensor.

        Returns:
            Computed BCE-with-logits loss tensor.
        """
        return F.binary_cross_entropy_with_logits(
            input,
            target,
            weight=self.weight,
            reduction=self.reduction,
            pos_weight=self.pos_weight,
        )


class CTCLoss(_Loss):
    """The Connectionist Temporal Classification loss."""

    __constants__: list[str] = ["blank", "reduction"]

    def __init__(
        self,
        blank: int = 0,
        reduction: str = "mean",
        zero_infinity: bool = False,
    ) -> None:
        """Initializes the CTCLoss module.

        Args:
            blank: Blank label index.
            reduction: Specifies the reduction to apply (`'none'`, `'mean'`, `'sum'`).
            zero_infinity: Whether to zero infinite losses and the associated gradients.
        """
        super().__init__(reduction=reduction)
        self.blank: int = blank
        self.zero_infinity: bool = zero_infinity

    def forward(
        self,
        log_probs: Tensor,
        targets: Tensor,
        input_lengths: Tensor | Sequence[int] | None = None,
        target_lengths: Tensor | Sequence[int] | None = None,
    ) -> Tensor:
        """Computes the CTC loss.

        Args:
            log_probs: Log-probabilities tensor.
            targets: Target sequences tensor.
            input_lengths: Lengths of the inputs.
            target_lengths: Lengths of the targets.

        Returns:
            Computed CTC loss tensor.
        """
        return F.ctc_loss(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            blank=self.blank,
            reduction=self.reduction,
            zero_infinity=self.zero_infinity,
        )


class CosineEmbeddingLoss(_Loss):
    """Measures the loss given two input tensors and a label tensor with values `1` or `-1`."""

    __constants__: list[str] = ["margin", "reduction"]

    def __init__(
        self,
        margin: float = 0.0,
        size_average: bool | None = None,
        reduce: bool | None = None,
        reduction: str = "mean",
    ) -> None:
        """Initializes the CosineEmbeddingLoss module.

        Args:
            margin: Margin value in `[-1, 1]`.
            size_average: Deprecated flag.
            reduce: Deprecated flag.
            reduction: Specifies the reduction to apply (`'none'`, `'mean'`, `'sum'`).
        """
        super().__init__(size_average, reduce, reduction)
        self.margin: float = margin

    def forward(
        self, input1: Tensor, input2: Tensor, target: Tensor | None = None
    ) -> Tensor:
        """Computes cosine embedding loss.

        Args:
            input1: First input tensor.
            input2: Second input tensor (or target when called with 2 args).
            target: Label tensor (`1` or `-1`).

        Returns:
            Computed cosine embedding loss tensor.
        """
        return F.cosine_embedding_loss(
            input1,
            input2,
            target,
            margin=self.margin,
            reduction=self.reduction,
        )


class CrossEntropyLoss(_WeightedLoss):
    """Computes the cross entropy loss between input logits and target."""

    __constants__: list[str] = ["ignore_index", "reduction", "label_smoothing"]

    def __init__(
        self,
        weight: Tensor | None = None,
        size_average: bool | None = None,
        ignore_index: int = -100,
        reduce: bool | None = None,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ) -> None:
        """Initializes the CrossEntropyLoss module.

        Args:
            weight: Manual rescaling weight given to each class.
            size_average: Deprecated flag.
            ignore_index: Target value that is ignored and does not contribute to gradient.
            reduce: Deprecated flag.
            reduction: Specifies the reduction to apply (`'none'`, `'mean'`, `'sum'`).
            label_smoothing: Amount of label smoothing in `[0.0, 1.0]`.
        """
        super().__init__(weight, size_average, reduce, reduction)
        self.ignore_index: int = ignore_index
        self.label_smoothing: float = label_smoothing

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Computes the cross-entropy loss.

        Args:
            input: Predicted unnormalized logits tensor.
            target: Ground-truth class indices or class probabilities tensor.

        Returns:
            Computed cross-entropy loss tensor.
        """
        return F.cross_entropy(
            input,
            target,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            label_smoothing=self.label_smoothing,
        )


class GaussianNLLLoss(_Loss):
    """Gaussian negative log likelihood loss."""

    def __init__(
        self,
        full: bool = False,
        eps: float = 1e-06,
        reduction: str = "mean",
    ) -> None:
        """Initializes the GaussianNLLLoss module.

        Args:
            full: Include the constant term in the loss calculation.
            eps: Value used to clamp the variance for stability.
            reduction: Specifies the reduction to apply (`'none'`, `'mean'`, `'sum'`).
        """
        super().__init__(reduction=reduction)
        self.full: bool = full
        self.eps: float = eps

    def forward(
        self, input: Tensor, target: Tensor, var: Tensor | float = 1.0
    ) -> Tensor:
        """Computes Gaussian negative log likelihood loss.

        Args:
            input: Expectation of the Gaussian distribution.
            target: Sample from the Gaussian distribution.
            var: Tensor of positive variance(s).

        Returns:
            Computed Gaussian NLL loss tensor.
        """
        return F.gaussian_nll_loss(
            input,
            target,
            var,
            full=self.full,
            eps=self.eps,
            reduction=self.reduction,
        )


class HingeEmbeddingLoss(_Loss):
    """Measures the loss given an input tensor `x` and a labels tensor `y` (`1` or `-1`)."""

    def __init__(
        self,
        margin: float = 1.0,
        size_average: bool | None = None,
        reduce: bool | None = None,
        reduction: str = "mean",
    ) -> None:
        """Initializes the HingeEmbeddingLoss module.

        Args:
            margin: Margin threshold `Delta`.
            size_average: Deprecated flag.
            reduce: Deprecated flag.
            reduction: Specifies the reduction to apply (`'none'`, `'mean'`, `'sum'`).
        """
        super().__init__(size_average, reduce, reduction)
        self.margin: float = margin

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Computes hinge embedding loss.

        Args:
            input: Input tensor.
            target: Labels tensor (`1` or `-1`).

        Returns:
            Computed hinge embedding loss tensor.
        """
        return F.hinge_embedding_loss(
            input, target, margin=self.margin, reduction=self.reduction
        )


class KLDivLoss(_Loss):
    """The Kullback-Leibler divergence loss measure."""

    def __init__(
        self,
        size_average: bool | None = None,
        reduce: bool | None = None,
        reduction: str = "mean",
        log_target: bool = False,
    ) -> None:
        """Initializes the KLDivLoss module.

        Args:
            size_average: Deprecated flag.
            reduce: Deprecated flag.
            reduction: Specifies the reduction to apply (`'none'`, `'batchmean'`, `'sum'`, `'mean'`).
            log_target: Specifies whether `target` is passed in log space.
        """
        super().__init__(size_average, reduce, reduction)
        self.log_target: bool = log_target

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Computes Kullback-Leibler divergence loss.

        Args:
            input: Tensor of arbitrary shape in log-probabilities.
            target: Tensor of the same shape as input.

        Returns:
            Computed KL-divergence loss tensor.
        """
        return F.kl_div(
            input, target, reduction=self.reduction, log_target=self.log_target
        )


class L1Loss(_Loss):
    """Creates a criterion that measures the mean absolute error (MAE) between `input` and `target`."""

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Computes L1 (mean absolute error) loss.

        Args:
            input: Input prediction tensor.
            target: Target tensor.

        Returns:
            Computed L1 loss tensor.
        """
        return F.l1_loss(input, target, reduction=self.reduction)


class MSELoss(_Loss):
    """Creates a criterion that measures the mean squared error (squared L2 norm) between `input` and `target`."""

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Computes mean squared error loss.

        Args:
            input: Input prediction tensor.
            target: Target tensor.

        Returns:
            Computed MSE loss tensor.
        """
        return F.mse_loss(input, target, reduction=self.reduction)


class NLLLoss(_WeightedLoss):
    """The negative log likelihood loss."""

    def __init__(
        self,
        weight: Tensor | None = None,
        size_average: bool | None = None,
        ignore_index: int = -100,
        reduce: bool | None = None,
        reduction: str = "mean",
    ) -> None:
        """Initializes the NLLLoss module.

        Args:
            weight: Manual rescaling weight given to each class.
            size_average: Deprecated flag.
            ignore_index: Target value that is ignored and does not contribute to gradient.
            reduce: Deprecated flag.
            reduction: Specifies the reduction to apply (`'none'`, `'mean'`, `'sum'`).
        """
        super().__init__(weight, size_average, reduce, reduction)
        self.ignore_index: int = ignore_index

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Computes negative log likelihood loss.

        Args:
            input: Log-probabilities tensor of shape `(N, C)` or `(N, C, d_1, ...)`.
            target: Class indices tensor.

        Returns:
            Computed NLL loss tensor.
        """
        return F.nll_loss(
            input,
            target,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
        )


class NLLLoss2d(NLLLoss):
    """The negative log likelihood loss for 2D images."""

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Computes 2D negative log likelihood loss.

        Args:
            input: Log-probabilities tensor of shape `(N, C, H, W)`.
            target: Class indices tensor of shape `(N, H, W)`.

        Returns:
            Computed 2D NLL loss tensor.
        """
        return F.nll_loss(
            input,
            target,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
        )


class PoissonNLLLoss(_Loss):
    """Negative log likelihood loss with Poisson distribution of target."""

    def __init__(
        self,
        log_input: bool = True,
        full: bool = False,
        size_average: bool | None = None,
        eps: float = 1e-08,
        reduce: bool | None = None,
        reduction: str = "mean",
    ) -> None:
        """Initializes the PoissonNLLLoss module.

        Args:
            log_input: If True, loss is computed as `exp(input) - target * input`.
            full: Whether to compute full loss including Stirling approximation term.
            size_average: Deprecated flag.
            eps: Small value to avoid evaluation of `log(0)`.
            reduce: Deprecated flag.
            reduction: Specifies the reduction to apply (`'none'`, `'mean'`, `'sum'`).
        """
        super().__init__(size_average, reduce, reduction)
        self.log_input: bool = log_input
        self.full: bool = full
        self.eps: float = eps

    def forward(self, log_input: Tensor, target: Tensor) -> Tensor:
        """Computes Poisson negative log likelihood loss.

        Args:
            log_input: Expectation of underlying Poisson distribution (or log expectation).
            target: Random sample `target ~ Poisson(input)`.

        Returns:
            Computed Poisson NLL loss tensor.
        """
        return F.poisson_nll_loss(
            log_input,
            target,
            log_input=self.log_input,
            full=self.full,
            eps=self.eps,
            reduction=self.reduction,
        )


class SmoothL1Loss(_Loss):
    """Uses a squared term if the absolute element-wise error falls below `beta` and an L1 term otherwise."""

    def __init__(
        self,
        size_average: bool | None = None,
        reduce: bool | None = None,
        reduction: str = "mean",
        beta: float = 1.0,
    ) -> None:
        """Initializes the SmoothL1Loss module.

        Args:
            size_average: Deprecated flag.
            reduce: Deprecated flag.
            reduction: Specifies the reduction to apply (`'none'`, `'mean'`, `'sum'`).
            beta: Threshold at which to change between L1 and L2 loss.
        """
        super().__init__(size_average, reduce, reduction)
        self.beta: float = beta

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Computes smooth L1 loss.

        Args:
            input: Input prediction tensor.
            target: Target tensor.

        Returns:
            Computed smooth L1 loss tensor.
        """
        return F.smooth_l1_loss(input, target, reduction=self.reduction, beta=self.beta)


class SoftMarginLoss(_Loss):
    """Creates a criterion that optimizes a two-class classification logistic loss."""

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Computes soft margin loss.

        Args:
            input: Input tensor.
            target: Target label tensor (`1` or `-1`).

        Returns:
            Computed soft margin loss tensor.
        """
        return F.soft_margin_loss(input, target, reduction=self.reduction)


class TripletMarginLoss(_Loss):
    """Creates a criterion that measures the triplet loss given `anchor`, `positive`, and `negative` tensors."""

    def __init__(
        self,
        margin: float = 1.0,
        p: float = 2.0,
        eps: float = 1e-06,
        swap: bool = False,
        size_average: bool | None = None,
        reduce: bool | None = None,
        reduction: str = "mean",
    ) -> None:
        """Initializes the TripletMarginLoss module.

        Args:
            margin: Margin threshold (> 0).
            p: The norm degree for pairwise distance.
            eps: Small constant for numerical stability.
            swap: Whether to use distance swap.
            size_average: Deprecated flag.
            reduce: Deprecated flag.
            reduction: Specifies the reduction to apply (`'none'`, `'mean'`, `'sum'`).
        """
        super().__init__(size_average, reduce, reduction)
        self.margin: float = margin
        self.p: float = p
        self.eps: float = eps
        self.swap: bool = swap

    def forward(self, anchor: Tensor, positive: Tensor, negative: Tensor) -> Tensor:
        """Computes triplet margin loss.

        Args:
            anchor: Anchor input tensor.
            positive: Positive input tensor.
            negative: Negative input tensor.

        Returns:
            Computed triplet margin loss tensor.
        """
        return F.triplet_margin_loss(
            anchor,
            positive,
            negative,
            margin=self.margin,
            p=self.p,
            eps=self.eps,
            swap=self.swap,
            reduction=self.reduction,
        )


class TripletMarginWithDistanceLoss(_Loss):
    """Measures the triplet loss given input tensors and a custom distance function."""

    def __init__(
        self,
        distance_function: Callable[[Tensor, Tensor], Tensor] | None = None,
        margin: float = 1.0,
        swap: bool = False,
        reduction: str = "mean",
    ) -> None:
        """Initializes the TripletMarginWithDistanceLoss module.

        Args:
            distance_function: Nonnegative real-valued function quantifying closeness of two tensors.
            margin: Margin threshold.
            swap: Whether to use distance swap.
            reduction: Specifies the reduction to apply (`'none'`, `'mean'`, `'sum'`).
        """
        super().__init__(reduction=reduction)
        self.distance_function: Callable[[Tensor, Tensor], Tensor] | None = (
            distance_function
        )
        self.margin: float = margin
        self.swap: bool = swap

    def forward(self, anchor: Tensor, positive: Tensor, negative: Tensor) -> Tensor:
        """Computes triplet margin with distance loss.

        Args:
            anchor: Anchor input tensor.
            positive: Positive input tensor.
            negative: Negative input tensor.

        Returns:
            Computed triplet loss tensor.
        """
        return F.triplet_margin_with_distance_loss(
            anchor,
            positive,
            negative,
            distance_function=self.distance_function,
            margin=self.margin,
            swap=self.swap,
            reduction=self.reduction,
        )


class AdaptiveLogSoftmaxWithLoss(Module):
    """Efficient softmax approximation for large number of classes."""

    def __init__(
        self,
        in_features: int,
        n_classes: int,
        cutoffs: Sequence[int],
        div_value: float = 4.0,
        head_bias: bool = False,
        head: Module | None = None,
        tail: Sequence[Module] | ModuleList | None = None,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        """Initializes AdaptiveLogSoftmaxWithLoss.

        Args:
            in_features: Number of features in the input tensor.
            n_classes: Number of classes in the dataset.
            cutoffs: Cutoffs used to assign targets to their buckets.
            div_value: Value used as an exponent to compute size of clusters.
            head_bias: If True, adds a bias term to the head of the adaptive softmax.
            head: Optional pre-constructed head module.
            tail: Optional pre-constructed sequence of tail modules.
            device: Desired device of parameters.
            dtype: Desired data type of parameters.
        """
        _ = (device, dtype)
        super().__init__()
        self.in_features: int = in_features
        self.n_classes: int = n_classes
        self.cutoffs: list[int] = list(cutoffs) + [n_classes]
        self.div_value: float = div_value
        self.head_bias: bool = head_bias
        self.n_clusters: int = len(self.cutoffs) - 1
        self.head_size: int = self.cutoffs[0] + self.n_clusters

        self.head: Module = (
            head
            if isinstance(head, Module)
            else Linear(in_features, max(1, n_classes), bias=head_bias)
        )
        self.add_module("head", self.head)

        if isinstance(tail, ModuleList):
            self.tail: ModuleList = tail
        else:
            tail_list: list[Module] = []
            for i in range(self.n_clusters):
                proj_sz = max(1, int(in_features // (div_value ** (i + 1))))
                clus_sz = max(1, self.cutoffs[i + 1] - self.cutoffs[i])
                tail_list.append(
                    Sequential(
                        Linear(in_features, proj_sz, bias=False),
                        Linear(proj_sz, clus_sz, bias=False),
                    )
                )
            self.tail = ModuleList(tail_list)
        self.add_module("tail", self.tail)

    def log_prob(self, input: Tensor) -> Tensor:
        """Computes log probabilities for all `n_classes` for each row in `input`.

        Args:
            input: Input tensor of shape `(N, in_features)`.

        Returns:
            Log probability tensor of shape `(N, n_classes)`.
        """
        logits = self.head(input)
        return F.log_softmax(logits, dim=-1)

    def predict(self, input: Tensor) -> Tensor:
        """Returns the class with the highest probability for each row in `input`.

        Args:
            input: Input tensor of shape `(N, in_features)`.

        Returns:
            Predicted class indices tensor of shape `(N,)`.
        """
        return self.log_prob(input).argmax(dim=-1)

    def forward(self, input: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
        """Computes adaptive log-softmax output and loss.

        Args:
            input: Input tensor of shape `(N, in_features)`.
            target: Target class indices tensor of shape `(N,)`.

        Returns:
            Tuple of `(output, loss)` tensors.
        """
        hw = self.head.weight if isinstance(self.head, Linear) else None
        hb = self.head.bias if isinstance(self.head, Linear) else None
        return adaptive_log_softmax_with_loss(
            input,
            target,
            self.in_features,
            self.n_classes,
            self.cutoffs[:-1],
            self.div_value,
            self.head_bias,
            head_weight=hw,
            head_bias_tensor=hb,
        )


class HuberLoss(_Loss):
    """Creates a criterion that uses a squared term if the absolute element-wise error falls below `delta` and a delta-scaled L1 term otherwise."""

    def __init__(self, reduction: str = "mean", delta: float = 1.0) -> None:
        """Initializes HuberLoss.

        Args:
            reduction: Specifies the reduction to apply (`'none'`, `'mean'`, `'sum'`).
            delta: Specifies the threshold at which to change between delta-scaled L1 and L2 loss.
        """
        super().__init__(reduction=reduction)
        self.delta: float = delta

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Computes Huber loss.

        Args:
            input: Input prediction tensor.
            target: Target tensor.

        Returns:
            Computed Huber loss tensor.
        """
        return F.huber_loss(input, target, reduction=self.reduction, delta=self.delta)


class MarginRankingLoss(_Loss):
    """Creates a criterion that measures the loss given inputs `x1`, `x2`, and a label tensor `y` (`1` or `-1`)."""

    def __init__(
        self,
        margin: float = 0.0,
        size_average: bool | None = None,
        reduce: bool | None = None,
        reduction: str = "mean",
    ) -> None:
        """Initializes MarginRankingLoss.

        Args:
            margin: Margin value.
            size_average: Deprecated flag.
            reduce: Deprecated flag.
            reduction: Specifies the reduction to apply (`'none'`, `'mean'`, `'sum'`).
        """
        super().__init__(size_average, reduce, reduction)
        self.margin: float = margin

    def forward(self, input1: Tensor, input2: Tensor, target: Tensor) -> Tensor:
        """Computes margin ranking loss.

        Args:
            input1: First input tensor.
            input2: Second input tensor.
            target: Label tensor (`1` or `-1`).

        Returns:
            Computed margin ranking loss tensor.
        """
        return F.margin_ranking_loss(
            input1, input2, target, margin=self.margin, reduction=self.reduction
        )


class MultiLabelMarginLoss(_Loss):
    """Creates a criterion that optimizes a multi-class multi-classification hinge loss (margin-based loss)."""

    def __init__(
        self,
        size_average: bool | None = None,
        reduce: bool | None = None,
        reduction: str = "mean",
    ) -> None:
        """Initializes MultiLabelMarginLoss.

        Args:
            size_average: Deprecated flag.
            reduce: Deprecated flag.
            reduction: Specifies the reduction to apply (`'none'`, `'mean'`, `'sum'`).
        """
        super().__init__(size_average, reduce, reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Computes multi-label margin loss.

        Args:
            input: Input score tensor.
            target: Target label index tensor.

        Returns:
            Computed multi-label margin loss tensor.
        """
        return F.multi_label_margin_loss(input, target, reduction=self.reduction)


class MultiLabelSoftMarginLoss(_WeightedLoss):
    """Creates a criterion that optimizes a multi-label one-versus-all loss based on max-entropy."""

    def __init__(
        self,
        weight: Tensor | None = None,
        size_average: bool | None = None,
        reduce: bool | None = None,
        reduction: str = "mean",
    ) -> None:
        """Initializes MultiLabelSoftMarginLoss.

        Args:
            weight: Manual rescaling weight given to each class.
            size_average: Deprecated flag.
            reduce: Deprecated flag.
            reduction: Specifies the reduction to apply (`'none'`, `'mean'`, `'sum'`).
        """
        super().__init__(weight, size_average, reduce, reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Computes multi-label soft margin loss.

        Args:
            input: Input logits tensor.
            target: Binary target label tensor.

        Returns:
            Computed multi-label soft margin loss tensor.
        """
        return F.multi_label_soft_margin_loss(
            input, target, weight=self.weight, reduction=self.reduction
        )


class MultiMarginLoss(_WeightedLoss):
    """Creates a criterion that optimizes a multi-class classification hinge loss (margin-based loss)."""

    def __init__(
        self,
        p: int = 1,
        margin: float = 1.0,
        weight: Tensor | None = None,
        size_average: bool | None = None,
        reduce: bool | None = None,
        reduction: str = "mean",
    ) -> None:
        """Initializes MultiMarginLoss.

        Args:
            p: Norm degree (`1` or `2`).
            margin: Margin threshold.
            weight: Manual rescaling weight given to each class.
            size_average: Deprecated flag.
            reduce: Deprecated flag.
            reduction: Specifies the reduction to apply (`'none'`, `'mean'`, `'sum'`).
        """
        super().__init__(weight, size_average, reduce, reduction)
        self.p: int = p
        self.margin: float = margin

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Computes multi-margin loss.

        Args:
            input: Input score tensor.
            target: Target class indices tensor.

        Returns:
            Computed multi-margin loss tensor.
        """
        return F.multi_margin_loss(
            input,
            target,
            p=self.p,
            margin=self.margin,
            weight=self.weight,
            reduction=self.reduction,
        )


class LinearCrossEntropyLoss(_Loss):
    """Fused linear projection and cross-entropy loss module."""

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        *,
        out_features: tuple[int, ...] = (),
        bias: bool = False,
        device: str | None = None,
        dtype: str | None = None,
        reduction: str = "mean",
        weight: Tensor | None = None,
        ignore_index: int | None = None,
        label_smoothing: float = 0.0,
        options: dict[str, int | float | bool | str] | None = None,
    ) -> None:
        """Initializes LinearCrossEntropyLoss.

        Args:
            in_features: Size of each input sample.
            num_classes: Number of target classes.
            out_features: Additional output dimensions.
            bias: If True, adds a learnable bias to the projection.
            device: Desired device of parameters.
            dtype: Desired data type of parameters.
            reduction: Reduction mode (`'none'`, `'mean'`, `'sum'`).
            weight: Manual rescaling weight given to each class.
            ignore_index: Target value that is ignored.
            label_smoothing: Amount of label smoothing in `[0.0, 1.0]`.
            options: Optional configuration dictionary.
        """
        super().__init__(reduction=reduction)
        self.in_features: int = in_features
        self.num_classes: int = num_classes
        self.out_features: tuple[int, ...] = out_features
        self.bias_enabled: bool = bias
        self.device: str | None = device
        self.dtype: str | None = dtype
        self.weight: Tensor | None = weight
        self.ignore_index: int = -100 if ignore_index is None else ignore_index
        self.label_smoothing: float = label_smoothing
        self.options: dict[str, int | float | bool | str] | None = options
        self.linear: Linear = Linear(in_features, num_classes, bias=bias)
        self.add_module("linear", self.linear)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Applies linear projection followed by cross-entropy loss.

        Args:
            input: Input feature tensor.
            target: Target class indices or probabilities.

        Returns:
            Computed cross-entropy loss tensor.
        """
        logits = self.linear(input)
        return F.cross_entropy(
            logits,
            target,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            label_smoothing=self.label_smoothing,
        )
