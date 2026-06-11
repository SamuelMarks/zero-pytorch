"Package init."

__all__ = [
    "AdaptiveAvgPool1d",
    "AdaptiveAvgPool2d",
    "AdaptiveAvgPool3d",
    "AdaptiveLogSoftmaxWithLoss",
    "AdaptiveMaxPool1d",
    "AdaptiveMaxPool2d",
    "AdaptiveMaxPool3d",
    "AlphaDropout",
    "AvgPool1d",
    "AvgPool2d",
    "AvgPool3d",
    "BCELoss",
    "BCEWithLogitsLoss",
    "BatchNorm1d",
    "BatchNorm2d",
    "BatchNorm3d",
    "Bilinear",
    "CELU",
    "CTCLoss",
    "ChannelShuffle",
    "CircularPad1d",
    "CircularPad2d",
    "CircularPad3d",
    "ConstantPad1d",
    "ConstantPad2d",
    "ConstantPad3d",
    "Container",
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "ConvTranspose1d",
    "ConvTranspose2d",
    "ConvTranspose3d",
    "CosineEmbeddingLoss",
    "CosineSimilarity",
    "CrossEntropyLoss",
    "CrossMapLRN2d",
    "DataParallel",
    "Dropout",
    "Dropout1d",
    "Dropout2d",
    "Dropout3d",
    "ELU",
    "Embedding",
    "EmbeddingBag",
    "F",
    "FeatureAlphaDropout",
    "Flatten",
    "Fold",
    "FractionalMaxPool2d",
    "FractionalMaxPool3d",
    "GELU",
    "GLU",
    "GRU",
    "GRUCell",
    "GaussianNLLLoss",
    "GroupNorm",
    "Hardshrink",
    "Hardsigmoid",
    "Hardswish",
    "Hardtanh",
    "HingeEmbeddingLoss",
    "HuberLoss",
    "Identity",
    "InstanceNorm1d",
    "InstanceNorm2d",
    "InstanceNorm3d",
    "KLDivLoss",
    "L1Loss",
    "LPPool1d",
    "LPPool2d",
    "LPPool3d",
    "LSTM",
    "LSTMCell",
    "LayerNorm",
    "LazyBatchNorm1d",
    "LazyBatchNorm2d",
    "LazyBatchNorm3d",
    "LazyConv1d",
    "LazyConv2d",
    "LazyConv3d",
    "LazyConvTranspose1d",
    "LazyConvTranspose2d",
    "LazyConvTranspose3d",
    "LazyInstanceNorm1d",
    "LazyInstanceNorm2d",
    "LazyInstanceNorm3d",
    "LazyLinear",
    "LeakyReLU",
    "Linear",
    "LocalResponseNorm",
    "LogSigmoid",
    "LogSoftmax",
    "MSELoss",
    "MarginRankingLoss",
    "MaxPool1d",
    "MaxPool2d",
    "MaxPool3d",
    "MaxUnpool1d",
    "MaxUnpool2d",
    "MaxUnpool3d",
    "Mish",
    "Module",
    "ModuleDict",
    "ModuleList",
    "MultiLabelMarginLoss",
    "MultiLabelSoftMarginLoss",
    "MultiMarginLoss",
    "MultiheadAttention",
    "NLLLoss",
    "NLLLoss2d",
    "PReLU",
    "PairwiseDistance",
    "Parameter",
    "ParameterDict",
    "ParameterList",
    "PixelShuffle",
    "PixelUnshuffle",
    "PoissonNLLLoss",
    "RMSNorm",
    "RNN",
    "RNNBase",
    "RNNCell",
    "RNNCellBase",
    "RReLU",
    "ReLU",
    "ReLU6",
    "ReflectionPad1d",
    "ReflectionPad2d",
    "ReflectionPad3d",
    "ReplicationPad1d",
    "ReplicationPad2d",
    "ReplicationPad3d",
    "SELU",
    "Sequential",
    "SiLU",
    "Sigmoid",
    "SmoothL1Loss",
    "SoftMarginLoss",
    "Softmax",
    "Softmax2d",
    "Softmin",
    "Softplus",
    "Softshrink",
    "Softsign",
    "SyncBatchNorm",
    "Tanh",
    "Tanhshrink",
    "Threshold",
    "Transformer",
    "TransformerDecoder",
    "TransformerDecoderLayer",
    "TransformerEncoder",
    "TransformerEncoderLayer",
    "TripletMarginLoss",
    "TripletMarginWithDistanceLoss",
    "Unflatten",
    "Unfold",
    "Upsample",
    "UpsamplingBilinear2d",
    "UpsamplingNearest2d",
    "ZeroPad1d",
    "ZeroPad2d",
    "ZeroPad3d",
]
from typing import Any
from .module import Module, Parameter, ModuleList, ParameterList
from . import functional as F
from .identity import Identity
from .linear import Linear
from .relu import ReLU, Sequential
from .parallel import DataParallel
from .loss import (
    BCELoss,
    BCEWithLogitsLoss,
    CTCLoss,
    CosineEmbeddingLoss,
    CrossEntropyLoss,
    GaussianNLLLoss,
    HingeEmbeddingLoss,
    HuberLoss,
    KLDivLoss,
    L1Loss,
    MSELoss,
    MarginRankingLoss,
    MultiLabelMarginLoss,
    MultiLabelSoftMarginLoss,
    MultiMarginLoss,
    NLLLoss,
    NLLLoss2d,
    PoissonNLLLoss,
    SmoothL1Loss,
    SoftMarginLoss,
    TripletMarginLoss,
    TripletMarginWithDistanceLoss,
)
from .dropout import (
    Dropout,
    Dropout1d,
    Dropout2d,
    Dropout3d,
    AlphaDropout,
    FeatureAlphaDropout,
)
from .pooling import (
    AvgPool1d,
    AvgPool2d,
    AvgPool3d,
    AdaptiveMaxPool1d,
    AdaptiveMaxPool2d,
    AdaptiveMaxPool3d,
    AdaptiveAvgPool1d,
    AdaptiveAvgPool2d,
    AdaptiveAvgPool3d,
)


class AdaptiveLogSoftmaxWithLoss:
    """Implementation of the AdaptiveLogSoftmaxWithLoss module."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class BatchNorm1d:
    """Applies BatchNorm1d over a mini-batch of inputs."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class BatchNorm2d:
    """Applies BatchNorm2d over a mini-batch of inputs."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass

    def __call__(self, *args, **kwargs):
        """Executes the call operation.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            Any: The result of the __call__ operation.
        """
        import ml_switcheroo.core.errors

        raise ml_switcheroo.core.errors.UnimplementedMathError


class BatchNorm3d:
    """Applies BatchNorm3d over a mini-batch of inputs."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class Bilinear:
    """Implementation of the Bilinear module."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class CELU:
    """Implementation of the CELU module."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class ChannelShuffle:
    """Implementation of the ChannelShuffle module."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class CircularPad1d:
    """Implementation of the CircularPad1d module."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class CircularPad2d:
    """Implementation of the CircularPad2d module."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class CircularPad3d:
    """Implementation of the CircularPad3d module."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class ConstantPad1d:
    """Implementation of the ConstantPad1d module."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class ConstantPad2d:
    """Implementation of the ConstantPad2d module."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class ConstantPad3d:
    """Implementation of the ConstantPad3d module."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class Container:
    """Implementation of the Container module."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class Conv1d:
    """Applies a Conv1d over an input signal."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class Conv2d:
    """Applies a Conv2d over an input signal."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass

    def __call__(self, *args, **kwargs):
        """Executes the call operation.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            Any: The result of the __call__ operation.
        """
        import ml_switcheroo.core.errors

        raise ml_switcheroo.core.errors.UnimplementedMathError


class Conv3d:
    """Applies a Conv3d over an input signal."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class ConvTranspose1d:
    """Applies a ConvTranspose1d over an input signal."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class ConvTranspose2d:
    """Applies a ConvTranspose2d over an input signal."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class ConvTranspose3d:
    """Applies a ConvTranspose3d over an input signal."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class CosineSimilarity:
    """Implementation of the CosineSimilarity module."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class CrossMapLRN2d:
    """Implementation of the CrossMapLRN2d module."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class ELU:
    """Implementation of the ELU module."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class Embedding:
    """Implementation of the Embedding module."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class EmbeddingBag:
    """Implementation of the EmbeddingBag module."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class Flatten:
    """Implementation of the Flatten module."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class Fold:
    """Implementation of the Fold module."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class FractionalMaxPool2d:
    """Applies FractionalMaxPool2d over an input signal."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class FractionalMaxPool3d:
    """Applies FractionalMaxPool3d over an input signal."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class GELU:
    """Implementation of the GELU module."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class GLU:
    """Implementation of the GLU module."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class GRU:
    """Implementation of the GRU module."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class GRUCell:
    """Implementation of the GRUCell module."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class GroupNorm:
    """Applies GroupNorm over a mini-batch of inputs."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class Hardshrink:
    """Implementation of the Hardshrink module."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class Hardsigmoid:
    """Implementation of the Hardsigmoid module."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class Hardswish:
    """Implementation of the Hardswish module."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class Hardtanh:
    """Implementation of the Hardtanh module."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class InstanceNorm1d:
    """Applies InstanceNorm1d over a mini-batch of inputs."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class InstanceNorm2d:
    """Applies InstanceNorm2d over a mini-batch of inputs."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class InstanceNorm3d:
    """Applies InstanceNorm3d over a mini-batch of inputs."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class LPPool1d:
    """Applies LPPool1d over an input signal."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class LPPool2d:
    """Applies LPPool2d over an input signal."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class LPPool3d:
    """Applies LPPool3d over an input signal."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class LSTM:
    """Implementation of the LSTM module."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class LSTMCell:
    """Implementation of the LSTMCell module."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class LayerNorm:
    """Applies LayerNorm over a mini-batch of inputs."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class LazyBatchNorm1d:
    """Applies LazyBatchNorm1d over a mini-batch of inputs."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class LazyBatchNorm2d:
    """Applies LazyBatchNorm2d over a mini-batch of inputs."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class LazyBatchNorm3d:
    """Applies LazyBatchNorm3d over a mini-batch of inputs."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class LazyConv1d:
    """Applies a LazyConv1d over an input signal."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class LazyConv2d:
    """Applies a LazyConv2d over an input signal."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class LazyConv3d:
    """Applies a LazyConv3d over an input signal."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class LazyConvTranspose1d:
    """Applies a LazyConvTranspose1d over an input signal."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class LazyConvTranspose2d:
    """Applies a LazyConvTranspose2d over an input signal."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class LazyConvTranspose3d:
    """Applies a LazyConvTranspose3d over an input signal."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class LazyInstanceNorm1d:
    """Applies LazyInstanceNorm1d over a mini-batch of inputs."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class LazyInstanceNorm2d:
    """Applies LazyInstanceNorm2d over a mini-batch of inputs."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class LazyInstanceNorm3d:
    """Applies LazyInstanceNorm3d over a mini-batch of inputs."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class LazyLinear:
    """Implementation of the LazyLinear module."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class LeakyReLU:
    """Implementation of the LeakyReLU module."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class LocalResponseNorm:
    """Applies LocalResponseNorm over a mini-batch of inputs."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class LogSigmoid:
    """Implementation of the LogSigmoid module."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class LogSoftmax:
    """Implementation of the LogSoftmax module."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class MaxPool1d:
    """Applies MaxPool1d over an input signal."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class MaxPool2d:
    """Applies MaxPool2d over an input signal."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class MaxPool3d:
    """Applies MaxPool3d over an input signal."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class MaxUnpool1d:
    """Implementation of the MaxUnpool1d module."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class MaxUnpool2d:
    """Implementation of the MaxUnpool2d module."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class MaxUnpool3d:
    """Implementation of the MaxUnpool3d module."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class Mish:
    """Implementation of the Mish module."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class ModuleDict:
    """Implementation of the ModuleDict module."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class MultiheadAttention:
    """Implementation of the MultiheadAttention module."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class PReLU:
    """Implementation of the PReLU module."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class PairwiseDistance:
    """Implementation of the PairwiseDistance module."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class ParameterDict:
    """Implementation of the ParameterDict module."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class PixelShuffle:
    """Implementation of the PixelShuffle module."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class PixelUnshuffle:
    """Implementation of the PixelUnshuffle module."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class RMSNorm:
    """Applies RMSNorm over a mini-batch of inputs."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class RNN:
    """Implementation of the RNN module."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class RNNBase:
    """Implementation of the RNNBase module."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class RNNCell:
    """Implementation of the RNNCell module."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class RNNCellBase:
    """Implementation of the RNNCellBase module."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class RReLU:
    """Implementation of the RReLU module."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class ReLU6:
    """Implementation of the ReLU6 module."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class ReflectionPad1d:
    """Implementation of the ReflectionPad1d module."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class ReflectionPad2d:
    """Implementation of the ReflectionPad2d module."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class ReflectionPad3d:
    """Implementation of the ReflectionPad3d module."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class ReplicationPad1d:
    """Implementation of the ReplicationPad1d module."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class ReplicationPad2d:
    """Implementation of the ReplicationPad2d module."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class ReplicationPad3d:
    """Implementation of the ReplicationPad3d module."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class SELU:
    """Implementation of the SELU module."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class SiLU:
    """Implementation of the SiLU module."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class Sigmoid:
    """Implementation of the Sigmoid module."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class Softmax:
    """Implementation of the Softmax module."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class Softmax2d:
    """Implementation of the Softmax2d module."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class Softmin:
    """Implementation of the Softmin module."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class Softplus:
    """Implementation of the Softplus module."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class Softshrink:
    """Implementation of the Softshrink module."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class Softsign:
    """Implementation of the Softsign module."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class SyncBatchNorm:
    """Applies SyncBatchNorm over a mini-batch of inputs."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class Tanh:
    """Implementation of the Tanh module."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class Tanhshrink:
    """Implementation of the Tanhshrink module."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class Threshold:
    """Implementation of the Threshold module."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class Transformer:
    """Implementation of the Transformer module."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class TransformerDecoder:
    """Implementation of the TransformerDecoder module."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class TransformerDecoderLayer:
    """Implementation of the TransformerDecoderLayer module."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class TransformerEncoder:
    """Implementation of the TransformerEncoder module."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class TransformerEncoderLayer:
    """Implementation of the TransformerEncoderLayer module."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class Unflatten:
    """Implementation of the Unflatten module."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class Unfold:
    """Implementation of the Unfold module."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class Upsample:
    """Implementation of the Upsample module."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class UpsamplingBilinear2d:
    """Implementation of the UpsamplingBilinear2d module."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class UpsamplingNearest2d:
    """Implementation of the UpsamplingNearest2d module."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class ZeroPad1d:
    """Implementation of the ZeroPad1d module."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class ZeroPad2d:
    """Implementation of the ZeroPad2d module."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass


class ZeroPad3d:
    """Implementation of the ZeroPad3d module."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        pass
