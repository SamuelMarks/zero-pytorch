"""Type stubs for torch.nn."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import TypeAlias

from torch import Tensor

DeviceLike: TypeAlias = str | None
DTypeLike: TypeAlias = str | None
Size1d: TypeAlias = int | tuple[int]
Size2d: TypeAlias = int | tuple[int, int]
Size3d: TypeAlias = int | tuple[int, int, int]
OptSize1d: TypeAlias = int | tuple[int | None]
OptSize2d: TypeAlias = int | tuple[int | None, int | None]
OptSize3d: TypeAlias = int | tuple[int | None, int | None, int | None]
PaddingArg1d: TypeAlias = Size1d | str
PaddingArg2d: TypeAlias = Size2d | str
PaddingArg3d: TypeAlias = Size3d | str
ShapeLike: TypeAlias = int | Sequence[int]

class GELU:
    def __init__(self, approximate: str = "none") -> None: ...

class LeakyReLU:
    def __init__(
        self,
        negative_slope: float = 0.01,
        inplace: bool = False,
    ) -> None: ...

class ReLU:
    def __init__(self, inplace: bool = False) -> None: ...

class SiLU:
    def __init__(self, inplace: bool = False) -> None: ...

class Sigmoid:
    def __init__(self) -> None: ...

class Softmax:
    def __init__(self, dim: int | None = None) -> None: ...

class Tanh:
    def __init__(self) -> None: ...

class AdaptiveAvgPool1d:
    def __init__(self, output_size: OptSize1d = ...) -> None: ...

class AdaptiveAvgPool2d:
    def __init__(self, output_size: OptSize2d = ...) -> None: ...

class AdaptiveAvgPool3d:
    def __init__(self, output_size: OptSize3d = ...) -> None: ...

class AdaptiveMaxPool1d:
    def __init__(
        self, output_size: OptSize1d = ..., return_indices: bool = False
    ) -> None: ...

class AdaptiveMaxPool2d:
    def __init__(
        self, output_size: OptSize2d = ..., return_indices: bool = False
    ) -> None: ...

class AdaptiveMaxPool3d:
    def __init__(
        self, output_size: OptSize3d = ..., return_indices: bool = False
    ) -> None: ...

class AlphaDropout:
    def __init__(self, p: float = 0.5, inplace: bool = False) -> None: ...

class AvgPool1d:
    def __init__(
        self,
        kernel_size: Size1d = ...,
        stride: Size1d | None = None,
        padding: Size1d = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
    ) -> None: ...

class AvgPool2d:
    def __init__(
        self,
        kernel_size: Size2d = ...,
        stride: Size2d | None = None,
        padding: Size2d = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        divisor_override: int | None = None,
    ) -> None: ...

class AvgPool3d:
    def __init__(
        self,
        kernel_size: Size3d = ...,
        stride: Size3d | None = None,
        padding: Size3d = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        divisor_override: int | None = None,
    ) -> None: ...

class BatchNorm1d:
    def __init__(
        self,
        num_features: int = ...,
        eps: float = 1e-05,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None: ...

class BatchNorm2d:
    def __init__(
        self,
        num_features: int = ...,
        eps: float = 1e-05,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None: ...

class BatchNorm3d:
    def __init__(
        self,
        num_features: int = ...,
        eps: float = 1e-05,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None: ...

class Bilinear:
    def __init__(
        self,
        in1_features: int = ...,
        in2_features: int = ...,
        out_features: int = ...,
        bias: bool = True,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None: ...

class CELU:
    def __init__(self, alpha: float = 1.0, inplace: bool = False) -> None: ...

class ChannelShuffle:
    def __init__(self, groups: int = ...) -> None: ...

class CircularPad1d:
    def __init__(self, padding: Size1d | tuple[int, int] = ...) -> None: ...

class CircularPad2d:
    def __init__(self, padding: int | tuple[int, int, int, int] = ...) -> None: ...

class CircularPad3d:
    def __init__(
        self, padding: int | tuple[int, int, int, int, int, int] = ...
    ) -> None: ...

class ConstantPad1d:
    def __init__(
        self, padding: Size1d | tuple[int, int] = ..., value: float = ...
    ) -> None: ...

class ConstantPad2d:
    def __init__(
        self,
        padding: int | tuple[int, int, int, int] = ...,
        value: float = ...,
    ) -> None: ...

class ConstantPad3d:
    def __init__(
        self,
        padding: int | tuple[int, int, int, int, int, int] = ...,
        value: float = ...,
    ) -> None: ...

class Container:
    def __init__(self, **kwargs: Module) -> None: ...

class Conv1d:
    def __init__(
        self,
        in_channels: int = ...,
        out_channels: int = ...,
        kernel_size: Size1d = ...,
        stride: Size1d = 1,
        padding: PaddingArg1d = 0,
        dilation: Size1d = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None: ...

class Conv2d:
    def __init__(
        self,
        in_channels: int = ...,
        out_channels: int = ...,
        kernel_size: Size2d = ...,
        stride: Size2d = 1,
        padding: PaddingArg2d = 0,
        dilation: Size2d = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None: ...

class Conv3d:
    def __init__(
        self,
        in_channels: int = ...,
        out_channels: int = ...,
        kernel_size: Size3d = ...,
        stride: Size3d = 1,
        padding: PaddingArg3d = 0,
        dilation: Size3d = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None: ...

class ConvTranspose1d:
    def __init__(
        self,
        in_channels: int = ...,
        out_channels: int = ...,
        kernel_size: Size1d = ...,
        stride: Size1d = 1,
        padding: Size1d = 0,
        output_padding: Size1d = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: Size1d = 1,
        padding_mode: str = "zeros",
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None: ...

class ConvTranspose2d:
    def __init__(
        self,
        in_channels: int = ...,
        out_channels: int = ...,
        kernel_size: Size2d = ...,
        stride: Size2d = 1,
        padding: Size2d = 0,
        output_padding: Size2d = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: Size2d = 1,
        padding_mode: str = "zeros",
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None: ...

class ConvTranspose3d:
    def __init__(
        self,
        in_channels: int = ...,
        out_channels: int = ...,
        kernel_size: Size3d = ...,
        stride: Size3d = 1,
        padding: Size3d = 0,
        output_padding: Size3d = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: Size3d = 1,
        padding_mode: str = "zeros",
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None: ...

class CosineSimilarity:
    def __init__(self, dim: int = 1, eps: float = 1e-08) -> None: ...

class CrossMapLRN2d:
    def __init__(
        self, size: int = ..., alpha: float = 0.0001, beta: float = 0.75, k: float = 1.0
    ) -> None: ...

class Dropout:
    def __init__(self, p: float = 0.5, inplace: bool = False) -> None: ...

class Dropout1d:
    def __init__(self, p: float = 0.5, inplace: bool = False) -> None: ...

class Dropout2d:
    def __init__(self, p: float = 0.5, inplace: bool = False) -> None: ...

class Dropout3d:
    def __init__(self, p: float = 0.5, inplace: bool = False) -> None: ...

class ELU:
    def __init__(self, alpha: float = 1.0, inplace: bool = False) -> None: ...

class Embedding:
    def __init__(
        self,
        num_embeddings: int = ...,
        embedding_dim: int = ...,
        padding_idx: int | None = None,
        max_norm: float | None = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        _weight: Tensor | None = None,
        _freeze: bool = False,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None: ...

class EmbeddingBag:
    def __init__(
        self,
        num_embeddings: int = ...,
        embedding_dim: int = ...,
        max_norm: float | None = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        mode: str = "mean",
        sparse: bool = False,
        _weight: Tensor | None = None,
        include_last_offset: bool = False,
        padding_idx: int | None = None,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None: ...

class FeatureAlphaDropout:
    def __init__(self, p: float = 0.5, inplace: bool = False) -> None: ...

class Flatten:
    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None: ...

class Fold:
    def __init__(
        self,
        output_size: Size2d = ...,
        kernel_size: Size2d = ...,
        dilation: Size2d = 1,
        padding: Size2d = 0,
        stride: Size2d = 1,
    ) -> None: ...

class FractionalMaxPool2d:
    def __init__(
        self,
        kernel_size: Size2d = ...,
        output_size: Size2d | None = None,
        output_ratio: float | tuple[float, float] | None = None,
        return_indices: bool = False,
        _random_samples: Tensor | None = None,
    ) -> None: ...

class FractionalMaxPool3d:
    def __init__(
        self,
        kernel_size: Size3d = ...,
        output_size: Size3d | None = None,
        output_ratio: float | tuple[float, float, float] | None = None,
        return_indices: bool = False,
        _random_samples: Tensor | None = None,
    ) -> None: ...

class GLU:
    def __init__(self, dim: int = -1) -> None: ...

class GRU:
    def __init__(
        self,
        input_size: int = ...,
        hidden_size: int = ...,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None: ...

class GRUCell:
    def __init__(
        self,
        input_size: int = ...,
        hidden_size: int = ...,
        bias: bool = True,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None: ...

class GroupNorm:
    def __init__(
        self,
        num_groups: int = ...,
        num_channels: int = ...,
        eps: float = 1e-05,
        affine: bool = True,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None: ...

class Hardshrink:
    def __init__(self, lambd: float = 0.5) -> None: ...

class Hardsigmoid:
    def __init__(self, inplace: bool = False) -> None: ...

class Hardswish:
    def __init__(self, inplace: bool = False) -> None: ...

class Hardtanh:
    def __init__(
        self,
        min_val: float = -1.0,
        max_val: float = 1.0,
        inplace: bool = False,
        min_value: float | None = None,
        max_value: float | None = None,
    ) -> None: ...

class Identity:
    def __init__(
        self, *args: Tensor | int | str, **kwargs: Tensor | int | str
    ) -> None: ...

class InstanceNorm1d:
    def __init__(
        self,
        num_features: int = ...,
        eps: float = 1e-05,
        momentum: float = 0.1,
        affine: bool = False,
        track_running_stats: bool = False,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None: ...

class InstanceNorm2d:
    def __init__(
        self,
        num_features: int = ...,
        eps: float = 1e-05,
        momentum: float = 0.1,
        affine: bool = False,
        track_running_stats: bool = False,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None: ...

class InstanceNorm3d:
    def __init__(
        self,
        num_features: int = ...,
        eps: float = 1e-05,
        momentum: float = 0.1,
        affine: bool = False,
        track_running_stats: bool = False,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None: ...

class LPPool1d:
    def __init__(
        self,
        norm_type: float = ...,
        kernel_size: Size1d = ...,
        stride: Size1d | None = None,
        ceil_mode: bool = False,
    ) -> None: ...

class LPPool2d:
    def __init__(
        self,
        norm_type: float = ...,
        kernel_size: Size2d = ...,
        stride: Size2d | None = None,
        ceil_mode: bool = False,
    ) -> None: ...

class LPPool3d:
    def __init__(
        self,
        norm_type: float = ...,
        kernel_size: Size3d = ...,
        stride: Size3d | None = None,
        ceil_mode: bool = False,
    ) -> None: ...

class LSTM:
    def __init__(
        self,
        input_size: int = ...,
        hidden_size: int = ...,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
        proj_size: int = 0,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None: ...

class LSTMCell:
    def __init__(
        self,
        input_size: int = ...,
        hidden_size: int = ...,
        bias: bool = True,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None: ...

class LayerNorm:
    def __init__(
        self,
        normalized_shape: ShapeLike = ...,
        eps: float = 1e-05,
        elementwise_affine: bool = True,
        bias: bool = True,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None: ...

class LazyBatchNorm1d:
    def __init__(
        self,
        eps: float = 1e-05,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None: ...

class LazyBatchNorm2d:
    def __init__(
        self,
        eps: float = 1e-05,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None: ...

class LazyBatchNorm3d:
    def __init__(
        self,
        eps: float = 1e-05,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None: ...

class LazyConv1d:
    def __init__(
        self,
        out_channels: int = ...,
        kernel_size: Size1d = ...,
        stride: Size1d = 1,
        padding: PaddingArg1d = 0,
        dilation: Size1d = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None: ...

class LazyConv2d:
    def __init__(
        self,
        out_channels: int = ...,
        kernel_size: Size2d = ...,
        stride: Size2d = 1,
        padding: PaddingArg2d = 0,
        dilation: Size2d = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None: ...

class LazyConv3d:
    def __init__(
        self,
        out_channels: int = ...,
        kernel_size: Size3d = ...,
        stride: Size3d = 1,
        padding: PaddingArg3d = 0,
        dilation: Size3d = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None: ...

class LazyConvTranspose1d:
    def __init__(
        self,
        out_channels: int = ...,
        kernel_size: Size1d = ...,
        stride: Size1d = 1,
        padding: Size1d = 0,
        output_padding: Size1d = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: Size1d = 1,
        padding_mode: str = "zeros",
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None: ...

class LazyConvTranspose2d:
    def __init__(
        self,
        out_channels: int = ...,
        kernel_size: Size2d = ...,
        stride: Size2d = 1,
        padding: Size2d = 0,
        output_padding: Size2d = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: Size2d = 1,
        padding_mode: str = "zeros",
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None: ...

class LazyConvTranspose3d:
    def __init__(
        self,
        out_channels: int = ...,
        kernel_size: Size3d = ...,
        stride: Size3d = 1,
        padding: Size3d = 0,
        output_padding: Size3d = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: Size3d = 1,
        padding_mode: str = "zeros",
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None: ...

class LazyInstanceNorm1d:
    def __init__(
        self,
        eps: float = 1e-05,
        momentum: float = 0.1,
        affine: bool = False,
        track_running_stats: bool = False,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None: ...

class LazyInstanceNorm2d:
    def __init__(
        self,
        eps: float = 1e-05,
        momentum: float = 0.1,
        affine: bool = False,
        track_running_stats: bool = False,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None: ...

class LazyInstanceNorm3d:
    def __init__(
        self,
        eps: float = 1e-05,
        momentum: float = 0.1,
        affine: bool = False,
        track_running_stats: bool = False,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None: ...

class LazyLinear:
    def __init__(
        self,
        out_features: int = ...,
        bias: bool = True,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None: ...

class Linear:
    def __init__(
        self,
        in_features: int = ...,
        out_features: int = ...,
        bias: bool = True,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None: ...

class LocalResponseNorm:
    def __init__(
        self, size: int = ..., alpha: float = 0.0001, beta: float = 0.75, k: float = 1.0
    ) -> None: ...

class LogSigmoid:
    def __init__(self) -> None: ...

class LogSoftmax:
    def __init__(self, dim: int | None = None) -> None: ...

class MaxPool1d:
    def __init__(
        self,
        kernel_size: Size1d = ...,
        stride: Size1d | None = None,
        padding: Size1d = 0,
        dilation: Size1d = 1,
        return_indices: bool = False,
        ceil_mode: bool = False,
    ) -> None: ...

class MaxPool2d:
    def __init__(
        self,
        kernel_size: Size2d = ...,
        stride: Size2d | None = None,
        padding: Size2d = 0,
        dilation: Size2d = 1,
        return_indices: bool = False,
        ceil_mode: bool = False,
    ) -> None: ...

class MaxPool3d:
    def __init__(
        self,
        kernel_size: Size3d = ...,
        stride: Size3d | None = None,
        padding: Size3d = 0,
        dilation: Size3d = 1,
        return_indices: bool = False,
        ceil_mode: bool = False,
    ) -> None: ...

class MaxUnpool1d:
    def __init__(
        self,
        kernel_size: Size1d = ...,
        stride: Size1d | None = None,
        padding: Size1d = 0,
    ) -> None: ...

class MaxUnpool2d:
    def __init__(
        self,
        kernel_size: Size2d = ...,
        stride: Size2d | None = None,
        padding: Size2d = 0,
    ) -> None: ...

class MaxUnpool3d:
    def __init__(
        self,
        kernel_size: Size3d = ...,
        stride: Size3d | None = None,
        padding: Size3d = 0,
    ) -> None: ...

class Mish:
    def __init__(self, inplace: bool = False) -> None: ...

class Module:
    def __init__(
        self, *args: Tensor | int | str, **kwargs: Tensor | int | str
    ) -> None: ...

class ModuleDict:
    def __init__(
        self,
        modules: Iterable[tuple[str, Module]] | dict[str, Module] | None = None,
    ) -> None: ...

class ModuleList:
    def __init__(self, modules: Iterable[Module] | None = None) -> None: ...

class MultiheadAttention:
    def __init__(
        self,
        embed_dim: int = ...,
        num_heads: int = ...,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        kdim: int | None = None,
        vdim: int | None = None,
        batch_first: bool = False,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None: ...

class NLLLoss2d:
    def __init__(
        self,
        weight: Tensor | None = None,
        size_average: bool | None = None,
        ignore_index: int = -100,
        reduce: bool | None = None,
        reduction: str = "mean",
    ) -> None: ...

class PReLU:
    def __init__(
        self,
        num_parameters: int = 1,
        init: float = 0.25,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None: ...

class PairwiseDistance:
    def __init__(
        self,
        p: float = 2.0,
        eps: float = 1e-06,
        keepdim: bool = False,
    ) -> None: ...

class ParameterDict:
    def __init__(
        self,
        parameters: dict[str, Tensor] | Iterable[tuple[str, Tensor]] | None = None,
    ) -> None: ...

class ParameterList:
    def __init__(self, values: Iterable[Tensor] | None = None) -> None: ...

class PixelShuffle:
    def __init__(self, upscale_factor: int = ...) -> None: ...

class PixelUnshuffle:
    def __init__(self, downscale_factor: int = ...) -> None: ...

class RMSNorm:
    def __init__(
        self,
        normalized_shape: ShapeLike = ...,
        eps: float | None = None,
        elementwise_affine: bool = True,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None: ...

class RNN:
    def __init__(
        self,
        input_size: int = ...,
        hidden_size: int = ...,
        num_layers: int = 1,
        nonlinearity: str = "tanh",
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None: ...

class RNNBase:
    def __init__(
        self,
        mode: str = ...,
        input_size: int = ...,
        hidden_size: int = ...,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
        proj_size: int = 0,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None: ...

class RNNCell:
    def __init__(
        self,
        input_size: int = ...,
        hidden_size: int = ...,
        bias: bool = True,
        nonlinearity: str = "tanh",
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None: ...

class RNNCellBase:
    def __init__(
        self,
        input_size: int = ...,
        hidden_size: int = ...,
        bias: bool = ...,
        num_chunks: int = ...,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None: ...

class RReLU:
    def __init__(
        self,
        lower: float = 0.125,
        upper: float = 0.3333333333333333,
        inplace: bool = False,
    ) -> None: ...

class ReLU6:
    def __init__(self, inplace: bool = False) -> None: ...

class ReflectionPad1d:
    def __init__(self, padding: Size1d | tuple[int, int] = ...) -> None: ...

class ReflectionPad2d:
    def __init__(self, padding: int | tuple[int, int, int, int] = ...) -> None: ...

class ReflectionPad3d:
    def __init__(
        self, padding: int | tuple[int, int, int, int, int, int] = ...
    ) -> None: ...

class ReplicationPad1d:
    def __init__(self, padding: Size1d | tuple[int, int] = ...) -> None: ...

class ReplicationPad2d:
    def __init__(self, padding: int | tuple[int, int, int, int] = ...) -> None: ...

class ReplicationPad3d:
    def __init__(
        self, padding: int | tuple[int, int, int, int, int, int] = ...
    ) -> None: ...

class SELU:
    def __init__(self, inplace: bool = False) -> None: ...

class Sequential:
    def __init__(self, *args: Module) -> None: ...

class Softmax2d:
    def __init__(self) -> None: ...

class Softmin:
    def __init__(self, dim: int | None = None) -> None: ...

class Softplus:
    def __init__(self, beta: float = 1.0, threshold: float = 20.0) -> None: ...

class Softshrink:
    def __init__(self, lambd: float = 0.5) -> None: ...

class Softsign:
    def __init__(self) -> None: ...

class SyncBatchNorm:
    def __init__(
        self,
        num_features: int = ...,
        eps: float = 1e-05,
        momentum: float | None = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        process_group: str | None = None,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None: ...

class Tanhshrink:
    def __init__(self) -> None: ...

class Threshold:
    def __init__(
        self,
        threshold: float = ...,
        value: float = ...,
        inplace: bool = False,
    ) -> None: ...

class Transformer:
    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str | Callable[[Tensor], Tensor] = "relu",
        custom_encoder: Module | None = None,
        custom_decoder: Module | None = None,
        layer_norm_eps: float = 1e-05,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None: ...

class TransformerDecoder:
    def __init__(
        self,
        decoder_layer: TransformerDecoderLayer = ...,
        num_layers: int = ...,
        norm: Module | None = None,
    ) -> None: ...

class TransformerDecoderLayer:
    def __init__(
        self,
        d_model: int = ...,
        nhead: int = ...,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str | Callable[[Tensor], Tensor] = "relu",
        layer_norm_eps: float = 1e-05,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None: ...

class TransformerEncoder:
    def __init__(
        self,
        encoder_layer: TransformerEncoderLayer = ...,
        num_layers: int = ...,
        norm: Module | None = None,
        enable_nested_tensor: bool = True,
        mask_check: bool = True,
    ) -> None: ...

class TransformerEncoderLayer:
    def __init__(
        self,
        d_model: int = ...,
        nhead: int = ...,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str | Callable[[Tensor], Tensor] = "relu",
        layer_norm_eps: float = 1e-05,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None: ...

class Unflatten:
    def __init__(
        self,
        dim: int | str = ...,
        unflattened_size: ShapeLike = ...,
    ) -> None: ...

class Unfold:
    def __init__(
        self,
        kernel_size: Size2d = ...,
        dilation: Size2d = 1,
        padding: Size2d = 0,
        stride: Size2d = 1,
    ) -> None: ...

class Upsample:
    def __init__(
        self,
        size: int | Sequence[int] | None = None,
        scale_factor: float | Sequence[float] | None = None,
        mode: str = "nearest",
        align_corners: bool | None = None,
        recompute_scale_factor: bool | None = None,
    ) -> None: ...

class UpsamplingBilinear2d:
    def __init__(
        self,
        size: int | tuple[int, int] | None = None,
        scale_factor: float | tuple[float, float] | None = None,
    ) -> None: ...

class UpsamplingNearest2d:
    def __init__(
        self,
        size: int | tuple[int, int] | None = None,
        scale_factor: float | tuple[float, float] | None = None,
    ) -> None: ...

class ZeroPad1d:
    def __init__(self, padding: Size1d | tuple[int, int] = ...) -> None: ...

class ZeroPad2d:
    def __init__(self, padding: int | tuple[int, int, int, int] = ...) -> None: ...

class ZeroPad3d:
    def __init__(
        self, padding: int | tuple[int, int, int, int, int, int] = ...
    ) -> None: ...

class AdaptiveLogSoftmaxWithLoss:
    def __init__(
        self,
        in_features: int = ...,
        n_classes: int = ...,
        cutoffs: Sequence[int] = ...,
        div_value: float = 4.0,
        head_bias: bool = False,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None: ...

class BCELoss:
    def __init__(
        self,
        weight: Tensor | None = None,
        size_average: bool | None = None,
        reduce: bool | None = None,
        reduction: str = "mean",
    ) -> None: ...

class BCEWithLogitsLoss:
    def __init__(
        self,
        weight: Tensor | None = None,
        size_average: bool | None = None,
        reduce: bool | None = None,
        reduction: str = "mean",
        pos_weight: Tensor | None = None,
    ) -> None: ...

class CTCLoss:
    def __init__(
        self,
        blank: int = 0,
        reduction: str = "mean",
        zero_infinity: bool = False,
    ) -> None: ...

class CosineEmbeddingLoss:
    def __init__(
        self,
        margin: float = 0.0,
        size_average: bool | None = None,
        reduce: bool | None = None,
        reduction: str = "mean",
    ) -> None: ...

class CrossEntropyLoss:
    def __init__(
        self,
        weight: Tensor | None = None,
        size_average: bool | None = None,
        ignore_index: int = -100,
        reduce: bool | None = None,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ) -> None: ...

class GaussianNLLLoss:
    def __init__(
        self,
        full: bool = False,
        eps: float = 1e-06,
        reduction: str = "mean",
    ) -> None: ...

class HingeEmbeddingLoss:
    def __init__(
        self,
        margin: float = 1.0,
        size_average: bool | None = None,
        reduce: bool | None = None,
        reduction: str = "mean",
    ) -> None: ...

class HuberLoss:
    def __init__(
        self,
        reduction: str = "mean",
        delta: float = 1.0,
    ) -> None: ...

class KLDivLoss:
    def __init__(
        self,
        size_average: bool | None = None,
        reduce: bool | None = None,
        reduction: str = "mean",
        log_target: bool = False,
    ) -> None: ...

class L1Loss:
    def __init__(
        self,
        size_average: bool | None = None,
        reduce: bool | None = None,
        reduction: str = "mean",
    ) -> None: ...

class MSELoss:
    def __init__(
        self,
        size_average: bool | None = None,
        reduce: bool | None = None,
        reduction: str = "mean",
    ) -> None: ...

class MarginRankingLoss:
    def __init__(
        self,
        margin: float = 0.0,
        size_average: bool | None = None,
        reduce: bool | None = None,
        reduction: str = "mean",
    ) -> None: ...

class MultiLabelMarginLoss:
    def __init__(
        self,
        size_average: bool | None = None,
        reduce: bool | None = None,
        reduction: str = "mean",
    ) -> None: ...

class MultiLabelSoftMarginLoss:
    def __init__(
        self,
        weight: Tensor | None = None,
        size_average: bool | None = None,
        reduce: bool | None = None,
        reduction: str = "mean",
    ) -> None: ...

class MultiMarginLoss:
    def __init__(
        self,
        p: int = 1,
        margin: float = 1.0,
        weight: Tensor | None = None,
        size_average: bool | None = None,
        reduce: bool | None = None,
        reduction: str = "mean",
    ) -> None: ...

class NLLLoss:
    def __init__(
        self,
        weight: Tensor | None = None,
        size_average: bool | None = None,
        ignore_index: int = -100,
        reduce: bool | None = None,
        reduction: str = "mean",
    ) -> None: ...

class PoissonNLLLoss:
    def __init__(
        self,
        log_input: bool = True,
        full: bool = False,
        eps: float = 1e-08,
        size_average: bool | None = None,
        reduce: bool | None = None,
        reduction: str = "mean",
    ) -> None: ...

class SmoothL1Loss:
    def __init__(
        self,
        size_average: bool | None = None,
        reduce: bool | None = None,
        reduction: str = "mean",
        beta: float = 1.0,
    ) -> None: ...

class SoftMarginLoss:
    def __init__(
        self,
        size_average: bool | None = None,
        reduce: bool | None = None,
        reduction: str = "mean",
    ) -> None: ...

class TripletMarginLoss:
    def __init__(
        self,
        margin: float = 1.0,
        p: float = 2.0,
        eps: float = 1e-06,
        swap: bool = False,
        size_average: bool | None = None,
        reduce: bool | None = None,
        reduction: str = "mean",
    ) -> None: ...

class TripletMarginWithDistanceLoss:
    def __init__(
        self,
        distance_function: Callable[[Tensor, Tensor], Tensor] | None = None,
        margin: float = 1.0,
        swap: bool = False,
        reduction: str = "mean",
    ) -> None: ...
