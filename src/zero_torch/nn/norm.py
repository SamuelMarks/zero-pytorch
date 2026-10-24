"""Normalization modules."""

from __future__ import annotations

from ml_switcheroo_compiler.ops.creation import ones, zeros
from ml_switcheroo_compiler.ops.nn.normalization import (
    BatchNormConfig,
    LRNConfig,
    batch_normalization,
    group_norm,
    instance_norm,
    layer_norm,
    lrn,
    rms_normalization,
)

import zero_torch
from zero_torch._types import DeviceLike, DTypeLike, ShapeLike
from zero_torch.tensor import Tensor, _to_tensor, _wrap

from .functional_norm import _norm_shape_tuple
from .module import Module, Parameter


def _init_batch_norm_state(
    module: Module,
    num_features: int,
    affine: bool,
    track_running_stats: bool,
) -> None:
    """Registers affine parameters and running statistics buffers on a batch/instance norm module.

    Args:
        module: Target normalization module.
        num_features: Number of channel features `C`.
        affine: Whether to register learnable `weight` and `bias` parameters.
        track_running_stats: Whether to register `running_mean`, `running_var`, and `num_batches_tracked` buffers.
    """
    if affine:
        module.register_parameter("weight", Parameter(zero_torch.ones(num_features)))
        module.register_parameter("bias", Parameter(zero_torch.zeros(num_features)))
    else:
        module.register_parameter("weight", None)
        module.register_parameter("bias", None)

    if track_running_stats:
        module.register_buffer("running_mean", zero_torch.zeros(num_features))
        module.register_buffer("running_var", zero_torch.ones(num_features))
        module.register_buffer("num_batches_tracked", zero_torch.tensor(0))
    else:
        module.register_buffer("running_mean", None)
        module.register_buffer("running_var", None)
        module.register_buffer("num_batches_tracked", None)


def _apply_channel_affine(
    out: Tensor,
    weight: Parameter | None,
    bias: Parameter | None,
) -> Tensor:
    """Applies per-channel affine scale and shift along dimension 1.

    Args:
        out: Normalized tensor of shape `(N, C, *)`.
        weight: Optional per-channel scale parameter of shape `(C,)`.
        bias: Optional per-channel bias parameter of shape `(C,)`.

    Returns:
        Affine-transformed tensor.
    """
    ndim = len(out.shape)
    if ndim < 2:
        res = out
        if weight is not None:
            res = res * weight
        if bias is not None:
            res = res + bias
        return res
    bcast_shape = (1, -1) + (1,) * (ndim - 2)
    res = out
    if weight is not None:
        res = res * zero_torch.reshape(weight, bcast_shape)
    if bias is not None:
        res = res + zero_torch.reshape(bias, bcast_shape)
    return res


class BatchNorm1d(Module):
    """Applies Batch Normalization over a 2D or 3D input."""

    weight: Parameter | None
    bias: Parameter | None
    running_mean: Tensor | None
    running_var: Tensor | None
    num_batches_tracked: Tensor | None

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-05,
        momentum: float | None = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        """Initializes the BatchNorm1d module.

        Args:
            num_features: Number of features or channels `C`.
            eps: Value added to the denominator for numerical stability.
            momentum: Value used for the running_mean and running_var computation.
            affine: If True, this module has learnable affine parameters.
            track_running_stats: If True, tracks the running mean and variance.
            device: Desired device of parameters.
            dtype: Desired data type of parameters.
        """
        _ = (device, dtype)
        super().__init__()
        self.num_features: int = num_features
        self.eps: float = eps
        self.momentum: float | None = momentum
        self.affine: bool = affine
        self.track_running_stats: bool = track_running_stats
        _init_batch_norm_state(self, num_features, affine, track_running_stats)

    def forward(self, input: Tensor) -> Tensor:
        """Applies 1D batch normalization to `input`.

        Args:
            input: Input tensor of shape `(N, C)` or `(N, C, L)`.

        Returns:
            Normalized tensor.
        """
        t = _to_tensor(input)
        out = _wrap(
            batch_normalization(
                t,
                zeros(t.shape),
                ones(t.shape),
                axis=1,
                config=BatchNormConfig(epsilon=self.eps),
            )
        )
        return _apply_channel_affine(out, self.weight, self.bias)


class BatchNorm2d(Module):
    """Applies Batch Normalization over a 4D input."""

    weight: Parameter | None
    bias: Parameter | None
    running_mean: Tensor | None
    running_var: Tensor | None
    num_batches_tracked: Tensor | None

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-05,
        momentum: float | None = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        """Initializes the BatchNorm2d module.

        Args:
            num_features: Number of channels `C` from an expected input of size `(N, C, H, W)`.
            eps: Value added to the denominator for numerical stability.
            momentum: Value used for the running_mean and running_var computation.
            affine: If True, this module has learnable affine parameters.
            track_running_stats: If True, tracks the running mean and variance.
            device: Desired device of parameters.
            dtype: Desired data type of parameters.
        """
        _ = (device, dtype)
        super().__init__()
        self.num_features: int = num_features
        self.eps: float = eps
        self.momentum: float | None = momentum
        self.affine: bool = affine
        self.track_running_stats: bool = track_running_stats
        _init_batch_norm_state(self, num_features, affine, track_running_stats)

    def forward(self, input: Tensor) -> Tensor:
        """Applies 2D batch normalization to `input`.

        Args:
            input: Input tensor of shape `(N, C, H, W)`.

        Returns:
            Normalized tensor.
        """
        t = _to_tensor(input)
        out = _wrap(
            batch_normalization(
                t,
                zeros(t.shape),
                ones(t.shape),
                axis=1,
                config=BatchNormConfig(epsilon=self.eps),
            )
        )
        return _apply_channel_affine(out, self.weight, self.bias)


class BatchNorm3d(Module):
    """Applies Batch Normalization over a 5D input."""

    weight: Parameter | None
    bias: Parameter | None
    running_mean: Tensor | None
    running_var: Tensor | None
    num_batches_tracked: Tensor | None

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-05,
        momentum: float | None = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        """Initializes the BatchNorm3d module.

        Args:
            num_features: Number of channels `C` from an expected input of size `(N, C, D, H, W)`.
            eps: Value added to the denominator for numerical stability.
            momentum: Value used for the running_mean and running_var computation.
            affine: If True, this module has learnable affine parameters.
            track_running_stats: If True, tracks the running mean and variance.
            device: Desired device of parameters.
            dtype: Desired data type of parameters.
        """
        _ = (device, dtype)
        super().__init__()
        self.num_features: int = num_features
        self.eps: float = eps
        self.momentum: float | None = momentum
        self.affine: bool = affine
        self.track_running_stats: bool = track_running_stats
        _init_batch_norm_state(self, num_features, affine, track_running_stats)

    def forward(self, input: Tensor) -> Tensor:
        """Applies 3D batch normalization to `input`.

        Args:
            input: Input tensor of shape `(N, C, D, H, W)`.

        Returns:
            Normalized tensor.
        """
        t = _to_tensor(input)
        out = _wrap(
            batch_normalization(
                t,
                zeros(t.shape),
                ones(t.shape),
                axis=1,
                config=BatchNormConfig(epsilon=self.eps),
            )
        )
        return _apply_channel_affine(out, self.weight, self.bias)


class CrossMapLRN2d(Module):
    """Applies CrossMapLRN2d over a 4D input."""

    def __init__(
        self, size: int, alpha: float = 1e-4, beta: float = 0.75, k: float = 1.0
    ) -> None:
        """Initializes CrossMapLRN2d.

        Args:
            size: Amount of neighbouring channels used for normalization.
            alpha: Multiplicative factor.
            beta: Exponent.
            k: Additive factor.
        """
        super().__init__()
        self.size: int = size
        self.alpha: float = alpha
        self.beta: float = beta
        self.k: float = k

    def forward(self, input: Tensor) -> Tensor:
        """Applies cross-map local response normalization to `input`.

        Args:
            input: Input tensor.

        Returns:
            Normalized tensor.
        """
        return _wrap(
            lrn(
                _to_tensor(input),
                config=LRNConfig(
                    depth_radius=self.size,
                    alpha=self.alpha,
                    beta=self.beta,
                    bias=self.k,
                ),
            )
        )


class GroupNorm(Module):
    """Applies Group Normalization over a mini-batch of inputs."""

    weight: Parameter | None
    bias: Parameter | None

    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        """Initializes GroupNorm.

        Args:
            num_groups: Number of groups to separate the channels into.
            num_channels: Number of channels expected in input.
            eps: Value added to the denominator for numerical stability.
            affine: If True, this module has learnable per-channel affine parameters.
            device: Desired device of parameters.
            dtype: Desired data type of parameters.
        """
        _ = (device, dtype)
        super().__init__()
        self.num_groups: int = num_groups
        self.num_channels: int = num_channels
        self.eps: float = eps
        self.affine: bool = affine
        if affine:
            self.register_parameter("weight", Parameter(zero_torch.ones(num_channels)))
            self.register_parameter("bias", Parameter(zero_torch.zeros(num_channels)))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, input: Tensor) -> Tensor:
        """Applies group normalization to `input`.

        Args:
            input: Input tensor of shape `(N, C, *)`.

        Returns:
            Normalized tensor.
        """
        out = _wrap(
            group_norm(_to_tensor(input), num_groups=self.num_groups, epsilon=self.eps)
        )
        return _apply_channel_affine(out, self.weight, self.bias)


class InstanceNorm1d(Module):
    """Applies Instance Normalization over a 2D or 3D input."""

    weight: Parameter | None
    bias: Parameter | None
    running_mean: Tensor | None
    running_var: Tensor | None
    num_batches_tracked: Tensor | None

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = False,
        track_running_stats: bool = False,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        """Initializes InstanceNorm1d.

        Args:
            num_features: Number of features or channels `C`.
            eps: Value added to the denominator for numerical stability.
            momentum: Value used for the running_mean and running_var computation.
            affine: If True, this module has learnable affine parameters.
            track_running_stats: If True, tracks running mean and variance.
            device: Desired device of parameters.
            dtype: Desired data type of parameters.
        """
        _ = (device, dtype)
        super().__init__()
        self.num_features: int = num_features
        self.eps: float = eps
        self.momentum: float = momentum
        self.affine: bool = affine
        self.track_running_stats: bool = track_running_stats
        _init_batch_norm_state(self, num_features, affine, track_running_stats)

    def forward(self, input: Tensor) -> Tensor:
        """Applies 1D instance normalization to `input`.

        Args:
            input: Input tensor.

        Returns:
            Normalized tensor.
        """
        out = _wrap(instance_norm(_to_tensor(input), epsilon=self.eps))
        return _apply_channel_affine(out, self.weight, self.bias)


class InstanceNorm2d(Module):
    """Applies Instance Normalization over a 4D input."""

    weight: Parameter | None
    bias: Parameter | None
    running_mean: Tensor | None
    running_var: Tensor | None
    num_batches_tracked: Tensor | None

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = False,
        track_running_stats: bool = False,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        """Initializes InstanceNorm2d.

        Args:
            num_features: Number of channels `C`.
            eps: Value added to the denominator for numerical stability.
            momentum: Value used for the running_mean and running_var computation.
            affine: If True, this module has learnable affine parameters.
            track_running_stats: If True, tracks running mean and variance.
            device: Desired device of parameters.
            dtype: Desired data type of parameters.
        """
        _ = (device, dtype)
        super().__init__()
        self.num_features: int = num_features
        self.eps: float = eps
        self.momentum: float = momentum
        self.affine: bool = affine
        self.track_running_stats: bool = track_running_stats
        _init_batch_norm_state(self, num_features, affine, track_running_stats)

    def forward(self, input: Tensor) -> Tensor:
        """Applies 2D instance normalization to `input`.

        Args:
            input: Input tensor.

        Returns:
            Normalized tensor.
        """
        out = _wrap(instance_norm(_to_tensor(input), epsilon=self.eps))
        return _apply_channel_affine(out, self.weight, self.bias)


class InstanceNorm3d(Module):
    """Applies Instance Normalization over a 5D input."""

    weight: Parameter | None
    bias: Parameter | None
    running_mean: Tensor | None
    running_var: Tensor | None
    num_batches_tracked: Tensor | None

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = False,
        track_running_stats: bool = False,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        """Initializes InstanceNorm3d.

        Args:
            num_features: Number of channels `C`.
            eps: Value added to the denominator for numerical stability.
            momentum: Value used for the running_mean and running_var computation.
            affine: If True, this module has learnable affine parameters.
            track_running_stats: If True, tracks running mean and variance.
            device: Desired device of parameters.
            dtype: Desired data type of parameters.
        """
        _ = (device, dtype)
        super().__init__()
        self.num_features: int = num_features
        self.eps: float = eps
        self.momentum: float = momentum
        self.affine: bool = affine
        self.track_running_stats: bool = track_running_stats
        _init_batch_norm_state(self, num_features, affine, track_running_stats)

    def forward(self, input: Tensor) -> Tensor:
        """Applies 3D instance normalization to `input`.

        Args:
            input: Input tensor.

        Returns:
            Normalized tensor.
        """
        out = _wrap(instance_norm(_to_tensor(input), epsilon=self.eps))
        return _apply_channel_affine(out, self.weight, self.bias)


class LayerNorm(Module):
    """Applies Layer Normalization over a mini-batch of inputs."""

    weight: Parameter | None
    bias: Parameter | None

    def __init__(
        self,
        normalized_shape: ShapeLike,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        """Initializes LayerNorm.

        Args:
            normalized_shape: Input shape from an expected input of size `[*, normalized_shape[0], ...]`.
            eps: Value added to the denominator for numerical stability.
            elementwise_affine: If True, this module has learnable per-element affine parameters.
            bias: If False, LayerNorm will not learn an additive bias.
            device: Desired device of parameters.
            dtype: Desired data type of parameters.
        """
        _ = (device, dtype)
        super().__init__()
        norm_tuple = _norm_shape_tuple(normalized_shape)
        self.normalized_shape: tuple[int, ...] = norm_tuple
        self.eps: float = eps
        self.elementwise_affine: bool = elementwise_affine
        if elementwise_affine:
            self.register_parameter("weight", Parameter(zero_torch.ones(*norm_tuple)))
            if bias:
                self.register_parameter(
                    "bias", Parameter(zero_torch.zeros(*norm_tuple))
                )
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, input: Tensor) -> Tensor:
        """Applies layer normalization to `input`.

        Args:
            input: Input tensor.

        Returns:
            Normalized tensor.
        """
        out = _wrap(
            layer_norm(
                _to_tensor(input),
                normalized_shape=self.normalized_shape,
                epsilon=self.eps,
            )
        )
        if self.weight is not None:
            out = out * self.weight
        if self.bias is not None:
            out = out + self.bias
        return out


class LocalResponseNorm(Module):
    """Applies local response normalization over an input signal."""

    def __init__(
        self, size: int, alpha: float = 1e-4, beta: float = 0.75, k: float = 1.0
    ) -> None:
        """Initializes LocalResponseNorm.

        Args:
            size: Amount of neighbouring channels used for normalization.
            alpha: Multiplicative factor.
            beta: Exponent.
            k: Additive factor.
        """
        super().__init__()
        self.size: int = size
        self.alpha: float = alpha
        self.beta: float = beta
        self.k: float = k

    def forward(self, input: Tensor) -> Tensor:
        """Applies local response normalization to `input`.

        Args:
            input: Input tensor.

        Returns:
            Normalized tensor.
        """
        return _wrap(
            lrn(
                _to_tensor(input),
                config=LRNConfig(
                    depth_radius=self.size,
                    alpha=self.alpha,
                    beta=self.beta,
                    bias=self.k,
                ),
            )
        )


class RMSNorm(Module):
    """Applies Root Mean Square Layer Normalization over a mini-batch of inputs."""

    weight: Parameter | None

    def __init__(
        self,
        normalized_shape: ShapeLike,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        """Initializes RMSNorm.

        Args:
            normalized_shape: Input shape over which to compute RMS normalization.
            eps: Value added to the denominator for numerical stability.
            elementwise_affine: If True, this module has learnable per-element affine parameters.
            device: Desired device of parameters.
            dtype: Desired data type of parameters.
        """
        _ = (device, dtype)
        super().__init__()
        norm_tuple = _norm_shape_tuple(normalized_shape)
        self.normalized_shape: tuple[int, ...] = norm_tuple
        self.eps: float = eps
        self.elementwise_affine: bool = elementwise_affine
        if elementwise_affine:
            self.register_parameter("weight", Parameter(zero_torch.ones(*norm_tuple)))
        else:
            self.register_parameter("weight", None)

    def forward(self, input: Tensor) -> Tensor:
        """Applies RMS normalization to `input`.

        Args:
            input: Input tensor.

        Returns:
            Normalized tensor.
        """
        t = _to_tensor(input)
        scale = (
            _to_tensor(self.weight)
            if self.weight is not None
            else ones(self.normalized_shape)
        )
        return _wrap(rms_normalization(t, scale=scale, epsilon=self.eps))


class SyncBatchNorm(Module):
    """Applies synchronized Batch Normalization over a N-Dimensional input."""

    weight: Parameter | None
    bias: Parameter | None
    running_mean: Tensor | None
    running_var: Tensor | None
    num_batches_tracked: Tensor | None

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        process_group: str | None = None,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        """Initializes SyncBatchNorm.

        Args:
            num_features: Number of features or channels `C`.
            eps: Value added to the denominator for numerical stability.
            momentum: Value used for the running_mean and running_var computation.
            affine: If True, this module has learnable affine parameters.
            track_running_stats: If True, tracks running mean and variance.
            process_group: Optional distributed process group identifier.
            device: Desired device of parameters.
            dtype: Desired data type of parameters.
        """
        _ = (device, dtype)
        super().__init__()
        self.num_features: int = num_features
        self.eps: float = eps
        self.momentum: float = momentum
        self.affine: bool = affine
        self.track_running_stats: bool = track_running_stats
        self.process_group: str | None = process_group
        _init_batch_norm_state(self, num_features, affine, track_running_stats)

    def forward(self, input: Tensor) -> Tensor:
        """Applies synchronized batch normalization to `input`.

        Args:
            input: Input tensor.

        Returns:
            Normalized tensor.
        """
        t = _to_tensor(input)
        out = _wrap(
            batch_normalization(
                t,
                zeros(t.shape),
                ones(t.shape),
                axis=1,
                config=BatchNormConfig(epsilon=self.eps),
            )
        )
        return _apply_channel_affine(out, self.weight, self.bias)
