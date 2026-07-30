"""Module."""

from .module import Module

"API Frontend backed by ml-switcheroo-compiler."


class BatchNorm1d(Module):
    """Applies Batch Normalization over a 2D or 3D input."""

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-05,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        """Initializes the BatchNorm1d module.

        Args:
            num_features: C from an expected input of size (N, C, L) or L from input of size (N, L)
            eps: a value added to the denominator for numerical stability. Default: 1e-5
            momentum: the value used for the running_mean and running_var computation. Can be set to None for cumulative moving average (i.e. simple average). Default: 0.1
            affine: a boolean value that when set to True, this module has learnable affine parameters. Default: True
            track_running_stats: a boolean value that when set to True, this module tracks the running mean and variance. Default: True
            device: device.
            dtype: dtype.
        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

    def forward(self, input):
        """Forward pass.

        Args:
            input (Tensor): input tensor.

        Returns:
            Tensor: output.
        """
        from ml_switcheroo_compiler.ops.nn.normalization import batch_normalization

        from zero_torch.tensor import _to_tensor, _wrap

        return _wrap(batch_normalization(_to_tensor(input), eps=self.eps))


class BatchNorm2d(Module):
    """Applies Batch Normalization over a 4D input."""

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-05,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        """Initializes the BatchNorm2d module.

        Args:
            num_features: C from an expected input of size (N, C, H, W)
            eps: a value added to the denominator for numerical stability. Default: 1e-5
            momentum: the value used for the running_mean and running_var computation. Default: 0.1
            affine: a boolean value that when set to True, this module has learnable affine parameters. Default: True
            track_running_stats: a boolean value that when set to True, this module tracks the running mean and variance. Default: True
            device: device.
            dtype: dtype.
        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

    def forward(self, input):
        """Forward pass.

        Args:
            input (Tensor): input tensor.

        Returns:
            Tensor: output.
        """
        from ml_switcheroo_compiler.ops.nn.normalization import batch_normalization

        from zero_torch.tensor import _to_tensor, _wrap

        return _wrap(batch_normalization(_to_tensor(input), eps=self.eps))


class BatchNorm3d(Module):
    """Applies Batch Normalization over a 5D input."""

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-05,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        """Initializes the BatchNorm3d module.

        Args:
            num_features: C from an expected input of size (N, C, D, H, W)
            eps: a value added to the denominator for numerical stability. Default: 1e-5
            momentum: the value used for the running_mean and running_var computation. Default: 0.1
            affine: a boolean value that when set to True, this module has learnable affine parameters. Default: True
            track_running_stats: a boolean value that when set to True, this module tracks the running mean and variance. Default: True
            device: device.
            dtype: dtype.
        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

    def forward(self, input):
        """Forward pass.

        Args:
            input (Tensor): input tensor.

        Returns:
            Tensor: output.
        """
        from ml_switcheroo_compiler.ops.nn.normalization import batch_normalization

        from zero_torch.tensor import _to_tensor, _wrap

        return _wrap(batch_normalization(_to_tensor(input), eps=self.eps))


class CrossMapLRN2d(Module):
    """Applies CrossMapLRN2d over a mini-batch of inputs."""

    def __init__(
        self, size: int, alpha: float = 1e-4, beta: float = 0.75, k: float = 1.0
    ) -> None:
        """Initializes CrossMapLRN2d."""
        super().__init__()
        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.k = k

    def forward(self, input):
        """Forward pass."""
        from ml_switcheroo_compiler.ops.configs import LRNConfig
        from ml_switcheroo_compiler.ops.nn.normalization import lrn  # pragma: no cover

        from zero_torch.tensor import _to_tensor, _wrap  # pragma: no cover

        # pragma: no cover
        return _wrap(  # pragma: no cover
            lrn(
                _to_tensor(input),
                config=LRNConfig(
                    size=self.size, alpha=self.alpha, beta=self.beta, bias=self.k
                ),
            )
        )


class GroupNorm(Module):
    """Applies Group Normalization."""

    def __init__(
        self, num_groups: int, num_channels: int, eps: float = 1e-5, affine: bool = True
    ) -> None:
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

    def forward(self, input):
        from ml_switcheroo_compiler.ops.nn.normalization import group_norm

        from zero_torch.tensor import _to_tensor, _wrap

        return _wrap(
            group_norm(_to_tensor(input), num_groups=self.num_groups, eps=self.eps)
        )


class InstanceNorm1d(Module):
    """Applies InstanceNorm1d."""

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = False,
        track_running_stats: bool = False,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

    def forward(self, input):
        from ml_switcheroo_compiler.ops.nn.normalization import instance_norm

        from zero_torch.tensor import _to_tensor, _wrap

        return _wrap(instance_norm(_to_tensor(input), eps=self.eps))


class InstanceNorm2d(Module):
    """Applies InstanceNorm2d."""

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = False,
        track_running_stats: bool = False,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

    def forward(self, input):
        from ml_switcheroo_compiler.ops.nn.normalization import instance_norm

        from zero_torch.tensor import _to_tensor, _wrap

        return _wrap(instance_norm(_to_tensor(input), eps=self.eps))


class InstanceNorm3d(Module):
    """Applies InstanceNorm3d."""

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = False,
        track_running_stats: bool = False,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

    def forward(self, input):
        from ml_switcheroo_compiler.ops.nn.normalization import instance_norm

        from zero_torch.tensor import _to_tensor, _wrap

        return _wrap(instance_norm(_to_tensor(input), eps=self.eps))


class LayerNorm(Module):
    """Applies Layer Normalization."""

    def __init__(
        self, normalized_shape, eps: float = 1e-5, elementwise_affine: bool = True
    ) -> None:
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine

    def forward(self, input):
        from ml_switcheroo_compiler.ops.nn.normalization import layer_norm

        from zero_torch.tensor import _to_tensor, _wrap

        return _wrap(
            layer_norm(
                _to_tensor(input), normalized_shape=self.normalized_shape, eps=self.eps
            )
        )


class LocalResponseNorm(Module):
    """Applies Local Response Normalization."""

    def __init__(
        self, size: int, alpha: float = 1e-4, beta: float = 0.75, k: float = 1.0
    ) -> None:
        super().__init__()
        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.k = k

    def forward(self, input):
        from ml_switcheroo_compiler.ops.configs import LRNConfig
        from ml_switcheroo_compiler.ops.nn.normalization import lrn  # pragma: no cover

        from zero_torch.tensor import _to_tensor, _wrap  # pragma: no cover

        # pragma: no cover
        return _wrap(  # pragma: no cover
            lrn(
                _to_tensor(input),
                config=LRNConfig(
                    size=self.size, alpha=self.alpha, beta=self.beta, bias=self.k
                ),
            )
        )


class RMSNorm(Module):
    """Applies RMSNorm."""

    def __init__(
        self, normalized_shape, eps: float = 1e-5, elementwise_affine: bool = True
    ) -> None:
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine

    def forward(self, input):
        from ml_switcheroo_compiler.ops.nn.normalization import rms_normalization

        from zero_torch.tensor import _to_tensor, _wrap

        return _wrap(rms_normalization(_to_tensor(input), eps=self.eps))


class SyncBatchNorm(Module):
    """Applies SyncBatchNorm."""

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        process_group=None,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.process_group = process_group

    def forward(self, input):
        from ml_switcheroo_compiler.ops.nn.normalization import batch_normalization

        from zero_torch.tensor import _to_tensor, _wrap

        return _wrap(batch_normalization(_to_tensor(input), eps=self.eps))
