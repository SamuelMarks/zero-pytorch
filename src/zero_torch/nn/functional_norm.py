"API Frontend backed by ml-switcheroo-compiler."


def batch_norm(*args, **kwargs):
    """Applies a normalization operation (batch_norm).

    Args:
        *args (Any): Variable length argument list.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        Any: The result of the batch_norm operation.
    """
    from ml_switcheroo_compiler.ops.nn.normalization import batch_normalization

    from zero_torch.tensor import _to_tensor, _wrap

    input_t = _to_tensor(args[0] if len(args) > 0 else kwargs.get("input"))
    return _wrap(batch_normalization(input_t, eps=kwargs.get("eps", 1e-5)))


def layer_norm(*args, **kwargs):
    """Applies a normalization operation (layer_norm).

    Args:
        *args (Any): Variable length argument list.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        Any: The result of the layer_norm operation.
    """
    from ml_switcheroo_compiler.ops.nn.normalization import layer_norm as _layer_norm

    from zero_torch.tensor import _to_tensor, _wrap

    input_t = _to_tensor(args[0] if len(args) > 0 else kwargs.get("input"))
    normalized_shape = args[1] if len(args) > 1 else kwargs.get("normalized_shape")
    return _wrap(
        _layer_norm(
            input_t, normalized_shape=normalized_shape, eps=kwargs.get("eps", 1e-5)
        )
    )


def instance_norm(*args, **kwargs):
    """Applies a normalization operation (instance_norm).

    Args:
        *args (Any): Variable length argument list.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        Any: The result of the instance_norm operation.
    """
    from ml_switcheroo_compiler.ops.nn.normalization import (
        instance_norm as _instance_norm,
    )

    from zero_torch.tensor import _to_tensor, _wrap

    input_t = _to_tensor(args[0] if len(args) > 0 else kwargs.get("input"))
    return _wrap(_instance_norm(input_t, eps=kwargs.get("eps", 1e-5)))


def normalize(*args, **kwargs):
    """Applies a normalization operation (normalize).

    Args:
        *args (Any): Variable length argument list.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        Any: The result of the normalize operation.
    """
    from ml_switcheroo_compiler.ops.nn.normalization import l2_normalize

    from zero_torch.tensor import _to_tensor, _wrap

    input_t = _to_tensor(args[0] if len(args) > 0 else kwargs.get("input"))
    dim = kwargs.get("dim", 1)
    eps = kwargs.get("eps", 1e-12)
    return _wrap(l2_normalize(input_t, axis=dim, epsilon=eps))


def rms_norm(*args, **kwargs):
    """Applies a normalization operation (rms_norm).

    Args:
        *args (Any): Variable length argument list.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        Any: The result of the rms_norm operation.
    """
    from ml_switcheroo_compiler.ops.nn.normalization import rms_normalization

    from zero_torch.tensor import _to_tensor, _wrap

    input_t = _to_tensor(args[0] if len(args) > 0 else kwargs.get("input"))
    return _wrap(rms_normalization(input_t, eps=kwargs.get("eps", 1e-5)))
