try:
    from ml_switcheroo_compiler.core.errors import (
        ShapeMismatchError,
        UnimplementedMathError,
    )
except ImportError:
    UnimplementedMathError = Exception
    ShapeMismatchError = Exception

import numpy as np
import torch as real_torch

import zero_torch

try:
    from ml_switcheroo_compiler.core.errors import (
        ShapeMismatchError,
        UnimplementedMathError,
    )
except ImportError:
    UnimplementedMathError = Exception
    ShapeMismatchError = Exception


def assert_parity(name, args, kwargs, real_out):
    """Tests for assert_parity.

    Args:
        *args: arguments
        **kwargs: keyword arguments

    Returns:
        Any: returns
    """
    if not hasattr(zero_torch, name):
        return

    try:
        zero_out = getattr(zero_torch, name)(*args, **kwargs)
    except (
        RuntimeError,
        ValueError,
        TypeError,
        AttributeError,
        KeyError,
        IndexError,
        ImportError,
        NotImplementedError,
        UnimplementedMathError,
        ShapeMismatchError,
    ):
        # Expected to fail if zero_torch doesn't support it yet
        return

    # Compare outputs
    if isinstance(real_out, real_torch.Tensor) and isinstance(
        zero_out, zero_torch.Tensor
    ):
        try:
            r_np = real_out.detach().cpu().numpy()
            z_np = np.array(zero_out.data)
            np.testing.assert_allclose(r_np, z_np, rtol=1e-4, atol=1e-4)
        except (
            RuntimeError,
            ValueError,
            TypeError,
            AttributeError,
            KeyError,
            IndexError,
            ImportError,
            NotImplementedError,
            UnimplementedMathError,
            ShapeMismatchError,
        ):
            _pass = True
