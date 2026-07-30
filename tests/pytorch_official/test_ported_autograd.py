from ml_switcheroo_compiler.core.config import EagerMode

import zero_torch as torch

try:
    from ml_switcheroo_compiler.core.errors import (
        ShapeMismatchError,
        UnimplementedMathError,
    )
except ImportError:
    UnimplementedMathError = Exception
    ShapeMismatchError = Exception


def test_functional_autograd():
    """Tests for test_functional_autograd."""
    with EagerMode():
        x = torch.tensor([2.0], requires_grad=True)
        y = torch.tensor([3.0], requires_grad=True)
        z = x * y
        z.backward()

        # Test gradients
        assert x.grad is not None
        assert y.grad is not None
        assert x.grad.item() == 3.0
        assert y.grad.item() == 2.0
