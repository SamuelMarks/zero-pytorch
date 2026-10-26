"""Advanced autograd tests for higher-order derivatives, hooks, and custom Functions."""

from __future__ import annotations

from ml_switcheroo_compiler.core.config import EagerMode

import zero_torch as torch
from tests.conftest import assert_allclose


def test_autograd_higher_order_and_hooks() -> None:
    """Verifies second-order derivatives via `create_graph=True` and `Tensor.register_hook`."""
    with EagerMode():
        x = torch.tensor([2.0, 3.0], requires_grad=True)
        y = (x * x * x).sum()
        (grad1,) = torch.autograd.grad(y, x, create_graph=True)
        assert grad1 is not None
        assert_allclose(grad1.tolist(), [12.0, 27.0], rtol=1e-5)

        (grad2,) = torch.autograd.grad(grad1.sum(), x)
        assert grad2 is not None
        assert_allclose(grad2.tolist(), [12.0, 18.0], rtol=1e-5)

        z = torch.tensor([4.0, -2.0], requires_grad=True)
        handle = z.register_hook(lambda g: g * 0.5)
        (z * z).sum().backward()
        assert z.grad is not None
        assert_allclose(z.grad.tolist(), [4.0, -2.0], rtol=1e-5)
        assert handle is not None
