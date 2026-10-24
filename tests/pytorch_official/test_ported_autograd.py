"""Ported autograd functional tests verifying analytical and numerical gradients."""

from __future__ import annotations

import numpy as np
from ml_switcheroo_compiler.core.config import EagerMode

import zero_torch as torch


def test_functional_autograd() -> None:
    """Verifies forward and backward gradients across arithmetic and reduction ops."""
    with EagerMode():
        x = torch.tensor([2.0], requires_grad=True)
        y = torch.tensor([3.0], requires_grad=True)
        z = x * y
        z.backward()

        assert x.grad is not None
        assert y.grad is not None
        assert x.grad.item() == 3.0
        assert y.grad.item() == 2.0

        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        b = torch.tensor([10.0, 20.0], requires_grad=True)
        out = (a + b).sum()
        out.backward()
        assert a.grad is not None
        assert b.grad is not None
        np.testing.assert_allclose(a.grad.tolist(), [[1.0, 1.0], [1.0, 1.0]])
        np.testing.assert_allclose(b.grad.tolist(), [2.0, 2.0])
