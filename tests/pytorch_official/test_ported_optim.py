"""Ported optimizer (`zero_torch.optim`) tests."""

from __future__ import annotations

import numpy as np
from ml_switcheroo_compiler.core.config import EagerMode

import zero_torch as torch
from zero_torch import nn, optim


def test_sgd() -> None:
    """Verifies `optim.SGD` parameter weight update."""
    with EagerMode():
        w = nn.Parameter(torch.ones((2, 2)))
        w.grad = torch.ones((2, 2)) * 2.0

        opt = optim.SGD([w], lr=0.1)
        opt.step()

        np.testing.assert_allclose(w.tolist(), 0.8)
