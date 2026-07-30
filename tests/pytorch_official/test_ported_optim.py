import numpy as np
from ml_switcheroo_compiler.core.config import EagerMode

import zero_torch as torch
from zero_torch import nn, optim

try:
    from ml_switcheroo_compiler.core.errors import (
        ShapeMismatchError,
        UnimplementedMathError,
    )
except ImportError:
    UnimplementedMathError = Exception
    ShapeMismatchError = Exception


def test_sgd():
    """Tests for test_sgd."""
    with EagerMode():
        # Parameter
        w = nn.Parameter(torch.ones((2, 2)))

        # We simulate functional grad by just manually assigning grad
        w.grad = torch.ones((2, 2)) * 2.0

        # Optimizer
        opt = optim.SGD([w], lr=0.1)

        # Step
        opt.step()

        # Weights should be 1.0 - 0.1 * 2.0 = 0.8
        np.testing.assert_allclose(w.numpy(), 0.8)
