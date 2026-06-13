import numpy as np
import ml_switcheroo_compiler as ml_switcheroo
import zero_torch as torch
import zero_torch.optim as optim
import zero_torch.nn as nn


def test_sgd():
    """Tests for test_sgd."""
    with ml_switcheroo.EagerMode():
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
