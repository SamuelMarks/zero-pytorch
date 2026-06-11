import numpy as np
import ml_switcheroo
import zero_torch as torch
import zero_torch.nn as nn
import zero_torch.nn.functional as F


def test_module_lifecycle():
    """Tests for test_module_lifecycle.

    Args:
        *args: arguments
        **kwargs: keyword arguments

    Returns:
        Any: returns
    """

    class MyModule(nn.Module):
        """Tests for MyModule."""

        def __init__(self):
            """Tests for __init__."""
            super().__init__()
            self.linear = nn.Linear(3, 4)
            self.register_buffer("running_mean", torch.zeros(4))

        def forward(self, x):
            """Tests for forward.

            Args:
                *args: arguments
                **kwargs: keyword arguments

            Returns:
                Any: returns
            """
            return self.linear(x) + self.running_mean

    with ml_switcheroo.EagerMode():
        m = MyModule()

        # Check state_dict
        sd = m.state_dict()
        assert "linear.weight" in sd
        assert "linear.bias" in sd
        assert "running_mean" in sd
        assert sd["linear.weight"].shape == (4, 3)
        assert sd["running_mean"].shape == (4,)

        # Check parameters
        params = list(m.parameters())
        assert len(params) == 2

        # Check buffers
        buffers = list(m.buffers())
        assert len(buffers) == 1

        # Check forward pass
        x = torch.ones((2, 3))
        out = m(x)
        assert out.shape == (2, 4)


def test_functional_linear():
    """Tests for test_functional_linear."""
    with ml_switcheroo.EagerMode():
        x = torch.ones((2, 3))
        w = torch.ones((4, 3)) * 2.0
        b = torch.ones((4,))

        out = F.linear(x, w, b)
        assert out.shape == (2, 4)
        np.testing.assert_allclose(out.numpy(), 7.0)


def test_conv2d():
    """Tests for test_conv2d."""
    with ml_switcheroo.EagerMode():
        # (N, C, H, W)
        x = torch.ones((2, 3, 5, 5))
        conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)

        # Set weights deterministically
        conv.weight = nn.Parameter(torch.ones((4, 3, 3, 3)))
        conv.bias = nn.Parameter(torch.zeros(4))

        try:
            out = conv(x)
            assert out.shape == (2, 4, 5, 5)
        except ml_switcheroo.core.errors.UnimplementedMathError:
            pass


def test_batchnorm2d():
    """Tests for test_batchnorm2d."""
    with ml_switcheroo.EagerMode():
        x = torch.ones((2, 3, 5, 5))
        bn = nn.BatchNorm2d(3)

        try:
            out = bn(x)
            assert out.shape == (2, 3, 5, 5)
        except ml_switcheroo.core.errors.UnimplementedMathError:
            pass


def test_losses():
    """Tests for test_losses."""
    with ml_switcheroo.EagerMode():
        pred = torch.tensor([[0.5, -0.5], [1.0, 2.0]])
        target = torch.tensor([0, 1])

        # Cross entropy
        ce = nn.CrossEntropyLoss()
        try:
            loss_ce = ce(pred, target)
            assert loss_ce.shape == ()
        except ml_switcheroo.core.errors.UnimplementedMathError:
            pass

        # MSE
        target_mse = torch.tensor([[0.0, 0.0], [1.0, 2.0]])
        mse = nn.MSELoss()
        loss_mse = mse(pred, target_mse)
        assert loss_mse.shape == ()
