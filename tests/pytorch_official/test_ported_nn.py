"""Ported neural network (`zero_torch.nn`) module and functional tests."""

from __future__ import annotations

from ml_switcheroo_compiler.core.config import EagerMode

import zero_torch as torch
import zero_torch.nn.functional as F
from tests.conftest import assert_allclose
from zero_torch import nn
from zero_torch.tensor import Tensor


def test_module_lifecycle() -> None:
    """Verifies `nn.Module` parameter/buffer registration, `state_dict`, and `forward`."""

    class MyModule(nn.Module):
        """Test module with a Linear layer and registered running_mean buffer."""

        def __init__(self) -> None:
            """Initializes MyModule."""
            super().__init__()
            self.linear = nn.Linear(3, 4)
            self.add_module("linear", self.linear)
            self.running_mean: Tensor | None = torch.zeros(4)
            self.register_buffer("running_mean", self.running_mean)

        def forward(self, x: Tensor) -> Tensor:
            """Applies the linear transformation and adds `running_mean`.

            Args:
                x: Input tensor of shape `(*, 3)`.

            Returns:
                Output tensor of shape `(*, 4)`.
            """
            assert self.running_mean is not None
            return self.linear(x) + self.running_mean

    with EagerMode():
        m = MyModule()

        sd = m.state_dict()
        assert "linear.weight" in sd
        assert "linear.bias" in sd
        assert "running_mean" in sd
        assert sd["linear.weight"].shape == (4, 3)
        assert sd["running_mean"].shape == (4,)

        params = list(m.parameters())
        assert len(params) == 2

        buffers = list(m.buffers())
        assert len(buffers) == 1

        x = torch.ones((2, 3))
        out = m(x)
        assert out.shape == (2, 4)


def test_functional_linear() -> None:
    """Verifies `F.linear` matrix multiplication and bias addition."""
    with EagerMode():
        x = torch.ones((2, 3))
        w = torch.ones((4, 3)) * 2.0
        b = torch.ones((4,))

        out = F.linear(x, w, b)
        assert out.shape == (2, 4)
        assert_allclose(out.tolist(), 7.0)


def test_conv2d() -> None:
    """Verifies `nn.Conv2d` forward pass with deterministic weights."""
    with EagerMode():
        x = torch.ones((2, 3, 5, 5))
        conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)

        conv.weight = nn.Parameter(torch.ones((4, 3, 3, 3)))
        conv.bias = nn.Parameter(torch.zeros(4))

        out = conv(x)
        assert out.shape == (2, 4, 5, 5)


def test_batchnorm2d() -> None:
    """Verifies `nn.BatchNorm2d` affine parameter registration, running stats update, and eval mode."""
    with EagerMode():
        bn = nn.BatchNorm2d(3)
        assert bn.weight is not None and bn.weight.shape == (3,)
        assert bn.bias is not None and bn.bias.shape == (3,)
        assert bn.running_mean is not None and bn.running_mean.shape == (3,)
        assert bn.running_var is not None and bn.running_var.shape == (3,)
        x = torch.ones((2, 3, 4, 4)) * 5.0
        out = bn(x)
        assert out.shape == (2, 3, 4, 4)
        assert_allclose(bn.running_mean.tolist(), [0.5, 0.5, 0.5], rtol=1e-4)
        bn.eval()
        out_eval = bn(x)
        assert out_eval.shape == (2, 3, 4, 4)

        lazy_lin = nn.LazyLinear(4)
        _ = lazy_lin(torch.ones((2, 3)))
        sd = lazy_lin.state_dict()
        assert "weight" in sd and "bias" in sd
        assert sd["weight"].shape == (4, 3)


def test_losses() -> None:
    """Verifies `nn.MSELoss` and `nn.CrossEntropyLoss` non-zero loss evaluation."""
    with EagerMode():
        pred = torch.tensor([[0.5, -0.5], [1.0, 2.0]])
        target_mse = torch.tensor([[0.0, 0.0], [1.0, 2.0]])
        mse = nn.MSELoss()
        loss_mse = mse(pred, target_mse)
        assert loss_mse.shape == ()
        assert_allclose(loss_mse.tolist(), 0.125, rtol=1e-4)

        ce = nn.CrossEntropyLoss()
        loss_ce = ce(pred, torch.tensor([0, 1]))
        assert loss_ce.shape == ()
        assert loss_ce.item() != 0.0
