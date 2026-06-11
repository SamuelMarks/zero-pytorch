"""Tests for coverage gaps."""

from zero_torch import Tensor
from zero_torch.nn import Module, Linear
from zero_torch.optim import SGD
import ml_switcheroo


def test_module_coverage():
    """Tests module coverage."""

    class MyMod(Module):
        """My dummy module."""

        def __init__(self):
            """Initializes MyMod."""
            super().__init__()
            self.register_buffer("my_buf", Tensor([1.0, 2.0]))
            self.linear = Linear(2, 3)

        def forward(self, x):
            """Forward pass."""
            return x

    mod = MyMod()
    assert hasattr(mod, "my_buf")

    params = list(mod.parameters(recurse=True))
    assert len(params) > 0


def test_optim_step():
    """Tests optim step."""

    class DummyParam:
        """Dummy parameter."""

        def __init__(self):
            """Initializes DummyParam."""
            self._tensor = [1.0]
            self.grad = Tensor([0.1])

        def __sub__(self, other):
            """Subtracts."""
            res = DummyParam()
            res._tensor = [self._tensor[0] - other._tensor[0]]
            return res

    p = Tensor([1.0])
    p.grad = Tensor([0.1])
    opt = SGD([p], lr=0.1)
    with ml_switcheroo.EagerMode():
        opt.step()


def test_norm_fallback():
    """Tests norm fallback."""
    from zero_torch import norm, Tensor
    import numpy as np
    import ml_switcheroo

    t = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
    with ml_switcheroo.EagerMode():
        res = norm(t)
        assert res.shape == ()

        # Test dim and keepdim
        res2 = norm(t, dim=0, keepdim=True)
        assert res2.shape == (1, 2)

        # Test p != 2
        res3 = norm(t, p=1)
        assert res3.shape == ()
