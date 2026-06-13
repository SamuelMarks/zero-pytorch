import pytest
import ml_switcheroo_compiler as ml_switcheroo
import zero_torch as torch


def test_functional_autograd():
    """Tests for test_functional_autograd."""
    with ml_switcheroo.EagerMode():
        x = torch.tensor([2.0], requires_grad=True)
        with pytest.raises(NotImplementedError):
            x.backward()
