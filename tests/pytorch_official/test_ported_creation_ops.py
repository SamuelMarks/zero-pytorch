import numpy as np
import ml_switcheroo
import zero_torch as torch

# Ported from PyTorch test_tensor_creation_ops.py


def test_zeros_empty_full():
    with ml_switcheroo.EagerMode():
        # zeros
        z = torch.zeros((2, 3))
        assert z.shape == (2, 3)
        assert np.all(z.numpy() == 0)

        # empty
        e = torch.empty((2, 3))
        assert e.shape == (2, 3)

        # full
        f = torch.full((2, 3), 5.0)
        assert f.shape == (2, 3)
        assert np.all(f.numpy() == 5.0)


def test_arange():
    with ml_switcheroo.EagerMode():
        a = torch.arange(5)
        assert a.shape == (5,)
        np.testing.assert_array_equal(a.numpy(), np.arange(5))

        b = torch.arange(1, 5, 2)
        assert b.shape == (2,)
        np.testing.assert_array_equal(b.numpy(), np.arange(1, 5, 2))


def test_cat():
    with ml_switcheroo.EagerMode():
        t1 = torch.zeros((2, 3))
        t2 = torch.ones((2, 3))

        c0 = torch.concatenate((t1, t2), dim=0)
        assert c0.shape == (4, 3)
        assert np.all(c0.numpy()[:2] == 0)
        assert np.all(c0.numpy()[2:] == 1)

        c1 = torch.concatenate((t1, t2), dim=1)
        assert c1.shape == (2, 6)


def test_linspace():
    with ml_switcheroo.EagerMode():
        lin = torch.linspace(0.0, 10.0, 5)
        assert lin.shape == (5,)
        np.testing.assert_allclose(lin.numpy(), np.linspace(0.0, 10.0, 5))


def test_eye():
    with ml_switcheroo.EagerMode():
        e = torch.eye(3)
        assert e.shape == (3, 3)
        np.testing.assert_array_equal(e.numpy(), np.eye(3))
