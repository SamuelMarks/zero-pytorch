"""Ported tensor creation operation tests from PyTorch `test_tensor_creation_ops.py`."""

from __future__ import annotations

import numpy as np
from ml_switcheroo_compiler.core.config import EagerMode

import zero_torch as torch


def test_zeros_empty_full() -> None:
    """Verifies `zeros`, `empty`, and `full` tensor creation ops."""
    with EagerMode():
        z = torch.zeros((2, 3))
        assert z.shape == (2, 3)
        assert z.tolist() == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

        e = torch.empty((2, 3))
        assert e.shape == (2, 3)

        f = torch.full((2, 3), 5.0)
        assert f.shape == (2, 3)
        assert f.tolist() == [[5.0, 5.0, 5.0], [5.0, 5.0, 5.0]]


def test_arange() -> None:
    """Verifies `arange` tensor creation with single and start/end/step arguments."""
    with EagerMode():
        a = torch.arange(5)
        assert a.shape == (5,)
        assert a.tolist() == [0, 1, 2, 3, 4]

        b = torch.arange(1, 5, 2)
        assert b.shape == (2,)
        assert b.tolist() == [1, 3]


def test_cat() -> None:
    """Verifies `concatenate` along dimensions 0 and 1."""
    with EagerMode():
        t1 = torch.zeros((2, 3))
        t2 = torch.ones((2, 3))

        c0 = torch.concatenate((t1, t2), dim=0)
        assert c0.shape == (4, 3)
        assert c0[:2].tolist() == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        assert c0[2:].tolist() == [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]

        c1 = torch.concatenate((t1, t2), dim=1)
        assert c1.shape == (2, 6)


def test_linspace() -> None:
    """Verifies `linspace` evenly spaced sequence generation."""
    with EagerMode():
        lin = torch.linspace(0.0, 10.0, 5)
        assert lin.shape == (5,)
        np.testing.assert_allclose(lin.tolist(), np.linspace(0.0, 10.0, 5))


def test_eye() -> None:
    """Verifies `eye` 2D identity matrix creation."""
    with EagerMode():
        e = torch.eye(3)
        assert e.shape == (3, 3)
        assert e.tolist() == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
