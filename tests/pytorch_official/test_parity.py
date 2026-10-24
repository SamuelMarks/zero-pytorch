"""Numerical parity tests against reference PyTorch."""

from __future__ import annotations

import numpy as np
import torch as real_torch
from ml_switcheroo_compiler.core.config import EagerMode

import zero_torch


def test_core_math_parity() -> None:
    """Verifies numerical parity between `zero_torch` and `torch` across core math operations."""
    with EagerMode():
        arr_a = np.array([[0.25, 0.5], [0.75, 1.25]], dtype=np.float32)
        arr_b = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        z_a = zero_torch.tensor(arr_a)
        z_b = zero_torch.tensor(arr_b)
        t_a = real_torch.tensor(arr_a)
        t_b = real_torch.tensor(arr_b)

        np.testing.assert_allclose(
            zero_torch.add(z_a, z_b).tolist(),
            real_torch.add(t_a, t_b).tolist(),
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            zero_torch.subtract(z_a, z_b).tolist(),
            real_torch.sub(t_a, t_b).tolist(),
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            zero_torch.multiply(z_a, z_b).tolist(),
            real_torch.mul(t_a, t_b).tolist(),
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            zero_torch.divide(z_a, z_b).tolist(),
            real_torch.div(t_a, t_b).tolist(),
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            zero_torch.matmul(z_a, z_b).tolist(),
            real_torch.matmul(t_a, t_b).tolist(),
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            zero_torch.exp(z_a).tolist(), real_torch.exp(t_a).tolist(), rtol=1e-5
        )
        np.testing.assert_allclose(
            zero_torch.log(z_a).tolist(), real_torch.log(t_a).tolist(), rtol=1e-5
        )
        np.testing.assert_allclose(
            zero_torch.sin(z_a).tolist(), real_torch.sin(t_a).tolist(), rtol=1e-5
        )
        np.testing.assert_allclose(
            zero_torch.cos(z_a).tolist(), real_torch.cos(t_a).tolist(), rtol=1e-5
        )
        np.testing.assert_allclose(
            zero_torch.tanh(z_a).tolist(), real_torch.tanh(t_a).tolist(), rtol=1e-5
        )
        np.testing.assert_allclose(
            zero_torch.sigmoid(z_a).tolist(),
            real_torch.sigmoid(t_a).tolist(),
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            zero_torch.softmax(z_a, dim=-1).tolist(),
            real_torch.softmax(t_a, dim=-1).tolist(),
            rtol=1e-5,
        )
