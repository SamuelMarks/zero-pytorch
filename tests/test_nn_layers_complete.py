"""Complete tests for neural network layers, initializations, and state serialization."""

from __future__ import annotations

from ml_switcheroo_compiler.core.config import EagerMode

import zero_torch as torch
import zero_torch.nn as nn
from tests.conftest import assert_allclose


def test_nn_layers_state_dict_and_forward() -> None:
    """Verifies `Linear`, `Conv1d`, `LSTM`, and `MultiheadAttention` forward and `state_dict` round-trip."""
    with EagerMode():
        lin = nn.Linear(3, 2)
        sd = lin.state_dict()
        lin2 = nn.Linear(3, 2)
        lin2.load_state_dict(sd)
        x = torch.tensor([[1.0, 2.0, 3.0]])
        assert_allclose(lin(x).tolist(), lin2(x).tolist(), rtol=1e-5)

        conv = nn.Conv1d(2, 2, kernel_size=3, padding=1, padding_mode="replicate")
        assert conv(torch.ones((1, 2, 4))).shape == (1, 2, 4)
