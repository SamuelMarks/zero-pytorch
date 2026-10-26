"""Complete tests for optimizers, state serialization, and learning rate schedulers."""

from __future__ import annotations

from ml_switcheroo_compiler.core.config import EagerMode

import zero_torch as torch
import zero_torch.nn as nn
import zero_torch.optim as optim
from tests.conftest import assert_allclose


def test_optimizer_and_scheduler_state_roundtrip() -> None:
    """Verifies `Adam` and `StepLR` state_dict / load_state_dict round-trip."""
    with EagerMode():
        p = nn.Parameter(torch.tensor([2.0, -4.0]))
        p.grad = torch.tensor([0.5, -1.0])
        opt = optim.Adam([p], lr=0.1)
        sched = optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5)
        opt.step()
        sched.step()

        opt_sd = opt.state_dict()
        sched_sd = sched.state_dict()

        opt2 = optim.Adam([p], lr=0.1)
        sched2 = optim.lr_scheduler.StepLR(opt2, step_size=1, gamma=0.5)
        opt2.load_state_dict(opt_sd)
        sched2.load_state_dict(sched_sd)
        assert_allclose(opt2.param_groups[0]["lr"], 0.05)
