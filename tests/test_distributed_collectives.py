"""Tests for distributed collectives, virtual rank reductions, and rendezvous stores."""

from __future__ import annotations

from ml_switcheroo_compiler.core.config import EagerMode

import zero_torch as torch
import zero_torch.distributed as dist
from tests.conftest import assert_allclose


def test_distributed_multi_rank_and_stores() -> None:
    """Verifies multi-rank `ReduceOp` reductions and `HashStore` / `TCPStore` operations."""
    with EagerMode():
        pg = dist.ProcessGroup(rank=0, world_size=2)
        pg.rank_buffers[1] = torch.tensor([10.0, 20.0])

        t = torch.tensor([2.0, 4.0])
        dist.all_reduce(t, op=dist.ReduceOp.SUM, group=pg)
        assert_allclose(t.tolist(), [12.0, 24.0])

        store = dist.HashStore()
        store.set("epoch", "5")
        assert store.get("epoch") == b"5"
        assert store.add("step", 1) == 1
