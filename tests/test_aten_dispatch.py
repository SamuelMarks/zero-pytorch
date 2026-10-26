"""Tests for `zero_torch.ops.aten` operator dispatch and overload packets."""

from __future__ import annotations

from ml_switcheroo_compiler.core.config import EagerMode

import zero_torch as torch
from tests.conftest import assert_allclose
from zero_torch.ops.aten import OpOverloadPacket


def test_aten_dispatch_and_overload_packet() -> None:
    """Verifies `torch.ops.aten` operator dispatch and `OpOverloadPacket` `.default` and `.out`."""
    with EagerMode():
        x = torch.tensor([-3.0, 4.0])
        res = torch.ops.aten.abs(x)
        assert_allclose(res.tolist(), [3.0, 4.0])

        pkt = OpOverloadPacket("abs", torch.ops.aten.abs)
        assert_allclose(pkt.default(x).tolist(), [3.0, 4.0])
        out_buf = torch.zeros((2,))
        pkt.out(x, out=out_buf)
        assert_allclose(out_buf.tolist(), [3.0, 4.0])
