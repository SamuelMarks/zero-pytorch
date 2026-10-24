"""Tests for `ProxyTensor` symbolic graph tracing operations and error handling."""

from __future__ import annotations

import pytest
from ml_switcheroo_compiler.core.errors import TracingError
from ml_switcheroo_compiler.ir.core import IRNode

from zero_torch.tracing import ProxyTensor, _tracer


def test_proxy_tensor_ops() -> None:
    """Verifies `ProxyTensor` unary, binary, bitwise, indexing, and matmul tracing ops."""
    _tracer.start_tracing()
    pt = ProxyTensor(id="t1", shape=(2, 3), dtype="float32")
    ProxyTensor(id="t2", shape=(2, 3), dtype="float32")

    _ = -pt
    _ = +pt
    _ = abs(pt)
    _ = ~pt

    _ = pt + 1
    _ = 1 + pt
    _ = pt - 1
    _ = 1 - pt
    _ = pt * 2
    _ = 2 * pt
    _ = pt / 2
    _ = 2 / pt
    _ = pt**2
    _ = pt // 2
    _ = 2 // pt
    _ = pt % 2
    _ = 2 % pt

    pt_int = ProxyTensor(id="ti1", shape=(2,), dtype="int32")
    _ = pt_int & 1
    _ = 1 & pt_int
    _ = pt_int | 1
    _ = 1 | pt_int
    _ = pt_int ^ 1
    _ = 1 ^ pt_int
    _ = pt_int << 1
    _ = 1 << pt_int
    _ = pt_int >> 1
    _ = 1 >> pt_int

    _ = pt[0]

    pt_mat2 = ProxyTensor(id="t3", shape=(3, 4), dtype="float32")
    _ = pt @ pt_mat2

    _tracer.stop_tracing()


def test_proxy_tensor_errors() -> None:
    """Verifies `ProxyTensor` raises appropriate errors when tracing is inactive or types mismatch."""
    pt = ProxyTensor(id="t1", shape=(2, 3), dtype="float32")

    with pytest.raises(RecursionError):
        _ = pt + 1

    with pytest.raises(RecursionError):
        _ = -pt

    with pytest.raises(TracingError, match="Cannot perform Slice"):
        _ = pt[0]

    with pytest.raises(TracingError, match="Cannot perform MatMul"):
        _ = pt @ pt

    _tracer.start_tracing()
    with pytest.raises(ValueError, match="must be a ProxyTensor"):
        _ = pt @ 1
    _tracer.stop_tracing()


def test_add_node_error() -> None:
    """Verifies `_tracer.add_node` is a no-op when `active_graph` is None."""
    _tracer.add_node(IRNode(id="x", op_type="Add", inputs=[]))
    assert _tracer.active_graph is None
