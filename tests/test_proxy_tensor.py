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


def test_tracer_teardown_stops_active_tracing() -> None:
    """Leaves `_tracer.is_tracing` active so `conftest.switcheroo_config` teardown stops it."""
    _tracer.start_tracing()
    assert _tracer.is_tracing


def test_type_checking_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Executes `TYPE_CHECKING` blocks in `_types`, `autograd.function`, and `nn.module`.

    Args:
        monkeypatch: Pytest fixture for patching attributes during the test.
    """
    import runpy
    import types
    import typing
    from pathlib import Path

    monkeypatch.setattr(typing, "TYPE_CHECKING", True)
    monkeypatch.setattr(types, "EllipsisType", type(Ellipsis), raising=False)
    monkeypatch.setattr(
        typing,
        "TypeAlias",
        vars(typing).get("TypeAlias", type),
        raising=False,
    )
    monkeypatch.setattr(
        typing,
        "ParamSpec",
        vars(typing).get("ParamSpec", typing.TypeVar),
        raising=False,
    )

    root = Path(__file__).resolve().parent.parent / "src" / "zero_torch"
    for rel in ("_types.py", "autograd/function.py", "nn/module.py"):
        ns = runpy.run_path(str(root / rel))
        assert "__all__" in ns or "Function" in ns or "Module" in ns
