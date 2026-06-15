import pytest
from zero_torch.tracing import ProxyTensor, _tracer


def test_proxy_tensor_ops():
    _tracer.start_tracing()
    pt = ProxyTensor(id="t1", shape=(2, 3), dtype="float32")
    ProxyTensor(id="t2", shape=(2, 3), dtype="float32")

    # Unary
    _ = -pt
    _ = +pt
    _ = abs(pt)
    _ = ~pt

    # Binary with scalar
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

    # Bitwise
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

    # Getitem
    _ = pt[0]

    # Matmul
    pt_mat2 = ProxyTensor(id="t3", shape=(3, 4), dtype="float32")
    _ = pt @ pt_mat2

    _tracer.stop_tracing()


def test_proxy_tensor_errors():
    pt = ProxyTensor(id="t1", shape=(2, 3), dtype="float32")

    with pytest.raises(RuntimeError, match="Cannot perform"):
        _ = pt + 1

    with pytest.raises(RuntimeError, match="Cannot perform"):
        _ = -pt

    with pytest.raises(RuntimeError, match="Cannot perform Slice"):
        _ = pt[0]

    with pytest.raises(RuntimeError, match="Cannot perform MatMul"):
        _ = pt @ pt

    _tracer.start_tracing()
    with pytest.raises(ValueError, match="must be a ProxyTensor"):
        _ = pt @ 1
    _tracer.stop_tracing()


def test_add_node_error():
    from ml_switcheroo_compiler.ir.core import IRNode

    with pytest.raises(RuntimeError, match="Cannot add node: not currently tracing"):
        _tracer.add_node(IRNode(id="x", op_type="Add", inputs=[]))
