"""Tracing engine for constructing LogicalGraphs via operator overloading."""

from __future__ import annotations

import threading
import uuid
import ml_switcheroo_compiler.tracing
from typing import TypeVar

from ml_switcheroo_ir import LogicalGraph
from ml_switcheroo_compiler.ir.core import IRNode

# We use simple broadcast_shapes from numpy since we can't import numpy in source!
# Wait! We can import from ml_switcheroo_compiler.ir.shape_system
from ml_switcheroo_compiler.ir.shape_system import broadcast_shapes

T = TypeVar("T", bound="ProxyTensor")


class TracerTape(threading.local):
    """Thread-local tape for tracking active graph construction."""

    def __init__(self) -> None:
        """Initialize the tracer tape."""
        self.active_graph: LogicalGraph | None = None
        self.is_tracing: bool = False

    def start_tracing(self, name: str = "Model") -> LogicalGraph:
        """Begin tracking a new graph."""
        self.active_graph = LogicalGraph(name=name)
        self.is_tracing = True
        return self.active_graph

    def stop_tracing(self) -> LogicalGraph | None:
        """Stop tracking and return the current graph."""
        graph = self.active_graph
        self.active_graph = None
        self.is_tracing = False
        return graph

    def add_node(self, node: IRNode) -> None:
        """Add a node to the active graph."""
        if not self.is_tracing or self.active_graph is None:
            msg = "Cannot add node: not currently tracing."
            raise RuntimeError(msg)

        if getattr(node, "source_ast_ref", None) is None:
            from ml_switcheroo_compiler.backends.linker import get_source_ast_ref

            node.source_ast_ref = get_source_ast_ref(back_frames=2)

        from ml_switcheroo_compiler.core.config import config

        if (
            hasattr(node, "stream")
            and getattr(node, "stream", None) is None
            and config.current_stream != "default"
        ):
            node.stream = config.current_stream

        self.active_graph.nodes[node.id] = node


# Global tracer instance

_tracer = ml_switcheroo_compiler.tracing._tracer
_tracer.__class__ = TracerTape


class ProxyTensor:
    """A proxy object that intercepts mathematical operations and builds the IR graph."""

    def __init__(
        self,
        id: str,
        shape: tuple[int | str, ...],
        dtype: str = "float32",
    ) -> None:
        """Initialize a ProxyTensor."""
        self.id = id
        self.shape = shape
        self.dtype = dtype

    def _binary_op(self, other: object, op_type: str) -> ProxyTensor:
        """Help with binary operations."""
        if not _tracer.is_tracing:
            msg = f"Cannot perform {op_type} outside of a tracing context."
            raise RuntimeError(msg)

        other_id = getattr(other, "id", None)
        other_shape = getattr(other, "shape", ())

        out_shape = broadcast_shapes(self.shape, other_shape)
        out_dtype = self.dtype

        if other_id is None:
            other_id = str(uuid.uuid4())
            const_node = IRNode(
                id=other_id,
                op_type="Constant",
                attributes={"value": other},
                shape_metadata=(),
            )
            _tracer.add_node(const_node)

        out_id = str(uuid.uuid4())
        node = IRNode(
            id=out_id,
            op_type=op_type,
            inputs=[self.id, other_id],
            shape_metadata=out_shape,
        )
        _tracer.add_node(node)

        return ProxyTensor(id=out_id, shape=out_shape, dtype=out_dtype)

    def __add__(self, other: object) -> ProxyTensor:
        return self._binary_op(other, "Add")

    def __radd__(self, other: object) -> ProxyTensor:
        return self._binary_op(other, "Add")

    def __sub__(self, other: object) -> ProxyTensor:
        return self._binary_op(other, "Sub")

    def __rsub__(self, other: object) -> ProxyTensor:
        return self._binary_op(other, "Sub")

    def __mul__(self, other: object) -> ProxyTensor:
        return self._binary_op(other, "Mul")

    def __rmul__(self, other: object) -> ProxyTensor:
        return self._binary_op(other, "Mul")

    def __truediv__(self, other: object) -> ProxyTensor:
        return self._binary_op(other, "Div")

    def __rtruediv__(self, other: object) -> ProxyTensor:
        return self._binary_op(other, "Div")

    def __pow__(self, other: object) -> ProxyTensor:
        return self._binary_op(other, "Pow")

    def __floordiv__(self, other: object) -> ProxyTensor:
        return self._binary_op(other, "FloorDiv")

    def __rfloordiv__(self, other: object) -> ProxyTensor:
        return self._binary_op(other, "FloorDiv")

    def __mod__(self, other: object) -> ProxyTensor:
        return self._binary_op(other, "Mod")

    def __rmod__(self, other: object) -> ProxyTensor:
        return self._binary_op(other, "Mod")

    def __and__(self, other: object) -> ProxyTensor:
        return self._binary_op(other, "BitwiseAnd")

    def __rand__(self, other: object) -> ProxyTensor:
        return self._binary_op(other, "BitwiseAnd")

    def __or__(self, other: object) -> ProxyTensor:
        return self._binary_op(other, "BitwiseOr")

    def __ror__(self, other: object) -> ProxyTensor:
        return self._binary_op(other, "BitwiseOr")

    def __xor__(self, other: object) -> ProxyTensor:
        return self._binary_op(other, "BitwiseXor")

    def __rxor__(self, other: object) -> ProxyTensor:
        return self._binary_op(other, "BitwiseXor")

    def __lshift__(self, other: object) -> ProxyTensor:
        return self._binary_op(other, "BitShiftLeft")

    def __rlshift__(self, other: object) -> ProxyTensor:
        return self._binary_op(other, "BitShiftLeft")

    def __rshift__(self, other: object) -> ProxyTensor:
        return self._binary_op(other, "BitShiftRight")

    def __rrshift__(self, other: object) -> ProxyTensor:
        return self._binary_op(other, "BitShiftRight")

    def _unary_op(self, op_type: str) -> ProxyTensor:
        if not _tracer.is_tracing:
            msg = f"Cannot perform {op_type} outside of a tracing context."
            raise RuntimeError(msg)

        out_id = str(uuid.uuid4())
        node = IRNode(
            id=out_id,
            op_type=op_type,
            inputs=[self.id],
            shape_metadata=self.shape,
        )
        _tracer.add_node(node)
        return ProxyTensor(id=out_id, shape=self.shape, dtype=self.dtype)

    def __neg__(self) -> ProxyTensor:
        return self._unary_op("Neg")

    def __pos__(self) -> ProxyTensor:
        return self

    def __abs__(self) -> ProxyTensor:
        return self._unary_op("Abs")

    def __invert__(self) -> ProxyTensor:
        return self._unary_op("BitwiseNot")

    def __getitem__(self, key: object) -> ProxyTensor:
        if not _tracer.is_tracing:
            msg = "Cannot perform Slice outside of a tracing context."
            raise RuntimeError(msg)

        out_id = str(uuid.uuid4())
        node = IRNode(
            id=out_id,
            op_type="Slice",
            inputs=[self.id],
            attributes={"slices": str(key)},
            shape_metadata=self.shape,
        )
        _tracer.add_node(node)
        return ProxyTensor(id=out_id, shape=self.shape, dtype=self.dtype)

    def __matmul__(self, other: object) -> ProxyTensor:
        from ml_switcheroo_compiler.ir.shape_system import matmul_shape

        if not _tracer.is_tracing:
            msg = "Cannot perform MatMul outside of a tracing context."
            raise RuntimeError(msg)

        other_id = getattr(other, "id", None)
        if other_id is None:
            msg = "MatMul right hand side must be a ProxyTensor."
            raise ValueError(msg)

        other_shape = getattr(other, "shape", ())

        out_shape = matmul_shape(self.shape, other_shape)
        out_dtype = self.dtype

        out_id = str(uuid.uuid4())
        node = IRNode(
            id=out_id,
            op_type="MatMul",
            inputs=[self.id, other_id],
            shape_metadata=out_shape,
        )
        _tracer.add_node(node)

        return ProxyTensor(id=out_id, shape=out_shape, dtype=out_dtype)


# Patch the compiler's tracer so that frontend tracing works seamlessly

ml_switcheroo_compiler.tracing._tracer = _tracer
ml_switcheroo_compiler.tracing.tracer._tracer = _tracer
ml_switcheroo_compiler.tracing.ProxyTensor = ProxyTensor
ml_switcheroo_compiler.tracing.tracer.ProxyTensor = ProxyTensor
