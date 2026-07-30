"""Tracing engine for constructing LogicalGraphs via operator overloading."""

from __future__ import annotations

from ml_switcheroo_compiler.tracing.state import global_tracing_state as _tracer
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor

__all__ = ["ProxyTensor", "_tracer"]
