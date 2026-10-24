"""Autograd engine module."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from .function import Function

if TYPE_CHECKING:
    from zero_torch.tensor import Tensor


def _backward(
    tensors: Tensor | Sequence[Tensor],
    grad_tensors: Tensor | Sequence[Tensor],
    retain_graph: bool | None = None,
    create_graph: bool = False,
) -> None:
    """Computes the sum of gradients of given tensors w.r.t. graph leaves.

    Args:
        tensors: Tensor or sequence of tensors of which the derivative will be computed.
        grad_tensors: The vector in the vector-Jacobian product matching `tensors`.
        retain_graph: If False, the graph used to compute the grad will be freed.
        create_graph: If True, graph of the derivative will be constructed.
    """
    from zero_torch.tensor import Tensor

    if retain_graph is None:
        retain_graph = create_graph

    tensor_seq: tuple[Tensor, ...] = (
        (tensors,) if isinstance(tensors, Tensor) else tuple(tensors)
    )
    grad_seq: tuple[Tensor, ...] = (
        (grad_tensors,) if isinstance(grad_tensors, Tensor) else tuple(grad_tensors)
    )

    topo: list[Function] = []
    visited: set[int] = set()

    def build_topo(v: Function) -> None:
        """Builds topological order of backward Function nodes.

        Args:
            v: Current Function node in the autograd graph.
        """
        if id(v) not in visited:
            visited.add(id(v))
            for next_fn, _ in v.next_functions:
                if isinstance(next_fn, Function):
                    build_topo(next_fn)
            topo.append(v)

    for t in tensor_seq:
        if t.grad_fn is not None:
            build_topo(t.grad_fn)

    node_grads: dict[int, list[Tensor]] = {}
    for t, g in zip(tensor_seq, grad_seq):
        if t.grad_fn is not None:
            node_grads[id(t.grad_fn)] = [g]
        elif t.requires_grad:
            if t.grad is None:
                t.grad = g
            else:
                t.grad = t.grad + g

    for node in reversed(topo):
        if id(node) not in node_grads:
            continue
        grads = node_grads[id(node)]
        g = grads[0]
        for idx in range(1, len(grads)):
            g = g + grads[idx]

        for out_t in node.output_tensors:
            if out_t._retains_grad:
                if out_t.grad is None:
                    out_t.grad = g
                else:
                    out_t.grad = out_t.grad + g

        raw_res = node(g)
        res_tuple: tuple[Tensor | None, ...] = (
            raw_res if isinstance(raw_res, tuple) else (raw_res,)
        )

        for i, (next_fn, _) in enumerate(node.next_functions):
            if next_fn is not None and i < len(res_tuple):
                grad_i = res_tuple[i]
                if grad_i is not None:
                    if isinstance(next_fn, Tensor):
                        if next_fn.grad is None:
                            next_fn.grad = grad_i
                        else:
                            next_fn.grad = next_fn.grad + grad_i
                    else:
                        if id(next_fn) not in node_grads:
                            node_grads[id(next_fn)] = []
                        node_grads[id(next_fn)].append(grad_i)

        if not retain_graph:
            node.next_functions = []
