"""Autograd engine module."""

from __future__ import annotations

from collections.abc import Sequence

import zero_torch
from zero_torch.tensor import Tensor

from .function import Function
from .grad_mode import set_grad_enabled


def _run_tensor_hooks(tensor: Tensor, grad: Tensor) -> Tensor:
    """Executes all registered gradient hooks on `tensor`.

    Args:
        tensor: Tensor whose `_grad_hooks` should be invoked.
        grad: Incoming gradient tensor.

    Returns:
        Modified gradient tensor if any hook returns a `Tensor`, else `grad`.
    """
    curr = grad
    for hook in list(tensor._grad_hooks.values()):
        res = hook(curr)
        if isinstance(res, Tensor):
            curr = res
    return curr


def _backward(
    tensors: Tensor | Sequence[Tensor],
    grad_tensors: Tensor | Sequence[Tensor | None] | None = None,
    retain_graph: bool | None = None,
    create_graph: bool = False,
    inputs: Sequence[Tensor] | Tensor | dict[str, Tensor] | None = None,
) -> dict[int, Tensor]:
    """Computes the sum of gradients of given tensors w.r.t. graph leaves.

    Args:
        tensors: Tensor or sequence of tensors of which the derivative will be computed.
        grad_tensors: The vector in the vector-Jacobian product matching `tensors`.
        retain_graph: If False, the graph used to compute the grad will be freed.
        create_graph: If True, graph of the derivative will be constructed.
        inputs: Optional leaf/intermediate tensors to restrict `.grad` accumulation to.

    Returns:
        Dictionary mapping `id(tensor)` to computed gradient `Tensor`.
    """
    if retain_graph is None:
        retain_graph = create_graph

    allowed_ids: set[int] | None = None
    if inputs is not None:
        if isinstance(inputs, Tensor):
            allowed_ids = {id(inputs)}
        elif isinstance(inputs, dict):
            allowed_ids = {id(v) for v in inputs.values()}
        else:
            allowed_ids = {id(t) for t in inputs}

    tensor_seq: tuple[Tensor, ...] = (
        (tensors,) if isinstance(tensors, Tensor) else tuple(tensors)
    )
    raw_grads: list[Tensor | None] = (
        [grad_tensors]
        if isinstance(grad_tensors, Tensor)
        else (
            list(grad_tensors) if grad_tensors is not None else [None] * len(tensor_seq)
        )
    )

    grad_seq: list[Tensor] = []
    for t, g in zip(tensor_seq, raw_grads):
        if g is None:
            grad_seq.append(
                zero_torch.ones_like(t, dtype=t.dtype, requires_grad=create_graph)
            )
        elif create_graph and not g.requires_grad:
            grad_seq.append(Tensor(g, requires_grad=True, dtype=g.dtype))
        else:
            grad_seq.append(g)

    computed_grads: dict[int, Tensor] = {}

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
        g_hooked = _run_tensor_hooks(t, g)
        computed_grads[id(t)] = g_hooked
        if t.grad_fn is not None:
            node_grads.setdefault(id(t.grad_fn), []).append(g_hooked)
        elif t.requires_grad and (allowed_ids is None or id(t) in allowed_ids):
            if t.grad is None:
                t.grad = g_hooked
            else:
                t.grad = t.grad + g_hooked

    with set_grad_enabled(create_graph):
        for node in reversed(topo):
            if id(node) not in node_grads:
                continue
            grads = node_grads[id(node)]
            g = grads[0]
            for idx in range(1, len(grads)):
                g = g + grads[idx]

            for out_t in node.output_tensors:
                g = _run_tensor_hooks(out_t, g)
                computed_grads[id(out_t)] = g
                if out_t._retains_grad and (
                    allowed_ids is None or id(out_t) in allowed_ids
                ):
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
                            grad_i = _run_tensor_hooks(next_fn, grad_i)
                            prev = computed_grads.get(id(next_fn))
                            computed_grads[id(next_fn)] = (
                                grad_i if prev is None else prev + grad_i
                            )
                            if allowed_ids is None or id(next_fn) in allowed_ids:
                                if next_fn.grad is None:
                                    next_fn.grad = grad_i
                                else:
                                    next_fn.grad = next_fn.grad + grad_i
                        else:
                            node_grads.setdefault(id(next_fn), []).append(grad_i)

            if not retain_graph:
                node.next_functions = []

    return computed_grads
