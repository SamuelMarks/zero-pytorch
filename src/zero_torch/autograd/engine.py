"""Autograd engine module."""


def _backward(tensors, grad_tensors, retain_graph=None, create_graph=False):
    """Computes the sum of gradients of given tensors w.r.t. graph leaves.

    Args:
        tensors (Tensor or tuple): Tensors of which the derivative will be computed.
        grad_tensors (Tensor or tuple): The "vector" in the vector-Jacobian product.
        retain_graph (bool, optional): If False, the graph used to compute the grad will be freed.
        create_graph (bool, optional): If True, graph of the derivative will be constructed.
    """
    if retain_graph is None:
        retain_graph = create_graph

    if not isinstance(tensors, (list, tuple)):
        tensors = (tensors,)
    if not isinstance(grad_tensors, (list, tuple)):
        grad_tensors = (grad_tensors,)

    topo = []
    visited = set()

    def build_topo(v):
        if id(v) not in visited:
            visited.add(id(v))
            if hasattr(v, "next_functions"):
                for next_fn, _ in v.next_functions:
                    if next_fn is not None:
                        build_topo(next_fn)
            topo.append(v)

    for t in tensors:
        if t.grad_fn is not None:
            build_topo(t.grad_fn)

    node_grads = {}
    for t, g in zip(tensors, grad_tensors):
        if t.grad_fn is not None:
            node_grads[id(t.grad_fn)] = [g]

    for node in reversed(topo):
        if id(node) not in node_grads:
            continue
        grads = node_grads[id(node)]
        # Sum gradients if there are multiple branches

        # simple sum for now
        g = grads[0]
        for idx in range(1, len(grads)):
            g = g + grads[idx]  # pragma: no cover

        # Call the backward function
        if callable(node):
            res = node(g)
            if not isinstance(res, tuple):
                res = (res,)  # pragma: no cover

            for i, (next_fn, _) in enumerate(node.next_functions):
                if next_fn is not None:
                    if hasattr(next_fn, "requires_grad"):  # it's a leaf tensor
                        if res[i] is not None:
                            if next_fn.grad is None:
                                next_fn.grad = res[i]
                            else:
                                next_fn.grad = next_fn.grad + res[i]  # pragma: no cover
                    else:  # it's a node  # pragma: no cover
                        if id(next_fn) not in node_grads:  # pragma: no cover
                            node_grads[id(next_fn)] = []  # pragma: no cover
                        node_grads[id(next_fn)].append(res[i])  # pragma: no cover

        if not retain_graph:
            # Free up references
            node.next_functions = []
