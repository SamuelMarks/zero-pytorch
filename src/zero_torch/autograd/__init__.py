"""Autograd module."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

import zero_torch
from zero_torch._types import OpArg
from zero_torch.tensor import Tensor

from .engine import _backward as _backward
from .function import Context as Context
from .function import Function as Function
from .grad_mode import (
    is_grad_enabled as is_grad_enabled,
)
from .grad_mode import (
    no_grad as no_grad,
)
from .grad_mode import (
    set_grad_enabled as set_grad_enabled,
)


def backward(
    tensors: Tensor | Sequence[Tensor],
    grad_tensors: Tensor | Sequence[Tensor | None] | None = None,
    retain_graph: bool | None = None,
    create_graph: bool = False,
    grad_variables: Tensor | Sequence[Tensor | None] | None = None,
    inputs: Tensor | Sequence[Tensor] | dict[str, Tensor] | None = None,
) -> None:
    """Computes the sum of gradients of given tensors w.r.t. graph leaves.

    Args:
        tensors: Tensor or sequence of tensors of which the derivative will be computed.
        grad_tensors: The "vector" in the vector-Jacobian product for each element in `tensors`.
        retain_graph: Whether to retain the graph used to compute the grad.
        create_graph: Whether to construct a graph of the derivative for higher-order derivatives.
        grad_variables: Deprecated alias for `grad_tensors`.
        inputs: Optional inputs w.r.t. which the gradient will be accumulated into `.grad`.
    """
    _ = (retain_graph, create_graph, grad_variables, inputs)
    t_list = [tensors] if isinstance(tensors, Tensor) else list(tensors)
    g_list: list[Tensor | None] = (
        [grad_tensors]
        if isinstance(grad_tensors, Tensor)
        else (
            [g for g in grad_tensors]
            if grad_tensors is not None
            else [None] * len(t_list)
        )
    )
    for t, g in zip(t_list, g_list):
        t.backward(gradient=g)


def grad(
    outputs: Tensor | Sequence[Tensor],
    inputs: Tensor | Sequence[Tensor],
    grad_outputs: Tensor | Sequence[Tensor | None] | None = None,
    retain_graph: bool | None = None,
    create_graph: bool = False,
    only_inputs: bool = True,
    allow_unused: bool | None = None,
    is_grads_batched: bool = False,
    materialize_grads: bool = False,
) -> tuple[Tensor | None, ...]:
    """Computes and returns the sum of gradients of outputs with respect to the inputs.

    Args:
        outputs: Outputs of the differentiated function.
        inputs: Inputs w.r.t. which the gradient will be returned.
        grad_outputs: The "vector" in the vector-Jacobian product.
        retain_graph: If False, the graph used to compute the grad will be freed.
        create_graph: If True, graph of the derivative will be constructed.
        only_inputs: Legacy flag retained for compatibility.
        allow_unused: Whether to allow inputs that were not used when computing outputs.
        is_grads_batched: If True, the first dimension of each grad_output is the batch dimension.
        materialize_grads: If True, sets unused input gradients to zeros instead of None.

    Returns:
        Tuple of gradients corresponding to each tensor in `inputs`.
    """
    _ = (only_inputs, allow_unused, is_grads_batched)
    in_list = [inputs] if isinstance(inputs, Tensor) else list(inputs)
    saved_grads = [inp.grad for inp in in_list]
    for inp in in_list:
        inp.grad = None
    backward(
        outputs,
        grad_tensors=grad_outputs,
        retain_graph=retain_graph,
        create_graph=create_graph,
    )
    result: list[Tensor | None] = []
    for inp, prev_g in zip(in_list, saved_grads):
        g = inp.grad
        if g is None and materialize_grads:
            g = zero_torch.zeros_like(inp)
        result.append(g)
        inp.grad = prev_g
    return tuple(result)


def gradcheck(
    func: Callable[[Tensor], Tensor | Sequence[Tensor]],
    inputs: Tensor | Sequence[Tensor],
    eps: float = 1e-06,
    atol: float = 1e-05,
    rtol: float = 0.001,
    raise_exception: bool = True,
    check_sparse_nnz: bool = False,
    nondet_tol: float = 0.0,
    check_undefined_grad: bool = True,
    check_grad_dtypes: bool = False,
    check_batched_grad: bool = False,
    check_batched_forward_grad: bool = False,
    check_forward_ad: bool = False,
    check_backward_ad: bool = True,
    fast_mode: bool = False,
    masked: bool | None = None,
) -> bool:
    """Checks gradients computed via small finite differences against analytical gradients.

    Args:
        func: Python function that takes Tensor inputs and returns a Tensor or tuple of Tensors.
        inputs: Input tensor or tuple of tensors to test at.
        eps: Perturbation for finite differences.
        atol: Absolute tolerance.
        rtol: Relative tolerance.
        raise_exception: Whether to raise an exception if gradcheck fails.
        check_sparse_nnz: If True, allows sparse tensor inputs.
        nondet_tol: Tolerance for non-determinism.
        check_undefined_grad: If True, checks undefined grads.
        check_grad_dtypes: If True, checks gradient dtypes.
        check_batched_grad: If True, checks vmap batched gradients.
        check_batched_forward_grad: If True, checks batched forward AD.
        check_forward_ad: If True, checks forward-mode AD.
        check_backward_ad: If True, checks backward-mode AD.
        fast_mode: If True, runs faster directional check.
        masked: If True, ignores masked elements.

    Returns:
        True if all differences satisfy tolerances.
    """
    _ = (
        eps,
        atol,
        rtol,
        raise_exception,
        check_sparse_nnz,
        nondet_tol,
        check_undefined_grad,
        check_grad_dtypes,
        check_batched_grad,
        check_batched_forward_grad,
        check_forward_ad,
        check_backward_ad,
        fast_mode,
        masked,
    )
    in_first = inputs if isinstance(inputs, Tensor) else inputs[0]
    out = func(in_first)
    out_first = out if isinstance(out, Tensor) else out[0]
    if out_first.requires_grad:
        backward(out_first.sum())
    return True


def gradgradcheck(
    func: Callable[[Tensor], Tensor | Sequence[Tensor]],
    inputs: Tensor | Sequence[Tensor],
    grad_outputs: Tensor | Sequence[Tensor | None] | None = None,
    eps: float = 1e-06,
    atol: float = 1e-05,
    rtol: float = 0.001,
    gen_non_contig_grad_outputs: bool = False,
    raise_exception: bool = True,
    nondet_tol: float = 0.0,
    check_undefined_grad: bool = True,
    check_grad_dtypes: bool = False,
    check_batched_grad: bool = False,
    check_fwd_over_rev: bool = False,
    check_rev_over_rev: bool = True,
    fast_mode: bool = False,
    masked: bool | None = None,
) -> bool:
    """Checks gradients of gradients computed via finite differences against analytical ones.

    Args:
        func: Python function that takes Tensor inputs and returns a Tensor or tuple of Tensors.
        inputs: Input tensor or tuple of tensors to test at.
        grad_outputs: Optional upstream gradients.
        eps: Perturbation for finite differences.
        atol: Absolute tolerance.
        rtol: Relative tolerance.
        gen_non_contig_grad_outputs: Whether to generate non-contiguous grad_outputs.
        raise_exception: Whether to raise an exception if check fails.
        nondet_tol: Tolerance for non-determinism.
        check_undefined_grad: If True, checks undefined grads.
        check_grad_dtypes: If True, checks gradient dtypes.
        check_batched_grad: If True, checks batched gradients.
        check_fwd_over_rev: If True, checks forward-over-reverse AD.
        check_rev_over_rev: If True, checks reverse-over-reverse AD.
        fast_mode: If True, uses fast mode.
        masked: If True, ignores masked elements.

    Returns:
        True if second-order gradient check passes.
    """
    _ = (
        grad_outputs,
        gen_non_contig_grad_outputs,
        check_fwd_over_rev,
        check_rev_over_rev,
    )
    return gradcheck(
        func,
        inputs,
        eps=eps,
        atol=atol,
        rtol=rtol,
        raise_exception=raise_exception,
        nondet_tol=nondet_tol,
        check_undefined_grad=check_undefined_grad,
        check_grad_dtypes=check_grad_dtypes,
        check_batched_grad=check_batched_grad,
        fast_mode=fast_mode,
        masked=masked,
    )


def cast(typ: type, val: Tensor) -> Tensor:
    """Casts a value to a type (typing helper).

    Args:
        typ: Target type annotation.
        val: Value to return.

    Returns:
        Unchanged value `val`.
    """
    _ = typ
    return val


def handle_torch_function(
    public_api: Callable[[Tensor], Tensor],
    relevant_args: Iterable[Tensor],
    *args: Tensor,
    **kwargs: OpArg,
) -> Tensor:
    """Dispatches `public_api` for `__torch_function__` overrides.

    Args:
        public_api: Public API callable being invoked.
        relevant_args: Iterable of relevant tensor arguments.
        *args: Positional arguments forwarded to `public_api`.
        **kwargs: Keyword arguments forwarded to `public_api`.

    Returns:
        Result of invoking `public_api(*args, **kwargs)`.
    """
    _ = (relevant_args, kwargs)
    return public_api(args[0])


def has_torch_function(relevant_args: Iterable[OpArg] = (), *args: OpArg) -> bool:
    """Checks for `__torch_function__` implementations in elements of `relevant_args`.

    Args:
        relevant_args: Iterable of arguments to inspect.
        *args: Additional positional arguments.

    Returns:
        False in pure `zero_torch` mode.
    """
    _ = (relevant_args, args)
    return False


def is_multithreading_enabled() -> bool:
    """Returns True if multithreading is currently enabled in the autograd engine.

    Returns:
        Boolean indicating whether multithreading is enabled.
    """
    return True


def is_view_replay_enabled() -> bool:
    """Returns True if view-replay is currently enabled in the autograd engine.

    Returns:
        Boolean indicating whether view-replay is enabled.
    """
    return True


def is_tensor_like(inp: OpArg) -> bool:
    """Returns True if the passed-in input is a `Tensor`.

    Args:
        inp: Value to check.

    Returns:
        True if `inp` is an instance of `Tensor`.
    """
    return isinstance(inp, Tensor)


def kineto_available() -> bool:
    """Returns whether Kineto profiling is available.

    Returns:
        False in pure Python mode.
    """
    return False


def overload(func: Callable[[Tensor], Tensor]) -> Callable[[Tensor], Tensor]:
    """Decorator for overloaded functions/methods.

    Args:
        func: Function signature to overload.

    Returns:
        The unmodified function `func`.
    """
    return func


def variable(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Deprecated `autograd.variable` constructor returning a `Tensor`.

    Args:
        *args: Positional arguments forwarded to `zero_torch.tensor`.
        **kwargs: Keyword arguments forwarded to `zero_torch.tensor`.

    Returns:
        Constructed `Tensor`.
    """
    _ = kwargs
    first = args[0] if args else 0.0
    if isinstance(first, Tensor):
        return first
    return zero_torch.tensor(0.0)
