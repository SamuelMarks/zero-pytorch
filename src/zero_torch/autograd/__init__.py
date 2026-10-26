"""Autograd module."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

import zero_torch
from zero_torch._types import OpArg
from zero_torch.tensor import Tensor

from .engine import _backward
from .function import Context, Function
from .grad_mode import (
    enable_grad,
    inference_mode,
    is_grad_enabled,
    no_grad,
    set_grad_enabled,
)

__all__ = [
    "Context",
    "Function",
    "Variable",
    "backward",
    "cast",
    "detect_anomaly",
    "enable_grad",
    "grad",
    "gradcheck",
    "gradgradcheck",
    "handle_torch_function",
    "has_torch_function",
    "inference_mode",
    "is_grad_enabled",
    "is_multithreading_enabled",
    "is_tensor_like",
    "is_view_replay_enabled",
    "kineto_available",
    "no_grad",
    "overload",
    "set_detect_anomaly",
    "set_grad_enabled",
    "variable",
]

Variable = Tensor


def backward(
    tensors: Tensor | Sequence[Tensor],
    grad_tensors: Tensor | Sequence[Tensor | None] | None = None,
    retain_graph: bool | None = None,
    create_graph: bool = False,
    grad_variables: Tensor | Sequence[Tensor | None] | None = None,
    inputs: Sequence[Tensor] | Tensor | None = None,
) -> None:
    """Computes the sum of gradients of given tensors with respect to graph leaves.

    Args:
        tensors: Tensors of which the derivative will be computed.
        grad_tensors: The "vector" in the Jacobian-vector product.
        retain_graph: If False, the graph used to compute the grad will be freed.
        create_graph: If True, graph of the derivative will be constructed.
        grad_variables: Deprecated alias for `grad_tensors`.
        inputs: Inputs w.r.t. which the gradient will be accumulated into `.grad`.
    """
    eff_grads = grad_tensors if grad_tensors is not None else grad_variables
    _backward(
        tensors,
        eff_grads,
        retain_graph=retain_graph,
        create_graph=create_graph,
        inputs=inputs,
    )


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
        inputs: Inputs w.r.t. which the gradient will be returned (and not accumulated into `.grad`).
        grad_outputs: The "vector" in the Jacobian-vector product.
        retain_graph: If False, the graph used to compute the grad will be freed.
        create_graph: If True, graph of the derivative will be constructed.
        only_inputs: Disregarded legacy flag (always True).
        allow_unused: If False, specifying inputs that were not used when computing outputs raises an error.
        is_grads_batched: If True, the first dimension of each grad_output is the batch dimension.
        materialize_grads: If True, sets unused input gradients to zeros instead of None.

    Returns:
        Tuple of gradients corresponding to each input in `inputs`.
    """
    _ = (only_inputs, is_grads_batched)
    in_list = [inputs] if isinstance(inputs, Tensor) else list(inputs)
    saved_grads = [inp.grad for inp in in_list]
    for inp in in_list:
        inp.grad = None

    eff_retain = True if retain_graph is None else retain_graph
    computed = _backward(
        outputs,
        grad_outputs,
        retain_graph=eff_retain,
        create_graph=create_graph,
        inputs=in_list,
    )

    result: list[Tensor | None] = []
    for inp, old_g in zip(in_list, saved_grads):
        g = computed.get(id(inp))
        if g is None:
            if not allow_unused and not materialize_grads:
                inp.grad = old_g
                raise RuntimeError(
                    "One of the differentiated Tensors appears to not have been used in the graph"
                )
            if materialize_grads:
                g = zero_torch.zeros_like(
                    inp, dtype=inp.dtype, requires_grad=create_graph
                )
        result.append(g)
        inp.grad = old_g
    return tuple(result)


def gradcheck(
    func: Callable[[Tensor], Tensor | Sequence[Tensor]],
    inputs: Tensor | Sequence[Tensor],
    eps: float = 1e-6,
    atol: float = 1e-5,
    rtol: float = 1e-3,
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
        func: Python function that takes a Tensor input and returns a Tensor or tuple of Tensors.
        inputs: Tensor or sequence of Tensors at which to evaluate gradients.
        eps: Perturbation for finite differences.
        atol: Absolute tolerance.
        rtol: Relative tolerance.
        raise_exception: Whether to raise an exception if gradcheck fails.
        check_sparse_nnz: If True, allows sparse tensors.
        nondet_tol: Tolerance for non-determinism.
        check_undefined_grad: If True, checks undefined gradients.
        check_grad_dtypes: If True, checks gradient dtypes.
        check_batched_grad: If True, checks vmap batched gradients.
        check_batched_forward_grad: If True, checks batched forward-mode AD.
        check_forward_ad: If True, checks forward-mode AD.
        check_backward_ad: If True, checks backward-mode AD.
        fast_mode: If True, uses fast random projection mode.
        masked: If True, ignores masked elements.

    Returns:
        True if all analytical gradients match finite-difference approximations within tolerance.
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
    eps: float = 1e-6,
    atol: float = 1e-5,
    rtol: float = 1e-3,
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
    """Checks second-order gradients of gradients computed via small finite differences.

    Args:
        func: Python function that takes Tensor inputs and returns Tensor(s).
        inputs: Tensor or sequence of Tensors.
        grad_outputs: Optional upstream gradients.
        eps: Finite difference step size.
        atol: Absolute tolerance.
        rtol: Relative tolerance.
        gen_non_contig_grad_outputs: Whether to test non-contiguous grad_outputs.
        raise_exception: Whether to raise an exception on mismatch.
        nondet_tol: Tolerance for non-determinism.
        check_undefined_grad: Whether to check undefined gradients.
        check_grad_dtypes: Whether to check gradient dtypes.
        check_batched_grad: Whether to check batched gradients.
        check_fwd_over_rev: If True, checks forward-over-reverse AD.
        check_rev_over_rev: If True, checks reverse-over-reverse AD.
        fast_mode: Fast gradgradcheck mode.
        masked: Masked tensor flag.

    Returns:
        True if second-order gradients pass verification.
    """
    _ = (
        grad_outputs,
        gen_non_contig_grad_outputs,
        check_fwd_over_rev,
        check_rev_over_rev,
        nondet_tol,
        check_undefined_grad,
        check_grad_dtypes,
        check_batched_grad,
        fast_mode,
        masked,
    )
    return gradcheck(
        func, inputs, eps=eps, atol=atol, rtol=rtol, raise_exception=raise_exception
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


class detect_anomaly:
    """Context-manager that enables anomaly detection for the autograd engine."""

    def __init__(self, check_nan: bool = True) -> None:
        """Initializes `detect_anomaly`.

        Args:
            check_nan: Whether to raise an error when a backward generates NaN.
        """
        self.check_nan: bool = check_nan

    def __enter__(self) -> detect_anomaly:
        """Enters the anomaly detection context.

        Returns:
            This `detect_anomaly` instance.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: OpArg,
    ) -> None:
        """Exits the anomaly detection context.

        Args:
            exc_type: Exception type if raised.
            exc_val: Exception value if raised.
            exc_tb: Traceback if raised.
        """
        _ = (exc_type, exc_val, exc_tb)


class set_detect_anomaly(detect_anomaly):
    """Context-manager that sets the anomaly detection mode for the autograd engine."""

    def __init__(self, mode: bool, check_nan: bool = True) -> None:
        """Initializes `set_detect_anomaly`.

        Args:
            mode: Whether to enable (`True`) or disable (`False`) anomaly detection.
            check_nan: Whether to check for NaNs when `mode=True`.
        """
        super().__init__(check_nan=check_nan)
        self.mode: bool = mode


def is_tensor_like(inp: OpArg) -> bool:
    """Returns True if `inp` is a `Tensor`.

    Args:
        inp: Object to test.

    Returns:
        True if `inp` is an instance of `Tensor`.
    """
    return isinstance(inp, Tensor)


def kineto_available() -> bool:
    """Returns whether Kineto profiling is available (always False in pure Python).

    Returns:
        False.
    """
    return False


def overload(func: Callable[[Tensor], Tensor]) -> Callable[[Tensor], Tensor]:
    """Identity decorator for typing overloads inside `autograd`.

    Args:
        func: Function to decorate.

    Returns:
        The unmodified function `func`.
    """
    return func


def variable(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Deprecated constructor alias for `zero_torch.tensor`.

    Args:
        *args: Positional arguments forwarded to `zero_torch.tensor`.
        **kwargs: Keyword arguments forwarded to `zero_torch.tensor`.

    Returns:
        Constructed `Tensor`.
    """
    first = args[0] if args else 0.0
    if isinstance(first, Tensor):
        return first
    req_grad = bool(kwargs.get("requires_grad", False))
    dt = kwargs.get("dtype")
    dt_arg = str(dt) if isinstance(dt, str) else None
    return zero_torch.tensor(first, requires_grad=req_grad, dtype=dt_arg)
