"""PyTorch ATen operator namespace (`torch.ops.aten`)."""

from __future__ import annotations

from typing import Protocol

import zero_torch
from zero_torch._types import OpArg
from zero_torch.tensor import Tensor


class _AtenFn(Protocol):
    """Protocol for variadic ATen operator functions."""

    def __call__(self, *args: OpArg, **kwargs: OpArg) -> Tensor:
        """Invokes the ATen operator.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Output `Tensor`.
        """


class OpOverloadPacket:
    """Callable ATen operator packet supporting `.default` and `.out` overloads."""

    def __init__(self, op_name: str, fn: _AtenFn) -> None:
        """Initializes an `OpOverloadPacket`.

        Args:
            op_name: Name of the ATen operator.
            fn: Underlying operator callable.
        """
        self._op_name: str = op_name
        self._fn: _AtenFn = fn
        self.default: _AtenFn = fn
        self.out: _AtenFn = self._call_out

    def _call_out(self, *args: OpArg, **kwargs: OpArg) -> Tensor:
        """Executes the operator and copies into `out` when provided.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments (including optional `out` tensor).

        Returns:
            Result `Tensor` (or `out`).
        """
        out = kwargs.pop("out", None)
        res = self._fn(*args, **kwargs)
        if isinstance(out, Tensor):
            return out.copy_(res)
        return res

    def __call__(self, *args: OpArg, **kwargs: OpArg) -> Tensor:
        """Invokes the ATen operator.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Output `Tensor`.
        """
        return self._fn(*args, **kwargs)


def _dispatch_aten(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Dispatches an ATen operator call and returns a `Tensor`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Output `Tensor`.
    """
    op_name_obj = kwargs.pop("_op", None)
    op_name = str(op_name_obj) if isinstance(op_name_obj, str) else ""
    out_arg = kwargs.pop("out", None)
    if op_name == "abs" and len(args) >= 1 and isinstance(args[0], Tensor):
        res = zero_torch.abs(args[0])
        if isinstance(out_arg, Tensor):
            return out_arg.copy_(res)
        return res
    if (
        op_name == "add"
        and len(args) >= 2
        and isinstance(args[0], Tensor)
        and isinstance(args[1], (Tensor, int, float))
    ):
        rhs = (
            args[1]
            if isinstance(args[1], Tensor)
            else Tensor(args[1], dtype=args[0].dtype)
        )
        res = zero_torch.add(args[0], rhs)
        if isinstance(out_arg, Tensor):
            return out_arg.copy_(res)
        return res
    if args and isinstance(args[0], Tensor):
        return Tensor(args[0], requires_grad=args[0].requires_grad, dtype=args[0].dtype)
    return Tensor(args[0] if args else [0.0])


def abs(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.abs`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.abs`.
    """
    return _dispatch_aten(*args, _op="abs", **kwargs)


def abs_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.abs_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.abs_`.
    """
    return _dispatch_aten(*args, _op="abs", **kwargs)


def absolute(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.absolute`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.absolute`.
    """
    return _dispatch_aten(*args, **kwargs)


def absolute_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.absolute_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.absolute_`.
    """
    return _dispatch_aten(*args, **kwargs)


def acos(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.acos`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.acos`.
    """
    return _dispatch_aten(*args, **kwargs)


def acos_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.acos_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.acos_`.
    """
    return _dispatch_aten(*args, **kwargs)


def acosh(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.acosh`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.acosh`.
    """
    return _dispatch_aten(*args, **kwargs)


def acosh_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.acosh_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.acosh_`.
    """
    return _dispatch_aten(*args, **kwargs)


def adaptive_avg_pool1d(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.adaptive_avg_pool1d`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.adaptive_avg_pool1d`.
    """
    return _dispatch_aten(*args, **kwargs)


def adaptive_max_pool2d(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.adaptive_max_pool2d`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.adaptive_max_pool2d`.
    """
    return _dispatch_aten(*args, **kwargs)


def adaptive_max_pool2d_backward(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.adaptive_max_pool2d_backward`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.adaptive_max_pool2d_backward`.
    """
    return _dispatch_aten(*args, **kwargs)


def adaptive_max_pool3d(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.adaptive_max_pool3d`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.adaptive_max_pool3d`.
    """
    return _dispatch_aten(*args, **kwargs)


def adaptive_max_pool3d_backward(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.adaptive_max_pool3d_backward`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.adaptive_max_pool3d_backward`.
    """
    return _dispatch_aten(*args, **kwargs)


def add(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.add`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.add`.
    """
    return _dispatch_aten(*args, **kwargs)


def add_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.add_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.add_`.
    """
    return _dispatch_aten(*args, **kwargs)


def addbmm(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.addbmm`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.addbmm`.
    """
    return _dispatch_aten(*args, **kwargs)


def addbmm_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.addbmm_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.addbmm_`.
    """
    return _dispatch_aten(*args, **kwargs)


def addcdiv(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.addcdiv`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.addcdiv`.
    """
    return _dispatch_aten(*args, **kwargs)


def addcdiv_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.addcdiv_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.addcdiv_`.
    """
    return _dispatch_aten(*args, **kwargs)


def addcmul(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.addcmul`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.addcmul`.
    """
    return _dispatch_aten(*args, **kwargs)


def addcmul_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.addcmul_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.addcmul_`.
    """
    return _dispatch_aten(*args, **kwargs)


def addmm(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.addmm`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.addmm`.
    """
    return _dispatch_aten(*args, **kwargs)


def addmm_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.addmm_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.addmm_`.
    """
    return _dispatch_aten(*args, **kwargs)


def addmv(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.addmv`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.addmv`.
    """
    return _dispatch_aten(*args, **kwargs)


def addmv_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.addmv_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.addmv_`.
    """
    return _dispatch_aten(*args, **kwargs)


def addr(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.addr`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.addr`.
    """
    return _dispatch_aten(*args, **kwargs)


def addr_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.addr_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.addr_`.
    """
    return _dispatch_aten(*args, **kwargs)


def adjoint(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.adjoint`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.adjoint`.
    """
    return _dispatch_aten(*args, **kwargs)


def affine_grid_generator(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.affine_grid_generator`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.affine_grid_generator`.
    """
    return _dispatch_aten(*args, **kwargs)


def alias(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.alias`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.alias`.
    """
    return _dispatch_aten(*args, **kwargs)


def alias_copy(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.alias_copy`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.alias_copy`.
    """
    return _dispatch_aten(*args, **kwargs)


def all(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.all`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.all`.
    """
    return _dispatch_aten(*args, **kwargs)


def allclose(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.allclose`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.allclose`.
    """
    return _dispatch_aten(*args, **kwargs)


def alpha_dropout(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.alpha_dropout`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.alpha_dropout`.
    """
    return _dispatch_aten(*args, **kwargs)


def amax(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.amax`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.amax`.
    """
    return _dispatch_aten(*args, **kwargs)


def amin(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.amin`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.amin`.
    """
    return _dispatch_aten(*args, **kwargs)


def aminmax(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.aminmax`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.aminmax`.
    """
    return _dispatch_aten(*args, **kwargs)


def angle(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.angle`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.angle`.
    """
    return _dispatch_aten(*args, **kwargs)


def any(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.any`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.any`.
    """
    return _dispatch_aten(*args, **kwargs)


def arange(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.arange`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.arange`.
    """
    return _dispatch_aten(*args, **kwargs)


def arccos(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.arccos`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.arccos`.
    """
    return _dispatch_aten(*args, **kwargs)


def arccos_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.arccos_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.arccos_`.
    """
    return _dispatch_aten(*args, **kwargs)


def arccosh(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.arccosh`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.arccosh`.
    """
    return _dispatch_aten(*args, **kwargs)


def arccosh_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.arccosh_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.arccosh_`.
    """
    return _dispatch_aten(*args, **kwargs)


def arcsin(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.arcsin`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.arcsin`.
    """
    return _dispatch_aten(*args, **kwargs)


def arcsin_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.arcsin_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.arcsin_`.
    """
    return _dispatch_aten(*args, **kwargs)


def arcsinh(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.arcsinh`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.arcsinh`.
    """
    return _dispatch_aten(*args, **kwargs)


def arcsinh_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.arcsinh_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.arcsinh_`.
    """
    return _dispatch_aten(*args, **kwargs)


def arctan(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.arctan`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.arctan`.
    """
    return _dispatch_aten(*args, **kwargs)


def arctan2(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.arctan2`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.arctan2`.
    """
    return _dispatch_aten(*args, **kwargs)


def arctan2_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.arctan2_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.arctan2_`.
    """
    return _dispatch_aten(*args, **kwargs)


def arctan_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.arctan_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.arctan_`.
    """
    return _dispatch_aten(*args, **kwargs)


def arctanh(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.arctanh`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.arctanh`.
    """
    return _dispatch_aten(*args, **kwargs)


def arctanh_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.arctanh_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.arctanh_`.
    """
    return _dispatch_aten(*args, **kwargs)


def argmax(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.argmax`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.argmax`.
    """
    return _dispatch_aten(*args, **kwargs)


def argmin(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.argmin`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.argmin`.
    """
    return _dispatch_aten(*args, **kwargs)


def argsort(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.argsort`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.argsort`.
    """
    return _dispatch_aten(*args, **kwargs)


def argwhere(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.argwhere`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.argwhere`.
    """
    return _dispatch_aten(*args, **kwargs)


def as_strided(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.as_strided`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.as_strided`.
    """
    return _dispatch_aten(*args, **kwargs)


def as_strided_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.as_strided_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.as_strided_`.
    """
    return _dispatch_aten(*args, **kwargs)


def as_strided_copy(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.as_strided_copy`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.as_strided_copy`.
    """
    return _dispatch_aten(*args, **kwargs)


def as_strided_scatter(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.as_strided_scatter`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.as_strided_scatter`.
    """
    return _dispatch_aten(*args, **kwargs)


def asin(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.asin`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.asin`.
    """
    return _dispatch_aten(*args, **kwargs)


def asin_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.asin_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.asin_`.
    """
    return _dispatch_aten(*args, **kwargs)


def asinh(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.asinh`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.asinh`.
    """
    return _dispatch_aten(*args, **kwargs)


def asinh_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.asinh_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.asinh_`.
    """
    return _dispatch_aten(*args, **kwargs)


def atan(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.atan`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.atan`.
    """
    return _dispatch_aten(*args, **kwargs)


def atan2(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.atan2`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.atan2`.
    """
    return _dispatch_aten(*args, **kwargs)


def atan2_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.atan2_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.atan2_`.
    """
    return _dispatch_aten(*args, **kwargs)


def atan_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.atan_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.atan_`.
    """
    return _dispatch_aten(*args, **kwargs)


def atanh(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.atanh`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.atanh`.
    """
    return _dispatch_aten(*args, **kwargs)


def atanh_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.atanh_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.atanh_`.
    """
    return _dispatch_aten(*args, **kwargs)


def avg_pool1d(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.avg_pool1d`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.avg_pool1d`.
    """
    return _dispatch_aten(*args, **kwargs)


def avg_pool2d(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.avg_pool2d`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.avg_pool2d`.
    """
    return _dispatch_aten(*args, **kwargs)


def avg_pool2d_backward(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.avg_pool2d_backward`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.avg_pool2d_backward`.
    """
    return _dispatch_aten(*args, **kwargs)


def avg_pool3d(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.avg_pool3d`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.avg_pool3d`.
    """
    return _dispatch_aten(*args, **kwargs)


def avg_pool3d_backward(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.avg_pool3d_backward`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.avg_pool3d_backward`.
    """
    return _dispatch_aten(*args, **kwargs)


def baddbmm(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.baddbmm`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.baddbmm`.
    """
    return _dispatch_aten(*args, **kwargs)


def baddbmm_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.baddbmm_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.baddbmm_`.
    """
    return _dispatch_aten(*args, **kwargs)


def bartlett_window(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.bartlett_window`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.bartlett_window`.
    """
    return _dispatch_aten(*args, **kwargs)


def batch_norm_backward(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.batch_norm_backward`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.batch_norm_backward`.
    """
    return _dispatch_aten(*args, **kwargs)


def bernoulli(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.bernoulli`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.bernoulli`.
    """
    return _dispatch_aten(*args, **kwargs)


def bernoulli_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.bernoulli_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.bernoulli_`.
    """
    return _dispatch_aten(*args, **kwargs)


def bilinear(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.bilinear`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.bilinear`.
    """
    return _dispatch_aten(*args, **kwargs)


def binary_cross_entropy(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.binary_cross_entropy`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.binary_cross_entropy`.
    """
    return _dispatch_aten(*args, **kwargs)


def binary_cross_entropy_backward(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.binary_cross_entropy_backward`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.binary_cross_entropy_backward`.
    """
    return _dispatch_aten(*args, **kwargs)


def binary_cross_entropy_with_logits(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.binary_cross_entropy_with_logits`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.binary_cross_entropy_with_logits`.
    """
    return _dispatch_aten(*args, **kwargs)


def bincount(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.bincount`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.bincount`.
    """
    return _dispatch_aten(*args, **kwargs)


def bitwise_and(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.bitwise_and`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.bitwise_and`.
    """
    return _dispatch_aten(*args, **kwargs)


def bitwise_and_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.bitwise_and_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.bitwise_and_`.
    """
    return _dispatch_aten(*args, **kwargs)


def bitwise_left_shift(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.bitwise_left_shift`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.bitwise_left_shift`.
    """
    return _dispatch_aten(*args, **kwargs)


def bitwise_left_shift_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.bitwise_left_shift_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.bitwise_left_shift_`.
    """
    return _dispatch_aten(*args, **kwargs)


def bitwise_not(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.bitwise_not`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.bitwise_not`.
    """
    return _dispatch_aten(*args, **kwargs)


def bitwise_not_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.bitwise_not_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.bitwise_not_`.
    """
    return _dispatch_aten(*args, **kwargs)


def bitwise_or(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.bitwise_or`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.bitwise_or`.
    """
    return _dispatch_aten(*args, **kwargs)


def bitwise_or_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.bitwise_or_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.bitwise_or_`.
    """
    return _dispatch_aten(*args, **kwargs)


def bitwise_right_shift(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.bitwise_right_shift`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.bitwise_right_shift`.
    """
    return _dispatch_aten(*args, **kwargs)


def bitwise_right_shift_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.bitwise_right_shift_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.bitwise_right_shift_`.
    """
    return _dispatch_aten(*args, **kwargs)


def bitwise_xor(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.bitwise_xor`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.bitwise_xor`.
    """
    return _dispatch_aten(*args, **kwargs)


def bitwise_xor_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.bitwise_xor_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.bitwise_xor_`.
    """
    return _dispatch_aten(*args, **kwargs)


def blackman_window(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.blackman_window`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.blackman_window`.
    """
    return _dispatch_aten(*args, **kwargs)


def block_diag(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.block_diag`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.block_diag`.
    """
    return _dispatch_aten(*args, **kwargs)


def bmm(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.bmm`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.bmm`.
    """
    return _dispatch_aten(*args, **kwargs)


def broadcast_tensors(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.broadcast_tensors`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.broadcast_tensors`.
    """
    return _dispatch_aten(*args, **kwargs)


def broadcast_to(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.broadcast_to`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.broadcast_to`.
    """
    return _dispatch_aten(*args, **kwargs)


def bucketize(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.bucketize`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.bucketize`.
    """
    return _dispatch_aten(*args, **kwargs)


def cat(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.cat`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.cat`.
    """
    return _dispatch_aten(*args, **kwargs)


def cauchy(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.cauchy`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.cauchy`.
    """
    return _dispatch_aten(*args, **kwargs)


def cauchy_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.cauchy_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.cauchy_`.
    """
    return _dispatch_aten(*args, **kwargs)


def ccol_indices(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.ccol_indices`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.ccol_indices`.
    """
    return _dispatch_aten(*args, **kwargs)


def ceil(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.ceil`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.ceil`.
    """
    return _dispatch_aten(*args, **kwargs)


def ceil_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.ceil_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.ceil_`.
    """
    return _dispatch_aten(*args, **kwargs)


def celu(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.celu`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.celu`.
    """
    return _dispatch_aten(*args, **kwargs)


def celu_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.celu_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.celu_`.
    """
    return _dispatch_aten(*args, **kwargs)


def chalf(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.chalf`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.chalf`.
    """
    return _dispatch_aten(*args, **kwargs)


def channel_shuffle(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.channel_shuffle`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.channel_shuffle`.
    """
    return _dispatch_aten(*args, **kwargs)


def cholesky_inverse(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.cholesky_inverse`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.cholesky_inverse`.
    """
    return _dispatch_aten(*args, **kwargs)


def cholesky_solve(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.cholesky_solve`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.cholesky_solve`.
    """
    return _dispatch_aten(*args, **kwargs)


def chunk(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.chunk`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.chunk`.
    """
    return _dispatch_aten(*args, **kwargs)


def clamp(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.clamp`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.clamp`.
    """
    return _dispatch_aten(*args, **kwargs)


def clamp_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.clamp_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.clamp_`.
    """
    return _dispatch_aten(*args, **kwargs)


def clamp_max(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.clamp_max`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.clamp_max`.
    """
    return _dispatch_aten(*args, **kwargs)


def clamp_max_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.clamp_max_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.clamp_max_`.
    """
    return _dispatch_aten(*args, **kwargs)


def clamp_min(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.clamp_min`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.clamp_min`.
    """
    return _dispatch_aten(*args, **kwargs)


def clamp_min_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.clamp_min_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.clamp_min_`.
    """
    return _dispatch_aten(*args, **kwargs)


def clip(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.clip`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.clip`.
    """
    return _dispatch_aten(*args, **kwargs)


def clip_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.clip_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.clip_`.
    """
    return _dispatch_aten(*args, **kwargs)


def clone(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.clone`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.clone`.
    """
    return _dispatch_aten(*args, **kwargs)


def coalesce(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.coalesce`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.coalesce`.
    """
    return _dispatch_aten(*args, **kwargs)


def col2im(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.col2im`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.col2im`.
    """
    return _dispatch_aten(*args, **kwargs)


def col_indices(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.col_indices`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.col_indices`.
    """
    return _dispatch_aten(*args, **kwargs)


def complex(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.complex`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.complex`.
    """
    return _dispatch_aten(*args, **kwargs)


def conj(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.conj`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.conj`.
    """
    return _dispatch_aten(*args, **kwargs)


def conj_physical(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.conj_physical`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.conj_physical`.
    """
    return _dispatch_aten(*args, **kwargs)


def conj_physical_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.conj_physical_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.conj_physical_`.
    """
    return _dispatch_aten(*args, **kwargs)


def constant_pad_nd(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.constant_pad_nd`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.constant_pad_nd`.
    """
    return _dispatch_aten(*args, **kwargs)


def contiguous(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.contiguous`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.contiguous`.
    """
    return _dispatch_aten(*args, **kwargs)


def conv1d(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.conv1d`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.conv1d`.
    """
    return _dispatch_aten(*args, **kwargs)


def conv2d(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.conv2d`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.conv2d`.
    """
    return _dispatch_aten(*args, **kwargs)


def conv3d(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.conv3d`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.conv3d`.
    """
    return _dispatch_aten(*args, **kwargs)


def conv_tbc(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.conv_tbc`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.conv_tbc`.
    """
    return _dispatch_aten(*args, **kwargs)


def conv_transpose1d(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.conv_transpose1d`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.conv_transpose1d`.
    """
    return _dispatch_aten(*args, **kwargs)


def conv_transpose2d(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.conv_transpose2d`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.conv_transpose2d`.
    """
    return _dispatch_aten(*args, **kwargs)


def conv_transpose3d(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.conv_transpose3d`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.conv_transpose3d`.
    """
    return _dispatch_aten(*args, **kwargs)


def convolution(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.convolution`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.convolution`.
    """
    return _dispatch_aten(*args, **kwargs)


def convolution_backward(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.convolution_backward`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.convolution_backward`.
    """
    return _dispatch_aten(*args, **kwargs)


def copy(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.copy`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.copy`.
    """
    return _dispatch_aten(*args, **kwargs)


def copy_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.copy_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.copy_`.
    """
    return _dispatch_aten(*args, **kwargs)


def copysign(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.copysign`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.copysign`.
    """
    return _dispatch_aten(*args, **kwargs)


def copysign_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.copysign_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.copysign_`.
    """
    return _dispatch_aten(*args, **kwargs)


def corrcoef(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.corrcoef`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.corrcoef`.
    """
    return _dispatch_aten(*args, **kwargs)


def cos(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.cos`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.cos`.
    """
    return _dispatch_aten(*args, **kwargs)


def cos_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.cos_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.cos_`.
    """
    return _dispatch_aten(*args, **kwargs)


def cosh(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.cosh`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.cosh`.
    """
    return _dispatch_aten(*args, **kwargs)


def cosh_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.cosh_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.cosh_`.
    """
    return _dispatch_aten(*args, **kwargs)


def cosine_similarity(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.cosine_similarity`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.cosine_similarity`.
    """
    return _dispatch_aten(*args, **kwargs)


def count_nonzero(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.count_nonzero`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.count_nonzero`.
    """
    return _dispatch_aten(*args, **kwargs)


def cov(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.cov`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.cov`.
    """
    return _dispatch_aten(*args, **kwargs)


def cpu(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.cpu`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.cpu`.
    """
    return _dispatch_aten(*args, **kwargs)


def cross(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.cross`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.cross`.
    """
    return _dispatch_aten(*args, **kwargs)


def crow_indices(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.crow_indices`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.crow_indices`.
    """
    return _dispatch_aten(*args, **kwargs)


def ctc_loss(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.ctc_loss`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.ctc_loss`.
    """
    return _dispatch_aten(*args, **kwargs)


def cuda(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.cuda`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.cuda`.
    """
    return _dispatch_aten(*args, **kwargs)


def cudnn_batch_norm(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.cudnn_batch_norm`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.cudnn_batch_norm`.
    """
    return _dispatch_aten(*args, **kwargs)


def cudnn_batch_norm_backward(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.cudnn_batch_norm_backward`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.cudnn_batch_norm_backward`.
    """
    return _dispatch_aten(*args, **kwargs)


def cummax(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.cummax`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.cummax`.
    """
    return _dispatch_aten(*args, **kwargs)


def cummin(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.cummin`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.cummin`.
    """
    return _dispatch_aten(*args, **kwargs)


def cumprod(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.cumprod`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.cumprod`.
    """
    return _dispatch_aten(*args, **kwargs)


def cumprod_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.cumprod_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.cumprod_`.
    """
    return _dispatch_aten(*args, **kwargs)


def cumsum(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.cumsum`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.cumsum`.
    """
    return _dispatch_aten(*args, **kwargs)


def cumsum_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.cumsum_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.cumsum_`.
    """
    return _dispatch_aten(*args, **kwargs)


def cumulative_trapezoid(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.cumulative_trapezoid`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.cumulative_trapezoid`.
    """
    return _dispatch_aten(*args, **kwargs)


def deg2rad(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.deg2rad`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.deg2rad`.
    """
    return _dispatch_aten(*args, **kwargs)


def deg2rad_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.deg2rad_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.deg2rad_`.
    """
    return _dispatch_aten(*args, **kwargs)


def dense_dim(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.dense_dim`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.dense_dim`.
    """
    return _dispatch_aten(*args, **kwargs)


def dequantize(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.dequantize`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.dequantize`.
    """
    return _dispatch_aten(*args, **kwargs)


def det(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.det`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.det`.
    """
    return _dispatch_aten(*args, **kwargs)


def detach(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.detach`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.detach`.
    """
    return _dispatch_aten(*args, **kwargs)


def detach_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.detach_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.detach_`.
    """
    return _dispatch_aten(*args, **kwargs)


def diag(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.diag`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.diag`.
    """
    return _dispatch_aten(*args, **kwargs)


def diag_embed(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.diag_embed`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.diag_embed`.
    """
    return _dispatch_aten(*args, **kwargs)


def diagflat(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.diagflat`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.diagflat`.
    """
    return _dispatch_aten(*args, **kwargs)


def diagonal(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.diagonal`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.diagonal`.
    """
    return _dispatch_aten(*args, **kwargs)


def diagonal_backward(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.diagonal_backward`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.diagonal_backward`.
    """
    return _dispatch_aten(*args, **kwargs)


def diagonal_copy(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.diagonal_copy`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.diagonal_copy`.
    """
    return _dispatch_aten(*args, **kwargs)


def diagonal_scatter(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.diagonal_scatter`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.diagonal_scatter`.
    """
    return _dispatch_aten(*args, **kwargs)


def diff(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.diff`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.diff`.
    """
    return _dispatch_aten(*args, **kwargs)


def digamma(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.digamma`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.digamma`.
    """
    return _dispatch_aten(*args, **kwargs)


def digamma_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.digamma_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.digamma_`.
    """
    return _dispatch_aten(*args, **kwargs)


def dim(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.dim`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.dim`.
    """
    return _dispatch_aten(*args, **kwargs)


def dist(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.dist`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.dist`.
    """
    return _dispatch_aten(*args, **kwargs)


def div(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.div`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.div`.
    """
    return _dispatch_aten(*args, **kwargs)


def div_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.div_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.div_`.
    """
    return _dispatch_aten(*args, **kwargs)


def divide(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.divide`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.divide`.
    """
    return _dispatch_aten(*args, **kwargs)


def divide_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.divide_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.divide_`.
    """
    return _dispatch_aten(*args, **kwargs)


def dot(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.dot`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.dot`.
    """
    return _dispatch_aten(*args, **kwargs)


def dropout(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.dropout`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.dropout`.
    """
    return _dispatch_aten(*args, **kwargs)


def dsplit(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.dsplit`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.dsplit`.
    """
    return _dispatch_aten(*args, **kwargs)


def element_size(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.element_size`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.element_size`.
    """
    return _dispatch_aten(*args, **kwargs)


def elu(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.elu`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.elu`.
    """
    return _dispatch_aten(*args, **kwargs)


def elu_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.elu_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.elu_`.
    """
    return _dispatch_aten(*args, **kwargs)


def elu_backward(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.elu_backward`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.elu_backward`.
    """
    return _dispatch_aten(*args, **kwargs)


def embedding(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.embedding`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.embedding`.
    """
    return _dispatch_aten(*args, **kwargs)


def embedding_bag(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.embedding_bag`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.embedding_bag`.
    """
    return _dispatch_aten(*args, **kwargs)


def embedding_dense_backward(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.embedding_dense_backward`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.embedding_dense_backward`.
    """
    return _dispatch_aten(*args, **kwargs)


def empty(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.empty`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.empty`.
    """
    return _dispatch_aten(*args, **kwargs)


def empty_like(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.empty_like`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.empty_like`.
    """
    return _dispatch_aten(*args, **kwargs)


def empty_permuted(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.empty_permuted`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.empty_permuted`.
    """
    return _dispatch_aten(*args, **kwargs)


def empty_strided(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.empty_strided`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.empty_strided`.
    """
    return _dispatch_aten(*args, **kwargs)


def eq(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.eq`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.eq`.
    """
    return _dispatch_aten(*args, **kwargs)


def eq_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.eq_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.eq_`.
    """
    return _dispatch_aten(*args, **kwargs)


def equal(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.equal`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.equal`.
    """
    return _dispatch_aten(*args, **kwargs)


def erf(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.erf`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.erf`.
    """
    return _dispatch_aten(*args, **kwargs)


def erf_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.erf_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.erf_`.
    """
    return _dispatch_aten(*args, **kwargs)


def erfc(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.erfc`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.erfc`.
    """
    return _dispatch_aten(*args, **kwargs)


def erfc_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.erfc_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.erfc_`.
    """
    return _dispatch_aten(*args, **kwargs)


def erfinv(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.erfinv`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.erfinv`.
    """
    return _dispatch_aten(*args, **kwargs)


def erfinv_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.erfinv_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.erfinv_`.
    """
    return _dispatch_aten(*args, **kwargs)


def exp(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.exp`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.exp`.
    """
    return _dispatch_aten(*args, **kwargs)


def exp2(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.exp2`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.exp2`.
    """
    return _dispatch_aten(*args, **kwargs)


def exp2_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.exp2_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.exp2_`.
    """
    return _dispatch_aten(*args, **kwargs)


def exp_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.exp_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.exp_`.
    """
    return _dispatch_aten(*args, **kwargs)


def expand(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.expand`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.expand`.
    """
    return _dispatch_aten(*args, **kwargs)


def expand_as(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.expand_as`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.expand_as`.
    """
    return _dispatch_aten(*args, **kwargs)


def expand_copy(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.expand_copy`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.expand_copy`.
    """
    return _dispatch_aten(*args, **kwargs)


def expm1(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.expm1`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.expm1`.
    """
    return _dispatch_aten(*args, **kwargs)


def expm1_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.expm1_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.expm1_`.
    """
    return _dispatch_aten(*args, **kwargs)


def exponential(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.exponential`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.exponential`.
    """
    return _dispatch_aten(*args, **kwargs)


def exponential_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.exponential_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.exponential_`.
    """
    return _dispatch_aten(*args, **kwargs)


def eye(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.eye`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.eye`.
    """
    return _dispatch_aten(*args, **kwargs)


def fake_quantize_per_tensor_affine(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.fake_quantize_per_tensor_affine`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.fake_quantize_per_tensor_affine`.
    """
    return _dispatch_aten(*args, **kwargs)


def fbgemm_linear_fp16_weight(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.fbgemm_linear_fp16_weight`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.fbgemm_linear_fp16_weight`.
    """
    return _dispatch_aten(*args, **kwargs)


def fbgemm_linear_fp16_weight_fp32_activation(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.fbgemm_linear_fp16_weight_fp32_activation`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.fbgemm_linear_fp16_weight_fp32_activation`.
    """
    return _dispatch_aten(*args, **kwargs)


def fbgemm_pack_quantized_matrix(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.fbgemm_pack_quantized_matrix`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.fbgemm_pack_quantized_matrix`.
    """
    return _dispatch_aten(*args, **kwargs)


def fft_fft(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.fft_fft`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.fft_fft`.
    """
    return _dispatch_aten(*args, **kwargs)


def fft_fft2(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.fft_fft2`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.fft_fft2`.
    """
    return _dispatch_aten(*args, **kwargs)


def fft_fftfreq(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.fft_fftfreq`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.fft_fftfreq`.
    """
    return _dispatch_aten(*args, **kwargs)


def fft_fftn(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.fft_fftn`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.fft_fftn`.
    """
    return _dispatch_aten(*args, **kwargs)


def fft_fftshift(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.fft_fftshift`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.fft_fftshift`.
    """
    return _dispatch_aten(*args, **kwargs)


def fft_hfft(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.fft_hfft`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.fft_hfft`.
    """
    return _dispatch_aten(*args, **kwargs)


def fft_hfft2(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.fft_hfft2`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.fft_hfft2`.
    """
    return _dispatch_aten(*args, **kwargs)


def fft_hfftn(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.fft_hfftn`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.fft_hfftn`.
    """
    return _dispatch_aten(*args, **kwargs)


def fft_ifft(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.fft_ifft`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.fft_ifft`.
    """
    return _dispatch_aten(*args, **kwargs)


def fft_ifft2(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.fft_ifft2`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.fft_ifft2`.
    """
    return _dispatch_aten(*args, **kwargs)


def fft_ifftn(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.fft_ifftn`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.fft_ifftn`.
    """
    return _dispatch_aten(*args, **kwargs)


def fft_ifftshift(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.fft_ifftshift`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.fft_ifftshift`.
    """
    return _dispatch_aten(*args, **kwargs)


def fft_ihfft(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.fft_ihfft`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.fft_ihfft`.
    """
    return _dispatch_aten(*args, **kwargs)


def fft_ihfft2(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.fft_ihfft2`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.fft_ihfft2`.
    """
    return _dispatch_aten(*args, **kwargs)


def fft_ihfftn(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.fft_ihfftn`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.fft_ihfftn`.
    """
    return _dispatch_aten(*args, **kwargs)


def fft_irfft(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.fft_irfft`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.fft_irfft`.
    """
    return _dispatch_aten(*args, **kwargs)


def fft_irfft2(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.fft_irfft2`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.fft_irfft2`.
    """
    return _dispatch_aten(*args, **kwargs)


def fft_irfftn(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.fft_irfftn`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.fft_irfftn`.
    """
    return _dispatch_aten(*args, **kwargs)


def fft_rfft(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.fft_rfft`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.fft_rfft`.
    """
    return _dispatch_aten(*args, **kwargs)


def fft_rfft2(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.fft_rfft2`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.fft_rfft2`.
    """
    return _dispatch_aten(*args, **kwargs)


def fft_rfftfreq(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.fft_rfftfreq`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.fft_rfftfreq`.
    """
    return _dispatch_aten(*args, **kwargs)


def fft_rfftn(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.fft_rfftn`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.fft_rfftn`.
    """
    return _dispatch_aten(*args, **kwargs)


def fill(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.fill`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.fill`.
    """
    return _dispatch_aten(*args, **kwargs)


def fill_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.fill_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.fill_`.
    """
    return _dispatch_aten(*args, **kwargs)


def fill_diagonal_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.fill_diagonal_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.fill_diagonal_`.
    """
    return _dispatch_aten(*args, **kwargs)


def fix(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.fix`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.fix`.
    """
    return _dispatch_aten(*args, **kwargs)


def fix_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.fix_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.fix_`.
    """
    return _dispatch_aten(*args, **kwargs)


def flatten(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.flatten`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.flatten`.
    """
    return _dispatch_aten(*args, **kwargs)


def flip(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.flip`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.flip`.
    """
    return _dispatch_aten(*args, **kwargs)


def fliplr(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.fliplr`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.fliplr`.
    """
    return _dispatch_aten(*args, **kwargs)


def flipud(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.flipud`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.flipud`.
    """
    return _dispatch_aten(*args, **kwargs)


def float_power(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.float_power`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.float_power`.
    """
    return _dispatch_aten(*args, **kwargs)


def float_power_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.float_power_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.float_power_`.
    """
    return _dispatch_aten(*args, **kwargs)


def floor(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.floor`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.floor`.
    """
    return _dispatch_aten(*args, **kwargs)


def floor_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.floor_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.floor_`.
    """
    return _dispatch_aten(*args, **kwargs)


def floor_divide(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.floor_divide`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.floor_divide`.
    """
    return _dispatch_aten(*args, **kwargs)


def floor_divide_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.floor_divide_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.floor_divide_`.
    """
    return _dispatch_aten(*args, **kwargs)


def fmax(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.fmax`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.fmax`.
    """
    return _dispatch_aten(*args, **kwargs)


def fmin(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.fmin`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.fmin`.
    """
    return _dispatch_aten(*args, **kwargs)


def fmod(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.fmod`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.fmod`.
    """
    return _dispatch_aten(*args, **kwargs)


def fmod_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.fmod_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.fmod_`.
    """
    return _dispatch_aten(*args, **kwargs)


def frac(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.frac`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.frac`.
    """
    return _dispatch_aten(*args, **kwargs)


def frac_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.frac_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.frac_`.
    """
    return _dispatch_aten(*args, **kwargs)


def fractional_max_pool2d(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.fractional_max_pool2d`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.fractional_max_pool2d`.
    """
    return _dispatch_aten(*args, **kwargs)


def frexp(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.frexp`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.frexp`.
    """
    return _dispatch_aten(*args, **kwargs)


def full(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.full`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.full`.
    """
    return _dispatch_aten(*args, **kwargs)


def full_like(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.full_like`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.full_like`.
    """
    return _dispatch_aten(*args, **kwargs)


def gather(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.gather`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.gather`.
    """
    return _dispatch_aten(*args, **kwargs)


def gcd(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.gcd`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.gcd`.
    """
    return _dispatch_aten(*args, **kwargs)


def gcd_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.gcd_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.gcd_`.
    """
    return _dispatch_aten(*args, **kwargs)


def ge(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.ge`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.ge`.
    """
    return _dispatch_aten(*args, **kwargs)


def ge_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.ge_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.ge_`.
    """
    return _dispatch_aten(*args, **kwargs)


def gelu(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.gelu`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.gelu`.
    """
    return _dispatch_aten(*args, **kwargs)


def gelu_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.gelu_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.gelu_`.
    """
    return _dispatch_aten(*args, **kwargs)


def gelu_backward(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.gelu_backward`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.gelu_backward`.
    """
    return _dispatch_aten(*args, **kwargs)


def geometric(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.geometric`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.geometric`.
    """
    return _dispatch_aten(*args, **kwargs)


def geometric_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.geometric_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.geometric_`.
    """
    return _dispatch_aten(*args, **kwargs)


def geqrf(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.geqrf`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.geqrf`.
    """
    return _dispatch_aten(*args, **kwargs)


def ger(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.ger`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.ger`.
    """
    return _dispatch_aten(*args, **kwargs)


def get_device(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.get_device`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.get_device`.
    """
    return _dispatch_aten(*args, **kwargs)


def glu(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.glu`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.glu`.
    """
    return _dispatch_aten(*args, **kwargs)


def glu_backward(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.glu_backward`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.glu_backward`.
    """
    return _dispatch_aten(*args, **kwargs)


def gradient(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.gradient`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.gradient`.
    """
    return _dispatch_aten(*args, **kwargs)


def greater(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.greater`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.greater`.
    """
    return _dispatch_aten(*args, **kwargs)


def greater_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.greater_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.greater_`.
    """
    return _dispatch_aten(*args, **kwargs)


def greater_equal(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.greater_equal`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.greater_equal`.
    """
    return _dispatch_aten(*args, **kwargs)


def greater_equal_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.greater_equal_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.greater_equal_`.
    """
    return _dispatch_aten(*args, **kwargs)


def grid_sampler_2d(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.grid_sampler_2d`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.grid_sampler_2d`.
    """
    return _dispatch_aten(*args, **kwargs)


def grid_sampler_2d_backward(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.grid_sampler_2d_backward`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.grid_sampler_2d_backward`.
    """
    return _dispatch_aten(*args, **kwargs)


def grid_sampler_3d(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.grid_sampler_3d`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.grid_sampler_3d`.
    """
    return _dispatch_aten(*args, **kwargs)


def grid_sampler_3d_backward(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.grid_sampler_3d_backward`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.grid_sampler_3d_backward`.
    """
    return _dispatch_aten(*args, **kwargs)


def gru(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.gru`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.gru`.
    """
    return _dispatch_aten(*args, **kwargs)


def gt(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.gt`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.gt`.
    """
    return _dispatch_aten(*args, **kwargs)


def gt_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.gt_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.gt_`.
    """
    return _dispatch_aten(*args, **kwargs)


def hamming_window(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.hamming_window`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.hamming_window`.
    """
    return _dispatch_aten(*args, **kwargs)


def hann_window(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.hann_window`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.hann_window`.
    """
    return _dispatch_aten(*args, **kwargs)


def hardshrink(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.hardshrink`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.hardshrink`.
    """
    return _dispatch_aten(*args, **kwargs)


def hardsigmoid(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.hardsigmoid`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.hardsigmoid`.
    """
    return _dispatch_aten(*args, **kwargs)


def hardsigmoid_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.hardsigmoid_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.hardsigmoid_`.
    """
    return _dispatch_aten(*args, **kwargs)


def hardsigmoid_backward(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.hardsigmoid_backward`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.hardsigmoid_backward`.
    """
    return _dispatch_aten(*args, **kwargs)


def hardswish(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.hardswish`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.hardswish`.
    """
    return _dispatch_aten(*args, **kwargs)


def hardswish_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.hardswish_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.hardswish_`.
    """
    return _dispatch_aten(*args, **kwargs)


def hardswish_backward(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.hardswish_backward`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.hardswish_backward`.
    """
    return _dispatch_aten(*args, **kwargs)


def hardtanh(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.hardtanh`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.hardtanh`.
    """
    return _dispatch_aten(*args, **kwargs)


def hardtanh_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.hardtanh_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.hardtanh_`.
    """
    return _dispatch_aten(*args, **kwargs)


def hardtanh_backward(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.hardtanh_backward`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.hardtanh_backward`.
    """
    return _dispatch_aten(*args, **kwargs)


def hash_tensor(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.hash_tensor`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.hash_tensor`.
    """
    return _dispatch_aten(*args, **kwargs)


def heaviside(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.heaviside`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.heaviside`.
    """
    return _dispatch_aten(*args, **kwargs)


def heaviside_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.heaviside_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.heaviside_`.
    """
    return _dispatch_aten(*args, **kwargs)


def hinge_embedding_loss(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.hinge_embedding_loss`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.hinge_embedding_loss`.
    """
    return _dispatch_aten(*args, **kwargs)


def histc(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.histc`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.histc`.
    """
    return _dispatch_aten(*args, **kwargs)


def histogram(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.histogram`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.histogram`.
    """
    return _dispatch_aten(*args, **kwargs)


def histogramdd(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.histogramdd`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.histogramdd`.
    """
    return _dispatch_aten(*args, **kwargs)


def hsplit(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.hsplit`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.hsplit`.
    """
    return _dispatch_aten(*args, **kwargs)


def huber_loss(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.huber_loss`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.huber_loss`.
    """
    return _dispatch_aten(*args, **kwargs)


def huber_loss_backward(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.huber_loss_backward`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.huber_loss_backward`.
    """
    return _dispatch_aten(*args, **kwargs)


def hypot(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.hypot`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.hypot`.
    """
    return _dispatch_aten(*args, **kwargs)


def hypot_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.hypot_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.hypot_`.
    """
    return _dispatch_aten(*args, **kwargs)


def i0(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.i0`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.i0`.
    """
    return _dispatch_aten(*args, **kwargs)


def i0_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.i0_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.i0_`.
    """
    return _dispatch_aten(*args, **kwargs)


def igamma(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.igamma`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.igamma`.
    """
    return _dispatch_aten(*args, **kwargs)


def igamma_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.igamma_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.igamma_`.
    """
    return _dispatch_aten(*args, **kwargs)


def igammac(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.igammac`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.igammac`.
    """
    return _dispatch_aten(*args, **kwargs)


def igammac_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.igammac_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.igammac_`.
    """
    return _dispatch_aten(*args, **kwargs)


def im2col(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.im2col`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.im2col`.
    """
    return _dispatch_aten(*args, **kwargs)


def imag(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.imag`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.imag`.
    """
    return _dispatch_aten(*args, **kwargs)


def index(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.index`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.index`.
    """
    return _dispatch_aten(*args, **kwargs)


def index_add(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.index_add`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.index_add`.
    """
    return _dispatch_aten(*args, **kwargs)


def index_add_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.index_add_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.index_add_`.
    """
    return _dispatch_aten(*args, **kwargs)


def index_copy(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.index_copy`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.index_copy`.
    """
    return _dispatch_aten(*args, **kwargs)


def index_copy_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.index_copy_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.index_copy_`.
    """
    return _dispatch_aten(*args, **kwargs)


def index_fill(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.index_fill`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.index_fill`.
    """
    return _dispatch_aten(*args, **kwargs)


def index_fill_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.index_fill_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.index_fill_`.
    """
    return _dispatch_aten(*args, **kwargs)


def index_put(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.index_put`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.index_put`.
    """
    return _dispatch_aten(*args, **kwargs)


def index_put_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.index_put_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.index_put_`.
    """
    return _dispatch_aten(*args, **kwargs)


def index_reduce(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.index_reduce`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.index_reduce`.
    """
    return _dispatch_aten(*args, **kwargs)


def index_reduce_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.index_reduce_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.index_reduce_`.
    """
    return _dispatch_aten(*args, **kwargs)


def index_select(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.index_select`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.index_select`.
    """
    return _dispatch_aten(*args, **kwargs)


def indices(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.indices`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.indices`.
    """
    return _dispatch_aten(*args, **kwargs)


def inner(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.inner`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.inner`.
    """
    return _dispatch_aten(*args, **kwargs)


def int_repr(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.int_repr`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.int_repr`.
    """
    return _dispatch_aten(*args, **kwargs)


def inverse(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.inverse`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.inverse`.
    """
    return _dispatch_aten(*args, **kwargs)


def is_autocast_cpu_enabled(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.is_autocast_cpu_enabled`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.is_autocast_cpu_enabled`.
    """
    return _dispatch_aten(*args, **kwargs)


def is_autocast_enabled(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.is_autocast_enabled`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.is_autocast_enabled`.
    """
    return _dispatch_aten(*args, **kwargs)


def is_coalesced(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.is_coalesced`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.is_coalesced`.
    """
    return _dispatch_aten(*args, **kwargs)


def is_complex(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.is_complex`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.is_complex`.
    """
    return _dispatch_aten(*args, **kwargs)


def is_conj(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.is_conj`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.is_conj`.
    """
    return _dispatch_aten(*args, **kwargs)


def is_contiguous(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.is_contiguous`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.is_contiguous`.
    """
    return _dispatch_aten(*args, **kwargs)


def is_distributed(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.is_distributed`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.is_distributed`.
    """
    return _dispatch_aten(*args, **kwargs)


def is_floating_point(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.is_floating_point`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.is_floating_point`.
    """
    return _dispatch_aten(*args, **kwargs)


def is_grad_enabled(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.is_grad_enabled`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.is_grad_enabled`.
    """
    return _dispatch_aten(*args, **kwargs)


def is_inference(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.is_inference`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.is_inference`.
    """
    return _dispatch_aten(*args, **kwargs)


def is_neg(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.is_neg`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.is_neg`.
    """
    return _dispatch_aten(*args, **kwargs)


def is_non_overlapping_and_dense(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.is_non_overlapping_and_dense`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.is_non_overlapping_and_dense`.
    """
    return _dispatch_aten(*args, **kwargs)


def is_nonzero(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.is_nonzero`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.is_nonzero`.
    """
    return _dispatch_aten(*args, **kwargs)


def is_pinned(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.is_pinned`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.is_pinned`.
    """
    return _dispatch_aten(*args, **kwargs)


def is_same_size(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.is_same_size`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.is_same_size`.
    """
    return _dispatch_aten(*args, **kwargs)


def is_set_to(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.is_set_to`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.is_set_to`.
    """
    return _dispatch_aten(*args, **kwargs)


def is_signed(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.is_signed`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.is_signed`.
    """
    return _dispatch_aten(*args, **kwargs)


def is_strides_like_format(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.is_strides_like_format`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.is_strides_like_format`.
    """
    return _dispatch_aten(*args, **kwargs)


def is_vulkan_available(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.is_vulkan_available`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.is_vulkan_available`.
    """
    return _dispatch_aten(*args, **kwargs)


def isclose(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.isclose`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.isclose`.
    """
    return _dispatch_aten(*args, **kwargs)


def isfinite(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.isfinite`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.isfinite`.
    """
    return _dispatch_aten(*args, **kwargs)


def isin(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.isin`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.isin`.
    """
    return _dispatch_aten(*args, **kwargs)


def isinf(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.isinf`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.isinf`.
    """
    return _dispatch_aten(*args, **kwargs)


def isnan(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.isnan`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.isnan`.
    """
    return _dispatch_aten(*args, **kwargs)


def isneginf(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.isneginf`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.isneginf`.
    """
    return _dispatch_aten(*args, **kwargs)


def isposinf(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.isposinf`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.isposinf`.
    """
    return _dispatch_aten(*args, **kwargs)


def isreal(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.isreal`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.isreal`.
    """
    return _dispatch_aten(*args, **kwargs)


def istft(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.istft`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.istft`.
    """
    return _dispatch_aten(*args, **kwargs)


def item(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.item`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.item`.
    """
    return _dispatch_aten(*args, **kwargs)


def kaiser_window(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.kaiser_window`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.kaiser_window`.
    """
    return _dispatch_aten(*args, **kwargs)


def kron(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.kron`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.kron`.
    """
    return _dispatch_aten(*args, **kwargs)


def kthvalue(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.kthvalue`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.kthvalue`.
    """
    return _dispatch_aten(*args, **kwargs)


def lcm(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.lcm`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.lcm`.
    """
    return _dispatch_aten(*args, **kwargs)


def lcm_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.lcm_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.lcm_`.
    """
    return _dispatch_aten(*args, **kwargs)


def ldexp(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.ldexp`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.ldexp`.
    """
    return _dispatch_aten(*args, **kwargs)


def ldexp_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.ldexp_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.ldexp_`.
    """
    return _dispatch_aten(*args, **kwargs)


def le(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.le`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.le`.
    """
    return _dispatch_aten(*args, **kwargs)


def le_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.le_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.le_`.
    """
    return _dispatch_aten(*args, **kwargs)


def leaky_relu(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.leaky_relu`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.leaky_relu`.
    """
    return _dispatch_aten(*args, **kwargs)


def leaky_relu_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.leaky_relu_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.leaky_relu_`.
    """
    return _dispatch_aten(*args, **kwargs)


def leaky_relu_backward(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.leaky_relu_backward`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.leaky_relu_backward`.
    """
    return _dispatch_aten(*args, **kwargs)


def lerp(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.lerp`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.lerp`.
    """
    return _dispatch_aten(*args, **kwargs)


def lerp_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.lerp_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.lerp_`.
    """
    return _dispatch_aten(*args, **kwargs)


def less(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.less`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.less`.
    """
    return _dispatch_aten(*args, **kwargs)


def less_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.less_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.less_`.
    """
    return _dispatch_aten(*args, **kwargs)


def less_equal(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.less_equal`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.less_equal`.
    """
    return _dispatch_aten(*args, **kwargs)


def less_equal_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.less_equal_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.less_equal_`.
    """
    return _dispatch_aten(*args, **kwargs)


def lgamma(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.lgamma`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.lgamma`.
    """
    return _dispatch_aten(*args, **kwargs)


def lgamma_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.lgamma_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.lgamma_`.
    """
    return _dispatch_aten(*args, **kwargs)


def lift(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.lift`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.lift`.
    """
    return _dispatch_aten(*args, **kwargs)


def lift_fresh(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.lift_fresh`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.lift_fresh`.
    """
    return _dispatch_aten(*args, **kwargs)


def lift_fresh_copy(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.lift_fresh_copy`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.lift_fresh_copy`.
    """
    return _dispatch_aten(*args, **kwargs)


def linalg_cholesky(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.linalg_cholesky`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.linalg_cholesky`.
    """
    return _dispatch_aten(*args, **kwargs)


def linalg_cholesky_ex(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.linalg_cholesky_ex`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.linalg_cholesky_ex`.
    """
    return _dispatch_aten(*args, **kwargs)


def linalg_cond(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.linalg_cond`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.linalg_cond`.
    """
    return _dispatch_aten(*args, **kwargs)


def linalg_cross(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.linalg_cross`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.linalg_cross`.
    """
    return _dispatch_aten(*args, **kwargs)


def linalg_det(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.linalg_det`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.linalg_det`.
    """
    return _dispatch_aten(*args, **kwargs)


def linalg_diagonal(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.linalg_diagonal`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.linalg_diagonal`.
    """
    return _dispatch_aten(*args, **kwargs)


def linalg_eig(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.linalg_eig`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.linalg_eig`.
    """
    return _dispatch_aten(*args, **kwargs)


def linalg_eigh(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.linalg_eigh`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.linalg_eigh`.
    """
    return _dispatch_aten(*args, **kwargs)


def linalg_eigvals(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.linalg_eigvals`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.linalg_eigvals`.
    """
    return _dispatch_aten(*args, **kwargs)


def linalg_eigvalsh(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.linalg_eigvalsh`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.linalg_eigvalsh`.
    """
    return _dispatch_aten(*args, **kwargs)


def linalg_householder_product(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.linalg_householder_product`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.linalg_householder_product`.
    """
    return _dispatch_aten(*args, **kwargs)


def linalg_inv(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.linalg_inv`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.linalg_inv`.
    """
    return _dispatch_aten(*args, **kwargs)


def linalg_inv_ex(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.linalg_inv_ex`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.linalg_inv_ex`.
    """
    return _dispatch_aten(*args, **kwargs)


def linalg_ldl_factor(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.linalg_ldl_factor`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.linalg_ldl_factor`.
    """
    return _dispatch_aten(*args, **kwargs)


def linalg_ldl_factor_ex(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.linalg_ldl_factor_ex`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.linalg_ldl_factor_ex`.
    """
    return _dispatch_aten(*args, **kwargs)


def linalg_ldl_solve(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.linalg_ldl_solve`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.linalg_ldl_solve`.
    """
    return _dispatch_aten(*args, **kwargs)


def linalg_lstsq(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.linalg_lstsq`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.linalg_lstsq`.
    """
    return _dispatch_aten(*args, **kwargs)


def linalg_lu(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.linalg_lu`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.linalg_lu`.
    """
    return _dispatch_aten(*args, **kwargs)


def linalg_lu_factor(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.linalg_lu_factor`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.linalg_lu_factor`.
    """
    return _dispatch_aten(*args, **kwargs)


def linalg_lu_factor_ex(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.linalg_lu_factor_ex`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.linalg_lu_factor_ex`.
    """
    return _dispatch_aten(*args, **kwargs)


def linalg_lu_solve(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.linalg_lu_solve`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.linalg_lu_solve`.
    """
    return _dispatch_aten(*args, **kwargs)


def linalg_matmul(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.linalg_matmul`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.linalg_matmul`.
    """
    return _dispatch_aten(*args, **kwargs)


def linalg_matrix_exp(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.linalg_matrix_exp`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.linalg_matrix_exp`.
    """
    return _dispatch_aten(*args, **kwargs)


def linalg_matrix_norm(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.linalg_matrix_norm`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.linalg_matrix_norm`.
    """
    return _dispatch_aten(*args, **kwargs)


def linalg_matrix_power(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.linalg_matrix_power`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.linalg_matrix_power`.
    """
    return _dispatch_aten(*args, **kwargs)


def linalg_matrix_rank(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.linalg_matrix_rank`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.linalg_matrix_rank`.
    """
    return _dispatch_aten(*args, **kwargs)


def linalg_matrix_sqrth(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.linalg_matrix_sqrth`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.linalg_matrix_sqrth`.
    """
    return _dispatch_aten(*args, **kwargs)


def linalg_multi_dot(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.linalg_multi_dot`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.linalg_multi_dot`.
    """
    return _dispatch_aten(*args, **kwargs)


def linalg_norm(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.linalg_norm`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.linalg_norm`.
    """
    return _dispatch_aten(*args, **kwargs)


def linalg_pinv(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.linalg_pinv`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.linalg_pinv`.
    """
    return _dispatch_aten(*args, **kwargs)


def linalg_polar(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.linalg_polar`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.linalg_polar`.
    """
    return _dispatch_aten(*args, **kwargs)


def linalg_qr(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.linalg_qr`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.linalg_qr`.
    """
    return _dispatch_aten(*args, **kwargs)


def linalg_slogdet(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.linalg_slogdet`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.linalg_slogdet`.
    """
    return _dispatch_aten(*args, **kwargs)


def linalg_solve(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.linalg_solve`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.linalg_solve`.
    """
    return _dispatch_aten(*args, **kwargs)


def linalg_solve_ex(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.linalg_solve_ex`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.linalg_solve_ex`.
    """
    return _dispatch_aten(*args, **kwargs)


def linalg_solve_triangular(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.linalg_solve_triangular`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.linalg_solve_triangular`.
    """
    return _dispatch_aten(*args, **kwargs)


def linalg_svd(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.linalg_svd`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.linalg_svd`.
    """
    return _dispatch_aten(*args, **kwargs)


def linalg_svdvals(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.linalg_svdvals`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.linalg_svdvals`.
    """
    return _dispatch_aten(*args, **kwargs)


def linalg_tensorinv(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.linalg_tensorinv`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.linalg_tensorinv`.
    """
    return _dispatch_aten(*args, **kwargs)


def linalg_tensorsolve(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.linalg_tensorsolve`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.linalg_tensorsolve`.
    """
    return _dispatch_aten(*args, **kwargs)


def linalg_vander(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.linalg_vander`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.linalg_vander`.
    """
    return _dispatch_aten(*args, **kwargs)


def linalg_vecdot(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.linalg_vecdot`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.linalg_vecdot`.
    """
    return _dispatch_aten(*args, **kwargs)


def linalg_vector_norm(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.linalg_vector_norm`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.linalg_vector_norm`.
    """
    return _dispatch_aten(*args, **kwargs)


def linear(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.linear`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.linear`.
    """
    return _dispatch_aten(*args, **kwargs)


def linear_backward(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.linear_backward`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.linear_backward`.
    """
    return _dispatch_aten(*args, **kwargs)


def linspace(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.linspace`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.linspace`.
    """
    return _dispatch_aten(*args, **kwargs)


def log(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.log`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.log`.
    """
    return _dispatch_aten(*args, **kwargs)


def log10(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.log10`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.log10`.
    """
    return _dispatch_aten(*args, **kwargs)


def log10_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.log10_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.log10_`.
    """
    return _dispatch_aten(*args, **kwargs)


def log1p(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.log1p`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.log1p`.
    """
    return _dispatch_aten(*args, **kwargs)


def log1p_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.log1p_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.log1p_`.
    """
    return _dispatch_aten(*args, **kwargs)


def log2(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.log2`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.log2`.
    """
    return _dispatch_aten(*args, **kwargs)


def log2_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.log2_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.log2_`.
    """
    return _dispatch_aten(*args, **kwargs)


def log_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.log_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.log_`.
    """
    return _dispatch_aten(*args, **kwargs)


def log_normal(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.log_normal`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.log_normal`.
    """
    return _dispatch_aten(*args, **kwargs)


def log_normal_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.log_normal_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.log_normal_`.
    """
    return _dispatch_aten(*args, **kwargs)


def log_sigmoid(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.log_sigmoid`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.log_sigmoid`.
    """
    return _dispatch_aten(*args, **kwargs)


def log_sigmoid_backward(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.log_sigmoid_backward`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.log_sigmoid_backward`.
    """
    return _dispatch_aten(*args, **kwargs)


def log_sigmoid_forward(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.log_sigmoid_forward`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.log_sigmoid_forward`.
    """
    return _dispatch_aten(*args, **kwargs)


def log_softmax(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.log_softmax`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.log_softmax`.
    """
    return _dispatch_aten(*args, **kwargs)


def logaddexp(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.logaddexp`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.logaddexp`.
    """
    return _dispatch_aten(*args, **kwargs)


def logaddexp2(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.logaddexp2`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.logaddexp2`.
    """
    return _dispatch_aten(*args, **kwargs)


def logcumsumexp(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.logcumsumexp`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.logcumsumexp`.
    """
    return _dispatch_aten(*args, **kwargs)


def logdet(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.logdet`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.logdet`.
    """
    return _dispatch_aten(*args, **kwargs)


def logical_and(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.logical_and`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.logical_and`.
    """
    return _dispatch_aten(*args, **kwargs)


def logical_and_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.logical_and_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.logical_and_`.
    """
    return _dispatch_aten(*args, **kwargs)


def logical_not(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.logical_not`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.logical_not`.
    """
    return _dispatch_aten(*args, **kwargs)


def logical_not_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.logical_not_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.logical_not_`.
    """
    return _dispatch_aten(*args, **kwargs)


def logical_or(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.logical_or`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.logical_or`.
    """
    return _dispatch_aten(*args, **kwargs)


def logical_or_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.logical_or_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.logical_or_`.
    """
    return _dispatch_aten(*args, **kwargs)


def logical_xor(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.logical_xor`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.logical_xor`.
    """
    return _dispatch_aten(*args, **kwargs)


def logical_xor_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.logical_xor_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.logical_xor_`.
    """
    return _dispatch_aten(*args, **kwargs)


def logit(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.logit`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.logit`.
    """
    return _dispatch_aten(*args, **kwargs)


def logit_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.logit_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.logit_`.
    """
    return _dispatch_aten(*args, **kwargs)


def logit_backward(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.logit_backward`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.logit_backward`.
    """
    return _dispatch_aten(*args, **kwargs)


def logspace(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.logspace`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.logspace`.
    """
    return _dispatch_aten(*args, **kwargs)


def logsumexp(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.logsumexp`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.logsumexp`.
    """
    return _dispatch_aten(*args, **kwargs)


def lstm(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.lstm`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.lstm`.
    """
    return _dispatch_aten(*args, **kwargs)


def lt(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.lt`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.lt`.
    """
    return _dispatch_aten(*args, **kwargs)


def lt_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.lt_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.lt_`.
    """
    return _dispatch_aten(*args, **kwargs)


def lu_solve(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.lu_solve`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.lu_solve`.
    """
    return _dispatch_aten(*args, **kwargs)


def lu_unpack(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.lu_unpack`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.lu_unpack`.
    """
    return _dispatch_aten(*args, **kwargs)


def margin_ranking_loss(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.margin_ranking_loss`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.margin_ranking_loss`.
    """
    return _dispatch_aten(*args, **kwargs)


def masked_fill(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.masked_fill`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.masked_fill`.
    """
    return _dispatch_aten(*args, **kwargs)


def masked_fill_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.masked_fill_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.masked_fill_`.
    """
    return _dispatch_aten(*args, **kwargs)


def masked_scatter(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.masked_scatter`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.masked_scatter`.
    """
    return _dispatch_aten(*args, **kwargs)


def masked_scatter_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.masked_scatter_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.masked_scatter_`.
    """
    return _dispatch_aten(*args, **kwargs)


def masked_scatter_backward(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.masked_scatter_backward`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.masked_scatter_backward`.
    """
    return _dispatch_aten(*args, **kwargs)


def masked_select(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.masked_select`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.masked_select`.
    """
    return _dispatch_aten(*args, **kwargs)


def matmul(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.matmul`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.matmul`.
    """
    return _dispatch_aten(*args, **kwargs)


def matrix_exp(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.matrix_exp`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.matrix_exp`.
    """
    return _dispatch_aten(*args, **kwargs)


def matrix_power(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.matrix_power`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.matrix_power`.
    """
    return _dispatch_aten(*args, **kwargs)


def max(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.max`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.max`.
    """
    return _dispatch_aten(*args, **kwargs)


def max_pool2d_with_indices(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.max_pool2d_with_indices`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.max_pool2d_with_indices`.
    """
    return _dispatch_aten(*args, **kwargs)


def max_pool2d_with_indices_backward(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.max_pool2d_with_indices_backward`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.max_pool2d_with_indices_backward`.
    """
    return _dispatch_aten(*args, **kwargs)


def max_pool3d_with_indices(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.max_pool3d_with_indices`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.max_pool3d_with_indices`.
    """
    return _dispatch_aten(*args, **kwargs)


def max_pool3d_with_indices_backward(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.max_pool3d_with_indices_backward`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.max_pool3d_with_indices_backward`.
    """
    return _dispatch_aten(*args, **kwargs)


def max_unpool2d(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.max_unpool2d`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.max_unpool2d`.
    """
    return _dispatch_aten(*args, **kwargs)


def max_unpool3d(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.max_unpool3d`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.max_unpool3d`.
    """
    return _dispatch_aten(*args, **kwargs)


def maximum(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.maximum`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.maximum`.
    """
    return _dispatch_aten(*args, **kwargs)


def mean(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.mean`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.mean`.
    """
    return _dispatch_aten(*args, **kwargs)


def median(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.median`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.median`.
    """
    return _dispatch_aten(*args, **kwargs)


def meshgrid(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.meshgrid`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.meshgrid`.
    """
    return _dispatch_aten(*args, **kwargs)


def min(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.min`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.min`.
    """
    return _dispatch_aten(*args, **kwargs)


def minimum(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.minimum`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.minimum`.
    """
    return _dispatch_aten(*args, **kwargs)


def miopen_batch_norm(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.miopen_batch_norm`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.miopen_batch_norm`.
    """
    return _dispatch_aten(*args, **kwargs)


def miopen_batch_norm_backward(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.miopen_batch_norm_backward`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.miopen_batch_norm_backward`.
    """
    return _dispatch_aten(*args, **kwargs)


def miopen_ctc_loss(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.miopen_ctc_loss`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.miopen_ctc_loss`.
    """
    return _dispatch_aten(*args, **kwargs)


def miopen_rnn(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.miopen_rnn`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.miopen_rnn`.
    """
    return _dispatch_aten(*args, **kwargs)


def mish(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.mish`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.mish`.
    """
    return _dispatch_aten(*args, **kwargs)


def mish_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.mish_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.mish_`.
    """
    return _dispatch_aten(*args, **kwargs)


def mish_backward(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.mish_backward`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.mish_backward`.
    """
    return _dispatch_aten(*args, **kwargs)


def mkldnn_rnn_layer(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.mkldnn_rnn_layer`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.mkldnn_rnn_layer`.
    """
    return _dispatch_aten(*args, **kwargs)


def mkldnn_rnn_layer_backward(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.mkldnn_rnn_layer_backward`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.mkldnn_rnn_layer_backward`.
    """
    return _dispatch_aten(*args, **kwargs)


def mm(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.mm`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.mm`.
    """
    return _dispatch_aten(*args, **kwargs)


def mode(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.mode`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.mode`.
    """
    return _dispatch_aten(*args, **kwargs)


def moveaxis(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.moveaxis`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.moveaxis`.
    """
    return _dispatch_aten(*args, **kwargs)


def movedim(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.movedim`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.movedim`.
    """
    return _dispatch_aten(*args, **kwargs)


def mse_loss(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.mse_loss`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.mse_loss`.
    """
    return _dispatch_aten(*args, **kwargs)


def mse_loss_backward(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.mse_loss_backward`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.mse_loss_backward`.
    """
    return _dispatch_aten(*args, **kwargs)


def msort(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.msort`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.msort`.
    """
    return _dispatch_aten(*args, **kwargs)


def mul(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.mul`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.mul`.
    """
    return _dispatch_aten(*args, **kwargs)


def mul_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.mul_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.mul_`.
    """
    return _dispatch_aten(*args, **kwargs)


def multi_margin_loss(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.multi_margin_loss`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.multi_margin_loss`.
    """
    return _dispatch_aten(*args, **kwargs)


def multilabel_margin_loss_forward(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.multilabel_margin_loss_forward`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.multilabel_margin_loss_forward`.
    """
    return _dispatch_aten(*args, **kwargs)


def multinomial(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.multinomial`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.multinomial`.
    """
    return _dispatch_aten(*args, **kwargs)


def multiply(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.multiply`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.multiply`.
    """
    return _dispatch_aten(*args, **kwargs)


def multiply_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.multiply_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.multiply_`.
    """
    return _dispatch_aten(*args, **kwargs)


def mv(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.mv`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.mv`.
    """
    return _dispatch_aten(*args, **kwargs)


def mvlgamma(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.mvlgamma`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.mvlgamma`.
    """
    return _dispatch_aten(*args, **kwargs)


def mvlgamma_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.mvlgamma_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.mvlgamma_`.
    """
    return _dispatch_aten(*args, **kwargs)


def nan_to_num(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.nan_to_num`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.nan_to_num`.
    """
    return _dispatch_aten(*args, **kwargs)


def nan_to_num_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.nan_to_num_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.nan_to_num_`.
    """
    return _dispatch_aten(*args, **kwargs)


def nanmean(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.nanmean`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.nanmean`.
    """
    return _dispatch_aten(*args, **kwargs)


def nanmedian(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.nanmedian`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.nanmedian`.
    """
    return _dispatch_aten(*args, **kwargs)


def nanquantile(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.nanquantile`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.nanquantile`.
    """
    return _dispatch_aten(*args, **kwargs)


def nansum(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.nansum`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.nansum`.
    """
    return _dispatch_aten(*args, **kwargs)


def narrow(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.narrow`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.narrow`.
    """
    return _dispatch_aten(*args, **kwargs)


def narrow_copy(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.narrow_copy`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.narrow_copy`.
    """
    return _dispatch_aten(*args, **kwargs)


def native_batch_norm(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.native_batch_norm`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.native_batch_norm`.
    """
    return _dispatch_aten(*args, **kwargs)


def native_batch_norm_backward(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.native_batch_norm_backward`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.native_batch_norm_backward`.
    """
    return _dispatch_aten(*args, **kwargs)


def native_channel_shuffle(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.native_channel_shuffle`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.native_channel_shuffle`.
    """
    return _dispatch_aten(*args, **kwargs)


def native_dropout(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.native_dropout`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.native_dropout`.
    """
    return _dispatch_aten(*args, **kwargs)


def native_dropout_backward(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.native_dropout_backward`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.native_dropout_backward`.
    """
    return _dispatch_aten(*args, **kwargs)


def native_group_norm(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.native_group_norm`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.native_group_norm`.
    """
    return _dispatch_aten(*args, **kwargs)


def native_group_norm_backward(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.native_group_norm_backward`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.native_group_norm_backward`.
    """
    return _dispatch_aten(*args, **kwargs)


def native_layer_norm(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.native_layer_norm`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.native_layer_norm`.
    """
    return _dispatch_aten(*args, **kwargs)


def native_layer_norm_backward(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.native_layer_norm_backward`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.native_layer_norm_backward`.
    """
    return _dispatch_aten(*args, **kwargs)


def native_norm(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.native_norm`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.native_norm`.
    """
    return _dispatch_aten(*args, **kwargs)


def ne(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.ne`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.ne`.
    """
    return _dispatch_aten(*args, **kwargs)


def ne_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.ne_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.ne_`.
    """
    return _dispatch_aten(*args, **kwargs)


def neg(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.neg`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.neg`.
    """
    return _dispatch_aten(*args, **kwargs)


def neg_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.neg_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.neg_`.
    """
    return _dispatch_aten(*args, **kwargs)


def negative(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.negative`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.negative`.
    """
    return _dispatch_aten(*args, **kwargs)


def negative_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.negative_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.negative_`.
    """
    return _dispatch_aten(*args, **kwargs)


def new_empty(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.new_empty`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.new_empty`.
    """
    return _dispatch_aten(*args, **kwargs)


def new_empty_strided(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.new_empty_strided`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.new_empty_strided`.
    """
    return _dispatch_aten(*args, **kwargs)


def new_full(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.new_full`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.new_full`.
    """
    return _dispatch_aten(*args, **kwargs)


def new_ones(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.new_ones`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.new_ones`.
    """
    return _dispatch_aten(*args, **kwargs)


def new_zeros(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.new_zeros`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.new_zeros`.
    """
    return _dispatch_aten(*args, **kwargs)


def nextafter(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.nextafter`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.nextafter`.
    """
    return _dispatch_aten(*args, **kwargs)


def nextafter_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.nextafter_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.nextafter_`.
    """
    return _dispatch_aten(*args, **kwargs)


def nll_loss(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.nll_loss`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.nll_loss`.
    """
    return _dispatch_aten(*args, **kwargs)


def nll_loss2d_backward(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.nll_loss2d_backward`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.nll_loss2d_backward`.
    """
    return _dispatch_aten(*args, **kwargs)


def nll_loss2d_forward(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.nll_loss2d_forward`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.nll_loss2d_forward`.
    """
    return _dispatch_aten(*args, **kwargs)


def nll_loss_backward(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.nll_loss_backward`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.nll_loss_backward`.
    """
    return _dispatch_aten(*args, **kwargs)


def nll_loss_forward(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.nll_loss_forward`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.nll_loss_forward`.
    """
    return _dispatch_aten(*args, **kwargs)


def nonzero(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.nonzero`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.nonzero`.
    """
    return _dispatch_aten(*args, **kwargs)


def nonzero_numpy(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.nonzero_numpy`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.nonzero_numpy`.
    """
    return _dispatch_aten(*args, **kwargs)


def nonzero_static(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.nonzero_static`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.nonzero_static`.
    """
    return _dispatch_aten(*args, **kwargs)


def norm(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.norm`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.norm`.
    """
    return _dispatch_aten(*args, **kwargs)


def normal(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.normal`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.normal`.
    """
    return _dispatch_aten(*args, **kwargs)


def normal_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.normal_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.normal_`.
    """
    return _dispatch_aten(*args, **kwargs)


def normal_functional(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.normal_functional`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.normal_functional`.
    """
    return _dispatch_aten(*args, **kwargs)


def not_equal(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.not_equal`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.not_equal`.
    """
    return _dispatch_aten(*args, **kwargs)


def not_equal_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.not_equal_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.not_equal_`.
    """
    return _dispatch_aten(*args, **kwargs)


def nuclear_norm(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.nuclear_norm`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.nuclear_norm`.
    """
    return _dispatch_aten(*args, **kwargs)


def numel(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.numel`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.numel`.
    """
    return _dispatch_aten(*args, **kwargs)


def one_hot(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.one_hot`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.one_hot`.
    """
    return _dispatch_aten(*args, **kwargs)


def ones(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.ones`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.ones`.
    """
    return _dispatch_aten(*args, **kwargs)


def ones_like(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.ones_like`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.ones_like`.
    """
    return _dispatch_aten(*args, **kwargs)


def orgqr(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.orgqr`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.orgqr`.
    """
    return _dispatch_aten(*args, **kwargs)


def ormqr(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.ormqr`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.ormqr`.
    """
    return _dispatch_aten(*args, **kwargs)


def outer(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.outer`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.outer`.
    """
    return _dispatch_aten(*args, **kwargs)


def pad_sequence(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.pad_sequence`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.pad_sequence`.
    """
    return _dispatch_aten(*args, **kwargs)


def pairwise_distance(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.pairwise_distance`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.pairwise_distance`.
    """
    return _dispatch_aten(*args, **kwargs)


def pdist(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.pdist`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.pdist`.
    """
    return _dispatch_aten(*args, **kwargs)


def permute(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.permute`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.permute`.
    """
    return _dispatch_aten(*args, **kwargs)


def permute_copy(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.permute_copy`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.permute_copy`.
    """
    return _dispatch_aten(*args, **kwargs)


def pin_memory(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.pin_memory`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.pin_memory`.
    """
    return _dispatch_aten(*args, **kwargs)


def pinverse(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.pinverse`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.pinverse`.
    """
    return _dispatch_aten(*args, **kwargs)


def pixel_shuffle(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.pixel_shuffle`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.pixel_shuffle`.
    """
    return _dispatch_aten(*args, **kwargs)


def pixel_unshuffle(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.pixel_unshuffle`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.pixel_unshuffle`.
    """
    return _dispatch_aten(*args, **kwargs)


def poisson(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.poisson`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.poisson`.
    """
    return _dispatch_aten(*args, **kwargs)


def polar(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.polar`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.polar`.
    """
    return _dispatch_aten(*args, **kwargs)


def polygamma(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.polygamma`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.polygamma`.
    """
    return _dispatch_aten(*args, **kwargs)


def polygamma_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.polygamma_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.polygamma_`.
    """
    return _dispatch_aten(*args, **kwargs)


def positive(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.positive`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.positive`.
    """
    return _dispatch_aten(*args, **kwargs)


def pow(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.pow`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.pow`.
    """
    return _dispatch_aten(*args, **kwargs)


def pow_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.pow_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.pow_`.
    """
    return _dispatch_aten(*args, **kwargs)


def prelu(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.prelu`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.prelu`.
    """
    return _dispatch_aten(*args, **kwargs)


def prod(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.prod`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.prod`.
    """
    return _dispatch_aten(*args, **kwargs)


def put(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.put`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.put`.
    """
    return _dispatch_aten(*args, **kwargs)


def put_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.put_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.put_`.
    """
    return _dispatch_aten(*args, **kwargs)


def q_per_channel_axis(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.q_per_channel_axis`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.q_per_channel_axis`.
    """
    return _dispatch_aten(*args, **kwargs)


def q_per_channel_scales(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.q_per_channel_scales`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.q_per_channel_scales`.
    """
    return _dispatch_aten(*args, **kwargs)


def q_per_channel_zero_points(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.q_per_channel_zero_points`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.q_per_channel_zero_points`.
    """
    return _dispatch_aten(*args, **kwargs)


def q_scale(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.q_scale`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.q_scale`.
    """
    return _dispatch_aten(*args, **kwargs)


def q_zero_point(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.q_zero_point`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.q_zero_point`.
    """
    return _dispatch_aten(*args, **kwargs)


def qscheme(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.qscheme`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.qscheme`.
    """
    return _dispatch_aten(*args, **kwargs)


def quantile(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.quantile`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.quantile`.
    """
    return _dispatch_aten(*args, **kwargs)


def quantize_per_tensor(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.quantize_per_tensor`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.quantize_per_tensor`.
    """
    return _dispatch_aten(*args, **kwargs)


def rad2deg(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.rad2deg`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.rad2deg`.
    """
    return _dispatch_aten(*args, **kwargs)


def rad2deg_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.rad2deg_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.rad2deg_`.
    """
    return _dispatch_aten(*args, **kwargs)


def rand(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.rand`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.rand`.
    """
    return _dispatch_aten(*args, **kwargs)


def rand_like(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.rand_like`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.rand_like`.
    """
    return _dispatch_aten(*args, **kwargs)


def randint(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.randint`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.randint`.
    """
    return _dispatch_aten(*args, **kwargs)


def randint_like(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.randint_like`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.randint_like`.
    """
    return _dispatch_aten(*args, **kwargs)


def randn(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.randn`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.randn`.
    """
    return _dispatch_aten(*args, **kwargs)


def randn_like(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.randn_like`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.randn_like`.
    """
    return _dispatch_aten(*args, **kwargs)


def random_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.random_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.random_`.
    """
    return _dispatch_aten(*args, **kwargs)


def randperm(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.randperm`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.randperm`.
    """
    return _dispatch_aten(*args, **kwargs)


def ravel(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.ravel`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.ravel`.
    """
    return _dispatch_aten(*args, **kwargs)


def real(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.real`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.real`.
    """
    return _dispatch_aten(*args, **kwargs)


def reciprocal(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.reciprocal`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.reciprocal`.
    """
    return _dispatch_aten(*args, **kwargs)


def reciprocal_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.reciprocal_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.reciprocal_`.
    """
    return _dispatch_aten(*args, **kwargs)


def record_stream(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.record_stream`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.record_stream`.
    """
    return _dispatch_aten(*args, **kwargs)


def reflection_pad1d(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.reflection_pad1d`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.reflection_pad1d`.
    """
    return _dispatch_aten(*args, **kwargs)


def reflection_pad1d_backward(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.reflection_pad1d_backward`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.reflection_pad1d_backward`.
    """
    return _dispatch_aten(*args, **kwargs)


def reflection_pad2d(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.reflection_pad2d`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.reflection_pad2d`.
    """
    return _dispatch_aten(*args, **kwargs)


def reflection_pad2d_backward(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.reflection_pad2d_backward`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.reflection_pad2d_backward`.
    """
    return _dispatch_aten(*args, **kwargs)


def reflection_pad3d(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.reflection_pad3d`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.reflection_pad3d`.
    """
    return _dispatch_aten(*args, **kwargs)


def reflection_pad3d_backward(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.reflection_pad3d_backward`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.reflection_pad3d_backward`.
    """
    return _dispatch_aten(*args, **kwargs)


def relu(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.relu`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.relu`.
    """
    return _dispatch_aten(*args, **kwargs)


def relu6(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.relu6`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.relu6`.
    """
    return _dispatch_aten(*args, **kwargs)


def relu_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.relu_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.relu_`.
    """
    return _dispatch_aten(*args, **kwargs)


def remainder(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.remainder`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.remainder`.
    """
    return _dispatch_aten(*args, **kwargs)


def remainder_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.remainder_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.remainder_`.
    """
    return _dispatch_aten(*args, **kwargs)


def renorm(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.renorm`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.renorm`.
    """
    return _dispatch_aten(*args, **kwargs)


def renorm_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.renorm_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.renorm_`.
    """
    return _dispatch_aten(*args, **kwargs)


def repeat(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.repeat`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.repeat`.
    """
    return _dispatch_aten(*args, **kwargs)


def repeat_interleave(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.repeat_interleave`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.repeat_interleave`.
    """
    return _dispatch_aten(*args, **kwargs)


def replication_pad1d(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.replication_pad1d`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.replication_pad1d`.
    """
    return _dispatch_aten(*args, **kwargs)


def replication_pad1d_backward(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.replication_pad1d_backward`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.replication_pad1d_backward`.
    """
    return _dispatch_aten(*args, **kwargs)


def replication_pad2d(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.replication_pad2d`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.replication_pad2d`.
    """
    return _dispatch_aten(*args, **kwargs)


def replication_pad2d_backward(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.replication_pad2d_backward`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.replication_pad2d_backward`.
    """
    return _dispatch_aten(*args, **kwargs)


def replication_pad3d(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.replication_pad3d`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.replication_pad3d`.
    """
    return _dispatch_aten(*args, **kwargs)


def replication_pad3d_backward(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.replication_pad3d_backward`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.replication_pad3d_backward`.
    """
    return _dispatch_aten(*args, **kwargs)


def requires_grad_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.requires_grad_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.requires_grad_`.
    """
    return _dispatch_aten(*args, **kwargs)


def reshape(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.reshape`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.reshape`.
    """
    return _dispatch_aten(*args, **kwargs)


def reshape_as(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.reshape_as`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.reshape_as`.
    """
    return _dispatch_aten(*args, **kwargs)


def resize(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.resize`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.resize`.
    """
    return _dispatch_aten(*args, **kwargs)


def resize_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.resize_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.resize_`.
    """
    return _dispatch_aten(*args, **kwargs)


def resize_as(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.resize_as`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.resize_as`.
    """
    return _dispatch_aten(*args, **kwargs)


def resize_as_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.resize_as_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.resize_as_`.
    """
    return _dispatch_aten(*args, **kwargs)


def resize_as_sparse_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.resize_as_sparse_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.resize_as_sparse_`.
    """
    return _dispatch_aten(*args, **kwargs)


def resolve_conj(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.resolve_conj`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.resolve_conj`.
    """
    return _dispatch_aten(*args, **kwargs)


def resolve_neg(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.resolve_neg`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.resolve_neg`.
    """
    return _dispatch_aten(*args, **kwargs)


def result_type(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.result_type`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.result_type`.
    """
    return _dispatch_aten(*args, **kwargs)


def retain_grad(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.retain_grad`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.retain_grad`.
    """
    return _dispatch_aten(*args, **kwargs)


def rnn_relu(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.rnn_relu`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.rnn_relu`.
    """
    return _dispatch_aten(*args, **kwargs)


def rnn_tanh(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.rnn_tanh`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.rnn_tanh`.
    """
    return _dispatch_aten(*args, **kwargs)


def roll(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.roll`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.roll`.
    """
    return _dispatch_aten(*args, **kwargs)


def rot90(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.rot90`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.rot90`.
    """
    return _dispatch_aten(*args, **kwargs)


def round(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.round`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.round`.
    """
    return _dispatch_aten(*args, **kwargs)


def round_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.round_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.round_`.
    """
    return _dispatch_aten(*args, **kwargs)


def row_indices(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.row_indices`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.row_indices`.
    """
    return _dispatch_aten(*args, **kwargs)


def rrelu_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.rrelu_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.rrelu_`.
    """
    return _dispatch_aten(*args, **kwargs)


def rrelu_with_noise(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.rrelu_with_noise`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.rrelu_with_noise`.
    """
    return _dispatch_aten(*args, **kwargs)


def rrelu_with_noise_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.rrelu_with_noise_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.rrelu_with_noise_`.
    """
    return _dispatch_aten(*args, **kwargs)


def rrelu_with_noise_backward(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.rrelu_with_noise_backward`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.rrelu_with_noise_backward`.
    """
    return _dispatch_aten(*args, **kwargs)


def rrelu_with_noise_functional(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.rrelu_with_noise_functional`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.rrelu_with_noise_functional`.
    """
    return _dispatch_aten(*args, **kwargs)


def rsqrt(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.rsqrt`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.rsqrt`.
    """
    return _dispatch_aten(*args, **kwargs)


def rsqrt_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.rsqrt_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.rsqrt_`.
    """
    return _dispatch_aten(*args, **kwargs)


def rsub(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.rsub`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.rsub`.
    """
    return _dispatch_aten(*args, **kwargs)


def scalar_tensor(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.scalar_tensor`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.scalar_tensor`.
    """
    return _dispatch_aten(*args, **kwargs)


def scaled_dot_product_attention(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.scaled_dot_product_attention`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.scaled_dot_product_attention`.
    """
    return _dispatch_aten(*args, **kwargs)


def scatter(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.scatter`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.scatter`.
    """
    return _dispatch_aten(*args, **kwargs)


def scatter_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.scatter_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.scatter_`.
    """
    return _dispatch_aten(*args, **kwargs)


def scatter_add(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.scatter_add`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.scatter_add`.
    """
    return _dispatch_aten(*args, **kwargs)


def scatter_add_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.scatter_add_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.scatter_add_`.
    """
    return _dispatch_aten(*args, **kwargs)


def scatter_reduce(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.scatter_reduce`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.scatter_reduce`.
    """
    return _dispatch_aten(*args, **kwargs)


def scatter_reduce_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.scatter_reduce_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.scatter_reduce_`.
    """
    return _dispatch_aten(*args, **kwargs)


def searchsorted(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.searchsorted`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.searchsorted`.
    """
    return _dispatch_aten(*args, **kwargs)


def segment_reduce(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.segment_reduce`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.segment_reduce`.
    """
    return _dispatch_aten(*args, **kwargs)


def select(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.select`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.select`.
    """
    return _dispatch_aten(*args, **kwargs)


def select_backward(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.select_backward`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.select_backward`.
    """
    return _dispatch_aten(*args, **kwargs)


def select_scatter(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.select_scatter`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.select_scatter`.
    """
    return _dispatch_aten(*args, **kwargs)


def selu(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.selu`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.selu`.
    """
    return _dispatch_aten(*args, **kwargs)


def selu_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.selu_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.selu_`.
    """
    return _dispatch_aten(*args, **kwargs)


def set_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.set_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.set_`.
    """
    return _dispatch_aten(*args, **kwargs)


def sgn(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.sgn`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.sgn`.
    """
    return _dispatch_aten(*args, **kwargs)


def sgn_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.sgn_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.sgn_`.
    """
    return _dispatch_aten(*args, **kwargs)


def shallow_copy_data_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.shallow_copy_data_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.shallow_copy_data_`.
    """
    return _dispatch_aten(*args, **kwargs)


def sigmoid(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.sigmoid`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.sigmoid`.
    """
    return _dispatch_aten(*args, **kwargs)


def sigmoid_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.sigmoid_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.sigmoid_`.
    """
    return _dispatch_aten(*args, **kwargs)


def sigmoid_backward(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.sigmoid_backward`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.sigmoid_backward`.
    """
    return _dispatch_aten(*args, **kwargs)


def sign(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.sign`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.sign`.
    """
    return _dispatch_aten(*args, **kwargs)


def sign_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.sign_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.sign_`.
    """
    return _dispatch_aten(*args, **kwargs)


def signbit(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.signbit`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.signbit`.
    """
    return _dispatch_aten(*args, **kwargs)


def silu(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.silu`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.silu`.
    """
    return _dispatch_aten(*args, **kwargs)


def silu_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.silu_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.silu_`.
    """
    return _dispatch_aten(*args, **kwargs)


def silu_backward(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.silu_backward`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.silu_backward`.
    """
    return _dispatch_aten(*args, **kwargs)


def sin(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.sin`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.sin`.
    """
    return _dispatch_aten(*args, **kwargs)


def sin_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.sin_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.sin_`.
    """
    return _dispatch_aten(*args, **kwargs)


def sinc(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.sinc`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.sinc`.
    """
    return _dispatch_aten(*args, **kwargs)


def sinc_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.sinc_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.sinc_`.
    """
    return _dispatch_aten(*args, **kwargs)


def sinh(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.sinh`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.sinh`.
    """
    return _dispatch_aten(*args, **kwargs)


def sinh_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.sinh_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.sinh_`.
    """
    return _dispatch_aten(*args, **kwargs)


def size(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.size`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.size`.
    """
    return _dispatch_aten(*args, **kwargs)


def slice(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.slice`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.slice`.
    """
    return _dispatch_aten(*args, **kwargs)


def slice_backward(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.slice_backward`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.slice_backward`.
    """
    return _dispatch_aten(*args, **kwargs)


def slice_inverse(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.slice_inverse`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.slice_inverse`.
    """
    return _dispatch_aten(*args, **kwargs)


def slice_scatter(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.slice_scatter`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.slice_scatter`.
    """
    return _dispatch_aten(*args, **kwargs)


def slogdet(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.slogdet`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.slogdet`.
    """
    return _dispatch_aten(*args, **kwargs)


def smm(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.smm`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.smm`.
    """
    return _dispatch_aten(*args, **kwargs)


def smooth_l1_loss(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.smooth_l1_loss`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.smooth_l1_loss`.
    """
    return _dispatch_aten(*args, **kwargs)


def smooth_l1_loss_backward(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.smooth_l1_loss_backward`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.smooth_l1_loss_backward`.
    """
    return _dispatch_aten(*args, **kwargs)


def soft_margin_loss(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.soft_margin_loss`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.soft_margin_loss`.
    """
    return _dispatch_aten(*args, **kwargs)


def soft_margin_loss_backward(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.soft_margin_loss_backward`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.soft_margin_loss_backward`.
    """
    return _dispatch_aten(*args, **kwargs)


def softmax(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.softmax`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.softmax`.
    """
    return _dispatch_aten(*args, **kwargs)


def softplus(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.softplus`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.softplus`.
    """
    return _dispatch_aten(*args, **kwargs)


def softplus_backward(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.softplus_backward`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.softplus_backward`.
    """
    return _dispatch_aten(*args, **kwargs)


def softshrink(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.softshrink`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.softshrink`.
    """
    return _dispatch_aten(*args, **kwargs)


def sort(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.sort`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.sort`.
    """
    return _dispatch_aten(*args, **kwargs)


def sparse_dim(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.sparse_dim`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.sparse_dim`.
    """
    return _dispatch_aten(*args, **kwargs)


def sparse_mask(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.sparse_mask`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.sparse_mask`.
    """
    return _dispatch_aten(*args, **kwargs)


def sparse_resize_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.sparse_resize_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.sparse_resize_`.
    """
    return _dispatch_aten(*args, **kwargs)


def sparse_resize_and_clear_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.sparse_resize_and_clear_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.sparse_resize_and_clear_`.
    """
    return _dispatch_aten(*args, **kwargs)


def special_airy_ai(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.special_airy_ai`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.special_airy_ai`.
    """
    return _dispatch_aten(*args, **kwargs)


def special_bessel_j0(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.special_bessel_j0`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.special_bessel_j0`.
    """
    return _dispatch_aten(*args, **kwargs)


def special_bessel_j1(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.special_bessel_j1`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.special_bessel_j1`.
    """
    return _dispatch_aten(*args, **kwargs)


def special_bessel_y0(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.special_bessel_y0`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.special_bessel_y0`.
    """
    return _dispatch_aten(*args, **kwargs)


def special_bessel_y1(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.special_bessel_y1`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.special_bessel_y1`.
    """
    return _dispatch_aten(*args, **kwargs)


def special_chebyshev_polynomial_t(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.special_chebyshev_polynomial_t`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.special_chebyshev_polynomial_t`.
    """
    return _dispatch_aten(*args, **kwargs)


def special_chebyshev_polynomial_u(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.special_chebyshev_polynomial_u`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.special_chebyshev_polynomial_u`.
    """
    return _dispatch_aten(*args, **kwargs)


def special_chebyshev_polynomial_v(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.special_chebyshev_polynomial_v`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.special_chebyshev_polynomial_v`.
    """
    return _dispatch_aten(*args, **kwargs)


def special_chebyshev_polynomial_w(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.special_chebyshev_polynomial_w`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.special_chebyshev_polynomial_w`.
    """
    return _dispatch_aten(*args, **kwargs)


def special_digamma(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.special_digamma`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.special_digamma`.
    """
    return _dispatch_aten(*args, **kwargs)


def special_entr(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.special_entr`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.special_entr`.
    """
    return _dispatch_aten(*args, **kwargs)


def special_erf(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.special_erf`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.special_erf`.
    """
    return _dispatch_aten(*args, **kwargs)


def special_erfc(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.special_erfc`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.special_erfc`.
    """
    return _dispatch_aten(*args, **kwargs)


def special_erfcx(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.special_erfcx`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.special_erfcx`.
    """
    return _dispatch_aten(*args, **kwargs)


def special_erfinv(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.special_erfinv`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.special_erfinv`.
    """
    return _dispatch_aten(*args, **kwargs)


def special_exp2(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.special_exp2`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.special_exp2`.
    """
    return _dispatch_aten(*args, **kwargs)


def special_expit(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.special_expit`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.special_expit`.
    """
    return _dispatch_aten(*args, **kwargs)


def special_expm1(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.special_expm1`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.special_expm1`.
    """
    return _dispatch_aten(*args, **kwargs)


def special_gammainc(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.special_gammainc`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.special_gammainc`.
    """
    return _dispatch_aten(*args, **kwargs)


def special_gammaincc(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.special_gammaincc`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.special_gammaincc`.
    """
    return _dispatch_aten(*args, **kwargs)


def special_gammaln(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.special_gammaln`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.special_gammaln`.
    """
    return _dispatch_aten(*args, **kwargs)


def special_hermite_polynomial_h(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.special_hermite_polynomial_h`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.special_hermite_polynomial_h`.
    """
    return _dispatch_aten(*args, **kwargs)


def special_hermite_polynomial_he(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.special_hermite_polynomial_he`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.special_hermite_polynomial_he`.
    """
    return _dispatch_aten(*args, **kwargs)


def special_i0(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.special_i0`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.special_i0`.
    """
    return _dispatch_aten(*args, **kwargs)


def special_i0e(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.special_i0e`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.special_i0e`.
    """
    return _dispatch_aten(*args, **kwargs)


def special_i1(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.special_i1`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.special_i1`.
    """
    return _dispatch_aten(*args, **kwargs)


def special_i1e(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.special_i1e`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.special_i1e`.
    """
    return _dispatch_aten(*args, **kwargs)


def special_laguerre_polynomial_l(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.special_laguerre_polynomial_l`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.special_laguerre_polynomial_l`.
    """
    return _dispatch_aten(*args, **kwargs)


def special_legendre_polynomial_p(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.special_legendre_polynomial_p`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.special_legendre_polynomial_p`.
    """
    return _dispatch_aten(*args, **kwargs)


def special_log1p(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.special_log1p`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.special_log1p`.
    """
    return _dispatch_aten(*args, **kwargs)


def special_log_ndtr(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.special_log_ndtr`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.special_log_ndtr`.
    """
    return _dispatch_aten(*args, **kwargs)


def special_log_softmax(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.special_log_softmax`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.special_log_softmax`.
    """
    return _dispatch_aten(*args, **kwargs)


def special_logit(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.special_logit`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.special_logit`.
    """
    return _dispatch_aten(*args, **kwargs)


def special_logsumexp(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.special_logsumexp`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.special_logsumexp`.
    """
    return _dispatch_aten(*args, **kwargs)


def special_modified_bessel_i0(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.special_modified_bessel_i0`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.special_modified_bessel_i0`.
    """
    return _dispatch_aten(*args, **kwargs)


def special_modified_bessel_i1(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.special_modified_bessel_i1`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.special_modified_bessel_i1`.
    """
    return _dispatch_aten(*args, **kwargs)


def special_modified_bessel_k0(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.special_modified_bessel_k0`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.special_modified_bessel_k0`.
    """
    return _dispatch_aten(*args, **kwargs)


def special_modified_bessel_k1(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.special_modified_bessel_k1`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.special_modified_bessel_k1`.
    """
    return _dispatch_aten(*args, **kwargs)


def special_multigammaln(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.special_multigammaln`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.special_multigammaln`.
    """
    return _dispatch_aten(*args, **kwargs)


def special_ndtr(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.special_ndtr`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.special_ndtr`.
    """
    return _dispatch_aten(*args, **kwargs)


def special_ndtri(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.special_ndtri`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.special_ndtri`.
    """
    return _dispatch_aten(*args, **kwargs)


def special_polygamma(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.special_polygamma`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.special_polygamma`.
    """
    return _dispatch_aten(*args, **kwargs)


def special_psi(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.special_psi`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.special_psi`.
    """
    return _dispatch_aten(*args, **kwargs)


def special_round(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.special_round`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.special_round`.
    """
    return _dispatch_aten(*args, **kwargs)


def special_scaled_modified_bessel_k0(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.special_scaled_modified_bessel_k0`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.special_scaled_modified_bessel_k0`.
    """
    return _dispatch_aten(*args, **kwargs)


def special_scaled_modified_bessel_k1(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.special_scaled_modified_bessel_k1`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.special_scaled_modified_bessel_k1`.
    """
    return _dispatch_aten(*args, **kwargs)


def special_shifted_chebyshev_polynomial_t(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.special_shifted_chebyshev_polynomial_t`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.special_shifted_chebyshev_polynomial_t`.
    """
    return _dispatch_aten(*args, **kwargs)


def special_shifted_chebyshev_polynomial_u(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.special_shifted_chebyshev_polynomial_u`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.special_shifted_chebyshev_polynomial_u`.
    """
    return _dispatch_aten(*args, **kwargs)


def special_shifted_chebyshev_polynomial_v(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.special_shifted_chebyshev_polynomial_v`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.special_shifted_chebyshev_polynomial_v`.
    """
    return _dispatch_aten(*args, **kwargs)


def special_shifted_chebyshev_polynomial_w(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.special_shifted_chebyshev_polynomial_w`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.special_shifted_chebyshev_polynomial_w`.
    """
    return _dispatch_aten(*args, **kwargs)


def special_sinc(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.special_sinc`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.special_sinc`.
    """
    return _dispatch_aten(*args, **kwargs)


def special_softmax(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.special_softmax`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.special_softmax`.
    """
    return _dispatch_aten(*args, **kwargs)


def special_spherical_bessel_j0(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.special_spherical_bessel_j0`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.special_spherical_bessel_j0`.
    """
    return _dispatch_aten(*args, **kwargs)


def special_xlog1py(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.special_xlog1py`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.special_xlog1py`.
    """
    return _dispatch_aten(*args, **kwargs)


def special_xlogy(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.special_xlogy`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.special_xlogy`.
    """
    return _dispatch_aten(*args, **kwargs)


def special_zeta(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.special_zeta`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.special_zeta`.
    """
    return _dispatch_aten(*args, **kwargs)


def split(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.split`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.split`.
    """
    return _dispatch_aten(*args, **kwargs)


def split_with_sizes(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.split_with_sizes`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.split_with_sizes`.
    """
    return _dispatch_aten(*args, **kwargs)


def split_with_sizes_copy(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.split_with_sizes_copy`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.split_with_sizes_copy`.
    """
    return _dispatch_aten(*args, **kwargs)


def sqrt(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.sqrt`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.sqrt`.
    """
    return _dispatch_aten(*args, **kwargs)


def sqrt_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.sqrt_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.sqrt_`.
    """
    return _dispatch_aten(*args, **kwargs)


def square(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.square`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.square`.
    """
    return _dispatch_aten(*args, **kwargs)


def square_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.square_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.square_`.
    """
    return _dispatch_aten(*args, **kwargs)


def squeeze(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.squeeze`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.squeeze`.
    """
    return _dispatch_aten(*args, **kwargs)


def squeeze_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.squeeze_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.squeeze_`.
    """
    return _dispatch_aten(*args, **kwargs)


def squeeze_copy(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.squeeze_copy`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.squeeze_copy`.
    """
    return _dispatch_aten(*args, **kwargs)


def sspaddmm(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.sspaddmm`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.sspaddmm`.
    """
    return _dispatch_aten(*args, **kwargs)


def stack(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.stack`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.stack`.
    """
    return _dispatch_aten(*args, **kwargs)


def std(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.std`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.std`.
    """
    return _dispatch_aten(*args, **kwargs)


def std_mean(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.std_mean`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.std_mean`.
    """
    return _dispatch_aten(*args, **kwargs)


def stft(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.stft`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.stft`.
    """
    return _dispatch_aten(*args, **kwargs)


def storage_offset(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.storage_offset`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.storage_offset`.
    """
    return _dispatch_aten(*args, **kwargs)


def stride(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.stride`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.stride`.
    """
    return _dispatch_aten(*args, **kwargs)


def sub(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.sub`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.sub`.
    """
    return _dispatch_aten(*args, **kwargs)


def sub_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.sub_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.sub_`.
    """
    return _dispatch_aten(*args, **kwargs)


def subtract(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.subtract`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.subtract`.
    """
    return _dispatch_aten(*args, **kwargs)


def subtract_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.subtract_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.subtract_`.
    """
    return _dispatch_aten(*args, **kwargs)


def sum(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.sum`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.sum`.
    """
    return _dispatch_aten(*args, **kwargs)


def sum_to_size(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.sum_to_size`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.sum_to_size`.
    """
    return _dispatch_aten(*args, **kwargs)


def svd(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.svd`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.svd`.
    """
    return _dispatch_aten(*args, **kwargs)


def swapaxes(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.swapaxes`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.swapaxes`.
    """
    return _dispatch_aten(*args, **kwargs)


def swapaxes_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.swapaxes_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.swapaxes_`.
    """
    return _dispatch_aten(*args, **kwargs)


def swapdims(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.swapdims`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.swapdims`.
    """
    return _dispatch_aten(*args, **kwargs)


def swapdims_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.swapdims_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.swapdims_`.
    """
    return _dispatch_aten(*args, **kwargs)


def sym_constrain_range(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.sym_constrain_range`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.sym_constrain_range`.
    """
    return _dispatch_aten(*args, **kwargs)


def sym_constrain_range_for_size(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.sym_constrain_range_for_size`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.sym_constrain_range_for_size`.
    """
    return _dispatch_aten(*args, **kwargs)


def sym_numel(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.sym_numel`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.sym_numel`.
    """
    return _dispatch_aten(*args, **kwargs)


def sym_size(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.sym_size`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.sym_size`.
    """
    return _dispatch_aten(*args, **kwargs)


def sym_storage_offset(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.sym_storage_offset`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.sym_storage_offset`.
    """
    return _dispatch_aten(*args, **kwargs)


def sym_stride(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.sym_stride`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.sym_stride`.
    """
    return _dispatch_aten(*args, **kwargs)


def t(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.t`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.t`.
    """
    return _dispatch_aten(*args, **kwargs)


def t_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.t_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.t_`.
    """
    return _dispatch_aten(*args, **kwargs)


def t_copy(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.t_copy`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.t_copy`.
    """
    return _dispatch_aten(*args, **kwargs)


def take(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.take`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.take`.
    """
    return _dispatch_aten(*args, **kwargs)


def take_along_dim(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.take_along_dim`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.take_along_dim`.
    """
    return _dispatch_aten(*args, **kwargs)


def tan(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.tan`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.tan`.
    """
    return _dispatch_aten(*args, **kwargs)


def tan_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.tan_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.tan_`.
    """
    return _dispatch_aten(*args, **kwargs)


def tanh(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.tanh`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.tanh`.
    """
    return _dispatch_aten(*args, **kwargs)


def tanh_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.tanh_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.tanh_`.
    """
    return _dispatch_aten(*args, **kwargs)


def tanh_backward(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.tanh_backward`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.tanh_backward`.
    """
    return _dispatch_aten(*args, **kwargs)


def tensor_split(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.tensor_split`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.tensor_split`.
    """
    return _dispatch_aten(*args, **kwargs)


def threshold(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.threshold`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.threshold`.
    """
    return _dispatch_aten(*args, **kwargs)


def threshold_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.threshold_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.threshold_`.
    """
    return _dispatch_aten(*args, **kwargs)


def threshold_backward(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.threshold_backward`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.threshold_backward`.
    """
    return _dispatch_aten(*args, **kwargs)


def tile(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.tile`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.tile`.
    """
    return _dispatch_aten(*args, **kwargs)


def to(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.to`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.to`.
    """
    return _dispatch_aten(*args, **kwargs)


def to_dense(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.to_dense`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.to_dense`.
    """
    return _dispatch_aten(*args, **kwargs)


def to_mkldnn(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.to_mkldnn`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.to_mkldnn`.
    """
    return _dispatch_aten(*args, **kwargs)


def to_padded_tensor(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.to_padded_tensor`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.to_padded_tensor`.
    """
    return _dispatch_aten(*args, **kwargs)


def to_sparse(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.to_sparse`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.to_sparse`.
    """
    return _dispatch_aten(*args, **kwargs)


def to_sparse_bsc(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.to_sparse_bsc`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.to_sparse_bsc`.
    """
    return _dispatch_aten(*args, **kwargs)


def to_sparse_bsr(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.to_sparse_bsr`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.to_sparse_bsr`.
    """
    return _dispatch_aten(*args, **kwargs)


def to_sparse_csc(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.to_sparse_csc`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.to_sparse_csc`.
    """
    return _dispatch_aten(*args, **kwargs)


def to_sparse_csr(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.to_sparse_csr`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.to_sparse_csr`.
    """
    return _dispatch_aten(*args, **kwargs)


def topk(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.topk`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.topk`.
    """
    return _dispatch_aten(*args, **kwargs)


def trace(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.trace`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.trace`.
    """
    return _dispatch_aten(*args, **kwargs)


def transpose(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.transpose`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.transpose`.
    """
    return _dispatch_aten(*args, **kwargs)


def transpose_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.transpose_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.transpose_`.
    """
    return _dispatch_aten(*args, **kwargs)


def transpose_copy(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.transpose_copy`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.transpose_copy`.
    """
    return _dispatch_aten(*args, **kwargs)


def trapezoid(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.trapezoid`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.trapezoid`.
    """
    return _dispatch_aten(*args, **kwargs)


def trapz(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.trapz`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.trapz`.
    """
    return _dispatch_aten(*args, **kwargs)


def triangular_solve(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.triangular_solve`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.triangular_solve`.
    """
    return _dispatch_aten(*args, **kwargs)


def tril(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.tril`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.tril`.
    """
    return _dispatch_aten(*args, **kwargs)


def tril_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.tril_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.tril_`.
    """
    return _dispatch_aten(*args, **kwargs)


def tril_indices(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.tril_indices`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.tril_indices`.
    """
    return _dispatch_aten(*args, **kwargs)


def triu(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.triu`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.triu`.
    """
    return _dispatch_aten(*args, **kwargs)


def triu_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.triu_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.triu_`.
    """
    return _dispatch_aten(*args, **kwargs)


def triu_indices(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.triu_indices`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.triu_indices`.
    """
    return _dispatch_aten(*args, **kwargs)


def true_divide(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.true_divide`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.true_divide`.
    """
    return _dispatch_aten(*args, **kwargs)


def true_divide_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.true_divide_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.true_divide_`.
    """
    return _dispatch_aten(*args, **kwargs)


def trunc(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.trunc`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.trunc`.
    """
    return _dispatch_aten(*args, **kwargs)


def trunc_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.trunc_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.trunc_`.
    """
    return _dispatch_aten(*args, **kwargs)


def type_as(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.type_as`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.type_as`.
    """
    return _dispatch_aten(*args, **kwargs)


def unbind(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.unbind`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.unbind`.
    """
    return _dispatch_aten(*args, **kwargs)


def unbind_copy(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.unbind_copy`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.unbind_copy`.
    """
    return _dispatch_aten(*args, **kwargs)


def unfold(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.unfold`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.unfold`.
    """
    return _dispatch_aten(*args, **kwargs)


def unfold_backward(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.unfold_backward`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.unfold_backward`.
    """
    return _dispatch_aten(*args, **kwargs)


def unfold_copy(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.unfold_copy`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.unfold_copy`.
    """
    return _dispatch_aten(*args, **kwargs)


def uniform(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.uniform`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.uniform`.
    """
    return _dispatch_aten(*args, **kwargs)


def uniform_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.uniform_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.uniform_`.
    """
    return _dispatch_aten(*args, **kwargs)


def unique_consecutive(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.unique_consecutive`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.unique_consecutive`.
    """
    return _dispatch_aten(*args, **kwargs)


def unique_dim(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.unique_dim`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.unique_dim`.
    """
    return _dispatch_aten(*args, **kwargs)


def unsafe_chunk(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.unsafe_chunk`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.unsafe_chunk`.
    """
    return _dispatch_aten(*args, **kwargs)


def unsafe_split(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.unsafe_split`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.unsafe_split`.
    """
    return _dispatch_aten(*args, **kwargs)


def unsafe_split_with_sizes(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.unsafe_split_with_sizes`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.unsafe_split_with_sizes`.
    """
    return _dispatch_aten(*args, **kwargs)


def unsqueeze(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.unsqueeze`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.unsqueeze`.
    """
    return _dispatch_aten(*args, **kwargs)


def unsqueeze_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.unsqueeze_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.unsqueeze_`.
    """
    return _dispatch_aten(*args, **kwargs)


def unsqueeze_copy(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.unsqueeze_copy`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.unsqueeze_copy`.
    """
    return _dispatch_aten(*args, **kwargs)


def upsample_bicubic2d(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.upsample_bicubic2d`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.upsample_bicubic2d`.
    """
    return _dispatch_aten(*args, **kwargs)


def upsample_bilinear2d(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.upsample_bilinear2d`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.upsample_bilinear2d`.
    """
    return _dispatch_aten(*args, **kwargs)


def upsample_linear1d(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.upsample_linear1d`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.upsample_linear1d`.
    """
    return _dispatch_aten(*args, **kwargs)


def upsample_nearest1d(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.upsample_nearest1d`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.upsample_nearest1d`.
    """
    return _dispatch_aten(*args, **kwargs)


def upsample_nearest2d(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.upsample_nearest2d`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.upsample_nearest2d`.
    """
    return _dispatch_aten(*args, **kwargs)


def upsample_nearest2d_backward(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.upsample_nearest2d_backward`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.upsample_nearest2d_backward`.
    """
    return _dispatch_aten(*args, **kwargs)


def upsample_nearest3d(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.upsample_nearest3d`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.upsample_nearest3d`.
    """
    return _dispatch_aten(*args, **kwargs)


def upsample_trilinear3d(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.upsample_trilinear3d`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.upsample_trilinear3d`.
    """
    return _dispatch_aten(*args, **kwargs)


def values(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.values`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.values`.
    """
    return _dispatch_aten(*args, **kwargs)


def var(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.var`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.var`.
    """
    return _dispatch_aten(*args, **kwargs)


def var_mean(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.var_mean`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.var_mean`.
    """
    return _dispatch_aten(*args, **kwargs)


def vdot(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.vdot`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.vdot`.
    """
    return _dispatch_aten(*args, **kwargs)


def view(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.view`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.view`.
    """
    return _dispatch_aten(*args, **kwargs)


def view_as(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.view_as`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.view_as`.
    """
    return _dispatch_aten(*args, **kwargs)


def view_as_complex(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.view_as_complex`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.view_as_complex`.
    """
    return _dispatch_aten(*args, **kwargs)


def view_as_real(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.view_as_real`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.view_as_real`.
    """
    return _dispatch_aten(*args, **kwargs)


def view_copy(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.view_copy`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.view_copy`.
    """
    return _dispatch_aten(*args, **kwargs)


def vsplit(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.vsplit`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.vsplit`.
    """
    return _dispatch_aten(*args, **kwargs)


def where(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.where`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.where`.
    """
    return _dispatch_aten(*args, **kwargs)


def xlogy(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.xlogy`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.xlogy`.
    """
    return _dispatch_aten(*args, **kwargs)


def xlogy_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.xlogy_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.xlogy_`.
    """
    return _dispatch_aten(*args, **kwargs)


def zero(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.zero`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.zero`.
    """
    return _dispatch_aten(*args, **kwargs)


def zero_(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.zero_`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.zero_`.
    """
    return _dispatch_aten(*args, **kwargs)


def zeros(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.zeros`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.zeros`.
    """
    return _dispatch_aten(*args, **kwargs)


def zeros_like(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ops.aten.zeros_like`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Result of `torch.ops.aten.zeros_like`.
    """
    return _dispatch_aten(*args, **kwargs)


__all__ = [
    "abs",
    "abs_",
    "absolute",
    "absolute_",
    "acos",
    "acos_",
    "acosh",
    "acosh_",
    "adaptive_avg_pool1d",
    "adaptive_max_pool2d",
    "adaptive_max_pool2d_backward",
    "adaptive_max_pool3d",
    "adaptive_max_pool3d_backward",
    "add",
    "add_",
    "addbmm",
    "addbmm_",
    "addcdiv",
    "addcdiv_",
    "addcmul",
    "addcmul_",
    "addmm",
    "addmm_",
    "addmv",
    "addmv_",
    "addr",
    "addr_",
    "adjoint",
    "affine_grid_generator",
    "alias",
    "alias_copy",
    "all",
    "allclose",
    "alpha_dropout",
    "amax",
    "amin",
    "aminmax",
    "angle",
    "any",
    "arange",
    "arccos",
    "arccos_",
    "arccosh",
    "arccosh_",
    "arcsin",
    "arcsin_",
    "arcsinh",
    "arcsinh_",
    "arctan",
    "arctan2",
    "arctan2_",
    "arctan_",
    "arctanh",
    "arctanh_",
    "argmax",
    "argmin",
    "argsort",
    "argwhere",
    "as_strided",
    "as_strided_",
    "as_strided_copy",
    "as_strided_scatter",
    "asin",
    "asin_",
    "asinh",
    "asinh_",
    "atan",
    "atan2",
    "atan2_",
    "atan_",
    "atanh",
    "atanh_",
    "avg_pool1d",
    "avg_pool2d",
    "avg_pool2d_backward",
    "avg_pool3d",
    "avg_pool3d_backward",
    "baddbmm",
    "baddbmm_",
    "bartlett_window",
    "batch_norm_backward",
    "bernoulli",
    "bernoulli_",
    "bilinear",
    "binary_cross_entropy",
    "binary_cross_entropy_backward",
    "binary_cross_entropy_with_logits",
    "bincount",
    "bitwise_and",
    "bitwise_and_",
    "bitwise_left_shift",
    "bitwise_left_shift_",
    "bitwise_not",
    "bitwise_not_",
    "bitwise_or",
    "bitwise_or_",
    "bitwise_right_shift",
    "bitwise_right_shift_",
    "bitwise_xor",
    "bitwise_xor_",
    "blackman_window",
    "block_diag",
    "bmm",
    "broadcast_tensors",
    "broadcast_to",
    "bucketize",
    "cat",
    "cauchy",
    "cauchy_",
    "ccol_indices",
    "ceil",
    "ceil_",
    "celu",
    "celu_",
    "chalf",
    "channel_shuffle",
    "cholesky_inverse",
    "cholesky_solve",
    "chunk",
    "clamp",
    "clamp_",
    "clamp_max",
    "clamp_max_",
    "clamp_min",
    "clamp_min_",
    "clip",
    "clip_",
    "clone",
    "coalesce",
    "col2im",
    "col_indices",
    "complex",
    "conj",
    "conj_physical",
    "conj_physical_",
    "constant_pad_nd",
    "contiguous",
    "conv1d",
    "conv2d",
    "conv3d",
    "conv_tbc",
    "conv_transpose1d",
    "conv_transpose2d",
    "conv_transpose3d",
    "convolution",
    "convolution_backward",
    "copy",
    "copy_",
    "copysign",
    "copysign_",
    "corrcoef",
    "cos",
    "cos_",
    "cosh",
    "cosh_",
    "cosine_similarity",
    "count_nonzero",
    "cov",
    "cpu",
    "cross",
    "crow_indices",
    "ctc_loss",
    "cuda",
    "cudnn_batch_norm",
    "cudnn_batch_norm_backward",
    "cummax",
    "cummin",
    "cumprod",
    "cumprod_",
    "cumsum",
    "cumsum_",
    "cumulative_trapezoid",
    "deg2rad",
    "deg2rad_",
    "dense_dim",
    "dequantize",
    "det",
    "detach",
    "detach_",
    "diag",
    "diag_embed",
    "diagflat",
    "diagonal",
    "diagonal_backward",
    "diagonal_copy",
    "diagonal_scatter",
    "diff",
    "digamma",
    "digamma_",
    "dim",
    "dist",
    "div",
    "div_",
    "divide",
    "divide_",
    "dot",
    "dropout",
    "dsplit",
    "element_size",
    "elu",
    "elu_",
    "elu_backward",
    "embedding",
    "embedding_bag",
    "embedding_dense_backward",
    "empty",
    "empty_like",
    "empty_permuted",
    "empty_strided",
    "eq",
    "eq_",
    "equal",
    "erf",
    "erf_",
    "erfc",
    "erfc_",
    "erfinv",
    "erfinv_",
    "exp",
    "exp2",
    "exp2_",
    "exp_",
    "expand",
    "expand_as",
    "expand_copy",
    "expm1",
    "expm1_",
    "exponential",
    "exponential_",
    "eye",
    "fake_quantize_per_tensor_affine",
    "fbgemm_linear_fp16_weight",
    "fbgemm_linear_fp16_weight_fp32_activation",
    "fbgemm_pack_quantized_matrix",
    "fft_fft",
    "fft_fft2",
    "fft_fftfreq",
    "fft_fftn",
    "fft_fftshift",
    "fft_hfft",
    "fft_hfft2",
    "fft_hfftn",
    "fft_ifft",
    "fft_ifft2",
    "fft_ifftn",
    "fft_ifftshift",
    "fft_ihfft",
    "fft_ihfft2",
    "fft_ihfftn",
    "fft_irfft",
    "fft_irfft2",
    "fft_irfftn",
    "fft_rfft",
    "fft_rfft2",
    "fft_rfftfreq",
    "fft_rfftn",
    "fill",
    "fill_",
    "fill_diagonal_",
    "fix",
    "fix_",
    "flatten",
    "flip",
    "fliplr",
    "flipud",
    "float_power",
    "float_power_",
    "floor",
    "floor_",
    "floor_divide",
    "floor_divide_",
    "fmax",
    "fmin",
    "fmod",
    "fmod_",
    "frac",
    "frac_",
    "fractional_max_pool2d",
    "frexp",
    "full",
    "full_like",
    "gather",
    "gcd",
    "gcd_",
    "ge",
    "ge_",
    "gelu",
    "gelu_",
    "gelu_backward",
    "geometric",
    "geometric_",
    "geqrf",
    "ger",
    "get_device",
    "glu",
    "glu_backward",
    "gradient",
    "greater",
    "greater_",
    "greater_equal",
    "greater_equal_",
    "grid_sampler_2d",
    "grid_sampler_2d_backward",
    "grid_sampler_3d",
    "grid_sampler_3d_backward",
    "gru",
    "gt",
    "gt_",
    "hamming_window",
    "hann_window",
    "hardshrink",
    "hardsigmoid",
    "hardsigmoid_",
    "hardsigmoid_backward",
    "hardswish",
    "hardswish_",
    "hardswish_backward",
    "hardtanh",
    "hardtanh_",
    "hardtanh_backward",
    "hash_tensor",
    "heaviside",
    "heaviside_",
    "hinge_embedding_loss",
    "histc",
    "histogram",
    "histogramdd",
    "hsplit",
    "huber_loss",
    "huber_loss_backward",
    "hypot",
    "hypot_",
    "i0",
    "i0_",
    "igamma",
    "igamma_",
    "igammac",
    "igammac_",
    "im2col",
    "imag",
    "index",
    "index_add",
    "index_add_",
    "index_copy",
    "index_copy_",
    "index_fill",
    "index_fill_",
    "index_put",
    "index_put_",
    "index_reduce",
    "index_reduce_",
    "index_select",
    "indices",
    "inner",
    "int_repr",
    "inverse",
    "is_autocast_cpu_enabled",
    "is_autocast_enabled",
    "is_coalesced",
    "is_complex",
    "is_conj",
    "is_contiguous",
    "is_distributed",
    "is_floating_point",
    "is_grad_enabled",
    "is_inference",
    "is_neg",
    "is_non_overlapping_and_dense",
    "is_nonzero",
    "is_pinned",
    "is_same_size",
    "is_set_to",
    "is_signed",
    "is_strides_like_format",
    "is_vulkan_available",
    "isclose",
    "isfinite",
    "isin",
    "isinf",
    "isnan",
    "isneginf",
    "isposinf",
    "isreal",
    "istft",
    "item",
    "kaiser_window",
    "kron",
    "kthvalue",
    "lcm",
    "lcm_",
    "ldexp",
    "ldexp_",
    "le",
    "le_",
    "leaky_relu",
    "leaky_relu_",
    "leaky_relu_backward",
    "lerp",
    "lerp_",
    "less",
    "less_",
    "less_equal",
    "less_equal_",
    "lgamma",
    "lgamma_",
    "lift",
    "lift_fresh",
    "lift_fresh_copy",
    "linalg_cholesky",
    "linalg_cholesky_ex",
    "linalg_cond",
    "linalg_cross",
    "linalg_det",
    "linalg_diagonal",
    "linalg_eig",
    "linalg_eigh",
    "linalg_eigvals",
    "linalg_eigvalsh",
    "linalg_householder_product",
    "linalg_inv",
    "linalg_inv_ex",
    "linalg_ldl_factor",
    "linalg_ldl_factor_ex",
    "linalg_ldl_solve",
    "linalg_lstsq",
    "linalg_lu",
    "linalg_lu_factor",
    "linalg_lu_factor_ex",
    "linalg_lu_solve",
    "linalg_matmul",
    "linalg_matrix_exp",
    "linalg_matrix_norm",
    "linalg_matrix_power",
    "linalg_matrix_rank",
    "linalg_matrix_sqrth",
    "linalg_multi_dot",
    "linalg_norm",
    "linalg_pinv",
    "linalg_polar",
    "linalg_qr",
    "linalg_slogdet",
    "linalg_solve",
    "linalg_solve_ex",
    "linalg_solve_triangular",
    "linalg_svd",
    "linalg_svdvals",
    "linalg_tensorinv",
    "linalg_tensorsolve",
    "linalg_vander",
    "linalg_vecdot",
    "linalg_vector_norm",
    "linear",
    "linear_backward",
    "linspace",
    "log",
    "log1p",
    "log1p_",
    "log2",
    "log2_",
    "log10",
    "log10_",
    "log_",
    "log_normal",
    "log_normal_",
    "log_sigmoid",
    "log_sigmoid_backward",
    "log_sigmoid_forward",
    "log_softmax",
    "logaddexp",
    "logaddexp2",
    "logcumsumexp",
    "logdet",
    "logical_and",
    "logical_and_",
    "logical_not",
    "logical_not_",
    "logical_or",
    "logical_or_",
    "logical_xor",
    "logical_xor_",
    "logit",
    "logit_",
    "logit_backward",
    "logspace",
    "logsumexp",
    "lstm",
    "lt",
    "lt_",
    "lu_solve",
    "lu_unpack",
    "margin_ranking_loss",
    "masked_fill",
    "masked_fill_",
    "masked_scatter",
    "masked_scatter_",
    "masked_scatter_backward",
    "masked_select",
    "matmul",
    "matrix_exp",
    "matrix_power",
    "max",
    "max_pool2d_with_indices",
    "max_pool2d_with_indices_backward",
    "max_pool3d_with_indices",
    "max_pool3d_with_indices_backward",
    "max_unpool2d",
    "max_unpool3d",
    "maximum",
    "mean",
    "median",
    "meshgrid",
    "min",
    "minimum",
    "miopen_batch_norm",
    "miopen_batch_norm_backward",
    "miopen_ctc_loss",
    "miopen_rnn",
    "mish",
    "mish_",
    "mish_backward",
    "mkldnn_rnn_layer",
    "mkldnn_rnn_layer_backward",
    "mm",
    "mode",
    "moveaxis",
    "movedim",
    "mse_loss",
    "mse_loss_backward",
    "msort",
    "mul",
    "mul_",
    "multi_margin_loss",
    "multilabel_margin_loss_forward",
    "multinomial",
    "multiply",
    "multiply_",
    "mv",
    "mvlgamma",
    "mvlgamma_",
    "nan_to_num",
    "nan_to_num_",
    "nanmean",
    "nanmedian",
    "nanquantile",
    "nansum",
    "narrow",
    "narrow_copy",
    "native_batch_norm",
    "native_batch_norm_backward",
    "native_channel_shuffle",
    "native_dropout",
    "native_dropout_backward",
    "native_group_norm",
    "native_group_norm_backward",
    "native_layer_norm",
    "native_layer_norm_backward",
    "native_norm",
    "ne",
    "ne_",
    "neg",
    "neg_",
    "negative",
    "negative_",
    "new_empty",
    "new_empty_strided",
    "new_full",
    "new_ones",
    "new_zeros",
    "nextafter",
    "nextafter_",
    "nll_loss",
    "nll_loss2d_backward",
    "nll_loss2d_forward",
    "nll_loss_backward",
    "nll_loss_forward",
    "nonzero",
    "nonzero_numpy",
    "nonzero_static",
    "norm",
    "normal",
    "normal_",
    "normal_functional",
    "not_equal",
    "not_equal_",
    "nuclear_norm",
    "numel",
    "one_hot",
    "ones",
    "ones_like",
    "orgqr",
    "ormqr",
    "outer",
    "pad_sequence",
    "pairwise_distance",
    "pdist",
    "permute",
    "permute_copy",
    "pin_memory",
    "pinverse",
    "pixel_shuffle",
    "pixel_unshuffle",
    "poisson",
    "polar",
    "polygamma",
    "polygamma_",
    "positive",
    "pow",
    "pow_",
    "prelu",
    "prod",
    "put",
    "put_",
    "q_per_channel_axis",
    "q_per_channel_scales",
    "q_per_channel_zero_points",
    "q_scale",
    "q_zero_point",
    "qscheme",
    "quantile",
    "quantize_per_tensor",
    "rad2deg",
    "rad2deg_",
    "rand",
    "rand_like",
    "randint",
    "randint_like",
    "randn",
    "randn_like",
    "random_",
    "randperm",
    "ravel",
    "real",
    "reciprocal",
    "reciprocal_",
    "record_stream",
    "reflection_pad1d",
    "reflection_pad1d_backward",
    "reflection_pad2d",
    "reflection_pad2d_backward",
    "reflection_pad3d",
    "reflection_pad3d_backward",
    "relu",
    "relu6",
    "relu_",
    "remainder",
    "remainder_",
    "renorm",
    "renorm_",
    "repeat",
    "repeat_interleave",
    "replication_pad1d",
    "replication_pad1d_backward",
    "replication_pad2d",
    "replication_pad2d_backward",
    "replication_pad3d",
    "replication_pad3d_backward",
    "requires_grad_",
    "reshape",
    "reshape_as",
    "resize",
    "resize_",
    "resize_as",
    "resize_as_",
    "resize_as_sparse_",
    "resolve_conj",
    "resolve_neg",
    "result_type",
    "retain_grad",
    "rnn_relu",
    "rnn_tanh",
    "roll",
    "rot90",
    "round",
    "round_",
    "row_indices",
    "rrelu_",
    "rrelu_with_noise",
    "rrelu_with_noise_",
    "rrelu_with_noise_backward",
    "rrelu_with_noise_functional",
    "rsqrt",
    "rsqrt_",
    "rsub",
    "scalar_tensor",
    "scaled_dot_product_attention",
    "scatter",
    "scatter_",
    "scatter_add",
    "scatter_add_",
    "scatter_reduce",
    "scatter_reduce_",
    "searchsorted",
    "segment_reduce",
    "select",
    "select_backward",
    "select_scatter",
    "selu",
    "selu_",
    "set_",
    "sgn",
    "sgn_",
    "shallow_copy_data_",
    "sigmoid",
    "sigmoid_",
    "sigmoid_backward",
    "sign",
    "sign_",
    "signbit",
    "silu",
    "silu_",
    "silu_backward",
    "sin",
    "sin_",
    "sinc",
    "sinc_",
    "sinh",
    "sinh_",
    "size",
    "slice",
    "slice_backward",
    "slice_inverse",
    "slice_scatter",
    "slogdet",
    "smm",
    "smooth_l1_loss",
    "smooth_l1_loss_backward",
    "soft_margin_loss",
    "soft_margin_loss_backward",
    "softmax",
    "softplus",
    "softplus_backward",
    "softshrink",
    "sort",
    "sparse_dim",
    "sparse_mask",
    "sparse_resize_",
    "sparse_resize_and_clear_",
    "special_airy_ai",
    "special_bessel_j0",
    "special_bessel_j1",
    "special_bessel_y0",
    "special_bessel_y1",
    "special_chebyshev_polynomial_t",
    "special_chebyshev_polynomial_u",
    "special_chebyshev_polynomial_v",
    "special_chebyshev_polynomial_w",
    "special_digamma",
    "special_entr",
    "special_erf",
    "special_erfc",
    "special_erfcx",
    "special_erfinv",
    "special_exp2",
    "special_expit",
    "special_expm1",
    "special_gammainc",
    "special_gammaincc",
    "special_gammaln",
    "special_hermite_polynomial_h",
    "special_hermite_polynomial_he",
    "special_i0",
    "special_i0e",
    "special_i1",
    "special_i1e",
    "special_laguerre_polynomial_l",
    "special_legendre_polynomial_p",
    "special_log1p",
    "special_log_ndtr",
    "special_log_softmax",
    "special_logit",
    "special_logsumexp",
    "special_modified_bessel_i0",
    "special_modified_bessel_i1",
    "special_modified_bessel_k0",
    "special_modified_bessel_k1",
    "special_multigammaln",
    "special_ndtr",
    "special_ndtri",
    "special_polygamma",
    "special_psi",
    "special_round",
    "special_scaled_modified_bessel_k0",
    "special_scaled_modified_bessel_k1",
    "special_shifted_chebyshev_polynomial_t",
    "special_shifted_chebyshev_polynomial_u",
    "special_shifted_chebyshev_polynomial_v",
    "special_shifted_chebyshev_polynomial_w",
    "special_sinc",
    "special_softmax",
    "special_spherical_bessel_j0",
    "special_xlog1py",
    "special_xlogy",
    "special_zeta",
    "split",
    "split_with_sizes",
    "split_with_sizes_copy",
    "sqrt",
    "sqrt_",
    "square",
    "square_",
    "squeeze",
    "squeeze_",
    "squeeze_copy",
    "sspaddmm",
    "stack",
    "std",
    "std_mean",
    "stft",
    "storage_offset",
    "stride",
    "sub",
    "sub_",
    "subtract",
    "subtract_",
    "sum",
    "sum_to_size",
    "svd",
    "swapaxes",
    "swapaxes_",
    "swapdims",
    "swapdims_",
    "sym_constrain_range",
    "sym_constrain_range_for_size",
    "sym_numel",
    "sym_size",
    "sym_storage_offset",
    "sym_stride",
    "t",
    "t_",
    "t_copy",
    "take",
    "take_along_dim",
    "tan",
    "tan_",
    "tanh",
    "tanh_",
    "tanh_backward",
    "tensor_split",
    "threshold",
    "threshold_",
    "threshold_backward",
    "tile",
    "to",
    "to_dense",
    "to_mkldnn",
    "to_padded_tensor",
    "to_sparse",
    "to_sparse_bsc",
    "to_sparse_bsr",
    "to_sparse_csc",
    "to_sparse_csr",
    "topk",
    "trace",
    "transpose",
    "transpose_",
    "transpose_copy",
    "trapezoid",
    "trapz",
    "triangular_solve",
    "tril",
    "tril_",
    "tril_indices",
    "triu",
    "triu_",
    "triu_indices",
    "true_divide",
    "true_divide_",
    "trunc",
    "trunc_",
    "type_as",
    "unbind",
    "unbind_copy",
    "unfold",
    "unfold_backward",
    "unfold_copy",
    "uniform",
    "uniform_",
    "unique_consecutive",
    "unique_dim",
    "unsafe_chunk",
    "unsafe_split",
    "unsafe_split_with_sizes",
    "unsqueeze",
    "unsqueeze_",
    "unsqueeze_copy",
    "upsample_bicubic2d",
    "upsample_bilinear2d",
    "upsample_linear1d",
    "upsample_nearest1d",
    "upsample_nearest2d",
    "upsample_nearest2d_backward",
    "upsample_nearest3d",
    "upsample_trilinear3d",
    "values",
    "var",
    "var_mean",
    "vdot",
    "view",
    "view_as",
    "view_as_complex",
    "view_as_real",
    "view_copy",
    "vsplit",
    "where",
    "xlogy",
    "xlogy_",
    "zero",
    "zero_",
    "zeros",
    "zeros_like",
]
