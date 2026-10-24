"""Autograd function module."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from zero_torch._types import OpArg, TensorLike

if TYPE_CHECKING:
    from zero_torch.tensor import Tensor


class Context:
    """Context object passed to forward and backward methods."""

    def __init__(self) -> None:
        """Initializes Context."""
        self.saved_tensors: tuple[TensorLike, ...] = ()

    def save_for_backward(self, *tensors: TensorLike) -> None:
        """Saves given tensors for a future call to backward.

        Args:
            *tensors: Tensors or scalars to save for the backward pass.
        """
        self.saved_tensors = tensors


class Function:
    """Base class to create custom autograd Functions."""

    @classmethod
    def apply(cls, *args: TensorLike, **kwargs: OpArg) -> Tensor:
        """Applies the custom autograd function.

        Args:
            *args: Positional input tensors or scalars.
            **kwargs: Keyword arguments forwarded to `forward`.

        Returns:
            Output tensor(s) produced by `forward`.
        """
        ctx = Context()

        from zero_torch.tensor import Tensor

        from .grad_mode import is_grad_enabled

        needs_grad = is_grad_enabled() and any(
            isinstance(arg, Tensor) and arg.requires_grad for arg in args
        )

        result = cls.forward(ctx, *args, **kwargs)

        if needs_grad:
            if isinstance(result, Tensor):
                fn = cls(ctx, args)
                fn.output_tensors = [result]
                result.grad_fn = fn
                result.requires_grad = True
            else:
                for idx, r in enumerate(result):
                    if isinstance(r, Tensor):
                        fn = cls(ctx, args)
                        fn.output_nr = idx
                        fn.output_tensors = [r]
                        r.grad_fn = fn
                        r.requires_grad = True

        return result

    def __init__(self, ctx: Context, args: Sequence[TensorLike]) -> None:
        """Initializes the function node instance.

        Args:
            ctx: The context object storing saved tensors.
            args: The input arguments passed to `apply`.
        """
        self.ctx: Context = ctx
        self.output_nr: int = 0
        self.output_tensors: list[Tensor] = []
        self.next_functions: list[tuple[Function | Tensor | None, int]] = []
        from zero_torch.tensor import Tensor

        for arg in args:
            if isinstance(arg, Tensor):
                if arg.requires_grad:
                    if arg.grad_fn is not None:
                        self.next_functions.append((arg.grad_fn, arg.grad_fn.output_nr))
                    else:
                        self.next_functions.append((arg, 0))
                else:
                    self.next_functions.append((None, 0))
            else:
                self.next_functions.append((None, 0))

    def __call__(
        self, *args: Tensor, **kwargs: OpArg
    ) -> Tensor | tuple[Tensor | None, ...] | None:
        """Calls the backward method on this autograd node.

        Args:
            *args: Upstream output gradients.
            **kwargs: Optional keyword arguments for backward.

        Returns:
            Computed input gradients.
        """
        return self.__class__.backward(self.ctx, *args, **kwargs)

    @staticmethod
    def forward(ctx: Context, *args: TensorLike, **kwargs: OpArg) -> Tensor:
        """Performs the forward operation.

        Args:
            ctx: The context object.
            *args: Positional input tensors or scalars.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Output tensor produced by the operation.
        """
        _ = (ctx, args, kwargs)
        raise NotImplementedError

    @staticmethod
    def backward(
        ctx: Context, *grad_outputs: Tensor
    ) -> Tensor | tuple[Tensor | None, ...] | None:
        """Defines the formula for differentiating the operation.

        Args:
            ctx: The context object.
            *grad_outputs: Gradients w.r.t. the forward outputs.

        Returns:
            Gradients w.r.t. the forward inputs.
        """
        _ = (ctx, grad_outputs)
        raise NotImplementedError
