"""Autograd function module."""

from typing import Any


class Context:
    """Context object passed to forward and backward methods."""

    def __init__(self):
        """Initializes Context."""
        self.saved_tensors: tuple[Any, ...] = ()

    def save_for_backward(self, *tensors: Any) -> None:
        """Saves given tensors for a future call to backward.

        Args:
            *tensors: Tensors to save.
        """
        self.saved_tensors = tensors


class Function:
    """Base class to create custom autograd Functions."""

    @classmethod
    def apply(cls, *args, **kwargs):
        """Applies the custom autograd function.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Any: The result of the forward pass.
        """
        ctx = Context()

        # Check if we need to track gradients
        from zero_torch.tensor import Tensor

        from .grad_mode import is_grad_enabled

        needs_grad = is_grad_enabled() and any(
            isinstance(arg, Tensor) and arg.requires_grad for arg in args
        )

        result = cls.forward(ctx, *args, **kwargs)

        if needs_grad:
            if isinstance(result, Tensor):
                result.grad_fn = cls(ctx, args)
                result.requires_grad = True
            elif isinstance(result, (tuple, list)):  # pragma: no cover
                for r in result:  # pragma: no cover
                    if isinstance(r, Tensor):  # pragma: no cover
                        r.grad_fn = cls(ctx, args)  # pragma: no cover
                        r.requires_grad = True  # pragma: no cover

        return result

    def __init__(self, ctx, args):
        """Initializes the function instance.

        Args:
            ctx: The context object.
            args: The arguments passed to apply.
        """
        self.ctx = ctx
        self.next_functions = []
        from zero_torch.tensor import Tensor

        for arg in args:
            if isinstance(arg, Tensor):
                if arg.requires_grad:
                    if arg.grad_fn is not None:
                        self.next_functions.append(
                            (arg.grad_fn, 0)
                        )  # assuming single output for now
                    else:
                        self.next_functions.append((arg, 0))  # leaf node
            else:
                self.next_functions.append((None, 0))  # pragma: no cover

    def __call__(self, *args, **kwargs):
        """Calls the backward method.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Any: The gradients.
        """
        return self.__class__.backward(self.ctx, *args, **kwargs)

    @staticmethod
    def forward(ctx: Context, *args, **kwargs) -> Any:
        """Performs the operation.

        Args:
            ctx (Context): The context object.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Raises:
            NotImplementedError: Must be implemented by subclass.
        """
        raise NotImplementedError

    @staticmethod
    def backward(ctx: Context, *grad_outputs: Any) -> Any:
        """Defines a formula for differentiating the operation.

        Args:
            ctx (Context): The context object.
            *grad_outputs: Gradients w.r.t. the outputs.

        Raises:
            NotImplementedError: Must be implemented by subclass.
        """
        raise NotImplementedError
