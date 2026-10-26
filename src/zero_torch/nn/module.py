"""NN Module System."""

from __future__ import annotations

import uuid
from collections.abc import Callable, Iterable, Iterator, Mapping
from types import TracebackType
from typing import TYPE_CHECKING, Generic, NamedTuple, TypeVar, overload

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import (
    Tensor as SwitcherooTensor,
)
from ml_switcheroo_compiler.core.tensor import (
    TensorConfig,
)
from ml_switcheroo_compiler.ir.core import LogicalNode

from zero_torch._types import (
    DTypeLike,
    ModuleDictArg,
    OpArg,
    ParamDictArg,
    Scalar,
    SliceIndexType,
)
from zero_torch.tensor import Tensor, _ArrayDataProtocol, _to_tensor
from zero_torch.tracing import ProxyTensor, _tracer

_H = TypeVar("_H")

if TYPE_CHECKING:
    from typing import ParamSpec, Protocol

    _P = ParamSpec("_P")
    _R_co = TypeVar("_R_co", covariant=True)

    class _HasForward(Protocol[_P, _R_co]):
        """Structural protocol for `Module` subclasses defining a `forward` method."""

        _forward_pre_hooks: dict[
            int,
            Callable[[Module, tuple[Tensor, ...]], tuple[Tensor, ...] | Tensor | None],
        ]
        _forward_hooks: dict[
            int, Callable[[Module, tuple[Tensor, ...], Tensor], Tensor | None]
        ]
        _backward_hooks: dict[
            int,
            Callable[
                [Module, tuple[Tensor | None, ...], tuple[Tensor | None, ...]],
                tuple[Tensor | None, ...] | None,
            ],
        ]

        def forward(self, *args: _P.args, **kwargs: _P.kwargs) -> _R_co:
            """Executes the forward pass of the module.

            Args:
                *args: Positional arguments for the forward pass.
                **kwargs: Keyword arguments for the forward pass.

            Returns:
                Output produced by the module.
            """


class IncompatibleKeys(NamedTuple):
    """Holds `missing_keys` and `unexpected_keys` returned by `Module.load_state_dict`."""

    missing_keys: list[str]
    unexpected_keys: list[str]


class RemovableHandle(Generic[_H]):
    """A handle which provides the capability to remove a hook from a dictionary."""

    def __init__(self, hooks_dict: dict[int, _H], hook_id: int) -> None:
        """Initializes a `RemovableHandle`.

        Args:
            hooks_dict: Dictionary mapping hook IDs to hook callables.
            hook_id: Unique integer identifier for the registered hook.
        """
        self._hooks_dict: dict[int, _H] = hooks_dict
        self.id: int = hook_id

    def remove(self) -> None:
        """Removes the hook from its parent dictionary if still present."""
        self._hooks_dict.pop(self.id, None)

    def __enter__(self) -> RemovableHandle[_H]:
        """Enters a context block managing this hook handle.

        Returns:
            This `RemovableHandle` instance.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Removes the hook upon exiting the context block.

        Args:
            exc_type: Exception type if raised.
            exc_val: Exception value if raised.
            exc_tb: Traceback object if raised.
        """
        _ = (exc_type, exc_val, exc_tb)
        self.remove()


class Parameter(Tensor):
    """A kind of Tensor that is to be considered a module parameter."""

    def __init__(
        self,
        data: OpArg | _ArrayDataProtocol = None,
        requires_grad: bool = True,
        *args: OpArg,
        **kwargs: OpArg,
    ) -> None:
        """Initializes a Parameter.

        Args:
            data: Parameter tensor data.
            requires_grad: Whether the parameter requires gradient.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        _ = (args, kwargs)
        super().__init__(data, requires_grad=requires_grad)
        self._param_id: str = "param_" + str(uuid.uuid4())

    @property
    def data(
        self,
    ) -> (
        SwitcherooTensor
        | ProxyTensor
        | _ArrayDataProtocol
        | list[Scalar]
        | Scalar
        | None
    ):
        """Gets the underlying parameter data or emits a ReadVariable node when tracing.

        Returns:
            The parameter data or traced `SwitcherooTensor`.
        """
        if _tracer.is_tracing:
            out_id = str(uuid.uuid4())
            node = LogicalNode(
                id=out_id,
                op_type="ReadVariable",
                inputs=[self._param_id],
                shape_metadata=self.shape,
            )
            _tracer.add_node(node)
            pt = ProxyTensor(id=out_id, shape=self.shape, dtype=str(self.dtype))
            dt = self.dtype if self.dtype is not None else config.default_float_dtype
            dev = (
                self._tensor.device
                if self._tensor is not None
                else config.default_device
            )
            return SwitcherooTensor(
                data=pt,
                config=TensorConfig(shape=self.shape, dtype=dt, device=dev),
            )
        return self._tensor.data if self._tensor is not None else None


class Module:
    """Base class for all neural network modules."""

    training: bool = True
    dump_patches: bool = False
    _version: int = 1
    _parameters: dict[str, Parameter] = {}
    _buffers: dict[str, Tensor] = {}
    _non_persistent_buffers_set: set[str] = set()
    _backward_pre_hooks: dict[int, str] = {}
    _backward_hooks: dict[
        int,
        Callable[
            [Module, tuple[Tensor | None, ...], tuple[Tensor | None, ...]],
            tuple[Tensor | None, ...] | None,
        ],
    ] = {}
    _is_full_backward_hook: bool | None = None
    _forward_hooks: dict[
        int, Callable[[Module, tuple[Tensor, ...], Tensor], Tensor | None]
    ] = {}
    _forward_hooks_with_kwargs: dict[int, bool] = {}
    _forward_hooks_always_called: dict[int, bool] = {}
    _forward_pre_hooks: dict[
        int, Callable[[Module, tuple[Tensor, ...]], tuple[Tensor, ...] | Tensor | None]
    ] = {}
    _forward_pre_hooks_with_kwargs: dict[int, bool] = {}
    _state_dict_hooks: dict[int, str] = {}
    _load_state_dict_pre_hooks: dict[int, str] = {}
    _state_dict_pre_hooks: dict[int, str] = {}
    _load_state_dict_post_hooks: dict[int, str] = {}
    _modules: dict[str, Module] = {}
    _next_hook_id: int = 0
    call_super_init: bool = False
    _compiled_call_impl: str | None = None
    T_destination: str = "T_destination"
    if not TYPE_CHECKING:
        forward: str = "_forward_unimplemented"

    def __init__(self, *args: OpArg, **kwargs: OpArg) -> None:
        """Initializes internal Module state.

        Args:
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        _ = (args, kwargs)
        self._modules = {}
        self._parameters = {}
        self._buffers = {}
        self._non_persistent_buffers_set = set()
        self._forward_pre_hooks = {}
        self._forward_hooks = {}
        self._backward_hooks = {}
        self._next_hook_id = 0
        self.training = True

    if TYPE_CHECKING:

        @overload
        def __call__(
            self: _HasForward[_P, _R_co], *args: _P.args, **kwargs: _P.kwargs
        ) -> _R_co:
            """Invokes the module's typed `forward` computation.

            Args:
                *args: Positional arguments for the forward pass.
                **kwargs: Keyword arguments for the forward pass.

            Returns:
                The output of `self.forward(*args, **kwargs)`.
            """

        @overload
        def __call__(self, *args: OpArg, **kwargs: OpArg) -> Tensor:
            """Invokes the base module's `forward` computation.

            Args:
                *args: Positional arguments for the forward pass.
                **kwargs: Keyword arguments for the forward pass.

            Returns:
                The output tensor.
            """

    def __call__(
        self: _HasForward[_P, _R_co], *args: _P.args, **kwargs: _P.kwargs
    ) -> _R_co:
        """Invokes the module's `forward` computation along with registered hooks.

        Args:
            *args: Positional arguments for the forward pass.
            **kwargs: Keyword arguments for the forward pass.

        Returns:
            The output of `self.forward(*args, **kwargs)`.
        """
        assert isinstance(self, Module)
        tensor_args: tuple[Tensor, ...] = tuple(
            a for a in args if isinstance(a, Tensor)
        )
        orig_tensors: tuple[Tensor, ...] = tensor_args
        for pre_hook in list(self._forward_pre_hooks.values()):
            pre_res = pre_hook(self, tensor_args)
            if pre_res is not None:
                tensor_args = pre_res if isinstance(pre_res, tuple) else (pre_res,)
                for orig_t, new_t in zip(orig_tensors, tensor_args):
                    orig_t._tensor = new_t._tensor

        result = self.forward(*args, **kwargs)
        if isinstance(result, Tensor):
            for hook in list(self._forward_hooks.values()):
                hook_res = hook(self, tensor_args, result)
                if isinstance(hook_res, Tensor):
                    result._tensor = hook_res._tensor
        return result

    def register_forward_pre_hook(
        self,
        hook: Callable[
            [Module, tuple[Tensor, ...]], tuple[Tensor, ...] | Tensor | None
        ],
        *,
        prepend: bool = False,
        with_kwargs: bool = False,
    ) -> RemovableHandle[
        Callable[[Module, tuple[Tensor, ...]], tuple[Tensor, ...] | Tensor | None]
    ]:
        """Registers a forward pre-hook on the module.

        Args:
            hook: Callable `(module, input) -> modified_input_or_None`.
            prepend: If True, prepends the hook before existing pre-hooks.
            with_kwargs: Whether the hook accepts keyword arguments.

        Returns:
            A `RemovableHandle` that can be used to remove the added hook.
        """
        _ = with_kwargs
        self._next_hook_id += 1
        hid = self._next_hook_id
        if prepend and self._forward_pre_hooks:
            reordered: dict[
                int,
                Callable[
                    [Module, tuple[Tensor, ...]], tuple[Tensor, ...] | Tensor | None
                ],
            ] = {hid: hook}
            reordered.update(self._forward_pre_hooks)
            self._forward_pre_hooks.clear()
            self._forward_pre_hooks.update(reordered)
        else:
            self._forward_pre_hooks[hid] = hook
        return RemovableHandle(self._forward_pre_hooks, hid)

    def register_forward_hook(
        self,
        hook: Callable[[Module, tuple[Tensor, ...], Tensor], Tensor | None],
        *,
        prepend: bool = False,
        with_kwargs: bool = False,
        always_call: bool = False,
    ) -> RemovableHandle[Callable[[Module, tuple[Tensor, ...], Tensor], Tensor | None]]:
        """Registers a forward hook on the module.

        Args:
            hook: Callable `(module, input, output) -> modified_output_or_None`.
            prepend: If True, prepends the hook before existing forward hooks.
            with_kwargs: Whether the hook accepts keyword arguments.
            always_call: If True, calls hook even if an exception occurs.

        Returns:
            A `RemovableHandle` that can be used to remove the added hook.
        """
        _ = (with_kwargs, always_call)
        self._next_hook_id += 1
        hid = self._next_hook_id
        if prepend and self._forward_hooks:
            reordered: dict[
                int, Callable[[Module, tuple[Tensor, ...], Tensor], Tensor | None]
            ] = {hid: hook}
            reordered.update(self._forward_hooks)
            self._forward_hooks.clear()
            self._forward_hooks.update(reordered)
        else:
            self._forward_hooks[hid] = hook
        return RemovableHandle(self._forward_hooks, hid)

    def register_full_backward_hook(
        self,
        hook: Callable[
            [Module, tuple[Tensor | None, ...], tuple[Tensor | None, ...]],
            tuple[Tensor | None, ...] | None,
        ],
        prepend: bool = False,
    ) -> RemovableHandle[
        Callable[
            [Module, tuple[Tensor | None, ...], tuple[Tensor | None, ...]],
            tuple[Tensor | None, ...] | None,
        ]
    ]:
        """Registers a backward hook on the module.

        Args:
            hook: Callable `(module, grad_input, grad_output) -> modified_grad_input_or_None`.
            prepend: If True, prepends the hook before existing backward hooks.

        Returns:
            A `RemovableHandle` that can be used to remove the added hook.
        """
        self._next_hook_id += 1
        hid = self._next_hook_id
        if prepend and self._backward_hooks:
            reordered: dict[
                int,
                Callable[
                    [Module, tuple[Tensor | None, ...], tuple[Tensor | None, ...]],
                    tuple[Tensor | None, ...] | None,
                ],
            ] = {hid: hook}
            reordered.update(self._backward_hooks)
            self._backward_hooks.clear()
            self._backward_hooks.update(reordered)
        else:
            self._backward_hooks[hid] = hook
        return RemovableHandle(self._backward_hooks, hid)

    def add_module(self, name: str, module: Module | None) -> None:
        """Adds a child module to the current module.

        Args:
            name: Name of the child module.
            module: Child module to register.
        """
        self._parameters.pop(name, None)
        self._buffers.pop(name, None)
        if module is not None:
            self._modules[name] = module
        else:
            self._modules.pop(name, None)

    def register_parameter(self, name: str, param: Parameter | None) -> None:
        """Adds a parameter to the module.

        Args:
            name: Name of the parameter.
            param: Parameter to register, or None.
        """
        self._modules.pop(name, None)
        self._buffers.pop(name, None)
        if param is not None:
            self._parameters[name] = param
        else:
            self._parameters.pop(name, None)

    def register_buffer(
        self, name: str, tensor: Tensor | None, persistent: bool = True
    ) -> None:
        """Adds a buffer to the module.

        Args:
            name: Name of the buffer.
            tensor: Buffer tensor to be registered, or None.
            persistent: Whether the buffer is part of this module's `state_dict`.
        """
        self._modules.pop(name, None)
        self._parameters.pop(name, None)
        if tensor is not None:
            self._buffers[name] = tensor
            if persistent:
                self._non_persistent_buffers_set.discard(name)
            else:
                self._non_persistent_buffers_set.add(name)
        else:
            self._buffers.pop(name, None)
            self._non_persistent_buffers_set.discard(name)

    def named_buffers(
        self,
        prefix: str = "",
        recurse: bool = True,
        remove_duplicate: bool = True,
    ) -> Iterator[tuple[str, Tensor]]:
        """Returns an iterator over module buffers, yielding both the name and the buffer.

        Args:
            prefix: Prefix to prepend to all buffer names.
            recurse: If True, yields buffers of this module and all submodules.
            remove_duplicate: Whether to remove duplicated buffers in the result.

        Yields:
            Tuple `(name, buffer_tensor)`.
        """
        memo: set[int] = set()
        for mod_prefix, module in self.named_modules(
            prefix=prefix, remove_duplicate=remove_duplicate
        ):
            if not recurse and module is not self:
                continue
            for name, buf in module._buffers.items():
                if remove_duplicate:
                    if id(buf) in memo:
                        continue
                    memo.add(id(buf))
                full_name = f"{mod_prefix}.{name}" if mod_prefix else name
                yield full_name, buf

    def buffers(self, recurse: bool = True) -> Iterator[Tensor]:
        """Returns an iterator over module buffers.

        Args:
            recurse: If True, yields buffers of this module and all submodules.

        Yields:
            Module buffer tensors.
        """
        for _, buf in self.named_buffers(recurse=recurse):
            yield buf

    def named_parameters(
        self,
        prefix: str = "",
        recurse: bool = True,
        remove_duplicate: bool = True,
    ) -> Iterator[tuple[str, Parameter]]:
        """Returns an iterator over module parameters, yielding both the name and the parameter.

        Args:
            prefix: Prefix to prepend to all parameter names.
            recurse: If True, yields parameters of this module and all submodules.
            remove_duplicate: Whether to remove duplicated parameters in the result.

        Yields:
            Tuple `(name, parameter)`.
        """
        memo: set[int] = set()
        for mod_prefix, module in self.named_modules(
            prefix=prefix, remove_duplicate=remove_duplicate
        ):
            if not recurse and module is not self:
                continue
            for name, param in module._parameters.items():
                if remove_duplicate:
                    if id(param) in memo:
                        continue
                    memo.add(id(param))
                full_name = f"{mod_prefix}.{name}" if mod_prefix else name
                yield full_name, param

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        """Returns an iterator over module parameters.

        Args:
            recurse: If True, yields parameters of this module and all submodules.

        Yields:
            Module parameters.
        """
        for _, param in self.named_parameters(recurse=recurse):
            yield param

    def children(self) -> Iterator[Module]:
        """Returns an iterator over immediate children modules.

        Yields:
            Immediate child modules.
        """
        yield from self._modules.values()

    def named_children(self) -> Iterator[tuple[str, Module]]:
        """Returns an iterator over immediate children modules, yielding `(name, module)`.

        Yields:
            Tuple containing the name and child module.
        """
        yield from self._modules.items()

    def named_modules(
        self,
        memo: set[int] | None = None,
        prefix: str = "",
        remove_duplicate: bool = True,
    ) -> Iterator[tuple[str, Module]]:
        """Returns an iterator over all modules in the network, yielding `(name, module)`.

        Args:
            memo: Optional set of visited module `id`s.
            prefix: Prefix to prepend to all module names.
            remove_duplicate: Whether to skip duplicate module references.

        Yields:
            Tuple `(name, module)`.
        """
        if memo is None:
            memo = set()
        if not remove_duplicate or id(self) not in memo:
            if remove_duplicate:
                memo.add(id(self))
            yield prefix, self
            for name, module in self._modules.items():
                submodule_prefix = f"{prefix}.{name}" if prefix else name
                yield from module.named_modules(
                    memo=memo,
                    prefix=submodule_prefix,
                    remove_duplicate=remove_duplicate,
                )

    def modules(self) -> Iterator[Module]:
        """Returns an iterator over all modules in the network.

        Yields:
            Module instances in the network.
        """
        for _, module in self.named_modules():
            yield module

    def apply(self, fn: Callable[[Module], None]) -> Module:
        """Applies `fn` recursively to every submodule as well as `self`.

        Args:
            fn: Function to be applied to each submodule.

        Returns:
            The module itself (`self`).
        """
        for module in self.children():
            module.apply(fn)
        fn(self)
        return self

    def requires_grad_(self, requires_grad: bool = True) -> Module:
        """Changes if autograd should record operations on parameters in this module.

        Args:
            requires_grad: Whether parameters of this module require gradients.

        Returns:
            The module itself (`self`).
        """
        for p in self.parameters():
            p.requires_grad = requires_grad
        return self

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Resets gradients of all model parameters.

        Args:
            set_to_none: If True, sets `.grad` to None instead of filling with zeros.
        """
        for p in self.parameters():
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.zero_()

    def state_dict(
        self,
        destination: dict[str, Tensor] | None = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> dict[str, Tensor]:
        """Returns a dictionary containing the whole persistent state of the module.

        Args:
            destination: Optional dictionary into which state will be stored.
            prefix: Prefix added to parameter and buffer names.
            keep_vars: If False, detached tensors are stored; if True, variables are kept.

        Returns:
            Dictionary mapping parameter and persistent buffer names to tensors.
        """
        state: dict[str, Tensor] = {} if destination is None else destination
        for name, param in self._parameters.items():
            key = f"{prefix}{name}"
            state[key] = param if keep_vars else param.detach()
        for name, buf in self._buffers.items():
            if name not in self._non_persistent_buffers_set:
                key = f"{prefix}{name}"
                state[key] = buf if keep_vars else buf.detach()
        for name, module in self._modules.items():
            module.state_dict(
                destination=state,
                prefix=f"{prefix}{name}.",
                keep_vars=keep_vars,
            )
        return state

    def load_state_dict(
        self,
        state_dict: Mapping[str, Tensor],
        strict: bool = True,
        assign: bool = False,
    ) -> IncompatibleKeys:
        """Copies parameters and buffers from `state_dict` into this module and its descendants.

        Args:
            state_dict: Mapping containing parameters and buffers.
            strict: Whether to strictly enforce that the keys in `state_dict` match this module.
            assign: If True, preserves parameter/buffer identity from `state_dict` instead of copying in-place.

        Returns:
            `IncompatibleKeys(missing_keys, unexpected_keys)` named tuple.
        """
        expected: dict[str, Tensor] = self.state_dict(keep_vars=True)
        missing_keys: list[str] = [k for k in expected if k not in state_dict]
        unexpected_keys: list[str] = [k for k in state_dict if k not in expected]

        if strict and (missing_keys or unexpected_keys):
            raise RuntimeError(
                f"Error(s) in loading state_dict: missing={missing_keys}, unexpected={unexpected_keys}"
            )

        for key, src_tensor in state_dict.items():
            if key in expected:
                target = expected[key]
                if assign:
                    target._tensor = _to_tensor(src_tensor, dtype=src_tensor.dtype)
                else:
                    target.copy_(src_tensor)

        return IncompatibleKeys(missing_keys, unexpected_keys)

    def _convert_tensors(self, dtype: DTypeLike) -> Module:
        """Converts all parameters and buffers to `dtype` in-place.

        Args:
            dtype: Target data type.

        Returns:
            The module itself (`self`).
        """
        if dtype is not None:
            dt_str = str(dtype)
            for p in self.parameters():
                p._tensor = _to_tensor(p, dtype=dt_str)
            for b in self.buffers():
                b._tensor = _to_tensor(b, dtype=dt_str)
        return self

    def to(self, *args: OpArg, **kwargs: OpArg) -> Module:
        """Moves and/or casts the parameters and buffers.

        Args:
            *args: Device and/or dtype positional arguments.
            **kwargs: Device and/or dtype keyword arguments.

        Returns:
            The module itself (`self`).
        """
        dtype_arg = kwargs.get("dtype")
        for arg in args:
            if isinstance(arg, str) and arg in (
                "float16",
                "float32",
                "float64",
                "bfloat16",
                "int8",
                "int16",
                "int32",
                "int64",
                "uint8",
                "bool",
                "complex64",
                "complex128",
            ):
                dtype_arg = arg
        if isinstance(dtype_arg, str):
            self._convert_tensors(dtype_arg)
        return self

    def cpu(self) -> Module:
        """Moves all model parameters and buffers to the CPU.

        Returns:
            The module itself (`self`).
        """
        return self

    def cuda(self, device: int | str | None = None) -> Module:
        """Moves all model parameters and buffers to the GPU (no-op on CPU backend).

        Args:
            device: Optional destination GPU device index or name.

        Returns:
            The module itself (`self`).
        """
        _ = device
        return self

    def float(self) -> Module:
        """Casts all floating point parameters and buffers to `float32` datatype.

        Returns:
            The module itself (`self`).
        """
        return self._convert_tensors("float32")

    def double(self) -> Module:
        """Casts all floating point parameters and buffers to `float64` datatype.

        Returns:
            The module itself (`self`).
        """
        return self._convert_tensors("float64")

    def half(self) -> Module:
        """Casts all floating point parameters and buffers to `float16` datatype.

        Returns:
            The module itself (`self`).
        """
        return self._convert_tensors("float16")

    def bfloat16(self) -> Module:
        """Casts all floating point parameters and buffers to `bfloat16` datatype.

        Returns:
            The module itself (`self`).
        """
        return self._convert_tensors("bfloat16")

    def type(self, dst_type: str) -> Module:
        """Casts all parameters and buffers to `dst_type`.

        Args:
            dst_type: Desired dtype string.

        Returns:
            The module itself (`self`).
        """
        return self._convert_tensors(dst_type)

    def train(self, mode: bool = True) -> Module:
        """Sets the module in training mode.

        Args:
            mode: Whether to set training mode (`True`) or evaluation mode (`False`).

        Returns:
            The module itself (`self`).
        """
        self.training = mode
        for module in self._modules.values():
            module.train(mode)
        return self

    def eval(self) -> Module:
        """Sets the module in evaluation mode.

        Returns:
            The module itself (`self`).
        """
        return self.train(False)


class ModuleList(Module):
    """Holds submodules in a list."""

    def __init__(
        self,
        modules: Iterable[Module] | None = None,
        *args: OpArg,
        **kwargs: OpArg,
    ) -> None:
        """Initializes the ModuleList.

        Args:
            modules: Optional iterable of modules to add.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        _ = (args, kwargs)
        super().__init__()
        self._modules_list: list[Module] = []
        if modules is not None:
            self.extend(modules)

    def _sync_modules_dict(self) -> None:
        """Synchronizes `self._modules` keys (`'0'`, `'1'`, ...) with `self._modules_list`."""
        self._modules.clear()
        for idx, mod in enumerate(self._modules_list):
            self._modules[str(idx)] = mod

    def __iter__(self) -> Iterator[Module]:
        """Returns an iterator over the modules in the list.

        Returns:
            Iterator over child modules.
        """
        return iter(self._modules_list)

    def __len__(self) -> int:
        """Returns the number of modules in the list.

        Returns:
            Number of modules.
        """
        return len(self._modules_list)

    @overload
    def __getitem__(self, idx: int) -> Module:
        """Retrieves a single module by integer index.

        Args:
            idx: Integer index.

        Returns:
            Module at `idx`.
        """

    @overload
    def __getitem__(self, idx: SliceIndexType) -> ModuleList:
        """Retrieves a `ModuleList` slice.

        Args:
            idx: Slice object.

        Returns:
            New `ModuleList` containing the sliced modules.
        """

    def __getitem__(self, idx: int | SliceIndexType) -> Module | ModuleList:
        """Returns the module at index `idx` or a sliced `ModuleList`.

        Args:
            idx: Integer index or slice.

        Returns:
            The module at `idx` or a new `ModuleList` when `idx` is a slice.
        """
        if isinstance(idx, slice):
            return ModuleList(self._modules_list[idx])
        return self._modules_list[idx]

    def __setitem__(self, idx: int, module: Module) -> None:
        """Sets the module at index `idx`.

        Args:
            idx: Index to replace.
            module: New module to store at `idx`.
        """
        self._modules_list[idx] = module
        self._sync_modules_dict()

    def __delitem__(self, idx: int | SliceIndexType) -> None:
        """Deletes the module(s) at `idx`.

        Args:
            idx: Integer index or slice to delete.
        """
        del self._modules_list[idx]
        self._sync_modules_dict()

    def __iadd__(self, modules: Iterable[Module]) -> ModuleList:
        """Appends modules from an iterable in-place.

        Args:
            modules: Iterable of modules to append.

        Returns:
            The modified `ModuleList` (`self`).
        """
        return self.extend(modules)

    def append(self, module: Module) -> ModuleList:
        """Appends a given module to the end of the list.

        Args:
            module: Module to append.

        Returns:
            The modified `ModuleList` (`self`).
        """
        self._modules_list.append(module)
        self._modules[str(len(self._modules_list) - 1)] = module
        return self

    def extend(self, modules: Iterable[Module]) -> ModuleList:
        """Appends modules from a Python iterable to the end of the list.

        Args:
            modules: Iterable of modules to append.

        Returns:
            The modified `ModuleList` (`self`).
        """
        for m in modules:
            self.append(m)
        return self

    def insert(self, index: int, module: Module) -> None:
        """Inserts a given module before a given index in the list.

        Args:
            index: Index before which to insert `module`.
            module: Module to insert.
        """
        self._modules_list.insert(index, module)
        self._sync_modules_dict()

    def pop(self, key: int = -1) -> Module:
        """Removes and returns the module at `key`.

        Args:
            key: Index of the module to pop (defaults to `-1`).

        Returns:
            The removed `Module`.
        """
        mod = self._modules_list.pop(key)
        self._sync_modules_dict()
        return mod


class ParameterList(Module):
    """Holds parameters in a list."""

    def __init__(
        self,
        values: Iterable[Parameter] | None = None,
        parameters: Iterable[Parameter] | None = None,
        *args: OpArg,
        **kwargs: OpArg,
    ) -> None:
        """Initializes the ParameterList.

        Args:
            values: Optional iterable of parameters to add.
            parameters: Optional alias for `values`.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        _ = (args, kwargs)
        super().__init__()
        self._params_list: list[Parameter] = []
        eff_params = values if values is not None else parameters
        if eff_params is not None:
            self.extend(eff_params)

    def _sync_params_dict(self) -> None:
        """Synchronizes `self._parameters` keys (`'0'`, `'1'`, ...) with `self._params_list`."""
        self._parameters.clear()
        for idx, p in enumerate(self._params_list):
            self._parameters[str(idx)] = p

    def __iter__(self) -> Iterator[Parameter]:
        """Returns an iterator over the parameters in the list.

        Returns:
            Iterator over parameters.
        """
        return iter(self._params_list)

    def __len__(self) -> int:
        """Returns the number of parameters in the list.

        Returns:
            Number of parameters.
        """
        return len(self._params_list)

    @overload
    def __getitem__(self, idx: int) -> Parameter:
        """Retrieves a single parameter by integer index.

        Args:
            idx: Integer index.

        Returns:
            Parameter at `idx`.
        """

    @overload
    def __getitem__(self, idx: SliceIndexType) -> ParameterList:
        """Retrieves a `ParameterList` slice.

        Args:
            idx: Slice object.

        Returns:
            New `ParameterList` containing the sliced parameters.
        """

    def __getitem__(self, idx: int | SliceIndexType) -> Parameter | ParameterList:
        """Returns the parameter at index `idx` or a sliced `ParameterList`.

        Args:
            idx: Integer index or slice.

        Returns:
            The parameter at `idx` or a new `ParameterList` when `idx` is a slice.
        """
        if isinstance(idx, slice):
            return ParameterList(self._params_list[idx])
        return self._params_list[idx]

    def __setitem__(self, idx: int, param: Parameter) -> None:
        """Sets the parameter at index `idx`.

        Args:
            idx: Index to replace.
            param: Parameter to store at `idx`.
        """
        self._params_list[idx] = param
        self._sync_params_dict()

    def __iadd__(self, parameters: Iterable[Parameter]) -> ParameterList:
        """Appends parameters from an iterable in-place.

        Args:
            parameters: Iterable of parameters to append.

        Returns:
            The modified `ParameterList` (`self`).
        """
        return self.extend(parameters)

    def append(self, param: Parameter) -> ParameterList:
        """Appends a given parameter to the end of the list.

        Args:
            param: Parameter to append.

        Returns:
            The modified `ParameterList` (`self`).
        """
        self._params_list.append(param)
        self._parameters[str(len(self._params_list) - 1)] = param
        return self

    def extend(self, parameters: Iterable[Parameter]) -> ParameterList:
        """Appends parameters from a Python iterable to the end of the list.

        Args:
            parameters: Iterable of parameters to append.

        Returns:
            The modified `ParameterList` (`self`).
        """
        for p in parameters:
            self.append(p)
        return self


class Container(Module):
    """Deprecated base container class provided for compatibility."""

    def __init__(self, *, training: bool = True, **kwargs: Module) -> None:
        """Initializes the Container with keyword child modules.

        Args:
            training: Whether the container is in training mode.
            **kwargs: Named child modules to register.
        """
        super().__init__()
        self.training = training
        for key, value in kwargs.items():
            self.add_module(key, value)


class ModuleDict(Module):
    """Holds submodules in a dictionary."""

    def __init__(self, modules: ModuleDictArg = None) -> None:
        """Initializes the ModuleDict.

        Args:
            modules: Mapping or iterable of `(key, module)` pairs.
        """
        super().__init__()
        if modules is not None:
            self.update(modules)

    def __getitem__(self, key: str) -> Module:
        """Returns the submodule associated with `key`.

        Args:
            key: Submodule key string.

        Returns:
            The registered `Module`.
        """
        return self._modules[key]

    def __setitem__(self, key: str, module: Module) -> None:
        """Registers `module` under `key`.

        Args:
            key: Submodule key string.
            module: Module to register.
        """
        self.add_module(key, module)

    def __delitem__(self, key: str) -> None:
        """Removes the submodule under `key`.

        Args:
            key: Submodule key string to remove.
        """
        del self._modules[key]

    def __len__(self) -> int:
        """Returns the number of submodules in the dictionary.

        Returns:
            Number of submodules.
        """
        return len(self._modules)

    def __iter__(self) -> Iterator[str]:
        """Returns an iterator over submodule keys.

        Returns:
            Iterator over keys.
        """
        return iter(self._modules)

    def __contains__(self, key: str) -> bool:
        """Checks whether `key` is present in the `ModuleDict`.

        Args:
            key: Key string to check.

        Returns:
            True if `key` is present, False otherwise.
        """
        return key in self._modules

    def clear(self) -> None:
        """Removes all submodules from the `ModuleDict`."""
        self._modules.clear()

    def pop(self, key: str) -> Module:
        """Removes and returns the submodule associated with `key`.

        Args:
            key: Key string to pop.

        Returns:
            The removed `Module`.
        """
        return self._modules.pop(key)

    def keys(self) -> Iterable[str]:
        """Returns an iterable of the `ModuleDict` keys.

        Returns:
            Keys view of `self._modules`.
        """
        return self._modules.keys()

    def items(self) -> Iterable[tuple[str, Module]]:
        """Returns an iterable of the `ModuleDict` key/value pairs.

        Returns:
            Items view of `self._modules`.
        """
        return self._modules.items()

    def values(self) -> Iterable[Module]:
        """Returns an iterable of the `ModuleDict` values.

        Returns:
            Values view of `self._modules`.
        """
        return self._modules.values()

    def update(
        self, modules: Mapping[str, Module] | Iterable[tuple[str, Module]]
    ) -> None:
        """Updates the ModuleDict with key-value pairs from a mapping or iterable.

        Args:
            modules: Mapping or iterable of `(key, module)` pairs.
        """
        if isinstance(modules, Mapping):
            for key, module in modules.items():
                self.add_module(key, module)
        else:
            for key, module in modules:
                self.add_module(key, module)

    def forward(self, *args: OpArg, **kwargs: OpArg) -> Tensor:
        """Raises NotImplementedError because ModuleDict is a container and not callable.

        Args:
            *args: Unused positional arguments.
            **kwargs: Unused keyword arguments.

        Returns:
            Never returns.
        """
        _ = (args, kwargs)
        raise NotImplementedError("ModuleDict should not be called.")


class ParameterDict(Module):
    """Holds parameters in a dictionary."""

    def __init__(
        self,
        parameters: ParamDictArg = None,
        values: ParamDictArg = None,
    ) -> None:
        """Initializes the ParameterDict.

        Args:
            parameters: Mapping or iterable of `(key, Parameter)` pairs.
            values: Optional alias for `parameters`.
        """
        super().__init__()
        eff = parameters if parameters is not None else values
        if eff is not None:
            self.update(eff)

    def __getitem__(self, key: str) -> Parameter:
        """Returns the parameter associated with `key`.

        Args:
            key: Parameter key string.

        Returns:
            The registered `Parameter`.
        """
        return self._parameters[key]

    def __setitem__(self, key: str, param: Parameter) -> None:
        """Registers `param` under `key`.

        Args:
            key: Parameter key string.
            param: Parameter to register.
        """
        self.register_parameter(key, param)

    def __delitem__(self, key: str) -> None:
        """Removes the parameter under `key`.

        Args:
            key: Parameter key string to remove.
        """
        del self._parameters[key]

    def __len__(self) -> int:
        """Returns the number of parameters in the dictionary.

        Returns:
            Number of parameters.
        """
        return len(self._parameters)

    def __iter__(self) -> Iterator[str]:
        """Returns an iterator over parameter keys.

        Returns:
            Iterator over keys.
        """
        return iter(self._parameters)

    def __contains__(self, key: str) -> bool:
        """Checks whether `key` is present in the `ParameterDict`.

        Args:
            key: Key string to check.

        Returns:
            True if `key` is present, False otherwise.
        """
        return key in self._parameters

    def clear(self) -> None:
        """Removes all parameters from the `ParameterDict`."""
        self._parameters.clear()

    def pop(self, key: str) -> Parameter:
        """Removes and returns the parameter associated with `key`.

        Args:
            key: Key string to pop.

        Returns:
            The removed `Parameter`.
        """
        return self._parameters.pop(key)

    def keys(self) -> Iterable[str]:
        """Returns an iterable of the `ParameterDict` keys.

        Returns:
            Keys view of `self._parameters`.
        """
        return self._parameters.keys()

    def items(self) -> Iterable[tuple[str, Parameter]]:
        """Returns an iterable of the `ParameterDict` key/value pairs.

        Returns:
            Items view of `self._parameters`.
        """
        return self._parameters.items()

    def values(self) -> Iterable[Parameter]:
        """Returns an iterable of the `ParameterDict` values.

        Returns:
            Values view of `self._parameters`.
        """
        return self._parameters.values()

    def update(
        self,
        parameters: Mapping[str, Parameter] | Iterable[tuple[str, Parameter]],
    ) -> None:
        """Updates the ParameterDict with key-value pairs from a mapping or iterable.

        Args:
            parameters: Mapping or iterable of `(key, Parameter)` pairs.
        """
        if isinstance(parameters, Mapping):
            for key, param in parameters.items():
                self.register_parameter(key, param)
        else:
            for key, param in parameters:
                self.register_parameter(key, param)

    def forward(self, *args: OpArg, **kwargs: OpArg) -> Tensor:
        """Raises NotImplementedError because ParameterDict is a container and not callable.

        Args:
            *args: Unused positional arguments.
            **kwargs: Unused keyword arguments.

        Returns:
            Never returns.
        """
        _ = (args, kwargs)
        raise NotImplementedError("ParameterDict should not be called.")
