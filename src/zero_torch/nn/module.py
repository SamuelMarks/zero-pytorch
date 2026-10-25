"""NN Module System."""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Iterator, Mapping
from typing import TYPE_CHECKING, overload

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import (
    Tensor as SwitcherooTensor,
)
from ml_switcheroo_compiler.core.tensor import (
    TensorConfig,
)
from ml_switcheroo_compiler.ir.core import LogicalNode

from zero_torch._types import (
    ModuleDictArg,
    OpArg,
    ParamDictArg,
    Scalar,
)
from zero_torch.tensor import Tensor, _ArrayDataProtocol
from zero_torch.tracing import ProxyTensor, _tracer

if TYPE_CHECKING:
    from typing import ParamSpec, Protocol, TypeVar

    _P = ParamSpec("_P")
    _R_co = TypeVar("_R_co", covariant=True)

    class _HasForward(Protocol[_P, _R_co]):
        """Structural protocol for `Module` subclasses defining a `forward` method."""

        def forward(self, *args: _P.args, **kwargs: _P.kwargs) -> _R_co:
            """Executes the forward pass of the module.

            Args:
                *args: Positional arguments for the forward pass.
                **kwargs: Keyword arguments for the forward pass.

            Returns:
                Output produced by the module.
            """


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

    _modules: dict[str, Module]
    _parameters: dict[str, Parameter]
    _buffers: dict[str, Tensor]
    training: bool

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
        """Invokes the module's `forward` computation.

        Args:
            *args: Positional arguments for the forward pass.
            **kwargs: Keyword arguments for the forward pass.

        Returns:
            The output of `self.forward(*args, **kwargs)`.
        """
        return self.forward(*args, **kwargs)

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
        _ = persistent
        self._modules.pop(name, None)
        self._parameters.pop(name, None)
        if tensor is not None:
            self._buffers[name] = tensor
        else:
            self._buffers.pop(name, None)

    def buffers(self, recurse: bool = True) -> Iterator[Tensor]:
        """Returns an iterator over module buffers.

        Args:
            recurse: If True, yields buffers of this module and all submodules.

        Yields:
            Module buffer tensors.
        """
        for buf in self._buffers.values():
            yield buf
        if recurse:
            for module in self._modules.values():
                for buf in module.buffers(recurse=True):
                    yield buf

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        """Returns an iterator over module parameters.

        Args:
            recurse: If True, yields parameters of this module and all submodules.

        Yields:
            Module parameters.
        """
        for param in self._parameters.values():
            yield param
        if recurse:
            for module in self._modules.values():
                for param in module.parameters(recurse=True):
                    yield param

    def named_children(self) -> Iterator[tuple[str, Module]]:
        """Returns an iterator over immediate children modules, yielding `(name, module)`.

        Yields:
            Tuple containing the name and child module.
        """
        yield from self._modules.items()

    def state_dict(self) -> dict[str, Tensor]:
        """Returns a dictionary containing the whole state of the module.

        Returns:
            Dictionary mapping parameter and buffer names to tensors.
        """
        state: dict[str, Tensor] = {}
        for name, param in self._parameters.items():
            state[name] = param
        for name, buf in self._buffers.items():
            state[name] = buf
        for name, module in self._modules.items():
            for child_name, child_param in module.state_dict().items():
                state[f"{name}.{child_name}"] = child_param
        return state

    def to(self, device: str) -> Module:
        """Moves and/or casts the parameters and buffers to `device`.

        Args:
            device: Desired device identifier string.

        Returns:
            The module itself (`self`).
        """
        _ = device
        return self

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
            for m in modules:
                self.append(m)

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

    def __getitem__(self, idx: int) -> Module:
        """Returns the module at index `idx`.

        Args:
            idx: Index of the module to retrieve.

        Returns:
            The module at `idx`.
        """
        return self._modules_list[idx]

    def append(self, module: Module) -> None:
        """Appends a given module to the end of the list.

        Args:
            module: Module to append.
        """
        self._modules_list.append(module)
        self._modules[str(len(self._modules_list) - 1)] = module


class ParameterList(Module):
    """Holds parameters in a list."""

    def __init__(
        self,
        parameters: Iterable[Parameter] | None = None,
        *args: OpArg,
        **kwargs: OpArg,
    ) -> None:
        """Initializes the ParameterList.

        Args:
            parameters: Optional iterable of parameters to add.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        _ = (args, kwargs)
        super().__init__()
        self._params_list: list[Parameter] = []
        if parameters is not None:
            for p in parameters:
                self.append(p)

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

    def append(self, param: Parameter) -> None:
        """Appends a given parameter to the end of the list.

        Args:
            param: Parameter to append.
        """
        self._params_list.append(param)
        self._parameters[str(len(self._params_list) - 1)] = param


class Container(Module):
    """Deprecated base container class provided for compatibility."""

    def __init__(self, **kwargs: Module) -> None:
        """Initializes the Container with keyword child modules.

        Args:
            **kwargs: Named child modules to register.
        """
        super().__init__()
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

    def __init__(self, parameters: ParamDictArg = None) -> None:
        """Initializes the ParameterDict.

        Args:
            parameters: Mapping or iterable of `(key, Parameter)` pairs.
        """
        super().__init__()
        if parameters is not None:
            self.update(parameters)

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
