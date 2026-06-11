"NN Module System."

from typing import Iterator, Tuple, Any
from zero_torch.tensor import Tensor
from ml_switcheroo.tracing import _tracer
from ml_switcheroo.ir.core import LogicalNode
import uuid


class Parameter(Tensor):
    """A kind of Tensor that is to be considered a module parameter."""

    def __init__(
        self, data: Any, requires_grad: bool = True, *args: Any, **kwargs: Any
    ):
        """Initializes a Parameter.

        Args:
            data (Any): Parameter tensor data.
            requires_grad (bool, optional): If the parameter requires gradient. Defaults to True.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(data, requires_grad=requires_grad)

    @property
    def data(self) -> Any:
        """Gets the underlying parameter data.

        Returns:
            Any: The parameter data or proxy for tracing.
        """
        if _tracer.is_tracing:
            if not hasattr(self, "_param_id"):
                self._param_id = "param_" + str(uuid.uuid4())
            # emit read variable
            out_id = str(uuid.uuid4())
            node = LogicalNode(
                id=out_id,
                op_type="ReadVariable",
                inputs=[self._param_id],
                shape_metadata=self.shape,
            )
            _tracer.add_node(node)
            from ml_switcheroo.tracing import ProxyTensor
            from ml_switcheroo import Tensor as SwitcherooTensor

            pt = ProxyTensor(id=out_id, shape=self.shape, dtype=str(self.dtype))
            return SwitcherooTensor(
                data=pt, shape=self.shape, dtype=self.dtype, device=self._tensor.device
            )
        return self._tensor.data


class Module:
    """Base class for all neural network modules."""

    def __init__(self, *args: Any, **kwargs: Any):
        """Initializes internal Module state.

        Args:
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        self._modules = {}
        self._parameters = {}
        self._buffers = {}
        self.training = True

    def __call__(self, *args, **kwargs) -> Any:
        """Defines the computation performed at every call.

        Args:
            *args: Positional arguments for the forward pass.
            **kwargs: Keyword arguments for the forward pass.

        Returns:
            Any: The result of the forward pass.
        """
        return self.forward(*args, **kwargs)

    def register_buffer(self, name: str, tensor: Tensor) -> None:
        """Adds a buffer to the module.

        Args:
            name (str): name of the buffer. The buffer can be accessed from this module using the given name.
            tensor (Tensor): buffer to be registered.
        """
        self._buffers[name] = tensor
        setattr(self, name, tensor)

    def forward(self, *args, **kwargs) -> Any:
        """Defines the computation performed at every call.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Raises:
            NotImplementedError: Modules must override this method.
        """
        raise NotImplementedError

    def __setattr__(self, name: str, value: Any) -> None:
        """Sets attributes, specifically handling Parameters and Modules.

        Args:
            name (str): Attribute name.
            value (Any): Attribute value to set.
        """
        if isinstance(value, Parameter):
            if not hasattr(self, "_parameters"):
                self.__dict__["_parameters"] = {}
            self._parameters[name] = value
        elif isinstance(value, Module):
            if not hasattr(self, "_modules"):
                self.__dict__["_modules"] = {}
            self._modules[name] = value
        super().__setattr__(name, value)

    def buffers(self, recurse: bool = True) -> Iterator[Tensor]:
        """Returns an iterator over module buffers.

        Args:
            recurse (bool, optional): If True, yields buffers of this module and all submodules. Defaults to True.

        Yields:
            Tensor: Module buffer.
        """
        for name, buf in self._buffers.items():
            yield buf
        if recurse:
            for name, module in self._modules.items():
                for buf in module.buffers(recurse=True):
                    yield buf

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        """Returns an iterator over module parameters.

        Args:
            recurse (bool, optional): If True, yields parameters of this module and all submodules. Defaults to True.

        Yields:
            Parameter: Module parameter.
        """
        for name, param in self._parameters.items():
            yield param
        if recurse:
            for name, module in self._modules.items():
                for param in module.parameters(recurse=True):
                    yield param

    def named_children(self) -> Iterator[Tuple[str, "Module"]]:
        """Returns an iterator over immediate children modules, yielding both the name of the module as well as the module itself.

        Yields:
            Tuple[str, Module]: Tuple containing name and child module.
        """
        for name, module in self._modules.items():
            yield name, module

    def state_dict(self) -> dict:
        """Returns a dictionary containing a whole state of the module.

        Returns:
            dict: A dictionary containing the module state.
        """
        state = {}
        for name, param in self._parameters.items():
            state[name] = param
        if hasattr(self, "_buffers"):
            for name, buf in self._buffers.items():
                state[name] = buf
        for name, module in self._modules.items():
            for child_name, child_param in module.state_dict().items():
                state[f"{name}.{child_name}"] = child_param
        return state

    def to(self, device: str) -> "Module":
        """Moves and/or casts the parameters and buffers.

        Args:
            device (str): The desired device of the parameters and buffers.

        Returns:
            Module: The module itself.
        """
        return self


class ModuleList(Module):
    """Holds submodules in a list."""

    def __init__(self, modules=None, *args: Any, **kwargs: Any):
        """Initializes the ModuleList.

        Args:
            modules (iterable, optional): An iterable of modules to add. Defaults to None.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()
        self._modules_list = list(modules) if modules else []

    def __iter__(self) -> Iterator[Module]:
        """Returns an iterator over the modules.

        Returns:
            Iterator[Module]: An iterator.
        """
        return iter(self._modules_list)

    def append(self, module: Module) -> None:
        """Appends a given module to the end of the list.

        Args:
            module (Module): Module to append.
        """
        self._modules_list.append(module)
        self._modules[str(len(self._modules_list) - 1)] = module


class ParameterList(Module):
    """Holds parameters in a list."""

    def __init__(self, parameters=None, *args: Any, **kwargs: Any):
        """Initializes the ParameterList.

        Args:
            parameters (iterable, optional): An iterable of parameters to add. Defaults to None.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()
        self._params_list = list(parameters) if parameters else []

    def __iter__(self) -> Iterator[Parameter]:
        """Returns an iterator over the parameters.

        Returns:
            Iterator[Parameter]: An iterator.
        """
        return iter(self._params_list)

    def append(self, param: Parameter) -> None:
        """Appends a given parameter to the end of the list.

        Args:
            param (Parameter): Parameter to append.
        """
        self._params_list.append(param)
        self._parameters[str(len(self._params_list) - 1)] = param


class Container(Module):
    """Base class for all neural network modules.

    Deprecated in PyTorch, but provided for compatibility.
    """

    def __init__(self, **kwargs):
        """Initializes the Container."""
        super().__init__()
        for key, value in kwargs.items():
            setattr(self, key, value)


class ModuleDict(Module):
    """Holds submodules in a dictionary."""

    def __init__(self, modules=None) -> None:
        super().__init__()
        self.update(modules)

    def update(self, modules):
        pass

    def forward(self, input):
        import ml_switcheroo.core.errors

        raise ml_switcheroo.core.errors.UnimplementedMathError


class ParameterDict(Module):
    """Holds parameters in a dictionary."""

    def __init__(self, parameters=None) -> None:
        super().__init__()
        self.update(parameters)

    def update(self, parameters):
        pass

    def forward(self, input):
        import ml_switcheroo.core.errors

        raise ml_switcheroo.core.errors.UnimplementedMathError
