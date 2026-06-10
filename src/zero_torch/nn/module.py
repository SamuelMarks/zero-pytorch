"NN Module System."

from typing import Iterator, Tuple, Any
from zero_torch.tensor import Tensor
from ml_switcheroo.tracing import _tracer
from ml_switcheroo_ir import LogicalNode
import uuid


class Parameter(Tensor):
    """Class."""

    def __init__(
        self, data: Any, requires_grad: bool = True, *args: Any, **kwargs: Any
    ):
        """Function."""
        super().__init__(data, requires_grad=requires_grad)

    @property
    def data(self):
        """Function."""
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
    """Class."""

    def __init__(self, *args: Any, **kwargs: Any):
        """Function."""
        self._modules = {}
        self._parameters = {}
        self._buffers = {}
        self.training = True

    def __call__(self, *args, **kwargs):
        """Function."""
        return self.forward(*args, **kwargs)

    def register_buffer(self, name, tensor):
        """Function."""
        self._buffers[name] = tensor
        setattr(self, name, tensor)

    def forward(self, *args, **kwargs):
        """Function."""
        raise NotImplementedError

    def __setattr__(self, name: str, value: Any):
        """Function."""
        if isinstance(value, Parameter):
            if not hasattr(self, "_parameters"):
                self.__dict__["_parameters"] = {}
            self._parameters[name] = value
        elif isinstance(value, Module):
            if not hasattr(self, "_modules"):
                self.__dict__["_modules"] = {}
            self._modules[name] = value
        super().__setattr__(name, value)

    def buffers(self, recurse: bool = True):
        """Function."""
        for name, buf in self._buffers.items():
            yield buf
        if recurse:
            for name, module in self._modules.items():
                for buf in module.buffers(recurse=True):
                    yield buf

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        """Function."""
        for name, param in self._parameters.items():
            yield param
        if recurse:
            for name, module in self._modules.items():
                for param in module.parameters(recurse=True):
                    yield param

    def named_children(self) -> Iterator[Tuple[str, "Module"]]:
        """Function."""
        for name, module in self._modules.items():
            yield name, module

    def state_dict(self) -> dict:
        """Function."""
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
        """Function."""
        return self


class ModuleList(Module):
    """Class."""

    def __init__(self, modules=None, *args: Any, **kwargs: Any):
        """Function."""
        super().__init__()
        self._modules_list = list(modules) if modules else []

    def __iter__(self):
        """Function."""
        return iter(self._modules_list)

    def append(self, module):
        """Function."""
        self._modules_list.append(module)
        self._modules[str(len(self._modules_list) - 1)] = module


class ParameterList(Module):
    """Class."""

    def __init__(self, parameters=None, *args: Any, **kwargs: Any):
        """Function."""
        super().__init__()
        self._params_list = list(parameters) if parameters else []

    def __iter__(self):
        """Function."""
        return iter(self._params_list)

    def append(self, param):
        """Function."""
        self._params_list.append(param)
        self._parameters[str(len(self._params_list) - 1)] = param
