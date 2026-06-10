"NN Module System."

from typing import Iterator, Tuple, Any
from zero_torch.tensor import Tensor
from ml_switcheroo.tracing import _tracer
from ml_switcheroo_ir import LogicalNode
import uuid


class Parameter(Tensor):
    def __init__(
        self, data: Any, requires_grad: bool = True, *args: Any, **kwargs: Any
    ):
        super().__init__(data, requires_grad=requires_grad)

    @property
    def data(self):
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
    def __init__(self, *args: Any, **kwargs: Any):
        self._modules = {}
        self._parameters = {}
        self.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __setattr__(self, name: str, value: Any):
        if isinstance(value, Parameter):
            if not hasattr(self, "_parameters"):
                self.__dict__["_parameters"] = {}
            self._parameters[name] = value
        elif isinstance(value, Module):
            if not hasattr(self, "_modules"):
                self.__dict__["_modules"] = {}
            self._modules[name] = value
        super().__setattr__(name, value)

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        for name, param in self._parameters.items():
            yield param
        if recurse:
            for name, module in self._modules.items():
                for param in module.parameters(recurse=True):
                    yield param

    def named_children(self) -> Iterator[Tuple[str, "Module"]]:
        for name, module in self._modules.items():
            yield name, module

    def state_dict(self) -> dict:
        state = {}
        for name, param in self._parameters.items():
            state[name] = param
        for name, module in self._modules.items():
            for child_name, child_param in module.state_dict().items():
                state[f"{name}.{child_name}"] = child_param
        return state

    def to(self, device: str) -> "Module":
        return self


class ModuleList(Module):
    def __init__(self, modules=None, *args: Any, **kwargs: Any):
        super().__init__()
        self._modules_list = list(modules) if modules else []

    def __iter__(self):
        return iter(self._modules_list)

    def append(self, module):
        self._modules_list.append(module)
        self._modules[str(len(self._modules_list) - 1)] = module


class ParameterList(Module):
    def __init__(self, parameters=None, *args: Any, **kwargs: Any):
        super().__init__()
        self._params_list = list(parameters) if parameters else []

    def __iter__(self):
        return iter(self._params_list)

    def append(self, param):
        self._params_list.append(param)
        self._parameters[str(len(self._params_list) - 1)] = param
