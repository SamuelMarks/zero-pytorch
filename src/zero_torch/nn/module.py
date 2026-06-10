"NN Module System."

from typing import Iterator, Tuple, Any
from zero_torch.tensor import Tensor


class Parameter(Tensor):
    def __init__(
        self, data: Any, requires_grad: bool = True, *args: Any, **kwargs: Any
    ):
        pass


class Module:
    def __call__(self, *args, **kwargs):
        pass

    def forward(self, *args, **kwargs):
        pass

    def __init__(self, *args: Any, **kwargs: Any):
        pass

    def __setattr__(self, name: str, value: Any):
        pass

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        pass

    def named_children(self) -> Iterator[Tuple[str, "Module"]]:
        pass

    def state_dict(self) -> dict:
        pass

    def to(self, device: str) -> "Module":
        pass


class ModuleList(Module):
    def __init__(self, modules=None, *args: Any, **kwargs: Any):
        pass


class ParameterList(Module):
    def __init__(self, parameters=None, *args: Any, **kwargs: Any):
        pass
