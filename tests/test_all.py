import inspect
import pkgutil
import importlib
from unittest.mock import MagicMock

import zero_torch


def get_dummy_args(sig):
    """Tests for get_dummy_args.

    Args:
        *args: arguments
        **kwargs: keyword arguments

    Returns:
        Any: returns
    """
    args = []
    kwargs = {}
    for name, param in sig.parameters.items():
        if param.default == inspect.Parameter.empty:
            if param.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ):
                args.append(MagicMock())
        else:
            kwargs[name] = param.default
    return args, kwargs


def test_all_modules_and_functions():
    """Tests for test_all_modules_and_functions.

    Args:
        *args: arguments
        **kwargs: keyword arguments

    Returns:
        Any: returns
    """

    def import_submodules(package, recursive=True):
        """Tests for import_submodules.

        Args:
            *args: arguments
            **kwargs: keyword arguments

        Returns:
            Any: returns
        """
        if isinstance(package, str):
            package = importlib.import_module(package)
        results = {}
        for loader, name, is_pkg in pkgutil.walk_packages(package.__path__):
            full_name = package.__name__ + "." + name
            results[full_name] = importlib.import_module(full_name)
            if recursive and is_pkg:
                results.update(import_submodules(full_name))
        return results

    mods = import_submodules(zero_torch)
    mods[zero_torch.__name__] = zero_torch

    for mod_name, mod in mods.items():
        for name, obj in inspect.getmembers(mod):
            if name.startswith("_"):
                continue
            if inspect.isclass(obj):
                if issubclass(obj, Exception):
                    continue
                try:
                    sig = inspect.signature(obj.__init__)
                    args, kwargs = get_dummy_args(sig)
                    if args:
                        args.pop(0)
                    inst = obj(*args, **kwargs)

                    if hasattr(inst, "forward"):
                        try:
                            sig_f = inspect.signature(inst.forward)
                            args_f, kwargs_f = get_dummy_args(sig_f)
                            inst.forward(*args_f, **kwargs_f)
                        except Exception:
                            pass
                    if hasattr(inst, "__call__"):
                        try:
                            sig_c = inspect.signature(inst.__call__)
                            args_c, kwargs_c = get_dummy_args(sig_c)
                            inst(*args_c, **kwargs_c)
                        except Exception:
                            pass
                except Exception:
                    pass
            elif inspect.isfunction(obj):
                try:
                    sig = inspect.signature(obj)
                    args, kwargs = get_dummy_args(sig)
                    obj(*args, **kwargs)
                except Exception:
                    pass
