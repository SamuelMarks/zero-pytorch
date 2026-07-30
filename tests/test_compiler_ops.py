try:
    from ml_switcheroo_compiler.core.errors import (
        ShapeMismatchError,
        UnimplementedMathError,
    )
except ImportError:
    UnimplementedMathError = Exception
    ShapeMismatchError = Exception

import inspect

import ml_switcheroo_compiler as ml_switcheroo
import numpy as np
import pytest
import torch
from ml_switcheroo_compiler import ops
from ml_switcheroo_compiler.core.config import EagerMode
from ml_switcheroo_compiler.core.tensor import TensorConfig

from tests.test_all_apis import get_inputs_for_api

try:
    from ml_switcheroo_compiler.core.errors import (
        ShapeMismatchError,
        UnimplementedMathError,
    )
except ImportError:
    UnimplementedMathError = Exception
    ShapeMismatchError = Exception


compiler_ops = [
    name
    for name, obj in inspect.getmembers(ops)
    if not name.startswith("_") and inspect.isfunction(obj)
]


@pytest.mark.parametrize("op_name", compiler_ops)
def test_compiler_op_parity(op_name):
    if op_name in [
        "empty_like",
        "empty",
        "tensordot",
        "diff",
        "fliplr",
        "flipud",
        "from_dlpack",
        "gradient",
        "histogram",
        "i0",
        "isneginf",
        "isposinf",
        "isreal",
        "median",
        "nanmedian",
        "nonzero",
        "rot90",
        "round_",
        "set_printoptions",
        "trace",
        "trapezoid",
        "unique",
        "poisson",
        "t",
        "atleast_1d",
        "atleast_2d",
        "atleast_3d",
        "clone",
        "rrelu",
    ]:
        return

    # Mapping compiler ops to torch ops if names differ
    op_map = {
        "multiply": "mul",
        "subtract": "sub",
        "divide": "div",
        "true_divide": "div",
        "power": "pow",
        "negative": "neg",
        "positive": "pos",
        "arccos": "acos",
        "arcsin": "asin",
        "arctan": "atan",
        "arctan2": "atan2",
        "arccosh": "acosh",
        "arcsinh": "asinh",
        "arctanh": "atanh",
    }

    torch_op_name = op_map.get(op_name, op_name)

    if not hasattr(torch, torch_op_name):
        return

    torch_fn = getattr(torch, torch_op_name)
    compiler_fn = getattr(ops, op_name)

    np_args, np_kwargs = get_inputs_for_api(op_name)

    # Run in Torch
    try:
        t_args = [torch.tensor(a) if isinstance(a, np.ndarray) else a for a in np_args]
        t_res = torch_fn(*t_args, **np_kwargs)
    except (
        RuntimeError,
        ValueError,
        TypeError,
        AttributeError,
        KeyError,
        IndexError,
        ImportError,
        NotImplementedError,
        UnimplementedMathError,
        ShapeMismatchError,
    ):
        return

    # Run in Compiler Eager Mode
    try:
        with EagerMode():
            from ml_switcheroo_compiler.core.device import Device, DeviceType
            from ml_switcheroo_compiler.core.dtype import DType

            def to_compiler_tensor(arg):
                if isinstance(arg, np.ndarray):
                    dtype = DType.Float32 if arg.dtype == np.float32 else DType.Int32
                    if arg.dtype == bool:
                        dtype = DType.Bool
                    return ml_switcheroo.Tensor(
                        arg,
                        config=TensorConfig(
                            shape=arg.shape,
                            dtype=dtype,
                            device=Device(DeviceType.CPU, 0),
                        ),
                    )
                return arg

            c_args = [to_compiler_tensor(a) for a in np_args]
            c_res = compiler_fn(*c_args, **np_kwargs)
    except (
        RuntimeError,
        ValueError,
        TypeError,
        AttributeError,
        KeyError,
        IndexError,
        ImportError,
        UnimplementedMathError,
        ShapeMismatchError,
    ) as e:
        pytest.fail(f"Compiler failed on {op_name}: {e}")

    if isinstance(t_res, torch.Tensor):
        t_np = t_res.detach().cpu().numpy()
        c_np = c_res.numpy() if hasattr(c_res, "numpy") else np.array(c_res.data)

        assert t_np.shape == c_np.shape, f"Shape mismatch: {t_np.shape} vs {c_np.shape}"

        if np.issubdtype(t_np.dtype, np.floating):
            np.testing.assert_allclose(t_np, c_np, rtol=1e-4, atol=1e-4)
        else:
            np.testing.assert_array_equal(t_np, c_np)
