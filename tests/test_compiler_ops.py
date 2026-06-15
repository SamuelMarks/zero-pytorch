import pytest
import numpy as np
import torch
import ml_switcheroo_compiler.ops as ops
import ml_switcheroo_compiler as ml_switcheroo
import inspect
from tests.test_all_apis import get_inputs_for_api

compiler_ops = [
    name
    for name, obj in inspect.getmembers(ops)
    if not name.startswith("_") and inspect.isfunction(obj)
]


@pytest.mark.parametrize("op_name", compiler_ops)
def test_compiler_op_parity(op_name):
    if op_name in ["empty_like", "empty", "tensordot"]:
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
    except Exception:
        return

    # Run in Compiler Eager Mode
    try:
        with ml_switcheroo.EagerMode():
            from ml_switcheroo_compiler.core.device import Device, DeviceType
            from ml_switcheroo_compiler.core.dtype import DType

            def to_compiler_tensor(arg):
                if isinstance(arg, np.ndarray):
                    dtype = DType.Float32 if arg.dtype == np.float32 else DType.Int32
                    if arg.dtype == bool:
                        dtype = DType.Bool
                    return ml_switcheroo.Tensor(
                        arg,
                        shape=arg.shape,
                        dtype=dtype,
                        device=Device(DeviceType.CPU, 0),
                    )
                return arg

            c_args = [to_compiler_tensor(a) for a in np_args]
            c_res = compiler_fn(*c_args, **np_kwargs)
    except Exception as e:
        pytest.fail(f"Compiler failed on {op_name}: {e}")

    if isinstance(t_res, torch.Tensor):
        t_np = t_res.detach().cpu().numpy()
        c_np = c_res.numpy() if hasattr(c_res, "numpy") else np.array(c_res.data)

        assert t_np.shape == c_np.shape, f"Shape mismatch: {t_np.shape} vs {c_np.shape}"

        if np.issubdtype(t_np.dtype, np.floating):
            np.testing.assert_allclose(t_np, c_np, rtol=1e-4, atol=1e-4)
        else:
            np.testing.assert_array_equal(t_np, c_np)
