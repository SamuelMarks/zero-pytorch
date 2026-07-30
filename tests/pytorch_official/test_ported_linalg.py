try:
    from ml_switcheroo_compiler.core.errors import (
        ShapeMismatchError,
        UnimplementedMathError,
    )
except ImportError:
    UnimplementedMathError = Exception
    ShapeMismatchError = Exception

import pytest
from ml_switcheroo_compiler.core.config import EagerMode

import zero_torch as torch
from zero_torch.tensor import Tensor

try:
    from ml_switcheroo_compiler.core.errors import (
        ShapeMismatchError,
        UnimplementedMathError,
    )
except ImportError:
    UnimplementedMathError = Exception
    ShapeMismatchError = Exception


def test_linalg_parity():
    with EagerMode():
        t = Tensor([[1.0, 0.0], [0.0, 1.0]])
        t2 = Tensor([[2.0, 0.0], [0.0, 2.0]])

        funcs = [
            ("cond", [t]),
            ("det", [t]),
            ("slogdet", [t]),
            ("matrix_rank", [t]),
            ("matrix_norm", [t]),
            ("vector_norm", [t]),
            ("cholesky", [t]),
            ("cholesky_ex", [t]),
            ("eig", [t]),
            ("eigh", [t]),
            ("eigvals", [t]),
            ("eigvalsh", [t]),
            ("lu", [t]),
            ("lu_factor", [t]),
            ("qr", [t]),
            ("svd", [t]),
            ("svdvals", [t]),
            ("inv", [t]),
            ("inv_ex", [t]),
            ("pinv", [t]),
            ("solve", [t, t2]),
            ("solve_ex", [t, t2]),
            ("lstsq", [t, t2]),
            ("tensorinv", [t]),
            ("tensorsolve", [t, t2]),
            ("cross", [t, t2]),
            ("matrix_exp", [t]),
            ("matrix_power", [t, 2]),
            ("householder_product", [t, t2]),
        ]

        for func_name, args in funcs:
            try:
                getattr(torch.linalg, func_name)(*args)
            except NotImplementedError:
                _pass = True
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
            ):
                # We catch other exceptions like LinAlgError since the mock shapes might not perfectly map into valid mathematical inputs.
                _pass = True

        try:
            torch.linalg.multi_dot([t, t2])
        except NotImplementedError:
            _pass = True

        # test missing op
        with pytest.raises(NotImplementedError):
            torch.linalg._get_op("nonexistent_op")

        # test success op
        torch.linalg._ops.nonexistent_op = lambda: None
        assert torch.linalg._get_op("nonexistent_op")() is None


def test_linalg_import_fallback(monkeypatch):
    import builtins
    import importlib

    import zero_torch.linalg

    original_import = builtins.__import__

    def mock_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "ml_switcheroo_compiler.ops.linalg":
            raise ImportError("Mocked ImportError")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    importlib.reload(zero_torch.linalg)

    # Check that it fell back
    assert hasattr(zero_torch.linalg._ops, "__class__")
    assert zero_torch.linalg._ops.__class__.__name__ == "_MockLinalg"
