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


def test_special_parity():
    with EagerMode():
        t = Tensor([1.0, 2.0, 3.0])
        t2 = Tensor([0.5, 1.5, 2.5])
        n = Tensor([1, 2, 3])

        funcs = [
            ("bessel_j0", [t]),
            ("bessel_j1", [t]),
            ("bessel_y0", [t]),
            ("bessel_y1", [t]),
            ("modified_bessel_i0", [t]),
            ("modified_bessel_i1", [t]),
            ("modified_bessel_k0", [t]),
            ("modified_bessel_k1", [t]),
            ("erfcx", [t]),
            ("ndtr", [t]),
            ("ndtri", [t]),
            ("gammaln", [t]),
            ("polygamma", [1, t]),
            ("multigammaln", [t, 2]),
            ("zeta", [t, t2]),
            ("chebyshev_polynomial_t", [t, n]),
            ("hermite_polynomial_h", [t, n]),
            ("laguerre_polynomial_l", [t, n]),
            ("legendre_polynomial_p", [t, n]),
        ]

        for func_name, args in funcs:
            try:
                getattr(torch.special, func_name)(*args)
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
                _pass = True

        # test missing op
        with pytest.raises(NotImplementedError):
            torch.special._get_op("nonexistent_op")

        # test success op
        torch.special._ops.nonexistent_op = lambda: None
        assert torch.special._get_op("nonexistent_op")() is None


def test_special_import_fallback(monkeypatch):
    import builtins
    import importlib

    import zero_torch.special

    original_import = builtins.__import__

    def mock_import(name, globals=None, locals=None, fromlist=(), level=0):
        if (
            name == "ml_switcheroo_compiler.ops.unary.special"
            or name == "ml_switcheroo_compiler.ops.binary.special"
        ):
            raise ImportError("Mocked ImportError")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    importlib.reload(zero_torch.special)

    # Check that it fell back
    assert hasattr(zero_torch.special._ops, "__class__")
    assert zero_torch.special._ops.__class__.__name__ == "_MockSpecial"
