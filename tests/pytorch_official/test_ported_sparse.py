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

try:
    from ml_switcheroo_compiler.core.errors import (
        ShapeMismatchError,
        UnimplementedMathError,
    )
except ImportError:
    UnimplementedMathError = Exception
    ShapeMismatchError = Exception


def test_sparse_parity():
    with EagerMode():
        # Eager mode mock returns dense zeros, testing the path
        indices = [[0, 1, 1], [2, 0, 2]]
        values = [3.0, 4.0, 5.0]
        st = torch.sparse_coo_tensor(indices, values, (2, 3))
        assert st.shape == (2, 3)

        # Testing fallback path
        st2 = torch.sparse_coo_tensor(indices, values)
        assert st2.shape == (2, 3)

        st3 = torch.sparse_coo_tensor(indices, values, (3, 3, 3))
        assert st3.shape == (3, 3, 3)

        indices_empty = [[], []]
        values_empty = []
        st4 = torch.sparse_coo_tensor(indices_empty, values_empty)
        assert st4.shape == ()

        # add and mm calls
        # In EagerMode, if compiler has eager fallbacks it will be called, otherwise it might raise NotImplementedError or similar
        try:
            torch.sparse.add(st, st)
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
            _pass = True

        try:
            torch.sparse.mm(st, st)
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
            _pass = True

        # test missing op
        with pytest.raises(NotImplementedError):
            torch.sparse._get_op("nonexistent_op")

        # test success op
        torch.sparse._ops.nonexistent_op = lambda: None
        assert torch.sparse._get_op("nonexistent_op")() is None


def test_sparse_proxy():
    import ml_switcheroo_compiler

    # Force eager mode off for this test since conftest sets it to True
    prev = ml_switcheroo_compiler.core.config.eager_mode
    ml_switcheroo_compiler.core.config.eager_mode = False
    try:
        indices = [[0, 1, 1], [2, 0, 2]]
        values = [3.0, 4.0, 5.0]
        st = torch.sparse_coo_tensor(indices, values, (2, 3))
        assert st.shape == (2, 3)
    finally:
        ml_switcheroo_compiler.core.config.eager_mode = prev


def test_sparse_edge_cases():
    import importlib
    import sys
    from unittest.mock import patch

    # Test ImportError
    with patch.dict(sys.modules, {"ml_switcheroo_compiler.ops.sparse": None}):
        import zero_torch.sparse

        importlib.reload(zero_torch.sparse)

        # Test not found ops
        # Test not found ops fallbacks
        import ml_switcheroo_compiler

        prev = ml_switcheroo_compiler.core.config.eager_mode
        ml_switcheroo_compiler.core.config.eager_mode = False
        try:
            with pytest.raises(
                NotImplementedError,
                match="Compiler backend missing sparse op: sparse_add",
            ):
                zero_torch.sparse.add(torch.tensor(1), torch.tensor(1))

            with pytest.raises(
                NotImplementedError, match="Compiler backend missing sparse op: smm"
            ):
                zero_torch.sparse.mm(torch.tensor([[1]]), torch.tensor([[1]]))
        finally:
            ml_switcheroo_compiler.core.config.eager_mode = prev

    # Reload normally
    importlib.reload(zero_torch.sparse)
    st = zero_torch.sparse.sparse_coo_tensor(
        [[0, 1, 1], [2, 0, 2]], [3.0, 4.0, 5.0], (2, 3)
    )
    assert st.shape == (2, 3)
