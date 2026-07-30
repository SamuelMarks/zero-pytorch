try:
    from ml_switcheroo_compiler.core.errors import (
        ShapeMismatchError,
        UnimplementedMathError,
    )
except ImportError:
    UnimplementedMathError = Exception
    ShapeMismatchError = Exception

import sys
from unittest import mock

import ml_switcheroo_compiler as ml_switcheroo

import zero_torch
import zero_torch.nn.functional_dropout as FD
import zero_torch.nn.functional_pooling as FP
import zero_torch.nn.functional_utils as FU
from zero_torch import Tensor

try:
    from ml_switcheroo_compiler.core.errors import (
        ShapeMismatchError,
        UnimplementedMathError,
    )
except ImportError:
    UnimplementedMathError = Exception
    ShapeMismatchError = Exception


def test_mvlgamma():
    t = Tensor([1.0, 2.0])
    try:
        zero_torch.mvlgamma(t, 2)
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

    zero_torch.nan_to_num(t, neginf=0.0)


def test_rand_shapes():
    zero_torch.rand([2, 3])
    zero_torch.randn([2, 3])
    try:
        zero_torch.randint(0, [2, 3])
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
    zero_torch.randint(0, 10, size=2)


def test_none_returns():
    class MockOpsNone:
        def __getattr__(self, name):
            return lambda *args, **kwargs: None

    with (
        mock.patch.object(sys.modules["zero_torch.tensor"], "ops", MockOpsNone()),
        mock.patch("zero_torch.nn.functional_dropout._nn", MockOpsNone()),
        mock.patch("zero_torch.nn.functional_pooling.ops", MockOpsNone()),
        mock.patch("zero_torch.nn.functional_pooling.avg_pool", MockOpsNone().avg_pool),
        mock.patch("zero_torch.nn.functional_pooling.max_pool", MockOpsNone().max_pool),
        mock.patch("zero_torch.nn.functional_utils._nn", MockOpsNone()),
    ):
        t = Tensor([1.0])
        try:
            FD.alpha_dropout(t)
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

        try:
            FP.adaptive_avg_pool1d(t, 1)
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
        try:
            FP.adaptive_avg_pool2d(t, 1)
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
        try:
            FP.adaptive_avg_pool3d(t, 1)
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

        try:
            FP.adaptive_max_pool1d(t, 1, return_indices=True)
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
        try:
            FP.adaptive_max_pool2d(t, 1, return_indices=True)
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
        try:
            FP.adaptive_max_pool3d(t, 1, return_indices=True)
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
        try:
            FP.adaptive_max_pool1d(t, 1, return_indices=False)
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
        try:
            FP.adaptive_max_pool2d(t, 1, return_indices=False)
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
        try:
            FP.adaptive_max_pool3d(t, 1, return_indices=False)
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

        try:
            FP.fractional_max_pool2d(t, 1, output_ratio=0.5)
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
        try:
            FP.fractional_max_pool3d(t, 1, output_ratio=0.5)
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

        try:
            FP.avg_pool1d(t, 1)
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
        try:
            FP.avg_pool2d(t, 1)
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
        try:
            FP.avg_pool3d(t, 1)
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

        try:
            FP.fractional_max_pool2d(t, 1, return_indices=True)
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
        try:
            FP.fractional_max_pool3d(t, 1, return_indices=True)
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

        try:
            FU._get_nn_op("nonexistent_op")
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

        class MockOpsTupleNone:
            def __getattr__(self, name):
                return lambda *args, **kwargs: (None, None)

        with mock.patch("zero_torch.nn.functional_utils._nn", MockOpsTupleNone()):
            FU.adaptive_log_softmax_with_loss(t, t, 1, 1, [1])


def test_tensor_misc():
    from zero_torch.tensor import _to_tensor

    class Inner2:
        def tolist(self):
            return [1.0]

    class Inner1:
        @property
        def data(self):
            return Inner2()

    class Val:
        def __init__(self):
            self.data = Inner2()
            self.shape = (1,)

    class X:
        def __init__(self):
            self.data = Val()
            self.shape = (1,)

    import ml_switcheroo_compiler as ml_switcheroo

    from zero_torch.tracing import _tracer

    old_eager = ml_switcheroo.core.config.eager_mode
    ml_switcheroo.core.config.eager_mode = False

    try:
        with _tracer:
            try:
                _to_tensor(X())
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

            try:
                _to_tensor([1.0], dtype="float_nonexistent")
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
    finally:
        ml_switcheroo.core.config.eager_mode = old_eager


def test_tracing_stream():
    from ml_switcheroo_compiler.ir.core import IRNode

    from zero_torch.tracing import _tracer

    old_stream = ml_switcheroo.core.config.current_stream
    old_eager = ml_switcheroo.core.config.eager_mode

    try:
        ml_switcheroo.core.config.current_stream = "custom_stream"
        ml_switcheroo.core.config.eager_mode = False
        with _tracer:
            node = IRNode("test_node", "Add", attributes={}, shape_metadata=())
            node.stream = None
            node.stream = None
            _tracer.add_node(node)
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
    finally:
        ml_switcheroo.core.config.current_stream = old_stream
        ml_switcheroo.core.config.eager_mode = old_eager
