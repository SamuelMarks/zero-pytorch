import zero_torch
from zero_torch import Tensor


def test_tensor_fix4():
    # boolean
    _ = Tensor(True)
    _ = Tensor(False)

    # get_shape_and_dt except
    _ = Tensor([[1], [2, 3]])  # Jagged list raises Exception in array conversion?

    # array_to_tensor except
    class BadArray:
        def __array__(self, dtype=None):
            raise ValueError("bad")

    try:
        Tensor(BadArray())
    except Exception:
        pass

    # __gt__, __le__, __ge__, __eq__
    t = Tensor([1.0])
    _ = t > 1
    _ = t <= 1
    _ = t >= 1
    _ = t == 1
    _ = t != 1
    _ = t < 1

    # detach None
    t_empty = Tensor([])
    t_empty._tensor = None
    t_empty.detach()

    # _wrap list and tuple
    from zero_torch.tensor import _wrap

    _wrap([t])
    _wrap((t,))

    # Val with data.tolist
    class MockData:
        def tolist(self):
            return [1]

    class MockVal:
        data = MockData()

    t._tensor = MockVal()
    try:
        Tensor(t)
    except Exception:
        pass


def test_init_fix4():
    from ml_switcheroo_compiler.core.config import config

    # Random ops eager
    zero_torch.rand(2)
    zero_torch.rand(2, 3)
    zero_torch.randn(2)
    zero_torch.randn(2, 3)
    zero_torch.randint(0, 10, (2,))
    zero_torch.randint(0, 10, size=(2,))

    # Random ops tracing
    config.eager_mode = False
    try:
        from zero_torch.tracing import _tracer

        _tracer.start_tracing()
        zero_torch.rand(2)
        zero_torch.randn(2)
        zero_torch.randint(0, 10, (2,))
        _tracer.stop_tracing()
    finally:
        config.eager_mode = True
