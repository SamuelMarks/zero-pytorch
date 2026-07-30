"""Sparse Tensors frontend module."""

from typing import Any

import zero_torch as torch
from zero_torch.tensor import Tensor, _wrap

try:
    import ml_switcheroo_compiler.ops.sparse as _ops
except ImportError:

    class _MockSparse:
        pass

    _ops = _MockSparse()


def _get_op(name):
    if not hasattr(_ops, name):
        raise NotImplementedError(f"Compiler backend missing sparse op: {name}")
    return getattr(_ops, name)


def sparse_coo_tensor(indices: Any, values: Any, size=None, **kwargs) -> Tensor:
    """Constructs a sparse COO tensor."""
    # Mocking sparse tensor creation for now. In reality this routes to compiler sparse config.
    import ml_switcheroo_compiler as ml_switcheroo
    from ml_switcheroo_compiler.core.tensor import Tensor as CompilerTensor
    from ml_switcheroo_compiler.core.tensor import TensorConfig

    indices_t = torch.tensor(indices)
    values_t = torch.tensor(values)

    # We construct a mock compiler tensor that has 'sparse' info in its config or attributes
    if size is None:
        if indices_t.shape[1] > 0:
            size = tuple((torch.max(indices_t, axis=1) + 1).tolist())
        else:
            size = ()

    # Use eager mode array creation if in eager mode
    if ml_switcheroo.core.config.eager_mode:
        dense_lst = torch.zeros(size, dtype=values_t.dtype).tolist()
        indices_lst = indices_t.tolist()
        values_lst = values_t.tolist()

        if len(indices_lst) > 0 and len(indices_lst[0]) > 0:
            if len(indices_lst) == 1:
                for i, v in zip(indices_lst[0], values_lst):  # pragma: no cover
                    dense_lst[i] = v  # pragma: no cover
            elif len(indices_lst) == 2:
                for i, j, v in zip(indices_lst[0], indices_lst[1], values_lst):
                    dense_lst[i][j] = v
            elif len(indices_lst) == 3:  # pragma: no cover
                for i, j, k, v in zip(  # pragma: no cover
                    indices_lst[0],
                    indices_lst[1],
                    indices_lst[2],
                    values_lst,  # pragma: no cover
                ):  # pragma: no cover
                    dense_lst[i][j][k] = v  # pragma: no cover
        return torch.tensor(dense_lst, dtype=values_t.dtype)

    config = TensorConfig(
        shape=size,
        dtype=values_t._tensor.config.dtype,
        device=values_t._tensor.config.device,
    )

    # Put kwargs into proxy data instead of config
    class SparseProxyData:
        def __init__(self, indices, values, kwargs):
            self.indices = indices
            self.values = values
            self.kwargs = kwargs
            self.id = "sparse_proxy"

    proxy = SparseProxyData(indices_t._tensor, values_t._tensor, kwargs)

    # Mock compiler tensor
    return _wrap(CompilerTensor(data=proxy, config=config))


def add(input: Tensor, other: Tensor, alpha=1, **kwargs) -> Tensor:
    """Adds other, scaled by alpha, to input."""
    from ml_switcheroo_compiler.core import config

    if config.eager_mode:
        import zero_torch as torch

        return torch.add(input, other * alpha, **kwargs)
    return _wrap(_get_op("sparse_add")(input._tensor, other._tensor, **kwargs))


def mm(mat1: Tensor, mat2: Tensor, **kwargs) -> Tensor:
    """Performs a matrix multiplication of the sparse matrix mat1 and dense matrix mat2."""
    from ml_switcheroo_compiler.core import config

    if config.eager_mode:
        import zero_torch as torch

        return torch.matmul(mat1, mat2, **kwargs)
    return _wrap(_get_op("smm")(mat1._tensor, mat2._tensor, **kwargs))
