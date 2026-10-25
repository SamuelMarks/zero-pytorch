"""Sparse Tensors frontend module."""

from __future__ import annotations

from collections.abc import Sequence

import ml_switcheroo_compiler as ml_switcheroo
import ml_switcheroo_compiler.ops.sparse as _ops
from ml_switcheroo_compiler.core.tensor import Tensor as CompilerTensor
from ml_switcheroo_compiler.core.tensor import TensorConfig

import zero_torch as torch
from zero_torch._types import OpArg, Scalar, TensorLike
from zero_torch.tensor import Tensor, _wrap


class SparseProxyData:
    """Proxy payload representing sparse COO tensor components during graph tracing."""

    def __init__(
        self,
        indices: CompilerTensor | None,
        values: CompilerTensor | None,
        kwargs: dict[str, OpArg],
    ) -> None:
        """Initializes SparseProxyData.

        Args:
            indices: Compiler tensor of sparse coordinate indices.
            values: Compiler tensor of non-zero values.
            kwargs: Additional keyword attributes.
        """
        self.indices: CompilerTensor | None = indices
        self.values: CompilerTensor | None = values
        self.kwargs: dict[str, OpArg] = kwargs
        self.id: str = "sparse_proxy"


def sparse_coo_tensor(
    indices: TensorLike,
    values: TensorLike,
    size: Sequence[int] | None = None,
    **kwargs: OpArg,
) -> Tensor:
    """Constructs a sparse tensor in Coordinate (COO) format.

    Args:
        indices: 2D array/tensor of coordinate indices of shape `(ndim, nnz)`.
        values: 1D array/tensor of values of length `nnz`.
        size: Optional dense shape tuple of the sparse tensor.
        **kwargs: Additional tensor creation arguments (`dtype`, `device`, `requires_grad`).

    Returns:
        Constructed sparse COO tensor.
    """
    indices_t = torch.tensor(indices)
    values_t = torch.tensor(values)

    resolved_size: tuple[int, ...]
    if size is None:
        if len(indices_t.shape) >= 2 and indices_t.shape[1] > 0:
            max_idx = (torch.max(indices_t, axis=1) + 1).tolist()
            assert isinstance(max_idx, list)
            resolved_size = tuple(
                int(x) for x in max_idx if isinstance(x, (int, float))
            )
        else:
            resolved_size = ()
    else:
        resolved_size = tuple(int(s) for s in size)

    if ml_switcheroo.core.config.eager_mode:
        dense_lst = torch.zeros(resolved_size, dtype=values_t.dtype).tolist()
        indices_lst = indices_t.tolist()
        values_lst = values_t.tolist()

        if (
            isinstance(dense_lst, list)
            and isinstance(indices_lst, list)
            and isinstance(values_lst, list)
            and len(indices_lst) > 0
            and isinstance(indices_lst[0], list)
            and len(indices_lst[0]) > 0
        ):
            if len(indices_lst) == 1:
                for i_val, v_val in zip(indices_lst[0], values_lst):
                    assert isinstance(i_val, int)
                    dense_lst[i_val] = v_val
            elif len(indices_lst) == 2 and isinstance(indices_lst[1], list):
                for i_val, j_val, v_val in zip(
                    indices_lst[0], indices_lst[1], values_lst
                ):
                    assert isinstance(i_val, int) and isinstance(j_val, int)
                    row = dense_lst[i_val]
                    assert isinstance(row, list)
                    row[j_val] = v_val
            else:
                assert isinstance(indices_lst[1], list)
                assert isinstance(indices_lst[2], list)
                for i_val, j_val, k_val, v_val in zip(
                    indices_lst[0],
                    indices_lst[1],
                    indices_lst[2],
                    values_lst,
                ):
                    assert isinstance(i_val, int)
                    assert isinstance(j_val, int)
                    assert isinstance(k_val, int)
                    plane = dense_lst[i_val]
                    assert isinstance(plane, list)
                    row = plane[j_val]
                    assert isinstance(row, list)
                    row[k_val] = v_val
        return torch.tensor(dense_lst, dtype=values_t.dtype)

    v_tensor = values_t._tensor
    assert v_tensor is not None
    cfg = TensorConfig(
        shape=resolved_size,
        dtype=v_tensor.config.dtype,
        device=v_tensor.config.device,
    )
    proxy = SparseProxyData(indices_t._tensor, v_tensor, dict(kwargs))
    return _wrap(CompilerTensor(data=proxy, config=cfg))


def add(input: Tensor, other: Tensor, alpha: Scalar = 1, **kwargs: OpArg) -> Tensor:
    """Adds `other`, scaled by `alpha`, to `input`.

    Args:
        input: First sparse/dense input tensor.
        other: Second tensor to scale and add.
        alpha: Scaling factor for `other`.
        **kwargs: Optional keyword arguments.

    Returns:
        Sum tensor `input + alpha * other`.
    """
    if ml_switcheroo.core.config.eager_mode:
        return torch.add(input, other * alpha)
    return _wrap(_ops.sparse_add(input._tensor, other._tensor, **kwargs))


def mm(mat1: Tensor, mat2: Tensor, **kwargs: OpArg) -> Tensor:
    """Performs a matrix multiplication of the sparse matrix `mat1` and dense matrix `mat2`.

    Args:
        mat1: Sparse matrix tensor.
        mat2: Dense matrix tensor.
        **kwargs: Optional keyword arguments.

    Returns:
        Matrix product tensor.
    """
    if ml_switcheroo.core.config.eager_mode:
        return torch.matmul(mat1, mat2, **kwargs)
    return _wrap(_ops.smm(mat1._tensor, mat2._tensor, **kwargs))
