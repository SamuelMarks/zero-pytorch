"""Sparse Tensors frontend module."""

from __future__ import annotations

from collections.abc import Sequence

import ml_switcheroo_compiler as ml_switcheroo
import ml_switcheroo_compiler.ops.sparse as _ops
from ml_switcheroo_compiler.core.tensor import Tensor as CompilerTensor
from ml_switcheroo_compiler.core.tensor import TensorConfig

import zero_torch as torch
from zero_torch._types import DTypeLike, OpArg, Scalar, TensorLike
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
    *,
    dtype: DTypeLike = None,
    device: OpArg = None,
    requires_grad: bool = False,
    check_invariants: bool | None = None,
    is_coalesced: bool | None = None,
    **kwargs: OpArg,
) -> Tensor:
    """Constructs a sparse tensor in Coordinate (COO) format.

    Args:
        indices: 2D array/tensor of coordinate indices of shape `(ndim, nnz)`.
        values: 1D array/tensor of values of length `nnz`.
        size: Optional dense shape tuple of the sparse tensor.
        dtype: Desired data type of returned tensor.
        device: Desired device of returned tensor.
        requires_grad: If autograd should record operations on the returned tensor.
        check_invariants: If Python/C++ invariants should be checked.
        is_coalesced: Whether the input indices are already coalesced.
        **kwargs: Additional tensor creation arguments.

    Returns:
        Constructed sparse COO tensor.
    """
    _ = (device, check_invariants, is_coalesced)
    indices_t = torch.tensor(indices)
    values_t = torch.tensor(values, dtype=dtype)

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
        res = torch.tensor(dense_lst, dtype=values_t.dtype, requires_grad=requires_grad)
        res._is_sparse = True
        res._sparse_indices = indices_t
        res._sparse_values = values_t
        res.layout = "torch.sparse_coo"
        return res

    v_tensor = values_t._tensor
    assert v_tensor is not None
    cfg = TensorConfig(
        shape=resolved_size,
        dtype=v_tensor.config.dtype,
        device=v_tensor.config.device,
    )
    proxy = SparseProxyData(indices_t._tensor, v_tensor, dict(kwargs))
    out_t = _wrap(CompilerTensor(data=proxy, config=cfg))
    out_t._is_sparse = True
    out_t._sparse_indices = indices_t
    out_t._sparse_values = values_t
    out_t.layout = "torch.sparse_coo"
    return out_t


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
        res = torch.add(input, other * alpha)
        if input._is_sparse and other._is_sparse:
            res._is_sparse = True
            res.layout = input.layout
        return res
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
        return torch.matmul(mat1, mat2)
    return _wrap(_ops.smm(mat1._tensor, mat2._tensor, **kwargs))


def addmm(
    mat: Tensor,
    mat1: Tensor,
    mat2: Tensor,
    *,
    beta: float = 1.0,
    alpha: float = 1.0,
) -> Tensor:
    """Performs `beta * mat + alpha * (mat1 @ mat2)` where `mat1` is sparse.

    Args:
        mat: Dense matrix to be added.
        mat1: Sparse matrix to be multiplied.
        mat2: Dense matrix to be multiplied.
        beta: Scaling factor for `mat`.
        alpha: Scaling factor for `mat1 @ mat2`.

    Returns:
        Resulting dense matrix tensor.
    """
    return mat * beta + mm(mat1, mat2) * alpha


def sum(
    input: Tensor,
    dim: int | tuple[int, ...] | None = None,
    *,
    dtype: DTypeLike = None,
) -> Tensor:
    """Computes the sum of elements of a sparse tensor along `dim`.

    Args:
        input: Input sparse tensor.
        dim: Dimension or tuple of dimensions to reduce.
        dtype: Optional desired output data type.

    Returns:
        Reduced sum tensor.
    """
    return torch.sum(input, dim=dim, dtype=dtype)


def softmax(
    input: Tensor,
    dim: int,
    *,
    dtype: DTypeLike = None,
) -> Tensor:
    """Applies a softmax function over the specified `dim` of a sparse tensor.

    Args:
        input: Input sparse tensor.
        dim: Dimension along which softmax will be computed.
        dtype: Optional desired output data type.

    Returns:
        Sparse tensor with softmax probabilities.
    """
    res = torch.nn.functional.softmax(input, dim=dim, dtype=dtype)
    res._is_sparse = input._is_sparse
    res.layout = input.layout
    return res


def log_softmax(
    input: Tensor,
    dim: int,
    *,
    dtype: DTypeLike = None,
) -> Tensor:
    """Applies a log-softmax function over the specified `dim` of a sparse tensor.

    Args:
        input: Input sparse tensor.
        dim: Dimension along which log_softmax will be computed.
        dtype: Optional desired output data type.

    Returns:
        Sparse tensor with log-softmax values.
    """
    res = torch.nn.functional.log_softmax(input, dim=dim, dtype=dtype)
    res._is_sparse = input._is_sparse
    res.layout = input.layout
    return res
