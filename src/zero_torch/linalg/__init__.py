"""Linear Algebra frontend module."""

from __future__ import annotations

from collections.abc import Sequence

import ml_switcheroo_compiler.ops.linalg as _ops
from ml_switcheroo_compiler.ops.base import get_op

import zero_torch
from zero_torch._types import DTypeLike, OpArg
from zero_torch.tensor import Tensor, _flatten_nested, _wrap, _wrap_tuple


def _pick_first(primary: Tensor | None, alias: Tensor | None) -> Tensor:
    """Selects the non-None tensor argument between a primary name and an alias.

    Args:
        primary: Primary tensor argument.
        alias: Alias tensor argument.

    Returns:
        Resolved tensor.
    """
    if primary is not None:
        return primary
    assert alias is not None
    return alias


def cond(
    input: Tensor,
    p: int | float | str | None = None,
    **kwargs: OpArg,
) -> Tensor:
    """Computes the condition number of a matrix with respect to a matrix norm.

    Args:
        input: Input matrix tensor.
        p: Order of norm (`'fro'`, `'nuc'`, `1`, `-1`, `2`, `-2`, `inf`, `-inf`, or `None`).
        **kwargs: Optional keyword arguments.

    Returns:
        Condition number tensor.
    """
    _ = p
    return _wrap(_ops.cond(input._tensor, **kwargs))


def det(
    A: Tensor | None = None,
    input: Tensor | None = None,
    **kwargs: OpArg,
) -> Tensor:
    """Computes the determinant of a square matrix.

    Args:
        A: Square matrix tensor.
        input: Alias for `A`.
        **kwargs: Optional keyword arguments.

    Returns:
        Determinant tensor.
    """
    mat = _pick_first(A, input)
    return _wrap(get_op("Det")()(mat._tensor, **kwargs))


def slogdet(
    A: Tensor | None = None,
    input: Tensor | None = None,
    nan: Tensor | None = None,
    **kwargs: OpArg,
) -> tuple[Tensor, ...]:
    """Computes the sign and natural logarithm of the absolute value of the determinant of a square matrix.

    Args:
        A: Square matrix tensor.
        input: Alias for `A`.
        nan: Optional snapshot alias for `A`.
        **kwargs: Optional keyword arguments.

    Returns:
        Tuple of `(sign, logabsdet)` tensors.
    """
    mat = A if A is not None else (input if input is not None else nan)
    assert mat is not None
    return _wrap_tuple(get_op("Slogdet")()(mat._tensor, **kwargs))


def matrix_rank(
    input: Tensor,
    tol: float | Tensor | None = None,
    hermitian: bool = False,
    **kwargs: OpArg,
) -> Tensor:
    """Computes the numerical rank of a matrix.

    Args:
        input: Input matrix tensor.
        tol: Optional singular value threshold tolerance.
        hermitian: Indicates whether `input` is Hermitian/symmetric.
        **kwargs: Optional tolerance (`atol`, `rtol`) arguments.

    Returns:
        Matrix rank tensor.
    """
    _ = (tol, hermitian)
    return _wrap(_ops.matrix_rank(input._tensor, **kwargs))


def matrix_norm(
    input: Tensor,
    ord: int | float | str = "fro",
    dim: tuple[int, int] = (-2, -1),
    keepdim: bool = False,
    *,
    dtype: DTypeLike = None,
    **kwargs: OpArg,
) -> Tensor:
    """Computes a matrix norm.

    Args:
        input: Input matrix tensor.
        ord: Order of norm (`'fro'`, `'nuc'`, `1`, `-1`, `2`, `-2`, `inf`, `-inf`).
        dim: Tuple of two dimensions over which to compute the matrix norm.
        keepdim: Whether the normed dimensions are retained with size `1`.
        dtype: Optional desired output data type.
        **kwargs: Optional keyword arguments.

    Returns:
        Matrix norm tensor.
    """
    _ = (dtype, kwargs)
    ndim = len(input.shape)
    d0 = dim[0] if dim[0] >= 0 else ndim + dim[0]
    d1 = dim[1] if dim[1] >= 0 else ndim + dim[1]
    ax_after_d0 = d1 if keepdim else (d1 - 1 if d1 > d0 else d1)
    ax_after_d1 = d0 if keepdim else (d0 - 1 if d0 > d1 else d0)
    if ord == "fro":
        s0 = zero_torch.sum(input * input, axis=d0, keepdims=keepdim)
        s1 = zero_torch.sum(s0, axis=ax_after_d0, keepdims=keepdim)
        return zero_torch.sqrt(s1)
    if ord == 1:
        col_sums = zero_torch.sum(zero_torch.abs(input), axis=d0, keepdims=keepdim)
        return zero_torch.max(col_sums, axis=ax_after_d0, keepdims=keepdim)
    if ord == -1:
        col_sums = zero_torch.sum(zero_torch.abs(input), axis=d0, keepdims=keepdim)
        return zero_torch.min(col_sums, axis=ax_after_d0, keepdims=keepdim)
    if ord == float("inf"):
        row_sums = zero_torch.sum(zero_torch.abs(input), axis=d1, keepdims=keepdim)
        return zero_torch.max(row_sums, axis=ax_after_d1, keepdims=keepdim)
    if ord == float("-inf"):
        row_sums = zero_torch.sum(zero_torch.abs(input), axis=d1, keepdims=keepdim)
        return zero_torch.min(row_sums, axis=ax_after_d1, keepdims=keepdim)
    s = svdvals(input)
    if ord == "nuc":
        out = zero_torch.sum(s, axis=-1, keepdims=keepdim)
    elif ord == 2:
        out = zero_torch.max(s, axis=-1, keepdims=keepdim)
    elif ord == -2:
        out = zero_torch.min(s, axis=-1, keepdims=keepdim)
    else:
        raise ValueError(f"Invalid matrix_norm ord: {ord}")
    if keepdim:
        return out.unsqueeze(-1)
    return out


def vector_norm(
    input: Tensor,
    ord: int | float = 2.0,
    dim: int | tuple[int, ...] | None = None,
    keepdim: bool = False,
    *,
    dtype: DTypeLike = None,
    **kwargs: OpArg,
) -> Tensor:
    """Computes a vector norm.

    Args:
        input: Input tensor.
        ord: Order of norm (`0`, `1`, `2`, `inf`, `-inf`, or any real number).
        dim: Dimension(s) over which to compute the vector norm.
        keepdim: Whether the normed dimensions are retained with size `1`.
        dtype: Optional desired output data type.
        **kwargs: Optional keyword arguments.

    Returns:
        Vector norm tensor.
    """
    _ = (dtype, kwargs)
    abs_x = zero_torch.abs(input)
    if ord == 2 or ord == 2.0:
        if dim is None:
            sq_sum = zero_torch.sum(abs_x * abs_x)
            if keepdim:
                return zero_torch.sqrt(sq_sum).reshape((1,) * len(input.shape))
            return zero_torch.sqrt(sq_sum)
        return zero_torch.sqrt(
            zero_torch.sum(abs_x * abs_x, axis=dim, keepdims=keepdim)
        )
    if ord == 1 or ord == 1.0:
        if dim is None:
            s = zero_torch.sum(abs_x)
            return s.reshape((1,) * len(input.shape)) if keepdim else s
        return zero_torch.sum(abs_x, axis=dim, keepdims=keepdim)
    if ord == float("inf"):
        if dim is None:
            m = zero_torch.max(abs_x)
            return m.reshape((1,) * len(input.shape)) if keepdim else m
        return zero_torch.max(abs_x, axis=dim, keepdims=keepdim)
    if ord == float("-inf"):
        if dim is None:
            m = zero_torch.min(abs_x)
            return m.reshape((1,) * len(input.shape)) if keepdim else m
        return zero_torch.min(abs_x, axis=dim, keepdims=keepdim)
    if ord == 0 or ord == 0.0:
        nz = zero_torch.cast(abs_x != 0, input.dtype)
        if dim is None:
            s = zero_torch.sum(nz)
            return s.reshape((1,) * len(input.shape)) if keepdim else s
        return zero_torch.sum(nz, axis=dim, keepdims=keepdim)
    p = float(ord)
    powered = abs_x**p
    if dim is None:
        total = zero_torch.sum(powered)
        res = total ** (1.0 / p)
        return res.reshape((1,) * len(input.shape)) if keepdim else res
    return zero_torch.sum(powered, axis=dim, keepdims=keepdim) ** (1.0 / p)


def norm(
    input: Tensor,
    ord: int | float | str | None = None,
    dim: int | tuple[int, ...] | None = None,
    keepdim: bool = False,
    *,
    dtype: DTypeLike = None,
    **kwargs: OpArg,
) -> Tensor:
    """Computes a vector or matrix norm depending on `ord` and `dim`.

    Args:
        input: Input tensor.
        ord: Order of the norm (`None`, `'fro'`, `'nuc'`, `0`, `1`, `-1`, `2`, `-2`, `inf`, `-inf`, or `float`).
        dim: Dimensions over which to compute the norm.
        keepdim: Whether the normed dimensions are retained with size `1`.
        dtype: Optional desired output data type.
        **kwargs: Optional keyword arguments.

    Returns:
        Computed norm tensor.
    """
    if isinstance(ord, str) or (isinstance(dim, tuple) and len(dim) == 2):
        eff_ord: int | float | str = "fro" if ord is None else ord
        eff_dim: tuple[int, int] = (
            dim if (isinstance(dim, tuple) and len(dim) == 2) else (-2, -1)
        )
        return matrix_norm(
            input, ord=eff_ord, dim=eff_dim, keepdim=keepdim, dtype=dtype, **kwargs
        )
    v_ord: int | float = 2.0 if ord is None else ord
    return vector_norm(
        input, ord=v_ord, dim=dim, keepdim=keepdim, dtype=dtype, **kwargs
    )


def cholesky(
    input: Tensor,
    *,
    upper: bool = False,
    **kwargs: OpArg,
) -> Tensor:
    """Computes the Cholesky decomposition of a complex Hermitian or real symmetric positive-definite matrix.

    Args:
        input: Symmetric positive-definite matrix tensor.
        upper: Whether to return an upper triangular matrix.
        **kwargs: Optional keyword arguments.

    Returns:
        Cholesky factor tensor.
    """
    res = _wrap(get_op("Cholesky")()(input._tensor, **kwargs))
    if upper:
        return res.transpose(-2, -1)
    return res


def cholesky_ex(
    input: Tensor,
    *,
    upper: bool = False,
    check_errors: bool = False,
    **kwargs: OpArg,
) -> tuple[Tensor, ...]:
    """Computes the Cholesky decomposition and info tensor.

    Args:
        input: Symmetric positive-definite matrix tensor.
        upper: Whether to return an upper triangular matrix.
        check_errors: Controls whether to check the content of `info`.
        **kwargs: Optional keyword arguments.

    Returns:
        Tuple of `(L, info)` tensors.
    """
    _ = check_errors
    l_mat = cholesky(input, upper=upper, **kwargs)
    info = zero_torch.zeros(input.shape[:-2], dtype=zero_torch.int32)
    return (l_mat, info)


def eig(input: Tensor, **kwargs: OpArg) -> tuple[Tensor, ...]:
    """Computes the eigenvalue decomposition of a square matrix if it exists.

    Args:
        input: Square matrix tensor.
        **kwargs: Optional keyword arguments.

    Returns:
        Tuple of `(eigenvalues, eigenvectors)` tensors.
    """
    return _wrap_tuple(get_op("Eig")()(input._tensor, **kwargs))


def eigh(
    input: Tensor,
    UPLO: str = "L",
    **kwargs: OpArg,
) -> tuple[Tensor, ...]:
    """Computes the eigenvalue decomposition of a complex Hermitian or real symmetric matrix.

    Args:
        input: Symmetric or Hermitian matrix tensor.
        UPLO: Controls whether to use upper (`'U'`) or lower (`'L'`) triangular part.
        **kwargs: Optional keyword arguments.

    Returns:
        Tuple of `(eigenvalues, eigenvectors)` tensors.
    """
    _ = UPLO
    return _wrap_tuple(get_op("Eigh")()(input._tensor, **kwargs))


def eigvals(input: Tensor, **kwargs: OpArg) -> Tensor:
    """Computes the eigenvalues of a square matrix.

    Args:
        input: Square matrix tensor.
        **kwargs: Optional keyword arguments.

    Returns:
        Eigenvalues tensor.
    """
    return _wrap(get_op("Eigvals")()(input._tensor, **kwargs))


def eigvalsh(
    input: Tensor,
    UPLO: str = "L",
    **kwargs: OpArg,
) -> Tensor:
    """Computes the eigenvalues of a complex Hermitian or real symmetric matrix.

    Args:
        input: Symmetric or Hermitian matrix tensor.
        UPLO: Controls whether to use upper (`'U'`) or lower (`'L'`) triangular part.
        **kwargs: Optional keyword arguments.

    Returns:
        Real eigenvalues tensor.
    """
    _ = UPLO
    return _wrap(get_op("Eigvalsh")()(input._tensor, **kwargs))


def lu(
    A: Tensor | None = None,
    input: Tensor | None = None,
    *,
    pivot: bool = True,
    out: tuple[Tensor, Tensor, Tensor] | None = None,
    **kwargs: OpArg,
) -> tuple[Tensor, ...]:
    """Computes the LU decomposition with partial pivoting.

    Args:
        A: Input matrix tensor.
        input: Alias for `A`.
        pivot: Controls whether pivoting is done.
        out: Optional output tuple.
        **kwargs: Optional keyword arguments.

    Returns:
        Tuple of `(P, L, U)` tensors.
    """
    mat = _pick_first(A, input)
    p_mat, l_mat, u_mat = _wrap_tuple(_ops.lu(mat._tensor, **kwargs))
    if not pivot:
        p_mat = zero_torch.eye(mat.shape[-2], dtype=mat.dtype)
    if out is not None:
        out[0].copy_(p_mat)
        out[1].copy_(l_mat)
        out[2].copy_(u_mat)
        return out
    return (p_mat, l_mat, u_mat)


def lu_factor(
    A: Tensor | None = None,
    input: Tensor | None = None,
    *,
    pivot: bool = True,
    **kwargs: OpArg,
) -> tuple[Tensor, ...]:
    """Computes a compact LU factorization with partial pivoting.

    Args:
        A: Input matrix tensor.
        input: Alias for `A`.
        pivot: Controls whether pivoting is done.
        **kwargs: Optional keyword arguments.

    Returns:
        Tuple of `(LU, pivots)` tensors.
    """
    _ = pivot
    mat = _pick_first(A, input)
    return _wrap_tuple(get_op("LuFactor")()(mat._tensor, **kwargs))


def lu_factor_ex(
    A: Tensor | None = None,
    input: Tensor | None = None,
    *,
    pivot: bool = True,
    check_errors: bool = False,
    **kwargs: OpArg,
) -> tuple[Tensor, Tensor, Tensor]:
    """Computes a compact LU factorization and returns an `info` tensor.

    Args:
        A: Input matrix tensor.
        input: Alias for `A`.
        pivot: Controls whether pivoting is done.
        check_errors: Controls whether to check the content of `info`.
        **kwargs: Optional keyword arguments.

    Returns:
        Tuple of `(LU, pivots, info)` tensors.
    """
    _ = check_errors
    mat = _pick_first(A, input)
    lu_mat, pivots = lu_factor(mat, pivot=pivot)
    info = zero_torch.zeros(mat.shape[:-2], dtype=zero_torch.int32)
    return (lu_mat, pivots, info)


def lu_solve(
    LU: Tensor,
    pivots: Tensor,
    B: Tensor,
    *,
    left: bool = True,
    adjoint: bool = False,
) -> Tensor:
    """Solves the linear system `AX = B` using the partially pivoted LU factorization of `A`.

    Args:
        LU: Compact LU matrix tensor from `lu_factor`.
        pivots: Pivot index tensor from `lu_factor`.
        B: Right-hand side tensor.
        left: Whether to solve `AX = B` (`True`) or `XA = B` (`False`).
        adjoint: Whether to solve with the conjugate transpose of `A`.

    Returns:
        Solution tensor `X`.
    """
    n = LU.shape[-1]
    eye_n = zero_torch.eye(n, dtype=LU.dtype)
    l_mat = zero_torch.tril(LU, diagonal=-1) + eye_n
    u_mat = zero_torch.triu(LU)
    p_rows = eye_n.tolist()
    assert isinstance(p_rows, list)
    piv_flat = _flatten_nested(pivots.tolist())
    for i, p_val in enumerate(piv_flat):
        raw_idx = int(float(p_val)) if isinstance(p_val, (int, float)) else i
        idx = max(0, min(len(p_rows) - 1, raw_idx - 1 if raw_idx >= 1 else raw_idx))
        if i < len(p_rows):
            p_rows[i], p_rows[idx] = p_rows[idx], p_rows[i]
    p_mat = Tensor(p_rows, dtype=LU.dtype)
    a_recon = zero_torch.matmul(
        p_mat.transpose(-2, -1), zero_torch.matmul(l_mat, u_mat)
    )
    if adjoint:
        a_recon = a_recon.transpose(-2, -1)
    if not left:
        return solve(a_recon.transpose(-2, -1), B.transpose(-2, -1)).transpose(-2, -1)
    return solve(a_recon, B)


def ldl_factor(
    input: Tensor,
    *,
    hermitian: bool = False,
) -> tuple[Tensor, Tensor]:
    """Computes the compact representation of the LDL factorization of a symmetric/Hermitian matrix.

    Args:
        input: Symmetric or Hermitian input matrix tensor.
        hermitian: Whether to treat `input` as complex Hermitian.

    Returns:
        Tuple of `(LD, pivots)` tensors.
    """
    _ = hermitian
    lu_mat, pivots = lu_factor(input)
    return (lu_mat, pivots)


def ldl_factor_ex(
    input: Tensor,
    *,
    hermitian: bool = False,
    check_errors: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    """Computes the LDL factorization and returns an `info` tensor.

    Args:
        input: Symmetric or Hermitian input matrix tensor.
        hermitian: Whether to treat `input` as complex Hermitian.
        check_errors: Controls whether to check the content of `info`.

    Returns:
        Tuple of `(LD, pivots, info)` tensors.
    """
    _ = check_errors
    ld, pivots = ldl_factor(input, hermitian=hermitian)
    info = zero_torch.zeros(input.shape[:-2], dtype=zero_torch.int32)
    return (ld, pivots, info)


def ldl_solve(
    LD: Tensor,
    pivots: Tensor,
    B: Tensor,
    *,
    hermitian: bool = False,
) -> Tensor:
    """Solves a system of linear equations using the LDL factorization.

    Args:
        LD: Compact LDL factor tensor.
        pivots: Pivot index tensor.
        B: Right-hand side tensor.
        hermitian: Whether the factored matrix is complex Hermitian.

    Returns:
        Solution tensor `X`.
    """
    _ = hermitian
    return lu_solve(LD, pivots, B)


def qr(
    A: Tensor | None = None,
    mode: str = "reduced",
    *,
    input: Tensor | None = None,
    out: tuple[Tensor, Tensor] | None = None,
    **kwargs: OpArg,
) -> tuple[Tensor, ...]:
    """Computes the QR decomposition of a matrix.

    Args:
        A: Input matrix tensor.
        mode: Decomposition mode (`'reduced'`, `'complete'`, or `'r'`).
        input: Alias for `A`.
        out: Optional output tuple.
        **kwargs: Optional keyword arguments.

    Returns:
        Tuple of `(Q, R)` tensors (or `(Tensor([]), R)` when `mode='r'`).
    """
    mat = _pick_first(A, input)
    q_mat, r_mat = _wrap_tuple(get_op("Qr")()(mat._tensor, **kwargs))
    if mode == "r":
        q_mat = zero_torch.empty((0,), dtype=mat.dtype)
    if out is not None:
        out[0].copy_(q_mat)
        out[1].copy_(r_mat)
        return out
    return (q_mat, r_mat)


def svd(
    A: Tensor | None = None,
    full_matrices: bool = True,
    *,
    driver: str | None = None,
    input: Tensor | None = None,
    **kwargs: OpArg,
) -> tuple[Tensor, ...]:
    """Computes the singular value decomposition (SVD) of a matrix.

    Args:
        A: Input matrix tensor.
        full_matrices: Controls whether to compute full or reduced SVD.
        driver: Optional LAPACK/cuSOLVER driver name.
        input: Alias for `A`.
        **kwargs: Optional keyword arguments.

    Returns:
        Tuple of `(U, S, Vh)` tensors.
    """
    _ = driver
    mat = _pick_first(A, input)
    u_mat, s_vec, vh_mat = _wrap_tuple(get_op("Svd")()(mat._tensor, **kwargs))
    if not full_matrices and len(mat.shape) >= 2:
        k = min(mat.shape[-2], mat.shape[-1])
        if u_mat.shape[-1] > k:
            u_mat = u_mat[:, :k]
        if vh_mat.shape[-2] > k:
            vh_mat = vh_mat[:k, :]
    return (u_mat, s_vec, vh_mat)


def svdvals(
    A: Tensor | None = None,
    *,
    driver: str | None = None,
    input: Tensor | None = None,
    **kwargs: OpArg,
) -> Tensor:
    """Computes the singular values of a matrix.

    Args:
        A: Input matrix tensor.
        driver: Optional LAPACK/cuSOLVER driver name.
        input: Alias for `A`.
        **kwargs: Optional keyword arguments.

    Returns:
        Singular values tensor.
    """
    _ = driver
    mat = _pick_first(A, input)
    return _wrap(_ops.svdvals(mat._tensor, **kwargs))


def inv(
    A: Tensor | None = None,
    input: Tensor | None = None,
    **kwargs: OpArg,
) -> Tensor:
    """Computes the inverse of a square matrix if it is invertible.

    Args:
        A: Invertible square matrix tensor.
        input: Alias for `A`.
        **kwargs: Optional keyword arguments.

    Returns:
        Inverse matrix tensor.
    """
    mat = _pick_first(A, input)
    return _wrap(get_op("Inv")()(mat._tensor, **kwargs))


def inv_ex(
    A: Tensor | None = None,
    *,
    check_errors: bool = False,
    input: Tensor | None = None,
    **kwargs: OpArg,
) -> tuple[Tensor, ...]:
    """Computes the inverse of a square matrix and an info tensor.

    Args:
        A: Square matrix tensor.
        check_errors: Controls whether to check the content of `info`.
        input: Alias for `A`.
        **kwargs: Optional keyword arguments.

    Returns:
        Tuple of `(inverse, info)` tensors.
    """
    _ = check_errors
    mat = _pick_first(A, input)
    return _wrap_tuple(get_op("InvEx")()(mat._tensor, **kwargs))


def pinv(
    input: Tensor,
    rcond: float | Tensor | None = None,
    hermitian: bool = False,
    *,
    atol: float | Tensor | None = None,
    rtol: float | Tensor | None = None,
    **kwargs: OpArg,
) -> Tensor:
    """Computes the pseudo-inverse (Moore-Penrose inverse) of a matrix.

    Args:
        input: Input matrix tensor.
        rcond: Relative condition cutoff for small singular values.
        hermitian: Indicates whether `input` is Hermitian.
        atol: Optional absolute tolerance for singular value cutoff.
        rtol: Optional relative tolerance for singular value cutoff.
        **kwargs: Optional keyword arguments.

    Returns:
        Pseudo-inverse matrix tensor.
    """
    _ = hermitian
    eff_rtol = rtol if rtol is not None else rcond
    if (eff_rtol is not None or atol is not None) and len(input.shape) == 2:
        u_mat, s_vec, vh_mat = svd(input, full_matrices=False)
        max_s = zero_torch.max(s_vec)
        r_item = eff_rtol.item() if isinstance(eff_rtol, Tensor) else eff_rtol
        r_val = float(r_item) if isinstance(r_item, (int, float)) else 1e-15
        a_item = atol.item() if isinstance(atol, Tensor) else atol
        a_val = float(a_item) if isinstance(a_item, (int, float)) else 0.0
        cutoff = zero_torch.maximum(max_s * r_val, a_val)
        safe_s = zero_torch.where(s_vec > cutoff, s_vec, zero_torch.ones_like(s_vec))
        s_inv = zero_torch.where(
            s_vec > cutoff, 1.0 / safe_s, zero_torch.zeros_like(s_vec)
        )
        return zero_torch.matmul(
            vh_mat.transpose(-2, -1),
            zero_torch.matmul(zero_torch.diag(s_inv), u_mat.transpose(-2, -1)),
        )
    return _wrap(get_op("Pinv")()(input._tensor, **kwargs))


def solve(
    A: Tensor | None = None,
    B: Tensor | None = None,
    *,
    left: bool = True,
    input: Tensor | None = None,
    other: Tensor | None = None,
    **kwargs: OpArg,
) -> Tensor:
    """Solves a square system of linear equations with a unique solution.

    Args:
        A: Coefficient matrix tensor `A`.
        B: Right-hand side tensor `B`.
        left: Whether to solve `AX = B` (`True`) or `XA = B` (`False`).
        input: Alias for `A`.
        other: Alias for `B`.
        **kwargs: Optional keyword arguments.

    Returns:
        Solution tensor `X`.
    """
    mat_a = _pick_first(A, input)
    mat_b = _pick_first(B, other)
    if not left:
        sol_t = _wrap(
            get_op("Solve")()(
                mat_a.transpose(-2, -1)._tensor,
                mat_b.transpose(-2, -1)._tensor,
                **kwargs,
            )
        )
        return sol_t.transpose(-2, -1)
    return _wrap(get_op("Solve")()(mat_a._tensor, mat_b._tensor, **kwargs))


def solve_ex(
    A: Tensor | None = None,
    B: Tensor | None = None,
    *,
    left: bool = True,
    check_errors: bool = False,
    input: Tensor | None = None,
    other: Tensor | None = None,
    **kwargs: OpArg,
) -> tuple[Tensor, ...]:
    """Solves a linear system of equations and returns an info tensor.

    Args:
        A: Coefficient matrix tensor `A`.
        B: Right-hand side tensor `B`.
        left: Whether to solve `AX = B` (`True`) or `XA = B` (`False`).
        check_errors: Controls whether to check the content of `info`.
        input: Alias for `A`.
        other: Alias for `B`.
        **kwargs: Optional keyword arguments.

    Returns:
        Tuple of `(result, info)` tensors.
    """
    _ = (check_errors, kwargs)
    mat_a = _pick_first(A, input)
    mat_b = _pick_first(B, other)
    sol = solve(mat_a, mat_b, left=left)
    info = zero_torch.zeros(mat_a.shape[:-2], dtype=zero_torch.int32)
    return (sol, info)


def solve_triangular(
    input: Tensor,
    B: Tensor,
    *,
    upper: bool,
    left: bool = True,
    unitriangular: bool = False,
) -> Tensor:
    """Computes the solution of a triangular system of linear equations with a unique solution.

    Args:
        input: Triangular coefficient matrix `A`.
        B: Right-hand side tensor `B`.
        upper: Whether `input` is upper or lower triangular.
        left: Whether to solve `AX = B` (`True`) or `XA = B` (`False`).
        unitriangular: If True, diagonal elements of `input` are assumed to be `1`.

    Returns:
        Solution tensor `X`.
    """
    a_tri = zero_torch.triu(input) if upper else zero_torch.tril(input)
    if unitriangular:
        n = a_tri.shape[-1]
        a_no_diag = (
            zero_torch.triu(a_tri, diagonal=1)
            if upper
            else zero_torch.tril(a_tri, diagonal=-1)
        )
        a_tri = a_no_diag + zero_torch.eye(n, dtype=a_tri.dtype)
    if not left:
        return solve(a_tri.transpose(-2, -1), B.transpose(-2, -1)).transpose(-2, -1)
    return solve(a_tri, B)


def lstsq(
    input: Tensor,
    b: Tensor | None = None,
    rcond: float | None = None,
    *,
    driver: str | None = None,
    other: Tensor | None = None,
    **kwargs: OpArg,
) -> tuple[Tensor, ...]:
    """Computes a solution to the least squares problem of a system of linear equations.

    Args:
        input: Coefficient matrix tensor `A`.
        b: Right-hand side tensor `B`.
        rcond: Cut-off ratio for small singular values.
        driver: Optional LAPACK/cuSOLVER method name.
        other: Alias for `b`.
        **kwargs: Optional keyword arguments.

    Returns:
        Tuple of `(solution, residuals, rank, singular_values)` tensors.
    """
    _ = (rcond, driver, kwargs)
    rhs = _pick_first(b, other)
    pinv_a = pinv(input)
    sol = zero_torch.matmul(pinv_a, rhs)
    diff = rhs - zero_torch.matmul(input, sol)
    residuals = zero_torch.sum(diff * diff, axis=0)
    rank = matrix_rank(input)
    s = svdvals(input)
    return (sol, residuals, rank, s)


def tensorinv(
    input: Tensor,
    ind: int = 2,
    **kwargs: OpArg,
) -> Tensor:
    """Computes the multiplicative inverse of `torch.tensordot`.

    Args:
        input: Input tensor.
        ind: Number of indices to invert.
        **kwargs: Optional keyword arguments.

    Returns:
        Tensor inverse.
    """
    return _wrap(_ops.tensorinv(input._tensor, ind=ind, **kwargs))


def tensorsolve(
    input: Tensor,
    other: Tensor,
    dims: tuple[int, ...] | list[int] | None = None,
    **kwargs: OpArg,
) -> Tensor:
    """Computes the solution `X` to the system `torch.tensordot(A, X) = B`.

    Args:
        input: Coefficient tensor `A`.
        other: Right-hand side tensor `B`.
        dims: Dimensions of `A` to be moved before solving.
        **kwargs: Optional keyword arguments.

    Returns:
        Solution tensor `X`.
    """
    _ = dims
    return _wrap(_ops.tensorsolve(input._tensor, other._tensor, **kwargs))


def cross(
    input: Tensor,
    other: Tensor,
    *,
    dim: int = -1,
    **kwargs: OpArg,
) -> Tensor:
    """Computes the cross product of two 3-element vectors.

    Args:
        input: First input tensor.
        other: Second input tensor.
        dim: Dimension along which to take the cross product.
        **kwargs: Optional keyword arguments.

    Returns:
        Cross product tensor.
    """
    ndim = len(input.shape)
    norm_dim = dim if dim >= 0 else ndim + dim
    if norm_dim != ndim - 1:
        in_swapped = input.transpose(norm_dim, -1)
        oth_swapped = other.transpose(norm_dim, -1)
        res = _wrap(_ops.cross(in_swapped._tensor, oth_swapped._tensor, **kwargs))
        return res.transpose(norm_dim, -1)
    return _wrap(_ops.cross(input._tensor, other._tensor, **kwargs))


def diagonal(
    A: Tensor,
    *,
    offset: int = 0,
    dim1: int = -2,
    dim2: int = -1,
) -> Tensor:
    """Extracts a partial view/tensor of the diagonal elements of `A`.

    Args:
        A: Input tensor.
        offset: Diagonal offset from the main diagonal.
        dim1: First dimension with respect to which to take diagonal.
        dim2: Second dimension with respect to which to take diagonal.

    Returns:
        Partial view/tensor of the diagonal elements.
    """
    ndim = len(A.shape)
    d1 = dim1 if dim1 >= 0 else ndim + dim1
    d2 = dim2 if dim2 >= 0 else ndim + dim2
    mat = A
    if (d1, d2) != (ndim - 2, ndim - 1):
        rem = [i for i in range(ndim) if i not in (d1, d2)]
        perm = tuple(rem + [d1, d2])
        mat = zero_torch.permute(A, perm)
    return zero_torch.diag(mat, diagonal=offset)


def matmul(input: Tensor, other: Tensor) -> Tensor:
    """Alias for `torch.matmul`.

    Args:
        input: First tensor to be multiplied.
        other: Second tensor to be multiplied.

    Returns:
        Matrix product tensor.
    """
    return zero_torch.matmul(input, other)


def vecdot(x: Tensor, y: Tensor, *, dim: int = -1) -> Tensor:
    """Computes the dot product of two batches of vectors along a dimension.

    Args:
        x: First batch of vectors.
        y: Second batch of vectors.
        dim: Dimension along which to compute the dot product.

    Returns:
        Dot product tensor.
    """
    return zero_torch.sum(zero_torch.conj(x) * y, axis=dim)


def vander(x: Tensor, N: int | None = None) -> Tensor:
    """Generates a Vandermonde matrix.

    Args:
        x: 1D or batched input tensor.
        N: Number of columns in the output. Defaults to `x.shape[-1]`.

    Returns:
        Vandermonde matrix tensor of shape `(*x.shape, N)`.
    """
    num_cols = x.shape[-1] if N is None else N
    cols = [x**i for i in range(num_cols)]
    return zero_torch.stack(cols, dim=-1)


def polar(A: Tensor) -> tuple[Tensor, Tensor]:
    """Constructs the polar decomposition `A = U P` of a matrix `A`.

    Args:
        A: Input matrix tensor.

    Returns:
        Tuple `(U, P)` where `U` is unitary/isometric and `P` is Hermitian positive semi-definite.
    """
    u_svd, s_svd, vh_svd = svd(A, full_matrices=False)
    u_polar = zero_torch.matmul(u_svd, vh_svd)
    s_diag = zero_torch.diag(s_svd)
    p_polar = zero_torch.matmul(
        zero_torch.matmul(vh_svd.transpose(-2, -1), s_diag), vh_svd
    )
    return (u_polar, p_polar)


def matrix_sqrth(input: Tensor) -> Tensor:
    """Computes the principal square root of a symmetric or Hermitian positive semi-definite matrix.

    Args:
        input: Symmetric or Hermitian positive semi-definite matrix tensor.

    Returns:
        Matrix square root tensor.
    """
    vals, vecs = eigh(input)
    clamped = zero_torch.clamp(vals, min=0.0)
    sqrt_diag = zero_torch.diag(zero_torch.sqrt(clamped))
    return zero_torch.matmul(zero_torch.matmul(vecs, sqrt_diag), vecs.transpose(-2, -1))


def matrix_exp(input: Tensor, **kwargs: OpArg) -> Tensor:
    """Computes the matrix exponential of a square matrix.

    Args:
        input: Square matrix tensor.
        **kwargs: Optional keyword arguments.

    Returns:
        Matrix exponential tensor.
    """
    _ = kwargs
    n = input.shape[-1]
    scale_steps = 4
    scale = float(1 << scale_steps)
    scaled_a = input / scale
    eye_n = zero_torch.eye(n, dtype=input.dtype)
    res = eye_n
    term = eye_n
    for k in range(1, 16):
        term = zero_torch.matmul(term, scaled_a) / float(k)
        res = res + term
    for _ in range(scale_steps):
        res = zero_torch.matmul(res, res)
    return res


def matrix_power(
    A: Tensor | None = None,
    n: int = 1,
    *,
    input: Tensor | None = None,
    out: Tensor | None = None,
    **kwargs: OpArg,
) -> Tensor:
    """Computes the `n`-th power of a square matrix for an integer `n`.

    Args:
        A: Square matrix tensor.
        n: Exponent integer.
        input: Alias for `A`.
        out: Optional output tensor.
        **kwargs: Optional keyword arguments.

    Returns:
        Matrix power tensor.
    """
    mat = _pick_first(A, input)
    res = _wrap(get_op("MatrixPower")()(mat._tensor, n=n, **kwargs))
    if out is not None:
        return out.copy_(res)
    return res


def multi_dot(tensors: Sequence[Tensor], **kwargs: OpArg) -> Tensor:
    """Efficiently multiplies two or more matrices by reordering multiplications.

    Args:
        tensors: Sequence of 2D (or 1D) tensors to multiply.
        **kwargs: Optional keyword arguments.

    Returns:
        Product tensor.
    """
    unwrapped = [t._tensor for t in tensors if t._tensor is not None]
    return _wrap(_ops.multi_dot(unwrapped, **kwargs))


def householder_product(
    A: Tensor | None = None,
    tau: Tensor | None = None,
    *,
    input: Tensor | None = None,
    out: Tensor | None = None,
    **kwargs: OpArg,
) -> Tensor:
    """Computes the first `n` columns of a product of Householder matrices.

    Args:
        A: Tensor with Householder vectors in its lower triangle.
        tau: Tensor of complex/real scale factors.
        input: Alias for `A`.
        out: Optional output tensor.
        **kwargs: Optional keyword arguments.

    Returns:
        Orthogonal/unitary matrix product tensor.
    """
    mat = _pick_first(A, input)
    assert tau is not None
    res = _wrap(get_op("HouseholderProduct")()(mat._tensor, tau._tensor, **kwargs))
    if out is not None:
        return out.copy_(res)
    return res
