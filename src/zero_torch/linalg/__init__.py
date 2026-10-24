"""Linear Algebra frontend module."""

from __future__ import annotations

from collections.abc import Sequence

import ml_switcheroo_compiler.ops.linalg as _ops
from ml_switcheroo_compiler.ops.base import get_op

import zero_torch
from zero_torch._types import OpArg
from zero_torch.tensor import Tensor, _wrap, _wrap_tuple


def cond(input: Tensor, **kwargs: OpArg) -> Tensor:
    """Computes the condition number of a matrix with respect to a matrix norm.

    Args:
        input: Input matrix tensor.
        **kwargs: Optional norm order (`p`) arguments.

    Returns:
        Condition number tensor.
    """
    return _wrap(_ops.cond(input._tensor, **kwargs))


def det(input: Tensor, **kwargs: OpArg) -> Tensor:
    """Computes the determinant of a square matrix.

    Args:
        input: Square matrix tensor.
        **kwargs: Optional keyword arguments.

    Returns:
        Determinant tensor.
    """
    return _wrap(get_op("Det")()(input._tensor, **kwargs))


def slogdet(input: Tensor, **kwargs: OpArg) -> tuple[Tensor, ...]:
    """Computes the sign and natural logarithm of the absolute value of the determinant of a square matrix.

    Args:
        input: Square matrix tensor.
        **kwargs: Optional keyword arguments.

    Returns:
        Tuple of `(sign, logabsdet)` tensors.
    """
    return _wrap_tuple(get_op("Slogdet")()(input._tensor, **kwargs))


def matrix_rank(input: Tensor, **kwargs: OpArg) -> Tensor:
    """Computes the numerical rank of a matrix.

    Args:
        input: Input matrix tensor.
        **kwargs: Optional tolerance (`atol`, `rtol`, `hermitian`) arguments.

    Returns:
        Matrix rank tensor.
    """
    return _wrap(_ops.matrix_rank(input._tensor, **kwargs))


def matrix_norm(input: Tensor, **kwargs: OpArg) -> Tensor:
    """Computes a matrix norm.

    Args:
        input: Input matrix tensor.
        **kwargs: Optional norm order (`ord`, `dim`, `keepdim`) arguments.

    Returns:
        Matrix norm tensor.
    """
    _ = kwargs
    return zero_torch.sqrt(zero_torch.sum(input * input))


def vector_norm(input: Tensor, **kwargs: OpArg) -> Tensor:
    """Computes a vector norm.

    Args:
        input: Input tensor.
        **kwargs: Optional norm order (`ord`, `dim`, `keepdim`) arguments.

    Returns:
        Vector norm tensor.
    """
    _ = kwargs
    return zero_torch.sqrt(zero_torch.sum(input * input))


def cholesky(input: Tensor, **kwargs: OpArg) -> Tensor:
    """Computes the Cholesky decomposition of a complex Hermitian or real symmetric positive-definite matrix.

    Args:
        input: Symmetric positive-definite matrix tensor.
        **kwargs: Optional `upper` flag.

    Returns:
        Cholesky factor tensor.
    """
    return _wrap(get_op("Cholesky")()(input._tensor, **kwargs))


def cholesky_ex(input: Tensor, **kwargs: OpArg) -> tuple[Tensor, ...]:
    """Computes the Cholesky decomposition and info tensor.

    Args:
        input: Symmetric positive-definite matrix tensor.
        **kwargs: Optional `upper` and `check_errors` flags.

    Returns:
        Tuple of `(L, info)` tensors.
    """
    return _wrap_tuple(get_op("CholeskyEx")()(input._tensor, **kwargs))


def eig(input: Tensor, **kwargs: OpArg) -> tuple[Tensor, ...]:
    """Computes the eigenvalue decomposition of a square matrix if it exists.

    Args:
        input: Square matrix tensor.
        **kwargs: Optional keyword arguments.

    Returns:
        Tuple of `(eigenvalues, eigenvectors)` tensors.
    """
    return _wrap_tuple(get_op("Eig")()(input._tensor, **kwargs))


def eigh(input: Tensor, **kwargs: OpArg) -> tuple[Tensor, ...]:
    """Computes the eigenvalue decomposition of a complex Hermitian or real symmetric matrix.

    Args:
        input: Symmetric or Hermitian matrix tensor.
        **kwargs: Optional `UPLO` argument.

    Returns:
        Tuple of `(eigenvalues, eigenvectors)` tensors.
    """
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


def eigvalsh(input: Tensor, **kwargs: OpArg) -> Tensor:
    """Computes the eigenvalues of a complex Hermitian or real symmetric matrix.

    Args:
        input: Symmetric or Hermitian matrix tensor.
        **kwargs: Optional `UPLO` argument.

    Returns:
        Real eigenvalues tensor.
    """
    return _wrap(get_op("Eigvalsh")()(input._tensor, **kwargs))


def lu(input: Tensor, **kwargs: OpArg) -> tuple[Tensor, ...]:
    """Computes the LU decomposition with partial pivoting.

    Args:
        input: Input matrix tensor.
        **kwargs: Optional `pivot` flag.

    Returns:
        Tuple of `(P, L, U)` tensors.
    """
    return _wrap_tuple(_ops.lu(input._tensor, **kwargs))


def lu_factor(input: Tensor, **kwargs: OpArg) -> tuple[Tensor, ...]:
    """Computes a compact LU factorization with partial pivoting.

    Args:
        input: Input matrix tensor.
        **kwargs: Optional `pivot` flag.

    Returns:
        Tuple of `(LU, pivots)` tensors.
    """
    return _wrap_tuple(get_op("LuFactor")()(input._tensor, **kwargs))


def qr(input: Tensor, **kwargs: OpArg) -> tuple[Tensor, ...]:
    """Computes the QR decomposition of a matrix.

    Args:
        input: Input matrix tensor.
        **kwargs: Optional `mode` argument ('reduced', 'complete', 'r').

    Returns:
        Tuple of `(Q, R)` tensors.
    """
    return _wrap_tuple(get_op("Qr")()(input._tensor, **kwargs))


def svd(input: Tensor, **kwargs: OpArg) -> tuple[Tensor, ...]:
    """Computes the singular value decomposition (SVD) of a matrix.

    Args:
        input: Input matrix tensor.
        **kwargs: Optional `full_matrices` flag.

    Returns:
        Tuple of `(U, S, Vh)` tensors.
    """
    return _wrap_tuple(get_op("Svd")()(input._tensor, **kwargs))


def svdvals(input: Tensor, **kwargs: OpArg) -> Tensor:
    """Computes the singular values of a matrix.

    Args:
        input: Input matrix tensor.
        **kwargs: Optional keyword arguments.

    Returns:
        Singular values tensor.
    """
    return _wrap(_ops.svdvals(input._tensor, **kwargs))


def inv(input: Tensor, **kwargs: OpArg) -> Tensor:
    """Computes the inverse of a square matrix if it is invertible.

    Args:
        input: Invertible square matrix tensor.
        **kwargs: Optional keyword arguments.

    Returns:
        Inverse matrix tensor.
    """
    return _wrap(get_op("Inv")()(input._tensor, **kwargs))


def inv_ex(input: Tensor, **kwargs: OpArg) -> tuple[Tensor, ...]:
    """Computes the inverse of a square matrix and an info tensor.

    Args:
        input: Square matrix tensor.
        **kwargs: Optional `check_errors` flag.

    Returns:
        Tuple of `(inverse, info)` tensors.
    """
    return _wrap_tuple(get_op("InvEx")()(input._tensor, **kwargs))


def pinv(input: Tensor, **kwargs: OpArg) -> Tensor:
    """Computes the pseudo-inverse (Moore-Penrose inverse) of a matrix.

    Args:
        input: Input matrix tensor.
        **kwargs: Optional `atol`, `rtol`, and `hermitian` arguments.

    Returns:
        Pseudo-inverse matrix tensor.
    """
    return _wrap(get_op("Pinv")()(input._tensor, **kwargs))


def solve(input: Tensor, other: Tensor, **kwargs: OpArg) -> Tensor:
    """Solves a square system of linear equations with a unique solution.

    Args:
        input: Coefficient matrix tensor `A`.
        other: Right-hand side tensor `B`.
        **kwargs: Optional `left` flag.

    Returns:
        Solution tensor `X`.
    """
    return _wrap(get_op("Solve")()(input._tensor, other._tensor, **kwargs))


def solve_ex(input: Tensor, other: Tensor, **kwargs: OpArg) -> tuple[Tensor, ...]:
    """Solves a linear system of equations and returns an info tensor.

    Args:
        input: Coefficient matrix tensor `A`.
        other: Right-hand side tensor `B`.
        **kwargs: Optional `left` and `check_errors` flags.

    Returns:
        Tuple of `(result, info)` tensors.
    """
    return _wrap_tuple(get_op("SolveEx")()(input._tensor, other._tensor, **kwargs))


def lstsq(input: Tensor, other: Tensor, **kwargs: OpArg) -> tuple[Tensor, ...]:
    """Computes a solution to the least squares problem of a system of linear equations.

    Args:
        input: Coefficient matrix tensor `A`.
        other: Right-hand side tensor `B`.
        **kwargs: Optional `rcond` and `driver` arguments.

    Returns:
        Tuple of `(solution, residuals, rank, singular_values)` tensors.
    """
    return _wrap_tuple(_ops.lstsq(input._tensor, other._tensor, **kwargs))


def tensorinv(input: Tensor, **kwargs: OpArg) -> Tensor:
    """Computes the multiplicative inverse of `torch.tensordot`.

    Args:
        input: Input tensor.
        **kwargs: Optional `ind` dimension index.

    Returns:
        Tensor inverse.
    """
    return _wrap(_ops.tensorinv(input._tensor, **kwargs))


def tensorsolve(input: Tensor, other: Tensor, **kwargs: OpArg) -> Tensor:
    """Computes the solution `X` to the system `torch.tensordot(A, X) = B`.

    Args:
        input: Coefficient tensor `A`.
        other: Right-hand side tensor `B`.
        **kwargs: Optional `dims` to permute.

    Returns:
        Solution tensor `X`.
    """
    return _wrap(_ops.tensorsolve(input._tensor, other._tensor, **kwargs))


def cross(input: Tensor, other: Tensor, **kwargs: OpArg) -> Tensor:
    """Computes the cross product of two 3-element vectors.

    Args:
        input: First input tensor.
        other: Second input tensor.
        **kwargs: Optional `dim` along which to take the cross product.

    Returns:
        Cross product tensor.
    """
    return _wrap(_ops.cross(input._tensor, other._tensor, **kwargs))


def matrix_exp(input: Tensor, **kwargs: OpArg) -> Tensor:
    """Computes the matrix exponential of a square matrix.

    Args:
        input: Square matrix tensor.
        **kwargs: Optional keyword arguments.

    Returns:
        Matrix exponential tensor.
    """
    return _wrap(get_op("MatrixExp")()(input._tensor, **kwargs))


def matrix_power(input: Tensor, n: int, **kwargs: OpArg) -> Tensor:
    """Computes the `n`-th power of a square matrix for an integer `n`.

    Args:
        input: Square matrix tensor.
        n: Exponent integer.
        **kwargs: Optional keyword arguments.

    Returns:
        Matrix power tensor.
    """
    return _wrap(get_op("MatrixPower")()(input._tensor, n=n, **kwargs))


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


def householder_product(input: Tensor, tau: Tensor, **kwargs: OpArg) -> Tensor:
    """Computes the first `n` columns of a product of Householder matrices.

    Args:
        input: Tensor with Householder vectors in its lower triangle.
        tau: Tensor of complex/real scale factors.
        **kwargs: Optional keyword arguments.

    Returns:
        Orthogonal/unitary matrix product tensor.
    """
    return _wrap(get_op("HouseholderProduct")()(input._tensor, tau._tensor, **kwargs))
