"""Linear Algebra frontend module."""

from zero_torch.tensor import Tensor, _wrap

try:
    import ml_switcheroo_compiler.ops.linalg as _ops
except ImportError:

    class _MockLinalg:
        pass

    _ops = _MockLinalg()


def _get_op(name):
    if not hasattr(_ops, name):
        raise NotImplementedError(f"Compiler backend missing linalg op: {name}")
    return getattr(_ops, name)


def cond(input: Tensor, **kwargs) -> Tensor:
    """Computes the condition number of a matrix with respect to a matrix norm."""
    return _wrap(_get_op("cond")(input._tensor, **kwargs))


def det(input: Tensor, **kwargs) -> Tensor:
    """Computes the determinant of a square matrix."""
    return _wrap(_get_op("det")(input._tensor, **kwargs))


def slogdet(input: Tensor, **kwargs) -> tuple:
    """Computes the sign and natural logarithm of the absolute value of the determinant of a square matrix."""
    return _wrap(_get_op("slogdet")(input._tensor, **kwargs))


def matrix_rank(input: Tensor, **kwargs) -> Tensor:
    """Computes the numerical rank of a matrix."""
    return _wrap(_get_op("matrix_rank")(input._tensor, **kwargs))


def matrix_norm(input: Tensor, **kwargs) -> Tensor:
    """Computes a matrix norm."""
    return _wrap(_get_op("matrix_norm")(input._tensor, **kwargs))


def vector_norm(input: Tensor, **kwargs) -> Tensor:
    """Computes a vector norm."""
    return _wrap(_get_op("vector_norm")(input._tensor, **kwargs))


def cholesky(input: Tensor, **kwargs) -> Tensor:
    """Computes the Cholesky decomposition of a complex Hermitian or real symmetric positive-definite matrix."""
    return _wrap(_get_op("cholesky")(input._tensor, **kwargs))


def cholesky_ex(input: Tensor, **kwargs) -> tuple:
    """Computes the Cholesky decomposition and info."""
    return _wrap(_get_op("cholesky_ex")(input._tensor, **kwargs))


def eig(input: Tensor, **kwargs) -> tuple:
    """Computes the eigenvalue decomposition of a square matrix if it exists."""
    return _wrap(_get_op("eig")(input._tensor, **kwargs))


def eigh(input: Tensor, **kwargs) -> tuple:
    """Computes the eigenvalue decomposition of a complex Hermitian or real symmetric matrix."""
    return _wrap(_get_op("eigh")(input._tensor, **kwargs))


def eigvals(input: Tensor, **kwargs) -> Tensor:
    """Computes the eigenvalues of a square matrix."""
    return _wrap(_get_op("eigvals")(input._tensor, **kwargs))


def eigvalsh(input: Tensor, **kwargs) -> Tensor:
    """Computes the eigenvalues of a complex Hermitian or real symmetric matrix."""
    return _wrap(_get_op("eigvalsh")(input._tensor, **kwargs))


def lu(input: Tensor, **kwargs) -> tuple:
    """Computes the LU decomposition with partial pivoting."""
    return _wrap(_get_op("lu")(input._tensor, **kwargs))


def lu_factor(input: Tensor, **kwargs) -> tuple:
    """Computes a compact LU factorization."""
    return _wrap(_get_op("lu_factor")(input._tensor, **kwargs))


def qr(input: Tensor, **kwargs) -> tuple:
    """Computes the QR decomposition of a matrix."""
    return _wrap(_get_op("qr")(input._tensor, **kwargs))


def svd(input: Tensor, **kwargs) -> tuple:
    """Computes the singular value decomposition (SVD) of a matrix."""
    return _wrap(_get_op("svd")(input._tensor, **kwargs))


def svdvals(input: Tensor, **kwargs) -> Tensor:
    """Computes the singular values of a matrix."""
    return _wrap(_get_op("svdvals")(input._tensor, **kwargs))


def inv(input: Tensor, **kwargs) -> Tensor:
    """Computes the inverse of a square matrix."""
    return _wrap(_get_op("inv")(input._tensor, **kwargs))


def inv_ex(input: Tensor, **kwargs) -> tuple:
    """Computes the inverse of a square matrix and info."""
    return _wrap(_get_op("inv_ex")(input._tensor, **kwargs))


def pinv(input: Tensor, **kwargs) -> Tensor:
    """Computes the pseudo-inverse (Moore-Penrose inverse) of a matrix."""
    return _wrap(_get_op("pinv")(input._tensor, **kwargs))


def solve(input: Tensor, other: Tensor, **kwargs) -> Tensor:
    """Solves a linear system of equations."""
    return _wrap(_get_op("solve")(input._tensor, other._tensor, **kwargs))


def solve_ex(input: Tensor, other: Tensor, **kwargs) -> tuple:
    """Solves a linear system of equations and info."""
    return _wrap(_get_op("solve_ex")(input._tensor, other._tensor, **kwargs))


def lstsq(input: Tensor, other: Tensor, **kwargs) -> tuple:
    """Computes a solution to the least squares problem."""
    return _wrap(_get_op("lstsq")(input._tensor, other._tensor, **kwargs))


def tensorinv(input: Tensor, **kwargs) -> Tensor:
    """Computes the tensor inverse."""
    return _wrap(_get_op("tensorinv")(input._tensor, **kwargs))


def tensorsolve(input: Tensor, other: Tensor, **kwargs) -> Tensor:
    """Solves a tensor system of equations."""
    return _wrap(_get_op("tensorsolve")(input._tensor, other._tensor, **kwargs))


def cross(input: Tensor, other: Tensor, **kwargs) -> Tensor:
    """Computes the cross product of two tensors."""
    return _wrap(_get_op("cross")(input._tensor, other._tensor, **kwargs))


def matrix_exp(input: Tensor, **kwargs) -> Tensor:
    """Computes the matrix exponential."""
    return _wrap(_get_op("matrix_exp")(input._tensor, **kwargs))


def matrix_power(input: Tensor, n: int, **kwargs) -> Tensor:
    """Computes the matrix power."""
    return _wrap(_get_op("matrix_power")(input._tensor, n=n, **kwargs))


def multi_dot(tensors, **kwargs) -> Tensor:
    """Computes the dot product of two or more matrices."""
    unwrapped = [t._tensor for t in tensors]
    return _wrap(_get_op("multi_dot")(unwrapped, **kwargs))


def householder_product(input: Tensor, tau: Tensor, **kwargs) -> Tensor:
    """Computes the first n columns of a product of Householder matrices."""
    return _wrap(_get_op("householder_product")(input._tensor, tau._tensor, **kwargs))
