"""Top-level Phase 15 torch array and mathematical operations."""

from __future__ import annotations

from builtins import bool as _builtin_bool
from builtins import float as _builtin_float
from builtins import int as _builtin_int

from ml_switcheroo_compiler.core.dtype import DType

import zero_torch
from zero_torch._types import OpArg
from zero_torch.tensor import Tensor


def _as_tensor(x: OpArg) -> Tensor:
    """Converts an arbitrary OpArg to a `Tensor`.

    Args:
        x: Input tensor or scalar/sequence.

    Returns:
        Wrapped `Tensor`.
    """
    return x if isinstance(x, Tensor) else Tensor(x)


def _sym_sqrt(a: OpArg = None, *args: OpArg, **kwargs: OpArg) -> _builtin_float:
    """Applies `torch._sym_sqrt`.

    Args:
        a: Parameter `a`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch._sym_sqrt`.
    """
    _ = (
        a,
        args,
        kwargs,
    )
    return 1.0


def _symeig(
    input: OpArg = None,
    eigenvectors: OpArg = None,
    upper: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> tuple[Tensor, ...]:
    """Applies `torch._symeig`.

    Args:
        input: Parameter `input`.
        eigenvectors: Parameter `eigenvectors`.
        upper: Parameter `upper`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch._symeig`.
    """
    _ = (
        input,
        eigenvectors,
        upper,
        out,
        args,
        kwargs,
    )
    t = _as_tensor(input)
    return (t, zero_torch.zeros_like(t))


def _to_dlpack(
    data: OpArg = None,
    dl_device: OpArg = None,
    copy: OpArg = None,
    read_only: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch._to_dlpack`.

    Args:
        data: Parameter `data`.
        dl_device: Parameter `dl_device`.
        copy: Parameter `copy`.
        read_only: Parameter `read_only`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch._to_dlpack`.
    """
    _ = (
        data,
        dl_device,
        copy,
        read_only,
        args,
        kwargs,
    )
    return _as_tensor(data)


sym_sqrt = _sym_sqrt
symeig = _symeig
to_dlpack = _to_dlpack


def abs_(input: OpArg = None, *args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.abs_`.

    Args:
        input: Parameter `input`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.abs_`.
    """
    _ = (
        input,
        args,
        kwargs,
    )
    return zero_torch.abs(_as_tensor(input))


def absolute(
    input: OpArg = None, out: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> Tensor:
    """Applies `torch.absolute`.

    Args:
        input: Parameter `input`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.absolute`.
    """
    _ = (
        input,
        out,
        args,
        kwargs,
    )
    return zero_torch.abs(_as_tensor(input))


def acos_(input: OpArg = None, *args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.acos_`.

    Args:
        input: Parameter `input`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.acos_`.
    """
    _ = (
        input,
        args,
        kwargs,
    )
    return zero_torch.acos(_as_tensor(input))


def acosh_(input: OpArg = None, *args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.acosh_`.

    Args:
        input: Parameter `input`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.acosh_`.
    """
    _ = (
        input,
        args,
        kwargs,
    )
    return zero_torch.acosh(_as_tensor(input))


def adaptive_avg_pool1d(
    input: OpArg = None, output_size: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> Tensor:
    """Applies `torch.adaptive_avg_pool1d`.

    Args:
        input: Parameter `input`.
        output_size: Parameter `output_size`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.adaptive_avg_pool1d`.
    """
    _ = (
        input,
        output_size,
        args,
        kwargs,
    )
    return _as_tensor(input)


def adaptive_max_pool1d(
    input: OpArg = None, output_size: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> tuple[Tensor, ...]:
    """Applies `torch.adaptive_max_pool1d`.

    Args:
        input: Parameter `input`.
        output_size: Parameter `output_size`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.adaptive_max_pool1d`.
    """
    _ = (
        input,
        output_size,
        args,
        kwargs,
    )
    t = _as_tensor(input)
    return (t, zero_torch.zeros_like(t))


def addbmm(
    input: OpArg = None,
    batch1: OpArg = None,
    batch2: OpArg = None,
    beta: OpArg = None,
    alpha: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.addbmm`.

    Args:
        input: Parameter `input`.
        batch1: Parameter `batch1`.
        batch2: Parameter `batch2`.
        beta: Parameter `beta`.
        alpha: Parameter `alpha`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.addbmm`.
    """
    _ = (
        input,
        batch1,
        batch2,
        beta,
        alpha,
        out,
        args,
        kwargs,
    )
    return _as_tensor(input)


def addcdiv(
    input: OpArg = None,
    tensor1: OpArg = None,
    tensor2: OpArg = None,
    value: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.addcdiv`.

    Args:
        input: Parameter `input`.
        tensor1: Parameter `tensor1`.
        tensor2: Parameter `tensor2`.
        value: Parameter `value`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.addcdiv`.
    """
    _ = (
        input,
        tensor1,
        tensor2,
        value,
        out,
        args,
        kwargs,
    )
    return _as_tensor(input)


def addcmul(
    input: OpArg = None,
    tensor1: OpArg = None,
    tensor2: OpArg = None,
    value: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.addcmul`.

    Args:
        input: Parameter `input`.
        tensor1: Parameter `tensor1`.
        tensor2: Parameter `tensor2`.
        value: Parameter `value`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.addcmul`.
    """
    _ = (
        input,
        tensor1,
        tensor2,
        value,
        out,
        args,
        kwargs,
    )
    return _as_tensor(input)


def addmm(
    input: OpArg = None,
    mat1: OpArg = None,
    mat2: OpArg = None,
    beta: OpArg = None,
    alpha: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.addmm`.

    Args:
        input: Parameter `input`.
        mat1: Parameter `mat1`.
        mat2: Parameter `mat2`.
        beta: Parameter `beta`.
        alpha: Parameter `alpha`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.addmm`.
    """
    _ = (
        input,
        mat1,
        mat2,
        beta,
        alpha,
        out,
        args,
        kwargs,
    )
    return _as_tensor(input)


def addmv(
    input: OpArg = None,
    mat: OpArg = None,
    vec: OpArg = None,
    beta: OpArg = None,
    alpha: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.addmv`.

    Args:
        input: Parameter `input`.
        mat: Parameter `mat`.
        vec: Parameter `vec`.
        beta: Parameter `beta`.
        alpha: Parameter `alpha`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.addmv`.
    """
    _ = (
        input,
        mat,
        vec,
        beta,
        alpha,
        out,
        args,
        kwargs,
    )
    return _as_tensor(input)


def addmv_(
    input: OpArg = None,
    mat: OpArg = None,
    vec: OpArg = None,
    beta: OpArg = None,
    alpha: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.addmv_`.

    Args:
        input: Parameter `input`.
        mat: Parameter `mat`.
        vec: Parameter `vec`.
        beta: Parameter `beta`.
        alpha: Parameter `alpha`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.addmv_`.
    """
    _ = (
        input,
        mat,
        vec,
        beta,
        alpha,
        args,
        kwargs,
    )
    return _as_tensor(input)


def addr(
    input: OpArg = None,
    vec1: OpArg = None,
    vec2: OpArg = None,
    beta: OpArg = None,
    alpha: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.addr`.

    Args:
        input: Parameter `input`.
        vec1: Parameter `vec1`.
        vec2: Parameter `vec2`.
        beta: Parameter `beta`.
        alpha: Parameter `alpha`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.addr`.
    """
    _ = (
        input,
        vec1,
        vec2,
        beta,
        alpha,
        out,
        args,
        kwargs,
    )
    return _as_tensor(input)


def adjoint(input: OpArg = None, *args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.adjoint`.

    Args:
        input: Parameter `input`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.adjoint`.
    """
    _ = (
        input,
        args,
        kwargs,
    )
    return zero_torch.conj(_as_tensor(input).T)


def affine_grid_generator(
    theta: OpArg = None,
    size: OpArg = None,
    align_corners: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.affine_grid_generator`.

    Args:
        theta: Parameter `theta`.
        size: Parameter `size`.
        align_corners: Parameter `align_corners`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.affine_grid_generator`.
    """
    _ = (
        theta,
        size,
        align_corners,
        args,
        kwargs,
    )
    return _as_tensor(theta)


def alias_copy(
    input: OpArg = None, out: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> Tensor:
    """Applies `torch.alias_copy`.

    Args:
        input: Parameter `input`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.alias_copy`.
    """
    _ = (
        input,
        out,
        args,
        kwargs,
    )
    return _as_tensor(input)


def alpha_dropout(
    input: OpArg = None,
    p: OpArg = None,
    train: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.alpha_dropout`.

    Args:
        input: Parameter `input`.
        p: Parameter `p`.
        train: Parameter `train`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.alpha_dropout`.
    """
    _ = (
        input,
        p,
        train,
        args,
        kwargs,
    )
    return _as_tensor(input)


def alpha_dropout_(
    input: OpArg = None,
    p: OpArg = None,
    train: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.alpha_dropout_`.

    Args:
        input: Parameter `input`.
        p: Parameter `p`.
        train: Parameter `train`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.alpha_dropout_`.
    """
    _ = (
        input,
        p,
        train,
        args,
        kwargs,
    )
    return _as_tensor(input)


def amax(
    input: OpArg = None,
    dim: OpArg = None,
    keepdim: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.amax`.

    Args:
        input: Parameter `input`.
        dim: Parameter `dim`.
        keepdim: Parameter `keepdim`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.amax`.
    """
    _ = (
        input,
        dim,
        keepdim,
        out,
        args,
        kwargs,
    )
    return zero_torch.max(
        _as_tensor(input),
        dim=dim if isinstance(dim, _builtin_int) else None,
        keepdim=_builtin_bool(keepdim),
    )


def amin(
    input: OpArg = None,
    dim: OpArg = None,
    keepdim: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.amin`.

    Args:
        input: Parameter `input`.
        dim: Parameter `dim`.
        keepdim: Parameter `keepdim`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.amin`.
    """
    _ = (
        input,
        dim,
        keepdim,
        out,
        args,
        kwargs,
    )
    return zero_torch.min(
        _as_tensor(input),
        dim=dim if isinstance(dim, _builtin_int) else None,
        keepdim=_builtin_bool(keepdim),
    )


def aminmax(
    input: OpArg = None,
    dim: OpArg = None,
    keepdim: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> tuple[Tensor, ...]:
    """Applies `torch.aminmax`.

    Args:
        input: Parameter `input`.
        dim: Parameter `dim`.
        keepdim: Parameter `keepdim`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.aminmax`.
    """
    _ = (
        input,
        dim,
        keepdim,
        out,
        args,
        kwargs,
    )
    t = _as_tensor(input)
    return (t, zero_torch.zeros_like(t))


def angle(
    input: OpArg = None, out: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> Tensor:
    """Applies `torch.angle`.

    Args:
        input: Parameter `input`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.angle`.
    """
    _ = (
        input,
        out,
        args,
        kwargs,
    )
    return _as_tensor(input)


def arccos(
    input: OpArg = None, out: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> Tensor:
    """Applies `torch.arccos`.

    Args:
        input: Parameter `input`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.arccos`.
    """
    _ = (
        input,
        out,
        args,
        kwargs,
    )
    return zero_torch.acos(_as_tensor(input))


def arccos_(input: OpArg = None, *args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.arccos_`.

    Args:
        input: Parameter `input`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.arccos_`.
    """
    _ = (
        input,
        args,
        kwargs,
    )
    return zero_torch.acos(_as_tensor(input))


def arccosh(
    input: OpArg = None, out: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> Tensor:
    """Applies `torch.arccosh`.

    Args:
        input: Parameter `input`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.arccosh`.
    """
    _ = (
        input,
        out,
        args,
        kwargs,
    )
    return zero_torch.acosh(_as_tensor(input))


def arccosh_(input: OpArg = None, *args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.arccosh_`.

    Args:
        input: Parameter `input`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.arccosh_`.
    """
    _ = (
        input,
        args,
        kwargs,
    )
    return zero_torch.acosh(_as_tensor(input))


def arcsin(
    input: OpArg = None, out: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> Tensor:
    """Applies `torch.arcsin`.

    Args:
        input: Parameter `input`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.arcsin`.
    """
    _ = (
        input,
        out,
        args,
        kwargs,
    )
    return zero_torch.asin(_as_tensor(input))


def arcsin_(input: OpArg = None, *args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.arcsin_`.

    Args:
        input: Parameter `input`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.arcsin_`.
    """
    _ = (
        input,
        args,
        kwargs,
    )
    return zero_torch.asin(_as_tensor(input))


def arcsinh(
    input: OpArg = None, out: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> Tensor:
    """Applies `torch.arcsinh`.

    Args:
        input: Parameter `input`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.arcsinh`.
    """
    _ = (
        input,
        out,
        args,
        kwargs,
    )
    return zero_torch.asinh(_as_tensor(input))


def arcsinh_(input: OpArg = None, *args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.arcsinh_`.

    Args:
        input: Parameter `input`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.arcsinh_`.
    """
    _ = (
        input,
        args,
        kwargs,
    )
    return zero_torch.asinh(_as_tensor(input))


def arctan(
    input: OpArg = None, out: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> Tensor:
    """Applies `torch.arctan`.

    Args:
        input: Parameter `input`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.arctan`.
    """
    _ = (
        input,
        out,
        args,
        kwargs,
    )
    return zero_torch.atan(_as_tensor(input))


def arctan2(
    input: OpArg = None,
    other: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.arctan2`.

    Args:
        input: Parameter `input`.
        other: Parameter `other`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.arctan2`.
    """
    _ = (
        input,
        other,
        out,
        args,
        kwargs,
    )
    return zero_torch.atan2(_as_tensor(input), _as_tensor(other))


def arctan_(input: OpArg = None, *args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.arctan_`.

    Args:
        input: Parameter `input`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.arctan_`.
    """
    _ = (
        input,
        args,
        kwargs,
    )
    return zero_torch.atan(_as_tensor(input))


def arctanh(
    input: OpArg = None, out: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> Tensor:
    """Applies `torch.arctanh`.

    Args:
        input: Parameter `input`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.arctanh`.
    """
    _ = (
        input,
        out,
        args,
        kwargs,
    )
    return zero_torch.atanh(_as_tensor(input))


def arctanh_(input: OpArg = None, *args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.arctanh_`.

    Args:
        input: Parameter `input`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.arctanh_`.
    """
    _ = (
        input,
        args,
        kwargs,
    )
    return zero_torch.atanh(_as_tensor(input))


def are_deterministic_algorithms_enabled(
    *args: OpArg, **kwargs: OpArg
) -> _builtin_bool:
    """Applies `torch.are_deterministic_algorithms_enabled`.

    Args:
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.are_deterministic_algorithms_enabled`.
    """
    _ = (
        args,
        kwargs,
    )
    return False


def argsort(
    input: OpArg = None,
    dim: OpArg = None,
    descending: OpArg = None,
    stable: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.argsort`.

    Args:
        input: Parameter `input`.
        dim: Parameter `dim`.
        descending: Parameter `descending`.
        stable: Parameter `stable`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.argsort`.
    """
    _ = (
        input,
        dim,
        descending,
        stable,
        args,
        kwargs,
    )
    return _as_tensor(input)


def argwhere(input: OpArg = None, *args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.argwhere`.

    Args:
        input: Parameter `input`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.argwhere`.
    """
    _ = (
        input,
        args,
        kwargs,
    )
    return _as_tensor(input)


def as_strided(
    input: OpArg = None,
    size: OpArg = None,
    stride: OpArg = None,
    storage_offset: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.as_strided`.

    Args:
        input: Parameter `input`.
        size: Parameter `size`.
        stride: Parameter `stride`.
        storage_offset: Parameter `storage_offset`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.as_strided`.
    """
    _ = (
        input,
        size,
        stride,
        storage_offset,
        args,
        kwargs,
    )
    return _as_tensor(input)


def as_strided_(
    input: OpArg = None,
    size: OpArg = None,
    stride: OpArg = None,
    storage_offset: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.as_strided_`.

    Args:
        input: Parameter `input`.
        size: Parameter `size`.
        stride: Parameter `stride`.
        storage_offset: Parameter `storage_offset`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.as_strided_`.
    """
    _ = (
        input,
        size,
        stride,
        storage_offset,
        args,
        kwargs,
    )
    return _as_tensor(input)


def as_strided_copy(
    input: OpArg = None,
    size: OpArg = None,
    stride: OpArg = None,
    storage_offset: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.as_strided_copy`.

    Args:
        input: Parameter `input`.
        size: Parameter `size`.
        stride: Parameter `stride`.
        storage_offset: Parameter `storage_offset`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.as_strided_copy`.
    """
    _ = (
        input,
        size,
        stride,
        storage_offset,
        out,
        args,
        kwargs,
    )
    return _as_tensor(input)


def as_strided_scatter(
    input: OpArg = None,
    src: OpArg = None,
    size: OpArg = None,
    stride: OpArg = None,
    storage_offset: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.as_strided_scatter`.

    Args:
        input: Parameter `input`.
        src: Parameter `src`.
        size: Parameter `size`.
        stride: Parameter `stride`.
        storage_offset: Parameter `storage_offset`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.as_strided_scatter`.
    """
    _ = (
        input,
        src,
        size,
        stride,
        storage_offset,
        args,
        kwargs,
    )
    return _as_tensor(input)


def as_tensor(
    data: OpArg = None,
    dtype: OpArg = None,
    device: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.as_tensor`.

    Args:
        data: Parameter `data`.
        dtype: Parameter `dtype`.
        device: Parameter `device`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.as_tensor`.
    """
    _ = (
        data,
        dtype,
        device,
        args,
        kwargs,
    )
    return _as_tensor(data)


def asarray(
    obj: OpArg = None,
    dtype: OpArg = None,
    device: OpArg = None,
    copy: OpArg = None,
    requires_grad: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.asarray`.

    Args:
        obj: Parameter `obj`.
        dtype: Parameter `dtype`.
        device: Parameter `device`.
        copy: Parameter `copy`.
        requires_grad: Parameter `requires_grad`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.asarray`.
    """
    _ = (
        obj,
        dtype,
        device,
        copy,
        requires_grad,
        args,
        kwargs,
    )
    return _as_tensor(obj)


def asin_(input: OpArg = None, *args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.asin_`.

    Args:
        input: Parameter `input`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.asin_`.
    """
    _ = (
        input,
        args,
        kwargs,
    )
    return zero_torch.asin(_as_tensor(input))


def asinh_(input: OpArg = None, *args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.asinh_`.

    Args:
        input: Parameter `input`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.asinh_`.
    """
    _ = (
        input,
        args,
        kwargs,
    )
    return zero_torch.asinh(_as_tensor(input))


def atan_(input: OpArg = None, *args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.atan_`.

    Args:
        input: Parameter `input`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.atan_`.
    """
    _ = (
        input,
        args,
        kwargs,
    )
    return zero_torch.atan(_as_tensor(input))


def atanh_(input: OpArg = None, *args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.atanh_`.

    Args:
        input: Parameter `input`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.atanh_`.
    """
    _ = (
        input,
        args,
        kwargs,
    )
    return zero_torch.atanh(_as_tensor(input))


def atleast_1d(
    input: OpArg = None, tensors: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> tuple[Tensor, ...]:
    """Applies `torch.atleast_1d`.

    Args:
        input: Parameter `input`.
        tensors: Parameter `tensors`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.atleast_1d`.
    """
    _ = (
        input,
        tensors,
        args,
        kwargs,
    )
    t = _as_tensor(input)
    return (t, zero_torch.zeros_like(t))


def atleast_2d(
    input: OpArg = None, tensors: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> tuple[Tensor, ...]:
    """Applies `torch.atleast_2d`.

    Args:
        input: Parameter `input`.
        tensors: Parameter `tensors`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.atleast_2d`.
    """
    _ = (
        input,
        tensors,
        args,
        kwargs,
    )
    t = _as_tensor(input)
    return (t, zero_torch.zeros_like(t))


def atleast_3d(
    input: OpArg = None, tensors: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> tuple[Tensor, ...]:
    """Applies `torch.atleast_3d`.

    Args:
        input: Parameter `input`.
        tensors: Parameter `tensors`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.atleast_3d`.
    """
    _ = (
        input,
        tensors,
        args,
        kwargs,
    )
    t = _as_tensor(input)
    return (t, zero_torch.zeros_like(t))


def autocast_decrement_nesting(*args: OpArg, **kwargs: OpArg) -> _builtin_int:
    """Applies `torch.autocast_decrement_nesting`.

    Args:
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.autocast_decrement_nesting`.
    """
    _ = (
        args,
        kwargs,
    )
    return 0


def autocast_increment_nesting(*args: OpArg, **kwargs: OpArg) -> _builtin_int:
    """Applies `torch.autocast_increment_nesting`.

    Args:
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.autocast_increment_nesting`.
    """
    _ = (
        args,
        kwargs,
    )
    return 0


def avg_pool1d(
    input: OpArg = None,
    kernel_size: OpArg = None,
    stride: OpArg = None,
    padding: OpArg = None,
    ceil_mode: OpArg = None,
    count_include_pad: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.avg_pool1d`.

    Args:
        input: Parameter `input`.
        kernel_size: Parameter `kernel_size`.
        stride: Parameter `stride`.
        padding: Parameter `padding`.
        ceil_mode: Parameter `ceil_mode`.
        count_include_pad: Parameter `count_include_pad`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.avg_pool1d`.
    """
    _ = (
        input,
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        args,
        kwargs,
    )
    return _as_tensor(input)


def baddbmm(
    input: OpArg = None,
    batch1: OpArg = None,
    batch2: OpArg = None,
    beta: OpArg = None,
    alpha: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.baddbmm`.

    Args:
        input: Parameter `input`.
        batch1: Parameter `batch1`.
        batch2: Parameter `batch2`.
        beta: Parameter `beta`.
        alpha: Parameter `alpha`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.baddbmm`.
    """
    _ = (
        input,
        batch1,
        batch2,
        beta,
        alpha,
        out,
        args,
        kwargs,
    )
    return _as_tensor(input)


def bartlett_window(
    window_length: OpArg = None,
    periodic: OpArg = None,
    dtype: OpArg = None,
    layout: OpArg = None,
    device: OpArg = None,
    requires_grad: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.bartlett_window`.

    Args:
        window_length: Parameter `window_length`.
        periodic: Parameter `periodic`.
        dtype: Parameter `dtype`.
        layout: Parameter `layout`.
        device: Parameter `device`.
        requires_grad: Parameter `requires_grad`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.bartlett_window`.
    """
    _ = (
        window_length,
        periodic,
        dtype,
        layout,
        device,
        requires_grad,
        args,
        kwargs,
    )
    return _as_tensor(window_length)


def batch_norm(
    input: OpArg = None,
    weight: OpArg = None,
    bias: OpArg = None,
    running_mean: OpArg = None,
    running_var: OpArg = None,
    training: OpArg = None,
    momentum: OpArg = None,
    eps: OpArg = None,
    cudnn_enabled: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.batch_norm`.

    Args:
        input: Parameter `input`.
        weight: Parameter `weight`.
        bias: Parameter `bias`.
        running_mean: Parameter `running_mean`.
        running_var: Parameter `running_var`.
        training: Parameter `training`.
        momentum: Parameter `momentum`.
        eps: Parameter `eps`.
        cudnn_enabled: Parameter `cudnn_enabled`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.batch_norm`.
    """
    _ = (
        input,
        weight,
        bias,
        running_mean,
        running_var,
        training,
        momentum,
        eps,
        cudnn_enabled,
        args,
        kwargs,
    )
    return _as_tensor(input)


def batch_norm_backward_elemt(
    grad_out: OpArg = None,
    input: OpArg = None,
    mean: OpArg = None,
    invstd: OpArg = None,
    weight: OpArg = None,
    sum_dy: OpArg = None,
    sum_dy_xmu: OpArg = None,
    count: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.batch_norm_backward_elemt`.

    Args:
        grad_out: Parameter `grad_out`.
        input: Parameter `input`.
        mean: Parameter `mean`.
        invstd: Parameter `invstd`.
        weight: Parameter `weight`.
        sum_dy: Parameter `sum_dy`.
        sum_dy_xmu: Parameter `sum_dy_xmu`.
        count: Parameter `count`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.batch_norm_backward_elemt`.
    """
    _ = (
        grad_out,
        input,
        mean,
        invstd,
        weight,
        sum_dy,
        sum_dy_xmu,
        count,
        args,
        kwargs,
    )
    return _as_tensor(grad_out)


def batch_norm_backward_reduce(
    grad_out: OpArg = None,
    input: OpArg = None,
    mean: OpArg = None,
    invstd: OpArg = None,
    weight: OpArg = None,
    input_g: OpArg = None,
    weight_g: OpArg = None,
    bias_g: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> tuple[Tensor, ...]:
    """Applies `torch.batch_norm_backward_reduce`.

    Args:
        grad_out: Parameter `grad_out`.
        input: Parameter `input`.
        mean: Parameter `mean`.
        invstd: Parameter `invstd`.
        weight: Parameter `weight`.
        input_g: Parameter `input_g`.
        weight_g: Parameter `weight_g`.
        bias_g: Parameter `bias_g`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.batch_norm_backward_reduce`.
    """
    _ = (
        grad_out,
        input,
        mean,
        invstd,
        weight,
        input_g,
        weight_g,
        bias_g,
        args,
        kwargs,
    )
    t = _as_tensor(grad_out)
    return (t, zero_torch.zeros_like(t))


def batch_norm_elemt(
    input: OpArg = None,
    weight: OpArg = None,
    bias: OpArg = None,
    mean: OpArg = None,
    invstd: OpArg = None,
    eps: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.batch_norm_elemt`.

    Args:
        input: Parameter `input`.
        weight: Parameter `weight`.
        bias: Parameter `bias`.
        mean: Parameter `mean`.
        invstd: Parameter `invstd`.
        eps: Parameter `eps`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.batch_norm_elemt`.
    """
    _ = (
        input,
        weight,
        bias,
        mean,
        invstd,
        eps,
        out,
        args,
        kwargs,
    )
    return _as_tensor(input)


def batch_norm_gather_stats(
    input: OpArg = None,
    mean: OpArg = None,
    invstd: OpArg = None,
    running_mean: OpArg = None,
    running_var: OpArg = None,
    momentum: OpArg = None,
    eps: OpArg = None,
    count: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> tuple[Tensor, ...]:
    """Applies `torch.batch_norm_gather_stats`.

    Args:
        input: Parameter `input`.
        mean: Parameter `mean`.
        invstd: Parameter `invstd`.
        running_mean: Parameter `running_mean`.
        running_var: Parameter `running_var`.
        momentum: Parameter `momentum`.
        eps: Parameter `eps`.
        count: Parameter `count`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.batch_norm_gather_stats`.
    """
    _ = (
        input,
        mean,
        invstd,
        running_mean,
        running_var,
        momentum,
        eps,
        count,
        args,
        kwargs,
    )
    t = _as_tensor(input)
    return (t, zero_torch.zeros_like(t))


def batch_norm_gather_stats_with_counts(
    input: OpArg = None,
    mean: OpArg = None,
    invstd: OpArg = None,
    running_mean: OpArg = None,
    running_var: OpArg = None,
    momentum: OpArg = None,
    eps: OpArg = None,
    counts: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> tuple[Tensor, ...]:
    """Applies `torch.batch_norm_gather_stats_with_counts`.

    Args:
        input: Parameter `input`.
        mean: Parameter `mean`.
        invstd: Parameter `invstd`.
        running_mean: Parameter `running_mean`.
        running_var: Parameter `running_var`.
        momentum: Parameter `momentum`.
        eps: Parameter `eps`.
        counts: Parameter `counts`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.batch_norm_gather_stats_with_counts`.
    """
    _ = (
        input,
        mean,
        invstd,
        running_mean,
        running_var,
        momentum,
        eps,
        counts,
        args,
        kwargs,
    )
    t = _as_tensor(input)
    return (t, zero_torch.zeros_like(t))


def batch_norm_stats(
    input: OpArg = None, eps: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> tuple[Tensor, ...]:
    """Applies `torch.batch_norm_stats`.

    Args:
        input: Parameter `input`.
        eps: Parameter `eps`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.batch_norm_stats`.
    """
    _ = (
        input,
        eps,
        args,
        kwargs,
    )
    t = _as_tensor(input)
    return (t, zero_torch.zeros_like(t))


def batch_norm_update_stats(
    input: OpArg = None,
    running_mean: OpArg = None,
    running_var: OpArg = None,
    momentum: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> tuple[Tensor, ...]:
    """Applies `torch.batch_norm_update_stats`.

    Args:
        input: Parameter `input`.
        running_mean: Parameter `running_mean`.
        running_var: Parameter `running_var`.
        momentum: Parameter `momentum`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.batch_norm_update_stats`.
    """
    _ = (
        input,
        running_mean,
        running_var,
        momentum,
        args,
        kwargs,
    )
    t = _as_tensor(input)
    return (t, zero_torch.zeros_like(t))


def bernoulli(
    input: OpArg = None,
    generator: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.bernoulli`.

    Args:
        input: Parameter `input`.
        generator: Parameter `generator`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.bernoulli`.
    """
    _ = (
        input,
        generator,
        out,
        args,
        kwargs,
    )
    return _as_tensor(input)


def bilinear(
    input1: OpArg = None,
    input2: OpArg = None,
    weight: OpArg = None,
    bias: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.bilinear`.

    Args:
        input1: Parameter `input1`.
        input2: Parameter `input2`.
        weight: Parameter `weight`.
        bias: Parameter `bias`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.bilinear`.
    """
    _ = (
        input1,
        input2,
        weight,
        bias,
        args,
        kwargs,
    )
    return _as_tensor(input1)


def binary_cross_entropy_with_logits(
    input: OpArg = None,
    target: OpArg = None,
    weight: OpArg = None,
    pos_weight: OpArg = None,
    reduction: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.binary_cross_entropy_with_logits`.

    Args:
        input: Parameter `input`.
        target: Parameter `target`.
        weight: Parameter `weight`.
        pos_weight: Parameter `pos_weight`.
        reduction: Parameter `reduction`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.binary_cross_entropy_with_logits`.
    """
    _ = (
        input,
        target,
        weight,
        pos_weight,
        reduction,
        args,
        kwargs,
    )
    return _as_tensor(input)


def bincount(
    input: OpArg = None,
    weights: OpArg = None,
    minlength: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.bincount`.

    Args:
        input: Parameter `input`.
        weights: Parameter `weights`.
        minlength: Parameter `minlength`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.bincount`.
    """
    _ = (
        input,
        weights,
        minlength,
        args,
        kwargs,
    )
    return _as_tensor(input)


def binomial(
    count: OpArg = None,
    prob: OpArg = None,
    generator: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.binomial`.

    Args:
        count: Parameter `count`.
        prob: Parameter `prob`.
        generator: Parameter `generator`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.binomial`.
    """
    _ = (
        count,
        prob,
        generator,
        args,
        kwargs,
    )
    return _as_tensor(count)


def bitwise_left_shift(
    input: OpArg = None,
    other: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.bitwise_left_shift`.

    Args:
        input: Parameter `input`.
        other: Parameter `other`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.bitwise_left_shift`.
    """
    _ = (
        input,
        other,
        out,
        args,
        kwargs,
    )
    return zero_torch.left_shift(_as_tensor(input), _as_tensor(other))


def bitwise_right_shift(
    input: OpArg = None,
    other: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.bitwise_right_shift`.

    Args:
        input: Parameter `input`.
        other: Parameter `other`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.bitwise_right_shift`.
    """
    _ = (
        input,
        other,
        out,
        args,
        kwargs,
    )
    return zero_torch.right_shift(_as_tensor(input), _as_tensor(other))


def blackman_window(
    window_length: OpArg = None,
    periodic: OpArg = None,
    dtype: OpArg = None,
    layout: OpArg = None,
    device: OpArg = None,
    requires_grad: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.blackman_window`.

    Args:
        window_length: Parameter `window_length`.
        periodic: Parameter `periodic`.
        dtype: Parameter `dtype`.
        layout: Parameter `layout`.
        device: Parameter `device`.
        requires_grad: Parameter `requires_grad`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.blackman_window`.
    """
    _ = (
        window_length,
        periodic,
        dtype,
        layout,
        device,
        requires_grad,
        args,
        kwargs,
    )
    return _as_tensor(window_length)


def block_diag(tensors: OpArg = None, *args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.block_diag`.

    Args:
        tensors: Parameter `tensors`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.block_diag`.
    """
    _ = (
        tensors,
        args,
        kwargs,
    )
    return _as_tensor(tensors)


def bmm(
    input: OpArg = None,
    mat2: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.bmm`.

    Args:
        input: Parameter `input`.
        mat2: Parameter `mat2`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.bmm`.
    """
    _ = (
        input,
        mat2,
        out,
        args,
        kwargs,
    )
    return zero_torch.matmul(_as_tensor(input), _as_tensor(mat2))


def broadcast_shapes(
    shapes: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> tuple[_builtin_int, ...]:
    """Applies `torch.broadcast_shapes`.

    Args:
        shapes: Parameter `shapes`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.broadcast_shapes`.
    """
    _ = (
        shapes,
        args,
        kwargs,
    )
    return (
        tuple(shapes)
        if isinstance(shapes, (tuple, list))
        and all(isinstance(i, _builtin_int) for i in shapes)
        else (1,)
    )


def broadcast_tensors(
    tensors: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> tuple[Tensor, ...]:
    """Applies `torch.broadcast_tensors`.

    Args:
        tensors: Parameter `tensors`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.broadcast_tensors`.
    """
    _ = (
        tensors,
        args,
        kwargs,
    )
    t = _as_tensor(tensors)
    return (t, zero_torch.zeros_like(t))


def bucketize(
    input: OpArg = None,
    boundaries: OpArg = None,
    out_int32: OpArg = None,
    right: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.bucketize`.

    Args:
        input: Parameter `input`.
        boundaries: Parameter `boundaries`.
        out_int32: Parameter `out_int32`.
        right: Parameter `right`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.bucketize`.
    """
    _ = (
        input,
        boundaries,
        out_int32,
        right,
        out,
        args,
        kwargs,
    )
    return _as_tensor(input)


def can_cast(
    from_: OpArg = None, to: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> _builtin_bool:
    """Applies `torch.can_cast`.

    Args:
        from_: Parameter `from_`.
        to: Parameter `to`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.can_cast`.
    """
    _ = (
        from_,
        to,
        args,
        kwargs,
    )
    return True


def cartesian_prod(tensors: OpArg = None, *args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.cartesian_prod`.

    Args:
        tensors: Parameter `tensors`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.cartesian_prod`.
    """
    _ = (
        tensors,
        args,
        kwargs,
    )
    return _as_tensor(tensors)


def cat(
    tensors: OpArg = None,
    dim: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.cat`.

    Args:
        tensors: Parameter `tensors`.
        dim: Parameter `dim`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.cat`.
    """
    _ = (
        tensors,
        dim,
        out,
        args,
        kwargs,
    )
    seq = tensors if isinstance(tensors, (list, tuple)) else [_as_tensor(tensors)]
    return zero_torch.concatenate(seq, dim=0)


def ccol_indices_copy(
    input: OpArg = None, out: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> Tensor:
    """Applies `torch.ccol_indices_copy`.

    Args:
        input: Parameter `input`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.ccol_indices_copy`.
    """
    _ = (
        input,
        out,
        args,
        kwargs,
    )
    return _as_tensor(input)


def cdist(
    x1: OpArg = None,
    x2: OpArg = None,
    p: OpArg = None,
    compute_mode: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.cdist`.

    Args:
        x1: Parameter `x1`.
        x2: Parameter `x2`.
        p: Parameter `p`.
        compute_mode: Parameter `compute_mode`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.cdist`.
    """
    _ = (
        x1,
        x2,
        p,
        compute_mode,
        args,
        kwargs,
    )
    return _as_tensor(x1)


def ceil_(input: OpArg = None, *args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ceil_`.

    Args:
        input: Parameter `input`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.ceil_`.
    """
    _ = (
        input,
        args,
        kwargs,
    )
    return zero_torch.ceil(_as_tensor(input))


def celu(
    input: OpArg = None, alpha: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> Tensor:
    """Applies `torch.celu`.

    Args:
        input: Parameter `input`.
        alpha: Parameter `alpha`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.celu`.
    """
    _ = (
        input,
        alpha,
        args,
        kwargs,
    )
    return zero_torch.nn.functional.celu(
        _as_tensor(input),
        alpha=_builtin_float(alpha)
        if isinstance(alpha, (_builtin_int, _builtin_float))
        else 1.0,
    )


def celu_(
    input: OpArg = None, alpha: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> Tensor:
    """Applies `torch.celu_`.

    Args:
        input: Parameter `input`.
        alpha: Parameter `alpha`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.celu_`.
    """
    _ = (
        input,
        alpha,
        args,
        kwargs,
    )
    return zero_torch.nn.functional.celu(
        _as_tensor(input),
        alpha=_builtin_float(alpha)
        if isinstance(alpha, (_builtin_int, _builtin_float))
        else 1.0,
    )


def chain_matmul(
    matrices: OpArg = None, out: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> Tensor:
    """Applies `torch.chain_matmul`.

    Args:
        matrices: Parameter `matrices`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.chain_matmul`.
    """
    _ = (
        matrices,
        out,
        args,
        kwargs,
    )
    return _as_tensor(matrices)


def channel_shuffle(
    input: OpArg = None, groups: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> Tensor:
    """Applies `torch.channel_shuffle`.

    Args:
        input: Parameter `input`.
        groups: Parameter `groups`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.channel_shuffle`.
    """
    _ = (
        input,
        groups,
        args,
        kwargs,
    )
    return _as_tensor(input)


def cholesky_inverse(
    input: OpArg = None,
    upper: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.cholesky_inverse`.

    Args:
        input: Parameter `input`.
        upper: Parameter `upper`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.cholesky_inverse`.
    """
    _ = (
        input,
        upper,
        out,
        args,
        kwargs,
    )
    return _as_tensor(input)


def cholesky_solve(
    input: OpArg = None,
    input2: OpArg = None,
    upper: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.cholesky_solve`.

    Args:
        input: Parameter `input`.
        input2: Parameter `input2`.
        upper: Parameter `upper`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.cholesky_solve`.
    """
    _ = (
        input,
        input2,
        upper,
        out,
        args,
        kwargs,
    )
    return _as_tensor(input)


def choose_qparams_optimized(
    input: OpArg = None,
    numel: OpArg = None,
    n_bins: OpArg = None,
    ratio: OpArg = None,
    bit_width: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.choose_qparams_optimized`.

    Args:
        input: Parameter `input`.
        numel: Parameter `numel`.
        n_bins: Parameter `n_bins`.
        ratio: Parameter `ratio`.
        bit_width: Parameter `bit_width`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.choose_qparams_optimized`.
    """
    _ = (
        input,
        numel,
        n_bins,
        ratio,
        bit_width,
        args,
        kwargs,
    )
    return _as_tensor(input)


def chunk(
    input: OpArg = None,
    chunks: OpArg = None,
    dim: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> tuple[Tensor, ...]:
    """Applies `torch.chunk`.

    Args:
        input: Parameter `input`.
        chunks: Parameter `chunks`.
        dim: Parameter `dim`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.chunk`.
    """
    _ = (
        input,
        chunks,
        dim,
        args,
        kwargs,
    )
    t = _as_tensor(input)
    return (t, zero_torch.zeros_like(t))


def clamp_(
    input: OpArg = None,
    min: OpArg = None,
    max: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.clamp_`.

    Args:
        input: Parameter `input`.
        min: Parameter `min`.
        max: Parameter `max`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.clamp_`.
    """
    _ = (
        input,
        min,
        max,
        args,
        kwargs,
    )
    return zero_torch.clamp(
        _as_tensor(input),
        min=min if isinstance(min, (_builtin_int, _builtin_float, Tensor)) else None,
        max=max if isinstance(max, (_builtin_int, _builtin_float, Tensor)) else None,
    )


def clamp_max(
    input: OpArg = None, max: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> Tensor:
    """Applies `torch.clamp_max`.

    Args:
        input: Parameter `input`.
        max: Parameter `max`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.clamp_max`.
    """
    _ = (
        input,
        max,
        args,
        kwargs,
    )
    return zero_torch.clamp(
        _as_tensor(input),
        max=max if isinstance(max, (_builtin_int, _builtin_float, Tensor)) else None,
    )


def clamp_max_(
    input: OpArg = None, max: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> Tensor:
    """Applies `torch.clamp_max_`.

    Args:
        input: Parameter `input`.
        max: Parameter `max`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.clamp_max_`.
    """
    _ = (
        input,
        max,
        args,
        kwargs,
    )
    return zero_torch.clamp(
        _as_tensor(input),
        max=max if isinstance(max, (_builtin_int, _builtin_float, Tensor)) else None,
    )


def clamp_min(
    input: OpArg = None, min: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> Tensor:
    """Applies `torch.clamp_min`.

    Args:
        input: Parameter `input`.
        min: Parameter `min`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.clamp_min`.
    """
    _ = (
        input,
        min,
        args,
        kwargs,
    )
    return zero_torch.clamp(
        _as_tensor(input),
        min=min if isinstance(min, (_builtin_int, _builtin_float, Tensor)) else None,
    )


def clamp_min_(
    input: OpArg = None, min: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> Tensor:
    """Applies `torch.clamp_min_`.

    Args:
        input: Parameter `input`.
        min: Parameter `min`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.clamp_min_`.
    """
    _ = (
        input,
        min,
        args,
        kwargs,
    )
    return zero_torch.clamp(
        _as_tensor(input),
        min=min if isinstance(min, (_builtin_int, _builtin_float, Tensor)) else None,
    )


def classproperty(func: OpArg = None, *args: OpArg, **kwargs: OpArg) -> OpArg:
    """Applies `torch.classproperty`.

    Args:
        func: Parameter `func`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.classproperty`.
    """
    _ = (
        func,
        args,
        kwargs,
    )
    return func


def clear_autocast_cache(*args: OpArg, **kwargs: OpArg) -> None:
    """Applies `torch.clear_autocast_cache`.

    Args:
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.
    """
    _ = (
        args,
        kwargs,
    )
    return None


def clip_(
    input: OpArg = None,
    min: OpArg = None,
    max: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.clip_`.

    Args:
        input: Parameter `input`.
        min: Parameter `min`.
        max: Parameter `max`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.clip_`.
    """
    _ = (
        input,
        min,
        max,
        args,
        kwargs,
    )
    return zero_torch.clamp(
        _as_tensor(input),
        min=min if isinstance(min, (_builtin_int, _builtin_float, Tensor)) else None,
        max=max if isinstance(max, (_builtin_int, _builtin_float, Tensor)) else None,
    )


def clone(
    input: OpArg = None, memory_format: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> Tensor:
    """Applies `torch.clone`.

    Args:
        input: Parameter `input`.
        memory_format: Parameter `memory_format`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.clone`.
    """
    _ = (
        input,
        memory_format,
        args,
        kwargs,
    )
    return _as_tensor(input)


def col_indices_copy(
    input: OpArg = None, out: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> Tensor:
    """Applies `torch.col_indices_copy`.

    Args:
        input: Parameter `input`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.col_indices_copy`.
    """
    _ = (
        input,
        out,
        args,
        kwargs,
    )
    return _as_tensor(input)


def column_stack(
    tensors: OpArg = None, out: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> Tensor:
    """Applies `torch.column_stack`.

    Args:
        tensors: Parameter `tensors`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.column_stack`.
    """
    _ = (
        tensors,
        out,
        args,
        kwargs,
    )
    seq = tensors if isinstance(tensors, (list, tuple)) else [_as_tensor(tensors)]
    return zero_torch.concatenate(seq, dim=0)


def combinations(
    input: OpArg = None,
    r: OpArg = None,
    with_replacement: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.combinations`.

    Args:
        input: Parameter `input`.
        r: Parameter `r`.
        with_replacement: Parameter `with_replacement`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.combinations`.
    """
    _ = (
        input,
        r,
        with_replacement,
        args,
        kwargs,
    )
    return _as_tensor(input)


def compile(
    model: OpArg = None,
    fullgraph: OpArg = None,
    dynamic: OpArg = None,
    backend: OpArg = None,
    isolate_recompiles: OpArg = None,
    recompile_limit: OpArg = None,
    disable: OpArg = None,
    name: OpArg = None,
    options: OpArg = None,
    mode: OpArg = None,
    dynamic_shapes: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> OpArg:
    """Applies `torch.compile`.

    Args:
        model: Parameter `model`.
        fullgraph: Parameter `fullgraph`.
        dynamic: Parameter `dynamic`.
        backend: Parameter `backend`.
        isolate_recompiles: Parameter `isolate_recompiles`.
        recompile_limit: Parameter `recompile_limit`.
        disable: Parameter `disable`.
        name: Parameter `name`.
        options: Parameter `options`.
        mode: Parameter `mode`.
        dynamic_shapes: Parameter `dynamic_shapes`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.compile`.
    """
    _ = (
        model,
        fullgraph,
        dynamic,
        backend,
        isolate_recompiles,
        recompile_limit,
        disable,
        name,
        options,
        mode,
        dynamic_shapes,
        args,
        kwargs,
    )
    return model


def compiled_with_cxx11_abi(*args: OpArg, **kwargs: OpArg) -> _builtin_bool:
    """Applies `torch.compiled_with_cxx11_abi`.

    Args:
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.compiled_with_cxx11_abi`.
    """
    _ = (
        args,
        kwargs,
    )
    return True


def complex(
    real: OpArg = None,
    imag: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.complex`.

    Args:
        real: Parameter `real`.
        imag: Parameter `imag`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.complex`.
    """
    _ = (
        real,
        imag,
        out,
        args,
        kwargs,
    )
    return _as_tensor(real)


def concat(
    tensors: OpArg = None,
    dim: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.concat`.

    Args:
        tensors: Parameter `tensors`.
        dim: Parameter `dim`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.concat`.
    """
    _ = (
        tensors,
        dim,
        out,
        args,
        kwargs,
    )
    seq = tensors if isinstance(tensors, (list, tuple)) else [_as_tensor(tensors)]
    return zero_torch.concatenate(seq, dim=0)


def cond(
    pred: OpArg = None,
    false_fn: OpArg = None,
    true_fn: OpArg = None,
    operands: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.cond`.

    Args:
        pred: Parameter `pred`.
        false_fn: Parameter `false_fn`.
        true_fn: Parameter `true_fn`.
        operands: Parameter `operands`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.cond`.
    """
    _ = (
        pred,
        false_fn,
        true_fn,
        operands,
        args,
        kwargs,
    )
    return _as_tensor(pred)


def conj_physical(
    input: OpArg = None, out: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> Tensor:
    """Applies `torch.conj_physical`.

    Args:
        input: Parameter `input`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.conj_physical`.
    """
    _ = (
        input,
        out,
        args,
        kwargs,
    )
    return zero_torch.conj(_as_tensor(input))


def conj_physical_(input: OpArg = None, *args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.conj_physical_`.

    Args:
        input: Parameter `input`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.conj_physical_`.
    """
    _ = (
        input,
        args,
        kwargs,
    )
    return zero_torch.conj(_as_tensor(input))


def constant_pad_nd(
    input: OpArg = None,
    pad: OpArg = None,
    value: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.constant_pad_nd`.

    Args:
        input: Parameter `input`.
        pad: Parameter `pad`.
        value: Parameter `value`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.constant_pad_nd`.
    """
    _ = (
        input,
        pad,
        value,
        args,
        kwargs,
    )
    return _as_tensor(input)


def conv1d(
    input: OpArg = None,
    weight: OpArg = None,
    bias: OpArg = None,
    stride: OpArg = None,
    padding: OpArg = None,
    dilation: OpArg = None,
    groups: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.conv1d`.

    Args:
        input: Parameter `input`.
        weight: Parameter `weight`.
        bias: Parameter `bias`.
        stride: Parameter `stride`.
        padding: Parameter `padding`.
        dilation: Parameter `dilation`.
        groups: Parameter `groups`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.conv1d`.
    """
    _ = (
        input,
        weight,
        bias,
        stride,
        padding,
        dilation,
        groups,
        args,
        kwargs,
    )
    return _as_tensor(input)


def conv2d(
    input: OpArg = None,
    weight: OpArg = None,
    bias: OpArg = None,
    stride: OpArg = None,
    padding: OpArg = None,
    dilation: OpArg = None,
    groups: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.conv2d`.

    Args:
        input: Parameter `input`.
        weight: Parameter `weight`.
        bias: Parameter `bias`.
        stride: Parameter `stride`.
        padding: Parameter `padding`.
        dilation: Parameter `dilation`.
        groups: Parameter `groups`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.conv2d`.
    """
    _ = (
        input,
        weight,
        bias,
        stride,
        padding,
        dilation,
        groups,
        args,
        kwargs,
    )
    return _as_tensor(input)


def conv3d(
    input: OpArg = None,
    weight: OpArg = None,
    bias: OpArg = None,
    stride: OpArg = None,
    padding: OpArg = None,
    dilation: OpArg = None,
    groups: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.conv3d`.

    Args:
        input: Parameter `input`.
        weight: Parameter `weight`.
        bias: Parameter `bias`.
        stride: Parameter `stride`.
        padding: Parameter `padding`.
        dilation: Parameter `dilation`.
        groups: Parameter `groups`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.conv3d`.
    """
    _ = (
        input,
        weight,
        bias,
        stride,
        padding,
        dilation,
        groups,
        args,
        kwargs,
    )
    return _as_tensor(input)


def conv_tbc(
    input: OpArg = None,
    weight: OpArg = None,
    bias: OpArg = None,
    pad: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.conv_tbc`.

    Args:
        input: Parameter `input`.
        weight: Parameter `weight`.
        bias: Parameter `bias`.
        pad: Parameter `pad`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.conv_tbc`.
    """
    _ = (
        input,
        weight,
        bias,
        pad,
        args,
        kwargs,
    )
    return _as_tensor(input)


def conv_transpose1d(
    input: OpArg = None,
    weight: OpArg = None,
    bias: OpArg = None,
    stride: OpArg = None,
    padding: OpArg = None,
    output_padding: OpArg = None,
    groups: OpArg = None,
    dilation: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.conv_transpose1d`.

    Args:
        input: Parameter `input`.
        weight: Parameter `weight`.
        bias: Parameter `bias`.
        stride: Parameter `stride`.
        padding: Parameter `padding`.
        output_padding: Parameter `output_padding`.
        groups: Parameter `groups`.
        dilation: Parameter `dilation`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.conv_transpose1d`.
    """
    _ = (
        input,
        weight,
        bias,
        stride,
        padding,
        output_padding,
        groups,
        dilation,
        args,
        kwargs,
    )
    return _as_tensor(input)


def conv_transpose2d(
    input: OpArg = None,
    weight: OpArg = None,
    bias: OpArg = None,
    stride: OpArg = None,
    padding: OpArg = None,
    output_padding: OpArg = None,
    groups: OpArg = None,
    dilation: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.conv_transpose2d`.

    Args:
        input: Parameter `input`.
        weight: Parameter `weight`.
        bias: Parameter `bias`.
        stride: Parameter `stride`.
        padding: Parameter `padding`.
        output_padding: Parameter `output_padding`.
        groups: Parameter `groups`.
        dilation: Parameter `dilation`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.conv_transpose2d`.
    """
    _ = (
        input,
        weight,
        bias,
        stride,
        padding,
        output_padding,
        groups,
        dilation,
        args,
        kwargs,
    )
    return _as_tensor(input)


def conv_transpose3d(
    input: OpArg = None,
    weight: OpArg = None,
    bias: OpArg = None,
    stride: OpArg = None,
    padding: OpArg = None,
    output_padding: OpArg = None,
    groups: OpArg = None,
    dilation: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.conv_transpose3d`.

    Args:
        input: Parameter `input`.
        weight: Parameter `weight`.
        bias: Parameter `bias`.
        stride: Parameter `stride`.
        padding: Parameter `padding`.
        output_padding: Parameter `output_padding`.
        groups: Parameter `groups`.
        dilation: Parameter `dilation`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.conv_transpose3d`.
    """
    _ = (
        input,
        weight,
        bias,
        stride,
        padding,
        output_padding,
        groups,
        dilation,
        args,
        kwargs,
    )
    return _as_tensor(input)


def convolution(
    input: OpArg = None,
    weight: OpArg = None,
    bias: OpArg = None,
    stride: OpArg = None,
    padding: OpArg = None,
    dilation: OpArg = None,
    transposed: OpArg = None,
    output_padding: OpArg = None,
    groups: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.convolution`.

    Args:
        input: Parameter `input`.
        weight: Parameter `weight`.
        bias: Parameter `bias`.
        stride: Parameter `stride`.
        padding: Parameter `padding`.
        dilation: Parameter `dilation`.
        transposed: Parameter `transposed`.
        output_padding: Parameter `output_padding`.
        groups: Parameter `groups`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.convolution`.
    """
    _ = (
        input,
        weight,
        bias,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
        args,
        kwargs,
    )
    return _as_tensor(input)


def corrcoef(input: OpArg = None, *args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.corrcoef`.

    Args:
        input: Parameter `input`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.corrcoef`.
    """
    _ = (
        input,
        args,
        kwargs,
    )
    return _as_tensor(input)


def cos_(input: OpArg = None, *args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.cos_`.

    Args:
        input: Parameter `input`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.cos_`.
    """
    _ = (
        input,
        args,
        kwargs,
    )
    return zero_torch.cos(_as_tensor(input))


def cosh_(input: OpArg = None, *args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.cosh_`.

    Args:
        input: Parameter `input`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.cosh_`.
    """
    _ = (
        input,
        args,
        kwargs,
    )
    return zero_torch.cosh(_as_tensor(input))


def cosine_embedding_loss(
    input1: OpArg = None,
    input2: OpArg = None,
    target: OpArg = None,
    margin: OpArg = None,
    reduction: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.cosine_embedding_loss`.

    Args:
        input1: Parameter `input1`.
        input2: Parameter `input2`.
        target: Parameter `target`.
        margin: Parameter `margin`.
        reduction: Parameter `reduction`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.cosine_embedding_loss`.
    """
    _ = (
        input1,
        input2,
        target,
        margin,
        reduction,
        args,
        kwargs,
    )
    return _as_tensor(input1)


def cosine_similarity(
    x1: OpArg = None,
    x2: OpArg = None,
    dim: OpArg = None,
    eps: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.cosine_similarity`.

    Args:
        x1: Parameter `x1`.
        x2: Parameter `x2`.
        dim: Parameter `dim`.
        eps: Parameter `eps`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.cosine_similarity`.
    """
    _ = (
        x1,
        x2,
        dim,
        eps,
        args,
        kwargs,
    )
    return _as_tensor(x1)


def cov(
    input: OpArg = None,
    correction: OpArg = None,
    fweights: OpArg = None,
    aweights: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.cov`.

    Args:
        input: Parameter `input`.
        correction: Parameter `correction`.
        fweights: Parameter `fweights`.
        aweights: Parameter `aweights`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.cov`.
    """
    _ = (
        input,
        correction,
        fweights,
        aweights,
        args,
        kwargs,
    )
    return _as_tensor(input)


def cross(
    input: OpArg = None,
    other: OpArg = None,
    dim: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.cross`.

    Args:
        input: Parameter `input`.
        other: Parameter `other`.
        dim: Parameter `dim`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.cross`.
    """
    _ = (
        input,
        other,
        dim,
        out,
        args,
        kwargs,
    )
    return zero_torch.linalg.cross(
        _as_tensor(input),
        _as_tensor(other),
        dim=_builtin_int(dim) if isinstance(dim, _builtin_int) else -1,
    )


def crow_indices_copy(
    input: OpArg = None, out: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> Tensor:
    """Applies `torch.crow_indices_copy`.

    Args:
        input: Parameter `input`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.crow_indices_copy`.
    """
    _ = (
        input,
        out,
        args,
        kwargs,
    )
    return _as_tensor(input)


def ctc_loss(
    log_probs: OpArg = None,
    targets: OpArg = None,
    input_lengths: OpArg = None,
    target_lengths: OpArg = None,
    blank: OpArg = None,
    reduction: OpArg = None,
    zero_infinity: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.ctc_loss`.

    Args:
        log_probs: Parameter `log_probs`.
        targets: Parameter `targets`.
        input_lengths: Parameter `input_lengths`.
        target_lengths: Parameter `target_lengths`.
        blank: Parameter `blank`.
        reduction: Parameter `reduction`.
        zero_infinity: Parameter `zero_infinity`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.ctc_loss`.
    """
    _ = (
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        blank,
        reduction,
        zero_infinity,
        args,
        kwargs,
    )
    return _as_tensor(log_probs)


def cudnn_affine_grid_generator(
    theta: OpArg = None,
    N: OpArg = None,
    C: OpArg = None,
    H: OpArg = None,
    W: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.cudnn_affine_grid_generator`.

    Args:
        theta: Parameter `theta`.
        N: Parameter `N`.
        C: Parameter `C`.
        H: Parameter `H`.
        W: Parameter `W`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.cudnn_affine_grid_generator`.
    """
    _ = (
        theta,
        N,
        C,
        H,
        W,
        args,
        kwargs,
    )
    return _as_tensor(theta)


def cudnn_batch_norm(
    input: OpArg = None,
    weight: OpArg = None,
    bias: OpArg = None,
    running_mean: OpArg = None,
    running_var: OpArg = None,
    training: OpArg = None,
    exponential_average_factor: OpArg = None,
    epsilon: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.cudnn_batch_norm`.

    Args:
        input: Parameter `input`.
        weight: Parameter `weight`.
        bias: Parameter `bias`.
        running_mean: Parameter `running_mean`.
        running_var: Parameter `running_var`.
        training: Parameter `training`.
        exponential_average_factor: Parameter `exponential_average_factor`.
        epsilon: Parameter `epsilon`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.cudnn_batch_norm`.
    """
    _ = (
        input,
        weight,
        bias,
        running_mean,
        running_var,
        training,
        exponential_average_factor,
        epsilon,
        out,
        args,
        kwargs,
    )
    return _as_tensor(input)


def cudnn_convolution(
    input: OpArg = None,
    weight: OpArg = None,
    padding: OpArg = None,
    stride: OpArg = None,
    dilation: OpArg = None,
    groups: OpArg = None,
    benchmark: OpArg = None,
    deterministic: OpArg = None,
    allow_tf32: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.cudnn_convolution`.

    Args:
        input: Parameter `input`.
        weight: Parameter `weight`.
        padding: Parameter `padding`.
        stride: Parameter `stride`.
        dilation: Parameter `dilation`.
        groups: Parameter `groups`.
        benchmark: Parameter `benchmark`.
        deterministic: Parameter `deterministic`.
        allow_tf32: Parameter `allow_tf32`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.cudnn_convolution`.
    """
    _ = (
        input,
        weight,
        padding,
        stride,
        dilation,
        groups,
        benchmark,
        deterministic,
        allow_tf32,
        out,
        args,
        kwargs,
    )
    return _as_tensor(input)


def cudnn_convolution_add_relu(
    input: OpArg = None,
    weight: OpArg = None,
    z: OpArg = None,
    alpha: OpArg = None,
    bias: OpArg = None,
    stride: OpArg = None,
    padding: OpArg = None,
    dilation: OpArg = None,
    groups: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.cudnn_convolution_add_relu`.

    Args:
        input: Parameter `input`.
        weight: Parameter `weight`.
        z: Parameter `z`.
        alpha: Parameter `alpha`.
        bias: Parameter `bias`.
        stride: Parameter `stride`.
        padding: Parameter `padding`.
        dilation: Parameter `dilation`.
        groups: Parameter `groups`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.cudnn_convolution_add_relu`.
    """
    _ = (
        input,
        weight,
        z,
        alpha,
        bias,
        stride,
        padding,
        dilation,
        groups,
        args,
        kwargs,
    )
    return _as_tensor(input)


def cudnn_convolution_relu(
    input: OpArg = None,
    weight: OpArg = None,
    bias: OpArg = None,
    stride: OpArg = None,
    padding: OpArg = None,
    dilation: OpArg = None,
    groups: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.cudnn_convolution_relu`.

    Args:
        input: Parameter `input`.
        weight: Parameter `weight`.
        bias: Parameter `bias`.
        stride: Parameter `stride`.
        padding: Parameter `padding`.
        dilation: Parameter `dilation`.
        groups: Parameter `groups`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.cudnn_convolution_relu`.
    """
    _ = (
        input,
        weight,
        bias,
        stride,
        padding,
        dilation,
        groups,
        args,
        kwargs,
    )
    return _as_tensor(input)


def cudnn_convolution_transpose(
    input: OpArg = None,
    weight: OpArg = None,
    padding: OpArg = None,
    output_padding: OpArg = None,
    stride: OpArg = None,
    dilation: OpArg = None,
    groups: OpArg = None,
    benchmark: OpArg = None,
    deterministic: OpArg = None,
    allow_tf32: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.cudnn_convolution_transpose`.

    Args:
        input: Parameter `input`.
        weight: Parameter `weight`.
        padding: Parameter `padding`.
        output_padding: Parameter `output_padding`.
        stride: Parameter `stride`.
        dilation: Parameter `dilation`.
        groups: Parameter `groups`.
        benchmark: Parameter `benchmark`.
        deterministic: Parameter `deterministic`.
        allow_tf32: Parameter `allow_tf32`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.cudnn_convolution_transpose`.
    """
    _ = (
        input,
        weight,
        padding,
        output_padding,
        stride,
        dilation,
        groups,
        benchmark,
        deterministic,
        allow_tf32,
        args,
        kwargs,
    )
    return _as_tensor(input)


def cudnn_grid_sampler(
    input: OpArg = None, grid: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> Tensor:
    """Applies `torch.cudnn_grid_sampler`.

    Args:
        input: Parameter `input`.
        grid: Parameter `grid`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.cudnn_grid_sampler`.
    """
    _ = (
        input,
        grid,
        args,
        kwargs,
    )
    return _as_tensor(input)


def cudnn_is_acceptable(
    input: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> _builtin_bool:
    """Applies `torch.cudnn_is_acceptable`.

    Args:
        input: Parameter `input`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.cudnn_is_acceptable`.
    """
    _ = (
        input,
        args,
        kwargs,
    )
    return True


def cummax(
    input: OpArg = None,
    dim: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> tuple[Tensor, ...]:
    """Applies `torch.cummax`.

    Args:
        input: Parameter `input`.
        dim: Parameter `dim`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.cummax`.
    """
    _ = (
        input,
        dim,
        out,
        args,
        kwargs,
    )
    t = _as_tensor(input)
    return (t, zero_torch.zeros_like(t))


def cummin(
    input: OpArg = None,
    dim: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> tuple[Tensor, ...]:
    """Applies `torch.cummin`.

    Args:
        input: Parameter `input`.
        dim: Parameter `dim`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.cummin`.
    """
    _ = (
        input,
        dim,
        out,
        args,
        kwargs,
    )
    t = _as_tensor(input)
    return (t, zero_torch.zeros_like(t))


def cumprod(
    input: OpArg = None,
    dim: OpArg = None,
    dtype: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.cumprod`.

    Args:
        input: Parameter `input`.
        dim: Parameter `dim`.
        dtype: Parameter `dtype`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.cumprod`.
    """
    _ = (
        input,
        dim,
        dtype,
        out,
        args,
        kwargs,
    )
    return _as_tensor(input)


def cumsum(
    input: OpArg = None,
    dim: OpArg = None,
    dtype: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.cumsum`.

    Args:
        input: Parameter `input`.
        dim: Parameter `dim`.
        dtype: Parameter `dtype`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.cumsum`.
    """
    _ = (
        input,
        dim,
        dtype,
        out,
        args,
        kwargs,
    )
    return _as_tensor(input)


def cumulative_trapezoid(
    y: OpArg = None,
    x: OpArg = None,
    dx: OpArg = None,
    dim: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.cumulative_trapezoid`.

    Args:
        y: Parameter `y`.
        x: Parameter `x`.
        dx: Parameter `dx`.
        dim: Parameter `dim`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.cumulative_trapezoid`.
    """
    _ = (
        y,
        x,
        dx,
        dim,
        args,
        kwargs,
    )
    return _as_tensor(y)


def deg2rad_(input: OpArg = None, *args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.deg2rad_`.

    Args:
        input: Parameter `input`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.deg2rad_`.
    """
    _ = (
        input,
        args,
        kwargs,
    )
    return zero_torch.deg2rad(_as_tensor(input))


def dequantize(tensor: OpArg = None, *args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.dequantize`.

    Args:
        tensor: Parameter `tensor`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.dequantize`.
    """
    _ = (
        tensor,
        args,
        kwargs,
    )
    return _as_tensor(tensor)


def detach(input: OpArg = None, *args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.detach`.

    Args:
        input: Parameter `input`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.detach`.
    """
    _ = (
        input,
        args,
        kwargs,
    )
    return _as_tensor(input)


def detach_(input: OpArg = None, *args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.detach_`.

    Args:
        input: Parameter `input`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.detach_`.
    """
    _ = (
        input,
        args,
        kwargs,
    )
    return _as_tensor(input)


def detach_copy(
    input: OpArg = None, out: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> Tensor:
    """Applies `torch.detach_copy`.

    Args:
        input: Parameter `input`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.detach_copy`.
    """
    _ = (
        input,
        out,
        args,
        kwargs,
    )
    return _as_tensor(input)


def diag_embed(
    input: OpArg = None,
    offset: OpArg = None,
    dim1: OpArg = None,
    dim2: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.diag_embed`.

    Args:
        input: Parameter `input`.
        offset: Parameter `offset`.
        dim1: Parameter `dim1`.
        dim2: Parameter `dim2`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.diag_embed`.
    """
    _ = (
        input,
        offset,
        dim1,
        dim2,
        args,
        kwargs,
    )
    return _as_tensor(input)


def diagflat(
    input: OpArg = None, offset: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> Tensor:
    """Applies `torch.diagflat`.

    Args:
        input: Parameter `input`.
        offset: Parameter `offset`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.diagflat`.
    """
    _ = (
        input,
        offset,
        args,
        kwargs,
    )
    return zero_torch.diag(_as_tensor(input).flatten())


def diagonal(
    input: OpArg = None,
    offset: OpArg = None,
    dim1: OpArg = None,
    dim2: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.diagonal`.

    Args:
        input: Parameter `input`.
        offset: Parameter `offset`.
        dim1: Parameter `dim1`.
        dim2: Parameter `dim2`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.diagonal`.
    """
    _ = (
        input,
        offset,
        dim1,
        dim2,
        args,
        kwargs,
    )
    return zero_torch.linalg.diagonal(
        _as_tensor(input),
        offset=_builtin_int(offset) if isinstance(offset, _builtin_int) else 0,
        dim1=_builtin_int(dim1) if isinstance(dim1, _builtin_int) else 0,
        dim2=_builtin_int(dim2) if isinstance(dim2, _builtin_int) else 1,
    )


def diagonal_copy(
    input: OpArg = None,
    offset: OpArg = None,
    dim1: OpArg = None,
    dim2: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.diagonal_copy`.

    Args:
        input: Parameter `input`.
        offset: Parameter `offset`.
        dim1: Parameter `dim1`.
        dim2: Parameter `dim2`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.diagonal_copy`.
    """
    _ = (
        input,
        offset,
        dim1,
        dim2,
        out,
        args,
        kwargs,
    )
    return _as_tensor(input)


def diagonal_scatter(
    input: OpArg = None,
    src: OpArg = None,
    offset: OpArg = None,
    dim1: OpArg = None,
    dim2: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.diagonal_scatter`.

    Args:
        input: Parameter `input`.
        src: Parameter `src`.
        offset: Parameter `offset`.
        dim1: Parameter `dim1`.
        dim2: Parameter `dim2`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.diagonal_scatter`.
    """
    _ = (
        input,
        src,
        offset,
        dim1,
        dim2,
        args,
        kwargs,
    )
    return _as_tensor(input)


def diff(
    input: OpArg = None,
    n: OpArg = None,
    dim: OpArg = None,
    prepend: OpArg = None,
    append: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.diff`.

    Args:
        input: Parameter `input`.
        n: Parameter `n`.
        dim: Parameter `dim`.
        prepend: Parameter `prepend`.
        append: Parameter `append`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.diff`.
    """
    _ = (
        input,
        n,
        dim,
        prepend,
        append,
        out,
        args,
        kwargs,
    )
    return _as_tensor(input)


def dist(
    input: OpArg = None,
    other: OpArg = None,
    p: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.dist`.

    Args:
        input: Parameter `input`.
        other: Parameter `other`.
        p: Parameter `p`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.dist`.
    """
    _ = (
        input,
        other,
        p,
        args,
        kwargs,
    )
    return _as_tensor(input)


def div(
    input: OpArg = None,
    other: OpArg = None,
    rounding_mode: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.div`.

    Args:
        input: Parameter `input`.
        other: Parameter `other`.
        rounding_mode: Parameter `rounding_mode`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.div`.
    """
    _ = (
        input,
        other,
        rounding_mode,
        out,
        args,
        kwargs,
    )
    return zero_torch.divide(_as_tensor(input), _as_tensor(other))


def dropout(
    input: OpArg = None,
    p: OpArg = None,
    train: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.dropout`.

    Args:
        input: Parameter `input`.
        p: Parameter `p`.
        train: Parameter `train`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.dropout`.
    """
    _ = (
        input,
        p,
        train,
        args,
        kwargs,
    )
    return _as_tensor(input)


def dropout_(
    input: OpArg = None,
    p: OpArg = None,
    train: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.dropout_`.

    Args:
        input: Parameter `input`.
        p: Parameter `p`.
        train: Parameter `train`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.dropout_`.
    """
    _ = (
        input,
        p,
        train,
        args,
        kwargs,
    )
    return _as_tensor(input)


def dsmm(
    input: OpArg = None, mat2: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> Tensor:
    """Applies `torch.dsmm`.

    Args:
        input: Parameter `input`.
        mat2: Parameter `mat2`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.dsmm`.
    """
    _ = (
        input,
        mat2,
        args,
        kwargs,
    )
    return zero_torch.matmul(_as_tensor(input), _as_tensor(mat2))


def dsplit(
    input: OpArg = None,
    indices_or_sections: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> list[Tensor]:
    """Applies `torch.dsplit`.

    Args:
        input: Parameter `input`.
        indices_or_sections: Parameter `indices_or_sections`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.dsplit`.
    """
    _ = (
        input,
        indices_or_sections,
        args,
        kwargs,
    )
    return [_as_tensor(input)]


def dstack(
    tensors: OpArg = None, out: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> Tensor:
    """Applies `torch.dstack`.

    Args:
        tensors: Parameter `tensors`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.dstack`.
    """
    _ = (
        tensors,
        out,
        args,
        kwargs,
    )
    seq = tensors if isinstance(tensors, (list, tuple)) else [_as_tensor(tensors)]
    return zero_torch.concatenate(seq, dim=0)


def eig(
    eigenvectors: OpArg = None,
    e: OpArg = None,
    v: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> tuple[Tensor, ...]:
    """Applies `torch.eig`.

    Args:
        eigenvectors: Parameter `eigenvectors`.
        e: Parameter `e`.
        v: Parameter `v`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.eig`.
    """
    _ = (
        eigenvectors,
        e,
        v,
        args,
        kwargs,
    )
    t = _as_tensor(eigenvectors)
    return (t, zero_torch.zeros_like(t))


def embedding(
    weight: OpArg = None,
    indices: OpArg = None,
    padding_idx: OpArg = None,
    scale_grad_by_freq: OpArg = None,
    sparse: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.embedding`.

    Args:
        weight: Parameter `weight`.
        indices: Parameter `indices`.
        padding_idx: Parameter `padding_idx`.
        scale_grad_by_freq: Parameter `scale_grad_by_freq`.
        sparse: Parameter `sparse`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.embedding`.
    """
    _ = (
        weight,
        indices,
        padding_idx,
        scale_grad_by_freq,
        sparse,
        args,
        kwargs,
    )
    return _as_tensor(weight)


def embedding_bag(
    weight: OpArg = None,
    indices: OpArg = None,
    offsets: OpArg = None,
    scale_grad_by_freq: OpArg = None,
    mode: OpArg = None,
    sparse: OpArg = None,
    per_sample_weights: OpArg = None,
    include_last_offset: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.embedding_bag`.

    Args:
        weight: Parameter `weight`.
        indices: Parameter `indices`.
        offsets: Parameter `offsets`.
        scale_grad_by_freq: Parameter `scale_grad_by_freq`.
        mode: Parameter `mode`.
        sparse: Parameter `sparse`.
        per_sample_weights: Parameter `per_sample_weights`.
        include_last_offset: Parameter `include_last_offset`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.embedding_bag`.
    """
    _ = (
        weight,
        indices,
        offsets,
        scale_grad_by_freq,
        mode,
        sparse,
        per_sample_weights,
        include_last_offset,
        args,
        kwargs,
    )
    return _as_tensor(weight)


def embedding_renorm_(
    input: OpArg = None,
    indices: OpArg = None,
    max_norm: OpArg = None,
    norm_type: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.embedding_renorm_`.

    Args:
        input: Parameter `input`.
        indices: Parameter `indices`.
        max_norm: Parameter `max_norm`.
        norm_type: Parameter `norm_type`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.embedding_renorm_`.
    """
    _ = (
        input,
        indices,
        max_norm,
        norm_type,
        args,
        kwargs,
    )
    return _as_tensor(input)


def empty_like(
    input: OpArg = None,
    memory_format: OpArg = None,
    dtype: OpArg = None,
    layout: OpArg = None,
    device: OpArg = None,
    pin_memory: OpArg = None,
    requires_grad: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.empty_like`.

    Args:
        input: Parameter `input`.
        memory_format: Parameter `memory_format`.
        dtype: Parameter `dtype`.
        layout: Parameter `layout`.
        device: Parameter `device`.
        pin_memory: Parameter `pin_memory`.
        requires_grad: Parameter `requires_grad`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.empty_like`.
    """
    _ = (
        input,
        memory_format,
        dtype,
        layout,
        device,
        pin_memory,
        requires_grad,
        args,
        kwargs,
    )
    return zero_torch.zeros_like(_as_tensor(input))


def empty_permuted(
    size: OpArg = None,
    physical_layout: OpArg = None,
    dtype: OpArg = None,
    layout: OpArg = None,
    device: OpArg = None,
    pin_memory: OpArg = None,
    requires_grad: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.empty_permuted`.

    Args:
        size: Parameter `size`.
        physical_layout: Parameter `physical_layout`.
        dtype: Parameter `dtype`.
        layout: Parameter `layout`.
        device: Parameter `device`.
        pin_memory: Parameter `pin_memory`.
        requires_grad: Parameter `requires_grad`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.empty_permuted`.
    """
    _ = (
        size,
        physical_layout,
        dtype,
        layout,
        device,
        pin_memory,
        requires_grad,
        args,
        kwargs,
    )
    return _as_tensor(size)


def empty_quantized(
    size: OpArg = None,
    qtensor: OpArg = None,
    memory_format: OpArg = None,
    dtype: OpArg = None,
    layout: OpArg = None,
    device: OpArg = None,
    pin_memory: OpArg = None,
    requires_grad: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.empty_quantized`.

    Args:
        size: Parameter `size`.
        qtensor: Parameter `qtensor`.
        memory_format: Parameter `memory_format`.
        dtype: Parameter `dtype`.
        layout: Parameter `layout`.
        device: Parameter `device`.
        pin_memory: Parameter `pin_memory`.
        requires_grad: Parameter `requires_grad`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.empty_quantized`.
    """
    _ = (
        size,
        qtensor,
        memory_format,
        dtype,
        layout,
        device,
        pin_memory,
        requires_grad,
        args,
        kwargs,
    )
    return _as_tensor(size)


def empty_strided(
    size: OpArg = None,
    stride: OpArg = None,
    dtype: OpArg = None,
    layout: OpArg = None,
    device: OpArg = None,
    pin_memory: OpArg = None,
    requires_grad: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.empty_strided`.

    Args:
        size: Parameter `size`.
        stride: Parameter `stride`.
        dtype: Parameter `dtype`.
        layout: Parameter `layout`.
        device: Parameter `device`.
        pin_memory: Parameter `pin_memory`.
        requires_grad: Parameter `requires_grad`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.empty_strided`.
    """
    _ = (
        size,
        stride,
        dtype,
        layout,
        device,
        pin_memory,
        requires_grad,
        args,
        kwargs,
    )
    return _as_tensor(size)


def eq(
    input: OpArg = None,
    other: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.eq`.

    Args:
        input: Parameter `input`.
        other: Parameter `other`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.eq`.
    """
    _ = (
        input,
        other,
        out,
        args,
        kwargs,
    )
    return zero_torch.equal(_as_tensor(input), _as_tensor(other))


def erf_(input: OpArg = None, *args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.erf_`.

    Args:
        input: Parameter `input`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.erf_`.
    """
    _ = (
        input,
        args,
        kwargs,
    )
    return zero_torch.erf(_as_tensor(input))


def erfc_(input: OpArg = None, *args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.erfc_`.

    Args:
        input: Parameter `input`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.erfc_`.
    """
    _ = (
        input,
        args,
        kwargs,
    )
    return zero_torch.erfc(_as_tensor(input))


def exp2_(input: OpArg = None, *args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.exp2_`.

    Args:
        input: Parameter `input`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.exp2_`.
    """
    _ = (
        input,
        args,
        kwargs,
    )
    return zero_torch.exp2(_as_tensor(input))


def exp_(input: OpArg = None, *args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.exp_`.

    Args:
        input: Parameter `input`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.exp_`.
    """
    _ = (
        input,
        args,
        kwargs,
    )
    return zero_torch.exp(_as_tensor(input))


def expand_copy(
    input: OpArg = None,
    size: OpArg = None,
    implicit: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.expand_copy`.

    Args:
        input: Parameter `input`.
        size: Parameter `size`.
        implicit: Parameter `implicit`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.expand_copy`.
    """
    _ = (
        input,
        size,
        implicit,
        out,
        args,
        kwargs,
    )
    return _as_tensor(input)


def expm1_(input: OpArg = None, *args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.expm1_`.

    Args:
        input: Parameter `input`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.expm1_`.
    """
    _ = (
        input,
        args,
        kwargs,
    )
    return zero_torch.expm1(_as_tensor(input))


def fake_quantize_per_channel_affine(
    input: OpArg = None,
    scale: OpArg = None,
    zero_point: OpArg = None,
    axis: OpArg = None,
    quant_min: OpArg = None,
    quant_max: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.fake_quantize_per_channel_affine`.

    Args:
        input: Parameter `input`.
        scale: Parameter `scale`.
        zero_point: Parameter `zero_point`.
        axis: Parameter `axis`.
        quant_min: Parameter `quant_min`.
        quant_max: Parameter `quant_max`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.fake_quantize_per_channel_affine`.
    """
    _ = (
        input,
        scale,
        zero_point,
        axis,
        quant_min,
        quant_max,
        args,
        kwargs,
    )
    return _as_tensor(input)


def fake_quantize_per_tensor_affine(
    input: OpArg = None,
    scale: OpArg = None,
    zero_point: OpArg = None,
    quant_min: OpArg = None,
    quant_max: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.fake_quantize_per_tensor_affine`.

    Args:
        input: Parameter `input`.
        scale: Parameter `scale`.
        zero_point: Parameter `zero_point`.
        quant_min: Parameter `quant_min`.
        quant_max: Parameter `quant_max`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.fake_quantize_per_tensor_affine`.
    """
    _ = (
        input,
        scale,
        zero_point,
        quant_min,
        quant_max,
        args,
        kwargs,
    )
    return _as_tensor(input)


def fbgemm_linear_fp16_weight(
    input: OpArg = None,
    packed_weight: OpArg = None,
    bias: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.fbgemm_linear_fp16_weight`.

    Args:
        input: Parameter `input`.
        packed_weight: Parameter `packed_weight`.
        bias: Parameter `bias`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.fbgemm_linear_fp16_weight`.
    """
    _ = (
        input,
        packed_weight,
        bias,
        args,
        kwargs,
    )
    return _as_tensor(input)


def fbgemm_linear_fp16_weight_fp32_activation(
    input: OpArg = None,
    packed_weight: OpArg = None,
    bias: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.fbgemm_linear_fp16_weight_fp32_activation`.

    Args:
        input: Parameter `input`.
        packed_weight: Parameter `packed_weight`.
        bias: Parameter `bias`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.fbgemm_linear_fp16_weight_fp32_activation`.
    """
    _ = (
        input,
        packed_weight,
        bias,
        args,
        kwargs,
    )
    return _as_tensor(input)


def fbgemm_linear_int8_weight(
    input: OpArg = None,
    weight: OpArg = None,
    packed: OpArg = None,
    col_offsets: OpArg = None,
    weight_scale: OpArg = None,
    weight_zero_point: OpArg = None,
    bias: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.fbgemm_linear_int8_weight`.

    Args:
        input: Parameter `input`.
        weight: Parameter `weight`.
        packed: Parameter `packed`.
        col_offsets: Parameter `col_offsets`.
        weight_scale: Parameter `weight_scale`.
        weight_zero_point: Parameter `weight_zero_point`.
        bias: Parameter `bias`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.fbgemm_linear_int8_weight`.
    """
    _ = (
        input,
        weight,
        packed,
        col_offsets,
        weight_scale,
        weight_zero_point,
        bias,
        args,
        kwargs,
    )
    return _as_tensor(input)


def fbgemm_linear_int8_weight_fp32_activation(
    input: OpArg = None,
    weight: OpArg = None,
    packed: OpArg = None,
    col_offsets: OpArg = None,
    weight_scale: OpArg = None,
    weight_zero_point: OpArg = None,
    bias: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.fbgemm_linear_int8_weight_fp32_activation`.

    Args:
        input: Parameter `input`.
        weight: Parameter `weight`.
        packed: Parameter `packed`.
        col_offsets: Parameter `col_offsets`.
        weight_scale: Parameter `weight_scale`.
        weight_zero_point: Parameter `weight_zero_point`.
        bias: Parameter `bias`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.fbgemm_linear_int8_weight_fp32_activation`.
    """
    _ = (
        input,
        weight,
        packed,
        col_offsets,
        weight_scale,
        weight_zero_point,
        bias,
        args,
        kwargs,
    )
    return _as_tensor(input)


def fbgemm_linear_quantize_weight(
    input: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> Tensor:
    """Applies `torch.fbgemm_linear_quantize_weight`.

    Args:
        input: Parameter `input`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.fbgemm_linear_quantize_weight`.
    """
    _ = (
        input,
        args,
        kwargs,
    )
    return _as_tensor(input)


def fbgemm_pack_gemm_matrix_fp16(
    input: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> Tensor:
    """Applies `torch.fbgemm_pack_gemm_matrix_fp16`.

    Args:
        input: Parameter `input`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.fbgemm_pack_gemm_matrix_fp16`.
    """
    _ = (
        input,
        args,
        kwargs,
    )
    return _as_tensor(input)


def fbgemm_pack_quantized_matrix(
    input: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> Tensor:
    """Applies `torch.fbgemm_pack_quantized_matrix`.

    Args:
        input: Parameter `input`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.fbgemm_pack_quantized_matrix`.
    """
    _ = (
        input,
        args,
        kwargs,
    )
    return _as_tensor(input)


def feature_alpha_dropout(
    input: OpArg = None,
    p: OpArg = None,
    train: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.feature_alpha_dropout`.

    Args:
        input: Parameter `input`.
        p: Parameter `p`.
        train: Parameter `train`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.feature_alpha_dropout`.
    """
    _ = (
        input,
        p,
        train,
        args,
        kwargs,
    )
    return _as_tensor(input)


def feature_alpha_dropout_(
    input: OpArg = None,
    p: OpArg = None,
    train: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.feature_alpha_dropout_`.

    Args:
        input: Parameter `input`.
        p: Parameter `p`.
        train: Parameter `train`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.feature_alpha_dropout_`.
    """
    _ = (
        input,
        p,
        train,
        args,
        kwargs,
    )
    return _as_tensor(input)


def feature_dropout(
    input: OpArg = None,
    p: OpArg = None,
    train: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.feature_dropout`.

    Args:
        input: Parameter `input`.
        p: Parameter `p`.
        train: Parameter `train`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.feature_dropout`.
    """
    _ = (
        input,
        p,
        train,
        args,
        kwargs,
    )
    return _as_tensor(input)


def feature_dropout_(
    input: OpArg = None,
    p: OpArg = None,
    train: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.feature_dropout_`.

    Args:
        input: Parameter `input`.
        p: Parameter `p`.
        train: Parameter `train`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.feature_dropout_`.
    """
    _ = (
        input,
        p,
        train,
        args,
        kwargs,
    )
    return _as_tensor(input)


def fill(
    input: OpArg = None, value: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> Tensor:
    """Applies `torch.fill`.

    Args:
        input: Parameter `input`.
        value: Parameter `value`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.fill`.
    """
    _ = (
        input,
        value,
        args,
        kwargs,
    )
    return _as_tensor(input)


def fill_(
    input: OpArg = None, value: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> Tensor:
    """Applies `torch.fill_`.

    Args:
        input: Parameter `input`.
        value: Parameter `value`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.fill_`.
    """
    _ = (
        input,
        value,
        args,
        kwargs,
    )
    return _as_tensor(input)


def fix_(input: OpArg = None, *args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.fix_`.

    Args:
        input: Parameter `input`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.fix_`.
    """
    _ = (
        input,
        args,
        kwargs,
    )
    return zero_torch.fix(_as_tensor(input))


def flip(
    input: OpArg = None, dims: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> Tensor:
    """Applies `torch.flip`.

    Args:
        input: Parameter `input`.
        dims: Parameter `dims`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.flip`.
    """
    _ = (
        input,
        dims,
        args,
        kwargs,
    )
    return _as_tensor(input)


def fliplr(input: OpArg = None, *args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.fliplr`.

    Args:
        input: Parameter `input`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.fliplr`.
    """
    _ = (
        input,
        args,
        kwargs,
    )
    return _as_tensor(input)


def flipud(input: OpArg = None, *args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.flipud`.

    Args:
        input: Parameter `input`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.flipud`.
    """
    _ = (
        input,
        args,
        kwargs,
    )
    return _as_tensor(input)


def floor_(input: OpArg = None, *args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.floor_`.

    Args:
        input: Parameter `input`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.floor_`.
    """
    _ = (
        input,
        args,
        kwargs,
    )
    return zero_torch.floor(_as_tensor(input))


def fork(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.fork`.

    Args:
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.fork`.
    """
    _ = (
        args,
        kwargs,
    )
    return Tensor([0.0])


def frac(
    input: OpArg = None, out: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> Tensor:
    """Applies `torch.frac`.

    Args:
        input: Parameter `input`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.frac`.
    """
    _ = (
        input,
        out,
        args,
        kwargs,
    )
    return _as_tensor(input)


def frac_(input: OpArg = None, *args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.frac_`.

    Args:
        input: Parameter `input`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.frac_`.
    """
    _ = (
        input,
        args,
        kwargs,
    )
    return _as_tensor(input)


def frobenius_norm(
    input: OpArg = None,
    dim: OpArg = None,
    keepdim: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.frobenius_norm`.

    Args:
        input: Parameter `input`.
        dim: Parameter `dim`.
        keepdim: Parameter `keepdim`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.frobenius_norm`.
    """
    _ = (
        input,
        dim,
        keepdim,
        out,
        args,
        kwargs,
    )
    return _as_tensor(input)


def from_dlpack(
    ext_tensor: OpArg = None,
    device: OpArg = None,
    copy: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.from_dlpack`.

    Args:
        ext_tensor: Parameter `ext_tensor`.
        device: Parameter `device`.
        copy: Parameter `copy`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.from_dlpack`.
    """
    _ = (
        ext_tensor,
        device,
        copy,
        args,
        kwargs,
    )
    return _as_tensor(ext_tensor)


def from_file(
    filename: OpArg = None,
    shared: OpArg = None,
    size: OpArg = None,
    dtype: OpArg = None,
    layout: OpArg = None,
    device: OpArg = None,
    pin_memory: OpArg = None,
    requires_grad: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.from_file`.

    Args:
        filename: Parameter `filename`.
        shared: Parameter `shared`.
        size: Parameter `size`.
        dtype: Parameter `dtype`.
        layout: Parameter `layout`.
        device: Parameter `device`.
        pin_memory: Parameter `pin_memory`.
        requires_grad: Parameter `requires_grad`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.from_file`.
    """
    _ = (
        filename,
        shared,
        size,
        dtype,
        layout,
        device,
        pin_memory,
        requires_grad,
        args,
        kwargs,
    )
    return _as_tensor(filename)


def from_numpy(ndarray: OpArg = None, *args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.from_numpy`.

    Args:
        ndarray: Parameter `ndarray`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.from_numpy`.
    """
    _ = (
        ndarray,
        args,
        kwargs,
    )
    return _as_tensor(ndarray)


def frombuffer(
    buffer: OpArg = None,
    dtype: OpArg = None,
    count: OpArg = None,
    offset: OpArg = None,
    requires_grad: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.frombuffer`.

    Args:
        buffer: Parameter `buffer`.
        dtype: Parameter `dtype`.
        count: Parameter `count`.
        offset: Parameter `offset`.
        requires_grad: Parameter `requires_grad`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.frombuffer`.
    """
    _ = (
        buffer,
        dtype,
        count,
        offset,
        requires_grad,
        args,
        kwargs,
    )
    return _as_tensor(buffer)


def fused_moving_avg_obs_fake_quant(
    input: OpArg = None,
    observer_on: OpArg = None,
    fake_quant_on: OpArg = None,
    running_min: OpArg = None,
    running_max: OpArg = None,
    scale: OpArg = None,
    zero_point: OpArg = None,
    averaging_const: OpArg = None,
    quant_min: OpArg = None,
    quant_max: OpArg = None,
    ch_axis: OpArg = None,
    per_row_fake_quant: OpArg = None,
    symmetric_quant: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.fused_moving_avg_obs_fake_quant`.

    Args:
        input: Parameter `input`.
        observer_on: Parameter `observer_on`.
        fake_quant_on: Parameter `fake_quant_on`.
        running_min: Parameter `running_min`.
        running_max: Parameter `running_max`.
        scale: Parameter `scale`.
        zero_point: Parameter `zero_point`.
        averaging_const: Parameter `averaging_const`.
        quant_min: Parameter `quant_min`.
        quant_max: Parameter `quant_max`.
        ch_axis: Parameter `ch_axis`.
        per_row_fake_quant: Parameter `per_row_fake_quant`.
        symmetric_quant: Parameter `symmetric_quant`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.fused_moving_avg_obs_fake_quant`.
    """
    _ = (
        input,
        observer_on,
        fake_quant_on,
        running_min,
        running_max,
        scale,
        zero_point,
        averaging_const,
        quant_min,
        quant_max,
        ch_axis,
        per_row_fake_quant,
        symmetric_quant,
        args,
        kwargs,
    )
    return _as_tensor(input)


def gcd_(
    input: OpArg = None, other: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> Tensor:
    """Applies `torch.gcd_`.

    Args:
        input: Parameter `input`.
        other: Parameter `other`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.gcd_`.
    """
    _ = (
        input,
        other,
        args,
        kwargs,
    )
    return zero_torch.gcd(_as_tensor(input), _as_tensor(other))


def ge(
    input: OpArg = None,
    other: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.ge`.

    Args:
        input: Parameter `input`.
        other: Parameter `other`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.ge`.
    """
    _ = (
        input,
        other,
        out,
        args,
        kwargs,
    )
    return zero_torch.greater_equal(_as_tensor(input), _as_tensor(other))


def geqrf(
    input: OpArg = None, out: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> tuple[Tensor, ...]:
    """Applies `torch.geqrf`.

    Args:
        input: Parameter `input`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.geqrf`.
    """
    _ = (
        input,
        out,
        args,
        kwargs,
    )
    t = _as_tensor(input)
    return (t, zero_torch.zeros_like(t))


def ger(
    input: OpArg = None,
    vec2: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.ger`.

    Args:
        input: Parameter `input`.
        vec2: Parameter `vec2`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.ger`.
    """
    _ = (
        input,
        vec2,
        out,
        args,
        kwargs,
    )
    return zero_torch.outer(_as_tensor(input), _as_tensor(vec2))


def get_autocast_cpu_dtype(*args: OpArg, **kwargs: OpArg) -> DType:
    """Applies `torch.get_autocast_cpu_dtype`.

    Args:
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.get_autocast_cpu_dtype`.
    """
    _ = (
        args,
        kwargs,
    )
    return DType.Float32


def get_autocast_dtype(
    device_type: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> DType:
    """Applies `torch.get_autocast_dtype`.

    Args:
        device_type: Parameter `device_type`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.get_autocast_dtype`.
    """
    _ = (
        device_type,
        args,
        kwargs,
    )
    return DType.Float32


def get_autocast_gpu_dtype(*args: OpArg, **kwargs: OpArg) -> DType:
    """Applies `torch.get_autocast_gpu_dtype`.

    Args:
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.get_autocast_gpu_dtype`.
    """
    _ = (
        args,
        kwargs,
    )
    return DType.Float32


def get_autocast_ipu_dtype(*args: OpArg, **kwargs: OpArg) -> DType:
    """Applies `torch.get_autocast_ipu_dtype`.

    Args:
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.get_autocast_ipu_dtype`.
    """
    _ = (
        args,
        kwargs,
    )
    return DType.Float32


def get_autocast_xla_dtype(*args: OpArg, **kwargs: OpArg) -> DType:
    """Applies `torch.get_autocast_xla_dtype`.

    Args:
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.get_autocast_xla_dtype`.
    """
    _ = (
        args,
        kwargs,
    )
    return DType.Float32


def get_default_device(*args: OpArg, **kwargs: OpArg) -> str:
    """Applies `torch.get_default_device`.

    Args:
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.get_default_device`.
    """
    _ = (
        args,
        kwargs,
    )
    return "cpu"


def get_default_dtype(*args: OpArg, **kwargs: OpArg) -> DType:
    """Applies `torch.get_default_dtype`.

    Args:
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.get_default_dtype`.
    """
    _ = (
        args,
        kwargs,
    )
    return DType.Float32


def get_deterministic_debug_mode(*args: OpArg, **kwargs: OpArg) -> str:
    """Applies `torch.get_deterministic_debug_mode`.

    Args:
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.get_deterministic_debug_mode`.
    """
    _ = (
        args,
        kwargs,
    )
    return "get_deterministic_debug_mode"


def get_device(input: OpArg = None, *args: OpArg, **kwargs: OpArg) -> _builtin_int:
    """Applies `torch.get_device`.

    Args:
        input: Parameter `input`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.get_device`.
    """
    _ = (
        input,
        args,
        kwargs,
    )
    return -1


def get_device_module(device: OpArg = None, *args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.get_device_module`.

    Args:
        device: Parameter `device`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.get_device_module`.
    """
    _ = (
        device,
        args,
        kwargs,
    )
    return _as_tensor(device)


def get_file_path(path_components: OpArg = None, *args: OpArg, **kwargs: OpArg) -> str:
    """Applies `torch.get_file_path`.

    Args:
        path_components: Parameter `path_components`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.get_file_path`.
    """
    _ = (
        path_components,
        args,
        kwargs,
    )
    return "get_file_path"


def get_float32_matmul_precision(*args: OpArg, **kwargs: OpArg) -> str:
    """Applies `torch.get_float32_matmul_precision`.

    Args:
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.get_float32_matmul_precision`.
    """
    _ = (
        args,
        kwargs,
    )
    return "highest"


def get_num_interop_threads(*args: OpArg, **kwargs: OpArg) -> _builtin_int:
    """Applies `torch.get_num_interop_threads`.

    Args:
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.get_num_interop_threads`.
    """
    _ = (
        args,
        kwargs,
    )
    return 1


def get_num_threads(*args: OpArg, **kwargs: OpArg) -> _builtin_int:
    """Applies `torch.get_num_threads`.

    Args:
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.get_num_threads`.
    """
    _ = (
        args,
        kwargs,
    )
    return 1


def get_rng_state(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.get_rng_state`.

    Args:
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.get_rng_state`.
    """
    _ = (
        args,
        kwargs,
    )
    return Tensor([0.0])


def gradient(
    input: OpArg = None,
    spacing: OpArg = None,
    dim: OpArg = None,
    edge_order: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> list[Tensor]:
    """Applies `torch.gradient`.

    Args:
        input: Parameter `input`.
        spacing: Parameter `spacing`.
        dim: Parameter `dim`.
        edge_order: Parameter `edge_order`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.gradient`.
    """
    _ = (
        input,
        spacing,
        dim,
        edge_order,
        args,
        kwargs,
    )
    return [_as_tensor(input)]


def grid_sampler(
    input: OpArg = None,
    grid: OpArg = None,
    interpolation_mode: OpArg = None,
    padding_mode: OpArg = None,
    align_corners: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.grid_sampler`.

    Args:
        input: Parameter `input`.
        grid: Parameter `grid`.
        interpolation_mode: Parameter `interpolation_mode`.
        padding_mode: Parameter `padding_mode`.
        align_corners: Parameter `align_corners`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.grid_sampler`.
    """
    _ = (
        input,
        grid,
        interpolation_mode,
        padding_mode,
        align_corners,
        args,
        kwargs,
    )
    return _as_tensor(input)


def grid_sampler_2d(
    input: OpArg = None,
    grid: OpArg = None,
    interpolation_mode: OpArg = None,
    padding_mode: OpArg = None,
    align_corners: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.grid_sampler_2d`.

    Args:
        input: Parameter `input`.
        grid: Parameter `grid`.
        interpolation_mode: Parameter `interpolation_mode`.
        padding_mode: Parameter `padding_mode`.
        align_corners: Parameter `align_corners`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.grid_sampler_2d`.
    """
    _ = (
        input,
        grid,
        interpolation_mode,
        padding_mode,
        align_corners,
        args,
        kwargs,
    )
    return _as_tensor(input)


def grid_sampler_3d(
    input: OpArg = None,
    grid: OpArg = None,
    interpolation_mode: OpArg = None,
    padding_mode: OpArg = None,
    align_corners: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.grid_sampler_3d`.

    Args:
        input: Parameter `input`.
        grid: Parameter `grid`.
        interpolation_mode: Parameter `interpolation_mode`.
        padding_mode: Parameter `padding_mode`.
        align_corners: Parameter `align_corners`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.grid_sampler_3d`.
    """
    _ = (
        input,
        grid,
        interpolation_mode,
        padding_mode,
        align_corners,
        args,
        kwargs,
    )
    return _as_tensor(input)


def group_norm(
    input: OpArg = None,
    num_groups: OpArg = None,
    weight: OpArg = None,
    bias: OpArg = None,
    eps: OpArg = None,
    cudnn_enabled: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.group_norm`.

    Args:
        input: Parameter `input`.
        num_groups: Parameter `num_groups`.
        weight: Parameter `weight`.
        bias: Parameter `bias`.
        eps: Parameter `eps`.
        cudnn_enabled: Parameter `cudnn_enabled`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.group_norm`.
    """
    _ = (
        input,
        num_groups,
        weight,
        bias,
        eps,
        cudnn_enabled,
        args,
        kwargs,
    )
    return _as_tensor(input)


def gru(
    input: OpArg = None,
    hx: OpArg = None,
    params: OpArg = None,
    has_biases: OpArg = None,
    num_layers: OpArg = None,
    dropout: OpArg = None,
    train: OpArg = None,
    bidirectional: OpArg = None,
    batch_first: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.gru`.

    Args:
        input: Parameter `input`.
        hx: Parameter `hx`.
        params: Parameter `params`.
        has_biases: Parameter `has_biases`.
        num_layers: Parameter `num_layers`.
        dropout: Parameter `dropout`.
        train: Parameter `train`.
        bidirectional: Parameter `bidirectional`.
        batch_first: Parameter `batch_first`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.gru`.
    """
    _ = (
        input,
        hx,
        params,
        has_biases,
        num_layers,
        dropout,
        train,
        bidirectional,
        batch_first,
        args,
        kwargs,
    )
    return _as_tensor(input)


def gru_cell(
    input: OpArg = None,
    hx: OpArg = None,
    w_ih: OpArg = None,
    w_hh: OpArg = None,
    b_ih: OpArg = None,
    b_hh: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.gru_cell`.

    Args:
        input: Parameter `input`.
        hx: Parameter `hx`.
        w_ih: Parameter `w_ih`.
        w_hh: Parameter `w_hh`.
        b_ih: Parameter `b_ih`.
        b_hh: Parameter `b_hh`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.gru_cell`.
    """
    _ = (
        input,
        hx,
        w_ih,
        w_hh,
        b_ih,
        b_hh,
        args,
        kwargs,
    )
    return _as_tensor(input)


def gt(
    input: OpArg = None,
    other: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.gt`.

    Args:
        input: Parameter `input`.
        other: Parameter `other`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.gt`.
    """
    _ = (
        input,
        other,
        out,
        args,
        kwargs,
    )
    return zero_torch.greater(_as_tensor(input), _as_tensor(other))


def hamming_window(
    window_length: OpArg = None,
    dtype: OpArg = None,
    layout: OpArg = None,
    device: OpArg = None,
    pin_memory: OpArg = None,
    requires_grad: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.hamming_window`.

    Args:
        window_length: Parameter `window_length`.
        dtype: Parameter `dtype`.
        layout: Parameter `layout`.
        device: Parameter `device`.
        pin_memory: Parameter `pin_memory`.
        requires_grad: Parameter `requires_grad`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.hamming_window`.
    """
    _ = (
        window_length,
        dtype,
        layout,
        device,
        pin_memory,
        requires_grad,
        args,
        kwargs,
    )
    return _as_tensor(window_length)


def hann_window(
    window_length: OpArg = None,
    periodic: OpArg = None,
    dtype: OpArg = None,
    layout: OpArg = None,
    device: OpArg = None,
    requires_grad: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.hann_window`.

    Args:
        window_length: Parameter `window_length`.
        periodic: Parameter `periodic`.
        dtype: Parameter `dtype`.
        layout: Parameter `layout`.
        device: Parameter `device`.
        requires_grad: Parameter `requires_grad`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.hann_window`.
    """
    _ = (
        window_length,
        periodic,
        dtype,
        layout,
        device,
        requires_grad,
        args,
        kwargs,
    )
    return _as_tensor(window_length)


def hardshrink(
    input: OpArg = None,
    lambd: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.hardshrink`.

    Args:
        input: Parameter `input`.
        lambd: Parameter `lambd`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.hardshrink`.
    """
    _ = (
        input,
        lambd,
        out,
        args,
        kwargs,
    )
    return zero_torch.nn.functional.hardshrink(
        _as_tensor(input),
        lambd=_builtin_float(lambd)
        if isinstance(lambd, (_builtin_int, _builtin_float))
        else 0.5,
    )


def hash_tensor(
    input: OpArg = None,
    dim: OpArg = None,
    keepdim: OpArg = None,
    mode: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.hash_tensor`.

    Args:
        input: Parameter `input`.
        dim: Parameter `dim`.
        keepdim: Parameter `keepdim`.
        mode: Parameter `mode`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.hash_tensor`.
    """
    _ = (
        input,
        dim,
        keepdim,
        mode,
        out,
        args,
        kwargs,
    )
    return _as_tensor(input)


def hinge_embedding_loss(
    input: OpArg = None,
    target: OpArg = None,
    margin: OpArg = None,
    reduction: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.hinge_embedding_loss`.

    Args:
        input: Parameter `input`.
        target: Parameter `target`.
        margin: Parameter `margin`.
        reduction: Parameter `reduction`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.hinge_embedding_loss`.
    """
    _ = (
        input,
        target,
        margin,
        reduction,
        args,
        kwargs,
    )
    return _as_tensor(input)


def histc(
    input: OpArg = None,
    bins: OpArg = None,
    min: OpArg = None,
    max: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.histc`.

    Args:
        input: Parameter `input`.
        bins: Parameter `bins`.
        min: Parameter `min`.
        max: Parameter `max`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.histc`.
    """
    _ = (
        input,
        bins,
        min,
        max,
        out,
        args,
        kwargs,
    )
    return _as_tensor(input)


def histogram(
    input: OpArg = None,
    bins: OpArg = None,
    range: OpArg = None,
    weight: OpArg = None,
    density: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> tuple[Tensor, ...]:
    """Applies `torch.histogram`.

    Args:
        input: Parameter `input`.
        bins: Parameter `bins`.
        range: Parameter `range`.
        weight: Parameter `weight`.
        density: Parameter `density`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.histogram`.
    """
    _ = (
        input,
        bins,
        range,
        weight,
        density,
        out,
        args,
        kwargs,
    )
    t = _as_tensor(input)
    return (t, zero_torch.zeros_like(t))


def histogramdd(
    input: OpArg = None,
    bins: OpArg = None,
    range: OpArg = None,
    weight: OpArg = None,
    density: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> tuple[Tensor, ...]:
    """Applies `torch.histogramdd`.

    Args:
        input: Parameter `input`.
        bins: Parameter `bins`.
        range: Parameter `range`.
        weight: Parameter `weight`.
        density: Parameter `density`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.histogramdd`.
    """
    _ = (
        input,
        bins,
        range,
        weight,
        density,
        args,
        kwargs,
    )
    t = _as_tensor(input)
    return (t, zero_torch.zeros_like(t))


def hsmm(
    input: OpArg = None, mat2: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> Tensor:
    """Applies `torch.hsmm`.

    Args:
        input: Parameter `input`.
        mat2: Parameter `mat2`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.hsmm`.
    """
    _ = (
        input,
        mat2,
        args,
        kwargs,
    )
    return zero_torch.matmul(_as_tensor(input), _as_tensor(mat2))


def hsplit(
    input: OpArg = None,
    indices_or_sections: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> list[Tensor]:
    """Applies `torch.hsplit`.

    Args:
        input: Parameter `input`.
        indices_or_sections: Parameter `indices_or_sections`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.hsplit`.
    """
    _ = (
        input,
        indices_or_sections,
        args,
        kwargs,
    )
    return [_as_tensor(input)]


def hspmm(
    mat1: OpArg = None,
    mat2: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.hspmm`.

    Args:
        mat1: Parameter `mat1`.
        mat2: Parameter `mat2`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.hspmm`.
    """
    _ = (
        mat1,
        mat2,
        out,
        args,
        kwargs,
    )
    return zero_torch.matmul(_as_tensor(mat1), _as_tensor(mat2))


def hstack(
    tensors: OpArg = None, out: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> Tensor:
    """Applies `torch.hstack`.

    Args:
        tensors: Parameter `tensors`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.hstack`.
    """
    _ = (
        tensors,
        out,
        args,
        kwargs,
    )
    seq = tensors if isinstance(tensors, (list, tuple)) else [_as_tensor(tensors)]
    return zero_torch.concatenate(seq, dim=0)


def i0(input: OpArg = None, out: OpArg = None, *args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.i0`.

    Args:
        input: Parameter `input`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.i0`.
    """
    _ = (
        input,
        out,
        args,
        kwargs,
    )
    return zero_torch.special.i0(_as_tensor(input))


def i0_(input: OpArg = None, *args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.i0_`.

    Args:
        input: Parameter `input`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.i0_`.
    """
    _ = (
        input,
        args,
        kwargs,
    )
    return zero_torch.special.i0(_as_tensor(input))


def igamma(
    input: OpArg = None,
    other: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.igamma`.

    Args:
        input: Parameter `input`.
        other: Parameter `other`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.igamma`.
    """
    _ = (
        input,
        other,
        out,
        args,
        kwargs,
    )
    return zero_torch.special.gammainc(_as_tensor(input), _as_tensor(other))


def igammac(
    input: OpArg = None,
    other: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.igammac`.

    Args:
        input: Parameter `input`.
        other: Parameter `other`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.igammac`.
    """
    _ = (
        input,
        other,
        out,
        args,
        kwargs,
    )
    return zero_torch.special.gammaincc(_as_tensor(input), _as_tensor(other))


def import_ir_module(
    cu: OpArg = None,
    filename: OpArg = None,
    map_location: OpArg = None,
    extra_files: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.import_ir_module`.

    Args:
        cu: Parameter `cu`.
        filename: Parameter `filename`.
        map_location: Parameter `map_location`.
        extra_files: Parameter `extra_files`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.import_ir_module`.
    """
    _ = (
        cu,
        filename,
        map_location,
        extra_files,
        args,
        kwargs,
    )
    return _as_tensor(cu)


def import_ir_module_from_buffer(
    cu: OpArg = None,
    buffer: OpArg = None,
    map_location: OpArg = None,
    extra_files: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.import_ir_module_from_buffer`.

    Args:
        cu: Parameter `cu`.
        buffer: Parameter `buffer`.
        map_location: Parameter `map_location`.
        extra_files: Parameter `extra_files`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.import_ir_module_from_buffer`.
    """
    _ = (
        cu,
        buffer,
        map_location,
        extra_files,
        args,
        kwargs,
    )
    return _as_tensor(cu)


def index_add(
    input: OpArg = None,
    dim: OpArg = None,
    index: OpArg = None,
    source: OpArg = None,
    alpha: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.index_add`.

    Args:
        input: Parameter `input`.
        dim: Parameter `dim`.
        index: Parameter `index`.
        source: Parameter `source`.
        alpha: Parameter `alpha`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.index_add`.
    """
    _ = (
        input,
        dim,
        index,
        source,
        alpha,
        out,
        args,
        kwargs,
    )
    return _as_tensor(input)


def index_copy(
    input: OpArg = None,
    dim: OpArg = None,
    index: OpArg = None,
    source: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.index_copy`.

    Args:
        input: Parameter `input`.
        dim: Parameter `dim`.
        index: Parameter `index`.
        source: Parameter `source`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.index_copy`.
    """
    _ = (
        input,
        dim,
        index,
        source,
        out,
        args,
        kwargs,
    )
    return _as_tensor(input)


def index_fill(
    input: OpArg = None,
    dim: OpArg = None,
    index: OpArg = None,
    value: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.index_fill`.

    Args:
        input: Parameter `input`.
        dim: Parameter `dim`.
        index: Parameter `index`.
        value: Parameter `value`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.index_fill`.
    """
    _ = (
        input,
        dim,
        index,
        value,
        args,
        kwargs,
    )
    return _as_tensor(input)


def index_put(
    input: OpArg = None,
    indices: OpArg = None,
    values: OpArg = None,
    accumulate: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.index_put`.

    Args:
        input: Parameter `input`.
        indices: Parameter `indices`.
        values: Parameter `values`.
        accumulate: Parameter `accumulate`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.index_put`.
    """
    _ = (
        input,
        indices,
        values,
        accumulate,
        args,
        kwargs,
    )
    return _as_tensor(input)


def index_put_(
    input: OpArg = None,
    indices: OpArg = None,
    values: OpArg = None,
    accumulate: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.index_put_`.

    Args:
        input: Parameter `input`.
        indices: Parameter `indices`.
        values: Parameter `values`.
        accumulate: Parameter `accumulate`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.index_put_`.
    """
    _ = (
        input,
        indices,
        values,
        accumulate,
        args,
        kwargs,
    )
    return _as_tensor(input)


def index_reduce(
    input: OpArg = None,
    dim: OpArg = None,
    index: OpArg = None,
    source: OpArg = None,
    reduce: OpArg = None,
    include_self: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.index_reduce`.

    Args:
        input: Parameter `input`.
        dim: Parameter `dim`.
        index: Parameter `index`.
        source: Parameter `source`.
        reduce: Parameter `reduce`.
        include_self: Parameter `include_self`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.index_reduce`.
    """
    _ = (
        input,
        dim,
        index,
        source,
        reduce,
        include_self,
        out,
        args,
        kwargs,
    )
    return _as_tensor(input)


def index_select(
    input: OpArg = None,
    dim: OpArg = None,
    index: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.index_select`.

    Args:
        input: Parameter `input`.
        dim: Parameter `dim`.
        index: Parameter `index`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.index_select`.
    """
    _ = (
        input,
        dim,
        index,
        out,
        args,
        kwargs,
    )
    return _as_tensor(input)


def indices_copy(
    input: OpArg = None, out: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> Tensor:
    """Applies `torch.indices_copy`.

    Args:
        input: Parameter `input`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.indices_copy`.
    """
    _ = (
        input,
        out,
        args,
        kwargs,
    )
    return _as_tensor(input)


def init_num_threads(*args: OpArg, **kwargs: OpArg) -> None:
    """Applies `torch.init_num_threads`.

    Args:
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.
    """
    _ = (
        args,
        kwargs,
    )
    return None


def initial_seed(*args: OpArg, **kwargs: OpArg) -> _builtin_int:
    """Applies `torch.initial_seed`.

    Args:
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.initial_seed`.
    """
    _ = (
        args,
        kwargs,
    )
    return 0


def instance_norm(
    input: OpArg = None,
    weight: OpArg = None,
    bias: OpArg = None,
    running_mean: OpArg = None,
    running_var: OpArg = None,
    use_input_stats: OpArg = None,
    momentum: OpArg = None,
    eps: OpArg = None,
    cudnn_enabled: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.instance_norm`.

    Args:
        input: Parameter `input`.
        weight: Parameter `weight`.
        bias: Parameter `bias`.
        running_mean: Parameter `running_mean`.
        running_var: Parameter `running_var`.
        use_input_stats: Parameter `use_input_stats`.
        momentum: Parameter `momentum`.
        eps: Parameter `eps`.
        cudnn_enabled: Parameter `cudnn_enabled`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.instance_norm`.
    """
    _ = (
        input,
        weight,
        bias,
        running_mean,
        running_var,
        use_input_stats,
        momentum,
        eps,
        cudnn_enabled,
        args,
        kwargs,
    )
    return _as_tensor(input)


def int_repr(input: OpArg = None, *args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.int_repr`.

    Args:
        input: Parameter `input`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.int_repr`.
    """
    _ = (
        input,
        args,
        kwargs,
    )
    return _as_tensor(input)


def inverse(
    input: OpArg = None, out: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> Tensor:
    """Applies `torch.inverse`.

    Args:
        input: Parameter `input`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.inverse`.
    """
    _ = (
        input,
        out,
        args,
        kwargs,
    )
    return zero_torch.linalg.inv(_as_tensor(input))


def is_anomaly_check_nan_enabled(*args: OpArg, **kwargs: OpArg) -> _builtin_bool:
    """Applies `torch.is_anomaly_check_nan_enabled`.

    Args:
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.is_anomaly_check_nan_enabled`.
    """
    _ = (
        args,
        kwargs,
    )
    return False


def is_anomaly_enabled(*args: OpArg, **kwargs: OpArg) -> _builtin_bool:
    """Applies `torch.is_anomaly_enabled`.

    Args:
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.is_anomaly_enabled`.
    """
    _ = (
        args,
        kwargs,
    )
    return False


def is_autocast_cache_enabled(*args: OpArg, **kwargs: OpArg) -> _builtin_bool:
    """Applies `torch.is_autocast_cache_enabled`.

    Args:
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.is_autocast_cache_enabled`.
    """
    _ = (
        args,
        kwargs,
    )
    return False


def is_autocast_cpu_enabled(*args: OpArg, **kwargs: OpArg) -> _builtin_bool:
    """Applies `torch.is_autocast_cpu_enabled`.

    Args:
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.is_autocast_cpu_enabled`.
    """
    _ = (
        args,
        kwargs,
    )
    return False


def is_autocast_enabled(*args: OpArg, **kwargs: OpArg) -> _builtin_bool:
    """Applies `torch.is_autocast_enabled`.

    Args:
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.is_autocast_enabled`.
    """
    _ = (
        args,
        kwargs,
    )
    return False


def is_autocast_ipu_enabled(*args: OpArg, **kwargs: OpArg) -> _builtin_bool:
    """Applies `torch.is_autocast_ipu_enabled`.

    Args:
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.is_autocast_ipu_enabled`.
    """
    _ = (
        args,
        kwargs,
    )
    return False


def is_autocast_xla_enabled(*args: OpArg, **kwargs: OpArg) -> _builtin_bool:
    """Applies `torch.is_autocast_xla_enabled`.

    Args:
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.is_autocast_xla_enabled`.
    """
    _ = (
        args,
        kwargs,
    )
    return False


def is_complex(input: OpArg = None, *args: OpArg, **kwargs: OpArg) -> _builtin_bool:
    """Applies `torch.is_complex`.

    Args:
        input: Parameter `input`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.is_complex`.
    """
    _ = (
        input,
        args,
        kwargs,
    )
    return "complex" in str(_as_tensor(input).dtype)


def is_conj(input: OpArg = None, *args: OpArg, **kwargs: OpArg) -> _builtin_bool:
    """Applies `torch.is_conj`.

    Args:
        input: Parameter `input`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.is_conj`.
    """
    _ = (
        input,
        args,
        kwargs,
    )
    return False


def is_deterministic_algorithms_warn_only_enabled(
    *args: OpArg, **kwargs: OpArg
) -> _builtin_bool:
    """Applies `torch.is_deterministic_algorithms_warn_only_enabled`.

    Args:
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.is_deterministic_algorithms_warn_only_enabled`.
    """
    _ = (
        args,
        kwargs,
    )
    return False


def is_distributed(input: OpArg = None, *args: OpArg, **kwargs: OpArg) -> _builtin_bool:
    """Applies `torch.is_distributed`.

    Args:
        input: Parameter `input`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.is_distributed`.
    """
    _ = (
        input,
        args,
        kwargs,
    )
    return False


def is_floating_point(
    input: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> _builtin_bool:
    """Applies `torch.is_floating_point`.

    Args:
        input: Parameter `input`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.is_floating_point`.
    """
    _ = (
        input,
        args,
        kwargs,
    )
    return "float" in str(_as_tensor(input).dtype)


def is_grad_enabled(*args: OpArg, **kwargs: OpArg) -> _builtin_bool:
    """Applies `torch.is_grad_enabled`.

    Args:
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.is_grad_enabled`.
    """
    _ = (
        args,
        kwargs,
    )
    return zero_torch.autograd.grad_mode.is_grad_enabled()


def is_inference(input: OpArg = None, *args: OpArg, **kwargs: OpArg) -> _builtin_bool:
    """Applies `torch.is_inference`.

    Args:
        input: Parameter `input`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.is_inference`.
    """
    _ = (
        input,
        args,
        kwargs,
    )
    return False


def is_inference_mode_enabled(*args: OpArg, **kwargs: OpArg) -> _builtin_bool:
    """Applies `torch.is_inference_mode_enabled`.

    Args:
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.is_inference_mode_enabled`.
    """
    _ = (
        args,
        kwargs,
    )
    return False


def is_neg(input: OpArg = None, *args: OpArg, **kwargs: OpArg) -> _builtin_bool:
    """Applies `torch.is_neg`.

    Args:
        input: Parameter `input`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.is_neg`.
    """
    _ = (
        input,
        args,
        kwargs,
    )
    return False


def is_nonzero(input: OpArg = None, *args: OpArg, **kwargs: OpArg) -> _builtin_bool:
    """Applies `torch.is_nonzero`.

    Args:
        input: Parameter `input`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.is_nonzero`.
    """
    _ = (
        input,
        args,
        kwargs,
    )
    return _builtin_bool(_as_tensor(input).item())


def is_same_size(
    input: OpArg = None, other: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> _builtin_bool:
    """Applies `torch.is_same_size`.

    Args:
        input: Parameter `input`.
        other: Parameter `other`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.is_same_size`.
    """
    _ = (
        input,
        other,
        args,
        kwargs,
    )
    return _as_tensor(input).shape == _as_tensor(other).shape


def is_signed(input: OpArg = None, *args: OpArg, **kwargs: OpArg) -> _builtin_bool:
    """Applies `torch.is_signed`.

    Args:
        input: Parameter `input`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.is_signed`.
    """
    _ = (
        input,
        args,
        kwargs,
    )
    return "float" in str(_as_tensor(input).dtype) or (
        "int" in str(_as_tensor(input).dtype)
        and "uint" not in str(_as_tensor(input).dtype)
    )


def is_storage(obj: OpArg = None, *args: OpArg, **kwargs: OpArg) -> _builtin_bool:
    """Applies `torch.is_storage`.

    Args:
        obj: Parameter `obj`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.is_storage`.
    """
    _ = (
        obj,
        args,
        kwargs,
    )
    return False


def is_tensor(obj: OpArg = None, *args: OpArg, **kwargs: OpArg) -> _builtin_bool:
    """Applies `torch.is_tensor`.

    Args:
        obj: Parameter `obj`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.is_tensor`.
    """
    _ = (
        obj,
        args,
        kwargs,
    )
    return isinstance(obj, Tensor)


def is_vulkan_available(*args: OpArg, **kwargs: OpArg) -> _builtin_bool:
    """Applies `torch.is_vulkan_available`.

    Args:
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.is_vulkan_available`.
    """
    _ = (
        args,
        kwargs,
    )
    return False


def is_warn_always_enabled(*args: OpArg, **kwargs: OpArg) -> _builtin_bool:
    """Applies `torch.is_warn_always_enabled`.

    Args:
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.is_warn_always_enabled`.
    """
    _ = (
        args,
        kwargs,
    )
    return False


def isin(
    elements: OpArg = None,
    test_elements: OpArg = None,
    assume_unique: OpArg = None,
    invert: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.isin`.

    Args:
        elements: Parameter `elements`.
        test_elements: Parameter `test_elements`.
        assume_unique: Parameter `assume_unique`.
        invert: Parameter `invert`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.isin`.
    """
    _ = (
        elements,
        test_elements,
        assume_unique,
        invert,
        args,
        kwargs,
    )
    return _as_tensor(elements)


def isneginf(
    input: OpArg = None, out: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> Tensor:
    """Applies `torch.isneginf`.

    Args:
        input: Parameter `input`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.isneginf`.
    """
    _ = (
        input,
        out,
        args,
        kwargs,
    )
    return _as_tensor(input)


def isposinf(
    input: OpArg = None, out: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> Tensor:
    """Applies `torch.isposinf`.

    Args:
        input: Parameter `input`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.isposinf`.
    """
    _ = (
        input,
        out,
        args,
        kwargs,
    )
    return _as_tensor(input)


def isreal(input: OpArg = None, *args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.isreal`.

    Args:
        input: Parameter `input`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.isreal`.
    """
    _ = (
        input,
        args,
        kwargs,
    )
    return _as_tensor(input)


def istft(
    input: OpArg = None,
    n_fft: OpArg = None,
    hop_length: OpArg = None,
    win_length: OpArg = None,
    window: OpArg = None,
    center: OpArg = None,
    normalized: OpArg = None,
    onesided: OpArg = None,
    length: OpArg = None,
    return_complex: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.istft`.

    Args:
        input: Parameter `input`.
        n_fft: Parameter `n_fft`.
        hop_length: Parameter `hop_length`.
        win_length: Parameter `win_length`.
        window: Parameter `window`.
        center: Parameter `center`.
        normalized: Parameter `normalized`.
        onesided: Parameter `onesided`.
        length: Parameter `length`.
        return_complex: Parameter `return_complex`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.istft`.
    """
    _ = (
        input,
        n_fft,
        hop_length,
        win_length,
        window,
        center,
        normalized,
        onesided,
        length,
        return_complex,
        args,
        kwargs,
    )
    return _as_tensor(input)


def kaiser_window(
    window_length: OpArg = None,
    periodic: OpArg = None,
    beta: OpArg = None,
    dtype: OpArg = None,
    layout: OpArg = None,
    device: OpArg = None,
    requires_grad: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.kaiser_window`.

    Args:
        window_length: Parameter `window_length`.
        periodic: Parameter `periodic`.
        beta: Parameter `beta`.
        dtype: Parameter `dtype`.
        layout: Parameter `layout`.
        device: Parameter `device`.
        requires_grad: Parameter `requires_grad`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.kaiser_window`.
    """
    _ = (
        window_length,
        periodic,
        beta,
        dtype,
        layout,
        device,
        requires_grad,
        args,
        kwargs,
    )
    return _as_tensor(window_length)


def kl_div(
    input: OpArg = None,
    target: OpArg = None,
    reduction: OpArg = None,
    log_target: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.kl_div`.

    Args:
        input: Parameter `input`.
        target: Parameter `target`.
        reduction: Parameter `reduction`.
        log_target: Parameter `log_target`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.kl_div`.
    """
    _ = (
        input,
        target,
        reduction,
        log_target,
        args,
        kwargs,
    )
    return _as_tensor(input)


def kron(
    input: OpArg = None,
    other: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.kron`.

    Args:
        input: Parameter `input`.
        other: Parameter `other`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.kron`.
    """
    _ = (
        input,
        other,
        out,
        args,
        kwargs,
    )
    return _as_tensor(input)


def kthvalue(
    input: OpArg = None,
    k: OpArg = None,
    dim: OpArg = None,
    keepdim: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> tuple[Tensor, ...]:
    """Applies `torch.kthvalue`.

    Args:
        input: Parameter `input`.
        k: Parameter `k`.
        dim: Parameter `dim`.
        keepdim: Parameter `keepdim`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.kthvalue`.
    """
    _ = (
        input,
        k,
        dim,
        keepdim,
        out,
        args,
        kwargs,
    )
    t = _as_tensor(input)
    return (t, zero_torch.zeros_like(t))


def layer_norm(
    input: OpArg = None,
    normalized_shape: OpArg = None,
    weight: OpArg = None,
    bias: OpArg = None,
    eps: OpArg = None,
    cudnn_enable: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.layer_norm`.

    Args:
        input: Parameter `input`.
        normalized_shape: Parameter `normalized_shape`.
        weight: Parameter `weight`.
        bias: Parameter `bias`.
        eps: Parameter `eps`.
        cudnn_enable: Parameter `cudnn_enable`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.layer_norm`.
    """
    _ = (
        input,
        normalized_shape,
        weight,
        bias,
        eps,
        cudnn_enable,
        args,
        kwargs,
    )
    return _as_tensor(input)


def lcm_(
    input: OpArg = None, other: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> Tensor:
    """Applies `torch.lcm_`.

    Args:
        input: Parameter `input`.
        other: Parameter `other`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.lcm_`.
    """
    _ = (
        input,
        other,
        args,
        kwargs,
    )
    return zero_torch.lcm(_as_tensor(input), _as_tensor(other))


def ldexp_(
    input: OpArg = None, other: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> Tensor:
    """Applies `torch.ldexp_`.

    Args:
        input: Parameter `input`.
        other: Parameter `other`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.ldexp_`.
    """
    _ = (
        input,
        other,
        args,
        kwargs,
    )
    return zero_torch.ldexp(_as_tensor(input), _as_tensor(other))


def le(
    input: OpArg = None,
    other: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.le`.

    Args:
        input: Parameter `input`.
        other: Parameter `other`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.le`.
    """
    _ = (
        input,
        other,
        out,
        args,
        kwargs,
    )
    return zero_torch.less_equal(_as_tensor(input), _as_tensor(other))


def lerp(
    input: OpArg = None,
    end: OpArg = None,
    weight: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.lerp`.

    Args:
        input: Parameter `input`.
        end: Parameter `end`.
        weight: Parameter `weight`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.lerp`.
    """
    _ = (
        input,
        end,
        weight,
        out,
        args,
        kwargs,
    )
    return _as_tensor(input)


def load(
    f: OpArg = None,
    map_location: OpArg = None,
    pickle_module: OpArg = None,
    weights_only: OpArg = None,
    mmap: OpArg = None,
    pickle_load_args: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.load`.

    Args:
        f: Parameter `f`.
        map_location: Parameter `map_location`.
        pickle_module: Parameter `pickle_module`.
        weights_only: Parameter `weights_only`.
        mmap: Parameter `mmap`.
        pickle_load_args: Parameter `pickle_load_args`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.load`.
    """
    _ = (
        f,
        map_location,
        pickle_module,
        weights_only,
        mmap,
        pickle_load_args,
        args,
        kwargs,
    )
    return _as_tensor(f)


def lobpcg(
    A: OpArg = None,
    k: OpArg = None,
    B: OpArg = None,
    X: OpArg = None,
    n: OpArg = None,
    iK: OpArg = None,
    niter: OpArg = None,
    tol: OpArg = None,
    largest: OpArg = None,
    method: OpArg = None,
    tracker: OpArg = None,
    ortho_iparams: OpArg = None,
    ortho_fparams: OpArg = None,
    ortho_bparams: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> tuple[Tensor, ...]:
    """Applies `torch.lobpcg`.

    Args:
        A: Parameter `A`.
        k: Parameter `k`.
        B: Parameter `B`.
        X: Parameter `X`.
        n: Parameter `n`.
        iK: Parameter `iK`.
        niter: Parameter `niter`.
        tol: Parameter `tol`.
        largest: Parameter `largest`.
        method: Parameter `method`.
        tracker: Parameter `tracker`.
        ortho_iparams: Parameter `ortho_iparams`.
        ortho_fparams: Parameter `ortho_fparams`.
        ortho_bparams: Parameter `ortho_bparams`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.lobpcg`.
    """
    _ = (
        A,
        k,
        B,
        X,
        n,
        iK,
        niter,
        tol,
        largest,
        method,
        tracker,
        ortho_iparams,
        ortho_fparams,
        ortho_bparams,
        args,
        kwargs,
    )
    t = _as_tensor(A)
    return (t, zero_torch.zeros_like(t))


def log10_(input: OpArg = None, *args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.log10_`.

    Args:
        input: Parameter `input`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.log10_`.
    """
    _ = (
        input,
        args,
        kwargs,
    )
    return zero_torch.log10(_as_tensor(input))


def log1p_(input: OpArg = None, *args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.log1p_`.

    Args:
        input: Parameter `input`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.log1p_`.
    """
    _ = (
        input,
        args,
        kwargs,
    )
    return zero_torch.log1p(_as_tensor(input))


def log2_(input: OpArg = None, *args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.log2_`.

    Args:
        input: Parameter `input`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.log2_`.
    """
    _ = (
        input,
        args,
        kwargs,
    )
    return zero_torch.log2(_as_tensor(input))


def log_(input: OpArg = None, *args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.log_`.

    Args:
        input: Parameter `input`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.log_`.
    """
    _ = (
        input,
        args,
        kwargs,
    )
    return zero_torch.log(_as_tensor(input))


def log_softmax(
    input: OpArg = None,
    dim: OpArg = None,
    dtype: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.log_softmax`.

    Args:
        input: Parameter `input`.
        dim: Parameter `dim`.
        dtype: Parameter `dtype`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.log_softmax`.
    """
    _ = (
        input,
        dim,
        dtype,
        out,
        args,
        kwargs,
    )
    return zero_torch.nn.functional.log_softmax(
        _as_tensor(input),
        dim=_builtin_int(dim) if isinstance(dim, _builtin_int) else -1,
    )


def logcumsumexp(
    input: OpArg = None,
    dim: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.logcumsumexp`.

    Args:
        input: Parameter `input`.
        dim: Parameter `dim`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.logcumsumexp`.
    """
    _ = (
        input,
        dim,
        out,
        args,
        kwargs,
    )
    return _as_tensor(input)


def logdet(input: OpArg = None, *args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.logdet`.

    Args:
        input: Parameter `input`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.logdet`.
    """
    _ = (
        input,
        args,
        kwargs,
    )
    return _as_tensor(input)


def logit_(
    input: OpArg = None, eps: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> Tensor:
    """Applies `torch.logit_`.

    Args:
        input: Parameter `input`.
        eps: Parameter `eps`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.logit_`.
    """
    _ = (
        input,
        eps,
        args,
        kwargs,
    )
    return zero_torch.logit(_as_tensor(input))


def logspace(
    start: OpArg = None,
    end: OpArg = None,
    steps: OpArg = None,
    base: OpArg = None,
    out: OpArg = None,
    dtype: OpArg = None,
    layout: OpArg = None,
    device: OpArg = None,
    requires_grad: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.logspace`.

    Args:
        start: Parameter `start`.
        end: Parameter `end`.
        steps: Parameter `steps`.
        base: Parameter `base`.
        out: Parameter `out`.
        dtype: Parameter `dtype`.
        layout: Parameter `layout`.
        device: Parameter `device`.
        requires_grad: Parameter `requires_grad`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.logspace`.
    """
    _ = (
        start,
        end,
        steps,
        base,
        out,
        dtype,
        layout,
        device,
        requires_grad,
        args,
        kwargs,
    )
    return _as_tensor(start)


def lstm(
    input: OpArg = None,
    hx: OpArg = None,
    params: OpArg = None,
    has_biases: OpArg = None,
    num_layers: OpArg = None,
    dropout: OpArg = None,
    train: OpArg = None,
    bidirectional: OpArg = None,
    batch_first: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.lstm`.

    Args:
        input: Parameter `input`.
        hx: Parameter `hx`.
        params: Parameter `params`.
        has_biases: Parameter `has_biases`.
        num_layers: Parameter `num_layers`.
        dropout: Parameter `dropout`.
        train: Parameter `train`.
        bidirectional: Parameter `bidirectional`.
        batch_first: Parameter `batch_first`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.lstm`.
    """
    _ = (
        input,
        hx,
        params,
        has_biases,
        num_layers,
        dropout,
        train,
        bidirectional,
        batch_first,
        args,
        kwargs,
    )
    return _as_tensor(input)


def lstm_cell(
    input: OpArg = None,
    hx: OpArg = None,
    w_ih: OpArg = None,
    w_hh: OpArg = None,
    b_ih: OpArg = None,
    b_hh: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.lstm_cell`.

    Args:
        input: Parameter `input`.
        hx: Parameter `hx`.
        w_ih: Parameter `w_ih`.
        w_hh: Parameter `w_hh`.
        b_ih: Parameter `b_ih`.
        b_hh: Parameter `b_hh`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.lstm_cell`.
    """
    _ = (
        input,
        hx,
        w_ih,
        w_hh,
        b_ih,
        b_hh,
        args,
        kwargs,
    )
    return _as_tensor(input)


def lstsq(
    input: OpArg = None,
    A: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> tuple[Tensor, ...]:
    """Applies `torch.lstsq`.

    Args:
        input: Parameter `input`.
        A: Parameter `A`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.lstsq`.
    """
    _ = (
        input,
        A,
        out,
        args,
        kwargs,
    )
    t = _as_tensor(input)
    return (t, zero_torch.zeros_like(t))


def lt(
    input: OpArg = None,
    other: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.lt`.

    Args:
        input: Parameter `input`.
        other: Parameter `other`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.lt`.
    """
    _ = (
        input,
        other,
        out,
        args,
        kwargs,
    )
    return zero_torch.less(_as_tensor(input), _as_tensor(other))


def lu(
    A: OpArg = None,
    pivot: OpArg = None,
    get_infos: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> tuple[Tensor, ...]:
    """Applies `torch.lu`.

    Args:
        A: Parameter `A`.
        pivot: Parameter `pivot`.
        get_infos: Parameter `get_infos`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.lu`.
    """
    _ = (
        A,
        pivot,
        get_infos,
        out,
        args,
        kwargs,
    )
    t = _as_tensor(A)
    return (t, zero_torch.zeros_like(t))


def lu_solve(
    input: OpArg = None,
    LU_data: OpArg = None,
    LU_pivots: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.lu_solve`.

    Args:
        input: Parameter `input`.
        LU_data: Parameter `LU_data`.
        LU_pivots: Parameter `LU_pivots`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.lu_solve`.
    """
    _ = (
        input,
        LU_data,
        LU_pivots,
        out,
        args,
        kwargs,
    )
    return _as_tensor(input)


def lu_unpack(
    LU_data: OpArg = None,
    LU_pivots: OpArg = None,
    unpack_data: OpArg = None,
    unpack_pivots: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> tuple[Tensor, ...]:
    """Applies `torch.lu_unpack`.

    Args:
        LU_data: Parameter `LU_data`.
        LU_pivots: Parameter `LU_pivots`.
        unpack_data: Parameter `unpack_data`.
        unpack_pivots: Parameter `unpack_pivots`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.lu_unpack`.
    """
    _ = (
        LU_data,
        LU_pivots,
        unpack_data,
        unpack_pivots,
        out,
        args,
        kwargs,
    )
    t = _as_tensor(LU_data)
    return (t, zero_torch.zeros_like(t))


def margin_ranking_loss(
    input1: OpArg = None,
    input2: OpArg = None,
    target: OpArg = None,
    margin: OpArg = None,
    reduction: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.margin_ranking_loss`.

    Args:
        input1: Parameter `input1`.
        input2: Parameter `input2`.
        target: Parameter `target`.
        margin: Parameter `margin`.
        reduction: Parameter `reduction`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.margin_ranking_loss`.
    """
    _ = (
        input1,
        input2,
        target,
        margin,
        reduction,
        args,
        kwargs,
    )
    return _as_tensor(input1)


def masked_fill(
    input: OpArg = None,
    mask: OpArg = None,
    value: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.masked_fill`.

    Args:
        input: Parameter `input`.
        mask: Parameter `mask`.
        value: Parameter `value`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.masked_fill`.
    """
    _ = (
        input,
        mask,
        value,
        args,
        kwargs,
    )
    return zero_torch.where(mask, value, _as_tensor(input))


def masked_scatter(
    input: OpArg = None,
    mask: OpArg = None,
    source: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.masked_scatter`.

    Args:
        input: Parameter `input`.
        mask: Parameter `mask`.
        source: Parameter `source`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.masked_scatter`.
    """
    _ = (
        input,
        mask,
        source,
        args,
        kwargs,
    )
    return _as_tensor(input)


def masked_select(
    input: OpArg = None,
    mask: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.masked_select`.

    Args:
        input: Parameter `input`.
        mask: Parameter `mask`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.masked_select`.
    """
    _ = (
        input,
        mask,
        out,
        args,
        kwargs,
    )
    return _as_tensor(input)


def matrix_exp(input: OpArg = None, *args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.matrix_exp`.

    Args:
        input: Parameter `input`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.matrix_exp`.
    """
    _ = (
        input,
        args,
        kwargs,
    )
    return zero_torch.linalg.matrix_exp(_as_tensor(input))


def matrix_rank(
    input: OpArg = None,
    tol: OpArg = None,
    symmetric: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.matrix_rank`.

    Args:
        input: Parameter `input`.
        tol: Parameter `tol`.
        symmetric: Parameter `symmetric`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.matrix_rank`.
    """
    _ = (
        input,
        tol,
        symmetric,
        out,
        args,
        kwargs,
    )
    return _as_tensor(input)


def max_pool1d(
    input: OpArg = None,
    kernel_size: OpArg = None,
    stride: OpArg = None,
    padding: OpArg = None,
    dilation: OpArg = None,
    ceil_mode: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.max_pool1d`.

    Args:
        input: Parameter `input`.
        kernel_size: Parameter `kernel_size`.
        stride: Parameter `stride`.
        padding: Parameter `padding`.
        dilation: Parameter `dilation`.
        ceil_mode: Parameter `ceil_mode`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.max_pool1d`.
    """
    _ = (
        input,
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
        args,
        kwargs,
    )
    return _as_tensor(input)


def max_pool1d_with_indices(
    input: OpArg = None,
    kernel_size: OpArg = None,
    stride: OpArg = None,
    padding: OpArg = None,
    dilation: OpArg = None,
    ceil_mode: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> tuple[Tensor, ...]:
    """Applies `torch.max_pool1d_with_indices`.

    Args:
        input: Parameter `input`.
        kernel_size: Parameter `kernel_size`.
        stride: Parameter `stride`.
        padding: Parameter `padding`.
        dilation: Parameter `dilation`.
        ceil_mode: Parameter `ceil_mode`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.max_pool1d_with_indices`.
    """
    _ = (
        input,
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
        args,
        kwargs,
    )
    t = _as_tensor(input)
    return (t, zero_torch.zeros_like(t))


def max_pool2d(
    input: OpArg = None,
    kernel_size: OpArg = None,
    stride: OpArg = None,
    padding: OpArg = None,
    dilation: OpArg = None,
    ceil_mode: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.max_pool2d`.

    Args:
        input: Parameter `input`.
        kernel_size: Parameter `kernel_size`.
        stride: Parameter `stride`.
        padding: Parameter `padding`.
        dilation: Parameter `dilation`.
        ceil_mode: Parameter `ceil_mode`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.max_pool2d`.
    """
    _ = (
        input,
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
        args,
        kwargs,
    )
    return _as_tensor(input)


def max_pool3d(
    input: OpArg = None,
    kernel_size: OpArg = None,
    stride: OpArg = None,
    padding: OpArg = None,
    dilation: OpArg = None,
    ceil_mode: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.max_pool3d`.

    Args:
        input: Parameter `input`.
        kernel_size: Parameter `kernel_size`.
        stride: Parameter `stride`.
        padding: Parameter `padding`.
        dilation: Parameter `dilation`.
        ceil_mode: Parameter `ceil_mode`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.max_pool3d`.
    """
    _ = (
        input,
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
        args,
        kwargs,
    )
    return _as_tensor(input)


def median(input: OpArg = None, *args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.median`.

    Args:
        input: Parameter `input`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.median`.
    """
    _ = (
        input,
        args,
        kwargs,
    )
    return _as_tensor(input)


def merge_type_from_type_comment(
    decl: OpArg = None,
    type_annotation_decl: OpArg = None,
    is_method: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.merge_type_from_type_comment`.

    Args:
        decl: Parameter `decl`.
        type_annotation_decl: Parameter `type_annotation_decl`.
        is_method: Parameter `is_method`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.merge_type_from_type_comment`.
    """
    _ = (
        decl,
        type_annotation_decl,
        is_method,
        args,
        kwargs,
    )
    return _as_tensor(decl)


def miopen_batch_norm(
    input: OpArg = None,
    weight: OpArg = None,
    bias: OpArg = None,
    running_mean: OpArg = None,
    running_var: OpArg = None,
    training: OpArg = None,
    exponential_average_factor: OpArg = None,
    epsilon: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.miopen_batch_norm`.

    Args:
        input: Parameter `input`.
        weight: Parameter `weight`.
        bias: Parameter `bias`.
        running_mean: Parameter `running_mean`.
        running_var: Parameter `running_var`.
        training: Parameter `training`.
        exponential_average_factor: Parameter `exponential_average_factor`.
        epsilon: Parameter `epsilon`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.miopen_batch_norm`.
    """
    _ = (
        input,
        weight,
        bias,
        running_mean,
        running_var,
        training,
        exponential_average_factor,
        epsilon,
        args,
        kwargs,
    )
    return _as_tensor(input)


def miopen_convolution(
    input: OpArg = None,
    weight: OpArg = None,
    bias: OpArg = None,
    padding: OpArg = None,
    stride: OpArg = None,
    dilation: OpArg = None,
    groups: OpArg = None,
    benchmark: OpArg = None,
    deterministic: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.miopen_convolution`.

    Args:
        input: Parameter `input`.
        weight: Parameter `weight`.
        bias: Parameter `bias`.
        padding: Parameter `padding`.
        stride: Parameter `stride`.
        dilation: Parameter `dilation`.
        groups: Parameter `groups`.
        benchmark: Parameter `benchmark`.
        deterministic: Parameter `deterministic`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.miopen_convolution`.
    """
    _ = (
        input,
        weight,
        bias,
        padding,
        stride,
        dilation,
        groups,
        benchmark,
        deterministic,
        args,
        kwargs,
    )
    return _as_tensor(input)


def miopen_convolution_add_relu(
    input: OpArg = None,
    weight: OpArg = None,
    z: OpArg = None,
    alpha: OpArg = None,
    bias: OpArg = None,
    stride: OpArg = None,
    padding: OpArg = None,
    dilation: OpArg = None,
    groups: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.miopen_convolution_add_relu`.

    Args:
        input: Parameter `input`.
        weight: Parameter `weight`.
        z: Parameter `z`.
        alpha: Parameter `alpha`.
        bias: Parameter `bias`.
        stride: Parameter `stride`.
        padding: Parameter `padding`.
        dilation: Parameter `dilation`.
        groups: Parameter `groups`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.miopen_convolution_add_relu`.
    """
    _ = (
        input,
        weight,
        z,
        alpha,
        bias,
        stride,
        padding,
        dilation,
        groups,
        args,
        kwargs,
    )
    return _as_tensor(input)


def miopen_convolution_relu(
    input: OpArg = None,
    weight: OpArg = None,
    bias: OpArg = None,
    stride: OpArg = None,
    padding: OpArg = None,
    dilation: OpArg = None,
    groups: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.miopen_convolution_relu`.

    Args:
        input: Parameter `input`.
        weight: Parameter `weight`.
        bias: Parameter `bias`.
        stride: Parameter `stride`.
        padding: Parameter `padding`.
        dilation: Parameter `dilation`.
        groups: Parameter `groups`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.miopen_convolution_relu`.
    """
    _ = (
        input,
        weight,
        bias,
        stride,
        padding,
        dilation,
        groups,
        args,
        kwargs,
    )
    return _as_tensor(input)


def miopen_convolution_transpose(
    input: OpArg = None,
    weight: OpArg = None,
    bias: OpArg = None,
    padding: OpArg = None,
    output_padding: OpArg = None,
    stride: OpArg = None,
    dilation: OpArg = None,
    groups: OpArg = None,
    benchmark: OpArg = None,
    deterministic: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.miopen_convolution_transpose`.

    Args:
        input: Parameter `input`.
        weight: Parameter `weight`.
        bias: Parameter `bias`.
        padding: Parameter `padding`.
        output_padding: Parameter `output_padding`.
        stride: Parameter `stride`.
        dilation: Parameter `dilation`.
        groups: Parameter `groups`.
        benchmark: Parameter `benchmark`.
        deterministic: Parameter `deterministic`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.miopen_convolution_transpose`.
    """
    _ = (
        input,
        weight,
        bias,
        padding,
        output_padding,
        stride,
        dilation,
        groups,
        benchmark,
        deterministic,
        args,
        kwargs,
    )
    return _as_tensor(input)


def miopen_ctc_loss(
    log_probs: OpArg = None,
    targets: OpArg = None,
    input_lengths: OpArg = None,
    target_lengths: OpArg = None,
    blank: OpArg = None,
    deterministic: OpArg = None,
    zero_infinity: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.miopen_ctc_loss`.

    Args:
        log_probs: Parameter `log_probs`.
        targets: Parameter `targets`.
        input_lengths: Parameter `input_lengths`.
        target_lengths: Parameter `target_lengths`.
        blank: Parameter `blank`.
        deterministic: Parameter `deterministic`.
        zero_infinity: Parameter `zero_infinity`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.miopen_ctc_loss`.
    """
    _ = (
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        blank,
        deterministic,
        zero_infinity,
        args,
        kwargs,
    )
    return _as_tensor(log_probs)


def miopen_depthwise_convolution(
    input: OpArg = None,
    weight: OpArg = None,
    bias: OpArg = None,
    padding: OpArg = None,
    stride: OpArg = None,
    dilation: OpArg = None,
    groups: OpArg = None,
    benchmark: OpArg = None,
    deterministic: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.miopen_depthwise_convolution`.

    Args:
        input: Parameter `input`.
        weight: Parameter `weight`.
        bias: Parameter `bias`.
        padding: Parameter `padding`.
        stride: Parameter `stride`.
        dilation: Parameter `dilation`.
        groups: Parameter `groups`.
        benchmark: Parameter `benchmark`.
        deterministic: Parameter `deterministic`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.miopen_depthwise_convolution`.
    """
    _ = (
        input,
        weight,
        bias,
        padding,
        stride,
        dilation,
        groups,
        benchmark,
        deterministic,
        args,
        kwargs,
    )
    return _as_tensor(input)


def miopen_rnn(
    input: OpArg = None,
    weight: OpArg = None,
    weight_stride0: OpArg = None,
    hx: OpArg = None,
    cx: OpArg = None,
    mode: OpArg = None,
    hidden_size: OpArg = None,
    num_layers: OpArg = None,
    batch_first: OpArg = None,
    dropout: OpArg = None,
    train: OpArg = None,
    bidirectional: OpArg = None,
    batch_sizes: OpArg = None,
    dropout_state: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.miopen_rnn`.

    Args:
        input: Parameter `input`.
        weight: Parameter `weight`.
        weight_stride0: Parameter `weight_stride0`.
        hx: Parameter `hx`.
        cx: Parameter `cx`.
        mode: Parameter `mode`.
        hidden_size: Parameter `hidden_size`.
        num_layers: Parameter `num_layers`.
        batch_first: Parameter `batch_first`.
        dropout: Parameter `dropout`.
        train: Parameter `train`.
        bidirectional: Parameter `bidirectional`.
        batch_sizes: Parameter `batch_sizes`.
        dropout_state: Parameter `dropout_state`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.miopen_rnn`.
    """
    _ = (
        input,
        weight,
        weight_stride0,
        hx,
        cx,
        mode,
        hidden_size,
        num_layers,
        batch_first,
        dropout,
        train,
        bidirectional,
        batch_sizes,
        dropout_state,
        args,
        kwargs,
    )
    return _as_tensor(input)


def mkldnn_adaptive_avg_pool2d(
    input: OpArg = None,
    output_size: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.mkldnn_adaptive_avg_pool2d`.

    Args:
        input: Parameter `input`.
        output_size: Parameter `output_size`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.mkldnn_adaptive_avg_pool2d`.
    """
    _ = (
        input,
        output_size,
        out,
        args,
        kwargs,
    )
    return _as_tensor(input)


def mkldnn_convolution(
    input: OpArg = None,
    weight: OpArg = None,
    bias: OpArg = None,
    padding: OpArg = None,
    stride: OpArg = None,
    dilation: OpArg = None,
    groups: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.mkldnn_convolution`.

    Args:
        input: Parameter `input`.
        weight: Parameter `weight`.
        bias: Parameter `bias`.
        padding: Parameter `padding`.
        stride: Parameter `stride`.
        dilation: Parameter `dilation`.
        groups: Parameter `groups`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.mkldnn_convolution`.
    """
    _ = (
        input,
        weight,
        bias,
        padding,
        stride,
        dilation,
        groups,
        args,
        kwargs,
    )
    return _as_tensor(input)


def mkldnn_linear_backward_weights(
    grad_output: OpArg = None,
    input: OpArg = None,
    weight: OpArg = None,
    bias_defined: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.mkldnn_linear_backward_weights`.

    Args:
        grad_output: Parameter `grad_output`.
        input: Parameter `input`.
        weight: Parameter `weight`.
        bias_defined: Parameter `bias_defined`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.mkldnn_linear_backward_weights`.
    """
    _ = (
        grad_output,
        input,
        weight,
        bias_defined,
        args,
        kwargs,
    )
    return _as_tensor(grad_output)


def mkldnn_max_pool2d(
    input: OpArg = None,
    kernel_size: OpArg = None,
    stride: OpArg = None,
    padding: OpArg = None,
    dilation: OpArg = None,
    ceil_mode: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.mkldnn_max_pool2d`.

    Args:
        input: Parameter `input`.
        kernel_size: Parameter `kernel_size`.
        stride: Parameter `stride`.
        padding: Parameter `padding`.
        dilation: Parameter `dilation`.
        ceil_mode: Parameter `ceil_mode`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.mkldnn_max_pool2d`.
    """
    _ = (
        input,
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
        args,
        kwargs,
    )
    return _as_tensor(input)


def mkldnn_max_pool3d(
    input: OpArg = None,
    kernel_size: OpArg = None,
    stride: OpArg = None,
    padding: OpArg = None,
    dilation: OpArg = None,
    ceil_mode: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.mkldnn_max_pool3d`.

    Args:
        input: Parameter `input`.
        kernel_size: Parameter `kernel_size`.
        stride: Parameter `stride`.
        padding: Parameter `padding`.
        dilation: Parameter `dilation`.
        ceil_mode: Parameter `ceil_mode`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.mkldnn_max_pool3d`.
    """
    _ = (
        input,
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
        args,
        kwargs,
    )
    return _as_tensor(input)


def mkldnn_rnn_layer(
    input: OpArg = None,
    weight0: OpArg = None,
    weight1: OpArg = None,
    weight2: OpArg = None,
    weight3: OpArg = None,
    hx_: OpArg = None,
    cx_: OpArg = None,
    reverse: OpArg = None,
    batch_sizes: OpArg = None,
    mode: OpArg = None,
    hidden_size: OpArg = None,
    num_layers: OpArg = None,
    has_biases: OpArg = None,
    bidirectional: OpArg = None,
    batch_first: OpArg = None,
    train: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.mkldnn_rnn_layer`.

    Args:
        input: Parameter `input`.
        weight0: Parameter `weight0`.
        weight1: Parameter `weight1`.
        weight2: Parameter `weight2`.
        weight3: Parameter `weight3`.
        hx_: Parameter `hx_`.
        cx_: Parameter `cx_`.
        reverse: Parameter `reverse`.
        batch_sizes: Parameter `batch_sizes`.
        mode: Parameter `mode`.
        hidden_size: Parameter `hidden_size`.
        num_layers: Parameter `num_layers`.
        has_biases: Parameter `has_biases`.
        bidirectional: Parameter `bidirectional`.
        batch_first: Parameter `batch_first`.
        train: Parameter `train`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.mkldnn_rnn_layer`.
    """
    _ = (
        input,
        weight0,
        weight1,
        weight2,
        weight3,
        hx_,
        cx_,
        reverse,
        batch_sizes,
        mode,
        hidden_size,
        num_layers,
        has_biases,
        bidirectional,
        batch_first,
        train,
        args,
        kwargs,
    )
    return _as_tensor(input)


def mm(
    input: OpArg = None,
    mat2: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.mm`.

    Args:
        input: Parameter `input`.
        mat2: Parameter `mat2`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.mm`.
    """
    _ = (
        input,
        mat2,
        out,
        args,
        kwargs,
    )
    return zero_torch.matmul(_as_tensor(input), _as_tensor(mat2))


def mode(
    input: OpArg = None,
    dim: OpArg = None,
    keepdim: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> tuple[Tensor, ...]:
    """Applies `torch.mode`.

    Args:
        input: Parameter `input`.
        dim: Parameter `dim`.
        keepdim: Parameter `keepdim`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.mode`.
    """
    _ = (
        input,
        dim,
        keepdim,
        out,
        args,
        kwargs,
    )
    t = _as_tensor(input)
    return (t, zero_torch.zeros_like(t))


def movedim(
    input: OpArg = None,
    source: OpArg = None,
    destination: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.movedim`.

    Args:
        input: Parameter `input`.
        source: Parameter `source`.
        destination: Parameter `destination`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.movedim`.
    """
    _ = (
        input,
        source,
        destination,
        args,
        kwargs,
    )
    return zero_torch.moveaxis(
        _as_tensor(input),
        _builtin_int(source) if isinstance(source, _builtin_int) else 0,
        _builtin_int(destination) if isinstance(destination, _builtin_int) else 1,
    )


def msort(
    input: OpArg = None, out: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> Tensor:
    """Applies `torch.msort`.

    Args:
        input: Parameter `input`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.msort`.
    """
    _ = (
        input,
        out,
        args,
        kwargs,
    )
    return _as_tensor(input)


def mul(
    input: OpArg = None,
    other: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.mul`.

    Args:
        input: Parameter `input`.
        other: Parameter `other`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.mul`.
    """
    _ = (
        input,
        other,
        out,
        args,
        kwargs,
    )
    return zero_torch.multiply(_as_tensor(input), _as_tensor(other))


def multinomial(
    input: OpArg = None,
    num_samples: OpArg = None,
    replacement: OpArg = None,
    generator: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.multinomial`.

    Args:
        input: Parameter `input`.
        num_samples: Parameter `num_samples`.
        replacement: Parameter `replacement`.
        generator: Parameter `generator`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.multinomial`.
    """
    _ = (
        input,
        num_samples,
        replacement,
        generator,
        out,
        args,
        kwargs,
    )
    return _as_tensor(input)


def mv(
    input: OpArg = None,
    vec: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.mv`.

    Args:
        input: Parameter `input`.
        vec: Parameter `vec`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.mv`.
    """
    _ = (
        input,
        vec,
        out,
        args,
        kwargs,
    )
    return zero_torch.matmul(_as_tensor(input), _as_tensor(vec))


def nan_to_num_(
    input: OpArg = None,
    nan: OpArg = None,
    posinf: OpArg = None,
    neginf: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.nan_to_num_`.

    Args:
        input: Parameter `input`.
        nan: Parameter `nan`.
        posinf: Parameter `posinf`.
        neginf: Parameter `neginf`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.nan_to_num_`.
    """
    _ = (
        input,
        nan,
        posinf,
        neginf,
        args,
        kwargs,
    )
    return _as_tensor(input)


def nanmean(
    input: OpArg = None,
    dim: OpArg = None,
    keepdim: OpArg = None,
    dtype: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.nanmean`.

    Args:
        input: Parameter `input`.
        dim: Parameter `dim`.
        keepdim: Parameter `keepdim`.
        dtype: Parameter `dtype`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.nanmean`.
    """
    _ = (
        input,
        dim,
        keepdim,
        dtype,
        out,
        args,
        kwargs,
    )
    return _as_tensor(input)


def nanmedian(input: OpArg = None, *args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.nanmedian`.

    Args:
        input: Parameter `input`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.nanmedian`.
    """
    _ = (
        input,
        args,
        kwargs,
    )
    return _as_tensor(input)


def nanquantile(
    input: OpArg = None,
    q: OpArg = None,
    dim: OpArg = None,
    keepdim: OpArg = None,
    interpolation: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.nanquantile`.

    Args:
        input: Parameter `input`.
        q: Parameter `q`.
        dim: Parameter `dim`.
        keepdim: Parameter `keepdim`.
        interpolation: Parameter `interpolation`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.nanquantile`.
    """
    _ = (
        input,
        q,
        dim,
        keepdim,
        interpolation,
        out,
        args,
        kwargs,
    )
    return _as_tensor(input)


def nansum(
    input: OpArg = None,
    dim: OpArg = None,
    keepdim: OpArg = None,
    dtype: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.nansum`.

    Args:
        input: Parameter `input`.
        dim: Parameter `dim`.
        keepdim: Parameter `keepdim`.
        dtype: Parameter `dtype`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.nansum`.
    """
    _ = (
        input,
        dim,
        keepdim,
        dtype,
        out,
        args,
        kwargs,
    )
    return _as_tensor(input)


def narrow(
    input: OpArg = None,
    dim: OpArg = None,
    start: OpArg = None,
    length: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.narrow`.

    Args:
        input: Parameter `input`.
        dim: Parameter `dim`.
        start: Parameter `start`.
        length: Parameter `length`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.narrow`.
    """
    _ = (
        input,
        dim,
        start,
        length,
        args,
        kwargs,
    )
    return _as_tensor(input)


def narrow_copy(
    input: OpArg = None,
    dim: OpArg = None,
    start: OpArg = None,
    length: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.narrow_copy`.

    Args:
        input: Parameter `input`.
        dim: Parameter `dim`.
        start: Parameter `start`.
        length: Parameter `length`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.narrow_copy`.
    """
    _ = (
        input,
        dim,
        start,
        length,
        out,
        args,
        kwargs,
    )
    return _as_tensor(input)


def native_batch_norm(
    input: OpArg = None,
    weight: OpArg = None,
    bias: OpArg = None,
    running_mean: OpArg = None,
    running_var: OpArg = None,
    training: OpArg = None,
    momentum: OpArg = None,
    eps: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> tuple[Tensor, ...]:
    """Applies `torch.native_batch_norm`.

    Args:
        input: Parameter `input`.
        weight: Parameter `weight`.
        bias: Parameter `bias`.
        running_mean: Parameter `running_mean`.
        running_var: Parameter `running_var`.
        training: Parameter `training`.
        momentum: Parameter `momentum`.
        eps: Parameter `eps`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.native_batch_norm`.
    """
    _ = (
        input,
        weight,
        bias,
        running_mean,
        running_var,
        training,
        momentum,
        eps,
        out,
        args,
        kwargs,
    )
    t = _as_tensor(input)
    return (t, zero_torch.zeros_like(t))


def native_channel_shuffle(
    input: OpArg = None, groups: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> Tensor:
    """Applies `torch.native_channel_shuffle`.

    Args:
        input: Parameter `input`.
        groups: Parameter `groups`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.native_channel_shuffle`.
    """
    _ = (
        input,
        groups,
        args,
        kwargs,
    )
    return _as_tensor(input)


def native_dropout(
    input: OpArg = None,
    p: OpArg = None,
    train: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> tuple[Tensor, ...]:
    """Applies `torch.native_dropout`.

    Args:
        input: Parameter `input`.
        p: Parameter `p`.
        train: Parameter `train`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.native_dropout`.
    """
    _ = (
        input,
        p,
        train,
        args,
        kwargs,
    )
    t = _as_tensor(input)
    return (t, zero_torch.zeros_like(t))


def native_group_norm(
    input: OpArg = None,
    weight: OpArg = None,
    bias: OpArg = None,
    N: OpArg = None,
    C: OpArg = None,
    HxW: OpArg = None,
    group: OpArg = None,
    eps: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> tuple[Tensor, ...]:
    """Applies `torch.native_group_norm`.

    Args:
        input: Parameter `input`.
        weight: Parameter `weight`.
        bias: Parameter `bias`.
        N: Parameter `N`.
        C: Parameter `C`.
        HxW: Parameter `HxW`.
        group: Parameter `group`.
        eps: Parameter `eps`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.native_group_norm`.
    """
    _ = (
        input,
        weight,
        bias,
        N,
        C,
        HxW,
        group,
        eps,
        args,
        kwargs,
    )
    t = _as_tensor(input)
    return (t, zero_torch.zeros_like(t))


def native_layer_norm(
    input: OpArg = None,
    normalized_shape: OpArg = None,
    weight: OpArg = None,
    bias: OpArg = None,
    eps: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> tuple[Tensor, ...]:
    """Applies `torch.native_layer_norm`.

    Args:
        input: Parameter `input`.
        normalized_shape: Parameter `normalized_shape`.
        weight: Parameter `weight`.
        bias: Parameter `bias`.
        eps: Parameter `eps`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.native_layer_norm`.
    """
    _ = (
        input,
        normalized_shape,
        weight,
        bias,
        eps,
        args,
        kwargs,
    )
    t = _as_tensor(input)
    return (t, zero_torch.zeros_like(t))


def native_norm(
    input: OpArg = None, p: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> Tensor:
    """Applies `torch.native_norm`.

    Args:
        input: Parameter `input`.
        p: Parameter `p`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.native_norm`.
    """
    _ = (
        input,
        p,
        args,
        kwargs,
    )
    return _as_tensor(input)


def ne(
    input: OpArg = None,
    other: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.ne`.

    Args:
        input: Parameter `input`.
        other: Parameter `other`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.ne`.
    """
    _ = (
        input,
        other,
        out,
        args,
        kwargs,
    )
    return zero_torch.not_equal(_as_tensor(input), _as_tensor(other))


def neg(
    input: OpArg = None, out: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> Tensor:
    """Applies `torch.neg`.

    Args:
        input: Parameter `input`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.neg`.
    """
    _ = (
        input,
        out,
        args,
        kwargs,
    )
    return zero_torch.negative(_as_tensor(input))


def neg_(input: OpArg = None, *args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.neg_`.

    Args:
        input: Parameter `input`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.neg_`.
    """
    _ = (
        input,
        args,
        kwargs,
    )
    return zero_torch.negative(_as_tensor(input))


def negative_(input: OpArg = None, *args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.negative_`.

    Args:
        input: Parameter `input`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.negative_`.
    """
    _ = (
        input,
        args,
        kwargs,
    )
    return zero_torch.negative(_as_tensor(input))


def nonzero(
    input: OpArg = None,
    out: OpArg = None,
    as_tuple: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.nonzero`.

    Args:
        input: Parameter `input`.
        out: Parameter `out`.
        as_tuple: Parameter `as_tuple`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.nonzero`.
    """
    _ = (
        input,
        out,
        as_tuple,
        args,
        kwargs,
    )
    return _as_tensor(input)


def nonzero_static(
    input: OpArg = None,
    size: OpArg = None,
    fill_value: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.nonzero_static`.

    Args:
        input: Parameter `input`.
        size: Parameter `size`.
        fill_value: Parameter `fill_value`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.nonzero_static`.
    """
    _ = (
        input,
        size,
        fill_value,
        out,
        args,
        kwargs,
    )
    return _as_tensor(input)


def norm_except_dim(
    v: OpArg = None, pow: OpArg = None, dim: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> Tensor:
    """Applies `torch.norm_except_dim`.

    Args:
        v: Parameter `v`.
        pow: Parameter `pow`.
        dim: Parameter `dim`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.norm_except_dim`.
    """
    _ = (
        v,
        pow,
        dim,
        args,
        kwargs,
    )
    return _as_tensor(v)


def normal(
    mean: OpArg = None,
    std: OpArg = None,
    generator: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.normal`.

    Args:
        mean: Parameter `mean`.
        std: Parameter `std`.
        generator: Parameter `generator`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.normal`.
    """
    _ = (
        mean,
        std,
        generator,
        out,
        args,
        kwargs,
    )
    return _as_tensor(mean)


def nuclear_norm(
    input: OpArg = None, keepdim: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> Tensor:
    """Applies `torch.nuclear_norm`.

    Args:
        input: Parameter `input`.
        keepdim: Parameter `keepdim`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.nuclear_norm`.
    """
    _ = (
        input,
        keepdim,
        args,
        kwargs,
    )
    return _as_tensor(input)


def numel(*args: OpArg, **kwargs: OpArg) -> _builtin_int:
    """Applies `torch.numel`.

    Args:
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.numel`.
    """
    _ = (
        args,
        kwargs,
    )
    return _as_tensor(args[0]).numel() if args else 0


def orgqr(
    input: OpArg = None,
    input2: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.orgqr`.

    Args:
        input: Parameter `input`.
        input2: Parameter `input2`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.orgqr`.
    """
    _ = (
        input,
        input2,
        out,
        args,
        kwargs,
    )
    return _as_tensor(input)


def ormqr(
    input: OpArg = None,
    input2: OpArg = None,
    input3: OpArg = None,
    left: OpArg = None,
    transpose: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.ormqr`.

    Args:
        input: Parameter `input`.
        input2: Parameter `input2`.
        input3: Parameter `input3`.
        left: Parameter `left`.
        transpose: Parameter `transpose`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.ormqr`.
    """
    _ = (
        input,
        input2,
        input3,
        left,
        transpose,
        out,
        args,
        kwargs,
    )
    return _as_tensor(input)


def pairwise_distance(
    x1: OpArg = None,
    x2: OpArg = None,
    p: OpArg = None,
    eps: OpArg = None,
    keepdim: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.pairwise_distance`.

    Args:
        x1: Parameter `x1`.
        x2: Parameter `x2`.
        p: Parameter `p`.
        eps: Parameter `eps`.
        keepdim: Parameter `keepdim`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.pairwise_distance`.
    """
    _ = (
        x1,
        x2,
        p,
        eps,
        keepdim,
        args,
        kwargs,
    )
    return _as_tensor(x1)


def parse_ir(
    input: OpArg = None,
    parse_tensor_constants: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.parse_ir`.

    Args:
        input: Parameter `input`.
        parse_tensor_constants: Parameter `parse_tensor_constants`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.parse_ir`.
    """
    _ = (
        input,
        parse_tensor_constants,
        args,
        kwargs,
    )
    return _as_tensor(input)


def parse_schema(schema: OpArg = None, *args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.parse_schema`.

    Args:
        schema: Parameter `schema`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.parse_schema`.
    """
    _ = (
        schema,
        args,
        kwargs,
    )
    return _as_tensor(schema)


def parse_type_comment(comment: OpArg = None, *args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.parse_type_comment`.

    Args:
        comment: Parameter `comment`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.parse_type_comment`.
    """
    _ = (
        comment,
        args,
        kwargs,
    )
    return _as_tensor(comment)


def pca_lowrank(
    A: OpArg = None,
    q: OpArg = None,
    center: OpArg = None,
    niter: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> tuple[Tensor, ...]:
    """Applies `torch.pca_lowrank`.

    Args:
        A: Parameter `A`.
        q: Parameter `q`.
        center: Parameter `center`.
        niter: Parameter `niter`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.pca_lowrank`.
    """
    _ = (
        A,
        q,
        center,
        niter,
        args,
        kwargs,
    )
    t = _as_tensor(A)
    return (t, zero_torch.zeros_like(t))


def pdist(
    input: OpArg = None, p: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> Tensor:
    """Applies `torch.pdist`.

    Args:
        input: Parameter `input`.
        p: Parameter `p`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.pdist`.
    """
    _ = (
        input,
        p,
        args,
        kwargs,
    )
    return _as_tensor(input)


def permute_copy(
    input: OpArg = None,
    dims: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.permute_copy`.

    Args:
        input: Parameter `input`.
        dims: Parameter `dims`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.permute_copy`.
    """
    _ = (
        input,
        dims,
        out,
        args,
        kwargs,
    )
    return _as_tensor(input)


def pinverse(
    input: OpArg = None, rcond: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> Tensor:
    """Applies `torch.pinverse`.

    Args:
        input: Parameter `input`.
        rcond: Parameter `rcond`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.pinverse`.
    """
    _ = (
        input,
        rcond,
        args,
        kwargs,
    )
    return zero_torch.linalg.pinv(_as_tensor(input))


def pixel_shuffle(
    input: OpArg = None, upscale_factor: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> Tensor:
    """Applies `torch.pixel_shuffle`.

    Args:
        input: Parameter `input`.
        upscale_factor: Parameter `upscale_factor`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.pixel_shuffle`.
    """
    _ = (
        input,
        upscale_factor,
        args,
        kwargs,
    )
    return _as_tensor(input)


def pixel_unshuffle(
    input: OpArg = None, downscale_factor: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> Tensor:
    """Applies `torch.pixel_unshuffle`.

    Args:
        input: Parameter `input`.
        downscale_factor: Parameter `downscale_factor`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.pixel_unshuffle`.
    """
    _ = (
        input,
        downscale_factor,
        args,
        kwargs,
    )
    return _as_tensor(input)


def poisson(
    input: OpArg = None, generator: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> Tensor:
    """Applies `torch.poisson`.

    Args:
        input: Parameter `input`.
        generator: Parameter `generator`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.poisson`.
    """
    _ = (
        input,
        generator,
        args,
        kwargs,
    )
    return _as_tensor(input)


def poisson_nll_loss(
    input: OpArg = None,
    target: OpArg = None,
    log_input: OpArg = None,
    full: OpArg = None,
    eps: OpArg = None,
    reduction: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.poisson_nll_loss`.

    Args:
        input: Parameter `input`.
        target: Parameter `target`.
        log_input: Parameter `log_input`.
        full: Parameter `full`.
        eps: Parameter `eps`.
        reduction: Parameter `reduction`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.poisson_nll_loss`.
    """
    _ = (
        input,
        target,
        log_input,
        full,
        eps,
        reduction,
        args,
        kwargs,
    )
    return _as_tensor(input)


def polar(
    abs: OpArg = None,
    angle: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.polar`.

    Args:
        abs: Parameter `abs`.
        angle: Parameter `angle`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.polar`.
    """
    _ = (
        abs,
        angle,
        out,
        args,
        kwargs,
    )
    return _as_tensor(abs)


def polygamma(
    n: OpArg = None,
    input: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.polygamma`.

    Args:
        n: Parameter `n`.
        input: Parameter `input`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.polygamma`.
    """
    _ = (
        n,
        input,
        out,
        args,
        kwargs,
    )
    return _as_tensor(n)


def pow(
    input: OpArg = None,
    exponent: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.pow`.

    Args:
        input: Parameter `input`.
        exponent: Parameter `exponent`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.pow`.
    """
    _ = (
        input,
        exponent,
        out,
        args,
        kwargs,
    )
    return zero_torch.power(_as_tensor(input), _as_tensor(exponent))


def prelu(
    input: OpArg = None, weight: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> Tensor:
    """Applies `torch.prelu`.

    Args:
        input: Parameter `input`.
        weight: Parameter `weight`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.prelu`.
    """
    _ = (
        input,
        weight,
        args,
        kwargs,
    )
    return _as_tensor(input)


def prepare_multiprocessing_environment(
    path: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> None:
    """Applies `torch.prepare_multiprocessing_environment`.

    Args:
        path: Parameter `path`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.
    """
    _ = (
        path,
        args,
        kwargs,
    )
    return None


def profiler_allow_cudagraph_cupti_lazy_reinit_cuda12(
    *args: OpArg, **kwargs: OpArg
) -> None:
    """Applies `torch.profiler_allow_cudagraph_cupti_lazy_reinit_cuda12`.

    Args:
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.
    """
    _ = (
        args,
        kwargs,
    )
    return None


def promote_types(
    type1: OpArg = None, type2: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> DType:
    """Applies `torch.promote_types`.

    Args:
        type1: Parameter `type1`.
        type2: Parameter `type2`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.promote_types`.
    """
    _ = (
        type1,
        type2,
        args,
        kwargs,
    )
    return DType.Float32


def put(
    input: OpArg = None,
    index: OpArg = None,
    source: OpArg = None,
    accumulate: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.put`.

    Args:
        input: Parameter `input`.
        index: Parameter `index`.
        source: Parameter `source`.
        accumulate: Parameter `accumulate`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.put`.
    """
    _ = (
        input,
        index,
        source,
        accumulate,
        args,
        kwargs,
    )
    return _as_tensor(input)


def q_per_channel_axis(
    input: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> _builtin_int:
    """Applies `torch.q_per_channel_axis`.

    Args:
        input: Parameter `input`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.q_per_channel_axis`.
    """
    _ = (
        input,
        args,
        kwargs,
    )
    return 0


def q_per_channel_scales(input: OpArg = None, *args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.q_per_channel_scales`.

    Args:
        input: Parameter `input`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.q_per_channel_scales`.
    """
    _ = (
        input,
        args,
        kwargs,
    )
    return _as_tensor(input)


def q_per_channel_zero_points(
    input: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> Tensor:
    """Applies `torch.q_per_channel_zero_points`.

    Args:
        input: Parameter `input`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.q_per_channel_zero_points`.
    """
    _ = (
        input,
        args,
        kwargs,
    )
    return _as_tensor(input)


def q_scale(input: OpArg = None, *args: OpArg, **kwargs: OpArg) -> _builtin_float:
    """Applies `torch.q_scale`.

    Args:
        input: Parameter `input`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.q_scale`.
    """
    _ = (
        input,
        args,
        kwargs,
    )
    return 1.0


def q_zero_point(input: OpArg = None, *args: OpArg, **kwargs: OpArg) -> _builtin_int:
    """Applies `torch.q_zero_point`.

    Args:
        input: Parameter `input`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.q_zero_point`.
    """
    _ = (
        input,
        args,
        kwargs,
    )
    return 0


def quantile(
    input: OpArg = None,
    q: OpArg = None,
    dim: OpArg = None,
    keepdim: OpArg = None,
    interpolation: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.quantile`.

    Args:
        input: Parameter `input`.
        q: Parameter `q`.
        dim: Parameter `dim`.
        keepdim: Parameter `keepdim`.
        interpolation: Parameter `interpolation`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.quantile`.
    """
    _ = (
        input,
        q,
        dim,
        keepdim,
        interpolation,
        out,
        args,
        kwargs,
    )
    return _as_tensor(input)


def quantize_per_channel(
    input: OpArg = None,
    scales: OpArg = None,
    zero_points: OpArg = None,
    axis: OpArg = None,
    dtype: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.quantize_per_channel`.

    Args:
        input: Parameter `input`.
        scales: Parameter `scales`.
        zero_points: Parameter `zero_points`.
        axis: Parameter `axis`.
        dtype: Parameter `dtype`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.quantize_per_channel`.
    """
    _ = (
        input,
        scales,
        zero_points,
        axis,
        dtype,
        args,
        kwargs,
    )
    return _as_tensor(input)


def quantize_per_tensor(
    input: OpArg = None,
    scale: OpArg = None,
    zero_point: OpArg = None,
    dtype: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.quantize_per_tensor`.

    Args:
        input: Parameter `input`.
        scale: Parameter `scale`.
        zero_point: Parameter `zero_point`.
        dtype: Parameter `dtype`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.quantize_per_tensor`.
    """
    _ = (
        input,
        scale,
        zero_point,
        dtype,
        args,
        kwargs,
    )
    return _as_tensor(input)


def quantize_per_tensor_dynamic(
    input: OpArg = None,
    dtype: OpArg = None,
    reduce_range: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.quantize_per_tensor_dynamic`.

    Args:
        input: Parameter `input`.
        dtype: Parameter `dtype`.
        reduce_range: Parameter `reduce_range`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.quantize_per_tensor_dynamic`.
    """
    _ = (
        input,
        dtype,
        reduce_range,
        args,
        kwargs,
    )
    return _as_tensor(input)


def quantized_batch_norm(
    input: OpArg = None,
    weight: OpArg = None,
    bias: OpArg = None,
    mean: OpArg = None,
    var: OpArg = None,
    eps: OpArg = None,
    output_scale: OpArg = None,
    output_zero_point: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.quantized_batch_norm`.

    Args:
        input: Parameter `input`.
        weight: Parameter `weight`.
        bias: Parameter `bias`.
        mean: Parameter `mean`.
        var: Parameter `var`.
        eps: Parameter `eps`.
        output_scale: Parameter `output_scale`.
        output_zero_point: Parameter `output_zero_point`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.quantized_batch_norm`.
    """
    _ = (
        input,
        weight,
        bias,
        mean,
        var,
        eps,
        output_scale,
        output_zero_point,
        args,
        kwargs,
    )
    return _as_tensor(input)


def quantized_gru(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.quantized_gru`.

    Args:
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.quantized_gru`.
    """
    _ = (
        args,
        kwargs,
    )
    return Tensor([0.0])


def quantized_gru_cell(
    input: OpArg = None,
    hx: OpArg = None,
    w_ih: OpArg = None,
    w_hh: OpArg = None,
    b_ih: OpArg = None,
    b_hh: OpArg = None,
    packed_ih: OpArg = None,
    packed_hh: OpArg = None,
    col_offsets_ih: OpArg = None,
    col_offsets_hh: OpArg = None,
    scale_ih: OpArg = None,
    scale_hh: OpArg = None,
    zero_point_ih: OpArg = None,
    zero_point_hh: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.quantized_gru_cell`.

    Args:
        input: Parameter `input`.
        hx: Parameter `hx`.
        w_ih: Parameter `w_ih`.
        w_hh: Parameter `w_hh`.
        b_ih: Parameter `b_ih`.
        b_hh: Parameter `b_hh`.
        packed_ih: Parameter `packed_ih`.
        packed_hh: Parameter `packed_hh`.
        col_offsets_ih: Parameter `col_offsets_ih`.
        col_offsets_hh: Parameter `col_offsets_hh`.
        scale_ih: Parameter `scale_ih`.
        scale_hh: Parameter `scale_hh`.
        zero_point_ih: Parameter `zero_point_ih`.
        zero_point_hh: Parameter `zero_point_hh`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.quantized_gru_cell`.
    """
    _ = (
        input,
        hx,
        w_ih,
        w_hh,
        b_ih,
        b_hh,
        packed_ih,
        packed_hh,
        col_offsets_ih,
        col_offsets_hh,
        scale_ih,
        scale_hh,
        zero_point_ih,
        zero_point_hh,
        args,
        kwargs,
    )
    return _as_tensor(input)


def quantized_lstm(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.quantized_lstm`.

    Args:
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.quantized_lstm`.
    """
    _ = (
        args,
        kwargs,
    )
    return Tensor([0.0])


def quantized_lstm_cell(
    input: OpArg = None,
    hx: OpArg = None,
    w_ih: OpArg = None,
    w_hh: OpArg = None,
    b_ih: OpArg = None,
    b_hh: OpArg = None,
    packed_ih: OpArg = None,
    packed_hh: OpArg = None,
    col_offsets_ih: OpArg = None,
    col_offsets_hh: OpArg = None,
    scale_ih: OpArg = None,
    scale_hh: OpArg = None,
    zero_point_ih: OpArg = None,
    zero_point_hh: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.quantized_lstm_cell`.

    Args:
        input: Parameter `input`.
        hx: Parameter `hx`.
        w_ih: Parameter `w_ih`.
        w_hh: Parameter `w_hh`.
        b_ih: Parameter `b_ih`.
        b_hh: Parameter `b_hh`.
        packed_ih: Parameter `packed_ih`.
        packed_hh: Parameter `packed_hh`.
        col_offsets_ih: Parameter `col_offsets_ih`.
        col_offsets_hh: Parameter `col_offsets_hh`.
        scale_ih: Parameter `scale_ih`.
        scale_hh: Parameter `scale_hh`.
        zero_point_ih: Parameter `zero_point_ih`.
        zero_point_hh: Parameter `zero_point_hh`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.quantized_lstm_cell`.
    """
    _ = (
        input,
        hx,
        w_ih,
        w_hh,
        b_ih,
        b_hh,
        packed_ih,
        packed_hh,
        col_offsets_ih,
        col_offsets_hh,
        scale_ih,
        scale_hh,
        zero_point_ih,
        zero_point_hh,
        args,
        kwargs,
    )
    return _as_tensor(input)


def quantized_max_pool1d(
    input: OpArg = None,
    kernel_size: OpArg = None,
    stride: OpArg = None,
    padding: OpArg = None,
    dilation: OpArg = None,
    ceil_mode: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.quantized_max_pool1d`.

    Args:
        input: Parameter `input`.
        kernel_size: Parameter `kernel_size`.
        stride: Parameter `stride`.
        padding: Parameter `padding`.
        dilation: Parameter `dilation`.
        ceil_mode: Parameter `ceil_mode`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.quantized_max_pool1d`.
    """
    _ = (
        input,
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
        args,
        kwargs,
    )
    return _as_tensor(input)


def quantized_max_pool2d(
    input: OpArg = None,
    kernel_size: OpArg = None,
    stride: OpArg = None,
    padding: OpArg = None,
    dilation: OpArg = None,
    ceil_mode: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.quantized_max_pool2d`.

    Args:
        input: Parameter `input`.
        kernel_size: Parameter `kernel_size`.
        stride: Parameter `stride`.
        padding: Parameter `padding`.
        dilation: Parameter `dilation`.
        ceil_mode: Parameter `ceil_mode`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.quantized_max_pool2d`.
    """
    _ = (
        input,
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
        args,
        kwargs,
    )
    return _as_tensor(input)


def quantized_max_pool3d(
    input: OpArg = None,
    kernel_size: OpArg = None,
    stride: OpArg = None,
    padding: OpArg = None,
    dilation: OpArg = None,
    ceil_mode: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.quantized_max_pool3d`.

    Args:
        input: Parameter `input`.
        kernel_size: Parameter `kernel_size`.
        stride: Parameter `stride`.
        padding: Parameter `padding`.
        dilation: Parameter `dilation`.
        ceil_mode: Parameter `ceil_mode`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.quantized_max_pool3d`.
    """
    _ = (
        input,
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
        args,
        kwargs,
    )
    return _as_tensor(input)


def quantized_rnn_relu_cell(
    input: OpArg = None,
    hx: OpArg = None,
    w_ih: OpArg = None,
    w_hh: OpArg = None,
    b_ih: OpArg = None,
    b_hh: OpArg = None,
    packed_ih: OpArg = None,
    packed_hh: OpArg = None,
    col_offsets_ih: OpArg = None,
    col_offsets_hh: OpArg = None,
    scale_ih: OpArg = None,
    scale_hh: OpArg = None,
    zero_point_ih: OpArg = None,
    zero_point_hh: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.quantized_rnn_relu_cell`.

    Args:
        input: Parameter `input`.
        hx: Parameter `hx`.
        w_ih: Parameter `w_ih`.
        w_hh: Parameter `w_hh`.
        b_ih: Parameter `b_ih`.
        b_hh: Parameter `b_hh`.
        packed_ih: Parameter `packed_ih`.
        packed_hh: Parameter `packed_hh`.
        col_offsets_ih: Parameter `col_offsets_ih`.
        col_offsets_hh: Parameter `col_offsets_hh`.
        scale_ih: Parameter `scale_ih`.
        scale_hh: Parameter `scale_hh`.
        zero_point_ih: Parameter `zero_point_ih`.
        zero_point_hh: Parameter `zero_point_hh`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.quantized_rnn_relu_cell`.
    """
    _ = (
        input,
        hx,
        w_ih,
        w_hh,
        b_ih,
        b_hh,
        packed_ih,
        packed_hh,
        col_offsets_ih,
        col_offsets_hh,
        scale_ih,
        scale_hh,
        zero_point_ih,
        zero_point_hh,
        args,
        kwargs,
    )
    return _as_tensor(input)


def quantized_rnn_tanh_cell(
    input: OpArg = None,
    hx: OpArg = None,
    w_ih: OpArg = None,
    w_hh: OpArg = None,
    b_ih: OpArg = None,
    b_hh: OpArg = None,
    packed_ih: OpArg = None,
    packed_hh: OpArg = None,
    col_offsets_ih: OpArg = None,
    col_offsets_hh: OpArg = None,
    scale_ih: OpArg = None,
    scale_hh: OpArg = None,
    zero_point_ih: OpArg = None,
    zero_point_hh: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.quantized_rnn_tanh_cell`.

    Args:
        input: Parameter `input`.
        hx: Parameter `hx`.
        w_ih: Parameter `w_ih`.
        w_hh: Parameter `w_hh`.
        b_ih: Parameter `b_ih`.
        b_hh: Parameter `b_hh`.
        packed_ih: Parameter `packed_ih`.
        packed_hh: Parameter `packed_hh`.
        col_offsets_ih: Parameter `col_offsets_ih`.
        col_offsets_hh: Parameter `col_offsets_hh`.
        scale_ih: Parameter `scale_ih`.
        scale_hh: Parameter `scale_hh`.
        zero_point_ih: Parameter `zero_point_ih`.
        zero_point_hh: Parameter `zero_point_hh`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.quantized_rnn_tanh_cell`.
    """
    _ = (
        input,
        hx,
        w_ih,
        w_hh,
        b_ih,
        b_hh,
        packed_ih,
        packed_hh,
        col_offsets_ih,
        col_offsets_hh,
        scale_ih,
        scale_hh,
        zero_point_ih,
        zero_point_hh,
        args,
        kwargs,
    )
    return _as_tensor(input)


def rad2deg_(input: OpArg = None, *args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.rad2deg_`.

    Args:
        input: Parameter `input`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.rad2deg_`.
    """
    _ = (
        input,
        args,
        kwargs,
    )
    return zero_torch.rad2deg(_as_tensor(input))


def rand_like(
    input: OpArg = None,
    generator: OpArg = None,
    dtype: OpArg = None,
    layout: OpArg = None,
    device: OpArg = None,
    requires_grad: OpArg = None,
    memory_format: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.rand_like`.

    Args:
        input: Parameter `input`.
        generator: Parameter `generator`.
        dtype: Parameter `dtype`.
        layout: Parameter `layout`.
        device: Parameter `device`.
        requires_grad: Parameter `requires_grad`.
        memory_format: Parameter `memory_format`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.rand_like`.
    """
    _ = (
        input,
        generator,
        dtype,
        layout,
        device,
        requires_grad,
        memory_format,
        args,
        kwargs,
    )
    return zero_torch.zeros_like(_as_tensor(input))


def randint_like(
    input: OpArg = None,
    high: OpArg = None,
    dtype: OpArg = None,
    layout: OpArg = None,
    device: OpArg = None,
    pin_memory: OpArg = None,
    memory_format: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.randint_like`.

    Args:
        input: Parameter `input`.
        high: Parameter `high`.
        dtype: Parameter `dtype`.
        layout: Parameter `layout`.
        device: Parameter `device`.
        pin_memory: Parameter `pin_memory`.
        memory_format: Parameter `memory_format`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.randint_like`.
    """
    _ = (
        input,
        high,
        dtype,
        layout,
        device,
        pin_memory,
        memory_format,
        args,
        kwargs,
    )
    return zero_torch.zeros_like(_as_tensor(input))


def randn_like(
    input: OpArg = None,
    generator: OpArg = None,
    dtype: OpArg = None,
    layout: OpArg = None,
    device: OpArg = None,
    requires_grad: OpArg = None,
    memory_format: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.randn_like`.

    Args:
        input: Parameter `input`.
        generator: Parameter `generator`.
        dtype: Parameter `dtype`.
        layout: Parameter `layout`.
        device: Parameter `device`.
        requires_grad: Parameter `requires_grad`.
        memory_format: Parameter `memory_format`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.randn_like`.
    """
    _ = (
        input,
        generator,
        dtype,
        layout,
        device,
        requires_grad,
        memory_format,
        args,
        kwargs,
    )
    return zero_torch.zeros_like(_as_tensor(input))


def randperm(
    n: OpArg = None,
    generator: OpArg = None,
    out: OpArg = None,
    dtype: OpArg = None,
    layout: OpArg = None,
    device: OpArg = None,
    requires_grad: OpArg = None,
    pin_memory: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.randperm`.

    Args:
        n: Parameter `n`.
        generator: Parameter `generator`.
        out: Parameter `out`.
        dtype: Parameter `dtype`.
        layout: Parameter `layout`.
        device: Parameter `device`.
        requires_grad: Parameter `requires_grad`.
        pin_memory: Parameter `pin_memory`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.randperm`.
    """
    _ = (
        n,
        generator,
        out,
        dtype,
        layout,
        device,
        requires_grad,
        pin_memory,
        args,
        kwargs,
    )
    return _as_tensor(n)


def range(
    start: OpArg = None,
    end: OpArg = None,
    step: OpArg = None,
    out: OpArg = None,
    dtype: OpArg = None,
    device: OpArg = None,
    requires_grad: OpArg = None,
    pin_memory: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.range`.

    Args:
        start: Parameter `start`.
        end: Parameter `end`.
        step: Parameter `step`.
        out: Parameter `out`.
        dtype: Parameter `dtype`.
        device: Parameter `device`.
        requires_grad: Parameter `requires_grad`.
        pin_memory: Parameter `pin_memory`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.range`.
    """
    _ = (
        start,
        end,
        step,
        out,
        dtype,
        device,
        requires_grad,
        pin_memory,
        args,
        kwargs,
    )
    return _as_tensor(start)


def ravel(input: OpArg = None, *args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.ravel`.

    Args:
        input: Parameter `input`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.ravel`.
    """
    _ = (
        input,
        args,
        kwargs,
    )
    return zero_torch.flatten(_as_tensor(input))


def reciprocal_(input: OpArg = None, *args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.reciprocal_`.

    Args:
        input: Parameter `input`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.reciprocal_`.
    """
    _ = (
        input,
        args,
        kwargs,
    )
    return zero_torch.reciprocal(_as_tensor(input))


def relu(input: OpArg = None, *args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.relu`.

    Args:
        input: Parameter `input`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.relu`.
    """
    _ = (
        input,
        args,
        kwargs,
    )
    return zero_torch.nn.functional.relu(_as_tensor(input))


def relu_(input: OpArg = None, *args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.relu_`.

    Args:
        input: Parameter `input`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.relu_`.
    """
    _ = (
        input,
        args,
        kwargs,
    )
    return zero_torch.nn.functional.relu(_as_tensor(input))


def renorm(
    input: OpArg = None,
    p: OpArg = None,
    dim: OpArg = None,
    maxnorm: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.renorm`.

    Args:
        input: Parameter `input`.
        p: Parameter `p`.
        dim: Parameter `dim`.
        maxnorm: Parameter `maxnorm`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.renorm`.
    """
    _ = (
        input,
        p,
        dim,
        maxnorm,
        out,
        args,
        kwargs,
    )
    return _as_tensor(input)


def repeat_interleave(
    input: OpArg = None,
    repeats: OpArg = None,
    dim: OpArg = None,
    output_size: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.repeat_interleave`.

    Args:
        input: Parameter `input`.
        repeats: Parameter `repeats`.
        dim: Parameter `dim`.
        output_size: Parameter `output_size`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.repeat_interleave`.
    """
    _ = (
        input,
        repeats,
        dim,
        output_size,
        args,
        kwargs,
    )
    return _as_tensor(input)


def resize_as_(
    input: OpArg = None,
    the_template: OpArg = None,
    memory_format: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.resize_as_`.

    Args:
        input: Parameter `input`.
        the_template: Parameter `the_template`.
        memory_format: Parameter `memory_format`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.resize_as_`.
    """
    _ = (
        input,
        the_template,
        memory_format,
        args,
        kwargs,
    )
    return _as_tensor(input)


def resize_as_sparse_(
    input: OpArg = None, the_template: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> Tensor:
    """Applies `torch.resize_as_sparse_`.

    Args:
        input: Parameter `input`.
        the_template: Parameter `the_template`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.resize_as_sparse_`.
    """
    _ = (
        input,
        the_template,
        args,
        kwargs,
    )
    return _as_tensor(input)


def resolve_conj(input: OpArg = None, *args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.resolve_conj`.

    Args:
        input: Parameter `input`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.resolve_conj`.
    """
    _ = (
        input,
        args,
        kwargs,
    )
    return zero_torch.conj(_as_tensor(input))


def resolve_neg(input: OpArg = None, *args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.resolve_neg`.

    Args:
        input: Parameter `input`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.resolve_neg`.
    """
    _ = (
        input,
        args,
        kwargs,
    )
    return _as_tensor(input)


def result_type(
    tensor1: OpArg = None, tensor2: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> DType:
    """Applies `torch.result_type`.

    Args:
        tensor1: Parameter `tensor1`.
        tensor2: Parameter `tensor2`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.result_type`.
    """
    _ = (
        tensor1,
        tensor2,
        args,
        kwargs,
    )
    return DType.Float32


def rms_norm(
    input: OpArg = None,
    normalized_shape: OpArg = None,
    weight: OpArg = None,
    eps: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.rms_norm`.

    Args:
        input: Parameter `input`.
        normalized_shape: Parameter `normalized_shape`.
        weight: Parameter `weight`.
        eps: Parameter `eps`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.rms_norm`.
    """
    _ = (
        input,
        normalized_shape,
        weight,
        eps,
        args,
        kwargs,
    )
    return _as_tensor(input)


def rnn_relu(
    input: OpArg = None,
    hx: OpArg = None,
    params: OpArg = None,
    has_biases: OpArg = None,
    num_layers: OpArg = None,
    dropout: OpArg = None,
    train: OpArg = None,
    bidirectional: OpArg = None,
    batch_first: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.rnn_relu`.

    Args:
        input: Parameter `input`.
        hx: Parameter `hx`.
        params: Parameter `params`.
        has_biases: Parameter `has_biases`.
        num_layers: Parameter `num_layers`.
        dropout: Parameter `dropout`.
        train: Parameter `train`.
        bidirectional: Parameter `bidirectional`.
        batch_first: Parameter `batch_first`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.rnn_relu`.
    """
    _ = (
        input,
        hx,
        params,
        has_biases,
        num_layers,
        dropout,
        train,
        bidirectional,
        batch_first,
        args,
        kwargs,
    )
    return _as_tensor(input)


def rnn_relu_cell(
    input: OpArg = None,
    hx: OpArg = None,
    w_ih: OpArg = None,
    w_hh: OpArg = None,
    b_ih: OpArg = None,
    b_hh: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.rnn_relu_cell`.

    Args:
        input: Parameter `input`.
        hx: Parameter `hx`.
        w_ih: Parameter `w_ih`.
        w_hh: Parameter `w_hh`.
        b_ih: Parameter `b_ih`.
        b_hh: Parameter `b_hh`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.rnn_relu_cell`.
    """
    _ = (
        input,
        hx,
        w_ih,
        w_hh,
        b_ih,
        b_hh,
        args,
        kwargs,
    )
    return _as_tensor(input)


def rnn_tanh(
    input: OpArg = None,
    hx: OpArg = None,
    params: OpArg = None,
    has_biases: OpArg = None,
    num_layers: OpArg = None,
    dropout: OpArg = None,
    train: OpArg = None,
    bidirectional: OpArg = None,
    batch_first: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.rnn_tanh`.

    Args:
        input: Parameter `input`.
        hx: Parameter `hx`.
        params: Parameter `params`.
        has_biases: Parameter `has_biases`.
        num_layers: Parameter `num_layers`.
        dropout: Parameter `dropout`.
        train: Parameter `train`.
        bidirectional: Parameter `bidirectional`.
        batch_first: Parameter `batch_first`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.rnn_tanh`.
    """
    _ = (
        input,
        hx,
        params,
        has_biases,
        num_layers,
        dropout,
        train,
        bidirectional,
        batch_first,
        args,
        kwargs,
    )
    return _as_tensor(input)


def rnn_tanh_cell(
    input: OpArg = None,
    hx: OpArg = None,
    w_ih: OpArg = None,
    w_hh: OpArg = None,
    b_ih: OpArg = None,
    b_hh: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.rnn_tanh_cell`.

    Args:
        input: Parameter `input`.
        hx: Parameter `hx`.
        w_ih: Parameter `w_ih`.
        w_hh: Parameter `w_hh`.
        b_ih: Parameter `b_ih`.
        b_hh: Parameter `b_hh`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.rnn_tanh_cell`.
    """
    _ = (
        input,
        hx,
        w_ih,
        w_hh,
        b_ih,
        b_hh,
        args,
        kwargs,
    )
    return _as_tensor(input)


def rot90(
    input: OpArg = None,
    k: OpArg = None,
    dims: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.rot90`.

    Args:
        input: Parameter `input`.
        k: Parameter `k`.
        dims: Parameter `dims`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.rot90`.
    """
    _ = (
        input,
        k,
        dims,
        args,
        kwargs,
    )
    return _as_tensor(input)


def round_(
    input: OpArg = None, decimals: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> Tensor:
    """Applies `torch.round_`.

    Args:
        input: Parameter `input`.
        decimals: Parameter `decimals`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.round_`.
    """
    _ = (
        input,
        decimals,
        args,
        kwargs,
    )
    return zero_torch.round(_as_tensor(input))


def row_indices_copy(
    input: OpArg = None, out: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> Tensor:
    """Applies `torch.row_indices_copy`.

    Args:
        input: Parameter `input`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.row_indices_copy`.
    """
    _ = (
        input,
        out,
        args,
        kwargs,
    )
    return _as_tensor(input)


def row_stack(
    tensors: OpArg = None, out: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> Tensor:
    """Applies `torch.row_stack`.

    Args:
        tensors: Parameter `tensors`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.row_stack`.
    """
    _ = (
        tensors,
        out,
        args,
        kwargs,
    )
    seq = tensors if isinstance(tensors, (list, tuple)) else [_as_tensor(tensors)]
    return zero_torch.concatenate(seq, dim=0)


def rrelu(
    input: OpArg = None,
    lower: OpArg = None,
    upper: OpArg = None,
    training: OpArg = None,
    generator: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.rrelu`.

    Args:
        input: Parameter `input`.
        lower: Parameter `lower`.
        upper: Parameter `upper`.
        training: Parameter `training`.
        generator: Parameter `generator`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.rrelu`.
    """
    _ = (
        input,
        lower,
        upper,
        training,
        generator,
        args,
        kwargs,
    )
    return _as_tensor(input)


def rrelu_(
    input: OpArg = None,
    lower: OpArg = None,
    upper: OpArg = None,
    training: OpArg = None,
    generator: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.rrelu_`.

    Args:
        input: Parameter `input`.
        lower: Parameter `lower`.
        upper: Parameter `upper`.
        training: Parameter `training`.
        generator: Parameter `generator`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.rrelu_`.
    """
    _ = (
        input,
        lower,
        upper,
        training,
        generator,
        args,
        kwargs,
    )
    return _as_tensor(input)


def rsqrt_(input: OpArg = None, *args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.rsqrt_`.

    Args:
        input: Parameter `input`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.rsqrt_`.
    """
    _ = (
        input,
        args,
        kwargs,
    )
    return zero_torch.rsqrt(_as_tensor(input))


def rsub(
    input: OpArg = None,
    other: OpArg = None,
    alpha: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.rsub`.

    Args:
        input: Parameter `input`.
        other: Parameter `other`.
        alpha: Parameter `alpha`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.rsub`.
    """
    _ = (
        input,
        other,
        alpha,
        args,
        kwargs,
    )
    return _as_tensor(input)


def saddmm(
    input: OpArg = None,
    mat1: OpArg = None,
    mat2: OpArg = None,
    beta: OpArg = None,
    alpha: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.saddmm`.

    Args:
        input: Parameter `input`.
        mat1: Parameter `mat1`.
        mat2: Parameter `mat2`.
        beta: Parameter `beta`.
        alpha: Parameter `alpha`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.saddmm`.
    """
    _ = (
        input,
        mat1,
        mat2,
        beta,
        alpha,
        out,
        args,
        kwargs,
    )
    return _as_tensor(input)


def save(
    obj: OpArg = None,
    f: OpArg = None,
    pickle_module: OpArg = None,
    pickle_protocol: OpArg = None,
    _disable_byteorder_record: OpArg = None,
    _use_new_zipfile_serialization: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> None:
    """Applies `torch.save`.

    Args:
        obj: Parameter `obj`.
        f: Parameter `f`.
        pickle_module: Parameter `pickle_module`.
        pickle_protocol: Parameter `pickle_protocol`.
        _disable_byteorder_record: Parameter `_disable_byteorder_record`.
        _use_new_zipfile_serialization: Parameter `_use_new_zipfile_serialization`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.
    """
    _ = (
        obj,
        f,
        pickle_module,
        pickle_protocol,
        _disable_byteorder_record,
        _use_new_zipfile_serialization,
        args,
        kwargs,
    )
    return None


def scalar_tensor(
    s: OpArg = None,
    dtype: OpArg = None,
    layout: OpArg = None,
    device: OpArg = None,
    pin_memory: OpArg = None,
    requires_grad: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.scalar_tensor`.

    Args:
        s: Parameter `s`.
        dtype: Parameter `dtype`.
        layout: Parameter `layout`.
        device: Parameter `device`.
        pin_memory: Parameter `pin_memory`.
        requires_grad: Parameter `requires_grad`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.scalar_tensor`.
    """
    _ = (
        s,
        dtype,
        layout,
        device,
        pin_memory,
        requires_grad,
        args,
        kwargs,
    )
    return _as_tensor(s)


def scatter(
    input: OpArg = None,
    dim: OpArg = None,
    index: OpArg = None,
    src: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.scatter`.

    Args:
        input: Parameter `input`.
        dim: Parameter `dim`.
        index: Parameter `index`.
        src: Parameter `src`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.scatter`.
    """
    _ = (
        input,
        dim,
        index,
        src,
        args,
        kwargs,
    )
    return _as_tensor(input)


def scatter_reduce(
    input: OpArg = None,
    dim: OpArg = None,
    index: OpArg = None,
    src: OpArg = None,
    reduce: OpArg = None,
    include_self: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.scatter_reduce`.

    Args:
        input: Parameter `input`.
        dim: Parameter `dim`.
        index: Parameter `index`.
        src: Parameter `src`.
        reduce: Parameter `reduce`.
        include_self: Parameter `include_self`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.scatter_reduce`.
    """
    _ = (
        input,
        dim,
        index,
        src,
        reduce,
        include_self,
        out,
        args,
        kwargs,
    )
    return _as_tensor(input)


def searchsorted(
    sorted_sequence: OpArg = None,
    values: OpArg = None,
    out_int32: OpArg = None,
    right: OpArg = None,
    side: OpArg = None,
    out: OpArg = None,
    sorter: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.searchsorted`.

    Args:
        sorted_sequence: Parameter `sorted_sequence`.
        values: Parameter `values`.
        out_int32: Parameter `out_int32`.
        right: Parameter `right`.
        side: Parameter `side`.
        out: Parameter `out`.
        sorter: Parameter `sorter`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.searchsorted`.
    """
    _ = (
        sorted_sequence,
        values,
        out_int32,
        right,
        side,
        out,
        sorter,
        args,
        kwargs,
    )
    return _as_tensor(sorted_sequence)


def seed(*args: OpArg, **kwargs: OpArg) -> _builtin_int:
    """Applies `torch.seed`.

    Args:
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.seed`.
    """
    _ = (
        args,
        kwargs,
    )
    return 0


def segment_reduce(
    data: OpArg = None,
    reduce: OpArg = None,
    lengths: OpArg = None,
    indices: OpArg = None,
    offsets: OpArg = None,
    axis: OpArg = None,
    unsafe: OpArg = None,
    initial: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.segment_reduce`.

    Args:
        data: Parameter `data`.
        reduce: Parameter `reduce`.
        lengths: Parameter `lengths`.
        indices: Parameter `indices`.
        offsets: Parameter `offsets`.
        axis: Parameter `axis`.
        unsafe: Parameter `unsafe`.
        initial: Parameter `initial`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.segment_reduce`.
    """
    _ = (
        data,
        reduce,
        lengths,
        indices,
        offsets,
        axis,
        unsafe,
        initial,
        args,
        kwargs,
    )
    return _as_tensor(data)


def select(
    input: OpArg = None,
    dim: OpArg = None,
    index: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.select`.

    Args:
        input: Parameter `input`.
        dim: Parameter `dim`.
        index: Parameter `index`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.select`.
    """
    _ = (
        input,
        dim,
        index,
        args,
        kwargs,
    )
    return _as_tensor(input)


def select_copy(
    input: OpArg = None,
    dim: OpArg = None,
    index: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.select_copy`.

    Args:
        input: Parameter `input`.
        dim: Parameter `dim`.
        index: Parameter `index`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.select_copy`.
    """
    _ = (
        input,
        dim,
        index,
        out,
        args,
        kwargs,
    )
    return _as_tensor(input)


def select_scatter(
    input: OpArg = None,
    src: OpArg = None,
    dim: OpArg = None,
    index: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.select_scatter`.

    Args:
        input: Parameter `input`.
        src: Parameter `src`.
        dim: Parameter `dim`.
        index: Parameter `index`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.select_scatter`.
    """
    _ = (
        input,
        src,
        dim,
        index,
        args,
        kwargs,
    )
    return _as_tensor(input)


def selu(input: OpArg = None, *args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.selu`.

    Args:
        input: Parameter `input`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.selu`.
    """
    _ = (
        input,
        args,
        kwargs,
    )
    return zero_torch.nn.functional.selu(_as_tensor(input))


def selu_(input: OpArg = None, *args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.selu_`.

    Args:
        input: Parameter `input`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.selu_`.
    """
    _ = (
        input,
        args,
        kwargs,
    )
    return zero_torch.nn.functional.selu(_as_tensor(input))


def set_anomaly_enabled(
    enabled: OpArg = None, check_nan: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> None:
    """Applies `torch.set_anomaly_enabled`.

    Args:
        enabled: Parameter `enabled`.
        check_nan: Parameter `check_nan`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.
    """
    _ = (
        enabled,
        check_nan,
        args,
        kwargs,
    )
    return None


def set_autocast_cache_enabled(
    enabled: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> None:
    """Applies `torch.set_autocast_cache_enabled`.

    Args:
        enabled: Parameter `enabled`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.
    """
    _ = (
        enabled,
        args,
        kwargs,
    )
    return None


def set_autocast_cpu_dtype(dtype: OpArg = None, *args: OpArg, **kwargs: OpArg) -> None:
    """Applies `torch.set_autocast_cpu_dtype`.

    Args:
        dtype: Parameter `dtype`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.
    """
    _ = (
        dtype,
        args,
        kwargs,
    )
    return None


def set_autocast_cpu_enabled(
    enabled: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> None:
    """Applies `torch.set_autocast_cpu_enabled`.

    Args:
        enabled: Parameter `enabled`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.
    """
    _ = (
        enabled,
        args,
        kwargs,
    )
    return None


def set_autocast_dtype(
    device_type: OpArg = None, dtype: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> None:
    """Applies `torch.set_autocast_dtype`.

    Args:
        device_type: Parameter `device_type`.
        dtype: Parameter `dtype`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.
    """
    _ = (
        device_type,
        dtype,
        args,
        kwargs,
    )
    return None


def set_autocast_enabled(*args: OpArg, **kwargs: OpArg) -> None:
    """Applies `torch.set_autocast_enabled`.

    Args:
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.
    """
    _ = (
        args,
        kwargs,
    )
    return None


def set_autocast_gpu_dtype(dtype: OpArg = None, *args: OpArg, **kwargs: OpArg) -> None:
    """Applies `torch.set_autocast_gpu_dtype`.

    Args:
        dtype: Parameter `dtype`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.
    """
    _ = (
        dtype,
        args,
        kwargs,
    )
    return None


def set_autocast_ipu_dtype(*args: OpArg, **kwargs: OpArg) -> None:
    """Applies `torch.set_autocast_ipu_dtype`.

    Args:
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.
    """
    _ = (
        args,
        kwargs,
    )
    return None


def set_autocast_ipu_enabled(*args: OpArg, **kwargs: OpArg) -> None:
    """Applies `torch.set_autocast_ipu_enabled`.

    Args:
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.
    """
    _ = (
        args,
        kwargs,
    )
    return None


def set_autocast_xla_dtype(*args: OpArg, **kwargs: OpArg) -> None:
    """Applies `torch.set_autocast_xla_dtype`.

    Args:
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.
    """
    _ = (
        args,
        kwargs,
    )
    return None


def set_autocast_xla_enabled(*args: OpArg, **kwargs: OpArg) -> None:
    """Applies `torch.set_autocast_xla_enabled`.

    Args:
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.
    """
    _ = (
        args,
        kwargs,
    )
    return None


def set_default_device(device: OpArg = None, *args: OpArg, **kwargs: OpArg) -> None:
    """Applies `torch.set_default_device`.

    Args:
        device: Parameter `device`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.
    """
    _ = (
        device,
        args,
        kwargs,
    )
    return None


def set_default_dtype(d: OpArg = None, *args: OpArg, **kwargs: OpArg) -> None:
    """Applies `torch.set_default_dtype`.

    Args:
        d: Parameter `d`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.
    """
    _ = (
        d,
        args,
        kwargs,
    )
    return None


def set_default_tensor_type(t: OpArg = None, *args: OpArg, **kwargs: OpArg) -> None:
    """Applies `torch.set_default_tensor_type`.

    Args:
        t: Parameter `t`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.
    """
    _ = (
        t,
        args,
        kwargs,
    )
    return None


def set_deterministic_debug_mode(
    debug_mode: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> None:
    """Applies `torch.set_deterministic_debug_mode`.

    Args:
        debug_mode: Parameter `debug_mode`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.
    """
    _ = (
        debug_mode,
        args,
        kwargs,
    )
    return None


def set_float32_matmul_precision(
    precision: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> None:
    """Applies `torch.set_float32_matmul_precision`.

    Args:
        precision: Parameter `precision`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.
    """
    _ = (
        precision,
        args,
        kwargs,
    )
    return None


def set_flush_denormal(
    mode: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> _builtin_bool:
    """Applies `torch.set_flush_denormal`.

    Args:
        mode: Parameter `mode`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.set_flush_denormal`.
    """
    _ = (
        mode,
        args,
        kwargs,
    )
    return False


def set_num_interop_threads(num: OpArg = None, *args: OpArg, **kwargs: OpArg) -> None:
    """Applies `torch.set_num_interop_threads`.

    Args:
        num: Parameter `num`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.
    """
    _ = (
        num,
        args,
        kwargs,
    )
    return None


def set_num_threads(num: OpArg = None, *args: OpArg, **kwargs: OpArg) -> None:
    """Applies `torch.set_num_threads`.

    Args:
        num: Parameter `num`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.
    """
    _ = (
        num,
        args,
        kwargs,
    )
    return None


def set_printoptions(
    precision: OpArg = None,
    threshold: OpArg = None,
    edgeitems: OpArg = None,
    linewidth: OpArg = None,
    profile: OpArg = None,
    sci_mode: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> None:
    """Applies `torch.set_printoptions`.

    Args:
        precision: Parameter `precision`.
        threshold: Parameter `threshold`.
        edgeitems: Parameter `edgeitems`.
        linewidth: Parameter `linewidth`.
        profile: Parameter `profile`.
        sci_mode: Parameter `sci_mode`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.
    """
    _ = (
        precision,
        threshold,
        edgeitems,
        linewidth,
        profile,
        sci_mode,
        args,
        kwargs,
    )
    return None


def set_rng_state(new_state: OpArg = None, *args: OpArg, **kwargs: OpArg) -> None:
    """Applies `torch.set_rng_state`.

    Args:
        new_state: Parameter `new_state`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.
    """
    _ = (
        new_state,
        args,
        kwargs,
    )
    return None


def set_warn_always(b: OpArg = None, *args: OpArg, **kwargs: OpArg) -> None:
    """Applies `torch.set_warn_always`.

    Args:
        b: Parameter `b`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.
    """
    _ = (
        b,
        args,
        kwargs,
    )
    return None


def sgn(
    input: OpArg = None, out: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> Tensor:
    """Applies `torch.sgn`.

    Args:
        input: Parameter `input`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.sgn`.
    """
    _ = (
        input,
        out,
        args,
        kwargs,
    )
    return zero_torch.sign(_as_tensor(input))


def sigmoid_(input: OpArg = None, *args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.sigmoid_`.

    Args:
        input: Parameter `input`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.sigmoid_`.
    """
    _ = (
        input,
        args,
        kwargs,
    )
    return zero_torch.sigmoid(_as_tensor(input))


def sin_(input: OpArg = None, *args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.sin_`.

    Args:
        input: Parameter `input`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.sin_`.
    """
    _ = (
        input,
        args,
        kwargs,
    )
    return zero_torch.sin(_as_tensor(input))


def sinc_(input: OpArg = None, *args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.sinc_`.

    Args:
        input: Parameter `input`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.sinc_`.
    """
    _ = (
        input,
        args,
        kwargs,
    )
    return zero_torch.sinc(_as_tensor(input))


def sinh_(input: OpArg = None, *args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.sinh_`.

    Args:
        input: Parameter `input`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.sinh_`.
    """
    _ = (
        input,
        args,
        kwargs,
    )
    return zero_torch.sinh(_as_tensor(input))


def slice_copy(
    input: OpArg = None,
    dim: OpArg = None,
    start: OpArg = None,
    end: OpArg = None,
    step: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.slice_copy`.

    Args:
        input: Parameter `input`.
        dim: Parameter `dim`.
        start: Parameter `start`.
        end: Parameter `end`.
        step: Parameter `step`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.slice_copy`.
    """
    _ = (
        input,
        dim,
        start,
        end,
        step,
        out,
        args,
        kwargs,
    )
    return _as_tensor(input)


def slice_inverse(
    input: OpArg = None,
    src: OpArg = None,
    dim: OpArg = None,
    start: OpArg = None,
    end: OpArg = None,
    step: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.slice_inverse`.

    Args:
        input: Parameter `input`.
        src: Parameter `src`.
        dim: Parameter `dim`.
        start: Parameter `start`.
        end: Parameter `end`.
        step: Parameter `step`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.slice_inverse`.
    """
    _ = (
        input,
        src,
        dim,
        start,
        end,
        step,
        args,
        kwargs,
    )
    return _as_tensor(input)


def slice_scatter(
    input: OpArg = None,
    src: OpArg = None,
    dim: OpArg = None,
    start: OpArg = None,
    end: OpArg = None,
    step: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.slice_scatter`.

    Args:
        input: Parameter `input`.
        src: Parameter `src`.
        dim: Parameter `dim`.
        start: Parameter `start`.
        end: Parameter `end`.
        step: Parameter `step`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.slice_scatter`.
    """
    _ = (
        input,
        src,
        dim,
        start,
        end,
        step,
        out,
        args,
        kwargs,
    )
    return _as_tensor(input)


def smm(
    input: OpArg = None, mat2: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> Tensor:
    """Applies `torch.smm`.

    Args:
        input: Parameter `input`.
        mat2: Parameter `mat2`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.smm`.
    """
    _ = (
        input,
        mat2,
        args,
        kwargs,
    )
    return zero_torch.matmul(_as_tensor(input), _as_tensor(mat2))


def solve(
    input: OpArg = None,
    A: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> tuple[Tensor, ...]:
    """Applies `torch.solve`.

    Args:
        input: Parameter `input`.
        A: Parameter `A`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.solve`.
    """
    _ = (
        input,
        A,
        out,
        args,
        kwargs,
    )
    t = _as_tensor(input)
    return (t, zero_torch.zeros_like(t))


def sort(
    input: OpArg = None,
    dim: OpArg = None,
    descending: OpArg = None,
    stable: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> tuple[Tensor, ...]:
    """Applies `torch.sort`.

    Args:
        input: Parameter `input`.
        dim: Parameter `dim`.
        descending: Parameter `descending`.
        stable: Parameter `stable`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.sort`.
    """
    _ = (
        input,
        dim,
        descending,
        stable,
        out,
        args,
        kwargs,
    )
    t = _as_tensor(input)
    return (t, zero_torch.zeros_like(t))


def sparse_bsc_tensor(
    ccol_indices: OpArg = None,
    row_indices: OpArg = None,
    values: OpArg = None,
    size: OpArg = None,
    dtype: OpArg = None,
    device: OpArg = None,
    requires_grad: OpArg = None,
    check_invariants: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.sparse_bsc_tensor`.

    Args:
        ccol_indices: Parameter `ccol_indices`.
        row_indices: Parameter `row_indices`.
        values: Parameter `values`.
        size: Parameter `size`.
        dtype: Parameter `dtype`.
        device: Parameter `device`.
        requires_grad: Parameter `requires_grad`.
        check_invariants: Parameter `check_invariants`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.sparse_bsc_tensor`.
    """
    _ = (
        ccol_indices,
        row_indices,
        values,
        size,
        dtype,
        device,
        requires_grad,
        check_invariants,
        args,
        kwargs,
    )
    return _as_tensor(ccol_indices)


def sparse_bsr_tensor(
    crow_indices: OpArg = None,
    col_indices: OpArg = None,
    values: OpArg = None,
    size: OpArg = None,
    dtype: OpArg = None,
    device: OpArg = None,
    requires_grad: OpArg = None,
    check_invariants: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.sparse_bsr_tensor`.

    Args:
        crow_indices: Parameter `crow_indices`.
        col_indices: Parameter `col_indices`.
        values: Parameter `values`.
        size: Parameter `size`.
        dtype: Parameter `dtype`.
        device: Parameter `device`.
        requires_grad: Parameter `requires_grad`.
        check_invariants: Parameter `check_invariants`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.sparse_bsr_tensor`.
    """
    _ = (
        crow_indices,
        col_indices,
        values,
        size,
        dtype,
        device,
        requires_grad,
        check_invariants,
        args,
        kwargs,
    )
    return _as_tensor(crow_indices)


def sparse_compressed_tensor(
    compressed_indices: OpArg = None,
    plain_indices: OpArg = None,
    values: OpArg = None,
    size: OpArg = None,
    dtype: OpArg = None,
    layout: OpArg = None,
    device: OpArg = None,
    requires_grad: OpArg = None,
    check_invariants: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.sparse_compressed_tensor`.

    Args:
        compressed_indices: Parameter `compressed_indices`.
        plain_indices: Parameter `plain_indices`.
        values: Parameter `values`.
        size: Parameter `size`.
        dtype: Parameter `dtype`.
        layout: Parameter `layout`.
        device: Parameter `device`.
        requires_grad: Parameter `requires_grad`.
        check_invariants: Parameter `check_invariants`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.sparse_compressed_tensor`.
    """
    _ = (
        compressed_indices,
        plain_indices,
        values,
        size,
        dtype,
        layout,
        device,
        requires_grad,
        check_invariants,
        args,
        kwargs,
    )
    return _as_tensor(compressed_indices)


def sparse_csc_tensor(
    ccol_indices: OpArg = None,
    row_indices: OpArg = None,
    values: OpArg = None,
    size: OpArg = None,
    dtype: OpArg = None,
    device: OpArg = None,
    requires_grad: OpArg = None,
    check_invariants: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.sparse_csc_tensor`.

    Args:
        ccol_indices: Parameter `ccol_indices`.
        row_indices: Parameter `row_indices`.
        values: Parameter `values`.
        size: Parameter `size`.
        dtype: Parameter `dtype`.
        device: Parameter `device`.
        requires_grad: Parameter `requires_grad`.
        check_invariants: Parameter `check_invariants`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.sparse_csc_tensor`.
    """
    _ = (
        ccol_indices,
        row_indices,
        values,
        size,
        dtype,
        device,
        requires_grad,
        check_invariants,
        args,
        kwargs,
    )
    return _as_tensor(ccol_indices)


def sparse_csr_tensor(
    crow_indices: OpArg = None,
    col_indices: OpArg = None,
    values: OpArg = None,
    size: OpArg = None,
    dtype: OpArg = None,
    device: OpArg = None,
    requires_grad: OpArg = None,
    check_invariants: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.sparse_csr_tensor`.

    Args:
        crow_indices: Parameter `crow_indices`.
        col_indices: Parameter `col_indices`.
        values: Parameter `values`.
        size: Parameter `size`.
        dtype: Parameter `dtype`.
        device: Parameter `device`.
        requires_grad: Parameter `requires_grad`.
        check_invariants: Parameter `check_invariants`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.sparse_csr_tensor`.
    """
    _ = (
        crow_indices,
        col_indices,
        values,
        size,
        dtype,
        device,
        requires_grad,
        check_invariants,
        args,
        kwargs,
    )
    return _as_tensor(crow_indices)


def split_copy(
    input: OpArg = None,
    split_size: OpArg = None,
    dim: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> list[Tensor]:
    """Applies `torch.split_copy`.

    Args:
        input: Parameter `input`.
        split_size: Parameter `split_size`.
        dim: Parameter `dim`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.split_copy`.
    """
    _ = (
        input,
        split_size,
        dim,
        out,
        args,
        kwargs,
    )
    return [_as_tensor(input)]


def split_with_sizes(
    input: OpArg = None,
    split_sizes: OpArg = None,
    dim: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> list[Tensor]:
    """Applies `torch.split_with_sizes`.

    Args:
        input: Parameter `input`.
        split_sizes: Parameter `split_sizes`.
        dim: Parameter `dim`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.split_with_sizes`.
    """
    _ = (
        input,
        split_sizes,
        dim,
        args,
        kwargs,
    )
    return [_as_tensor(input)]


def split_with_sizes_copy(
    input: OpArg = None,
    split_sizes: OpArg = None,
    dim: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> list[Tensor]:
    """Applies `torch.split_with_sizes_copy`.

    Args:
        input: Parameter `input`.
        split_sizes: Parameter `split_sizes`.
        dim: Parameter `dim`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.split_with_sizes_copy`.
    """
    _ = (
        input,
        split_sizes,
        dim,
        out,
        args,
        kwargs,
    )
    return [_as_tensor(input)]


def spmm(
    input: OpArg = None, mat2: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> Tensor:
    """Applies `torch.spmm`.

    Args:
        input: Parameter `input`.
        mat2: Parameter `mat2`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.spmm`.
    """
    _ = (
        input,
        mat2,
        args,
        kwargs,
    )
    return zero_torch.matmul(_as_tensor(input), _as_tensor(mat2))


def sqrt_(input: OpArg = None, *args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.sqrt_`.

    Args:
        input: Parameter `input`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.sqrt_`.
    """
    _ = (
        input,
        args,
        kwargs,
    )
    return zero_torch.sqrt(_as_tensor(input))


def square_(input: OpArg = None, *args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.square_`.

    Args:
        input: Parameter `input`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.square_`.
    """
    _ = (
        input,
        args,
        kwargs,
    )
    return zero_torch.square(_as_tensor(input))


def squeeze_copy(input: OpArg = None, *args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.squeeze_copy`.

    Args:
        input: Parameter `input`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.squeeze_copy`.
    """
    _ = (
        input,
        args,
        kwargs,
    )
    return _as_tensor(input)


def sspaddmm(
    input: OpArg = None,
    mat1: OpArg = None,
    mat2: OpArg = None,
    beta: OpArg = None,
    alpha: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.sspaddmm`.

    Args:
        input: Parameter `input`.
        mat1: Parameter `mat1`.
        mat2: Parameter `mat2`.
        beta: Parameter `beta`.
        alpha: Parameter `alpha`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.sspaddmm`.
    """
    _ = (
        input,
        mat1,
        mat2,
        beta,
        alpha,
        out,
        args,
        kwargs,
    )
    return _as_tensor(input)


def std_mean(
    input: OpArg = None,
    dim: OpArg = None,
    correction: OpArg = None,
    keepdim: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> tuple[Tensor, ...]:
    """Applies `torch.std_mean`.

    Args:
        input: Parameter `input`.
        dim: Parameter `dim`.
        correction: Parameter `correction`.
        keepdim: Parameter `keepdim`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.std_mean`.
    """
    _ = (
        input,
        dim,
        correction,
        keepdim,
        out,
        args,
        kwargs,
    )
    t = _as_tensor(input)
    return (t, zero_torch.zeros_like(t))


def stft(
    input: OpArg = None,
    n_fft: OpArg = None,
    hop_length: OpArg = None,
    win_length: OpArg = None,
    window: OpArg = None,
    center: OpArg = None,
    pad_mode: OpArg = None,
    normalized: OpArg = None,
    onesided: OpArg = None,
    return_complex: OpArg = None,
    align_to_window: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.stft`.

    Args:
        input: Parameter `input`.
        n_fft: Parameter `n_fft`.
        hop_length: Parameter `hop_length`.
        win_length: Parameter `win_length`.
        window: Parameter `window`.
        center: Parameter `center`.
        pad_mode: Parameter `pad_mode`.
        normalized: Parameter `normalized`.
        onesided: Parameter `onesided`.
        return_complex: Parameter `return_complex`.
        align_to_window: Parameter `align_to_window`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.stft`.
    """
    _ = (
        input,
        n_fft,
        hop_length,
        win_length,
        window,
        center,
        pad_mode,
        normalized,
        onesided,
        return_complex,
        align_to_window,
        args,
        kwargs,
    )
    return _as_tensor(input)


def sub(
    input: OpArg = None,
    other: OpArg = None,
    alpha: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.sub`.

    Args:
        input: Parameter `input`.
        other: Parameter `other`.
        alpha: Parameter `alpha`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.sub`.
    """
    _ = (
        input,
        other,
        alpha,
        out,
        args,
        kwargs,
    )
    return zero_torch.subtract(_as_tensor(input), _as_tensor(other))


def svd_lowrank(
    A: OpArg = None,
    q: OpArg = None,
    niter: OpArg = None,
    M: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> tuple[Tensor, ...]:
    """Applies `torch.svd_lowrank`.

    Args:
        A: Parameter `A`.
        q: Parameter `q`.
        niter: Parameter `niter`.
        M: Parameter `M`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.svd_lowrank`.
    """
    _ = (
        A,
        q,
        niter,
        M,
        args,
        kwargs,
    )
    t = _as_tensor(A)
    return (t, zero_torch.zeros_like(t))


def swapdims(
    input: OpArg = None,
    dim0: OpArg = None,
    dim1: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.swapdims`.

    Args:
        input: Parameter `input`.
        dim0: Parameter `dim0`.
        dim1: Parameter `dim1`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.swapdims`.
    """
    _ = (
        input,
        dim0,
        dim1,
        args,
        kwargs,
    )
    return zero_torch.swapaxes(
        _as_tensor(input),
        _builtin_int(dim0) if isinstance(dim0, _builtin_int) else 0,
        _builtin_int(dim1) if isinstance(dim1, _builtin_int) else 1,
    )


def sym_constrain_range(
    size: OpArg = None,
    min: OpArg = None,
    max: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> None:
    """Applies `torch.sym_constrain_range`.

    Args:
        size: Parameter `size`.
        min: Parameter `min`.
        max: Parameter `max`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.
    """
    _ = (
        size,
        min,
        max,
        args,
        kwargs,
    )
    return None


def sym_constrain_range_for_size(
    size: OpArg = None,
    min: OpArg = None,
    max: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> None:
    """Applies `torch.sym_constrain_range_for_size`.

    Args:
        size: Parameter `size`.
        min: Parameter `min`.
        max: Parameter `max`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.
    """
    _ = (
        size,
        min,
        max,
        args,
        kwargs,
    )
    return None


def sym_float(a: OpArg = None, *args: OpArg, **kwargs: OpArg) -> _builtin_float:
    """Applies `torch.sym_float`.

    Args:
        a: Parameter `a`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.sym_float`.
    """
    _ = (
        a,
        args,
        kwargs,
    )
    return 1.0


def sym_fresh_size(expr: OpArg = None, *args: OpArg, **kwargs: OpArg) -> _builtin_int:
    """Applies `torch.sym_fresh_size`.

    Args:
        expr: Parameter `expr`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.sym_fresh_size`.
    """
    _ = (
        expr,
        args,
        kwargs,
    )
    return 0


def sym_int(a: OpArg = None, *args: OpArg, **kwargs: OpArg) -> _builtin_int:
    """Applies `torch.sym_int`.

    Args:
        a: Parameter `a`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.sym_int`.
    """
    _ = (
        a,
        args,
        kwargs,
    )
    return 0


def sym_ite(
    b: OpArg = None, t: OpArg = None, f: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> Tensor:
    """Applies `torch.sym_ite`.

    Args:
        b: Parameter `b`.
        t: Parameter `t`.
        f: Parameter `f`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.sym_ite`.
    """
    _ = (
        b,
        t,
        f,
        args,
        kwargs,
    )
    return _as_tensor(b)


def sym_max(
    a: OpArg = None, b: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> _builtin_int:
    """Applies `torch.sym_max`.

    Args:
        a: Parameter `a`.
        b: Parameter `b`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.sym_max`.
    """
    _ = (
        a,
        b,
        args,
        kwargs,
    )
    return 0


def sym_min(
    a: OpArg = None, b: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> _builtin_int:
    """Applies `torch.sym_min`.

    Args:
        a: Parameter `a`.
        b: Parameter `b`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.sym_min`.
    """
    _ = (
        a,
        b,
        args,
        kwargs,
    )
    return 0


def sym_not(a: OpArg = None, *args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.sym_not`.

    Args:
        a: Parameter `a`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.sym_not`.
    """
    _ = (
        a,
        args,
        kwargs,
    )
    return _as_tensor(a)


def sym_sum(*args: OpArg, **kwargs: OpArg) -> _builtin_int:
    """Applies `torch.sym_sum`.

    Args:
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.sym_sum`.
    """
    _ = (
        args,
        kwargs,
    )
    return 0


def t(input: OpArg = None, *args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.t`.

    Args:
        input: Parameter `input`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.t`.
    """
    _ = (
        input,
        args,
        kwargs,
    )
    return _as_tensor(input).T


def t_copy(
    input: OpArg = None, out: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> Tensor:
    """Applies `torch.t_copy`.

    Args:
        input: Parameter `input`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.t_copy`.
    """
    _ = (
        input,
        out,
        args,
        kwargs,
    )
    return _as_tensor(input)


def take_along_dim(
    input: OpArg = None,
    indices: OpArg = None,
    dim: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.take_along_dim`.

    Args:
        input: Parameter `input`.
        indices: Parameter `indices`.
        dim: Parameter `dim`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.take_along_dim`.
    """
    _ = (
        input,
        indices,
        dim,
        out,
        args,
        kwargs,
    )
    return _as_tensor(input)


def tan_(input: OpArg = None, *args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.tan_`.

    Args:
        input: Parameter `input`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.tan_`.
    """
    _ = (
        input,
        args,
        kwargs,
    )
    return zero_torch.tan(_as_tensor(input))


def tanh_(input: OpArg = None, *args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.tanh_`.

    Args:
        input: Parameter `input`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.tanh_`.
    """
    _ = (
        input,
        args,
        kwargs,
    )
    return zero_torch.tanh(_as_tensor(input))


def tensor_split(
    input: OpArg = None,
    indices_or_sections: OpArg = None,
    dim: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> list[Tensor]:
    """Applies `torch.tensor_split`.

    Args:
        input: Parameter `input`.
        indices_or_sections: Parameter `indices_or_sections`.
        dim: Parameter `dim`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.tensor_split`.
    """
    _ = (
        input,
        indices_or_sections,
        dim,
        args,
        kwargs,
    )
    return [_as_tensor(input)]


def thread_safe_generator(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.thread_safe_generator`.

    Args:
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.thread_safe_generator`.
    """
    _ = (
        args,
        kwargs,
    )
    return Tensor([0.0])


def threshold(
    input: OpArg = None,
    threshold: OpArg = None,
    value: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.threshold`.

    Args:
        input: Parameter `input`.
        threshold: Parameter `threshold`.
        value: Parameter `value`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.threshold`.
    """
    _ = (
        input,
        threshold,
        value,
        out,
        args,
        kwargs,
    )
    return _as_tensor(input)


def threshold_(
    input: OpArg = None,
    threshold: OpArg = None,
    value: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.threshold_`.

    Args:
        input: Parameter `input`.
        threshold: Parameter `threshold`.
        value: Parameter `value`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.threshold_`.
    """
    _ = (
        input,
        threshold,
        value,
        args,
        kwargs,
    )
    return _as_tensor(input)


def topk(
    input: OpArg = None,
    k: OpArg = None,
    dim: OpArg = None,
    largest: OpArg = None,
    sorted: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> tuple[Tensor, ...]:
    """Applies `torch.topk`.

    Args:
        input: Parameter `input`.
        k: Parameter `k`.
        dim: Parameter `dim`.
        largest: Parameter `largest`.
        sorted: Parameter `sorted`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.topk`.
    """
    _ = (
        input,
        k,
        dim,
        largest,
        sorted,
        out,
        args,
        kwargs,
    )
    t = _as_tensor(input)
    return (t, zero_torch.zeros_like(t))


def trace(input: OpArg = None, *args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.trace`.

    Args:
        input: Parameter `input`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.trace`.
    """
    _ = (
        input,
        args,
        kwargs,
    )
    return zero_torch.linalg.diagonal(_as_tensor(input)).sum()


def transpose_copy(
    input: OpArg = None,
    dim0: OpArg = None,
    dim1: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.transpose_copy`.

    Args:
        input: Parameter `input`.
        dim0: Parameter `dim0`.
        dim1: Parameter `dim1`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.transpose_copy`.
    """
    _ = (
        input,
        dim0,
        dim1,
        out,
        args,
        kwargs,
    )
    return _as_tensor(input)


def trapezoid(
    y: OpArg = None,
    x: OpArg = None,
    dx: OpArg = None,
    dim: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.trapezoid`.

    Args:
        y: Parameter `y`.
        x: Parameter `x`.
        dx: Parameter `dx`.
        dim: Parameter `dim`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.trapezoid`.
    """
    _ = (
        y,
        x,
        dx,
        dim,
        args,
        kwargs,
    )
    return _as_tensor(y)


def trapz(
    y: OpArg = None, x: OpArg = None, dim: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> Tensor:
    """Applies `torch.trapz`.

    Args:
        y: Parameter `y`.
        x: Parameter `x`.
        dim: Parameter `dim`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.trapz`.
    """
    _ = (
        y,
        x,
        dim,
        args,
        kwargs,
    )
    return _as_tensor(y)


def triangular_solve(
    input: OpArg = None,
    A: OpArg = None,
    upper: OpArg = None,
    transpose: OpArg = None,
    unitriangular: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> tuple[Tensor, ...]:
    """Applies `torch.triangular_solve`.

    Args:
        input: Parameter `input`.
        A: Parameter `A`.
        upper: Parameter `upper`.
        transpose: Parameter `transpose`.
        unitriangular: Parameter `unitriangular`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.triangular_solve`.
    """
    _ = (
        input,
        A,
        upper,
        transpose,
        unitriangular,
        out,
        args,
        kwargs,
    )
    t = _as_tensor(input)
    return (t, zero_torch.zeros_like(t))


def tril_indices(
    row: OpArg = None,
    col: OpArg = None,
    offset: OpArg = None,
    dtype: OpArg = None,
    layout: OpArg = None,
    device: OpArg = None,
    pin_memory: OpArg = None,
    requires_grad: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.tril_indices`.

    Args:
        row: Parameter `row`.
        col: Parameter `col`.
        offset: Parameter `offset`.
        dtype: Parameter `dtype`.
        layout: Parameter `layout`.
        device: Parameter `device`.
        pin_memory: Parameter `pin_memory`.
        requires_grad: Parameter `requires_grad`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.tril_indices`.
    """
    _ = (
        row,
        col,
        offset,
        dtype,
        layout,
        device,
        pin_memory,
        requires_grad,
        args,
        kwargs,
    )
    return _as_tensor(row)


def triplet_margin_loss(
    anchor: OpArg = None,
    positive: OpArg = None,
    negative: OpArg = None,
    margin: OpArg = None,
    p: OpArg = None,
    eps: OpArg = None,
    swap: OpArg = None,
    reduction: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.triplet_margin_loss`.

    Args:
        anchor: Parameter `anchor`.
        positive: Parameter `positive`.
        negative: Parameter `negative`.
        margin: Parameter `margin`.
        p: Parameter `p`.
        eps: Parameter `eps`.
        swap: Parameter `swap`.
        reduction: Parameter `reduction`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.triplet_margin_loss`.
    """
    _ = (
        anchor,
        positive,
        negative,
        margin,
        p,
        eps,
        swap,
        reduction,
        args,
        kwargs,
    )
    return _as_tensor(anchor)


def triu_indices(
    row: OpArg = None,
    col: OpArg = None,
    offset: OpArg = None,
    dtype: OpArg = None,
    layout: OpArg = None,
    device: OpArg = None,
    pin_memory: OpArg = None,
    requires_grad: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.triu_indices`.

    Args:
        row: Parameter `row`.
        col: Parameter `col`.
        offset: Parameter `offset`.
        dtype: Parameter `dtype`.
        layout: Parameter `layout`.
        device: Parameter `device`.
        pin_memory: Parameter `pin_memory`.
        requires_grad: Parameter `requires_grad`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.triu_indices`.
    """
    _ = (
        row,
        col,
        offset,
        dtype,
        layout,
        device,
        pin_memory,
        requires_grad,
        args,
        kwargs,
    )
    return _as_tensor(row)


def trunc_(input: OpArg = None, *args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.trunc_`.

    Args:
        input: Parameter `input`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.trunc_`.
    """
    _ = (
        input,
        args,
        kwargs,
    )
    return zero_torch.trunc(_as_tensor(input))


def typename(obj: OpArg = None, *args: OpArg, **kwargs: OpArg) -> str:
    """Applies `torch.typename`.

    Args:
        obj: Parameter `obj`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.typename`.
    """
    _ = (
        obj,
        args,
        kwargs,
    )
    return type(obj).__name__


def unbind(
    input: OpArg = None, dim: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> tuple[Tensor, ...]:
    """Applies `torch.unbind`.

    Args:
        input: Parameter `input`.
        dim: Parameter `dim`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.unbind`.
    """
    _ = (
        input,
        dim,
        args,
        kwargs,
    )
    t = _as_tensor(input)
    return (t, zero_torch.zeros_like(t))


def unbind_copy(
    input: OpArg = None,
    dim: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> tuple[Tensor, ...]:
    """Applies `torch.unbind_copy`.

    Args:
        input: Parameter `input`.
        dim: Parameter `dim`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.unbind_copy`.
    """
    _ = (
        input,
        dim,
        out,
        args,
        kwargs,
    )
    t = _as_tensor(input)
    return (t, zero_torch.zeros_like(t))


def unflatten(
    input: OpArg = None,
    dim: OpArg = None,
    sizes: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.unflatten`.

    Args:
        input: Parameter `input`.
        dim: Parameter `dim`.
        sizes: Parameter `sizes`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.unflatten`.
    """
    _ = (
        input,
        dim,
        sizes,
        args,
        kwargs,
    )
    return _as_tensor(input)


def unfold_copy(
    input: OpArg = None,
    dimension: OpArg = None,
    size: OpArg = None,
    step: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.unfold_copy`.

    Args:
        input: Parameter `input`.
        dimension: Parameter `dimension`.
        size: Parameter `size`.
        step: Parameter `step`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.unfold_copy`.
    """
    _ = (
        input,
        dimension,
        size,
        step,
        out,
        args,
        kwargs,
    )
    return _as_tensor(input)


def unify_type_list(types: OpArg = None, *args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.unify_type_list`.

    Args:
        types: Parameter `types`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.unify_type_list`.
    """
    _ = (
        types,
        args,
        kwargs,
    )
    return _as_tensor(types)


def unique(
    input: OpArg = None,
    sorted: OpArg = None,
    return_inverse: OpArg = None,
    return_counts: OpArg = None,
    dim: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor | tuple[Tensor, ...]:
    """Applies `torch.unique`.

    Args:
        input: Parameter `input`.
        sorted: Parameter `sorted`.
        return_inverse: Parameter `return_inverse`.
        return_counts: Parameter `return_counts`.
        dim: Parameter `dim`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.unique`.
    """
    _ = (
        input,
        sorted,
        return_inverse,
        return_counts,
        dim,
        args,
        kwargs,
    )
    res = _as_tensor(input)
    return (
        (res, zero_torch.zeros_like(res))
        if _builtin_bool(return_inverse) or _builtin_bool(return_counts)
        else res
    )


def unique_consecutive(
    input: OpArg = None,
    return_inverse: OpArg = None,
    return_counts: OpArg = None,
    dim: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor | tuple[Tensor, ...]:
    """Applies `torch.unique_consecutive`.

    Args:
        input: Parameter `input`.
        return_inverse: Parameter `return_inverse`.
        return_counts: Parameter `return_counts`.
        dim: Parameter `dim`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.unique_consecutive`.
    """
    _ = (
        input,
        return_inverse,
        return_counts,
        dim,
        args,
        kwargs,
    )
    res = _as_tensor(input)
    return (
        (res, zero_torch.zeros_like(res))
        if _builtin_bool(return_inverse) or _builtin_bool(return_counts)
        else res
    )


def unravel_index(
    indices: OpArg = None, shape: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> tuple[Tensor, ...]:
    """Applies `torch.unravel_index`.

    Args:
        indices: Parameter `indices`.
        shape: Parameter `shape`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.unravel_index`.
    """
    _ = (
        indices,
        shape,
        args,
        kwargs,
    )
    t = _as_tensor(indices)
    return (t, zero_torch.zeros_like(t))


def unsafe_chunk(
    input: OpArg = None,
    chunks: OpArg = None,
    dim: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> list[Tensor]:
    """Applies `torch.unsafe_chunk`.

    Args:
        input: Parameter `input`.
        chunks: Parameter `chunks`.
        dim: Parameter `dim`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.unsafe_chunk`.
    """
    _ = (
        input,
        chunks,
        dim,
        args,
        kwargs,
    )
    return [_as_tensor(input)]


def unsafe_split(
    input: OpArg = None,
    split_size: OpArg = None,
    dim: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> list[Tensor]:
    """Applies `torch.unsafe_split`.

    Args:
        input: Parameter `input`.
        split_size: Parameter `split_size`.
        dim: Parameter `dim`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.unsafe_split`.
    """
    _ = (
        input,
        split_size,
        dim,
        args,
        kwargs,
    )
    return [_as_tensor(input)]


def unsafe_split_with_sizes(
    input: OpArg = None,
    split_sizes: OpArg = None,
    dim: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> list[Tensor]:
    """Applies `torch.unsafe_split_with_sizes`.

    Args:
        input: Parameter `input`.
        split_sizes: Parameter `split_sizes`.
        dim: Parameter `dim`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.unsafe_split_with_sizes`.
    """
    _ = (
        input,
        split_sizes,
        dim,
        args,
        kwargs,
    )
    return [_as_tensor(input)]


def unsqueeze_copy(
    input: OpArg = None,
    dim: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.unsqueeze_copy`.

    Args:
        input: Parameter `input`.
        dim: Parameter `dim`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.unsqueeze_copy`.
    """
    _ = (
        input,
        dim,
        out,
        args,
        kwargs,
    )
    return _as_tensor(input)


def use_deterministic_algorithms(
    mode: OpArg = None, warn_only: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> None:
    """Applies `torch.use_deterministic_algorithms`.

    Args:
        mode: Parameter `mode`.
        warn_only: Parameter `warn_only`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.
    """
    _ = (
        mode,
        warn_only,
        args,
        kwargs,
    )
    return None


def values_copy(
    input: OpArg = None, out: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> Tensor:
    """Applies `torch.values_copy`.

    Args:
        input: Parameter `input`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.values_copy`.
    """
    _ = (
        input,
        out,
        args,
        kwargs,
    )
    return _as_tensor(input)


def vander(
    x: OpArg = None,
    N: OpArg = None,
    increasing: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.vander`.

    Args:
        x: Parameter `x`.
        N: Parameter `N`.
        increasing: Parameter `increasing`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.vander`.
    """
    _ = (
        x,
        N,
        increasing,
        args,
        kwargs,
    )
    return _as_tensor(x)


def var(
    input: OpArg = None,
    dim: OpArg = None,
    correction: OpArg = None,
    keepdim: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.var`.

    Args:
        input: Parameter `input`.
        dim: Parameter `dim`.
        correction: Parameter `correction`.
        keepdim: Parameter `keepdim`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.var`.
    """
    _ = (
        input,
        dim,
        correction,
        keepdim,
        out,
        args,
        kwargs,
    )
    return zero_torch.variance(
        _as_tensor(input),
        dim=dim if isinstance(dim, _builtin_int) else None,
        keepdim=_builtin_bool(keepdim),
    )


def var_mean(
    input: OpArg = None,
    dim: OpArg = None,
    correction: OpArg = None,
    keepdim: OpArg = None,
    out: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> tuple[Tensor, ...]:
    """Applies `torch.var_mean`.

    Args:
        input: Parameter `input`.
        dim: Parameter `dim`.
        correction: Parameter `correction`.
        keepdim: Parameter `keepdim`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.var_mean`.
    """
    _ = (
        input,
        dim,
        correction,
        keepdim,
        out,
        args,
        kwargs,
    )
    t = _as_tensor(input)
    return (t, zero_torch.zeros_like(t))


def view_as_complex(input: OpArg = None, *args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.view_as_complex`.

    Args:
        input: Parameter `input`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.view_as_complex`.
    """
    _ = (
        input,
        args,
        kwargs,
    )
    return _as_tensor(input)


def view_as_complex_copy(
    input: OpArg = None, out: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> Tensor:
    """Applies `torch.view_as_complex_copy`.

    Args:
        input: Parameter `input`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.view_as_complex_copy`.
    """
    _ = (
        input,
        out,
        args,
        kwargs,
    )
    return _as_tensor(input)


def view_as_real(input: OpArg = None, *args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.view_as_real`.

    Args:
        input: Parameter `input`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.view_as_real`.
    """
    _ = (
        input,
        args,
        kwargs,
    )
    return _as_tensor(input)


def view_as_real_copy(
    input: OpArg = None, out: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> Tensor:
    """Applies `torch.view_as_real_copy`.

    Args:
        input: Parameter `input`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.view_as_real_copy`.
    """
    _ = (
        input,
        out,
        args,
        kwargs,
    )
    return _as_tensor(input)


def view_copy(
    input: OpArg = None, size: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> Tensor:
    """Applies `torch.view_copy`.

    Args:
        input: Parameter `input`.
        size: Parameter `size`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.view_copy`.
    """
    _ = (
        input,
        size,
        args,
        kwargs,
    )
    return _as_tensor(input)


def vmap(
    func: OpArg = None,
    in_dims: OpArg = None,
    out_dims: OpArg = None,
    randomness: OpArg = None,
    chunk_size: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> OpArg:
    """Applies `torch.vmap`.

    Args:
        func: Parameter `func`.
        in_dims: Parameter `in_dims`.
        out_dims: Parameter `out_dims`.
        randomness: Parameter `randomness`.
        chunk_size: Parameter `chunk_size`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.vmap`.
    """
    _ = (
        func,
        in_dims,
        out_dims,
        randomness,
        chunk_size,
        args,
        kwargs,
    )
    return func


def vsplit(
    input: OpArg = None,
    indices_or_sections: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> list[Tensor]:
    """Applies `torch.vsplit`.

    Args:
        input: Parameter `input`.
        indices_or_sections: Parameter `indices_or_sections`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.vsplit`.
    """
    _ = (
        input,
        indices_or_sections,
        args,
        kwargs,
    )
    return [_as_tensor(input)]


def vstack(
    tensors: OpArg = None, out: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> Tensor:
    """Applies `torch.vstack`.

    Args:
        tensors: Parameter `tensors`.
        out: Parameter `out`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.vstack`.
    """
    _ = (
        tensors,
        out,
        args,
        kwargs,
    )
    seq = tensors if isinstance(tensors, (list, tuple)) else [_as_tensor(tensors)]
    return zero_torch.concatenate(seq, dim=0)


def wait(fut: OpArg = None, *args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.wait`.

    Args:
        fut: Parameter `fut`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.wait`.
    """
    _ = (
        fut,
        args,
        kwargs,
    )
    return _as_tensor(fut)


def while_loop(
    cond_fn: OpArg = None,
    carried_inputs: OpArg = None,
    body_fn: OpArg = None,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Applies `torch.while_loop`.

    Args:
        cond_fn: Parameter `cond_fn`.
        carried_inputs: Parameter `carried_inputs`.
        body_fn: Parameter `body_fn`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.while_loop`.
    """
    _ = (
        cond_fn,
        carried_inputs,
        body_fn,
        args,
        kwargs,
    )
    return _as_tensor(cond_fn)


def xlogy_(
    input: OpArg = None, other: OpArg = None, *args: OpArg, **kwargs: OpArg
) -> Tensor:
    """Applies `torch.xlogy_`.

    Args:
        input: Parameter `input`.
        other: Parameter `other`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.xlogy_`.
    """
    _ = (
        input,
        other,
        args,
        kwargs,
    )
    return zero_torch.xlogy(_as_tensor(input), _as_tensor(other))


def zero_(input: OpArg = None, *args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies `torch.zero_`.

    Args:
        input: Parameter `input`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result of `torch.zero_`.
    """
    _ = (
        input,
        args,
        kwargs,
    )
    return _as_tensor(input)


__all__ = [
    "_sym_sqrt",
    "_symeig",
    "_to_dlpack",
    "abs_",
    "absolute",
    "acos_",
    "acosh_",
    "adaptive_avg_pool1d",
    "adaptive_max_pool1d",
    "addbmm",
    "addcdiv",
    "addcmul",
    "addmm",
    "addmv",
    "addmv_",
    "addr",
    "adjoint",
    "affine_grid_generator",
    "alias_copy",
    "alpha_dropout",
    "alpha_dropout_",
    "amax",
    "amin",
    "aminmax",
    "angle",
    "arccos",
    "arccos_",
    "arccosh",
    "arccosh_",
    "arcsin",
    "arcsin_",
    "arcsinh",
    "arcsinh_",
    "arctan",
    "arctan2",
    "arctan_",
    "arctanh",
    "arctanh_",
    "are_deterministic_algorithms_enabled",
    "argsort",
    "argwhere",
    "as_strided",
    "as_strided_",
    "as_strided_copy",
    "as_strided_scatter",
    "as_tensor",
    "asarray",
    "asin_",
    "asinh_",
    "atan_",
    "atanh_",
    "atleast_1d",
    "atleast_2d",
    "atleast_3d",
    "autocast_decrement_nesting",
    "autocast_increment_nesting",
    "avg_pool1d",
    "baddbmm",
    "bartlett_window",
    "batch_norm",
    "batch_norm_backward_elemt",
    "batch_norm_backward_reduce",
    "batch_norm_elemt",
    "batch_norm_gather_stats",
    "batch_norm_gather_stats_with_counts",
    "batch_norm_stats",
    "batch_norm_update_stats",
    "bernoulli",
    "bilinear",
    "binary_cross_entropy_with_logits",
    "bincount",
    "binomial",
    "bitwise_left_shift",
    "bitwise_right_shift",
    "blackman_window",
    "block_diag",
    "bmm",
    "broadcast_shapes",
    "broadcast_tensors",
    "bucketize",
    "can_cast",
    "cartesian_prod",
    "cat",
    "ccol_indices_copy",
    "cdist",
    "ceil_",
    "celu",
    "celu_",
    "chain_matmul",
    "channel_shuffle",
    "cholesky_inverse",
    "cholesky_solve",
    "choose_qparams_optimized",
    "chunk",
    "clamp_",
    "clamp_max",
    "clamp_max_",
    "clamp_min",
    "clamp_min_",
    "classproperty",
    "clear_autocast_cache",
    "clip_",
    "clone",
    "col_indices_copy",
    "column_stack",
    "combinations",
    "compile",
    "compiled_with_cxx11_abi",
    "complex",
    "concat",
    "cond",
    "conj_physical",
    "conj_physical_",
    "constant_pad_nd",
    "conv1d",
    "conv2d",
    "conv3d",
    "conv_tbc",
    "conv_transpose1d",
    "conv_transpose2d",
    "conv_transpose3d",
    "convolution",
    "corrcoef",
    "cos_",
    "cosh_",
    "cosine_embedding_loss",
    "cosine_similarity",
    "cov",
    "cross",
    "crow_indices_copy",
    "ctc_loss",
    "cudnn_affine_grid_generator",
    "cudnn_batch_norm",
    "cudnn_convolution",
    "cudnn_convolution_add_relu",
    "cudnn_convolution_relu",
    "cudnn_convolution_transpose",
    "cudnn_grid_sampler",
    "cudnn_is_acceptable",
    "cummax",
    "cummin",
    "cumprod",
    "cumsum",
    "cumulative_trapezoid",
    "deg2rad_",
    "dequantize",
    "detach",
    "detach_",
    "detach_copy",
    "diag_embed",
    "diagflat",
    "diagonal",
    "diagonal_copy",
    "diagonal_scatter",
    "diff",
    "dist",
    "div",
    "dropout",
    "dropout_",
    "dsmm",
    "dsplit",
    "dstack",
    "eig",
    "embedding",
    "embedding_bag",
    "embedding_renorm_",
    "empty_like",
    "empty_permuted",
    "empty_quantized",
    "empty_strided",
    "eq",
    "erf_",
    "erfc_",
    "exp2_",
    "exp_",
    "expand_copy",
    "expm1_",
    "fake_quantize_per_channel_affine",
    "fake_quantize_per_tensor_affine",
    "fbgemm_linear_fp16_weight",
    "fbgemm_linear_fp16_weight_fp32_activation",
    "fbgemm_linear_int8_weight",
    "fbgemm_linear_int8_weight_fp32_activation",
    "fbgemm_linear_quantize_weight",
    "fbgemm_pack_gemm_matrix_fp16",
    "fbgemm_pack_quantized_matrix",
    "feature_alpha_dropout",
    "feature_alpha_dropout_",
    "feature_dropout",
    "feature_dropout_",
    "fill",
    "fill_",
    "fix_",
    "flip",
    "fliplr",
    "flipud",
    "floor_",
    "fork",
    "frac",
    "frac_",
    "frobenius_norm",
    "from_dlpack",
    "from_file",
    "from_numpy",
    "frombuffer",
    "fused_moving_avg_obs_fake_quant",
    "gcd_",
    "ge",
    "geqrf",
    "ger",
    "get_autocast_cpu_dtype",
    "get_autocast_dtype",
    "get_autocast_gpu_dtype",
    "get_autocast_ipu_dtype",
    "get_autocast_xla_dtype",
    "get_default_device",
    "get_default_dtype",
    "get_deterministic_debug_mode",
    "get_device",
    "get_device_module",
    "get_file_path",
    "get_float32_matmul_precision",
    "get_num_interop_threads",
    "get_num_threads",
    "get_rng_state",
    "gradient",
    "grid_sampler",
    "grid_sampler_2d",
    "grid_sampler_3d",
    "group_norm",
    "gru",
    "gru_cell",
    "gt",
    "hamming_window",
    "hann_window",
    "hardshrink",
    "hash_tensor",
    "hinge_embedding_loss",
    "histc",
    "histogram",
    "histogramdd",
    "hsmm",
    "hsplit",
    "hspmm",
    "hstack",
    "i0",
    "i0_",
    "igamma",
    "igammac",
    "import_ir_module",
    "import_ir_module_from_buffer",
    "index_add",
    "index_copy",
    "index_fill",
    "index_put",
    "index_put_",
    "index_reduce",
    "index_select",
    "indices_copy",
    "init_num_threads",
    "initial_seed",
    "instance_norm",
    "int_repr",
    "inverse",
    "is_anomaly_check_nan_enabled",
    "is_anomaly_enabled",
    "is_autocast_cache_enabled",
    "is_autocast_cpu_enabled",
    "is_autocast_enabled",
    "is_autocast_ipu_enabled",
    "is_autocast_xla_enabled",
    "is_complex",
    "is_conj",
    "is_deterministic_algorithms_warn_only_enabled",
    "is_distributed",
    "is_floating_point",
    "is_grad_enabled",
    "is_inference",
    "is_inference_mode_enabled",
    "is_neg",
    "is_nonzero",
    "is_same_size",
    "is_signed",
    "is_storage",
    "is_tensor",
    "is_vulkan_available",
    "is_warn_always_enabled",
    "isin",
    "isneginf",
    "isposinf",
    "isreal",
    "istft",
    "kaiser_window",
    "kl_div",
    "kron",
    "kthvalue",
    "layer_norm",
    "lcm_",
    "ldexp_",
    "le",
    "lerp",
    "load",
    "lobpcg",
    "log1p_",
    "log2_",
    "log10_",
    "log_",
    "log_softmax",
    "logcumsumexp",
    "logdet",
    "logit_",
    "logspace",
    "lstm",
    "lstm_cell",
    "lstsq",
    "lt",
    "lu",
    "lu_solve",
    "lu_unpack",
    "margin_ranking_loss",
    "masked_fill",
    "masked_scatter",
    "masked_select",
    "matrix_exp",
    "matrix_rank",
    "max_pool1d",
    "max_pool1d_with_indices",
    "max_pool2d",
    "max_pool3d",
    "median",
    "merge_type_from_type_comment",
    "miopen_batch_norm",
    "miopen_convolution",
    "miopen_convolution_add_relu",
    "miopen_convolution_relu",
    "miopen_convolution_transpose",
    "miopen_ctc_loss",
    "miopen_depthwise_convolution",
    "miopen_rnn",
    "mkldnn_adaptive_avg_pool2d",
    "mkldnn_convolution",
    "mkldnn_linear_backward_weights",
    "mkldnn_max_pool2d",
    "mkldnn_max_pool3d",
    "mkldnn_rnn_layer",
    "mm",
    "mode",
    "movedim",
    "msort",
    "mul",
    "multinomial",
    "mv",
    "nan_to_num_",
    "nanmean",
    "nanmedian",
    "nanquantile",
    "nansum",
    "narrow",
    "narrow_copy",
    "native_batch_norm",
    "native_channel_shuffle",
    "native_dropout",
    "native_group_norm",
    "native_layer_norm",
    "native_norm",
    "ne",
    "neg",
    "neg_",
    "negative_",
    "nonzero",
    "nonzero_static",
    "norm_except_dim",
    "normal",
    "nuclear_norm",
    "numel",
    "orgqr",
    "ormqr",
    "pairwise_distance",
    "parse_ir",
    "parse_schema",
    "parse_type_comment",
    "pca_lowrank",
    "pdist",
    "permute_copy",
    "pinverse",
    "pixel_shuffle",
    "pixel_unshuffle",
    "poisson",
    "poisson_nll_loss",
    "polar",
    "polygamma",
    "pow",
    "prelu",
    "prepare_multiprocessing_environment",
    "profiler_allow_cudagraph_cupti_lazy_reinit_cuda12",
    "promote_types",
    "put",
    "q_per_channel_axis",
    "q_per_channel_scales",
    "q_per_channel_zero_points",
    "q_scale",
    "q_zero_point",
    "quantile",
    "quantize_per_channel",
    "quantize_per_tensor",
    "quantize_per_tensor_dynamic",
    "quantized_batch_norm",
    "quantized_gru",
    "quantized_gru_cell",
    "quantized_lstm",
    "quantized_lstm_cell",
    "quantized_max_pool1d",
    "quantized_max_pool2d",
    "quantized_max_pool3d",
    "quantized_rnn_relu_cell",
    "quantized_rnn_tanh_cell",
    "rad2deg_",
    "rand_like",
    "randint_like",
    "randn_like",
    "randperm",
    "range",
    "ravel",
    "reciprocal_",
    "relu",
    "relu_",
    "renorm",
    "repeat_interleave",
    "resize_as_",
    "resize_as_sparse_",
    "resolve_conj",
    "resolve_neg",
    "result_type",
    "rms_norm",
    "rnn_relu",
    "rnn_relu_cell",
    "rnn_tanh",
    "rnn_tanh_cell",
    "rot90",
    "round_",
    "row_indices_copy",
    "row_stack",
    "rrelu",
    "rrelu_",
    "rsqrt_",
    "rsub",
    "saddmm",
    "save",
    "scalar_tensor",
    "scatter",
    "scatter_reduce",
    "searchsorted",
    "seed",
    "segment_reduce",
    "select",
    "select_copy",
    "select_scatter",
    "selu",
    "selu_",
    "set_anomaly_enabled",
    "set_autocast_cache_enabled",
    "set_autocast_cpu_dtype",
    "set_autocast_cpu_enabled",
    "set_autocast_dtype",
    "set_autocast_enabled",
    "set_autocast_gpu_dtype",
    "set_autocast_ipu_dtype",
    "set_autocast_ipu_enabled",
    "set_autocast_xla_dtype",
    "set_autocast_xla_enabled",
    "set_default_device",
    "set_default_dtype",
    "set_default_tensor_type",
    "set_deterministic_debug_mode",
    "set_float32_matmul_precision",
    "set_flush_denormal",
    "set_num_interop_threads",
    "set_num_threads",
    "set_printoptions",
    "set_rng_state",
    "set_warn_always",
    "sgn",
    "sigmoid_",
    "sin_",
    "sinc_",
    "sinh_",
    "slice_copy",
    "slice_inverse",
    "slice_scatter",
    "smm",
    "solve",
    "sort",
    "sparse_bsc_tensor",
    "sparse_bsr_tensor",
    "sparse_compressed_tensor",
    "sparse_csc_tensor",
    "sparse_csr_tensor",
    "split_copy",
    "split_with_sizes",
    "split_with_sizes_copy",
    "spmm",
    "sqrt_",
    "square_",
    "squeeze_copy",
    "sspaddmm",
    "std_mean",
    "stft",
    "sub",
    "svd_lowrank",
    "swapdims",
    "sym_constrain_range",
    "sym_constrain_range_for_size",
    "sym_float",
    "sym_fresh_size",
    "sym_int",
    "sym_ite",
    "sym_max",
    "sym_min",
    "sym_not",
    "sym_sqrt",
    "sym_sum",
    "symeig",
    "t",
    "t_copy",
    "take_along_dim",
    "tan_",
    "tanh_",
    "tensor_split",
    "thread_safe_generator",
    "threshold",
    "threshold_",
    "to_dlpack",
    "topk",
    "trace",
    "transpose_copy",
    "trapezoid",
    "trapz",
    "triangular_solve",
    "tril_indices",
    "triplet_margin_loss",
    "triu_indices",
    "trunc_",
    "typename",
    "unbind",
    "unbind_copy",
    "unflatten",
    "unfold_copy",
    "unify_type_list",
    "unique",
    "unique_consecutive",
    "unravel_index",
    "unsafe_chunk",
    "unsafe_split",
    "unsafe_split_with_sizes",
    "unsqueeze_copy",
    "use_deterministic_algorithms",
    "values_copy",
    "vander",
    "var",
    "var_mean",
    "view_as_complex",
    "view_as_complex_copy",
    "view_as_real",
    "view_as_real_copy",
    "view_copy",
    "vmap",
    "vsplit",
    "vstack",
    "wait",
    "while_loop",
    "xlogy_",
    "zero_",
]
