"""Pytest configuration and shared pure-Python assertion/tensor helpers for zero_torch tests."""

from __future__ import annotations

import cmath
import math
from collections.abc import Iterator, Sequence

import pytest
from ml_switcheroo_compiler.core.config import EagerMode
from ml_switcheroo_compiler.core.dtype import DType

from zero_torch._types import NestedList, NestedSequence, OpArg, Scalar


def flatten_scalars(data: NestedSequence | NestedList | Scalar) -> list[Scalar]:
    """Flattens a nested sequence or scalar into a 1D list of scalars.

    Args:
        data: Nested list/sequence of scalars or a single scalar value.

    Returns:
        Flat list of scalar values.
    """
    if isinstance(data, (int, float, bool, complex)):
        return [data]
    result: list[Scalar] = []
    for item in data:
        result.extend(flatten_scalars(item))
    return result


def eye_nested(n: int) -> NestedList:
    """Constructs a 2D `n x n` identity matrix as a nested Python list of floats.

    Args:
        n: Dimension of the square identity matrix.

    Returns:
        2D nested list with `1.0` on the diagonal and `0.0` elsewhere.
    """
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def linspace_list(start: float, end: float, steps: int) -> list[float]:
    """Constructs `steps` evenly spaced floats from `start` to `end` inclusive.

    Args:
        start: Starting scalar value.
        end: Ending scalar value.
        steps: Number of points to generate.

    Returns:
        List of evenly spaced float values.
    """
    if steps <= 0:
        return []
    if steps == 1:
        return [float(start)]
    step_size = (end - start) / (steps - 1)
    return [float(start + i * step_size) for i in range(steps)]


def _scalars_allclose(
    actual: Scalar,
    expected: Scalar,
    rtol: float,
    atol: float,
) -> bool:
    """Checks whether two scalar values are equal within relative and absolute tolerances.

    Args:
        actual: Actual scalar value.
        expected: Expected scalar value.
        rtol: Relative tolerance.
        atol: Absolute tolerance.

    Returns:
        True if `actual` and `expected` are close within tolerance, False otherwise.
    """
    if isinstance(actual, complex) or isinstance(expected, complex):
        a_c = complex(actual)
        e_c = complex(expected)
        if cmath.isnan(a_c) and cmath.isnan(e_c):
            return True
        if cmath.isinf(a_c) or cmath.isinf(e_c):
            return a_c == e_c
        return cmath.isclose(a_c, e_c, rel_tol=rtol, abs_tol=atol)

    a_f = float(actual)
    e_f = float(expected)
    if math.isnan(a_f) and math.isnan(e_f):
        return True
    if math.isinf(a_f) or math.isinf(e_f):
        return a_f == e_f
    return math.isclose(a_f, e_f, rel_tol=rtol, abs_tol=atol)


def assert_allclose(
    actual: NestedSequence | NestedList | Scalar,
    expected: NestedSequence | NestedList | Scalar,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> None:
    """Asserts that two scalars or nested sequences are element-wise close within tolerance.

    Supports scalar-to-sequence broadcasting when either `actual` or `expected` is a scalar.

    Args:
        actual: Actual scalar or nested sequence (e.g. from `tensor.tolist()`).
        expected: Expected scalar or nested sequence.
        rtol: Relative tolerance (default `1e-5`).
        atol: Absolute tolerance (default `1e-8`).
    """
    if isinstance(actual, (int, float, bool, complex)) and isinstance(
        expected, (int, float, bool, complex)
    ):
        if not _scalars_allclose(actual, expected, rtol=rtol, atol=atol):
            raise AssertionError(
                f"Scalars not close: actual={actual!r}, expected={expected!r} "
                f"(rtol={rtol}, atol={atol})"
            )
        return

    if isinstance(actual, (int, float, bool, complex)) and isinstance(
        expected, Sequence
    ):
        for item in expected:
            assert_allclose(actual, item, rtol=rtol, atol=atol)
        return

    if isinstance(actual, Sequence) and isinstance(
        expected, (int, float, bool, complex)
    ):
        for item in actual:
            assert_allclose(item, expected, rtol=rtol, atol=atol)
        return

    if isinstance(actual, Sequence) and isinstance(expected, Sequence):
        if len(actual) != len(expected):
            raise AssertionError(
                f"Sequence length mismatch: len(actual)={len(actual)} != "
                f"len(expected)={len(expected)} (actual={actual!r}, expected={expected!r})"
            )
        for a_item, e_item in zip(actual, expected):
            assert_allclose(a_item, e_item, rtol=rtol, atol=atol)


def assert_op_output(res: OpArg) -> None:
    """Asserts that an operation return value is a well-formed Tensor, container of Tensors, or valid primitive/None.

    Args:
        res: Operation return value to validate.
    """
    from zero_torch.tensor import Tensor

    if isinstance(res, Tensor):
        assert isinstance(res.shape, tuple) and res.dtype is not None
    elif isinstance(res, (tuple, list)):
        assert len(res) >= 0
    else:
        assert res is None or isinstance(res, (int, float, bool, str, DType))


@pytest.fixture(autouse=True)
def switcheroo_config() -> Iterator[None]:
    """Configures `EagerMode` by default and resets the tracing state after each test.

    Yields:
        None during test execution.
    """
    with EagerMode():
        try:
            yield
        finally:
            from zero_torch.tracing import _tracer

            if _tracer.is_tracing:
                _tracer.stop_tracing()
