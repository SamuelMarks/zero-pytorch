"""Domain module for global runtime state, RNG, threads, and deterministic mode."""

from __future__ import annotations

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType

from zero_torch._types import OpArg

_RUNTIME_STATE: dict[str, int | bool | str | DType] = {
    "default_device": "cpu",
    "default_dtype": DType.Float32,
    "deterministic_debug_mode": "default",
    "deterministic_algorithms": False,
    "num_threads": 1,
    "num_interop_threads": 1,
    "anomaly_enabled": False,
}


def _get_default_device() -> str:
    """Returns the current default device string.

    Returns:
        Default device string (e.g. `'cpu'`).
    """
    val = _RUNTIME_STATE["default_device"]
    return str(val)


def _set_default_device(device: OpArg) -> None:
    """Sets the default device string.

    Args:
        device: Target device identifier.
    """
    _RUNTIME_STATE["default_device"] = str(device) if device is not None else "cpu"


def _get_default_dtype() -> DType:
    """Returns the current default floating-point `DType`.

    Returns:
        Default `DType`.
    """
    val = _RUNTIME_STATE["default_dtype"]
    return val if isinstance(val, DType) else DType.Float32


def _set_default_dtype(d: OpArg) -> None:
    """Sets the default floating-point `DType`.

    Args:
        d: Target `DType` or dtype string.
    """
    if isinstance(d, DType):
        _RUNTIME_STATE["default_dtype"] = d
        config.default_float_dtype = d
    else:
        try:
            dt = DType(str(d))
            _RUNTIME_STATE["default_dtype"] = dt
            config.default_float_dtype = dt
        except ValueError:
            _RUNTIME_STATE["default_dtype"] = DType.Float32


def _get_deterministic_debug_mode() -> str:
    """Returns the current deterministic debug mode (`'default'`, `'warn'`, `'error'`).

    Returns:
        Deterministic debug mode string.
    """
    return str(_RUNTIME_STATE["deterministic_debug_mode"])


def _set_deterministic_debug_mode(debug_mode: OpArg) -> None:
    """Sets the deterministic debug mode.

    Args:
        debug_mode: Mode int (`0`, `1`, `2`) or string (`'default'`, `'warn'`, `'error'`).
    """
    if debug_mode in (1, "warn"):
        _RUNTIME_STATE["deterministic_debug_mode"] = "warn"
    elif debug_mode in (2, "error"):
        _RUNTIME_STATE["deterministic_debug_mode"] = "error"
    else:
        _RUNTIME_STATE["deterministic_debug_mode"] = "default"


def _are_deterministic_algorithms_enabled() -> bool:
    """Returns True if deterministic algorithms mode is enabled.

    Returns:
        Boolean flag.
    """
    return bool(_RUNTIME_STATE["deterministic_algorithms"])


def _use_deterministic_algorithms(mode: OpArg) -> None:
    """Enables or disables deterministic algorithms mode.

    Args:
        mode: Boolean flag.
    """
    _RUNTIME_STATE["deterministic_algorithms"] = bool(mode)


def _get_num_threads() -> int:
    """Returns the number of intra-op parallelism threads.

    Returns:
        Thread count (`>= 1`).
    """
    val = _RUNTIME_STATE["num_threads"]
    return int(val) if isinstance(val, int) else 1


def _set_num_threads(num: OpArg) -> None:
    """Sets the number of intra-op parallelism threads.

    Args:
        num: Positive integer thread count.
    """
    _RUNTIME_STATE["num_threads"] = num if isinstance(num, int) and num >= 1 else 1


def _get_num_interop_threads() -> int:
    """Returns the number of inter-op parallelism threads.

    Returns:
        Inter-op thread count (`>= 1`).
    """
    val = _RUNTIME_STATE["num_interop_threads"]
    return int(val) if isinstance(val, int) else 1


def _set_num_interop_threads(num: OpArg) -> None:
    """Sets the number of inter-op parallelism threads.

    Args:
        num: Positive integer thread count.
    """
    _RUNTIME_STATE["num_interop_threads"] = (
        num if isinstance(num, int) and num >= 1 else 1
    )


def _is_anomaly_enabled() -> bool:
    """Returns True if autograd anomaly detection is enabled.

    Returns:
        Boolean flag.
    """
    return bool(_RUNTIME_STATE["anomaly_enabled"])
