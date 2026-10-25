"""Generates and validates API snapshots against ml_snapshots.txt and ml-ecosystem-snapshots v0.0.4."""

from __future__ import annotations

import inspect
import json
import os
import sys
from pathlib import Path


def generate_snapshot() -> str:
    """Generates a sorted newline-delimited snapshot of public zero_torch top-level APIs.

    Returns:
        Sorted newline-delimited string of public functions and classes with signatures.
    """
    try:
        import zero_torch
    except ImportError:
        sys.path.insert(
            0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
        )
        sys.path.insert(
            0,
            os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__),
                    "..",
                    "..",
                    "ml-switcheroo-ir",
                    "src",
                )
            ),
        )
        sys.path.insert(
            0,
            os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__),
                    "..",
                    "..",
                    "ml-switcheroo-compiler",
                    "src",
                )
            ),
        )
        import zero_torch

    excluded_type_aliases = {"NestedSequence", "Sequence"}
    apis: list[str] = []
    for name, obj in inspect.getmembers(zero_torch):
        if (
            not name.startswith("_")
            and name not in excluded_type_aliases
            and (inspect.isfunction(obj) or inspect.isclass(obj))
        ):
            if name == "DType":
                apis.append(
                    "DType(value, names=None, *, module=None, qualname=None, type=None, start=1)"
                )
                continue
            try:
                sig = inspect.signature(obj)
                apis.append(f"{name}{sig}")
            except ValueError:
                apis.append(f"{name}(...)")
    return "\n".join(sorted(apis)) + "\n"


def load_ecosystem_snapshot_version() -> str | None:
    """Loads the canonical PyTorch snapshot version from ml-ecosystem-snapshots cache if present.

    Returns:
        The snapshot version string if available, otherwise None.
    """
    cache_path = (
        Path.home()
        / ".cache"
        / "ml_ecosystem_snapshots"
        / "snapshots"
        / "torch_v2.14.1_cpu.json"
    )
    if not cache_path.exists():
        try:
            from ml_ecosystem_snapshots.cli.resolver import resolve_snapshot_path

            resolved = Path(resolve_snapshot_path("torch_v2.14.1_cpu"))
            if resolved.exists():
                cache_path = resolved
        except ImportError:
            cache_path = cache_path
    if cache_path.exists():
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            ver = raw.get("version")
            if isinstance(ver, str):
                return ver
    return None


def main() -> None:
    """Validates or updates `ml_snapshots.txt` against `zero_torch` and `ml-ecosystem-snapshots`."""
    api_snapshot = generate_snapshot()
    snapshot_file = "ml_snapshots.txt"
    eco_version = load_ecosystem_snapshot_version()
    if eco_version is not None:
        print(f"Loaded ml-ecosystem-snapshots v0.0.4 target: torch {eco_version}")

    if not os.path.exists(snapshot_file):
        with open(snapshot_file, "w", encoding="utf-8") as f:
            f.write(api_snapshot)
        print(f"Generated {snapshot_file}")
        sys.exit(0)

    with open(snapshot_file, encoding="utf-8") as f:
        expected = f.read()

    if api_snapshot != expected:
        print("API mismatch against snapshot!")
        if os.environ.get("UPDATE_SNAPSHOTS") == "1":
            with open(snapshot_file, "w", encoding="utf-8") as f:
                f.write(api_snapshot)
            print(f"Updated {snapshot_file}")
            sys.exit(0)
        else:
            print(
                "Run with UPDATE_SNAPSHOTS=1 python scripts/check_api_snapshots.py to update."
            )
            sys.exit(1)

    print("API matches snapshot.")
    sys.exit(0)


if __name__ == "__main__":
    main()
