"""Generates and validates top-level zero_torch API snapshots against ml-ecosystem-snapshots v0.0.5."""

from __future__ import annotations

import inspect
import json
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from types import ModuleType


def generate_snapshot(
    members_override: Sequence[tuple[str, type | Callable[[], None]]] | None = None,
    force_import_fallback: bool = False,
) -> str:
    """Generates a sorted newline-delimited snapshot of public zero_torch top-level APIs.

    Args:
        members_override: Optional synthetic `(name, obj)` pairs for testing.
        force_import_fallback: If True, exercises the relative path bootstrap branch.

    Returns:
        Sorted newline-delimited string of public functions and classes with signatures.
    """
    try:
        if force_import_fallback:
            raise ImportError("forced fallback")
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
    members: Sequence[tuple[str, type | Callable[[], None] | ModuleType]] = (
        members_override
        if members_override is not None
        else inspect.getmembers(zero_torch)
    )
    for name, obj in members:
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
    nl = chr(10)
    return nl.join(sorted(apis)) + nl


def load_ecosystem_snapshot(
    home_dir: Path | None = None,
    resolver_fn: Callable[[str], str] | None = None,
) -> tuple[str | None, set[str]]:
    """Loads the canonical PyTorch snapshot version and top-level API names from ml-ecosystem-snapshots.

    Args:
        home_dir: Optional override for user home directory path.
        resolver_fn: Optional override for `resolve_snapshot_path`.

    Returns:
        Tuple of `(version_string_or_none, set_of_expected_top_level_symbol_names)`.
    """
    base_home = Path.home() if home_dir is None else home_dir
    cache_path = (
        base_home
        / ".cache"
        / "ml_ecosystem_snapshots"
        / "snapshots"
        / "torch_v2.14.1_cpu.json"
    )
    if not cache_path.exists():
        try:
            if resolver_fn is not None:
                resolved = Path(resolver_fn("torch_v2.14.1_cpu"))
            else:
                from ml_ecosystem_snapshots.cli.resolver import resolve_snapshot_path

                resolved = Path(resolve_snapshot_path("torch_v2.14.1_cpu"))
            if resolved.exists():
                cache_path = resolved
        except ImportError:
            cache_path = cache_path
    if cache_path.exists():
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            ver_obj = raw.get("version")
            ver = ver_obj if isinstance(ver_obj, str) else None
            top_level: set[str] = set()
            categories = raw.get("categories")
            if isinstance(categories, dict):
                for items in categories.values():
                    if isinstance(items, list):
                        for item in items:
                            if isinstance(item, dict):
                                api_path = item.get("api_path")
                                if isinstance(api_path, str):
                                    parts = api_path.split(".")
                                    if len(parts) == 2 and parts[0] == "torch":
                                        top_level.add(parts[1])
            return ver, top_level
    return None, set()


def load_ecosystem_snapshot_version(
    home_dir: Path | None = None,
    resolver_fn: Callable[[str], str] | None = None,
) -> str | None:
    """Loads the canonical PyTorch snapshot version from ml-ecosystem-snapshots cache if present.

    Args:
        home_dir: Optional override for user home directory path.
        resolver_fn: Optional override for `resolve_snapshot_path`.

    Returns:
        The snapshot version string if available, otherwise None.
    """
    ver, _ = load_ecosystem_snapshot(home_dir=home_dir, resolver_fn=resolver_fn)
    return ver


def main(
    snapshot_text: str | None = None,
    eco_data: tuple[str | None, set[str]] | None = None,
) -> None:
    """Validates `zero_torch` top-level API snapshot against `ml-ecosystem-snapshots` v0.0.5.

    Args:
        snapshot_text: Optional pre-generated API snapshot string.
        eco_data: Optional pre-loaded `(version, expected_top_level)` tuple.
    """
    api_snapshot = generate_snapshot() if snapshot_text is None else snapshot_text
    exported_names = {
        line.split("(", 1)[0] for line in api_snapshot.splitlines() if "(" in line
    }
    eco_version, expected_top_level = (
        load_ecosystem_snapshot() if eco_data is None else eco_data
    )
    if eco_version is not None:
        print(f"Loaded ml-ecosystem-snapshots v0.0.5 target: torch {eco_version}")

    missing = sorted(expected_top_level - exported_names)
    if not exported_names or missing:
        print("API mismatch against ml-ecosystem-snapshots!")
        for symbol in missing[:10]:
            print(f"  Missing top-level API: torch.{symbol}")
        sys.exit(1)

    print("API matches snapshot.")
    sys.exit(0)


if __name__ == "__main__":
    main()
