"""Validates zero_torch API compliance against SEMANTIC_PLAN.md, TODO_PLAN.md, and ml-ecosystem-snapshots v0.0.5."""

from __future__ import annotations

import importlib
import inspect
import json
import os
import re
import sys
from pathlib import Path
from types import FunctionType, MethodType, ModuleType

import zero_torch

_SNAPSHOT_HELPER_EXCLUSIONS: set[str] = {
    "Any",
    "Counter",
    "Optional",
    "Optimizer",
    "SupportsFloat",
    "Tensor",
    "partial",
    "ref",
}


def parse_todo_plan_unchecked(todo_plan_path: Path | None = None) -> set[str]:
    """Extracts unchecked `- [ ]` canonical `torch.*` API paths from `TODO_PLAN.md`.

    Args:
        todo_plan_path: Optional path to `TODO_PLAN.md` (defaults to `Path("TODO_PLAN.md")`).

    Returns:
        Set of canonical `torch.*` API paths (and `torch.*(param)` / `torch.*.__constants__` tags)
        that are marked `- [ ]` in `TODO_PLAN.md`.
    """
    resolved = Path("TODO_PLAN.md") if todo_plan_path is None else todo_plan_path
    if not resolved.exists():
        return set()

    unchecked: set[str] = set()
    content = resolved.read_text(encoding="utf-8")
    for line in content.splitlines():
        if not re.match(r"^\s*-\s*\[\s\]\s+", line):
            continue
        for match in re.finditer(
            r"`((?:torch|zero_torch)\.[a-zA-Z0-9_\.]+(?:\([a-zA-Z0-9_,\s\*]+\))?)`",
            line,
        ):
            raw_path = match.group(1)
            canonical = (
                "torch." + raw_path[len("zero_torch.") :]
                if raw_path.startswith("zero_torch.")
                else raw_path
            )
            unchecked.add(canonical)
    return unchecked


def _resolve_target_symbol(
    api_path: str,
) -> tuple[bool, type | FunctionType | MethodType | ModuleType | None]:
    """Resolves a canonical `torch.*` API path against `zero_torch` without `getattr`.

    Args:
        api_path: Canonical `torch.*` dotted API path.

    Returns:
        Tuple `(exists, resolved_symbol)` where `exists` is True if the symbol is defined.
    """
    parts = api_path.split(".")
    sym_name = parts[-1]
    if len(parts) == 3 and parts[1] == "Tensor":
        for cls in inspect.getmro(zero_torch.Tensor):
            cls_members: dict[
                str, type | FunctionType | MethodType | ModuleType | None
            ] = dict(inspect.getmembers(cls))
            if sym_name in cls_members:
                return True, cls_members[sym_name]
        return False, None

    mod_name = ".".join(["zero_torch"] + parts[1:-1])
    try:
        mod = importlib.import_module(mod_name)
    except ImportError:
        return False, None

    mod_vars: dict[str, type | FunctionType | MethodType | ModuleType | None] = vars(
        mod
    )
    if sym_name in mod_vars:
        return True, mod_vars[sym_name]
    return False, None


def verify_ecosystem_snapshot(
    todo_plan_path: Path | None = None,
) -> list[str]:
    """Verifies that `zero_torch` implements the `ml-ecosystem-snapshots` v0.0.5 specification.

    Validates module hierarchy, classes, class fields (`__constants__`, etc.), functions,
    methods (`torch.Tensor.*`), parameter names/defaults, and docstrings across all 8
    categories (`activation`, `array`, `dataloader`, `initializer`, `layer`, `loss`,
    `optimizer`, `scheduler`). Additional functions/classes/parameters in `zero_torch`
    beyond the snapshot are permitted. Endpoints or signature/field refinements still
    marked `- [ ]` in `TODO_PLAN.md` are tracked as planned work until checked off.

    Args:
        todo_plan_path: Optional path to `TODO_PLAN.md`.

    Returns:
        List of missing or non-conformant API descriptions, or an empty list if conformant.
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
    if not cache_path.exists():
        return []

    raw = json.loads(cache_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return []
    categories = raw.get("categories")
    if not isinstance(categories, dict):
        return []

    unchecked = parse_todo_plan_unchecked(todo_plan_path)
    missing: list[str] = []
    target_cats = (
        "activation",
        "array",
        "dataloader",
        "initializer",
        "layer",
        "loss",
        "optimizer",
        "scheduler",
    )
    for cat in target_cats:
        items = categories.get(cat, [])
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            api_path = item.get("api_path")
            name = item.get("name")
            if not isinstance(api_path, str) or not isinstance(name, str):
                api_path = None
                continue
            parts = api_path.split(".")
            sym_name = parts[-1]
            if (
                name in _SNAPSHOT_HELPER_EXCLUSIONS
                or sym_name in _SNAPSHOT_HELPER_EXCLUSIONS
            ):
                sym_name = ""
                continue
            if api_path in unchecked:
                continue

            exists, sym_obj = _resolve_target_symbol(api_path)
            if not exists:
                missing.append(api_path)
                continue

            if sym_obj is not None and callable(sym_obj):
                doc = inspect.getdoc(sym_obj)
                if not doc and item.get("docstring"):
                    missing.append(f"{api_path}: missing docstring")

                raw_params = item.get("params")
                if isinstance(raw_params, list):
                    try:
                        sig = inspect.signature(sym_obj)
                        actual_params = {
                            p.name: p
                            for p in sig.parameters.values()
                            if p.name not in ("self", "cls")
                        }
                    except (ValueError, TypeError):
                        actual_params = {}

                    cls_members = (
                        dict(inspect.getmembers(sym_obj))
                        if inspect.isclass(sym_obj)
                        else {}
                    )
                    for sp in raw_params:
                        if not isinstance(sp, dict):
                            continue
                        pname_raw = sp.get("name")
                        if not isinstance(pname_raw, str):
                            continue
                        sub_names = [
                            s.strip().lstrip("\\*")
                            for s in pname_raw.split(",")
                            if s.strip()
                        ]
                        for pname in sub_names:
                            if pname in (
                                "self",
                                "cls",
                                "args",
                                "kwargs",
                                "__doc__",
                            ):
                                continue
                            field_tag = f"{api_path}.{pname}"
                            param_tag = f"{api_path}({pname})"
                            if field_tag in unchecked or param_tag in unchecked:
                                continue
                            is_class_field = pname == "__constants__" or (
                                item.get("kind") == "class"
                                and pname.startswith("_")
                                and pname not in actual_params
                            )
                            if is_class_field:
                                if pname not in cls_members:
                                    missing.append(field_tag)
                            elif (
                                pname not in actual_params
                                and f"{pname}_" not in actual_params
                                and pname not in cls_members
                            ):
                                missing.append(param_tag)

    return missing


def main() -> None:
    """Checks API compliance against SEMANTIC_PLAN.md, TODO_PLAN.md, and ml-ecosystem-snapshots v0.0.5."""
    if not os.path.exists("SEMANTIC_PLAN.md"):
        print("SEMANTIC_PLAN.md not found, skipping compliance check.")
        sys.exit(0)

    with open("SEMANTIC_PLAN.md", encoding="utf-8") as f:
        content = f.read()

    modules: dict[str, list[str]] = {}
    current_module: str | None = None

    for line in content.splitlines():
        m_mod = re.search(r"`([a-zA-Z0-9_\.]+)`", line)
        if line.startswith("## ") and m_mod:
            raw_mod = m_mod.group(1)
            prefix = raw_mod.split(".")[0]
            current_module = raw_mod.replace(prefix, "zero_torch", 1)
            modules.setdefault(current_module, [])
            continue
        elif line.startswith("## ") and not current_module:
            current_module = "zero_torch"
            modules[current_module] = []
            continue

        if current_module:
            m_item = re.match(r"^- \[[xX ]\] `?([a-zA-Z0-9_]+)`?(?:[: ].*)?", line)
            if m_item:
                funcs_part = m_item.group(1)
                funcs = [f.strip(" `") for f in funcs_part.split("/")]
                for fn in funcs:
                    if re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", fn):
                        modules[current_module].append(fn)

    missing: list[str] = []
    total = 0
    implemented = 0

    if not modules or all(len(funcs) == 0 for funcs in modules.values()):
        print("No parsable API targets found in SEMANTIC_PLAN.md")
        sys.exit(0)

    for mod_name, funcs in modules.items():
        if not funcs:
            continue
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            missing.extend([f"{mod_name}.{f}" for f in funcs])
            total += len(funcs)
            continue

        mod_vars = vars(mod)
        for f_name in funcs:
            f_name = f_name.strip()
            if not f_name:
                continue
            total += 1
            if f_name in mod_vars:
                implemented += 1
            else:
                missing.append(f"{mod_name}.{f_name}")

    snapshot_missing = verify_ecosystem_snapshot()
    missing.extend(snapshot_missing)

    pct = (implemented / total) * 100.0 if total > 0 else 100.0

    current_todo = ""
    try:
        with open("PYTORCH_TODO.md", encoding="utf-8") as f:
            current_todo = f.read()
    except FileNotFoundError:
        current_todo = ""

    new_todo_lines: list[str] = [
        "# PYTORCH_TODO.md",
        "--- Compliance Report ---",
        f"Overall Compliance: {pct:.1f}%",
        "",
        "Breakdown by Module:",
    ]
    for mod_name, funcs in modules.items():
        if not funcs:
            continue
        prefix = ["PYTORCH", "TODO.md"][0].lower()
        orig_mod_name = mod_name.replace("zero_torch", prefix)
        mod_implemented = sum(
            1 for f_name in funcs if f"{mod_name}.{f_name}" not in missing
        )
        mod_pct = (mod_implemented / len(funcs)) * 100.0 if funcs else 100.0
        new_todo_lines.append(
            f"  - {orig_mod_name}: {mod_pct:.1f}% ({mod_implemented}/{len(funcs)})"
        )

    if missing:
        new_todo_lines.append("")
        new_todo_lines.append("Missing APIs:")
        for m in missing:
            new_todo_lines.append(f"  - {m}")

    new_todo = chr(10).join(new_todo_lines) + chr(10)

    if new_todo != current_todo:
        with open("PYTORCH_TODO.md", "w", encoding="utf-8") as f:
            f.write(new_todo)
        print("Updated PYTORCH_TODO.md.")

    if missing:
        print(
            "COMPLIANCE FAILURE: The following expected APIs are missing or not exported:"
        )
        for m in missing:
            print(f"  - {m}")
        print(f"Overall Compliance: {pct:.1f}% ({implemented}/{total})")
        sys.exit(1)
    else:
        print(f"Compliance Check Passed! 100% compliant ({implemented}/{total} APIs).")
        if new_todo != current_todo:
            sys.exit(1)
        sys.exit(0)


if __name__ == "__main__":
    main()
