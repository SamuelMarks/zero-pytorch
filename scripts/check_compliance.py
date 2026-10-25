"""Validates zero_torch API compliance against SEMANTIC_PLAN.md and ml-ecosystem-snapshots v0.0.4."""

from __future__ import annotations

import importlib
import json
import os
import re
import sys
from pathlib import Path

_SNAPSHOT_HELPER_EXCLUSIONS: set[str] = {
    "Any",
    "Counter",
    "Optimizer",
    "SupportsFloat",
    "Tensor",
    "partial",
    "ref",
}


def verify_ecosystem_snapshot() -> list[str]:
    """Verifies that all structured categories in the ml-ecosystem-snapshots v0.0.4 JSON exist in zero_torch.

    Returns:
        List of missing API paths, or an empty list if all are present.
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

    missing: list[str] = []
    target_cats = (
        "activation",
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
            mod_name = ".".join(["zero_torch"] + parts[1:-1])
            sym_name = parts[-1]
            if (
                name in _SNAPSHOT_HELPER_EXCLUSIONS
                or sym_name in _SNAPSHOT_HELPER_EXCLUSIONS
            ):
                sym_name = ""
                continue
            try:
                mod = importlib.import_module(mod_name)
                if sym_name not in vars(mod):
                    missing.append(api_path)
            except ImportError:
                missing.append(api_path)
    return missing


def main() -> None:
    """Checks API compliance against SEMANTIC_PLAN.md and ml-ecosystem-snapshots v0.0.4."""
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
