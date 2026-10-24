#!/usr/bin/env python3
"""Checks for disallowed 3rd party dependencies and dynamic imports in non-test code."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

# Allowed 3rd-party root modules:
ALLOWED_MODULES: set[str] = {
    "pydantic",
    "cdd_python",
    "cdd",
    "ml_switcheroo_ir",
    "ml_switcheroo_compiler",
    "ml_ecosystem_snapshots",
    "zero_torch",
}


def is_std_lib(module_name: str) -> bool:
    """Checks whether a module name belongs to the Python standard library.

    Args:
        module_name: The root module name to check.

    Returns:
        True if the module is part of the Python standard library, False otherwise.
    """
    if not module_name:
        return True

    if module_name == "importlib":
        return True

    if sys.version_info >= (3, 10) and module_name in sys.stdlib_module_names:
        return True

    common_stdlib = {
        "os",
        "sys",
        "re",
        "math",
        "collections",
        "typing",
        "itertools",
        "functools",
        "json",
        "uuid",
        "pathlib",
        "datetime",
        "time",
        "random",
        "logging",
        "ast",
        "warnings",
        "subprocess",
        "threading",
        "__future__",
        "abc",
        "contextlib",
        "dataclasses",
        "enum",
        "weakref",
        "copy",
        "traceback",
        "inspect",
        "types",
        "operator",
        "hashlib",
        "io",
        "urllib",
        "cgi",
        "shutil",
        "tempfile",
        "importlib",
        "builtins",
    }
    return module_name in common_stdlib


def extract_constant_string(node: ast.AST) -> str | None:
    """Extracts a string literal value from an AST node if it is a string constant.

    Args:
        node: The AST node to inspect.

    Returns:
        The string value if the node is a constant string, otherwise None.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def is_sys_modules(node: ast.AST) -> bool:
    """Checks whether an AST node represents the `sys.modules` attribute access.

    Args:
        node: The AST node to inspect.

    Returns:
        True if the node represents `sys.modules`, False otherwise.
    """
    return bool(
        isinstance(node, ast.Attribute)
        and node.attr == "modules"
        and isinstance(node.value, ast.Name)
        and node.value.id == "sys"
    )


def check_file(file_path: Path) -> bool:
    """Checks a single Python source file for disallowed imports and dynamic imports.

    Args:
        file_path: Path to the Python file to inspect.

    Returns:
        True if the file has no disallowed imports, False otherwise.
    """
    with open(file_path, encoding="utf-8") as f:
        try:
            tree = ast.parse(f.read(), filename=str(file_path))
        except SyntaxError:
            return True

    violations: list[tuple[int, str]] = []

    def check_module(lineno: int, full_name: str, context: str = "") -> None:
        """Validates a single imported module name against allowed modules.

        Args:
            lineno: Line number of the import statement.
            full_name: Full dotted module name.
            context: Optional context string describing how the import occurred.
        """
        base_module = full_name.split(".")[0]
        if not is_std_lib(base_module) and base_module not in ALLOWED_MODULES:
            violations.append((lineno, f"{full_name} {context}".strip()))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                check_module(node.lineno, alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                check_module(node.lineno, node.module)
        elif isinstance(node, ast.Call):
            is_dunder_import = (
                isinstance(node.func, ast.Name) and node.func.id == "__import__"
            ) or (
                isinstance(node.func, ast.Attribute) and node.func.attr == "__import__"
            )

            is_importlib = (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "import_module"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "importlib"
            )

            if is_dunder_import or is_importlib:
                if node.args:
                    mod_name = extract_constant_string(node.args[0])
                    func_label = (
                        node.func.id
                        if isinstance(node.func, ast.Name)
                        else (
                            node.func.attr
                            if isinstance(node.func, ast.Attribute)
                            else "import"
                        )
                    )
                    if mod_name:
                        check_module(
                            node.lineno,
                            mod_name,
                            f"(via {func_label})",
                        )
                    else:
                        violations.append(
                            (node.lineno, "dynamic import with non-constant string")
                        )
                else:
                    violations.append((node.lineno, "dynamic import without arguments"))

            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr in ("get", "pop", "setdefault")
                and is_sys_modules(node.func.value)
                and node.args
            ):
                mod_name = extract_constant_string(node.args[0])
                if mod_name:
                    check_module(
                        node.lineno,
                        mod_name,
                        f"(via sys.modules.{node.func.attr})",
                    )
                else:
                    violations.append(
                        (
                            node.lineno,
                            f"sys.modules.{node.func.attr} with non-constant string",
                        )
                    )

        elif isinstance(node, ast.Subscript) and is_sys_modules(node.value):
            slice_val = node.slice
            mod_name = extract_constant_string(slice_val)
            if mod_name:
                check_module(node.lineno, mod_name, "(via sys.modules subscript)")
            else:
                violations.append(
                    (node.lineno, "sys.modules accessed with non-constant string")
                )

    if violations:
        for lineno, module in violations:
            print(f"{file_path}:{lineno} - Disallowed import found: '{module}'")
        return False
    return True


def main() -> int:
    """Runs the dependency checker across all Python files in `src/`.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    success = True
    src_dir = Path("src")
    if not src_dir.exists():
        return 0

    for py_file in sorted(src_dir.rglob("*.py")):
        if not check_file(py_file):
            success = False

    if not success:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
