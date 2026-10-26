#!/usr/bin/env python3
"""Checks for disallowed 3rd party dependencies and dynamic imports across source, scripts, and tests."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

# Allowed 3rd-party root modules in `src/` and `scripts/`:
ALLOWED_MODULES: set[str] = {
    "pydantic",
    "cdd_python",
    "cdd",
    "ml_switcheroo_ir",
    "ml_switcheroo_compiler",
    "ml_ecosystem_snapshots",
    "zero_torch",
    "zero_jax",
    "zero_flax",
    "zero_keras",
    "zero_tensorflow",
    "zero_mlx",
    "zero_pax",
    "zero_chex",
    "zero_optax",
    "zero_orbax",
    "zero_grain",
    "zero_zoo",
}

# Additional allowed root modules in `tests/`:
ALLOWED_TEST_MODULES: set[str] = {
    "pytest",
    "scripts",
    "tests",
    "conftest",
}

# Singular parity test exception allowed to import reference `torch`:
PARITY_TEST_SUFFIX: tuple[str, ...] = ("tests", "pytorch_official", "test_parity.py")


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
        "cmath",
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
        "runpy",
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


def _allowed_modules_for_path(file_path: Path) -> set[str]:
    """Determines the set of allowed third-party root modules for a given file path.

    Args:
        file_path: Path to the Python file being checked.

    Returns:
        Set of allowed root module names for `file_path`.
    """
    parts = file_path.parts
    allowed = set(ALLOWED_MODULES)
    if "tests" in parts:
        allowed |= ALLOWED_TEST_MODULES
        if parts[-3:] == PARITY_TEST_SUFFIX:
            allowed.add("torch")
    return allowed


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
    allowed_for_file = _allowed_modules_for_path(file_path)

    def check_module(lineno: int, full_name: str, context: str = "") -> None:
        """Validates a single imported module name against allowed modules.

        Args:
            lineno: Line number of the import statement.
            full_name: Full dotted module name.
            context: Optional context string describing how the import occurred.
        """
        base_module = full_name.split(".")[0]
        if not is_std_lib(base_module) and base_module not in allowed_for_file:
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
    """Runs the dependency checker across all Python files in `src/` and `tests/`.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    success = True
    target_dirs = [Path("src"), Path("tests")]
    if not any(d.exists() for d in target_dirs):
        return 0

    for target_dir in target_dirs:
        if target_dir.exists():
            for py_file in sorted(target_dir.rglob("*.py")):
                if not check_file(py_file):
                    success = False

    if not success:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
