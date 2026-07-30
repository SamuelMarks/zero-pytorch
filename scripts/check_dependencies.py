#!/usr/bin/env python3
"""Checks for disallowed 3rd party dependencies and dynamic imports in non-test code."""

import ast
import sys
from pathlib import Path

# Allowed 3rd-party root modules:
ALLOWED_MODULES = {
    "pydantic",
    "cdd_python",
    "cdd",
    "ml_switcheroo_ir",
    "ml_switcheroo_compiler",
    "zero_torch",  # self
}


def is_std_lib(module_name):
    """Check if a module is in the standard library."""
    if not module_name:
        return True  # Relative imports are handled below

    # 'importlib' is no longer unconditionally forbidden, just its non-whitelisted usages
    if module_name == "importlib":
        return True  # we'll inspect its calls instead

    if sys.version_info >= (3, 10):
        if module_name in sys.stdlib_module_names:
            return True

    # Fallback for Python 3.9
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
    }
    return module_name in common_stdlib


def extract_constant_string(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    # For python < 3.8 support
    if isinstance(node, ast.Str):
        return node.s
    return None


def is_sys_modules(node):
    # Check if a node represents `sys.modules`
    return bool(
        isinstance(node, ast.Attribute)
        and (
            node.attr == "modules"
            and isinstance(node.value, ast.Name)
            and node.value.id == "sys"
        )
    )


def check_file(file_path):
    """Check a single file for disallowed imports."""
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            tree = ast.parse(f.read(), filename=str(file_path))
        except SyntaxError:
            return True  # Not a valid python file, ignore

    violations = []

    def check_module(lineno, full_name, context=""):
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
            # Check __import__("name")
            is_dunder_import = False
            if (
                isinstance(node.func, ast.Name)
                and node.func.id == "__import__"
                or (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr == "__import__"
                )
            ):
                is_dunder_import = True

            # Check importlib.import_module("name")
            is_importlib = False
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "import_module"
            ) and (
                isinstance(node.func.value, ast.Name)
                and node.func.value.id == "importlib"
            ):
                is_importlib = True

            if is_dunder_import or is_importlib:
                if node.args:
                    mod_name = extract_constant_string(node.args[0])
                    if mod_name:
                        check_module(
                            node.lineno,
                            mod_name,
                            f"(via {node.func.id if isinstance(node.func, ast.Name) else node.func.attr})",
                        )
                    else:
                        violations.append(
                            (node.lineno, "dynamic import with non-constant string")
                        )
                else:
                    violations.append((node.lineno, "dynamic import without arguments"))

            # Check sys.modules.get("name") or sys.modules.pop("name")
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
            # Check sys.modules["name"]
            slice_val = None
            if isinstance(node.slice, ast.Index):  # Python < 3.9
                slice_val = node.slice.value
            else:
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


def main():
    success = True
    src_dir = Path("src")
    if not src_dir.exists():
        return 0

    for py_file in src_dir.rglob("*.py"):
        if not check_file(py_file):
            success = False

    if not success:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
