#!/usr/bin/env python3
"""Checks for disallowed 3rd party dependencies in non-test code."""

import ast
import sys
from pathlib import Path

# Allowed root modules based on constraints:
# exceptions: numpy, pydantic, ../ml-switcheroo, ../cdd-python, ../ml-switcheroo-ir, ../ml-switcheroo-compiler
ALLOWED_MODULES = {
    "numpy",
    "np",  # if alias is somehow checked
    "pydantic",
    "ml_switcheroo",
    "cdd",
    "ml_switcheroo_ir",
    "ml_switcheroo_compiler",
    "zero_torch",  # self
}


def is_std_lib(module_name):
    """Check if a module is in the standard library."""
    if not module_name:
        return True  # Relative imports are handled below
    if sys.version_info >= (3, 10):
        return module_name in sys.stdlib_module_names
    else:
        # Fallback for Python 3.9
        # Simple heuristic, hardcode some common ones
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
        }
        if module_name in common_stdlib:
            return True
        return False


def check_file(file_path):
    """Check a single file for disallowed imports."""
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            tree = ast.parse(f.read(), filename=str(file_path))
        except SyntaxError:
            return True  # Not a valid python file, ignore

    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                base_module = alias.name.split(".")[0]
                if not is_std_lib(base_module) and base_module not in ALLOWED_MODULES:
                    violations.append((node.lineno, base_module))
        elif isinstance(node, ast.ImportFrom):
            # level > 0 means relative import, which is allowed (internal)
            if node.level == 0 and node.module:
                base_module = node.module.split(".")[0]
                if not is_std_lib(base_module) and base_module not in ALLOWED_MODULES:
                    violations.append((node.lineno, base_module))

    if violations:
        for lineno, module in violations:
            print(
                f"{file_path}:{lineno} - Disallowed 3rd-party import found: '{module}'"
            )
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
