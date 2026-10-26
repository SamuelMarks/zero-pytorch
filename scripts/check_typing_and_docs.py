#!/usr/bin/env python3
"""Enforces 100% static typing (no Any/object/cast/getattr/setattr/hasattr) and 100% docstring (module, class, function, arg, return) coverage."""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

FORBIDDEN_CALLS: set[str] = {
    "cast",
    "getattr",
    "setattr",
    "hasattr",
    "delattr",
}

FORBIDDEN_NAMES: set[str] = {
    "Any",
    "object",
}

ARGS_HEADER_RE = re.compile(r"^\s*Args:\s*$", re.MULTILINE)
RETURNS_HEADER_RE = re.compile(r"^\s*(Returns|Yields):\s*$", re.MULTILINE)


def _is_none_return(annotation: ast.AST | None) -> bool:
    """Checks whether a return annotation represents None.

    Args:
        annotation: AST node of the function return annotation.

    Returns:
        True if the annotation is None, False otherwise.
    """
    if annotation is None:
        return True
    if isinstance(annotation, ast.Constant) and annotation.value in (None, "None"):
        return True
    return bool(
        isinstance(annotation, ast.Name) and annotation.id in ("None", "NoReturn")
    )


def _extract_documented_args(docstring: str) -> set[str]:
    """Extracts documented parameter names from the Args section of a Google-style docstring.

    Args:
        docstring: The raw docstring text.

    Returns:
        Set of parameter names documented in the docstring (stripped of leading `*`).
    """
    documented: set[str] = set()
    in_args = False
    args_indent: int | None = None
    for line in docstring.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        indent = len(line) - len(line.lstrip())
        if stripped == "Args:":
            in_args = True
            args_indent = indent
            continue
        if in_args:
            if (
                args_indent is not None
                and indent <= args_indent
                and stripped.endswith(":")
            ):
                break
            m = re.match(
                r"^\*{0,2}([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:\([^)]*\))?\s*:", stripped
            )
            if m:
                documented.add(m.group(1))
    return documented


def check_file(file_path: Path) -> list[str]:
    """Audits a single Python source file for typing and docstring compliance.

    Args:
        file_path: Path to the Python file to audit.

    Returns:
        List of human-readable violation strings found in the file.
    """
    errors: list[str] = []
    source = file_path.read_text(encoding="utf-8")

    for idx, line in enumerate(source.splitlines(), start=1):
        if "type: ignore" in line and file_path.name != "check_typing_and_docs.py":
            errors.append(f"{file_path}:{idx}: forbidden '# type: ignore' comment")
        if "pragma: no cover" in line and file_path.name != "check_typing_and_docs.py":
            errors.append(f"{file_path}:{idx}: forbidden '# pragma: no cover' comment")
        if "# noqa" in line and file_path.name != "check_typing_and_docs.py":
            errors.append(f"{file_path}:{idx}: forbidden '# noqa' comment")

    tree = ast.parse(source, filename=str(file_path))
    is_stub = file_path.suffix == ".pyi"
    is_test = file_path.name.startswith("test_")

    mod_doc = ast.get_docstring(tree)
    if not mod_doc or not mod_doc.strip():
        errors.append(f"{file_path}:1: missing module docstring")

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name in ("Any", "cast"):
                    errors.append(
                        f"{file_path}:{node.lineno}: forbidden import '{alias.name}'"
                    )
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if is_stub and ("```(" in node.value or ":attr:`" in node.value):
                errors.append(
                    f"{file_path}:{node.lineno}: malformed scraped docstring default '{node.value}'"
                )
        elif isinstance(node, ast.Name):
            if node.id in FORBIDDEN_NAMES:
                errors.append(
                    f"{file_path}:{node.lineno}: forbidden symbol '{node.id}'"
                )
            elif node.id in (
                "__dict__",
                "__builtins__",
                "__getattr__",
                "__setattr__",
                "__delattr__",
            ):
                errors.append(
                    f"{file_path}:{node.lineno}: forbidden dunder '{node.id}'"
                )
        elif isinstance(node, ast.Attribute):
            if node.attr in (
                "__dict__",
                "__builtins__",
                "__getattr__",
                "__setattr__",
                "__delattr__",
            ):
                errors.append(
                    f"{file_path}:{node.lineno}: forbidden dunder attribute '{node.attr}'"
                )
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "_pass":
                    errors.append(
                        f"{file_path}:{node.lineno}: tautological '_pass = ...' exception swallowing is forbidden"
                    )
                elif is_test and isinstance(target, ast.Name) and target.id == "_":
                    errors.append(
                        f"{file_path}:{node.lineno}: discarded '_ = ...' call in test is forbidden"
                    )
        elif (
            isinstance(node, ast.Assert)
            and isinstance(node.test, ast.Constant)
            and node.test.value is True
        ):
            errors.append(
                f"{file_path}:{node.lineno}: tautological 'assert True' is forbidden"
            )
        elif isinstance(node, ast.ClassDef):
            cls_doc = ast.get_docstring(node)
            if not is_stub and (not cls_doc or not cls_doc.strip()):
                errors.append(
                    f"{file_path}:{node.lineno}: class '{node.name}' missing docstring"
                )
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in ("__getattr__", "__setattr__", "__delattr__"):
                errors.append(
                    f"{file_path}:{node.lineno}: forbidden dunder method '{node.name}'"
                )
            is_overload = any(
                (isinstance(d, ast.Name) and d.id == "overload")
                or (isinstance(d, ast.Attribute) and d.attr == "overload")
                for d in node.decorator_list
            )
            fn_doc = ast.get_docstring(node)
            if not is_stub and not is_overload and (not fn_doc or not fn_doc.strip()):
                errors.append(
                    f"{file_path}:{node.lineno}: function '{node.name}' missing docstring"
                )
            if node.returns is None:
                errors.append(
                    f"{file_path}:{node.lineno}: function '{node.name}' missing return type annotation"
                )
            elif (
                isinstance(node.returns, ast.Subscript)
                and isinstance(node.returns.value, ast.Name)
                and node.returns.value.id == "TypeGuard"
            ):
                for stmt in node.body:
                    if isinstance(stmt, ast.Return) and (
                        (
                            isinstance(stmt.value, ast.Constant)
                            and stmt.value.value is True
                        )
                        or isinstance(stmt.value, ast.BoolOp)
                    ):
                        errors.append(
                            f"{file_path}:{node.lineno}: tautological TypeGuard '{node.name}' is forbidden"
                        )
            all_args = (
                list(node.args.posonlyargs)
                + list(node.args.args)
                + list(node.args.kwonlyargs)
            )
            required_doc_args: list[str] = []
            for arg in all_args:
                if arg.arg in ("self", "cls"):
                    continue
                required_doc_args.append(arg.arg)
                if (
                    isinstance(arg.annotation, ast.Name)
                    and arg.annotation.id == "object"
                    and node.name not in ("__eq__", "__ne__")
                ):
                    errors.append(
                        f"{file_path}:{node.lineno}: forbidden 'object' annotation on parameter '{arg.arg}' in '{node.name}'"
                    )
                if arg.annotation is None:
                    errors.append(
                        f"{file_path}:{node.lineno}: parameter '{arg.arg}' in '{node.name}' missing type annotation"
                    )
            if node.args.vararg is not None:
                required_doc_args.append(node.args.vararg.arg)
                if node.args.vararg.annotation is None:
                    errors.append(
                        f"{file_path}:{node.lineno}: vararg '*{node.args.vararg.arg}' in '{node.name}' missing type annotation"
                    )
            if node.args.kwarg is not None:
                required_doc_args.append(node.args.kwarg.arg)
                if node.args.kwarg.annotation is None:
                    errors.append(
                        f"{file_path}:{node.lineno}: kwarg '**{node.args.kwarg.arg}' in '{node.name}' missing type annotation"
                    )

            if fn_doc and not is_stub:
                doc_args = _extract_documented_args(fn_doc)
                for req_arg in required_doc_args:
                    if req_arg not in doc_args:
                        errors.append(
                            f"{file_path}:{node.lineno}: parameter '{req_arg}' in '{node.name}' not documented in Args:"
                        )
                if not _is_none_return(node.returns) and not RETURNS_HEADER_RE.search(
                    fn_doc
                ):
                    errors.append(
                        f"{file_path}:{node.lineno}: function '{node.name}' missing Returns:/Yields: section in docstring"
                    )
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
                errors.append(
                    f"{file_path}:{node.lineno}: forbidden call '{node.func.id}()'"
                )
            elif isinstance(node.func, ast.Attribute) and (
                node.func.attr in ("getattr", "setattr", "hasattr", "delattr")
                or (
                    node.func.attr == "cast"
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id in ("typing", "builtins")
                )
            ):
                receiver = (
                    node.func.value.id
                    if isinstance(node.func.value, ast.Name)
                    else "attr"
                )
                errors.append(
                    f"{file_path}:{node.lineno}: forbidden call '{receiver}.{node.func.attr}()'"
                )

    return errors


def main() -> int:
    """Runs typing and docstring checks across `src/zero_torch`, `scripts`, `tests`, and `stubs`.

    Returns:
        Exit code (0 if all checks pass, 1 otherwise).
    """
    target_dirs = [
        Path("src/zero_torch"),
        Path("scripts"),
        Path("tests"),
        Path("stubs"),
    ]
    all_errors: list[str] = []
    for d in target_dirs:
        if not d.exists():
            continue
        for py_file in sorted(list(d.rglob("*.py")) + list(d.rglob("*.pyi"))):
            all_errors.extend(check_file(py_file))

    if all_errors:
        for err in all_errors[:200]:
            print(err)
        if len(all_errors) > 200:
            print(f"... and {len(all_errors) - 200} more violations.")
        print(f"Total violations: {len(all_errors)}")
        sys.exit(1)

    print("100% typing and docstring check passed with 0 violations!")
    sys.exit(0)


if __name__ == "__main__":
    main()
