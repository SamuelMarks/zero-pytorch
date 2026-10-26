"""Comprehensive unit tests covering 100% of functions, lines, and branches in `scripts/`."""

from __future__ import annotations

import ast
import builtins
import inspect
import json
import re
import runpy
import subprocess
import sys
import types
from collections.abc import Callable
from pathlib import Path
from typing import SupportsIndex

import pytest

from scripts import (
    check_api_snapshots,
    check_compliance,
    check_dependencies,
    check_typing_and_docs,
    update_badges,
)

NL = chr(10)


def test_check_api_snapshots_generate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Covers `generate_snapshot` including ImportError fallback and `ValueError` signature fallback.

    Args:
        monkeypatch: Pytest fixture for patching environment and modules.
    """
    orig_import = builtins.__import__
    fail_once = {"done": False}

    def fake_import(
        name: str,
        globals_dict: dict[str, str] | None = None,
        locals_dict: dict[str, str] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> types.ModuleType:
        """Simulates `ImportError` on the first `import zero_torch`.

        Args:
            name: Module name.
            globals_dict: Globals mapping.
            locals_dict: Locals mapping.
            fromlist: From-import list.
            level: Relative import level.

        Returns:
            Imported module.
        """
        if name == "zero_torch" and not fail_once["done"]:
            fail_once["done"] = True
            raise ImportError("simulated missing zero_torch")
        return orig_import(name, globals_dict, locals_dict, fromlist, level)

    def fake_getmembers(
        _mod: types.ModuleType,
    ) -> list[tuple[str, type | Callable[[], None]]]:
        """Returns synthetic members to exercise all branches of `generate_snapshot`.

        Args:
            _mod: Module being inspected.

        Returns:
            List of `(name, member)` pairs.
        """

        def normal_fn() -> None:
            """Sample function."""

        class NormalClass:
            """Sample class."""

        return [
            ("_private", normal_fn),
            ("NestedSequence", NormalClass),
            ("DType", NormalClass),
            ("normal_fn", normal_fn),
            ("bad_sig_fn", normal_fn),
        ]

    orig_signature = inspect.signature

    def fake_signature(
        obj: Callable[[], None],
    ) -> inspect.Signature:
        """Raises `ValueError` on the second call to test the `except ValueError` branch.

        Args:
            obj: Callable to inspect.

        Returns:
            Inspect signature.
        """
        if fake_calls["count"] > 0:
            raise ValueError("no signature")
        fake_calls["count"] += 1
        return orig_signature(obj)

    fake_calls = {"count": 0}
    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(inspect, "getmembers", fake_getmembers)
    monkeypatch.setattr(inspect, "signature", fake_signature)

    snap = check_api_snapshots.generate_snapshot()
    assert (
        "DType(value, names=None, *, module=None, qualname=None, type=None, start=1)"
        in snap
    )
    assert "normal_fn()" in snap
    assert "bad_sig_fn(...)" in snap


def test_check_api_snapshots_load_version(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Covers all branches of `load_ecosystem_snapshot` and `load_ecosystem_snapshot_version`.

    Args:
        monkeypatch: Pytest fixture for patching attributes and modules.
        tmp_path: Temporary directory path fixture.
    """
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    # 1. Cache does not exist, resolver resolves to existing valid file with categories
    resolved_file = tmp_path / "resolved_torch.json"
    resolved_file.write_text(
        json.dumps(
            {
                "version": "2.14.1",
                "categories": {
                    "array": [
                        {"api_path": "torch.abs"},
                        {"api_path": "torch.linalg.inv"},
                        {"api_path": 123},
                        "not_a_dict",
                    ],
                    "bad_cat": "not_a_list",
                },
            }
        ),
        encoding="utf-8",
    )

    fake_resolver_mod = types.ModuleType("ml_ecosystem_snapshots.cli.resolver")
    vars(fake_resolver_mod)["resolve_snapshot_path"] = lambda _name: str(resolved_file)
    monkeypatch.setitem(
        sys.modules, "ml_ecosystem_snapshots.cli.resolver", fake_resolver_mod
    )
    assert check_api_snapshots.load_ecosystem_snapshot_version() == "2.14.1"
    ver, top_level = check_api_snapshots.load_ecosystem_snapshot()
    assert ver == "2.14.1"
    assert top_level == {"abs"}

    # 2. Resolver resolves to non-existent file
    vars(fake_resolver_mod)["resolve_snapshot_path"] = lambda _name: str(
        tmp_path / "nonexistent.json"
    )
    assert check_api_snapshots.load_ecosystem_snapshot_version() is None

    # 3. Resolver raises ImportError
    orig_import = builtins.__import__

    def fail_resolver_import(
        name: str,
        globals_dict: dict[str, str] | None = None,
        locals_dict: dict[str, str] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> types.ModuleType:
        """Raises `ImportError` when importing `ml_ecosystem_snapshots.cli.resolver`.

        Args:
            name: Module name.
            globals_dict: Globals mapping.
            locals_dict: Locals mapping.
            fromlist: From-import list.
            level: Relative import level.

        Returns:
            Imported module.
        """
        if name == "ml_ecosystem_snapshots.cli.resolver":
            raise ImportError("no resolver")
        return orig_import(name, globals_dict, locals_dict, fromlist, level)

    monkeypatch.delitem(
        sys.modules, "ml_ecosystem_snapshots.cli.resolver", raising=False
    )
    monkeypatch.setattr(builtins, "__import__", fail_resolver_import)
    assert check_api_snapshots.load_ecosystem_snapshot_version() is None
    monkeypatch.setattr(builtins, "__import__", orig_import)

    # 4. Cache file exists directly under ~/.cache/...
    cache_dir = tmp_path / ".cache" / "ml_ecosystem_snapshots" / "snapshots"
    cache_dir.mkdir(parents=True)
    cache_file = cache_dir / "torch_v2.14.1_cpu.json"

    # Non-dict JSON
    cache_file.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    assert check_api_snapshots.load_ecosystem_snapshot_version() is None

    # Dict with non-str version and non-dict categories
    cache_file.write_text(
        json.dumps({"version": 123, "categories": []}), encoding="utf-8"
    )
    assert check_api_snapshots.load_ecosystem_snapshot_version() is None


def test_check_api_snapshots_main(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Covers all branches of `check_api_snapshots.main` and `__main__` entrypoint.

    Args:
        monkeypatch: Pytest fixture for patching attributes and environment.
        tmp_path: Temporary directory path fixture.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        check_api_snapshots, "generate_snapshot", lambda: "fn()" + NL + "no_parens" + NL
    )
    monkeypatch.setattr(
        check_api_snapshots,
        "load_ecosystem_snapshot",
        lambda: ("2.14.1", {"fn"}),
    )

    # 1. Snapshot matches with eco_version present -> exits 0
    with pytest.raises(SystemExit) as exc:
        check_api_snapshots.main()
    assert exc.value.code == 0

    # 2. Snapshot matches with eco_version None -> exits 0
    monkeypatch.setattr(
        check_api_snapshots, "load_ecosystem_snapshot", lambda: (None, {"fn"})
    )
    with pytest.raises(SystemExit) as exc2:
        check_api_snapshots.main()
    assert exc2.value.code == 0

    # 3. Mismatch (missing symbol) -> exits 1
    monkeypatch.setattr(
        check_api_snapshots,
        "load_ecosystem_snapshot",
        lambda: ("2.14.1", {"fn", "missing_fn"}),
    )
    with pytest.raises(SystemExit) as exc3:
        check_api_snapshots.main()
    assert exc3.value.code == 1

    # 4. Execute __main__ block via runpy against live zero_torch and ml-ecosystem-snapshots
    with pytest.raises(SystemExit) as exc4:
        runpy.run_path(
            str(Path(check_api_snapshots.__file__).resolve()), run_name="__main__"
        )
    assert exc4.value.code == 0


def test_check_compliance_verify_ecosystem_snapshot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Covers all branches of `check_compliance.verify_ecosystem_snapshot`.

    Args:
        monkeypatch: Pytest fixture for patching attributes and modules.
        tmp_path: Temporary directory path fixture.
    """
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    # 1. Cache missing and resolver raises ImportError
    orig_import = builtins.__import__

    def fail_resolver_import(
        name: str,
        globals_dict: dict[str, str] | None = None,
        locals_dict: dict[str, str] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> types.ModuleType:
        """Raises `ImportError` when importing `ml_ecosystem_snapshots.cli.resolver`.

        Args:
            name: Module name.
            globals_dict: Globals mapping.
            locals_dict: Locals mapping.
            fromlist: From-import list.
            level: Relative import level.

        Returns:
            Imported module.
        """
        if name == "ml_ecosystem_snapshots.cli.resolver":
            raise ImportError("no resolver")
        return orig_import(name, globals_dict, locals_dict, fromlist, level)

    monkeypatch.delitem(
        sys.modules, "ml_ecosystem_snapshots.cli.resolver", raising=False
    )
    monkeypatch.setattr(builtins, "__import__", fail_resolver_import)
    assert check_compliance.verify_ecosystem_snapshot() == []
    monkeypatch.setattr(builtins, "__import__", orig_import)

    # 2. Resolver returns non-existent path
    fake_resolver_mod = types.ModuleType("ml_ecosystem_snapshots.cli.resolver")
    vars(fake_resolver_mod)["resolve_snapshot_path"] = lambda _name: str(
        tmp_path / "missing.json"
    )
    monkeypatch.setitem(
        sys.modules, "ml_ecosystem_snapshots.cli.resolver", fake_resolver_mod
    )
    assert check_compliance.verify_ecosystem_snapshot() == []

    # 3. Resolver returns existing file with non-dict JSON
    resolved_file = tmp_path / "resolved.json"
    resolved_file.write_text(json.dumps([1, 2]), encoding="utf-8")
    vars(fake_resolver_mod)["resolve_snapshot_path"] = lambda _name: str(resolved_file)
    assert check_compliance.verify_ecosystem_snapshot() == []

    # 4. Cache exists directly, categories not a dict
    cache_dir = tmp_path / ".cache" / "ml_ecosystem_snapshots" / "snapshots"
    cache_dir.mkdir(parents=True)
    cache_file = cache_dir / "torch_v2.14.1_cpu.json"
    cache_file.write_text(json.dumps({"categories": "bad"}), encoding="utf-8")
    assert check_compliance.verify_ecosystem_snapshot() == []

    # 5. Categories with all item variants, TODO_PLAN.md unchecked items, Tensor methods, docstrings, fields, and params
    todo_plan = tmp_path / "TODO_PLAN.md"
    todo_plan.write_text(
        NL.join(
            [
                "# Heading",
                "- [x] `torch.nn.ReLU`",
                "- [ ] `torch.nn.PlannedSym`",
                "- [ ] `zero_torch.nn.CELU.__constants__`",
                "- [ ] `torch.nn.CELU(planned_param)`",
            ]
        )
        + NL,
        encoding="utf-8",
    )
    assert check_compliance.parse_todo_plan_unchecked(todo_plan) == {
        "torch.nn.PlannedSym",
        "torch.nn.CELU.__constants__",
        "torch.nn.CELU(planned_param)",
    }
    assert check_compliance.parse_todo_plan_unchecked(tmp_path / "no_todo.md") == set()

    # Exercise undocumented callable, un-signaturable callable, and non-callable symbol in zero_torch.nn
    import zero_torch.nn as zt_nn

    def undoc_callable() -> None:
        """Temporary docstring cleared at runtime to test missing docstring detection."""
        return None

    undoc_callable.__doc__ = None
    monkeypatch.setitem(vars(zt_nn), "_test_undoc_fn", undoc_callable)
    monkeypatch.setitem(vars(zt_nn), "_test_builtin_int", int)
    monkeypatch.setitem(vars(zt_nn), "_test_non_callable", 42)

    orig_sig = inspect.signature

    def fake_sig(obj: type | types.FunctionType) -> inspect.Signature:
        """Raises `ValueError` for `int` to cover the `except (ValueError, TypeError)` branch.

        Args:
            obj: Callable being inspected.

        Returns:
            Signature object.
        """
        if obj is int:
            raise ValueError("no signature for builtin int")
        return orig_sig(obj)

    monkeypatch.setattr(inspect, "signature", fake_sig)

    payload = {
        "categories": {
            "activation": "not_a_list",
            "dataloader": [
                "not_a_dict",
                {"api_path": 123, "name": "bad"},
                {"api_path": "torch.nn.Tensor", "name": "Tensor"},
                {"api_path": "torch.nn.PlannedSym", "name": "PlannedSym"},
                {"api_path": "torch.Tensor.reshape", "name": "reshape"},
                {
                    "api_path": "torch.Tensor.nonexistent_method",
                    "name": "nonexistent_method",
                },
                {
                    "api_path": "torch.nn._test_undoc_fn",
                    "name": "_test_undoc_fn",
                    "docstring": "Expected doc.",
                    "params": "not_a_list",
                },
                {
                    "api_path": "torch.nn._test_builtin_int",
                    "name": "_test_builtin_int",
                    "docstring": "",
                    "params": [{"name": "x"}],
                },
                {
                    "api_path": "torch.nn._test_non_callable",
                    "name": "_test_non_callable",
                },
                {
                    "api_path": "torch.nn.ReLU",
                    "name": "ReLU",
                    "kind": "class",
                    "docstring": "ReLU doc.",
                    "params": [
                        "not_a_dict",
                        {"name": 123},
                        {"name": "self"},
                        {"name": "inplace"},
                        {"name": "__module__"},
                        {"name": "_missing_const"},
                    ],
                },
                {
                    "api_path": "torch.nn.CELU",
                    "name": "CELU",
                    "kind": "class",
                    "docstring": "CELU doc.",
                    "params": [
                        {"name": "__constants__"},
                        {"name": "planned_param"},
                        {"name": "missing_param"},
                        {"name": "forward"},
                    ],
                },
                {"api_path": "torch.nn.NonExistentSym", "name": "NonExistentSym"},
                {
                    "api_path": "torch.nonexistent_submod.Sym",
                    "name": "Sym",
                },
            ],
        }
    }
    cache_file.write_text(json.dumps(payload), encoding="utf-8")
    missing = check_compliance.verify_ecosystem_snapshot(todo_plan_path=todo_plan)
    assert missing == [
        "torch.Tensor.nonexistent_method",
        "torch.nn._test_undoc_fn: missing docstring",
        "torch.nn._test_builtin_int(x)",
        "torch.nn.ReLU._missing_const",
        "torch.nn.CELU(missing_param)",
        "torch.nn.NonExistentSym",
        "torch.nonexistent_submod.Sym",
    ]


def test_check_compliance_main(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Covers all branches of `check_compliance.main` and `__main__` entrypoint.

    Args:
        monkeypatch: Pytest fixture for patching attributes and working directory.
        tmp_path: Temporary directory path fixture.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(check_compliance, "verify_ecosystem_snapshot", lambda: [])

    # 1. SEMANTIC_PLAN.md does not exist -> exits 0
    with pytest.raises(SystemExit) as exc1:
        check_compliance.main()
    assert exc1.value.code == 0

    # 2. SEMANTIC_PLAN.md has no parsable API targets -> exits 0
    (tmp_path / "SEMANTIC_PLAN.md").write_text(
        NL.join(["## Overview", "## `torch.nn`", "- [x] `123invalid`", "Just text"]),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit) as exc2:
        check_compliance.main()
    assert exc2.value.code == 0

    # 3. Compliance passes, PYTORCH_TODO.md does not exist yet -> writes and exits 1
    plan_pass = NL.join(
        [
            "- [x] `ignored_before_header`",
            "## General",
            "## SecondHeaderWithoutModule",
            "## `torch.nn`",
            "## `torch`",
            "## `torch`",
            "- [x] `abs`",
            "- [x] `123bad`",
            "Not a list item",
        ]
    )
    (tmp_path / "SEMANTIC_PLAN.md").write_text(plan_pass, encoding="utf-8")
    with pytest.raises(SystemExit) as exc3:
        check_compliance.main()
    assert exc3.value.code == 1
    assert (tmp_path / "PYTORCH_TODO.md").exists()

    # 4. Run again when PYTORCH_TODO.md already matches -> exits 0
    with pytest.raises(SystemExit) as exc4:
        check_compliance.main()
    assert exc4.value.code == 0

    # 5. Missing APIs (bad module + missing function) -> exits 1
    plan_fail = NL.join(
        [
            "## `torch.nonexistent_mod`",
            "- [ ] `foo`",
            "## `torch`",
            "- [x] `abs`",
            "- [ ] `missing_fn`",
        ]
    )
    (tmp_path / "SEMANTIC_PLAN.md").write_text(plan_fail, encoding="utf-8")
    with pytest.raises(SystemExit) as exc5:
        check_compliance.main()
    assert exc5.value.code == 1

    # Run again when PYTORCH_TODO.md already matches the failure output -> covers new_todo == current_todo branch
    with pytest.raises(SystemExit) as exc5b:
        check_compliance.main()
    assert exc5b.value.code == 1

    # 6. Exercise empty `f_name` branch inside `for f_name in funcs:` via custom str subclass
    class BlankOnSecondStrip(str):
        """String subclass whose second `.strip()` returns an empty string."""

        def __new__(cls, val: str) -> BlankOnSecondStrip:
            """Creates a new `BlankOnSecondStrip` instance.

            Args:
                val: String value.

            Returns:
                New `BlankOnSecondStrip` instance.
            """
            instance = super().__new__(cls, val)
            instance.calls = 0
            return instance

        calls: int

        def strip(self, chars: str | None = None) -> str:
            """Returns `'abs'` on the first strip and `''` thereafter.

            Args:
                chars: Characters to strip.

            Returns:
                Stripped string.
            """
            _ = chars
            self.calls += 1
            if self.calls == 1:
                return self
            return ""

    orig_match = re.match

    class FakeMatchItem:
        """Fake regex match returning a `BlankOnSecondStrip` group."""

        def group(self, _idx: int) -> str:
            """Returns a custom string whose `.split('/')` contains `BlankOnSecondStrip`.

            Args:
                _idx: Group index.

            Returns:
                Matched string.
            """

            class CustomPart(str):
                """Custom string whose `.split('/')` returns `BlankOnSecondStrip`."""

                def split(
                    self, sep: str | None = None, maxsplit: SupportsIndex = -1
                ) -> list[str]:
                    """Returns a list containing `BlankOnSecondStrip` and `'abs'`.

                    Args:
                        sep: Separator.
                        maxsplit: Max splits.

                    Returns:
                        List of strings.
                    """
                    _ = (sep, maxsplit)
                    return [BlankOnSecondStrip("abs"), "abs"]

            return CustomPart("abs")

    def fake_match(pattern: str, string: str) -> re.Match[str] | FakeMatchItem | None:
        """Intercepts list-item regex matches to inject `FakeMatchItem`.

        Args:
            pattern: Regex pattern.
            string: Input string.

        Returns:
            Match object or None.
        """
        if pattern.startswith("^-") and string.startswith("- [x] `abs`"):
            return FakeMatchItem()
        return orig_match(pattern, string)

    monkeypatch.setattr(re, "match", fake_match)
    (tmp_path / "SEMANTIC_PLAN.md").write_text(
        NL.join(["## `torch`", "- [x] `abs`"]), encoding="utf-8"
    )
    with pytest.raises(SystemExit):
        check_compliance.main()
    monkeypatch.setattr(re, "match", orig_match)

    # 7. Execute `__main__` via runpy when SEMANTIC_PLAN.md is absent
    (tmp_path / "SEMANTIC_PLAN.md").unlink()
    with pytest.raises(SystemExit) as exc7:
        runpy.run_path(
            str(Path(check_compliance.__file__).resolve()), run_name="__main__"
        )
    assert exc7.value.code == 0


def test_check_dependencies(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Covers 100% of functions, lines, and branches in `scripts/check_dependencies.py`.

    Args:
        monkeypatch: Pytest fixture for patching attributes and working directory.
        tmp_path: Temporary directory path fixture.
    """
    monkeypatch.setattr(sys, "version_info", (3, 10, 0), raising=False)
    monkeypatch.setattr(
        sys, "stdlib_module_names", frozenset({"math", "os"}), raising=False
    )
    assert check_dependencies.is_std_lib("") is True
    assert check_dependencies.is_std_lib("importlib") is True
    assert check_dependencies.is_std_lib("math") is True
    assert check_dependencies.is_std_lib("nonexistent_third_party_pkg") is False

    # Force fallback to `common_stdlib` by patching `stdlib_module_names` to empty
    monkeypatch.setattr(sys, "stdlib_module_names", frozenset(), raising=False)
    assert check_dependencies.is_std_lib("os") is True
    assert check_dependencies.is_std_lib("unknown_pkg") is False

    assert (
        check_dependencies.extract_constant_string(ast.Constant(value="abc")) == "abc"
    )
    assert check_dependencies.extract_constant_string(ast.Constant(value=123)) is None

    # SyntaxError file returns True
    syntax_err_file = tmp_path / "syntax_err.py"
    syntax_err_file.write_text("def broken(:" + NL, encoding="utf-8")
    assert check_dependencies.check_file(syntax_err_file) is True

    # Valid file with allowed imports, relative imports, and non-matching calls
    clean_file = tmp_path / "clean.py"
    clean_file.write_text(
        NL.join(
            [
                "import os",
                "import pydantic.main",
                "from . import sibling",
                "from os import path",
                "__import__('os')",
                "builtins.__import__('json')",
                "importlib.import_module('zero_torch')",
                "sys.modules.get('os')",
                "sys.modules['os']",
                "other_func('foo')",
            ]
        )
        + NL,
        encoding="utf-8",
    )
    assert check_dependencies.check_file(clean_file) is True

    # File with all kinds of violations
    bad_file = tmp_path / "bad.py"
    bad_file.write_text(
        NL.join(
            [
                "import disallowed_pkg",
                "from disallowed_pkg2 import something",
                "__import__('disallowed_pkg3')",
                "builtins.__import__('disallowed_pkg4')",
                "__import__(var_name)",
                "__import__()",
                "importlib.import_module('disallowed_pkg5')",
                "sys.modules.get('disallowed_pkg6')",
                "sys.modules.pop(var_name)",
                "sys.modules['disallowed_pkg7']",
                "sys.modules[var_name]",
            ]
        )
        + NL,
        encoding="utf-8",
    )
    assert check_dependencies.check_file(bad_file) is False

    # Test `main()` branches and test-path allowlists
    monkeypatch.chdir(tmp_path)
    # 1. None of `src`, `scripts`, `tests` exist -> returns 0
    assert check_dependencies.main() == 0

    # 2. Only `src` exists first (so `tests` does not exist -> covers `if target_dir.exists():` False branch)
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "ok.py").write_text("import os" + NL, encoding="utf-8")
    with pytest.raises(SystemExit) as exc0:
        check_dependencies.main()
    assert exc0.value.code == 0

    # 2b. `src` and `tests` exist with clean files (including parity test exception) -> exits 0
    parity_dir = tmp_path / "tests" / "pytorch_official"
    parity_dir.mkdir(parents=True)
    parity_file = parity_dir / "test_parity.py"
    parity_file.write_text(
        "import pytest" + NL + "import torch as real_torch" + NL,
        encoding="utf-8",
    )
    assert check_dependencies.check_file(parity_file) is True

    # Non-parity test file importing `torch` or `numpy` is rejected
    non_parity_test = tmp_path / "tests" / "test_other.py"
    non_parity_test.write_text("import numpy as np" + NL, encoding="utf-8")
    assert check_dependencies.check_file(non_parity_test) is False
    non_parity_test.unlink()

    with pytest.raises(SystemExit) as exc1:
        check_dependencies.main()
    assert exc1.value.code == 0

    # 3. `src` has violation -> exits 1
    (src_dir / "bad.py").write_text("import forbidden_mod" + NL, encoding="utf-8")
    with pytest.raises(SystemExit) as exc2:
        check_dependencies.main()
    assert exc2.value.code == 1

    # 4. Run `__main__` via runpy when directories have only clean files
    (src_dir / "bad.py").unlink()
    with pytest.raises(SystemExit) as exc3:
        runpy.run_path(
            str(Path(check_dependencies.__file__).resolve()), run_name="__main__"
        )
    assert exc3.value.code == 0


def test_check_typing_and_docs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Covers 100% of functions, lines, and branches in `scripts/check_typing_and_docs.py`.

    Args:
        monkeypatch: Pytest fixture for patching attributes and working directory.
        tmp_path: Temporary directory path fixture.
    """
    assert check_typing_and_docs._is_none_return(None) is True
    assert check_typing_and_docs._is_none_return(ast.Constant(value=None)) is True
    assert check_typing_and_docs._is_none_return(ast.Constant(value="None")) is True
    assert check_typing_and_docs._is_none_return(ast.Name(id="None")) is True
    assert check_typing_and_docs._is_none_return(ast.Name(id="NoReturn")) is True
    assert check_typing_and_docs._is_none_return(ast.Name(id="int")) is False

    doc_sample = NL.join(
        [
            "Summary.",
            "",
            "Args:",
            "    a (int): First arg.",
            "      continuation line",
            "",
            "    *args: Varargs.",
            "    **kwargs: Kwargs.",
            "Returns:",
            "    int: Result.",
        ]
    )
    assert check_typing_and_docs._extract_documented_args(doc_sample) == {
        "a",
        "args",
        "kwargs",
    }

    # Test file containing every possible violation in `check_typing_and_docs.check_file`
    viol_file = tmp_path / "violations.py"
    ti_tag = ": ".join(["type", "ignore"])
    pnc_tag = ": ".join(["pragma", "no cover"])
    viol_lines = [
        "# " + ti_tag,
        "# " + pnc_tag,
        "from typing import " + "Any, " + "cast, List",
        "x = " + "Any",
        "d = " + "__dict__",
        "a = obj." + "__dict__",
        "c = " + "cast(int, 1)",
        "g = typing." + "cast(int, 1)",
        "ok_call = len([])",
        "_" + "pass = 1",
        "y = 1",
        "class UndocClass:",
        "    pass",
        "class DocClass:",
        '    """Doc."""',
        "    def " + "__getattr__(self, name: str) -> int:",
        '        """Doc.'
        + NL
        + NL
        + "        Args:"
        + NL
        + "            name: n."
        + NL
        + NL
        + "        Returns:"
        + NL
        + "            int."
        + NL
        + '        """',
        "        return 1",
        "    @overload",
        "    def ov(self, x: int) -> int: ...",
        "def undoc_fn(a, b: " + "object, *args, **kwargs):",
        "    return 1",
        "def bad_doc_fn(a: int, *args: int, **kwargs: int) -> int:",
        '    """Summary without Args or Returns."""',
        "    return a",
        "def bad_tg_true(x: int) -> TypeGuard[int]:",
        '    """Doc.'
        + NL
        + NL
        + "    Args:"
        + NL
        + "        x: val."
        + NL
        + NL
        + "    Returns:"
        + NL
        + "        bool."
        + NL
        + '    """',
        "    x = 1",
        "    return True",
        "def bad_tg_boolop(x: int) -> TypeGuard[int]:",
        '    """Doc.'
        + NL
        + NL
        + "    Args:"
        + NL
        + "        x: val."
        + NL
        + NL
        + "    Returns:"
        + NL
        + "        bool."
        + NL
        + '    """',
        "    return x > 0 or True",
    ]
    viol_file.write_text(NL.join(viol_lines) + NL, encoding="utf-8")
    errs = check_typing_and_docs.check_file(viol_file)
    assert len(errs) >= 18

    # Clean file with check_typing_and_docs.py filename exemption
    exempt_dir = tmp_path / "exempt"
    exempt_dir.mkdir()
    exempt_file = exempt_dir / "check_typing_and_docs.py"
    exempt_file.write_text(
        NL.join(
            [
                '"""Module doc."""',
                "# " + ti_tag,
                "# " + pnc_tag,
            ]
        )
        + NL,
        encoding="utf-8",
    )
    assert check_typing_and_docs.check_file(exempt_file) == []

    # Test `main()` in a temporary workspace
    monkeypatch.chdir(tmp_path)
    # 1. No target dirs exist -> passes and exits 0
    with pytest.raises(SystemExit) as exc1:
        check_typing_and_docs.main()
    assert exc1.value.code == 0

    # 2. Target dir exists with > 200 violations -> prints truncation message and exits 1
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    many_viols = ["x = " + "Any" for _ in range(205)]
    (scripts_dir / "many.py").write_text(NL.join(many_viols) + NL, encoding="utf-8")
    with pytest.raises(SystemExit) as exc2:
        check_typing_and_docs.main()
    assert exc2.value.code == 1

    # 2b. Target dir exists with <= 200 violations -> exits 1 without truncation message
    (scripts_dir / "many.py").write_text("x = " + "Any" + NL, encoding="utf-8")
    with pytest.raises(SystemExit) as exc2b:
        check_typing_and_docs.main()
    assert exc2b.value.code == 1

    # 3. Clean file in scripts_dir -> exits 0 via runpy `__main__`
    (scripts_dir / "many.py").write_text('"""Clean module."""' + NL, encoding="utf-8")
    with pytest.raises(SystemExit) as exc3:
        runpy.run_path(
            str(Path(check_typing_and_docs.__file__).resolve()), run_name="__main__"
        )
    assert exc3.value.code == 0


def test_update_badges(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Covers 100% of functions, lines, and branches in `scripts/update_badges.py`.

    Args:
        monkeypatch: Pytest fixture for patching attributes and working directory.
        tmp_path: Temporary directory path fixture.
    """
    assert update_badges.get_color(100.0) == "brightgreen"
    assert update_badges.get_color(95.0) == "green"
    assert update_badges.get_color(85.0) == "yellowgreen"
    assert update_badges.get_color(75.0) == "yellow"
    assert update_badges.get_color(65.0) == "orange"
    assert update_badges.get_color(50.0) == "red"

    assert update_badges.format_cov(100.0) == "100"
    assert update_badges.format_cov(98.5) == "98.5"
    assert update_badges.get_doc_coverage() == 100.0

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0),
    )

    # get_test_coverage branches
    # 1. FileNotFoundError -> 0.0
    assert update_badges.get_test_coverage() == 0.0

    # 2. Non-dict JSON -> 0.0
    cov_json = tmp_path / "coverage.json"
    cov_json.write_text(json.dumps([1, 2]), encoding="utf-8")
    assert update_badges.get_test_coverage() == 0.0

    # 3. Dict without dict totals -> 0.0
    cov_json.write_text(json.dumps({"totals": "bad"}), encoding="utf-8")
    assert update_badges.get_test_coverage() == 0.0

    # 4. Totals without numeric percent_covered -> 0.0
    cov_json.write_text(
        json.dumps({"totals": {"percent_covered": "bad"}}), encoding="utf-8"
    )
    assert update_badges.get_test_coverage() == 0.0

    # 5. Valid percent_covered -> float
    cov_json.write_text(
        json.dumps({"totals": {"percent_covered": 100.0}}), encoding="utf-8"
    )
    assert update_badges.get_test_coverage() == 100.0

    # get_api_compliance branches
    # 1. No *_TODO.md files -> 0.0
    assert update_badges.get_api_compliance() == 0.0

    # 2. *_TODO.md without Overall Compliance line -> 0.0
    todo_file = tmp_path / "PYTORCH_TODO.md"
    todo_file.write_text(NL.join(["# Header", "No compliance line"]), encoding="utf-8")
    assert update_badges.get_api_compliance() == 0.0

    # 3. *_TODO.md with invalid float -> catches ValueError -> 0.0
    todo_file.write_text("Overall Compliance: bad%" + NL, encoding="utf-8")
    assert update_badges.get_api_compliance() == 0.0

    # 4. *_TODO.md with valid compliance -> returns float
    todo_file.write_text("Overall Compliance: 100.0%" + NL, encoding="utf-8")
    assert update_badges.get_api_compliance() == 100.0

    # update_readme branches
    # 1. No README.md -> returns early
    update_badges.update_readme()

    # 2. README.md with existing Test, Doc, and API Compliance badges
    readme = tmp_path / "README.md"
    readme.write_text(
        NL.join(
            [
                "[![Test Coverage](https://img.shields.io/badge/test_coverage-0%25-red.svg)](#)",
                "[![Doc Coverage](https://img.shields.io/badge/doc_coverage-0%25-red.svg)](#)",
                "[![API Compliance](https://img.shields.io/badge/api_compliance-0%25-red.svg)](#)",
            ]
        )
        + NL,
        encoding="utf-8",
    )
    update_badges.update_readme()
    updated = readme.read_text(encoding="utf-8")
    assert "test_coverage-100%25-brightgreen.svg" in updated
    assert "doc_coverage-100%25-brightgreen.svg" in updated
    assert "api_compliance-100%25-brightgreen.svg" in updated

    # 3. README.md without API Compliance badge, with Doc Coverage badge -> inserts after Doc Coverage
    readme.write_text(
        "[![Doc Coverage](https://img.shields.io/badge/doc_coverage-100%25-brightgreen.svg)](#)"
        + NL,
        encoding="utf-8",
    )
    update_badges.update_readme()
    assert "api_compliance-100%25-brightgreen.svg" in readme.read_text(encoding="utf-8")

    # 4. README.md without API Compliance badge and without Doc Coverage badge -> via runpy `__main__`
    readme.write_text("# Title only" + NL, encoding="utf-8")
    runpy.run_path(str(Path(update_badges.__file__).resolve()), run_name="__main__")
    assert readme.read_text(encoding="utf-8") == "# Title only" + NL
