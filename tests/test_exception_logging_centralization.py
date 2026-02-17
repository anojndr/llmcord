from __future__ import annotations

import ast
import re
from pathlib import Path

EXCEPTION_LOG_PATTERNS = (
    "logger.exception(",
    "LOGGER.exception(",
    "logging.exception(",
)
BROAD_EXCEPTION_PATTERN = re.compile(r"except\s+(Exception|BaseException)\b")
BROAD_EXCEPTION_IDENTIFIERS = {"Exception", "BaseException"}


def _iter_src_python_files() -> list[Path]:
    src_root = Path(__file__).resolve().parents[1] / "src"
    return sorted(src_root.rglob("*.py"))


def test_no_direct_exception_logging_calls_in_src() -> None:
    violations: list[str] = []

    src_root = Path(__file__).resolve().parents[1] / "src"
    for python_file in _iter_src_python_files():
        source = python_file.read_text(encoding="utf-8")
        for pattern in EXCEPTION_LOG_PATTERNS:
            if pattern not in source:
                continue
            relative_path = python_file.relative_to(src_root)
            violations.append(f"{relative_path}: {pattern}")

    assert not violations, "\n".join(sorted(violations))


def test_no_broad_exception_handlers_in_src() -> None:
    violations: list[str] = []
    src_root = Path(__file__).resolve().parents[1] / "src"

    for python_file in _iter_src_python_files():
        source = python_file.read_text(encoding="utf-8")
        relative_path = python_file.relative_to(src_root).as_posix()

        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Try):
                continue

            for handler in node.handlers:
                if handler.type is None:
                    violations.append(
                        f"{relative_path}:{handler.lineno}: bare except is not allowed",
                    )
                    continue

                exception_type = handler.type
                if (
                    isinstance(exception_type, ast.Name)
                    and exception_type.id in BROAD_EXCEPTION_IDENTIFIERS
                ):
                    violations.append(
                        (
                            f"{relative_path}:{handler.lineno}: broad catch "
                            f"`except {exception_type.id}` is not allowed"
                        ),
                    )

    assert not violations, "\n".join(sorted(violations))


def test_no_ble001_noqa_markers_in_src() -> None:
    violations: list[str] = []
    src_root = Path(__file__).resolve().parents[1] / "src"

    for python_file in _iter_src_python_files():
        source = python_file.read_text(encoding="utf-8")
        if "noqa: BLE001" not in source:
            continue
        relative_path = python_file.relative_to(src_root).as_posix()
        violations.append(f"{relative_path}: contains disallowed `noqa: BLE001`")

    assert not violations, "\n".join(sorted(violations))


def test_no_textual_broad_exception_patterns_in_src() -> None:
    violations: list[str] = []
    src_root = Path(__file__).resolve().parents[1] / "src"

    for python_file in _iter_src_python_files():
        source = python_file.read_text(encoding="utf-8")
        if not BROAD_EXCEPTION_PATTERN.search(source):
            continue

        relative_path = python_file.relative_to(src_root).as_posix()
        for line_number, line in enumerate(source.splitlines(), start=1):
            if BROAD_EXCEPTION_PATTERN.search(line):
                violations.append(
                    f"{relative_path}:{line_number}: {line.strip()}",
                )

    assert not violations, "\n".join(sorted(violations))
