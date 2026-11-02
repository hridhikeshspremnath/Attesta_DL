"""Provenance-neutral text normalisation for the byte model (`data.normalize: true`).

Input is the stored `text` (blocks of "### <path>\\n<body>\\n"). Removes formatting/layout cues that differ by data
source, not behaviour: CRLF, BOM, trailing-newline state, tmp-dir / top-level-dir path prefixes, archive file order.
Limitation: a source line that itself starts with "### " would be read as a header (rare in selected code files).
"""
from __future__ import annotations

import posixpath
import re

_HDR = re.compile(r"^### (.+)\n", re.M)
_TMP = re.compile(r"^(?:[A-Za-z0-9_.]+/)?tmp[a-z0-9_]{6,}/")  # DataDog build dirs: tmp/tmpq1uzlfp0/...
_VAR = re.compile(r"^(?:private/)?var/folders/[^/]+/[^/]+/T/(?:tmp[^/]*/)?")
_INSTALL_FIRST = {"package.json", "setup.py", "pyproject.toml", "setup.cfg"}


def norm_path(p: str) -> str:
    p = p.replace("\\", "/")
    p = re.sub(r"^(?:\./|/)+", "", p)
    p = _VAR.sub("", _TMP.sub("", p))
    return re.sub(r"^package/", "", p)


def split_files(text: str) -> list[tuple[str, str]]:
    """[(path, body)] from build_text output."""
    parts = _HDR.split(text)
    return list(zip(parts[1::2], parts[2::2]))


def _body(b: str) -> str:
    b = b.replace("\r\n", "\n").replace("\r", "\n").replace("﻿", "")
    return b.rstrip("\n") + "\n"


def _rank(path: str) -> tuple[int, str]:
    return (0 if path in _INSTALL_FIRST else 1, path)  # depends on the path only, never on class or archive order


def normalize_text(text: str) -> str:
    files = [(norm_path(p), _body(b)) for p, b in split_files(text)]
    if not files:
        return _body(text) if text else text
    if len(files) > 1 and all("/" in p for p, _ in files) and len({p.split("/")[0] for p, _ in files}) == 1:
        files = [(p.split("/", 1)[1], b) for p, b in files]  # shared top-level dir prefix
    files.sort(key=lambda f: _rank(f[0]))
    return "".join(f"### {posixpath.normpath(p)}\n{b}" for p, b in files)
