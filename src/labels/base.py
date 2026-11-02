"""Shared label plumbing: result type + AST-first/regex-fallback runner."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Iterable

from src.labels.facts import Facts

SNIPPET = 120


@dataclass
class LabelOut:
    value: bool = False
    evidence: list = field(default_factory=list)   # [{file, kind, detail}]
    parsed: int = 0                                # files handled by AST
    fallback: int = 0                              # files handled by regex


def ev(f: Facts, kind: str, detail: str) -> dict:
    return {"file": f.path, "kind": kind, "detail": detail[:SNIPPET]}


def rx_hits(text: str, patterns: Iterable[re.Pattern], limit: int = 3) -> list[str]:
    out = []
    for p in patterns:
        m = p.search(text)
        if m:
            s = max(0, m.start() - 20)
            out.append(text[s:m.end() + 40].replace("\n", " "))
            if len(out) >= limit:
                break
    return out


def run_label(facts: list[Facts], ast_check: Callable[[Facts], list[str]],
              regex_check: Callable[[Facts], list[str]], langs=("py", "js")) -> LabelOut:
    """AST per file when it parsed, else regex on that file; only files of `langs` are inspected."""
    out = LabelOut()
    for f in facts:
        if f.lang not in langs:
            continue
        if f.ok:
            hits, kind = ast_check(f), "ast"
            out.parsed += 1
        else:
            hits, kind = regex_check(f), "regex"
            out.fallback += 1
        out.evidence += [ev(f, kind, h) for h in hits[:3]]
    out.value = bool(out.evidence)
    return out
