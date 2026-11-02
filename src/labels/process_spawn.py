"""aux_process_spawn: spawns OS processes / shells."""
from __future__ import annotations

import re

from src.labels.base import LabelOut, rx_hits, run_label
from src.labels.facts import Facts

PY_EXACT = {"os.system", "os.popen", "pty.spawn", "commands.getoutput", "commands.getstatusoutput",
            "asyncio.create_subprocess_exec", "asyncio.create_subprocess_shell"}
PY_PREFIX = ("subprocess.", "os.exec", "os.spawn", "os.posix_spawn")
JS_FUNCS = {"exec", "execSync", "spawn", "spawnSync", "fork", "execFile", "execFileSync"}
RX = [re.compile(p) for p in (
    r"\bos\.(system|popen|exec\w*|spawn\w*)\s*\(", r"\bsubprocess\.\w+\s*\(", r"\bpty\.spawn\b",
    r"child_process", r"\b(execSync|spawnSync|execFileSync)\s*\(")]


def _ast(f: Facts) -> list[str]:
    if f.lang == "py":
        return [c for c in f.calls if c in PY_EXACT or c.startswith(PY_PREFIX)]
    hits = [f"require {i}" for i in f.imports if i in ("child_process", "node:child_process")]
    hits += [c for c in f.calls if c.split(".")[0] in ("child_process", "node:child_process")
             or (c in JS_FUNCS and "child_process" in f.imports)]
    return hits


def aux_process_spawn(rec: dict, facts: list[Facts]) -> LabelOut:
    return run_label(facts, _ast, lambda f: rx_hits(f.text, RX))
