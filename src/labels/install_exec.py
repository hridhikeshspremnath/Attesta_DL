"""aux_install_exec: code that runs automatically at install time."""
from __future__ import annotations

import json
import re

from src.labels.base import LabelOut, ev, rx_hits
from src.labels.facts import Facts

NPM_HOOKS = ("preinstall", "install", "postinstall")
PY_HOOK_BASES = ("install", "develop", "egg_info", "build_py")
PY_DANGER = {"os.system", "os.popen", "exec", "eval", "__import__", "urllib.request.urlopen",
             "requests.get", "requests.post"}
RX_PY = [re.compile(p) for p in (r"cmdclass\s*=", r"\b(os\.system|subprocess\.\w+|exec|eval|urlopen)\s*\(")]
RX_NPM = re.compile(r"\"(pre|post)?install\"\s*:\s*\"([^\"]+)\"")


def _npm(rec: dict, facts: list[Facts]) -> LabelOut:
    out = LabelOut()
    pj = next((f for f in rec["files"] if f["path"] == "package.json"), None)
    if pj is None:
        return out
    try:
        data = json.loads(pj["text"])
        scripts = data.get("scripts") if isinstance(data, dict) else None
        scripts = scripts if isinstance(scripts, dict) else {}
        out.parsed = 1
        hooks = {k: v for k, v in scripts.items() if k in NPM_HOOKS and isinstance(v, str) and v.strip()}
        out.evidence = [{"file": "package.json", "kind": "ast", "detail": f"{k}: {v}"[:120]} for k, v in hooks.items()]
    except ValueError:  # truncated / invalid JSON -> regex fallback
        out.fallback = 1
        out.evidence = [{"file": "package.json", "kind": "regex", "detail": m.group(0)[:120]}
                        for m in RX_NPM.finditer(pj["text"])][:3]
    out.value = bool(out.evidence)
    return out


def _pypi(rec: dict, facts: list[Facts]) -> LabelOut:
    out = LabelOut()
    for f in facts:
        if f.path != "setup.py":
            continue
        if f.ok:
            out.parsed += 1
            hits = []
            if "cmdclass" in f.kwargs:
                hits.append("setup(cmdclass=...)")
            hits += [f"class {b}" for b in f.class_bases if any(h in b.lower() for h in PY_HOOK_BASES)]
            hits += [f"toplevel {c}" for c in f.toplevel_calls if c in PY_DANGER or c.startswith("subprocess.")]
            out.evidence += [ev(f, "ast", h) for h in hits[:3]]
        else:
            out.fallback += 1
            out.evidence += [ev(f, "regex", h) for h in rx_hits(f.text, RX_PY)]
    out.value = bool(out.evidence)
    return out


def aux_install_exec(rec: dict, facts: list[Facts]) -> LabelOut:
    return _npm(rec, facts) if rec["eco"] == "npm" else _pypi(rec, facts)
