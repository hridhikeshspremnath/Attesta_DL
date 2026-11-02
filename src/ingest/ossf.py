"""OSSF OSV reports -> exclusion index + known-malicious name list (no training use).

Rule (decision 4): report without versions -> exclude whole package name from benign;
report with versions -> exclude only those versions.
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.utils.common import lp, norm_name


def _scan_dir(args: tuple[str, str]) -> list[tuple[str, str, list[str]]]:
    eco, d = args
    out = []
    for root, _dirs, files in os.walk(d):
        for f in files:
            if not f.endswith(".json"):
                continue
            try:
                with open(root + "\\" + f, encoding="utf8") as fh:
                    rep = json.load(fh)
            except Exception:
                continue
            for a in rep.get("affected") or []:
                name = (a.get("package") or {}).get("name")
                if name:
                    out.append((eco, norm_name(eco, name), list(a.get("versions") or [])))
    return out


def build_ossf_index(raw: Path, out: Path, workers: int = 16) -> dict:
    if out.exists():
        return json.loads(out.read_text(encoding="utf8"))
    base = lp(raw / "malicious-packages" / "osv" / "malicious")
    jobs = []
    for eco in ("pypi", "npm"):
        for sub in os.listdir(base + "\\" + eco):
            jobs.append((eco, f"{base}\\{eco}\\{sub}"))
    whole: dict[str, set] = {"pypi": set(), "npm": set()}
    vers: dict[str, dict[str, set]] = {"pypi": {}, "npm": {}}
    with ThreadPoolExecutor(workers) as ex:
        for rows in ex.map(_scan_dir, jobs):
            for eco, name, vs in rows:
                if vs:
                    vers[eco].setdefault(name, set()).update(vs)
                else:
                    whole[eco].add(name)
    idx = {eco: {"whole": sorted(whole[eco]), "versions": {k: sorted(v) for k, v in vers[eco].items()},
                 "names": sorted(whole[eco] | set(vers[eco]))} for eco in whole}
    out.write_text(json.dumps(idx), encoding="utf8")
    return idx


def load_unmergable(raw: Path) -> dict[str, dict]:
    """osv/unmergable/ reports as exclusions, same rule as OSSF proper: versions listed -> only those
    versions; none listed -> the whole name. Returns {eco: {"whole": set, "versions": {name: set}}}."""
    out = {e: {"whole": set(), "versions": {}} for e in ("pypi", "npm")}
    base = lp(raw / "malicious-packages" / "osv" / "unmergable")
    for root, _d, files in os.walk(base):
        eco = "pypi" if "\\pypi" in root else "npm"
        for f in files:
            if f.endswith(".json"):
                with open(root + "\\" + f, encoding="utf8") as fh:
                    for a in json.load(fh).get("affected") or []:
                        name = norm_name(eco, a["package"]["name"])
                        vs = a.get("versions") or []
                        if vs:
                            out[eco]["versions"].setdefault(name, set()).update(vs)
                        else:
                            out[eco]["whole"].add(name)
    return out


_WHOLE: dict[tuple, frozenset] = {}


def is_excluded(idx: dict, eco: str, name: str, version: str | None = None) -> bool:
    key = (id(idx), eco)
    if key not in _WHOLE:  # build the lookup set once per index
        _WHOLE[key] = frozenset(idx[eco]["whole"])
    if name in _WHOLE[key]:
        return True
    return version is not None and version in idx[eco]["versions"].get(name, [])


if __name__ == "__main__":
    from src.utils.common import data_dir, load_config, ROOT
    cfg = load_config()
    i = build_ossf_index(ROOT / cfg["paths"]["raw"], data_dir(cfg, "interim") / "ossf_index.json")
    print({e: {k: len(v) for k, v in d.items()} for e, d in i.items()})
