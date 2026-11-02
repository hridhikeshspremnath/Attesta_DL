"""Enumerate malicious samples (listing only, no reading): DataDog zips + pypi_malregistry tarballs."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from src.utils.common import ROOT, dd_npm_name, lp, norm_name, vkey

DD = "malicious-software-packages-dataset/samples"
MR = "pypi_malregistry"


def index_datadog(raw: Path) -> list[dict]:
    out = []
    for eco in ("npm", "pypi"):
        for kind in ("malicious_intent", "compromised_lib"):
            base = lp(raw / DD / eco / kind)
            for pkg in os.listdir(base):
                pdir = base + "\\" + pkg
                name = norm_name(eco, dd_npm_name(pkg) if eco == "npm" else pkg)
                for ent in os.listdir(pdir):
                    full = pdir + "\\" + ent
                    if os.path.isdir(full):  # <pkg>/<version>/<date>-<pkg>-v<version>.zip
                        zips = [z for z in os.listdir(full) if z.endswith(".zip")]
                        ver, zp = ent, (full + "\\" + zips[0] if zips else None)
                    elif ent.endswith(".zip"):  # flat layout: <pkg>/<date>-<pkg>[-v<version>].zip
                        m = re.search(r"-v(.+)\.zip$", ent)
                        ver, zp = (m.group(1) if m else "unknown"), full
                    else:
                        continue
                    if zp:
                        out.append({"eco": eco, "package": name, "version": ver, "source_dataset": "datadog",
                                    "is_compromised": kind == "compromised_lib", "path": zp})
    return out


def index_malregistry(raw: Path) -> list[dict]:
    out = []
    base = lp(raw / MR)
    for pkg in os.listdir(base):
        pdir = base + "\\" + pkg
        if not os.path.isdir(pdir):
            continue
        for ver in os.listdir(pdir):
            vdir = pdir + "\\" + ver
            if not os.path.isdir(vdir):
                continue
            tars = [t for t in os.listdir(vdir) if t.endswith((".tar.gz", ".tgz", ".zip"))]
            if tars:
                out.append({"eco": "pypi", "package": norm_name("pypi", pkg), "version": ver,
                            "source_dataset": "malregistry", "is_compromised": False,
                            "path": f"{vdir}\\{tars[0]}"})
    return out


def pre_cap(rows: list[dict], cap: int) -> list[dict]:
    """Per (eco, package, source) keep at most `cap` versions, evenly spread. Cheap pre-filter."""
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r["eco"], r["package"]), []).append(r)
    kept = []
    for g in groups.values():
        g.sort(key=lambda r: vkey(r["version"]))
        if len(g) > cap:
            step = (len(g) - 1) / (cap - 1)
            g = [g[round(i * step)] for i in range(cap)]
        kept += g
    return kept


def build_index(raw: Path, cap: int, out: Path) -> list[dict]:
    if out.exists():
        return [json.loads(l) for l in out.read_text(encoding="utf8").splitlines()]
    rows = pre_cap(index_datadog(raw) + index_malregistry(raw), cap)
    rows.sort(key=lambda r: (r["eco"], r["package"], vkey(r["version"]), r["source_dataset"]))
    out.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf8")
    return rows
