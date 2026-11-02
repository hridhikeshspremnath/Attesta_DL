"""Choose which files of a package to keep as text (priority-ordered, budgeted)."""
from __future__ import annotations

import json
import posixpath
import re

BINARY_EXT = {
    ".so", ".pyd", ".dll", ".dylib", ".exe", ".bin", ".node", ".o", ".a", ".class", ".jar", ".whl",
    ".zip", ".gz", ".tgz", ".bz2", ".xz", ".7z", ".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf",
    ".woff", ".woff2", ".ttf", ".eot", ".mp3", ".mp4", ".wasm", ".pyc", ".pyo", ".db", ".sqlite",
}
INSTALL_KEYS = ("preinstall", "install", "postinstall", "prepare", "prepublish", "preuninstall")
_PATH_TOKEN = re.compile(r"[\w@./\\-]+\.(?:js|mjs|cjs|sh|py|ps1|bat)\b")


def ext(path: str) -> str:
    return posixpath.splitext(path.lower())[1]


def find_root(names: list[str], eco: str) -> str:
    """Directory prefix that holds the package proper (strips wrapper dirs)."""
    markers = ("package.json",) if eco == "npm" else ("setup.py", "pyproject.toml", "setup.cfg", "PKG-INFO")
    cands = []
    for n in names:
        if "node_modules/" in n:
            continue
        base = posixpath.basename(n)
        if base in markers:
            d = posixpath.dirname(n)
            # DD pypi stores the pip metadata dump next to the package; egg-info isn't a root
            if d.endswith(".egg-info") or d.endswith(".dist-info"):
                continue
            cands.append(d)
    if not cands:
        return ""
    best = min(cands, key=lambda d: (d.count("/"), len(d)))
    return best + "/" if best else ""


def _rel(names: list[str], root: str) -> dict[str, str]:
    return {n[len(root):]: n for n in names if n.startswith(root)}


def _npm_priority(rel: dict[str, str], pkg_json: dict) -> list[str]:
    order: list[str] = ["package.json"]
    scripts = pkg_json.get("scripts") if isinstance(pkg_json.get("scripts"), dict) else {}
    refs: list[str] = []
    for k in INSTALL_KEYS:
        if isinstance(scripts.get(k), str):
            refs += [posixpath.normpath(t.replace("\\", "/")) for t in _PATH_TOKEN.findall(scripts[k])]
    binf = pkg_json.get("bin")
    if isinstance(binf, str):
        refs.append(binf)
    elif isinstance(binf, dict):
        refs += [v for v in binf.values() if isinstance(v, str)]
    for k in ("main", "module", "browser"):
        if isinstance(pkg_json.get(k), str):
            refs.append(pkg_json[k])
    for r in refs:
        r = posixpath.normpath(r.lstrip("./"))
        for cand in (r, r + ".js", posixpath.join(r, "index.js")):
            if cand in rel and cand not in order:
                order.append(cand)
    if "index.js" in rel and "index.js" not in order:
        order.append("index.js")
    rest = [p for p in rel if ext(p) in (".js", ".mjs", ".cjs") and p not in order and not p.endswith(".d.ts")]
    rest.sort(key=lambda p: (("test" in p.lower()) or ("spec" in p.lower()), p.count("/"), p))
    return order + rest


def _py_entry_modules(texts: dict[str, str]) -> list[str]:
    mods = []
    for t in texts.values():
        for m in re.findall(r"['\"][\w.-]+\s*=\s*([\w.]+):[\w.]+['\"]", t):
            mods.append(m.replace(".", "/") + ".py")
    return mods


def _py_priority(rel: dict[str, str], read_small) -> list[str]:
    order = [p for p in ("setup.py", "pyproject.toml", "setup.cfg") if p in rel]
    entry = _py_entry_modules({p: read_small(rel[p]) for p in order})
    inits = sorted((p for p in rel if posixpath.basename(p) == "__init__.py"), key=lambda p: (p.count("/"), p))
    for p in inits[:3] + entry:
        for cand in (p, "src/" + p):
            if cand in rel and cand not in order:
                order.append(cand)
    rest = [p for p in rel if p.endswith(".py") and p not in order]
    rest.sort(key=lambda p: (("test" in p.lower()), p.count("/"), p))
    return order + rest


def select_files(arc, eco: str, limits: dict) -> dict:
    """Return {files, binaries, root, n_members, size_bytes, pkg_json}; text only, budgeted."""
    names = list(arc.sizes)
    root = find_root(names, eco)
    rel = _rel(names, root)
    size_bytes = sum(arc.sizes.values())
    binaries: dict[str, int] = {}
    for p in rel:
        e = ext(p)
        if e in BINARY_EXT:
            binaries[e] = binaries.get(e, 0) + 1

    def read_small(member: str) -> str:
        return arc.read(member, 20_000).decode("utf8", "replace")

    pkg_json: dict = {}
    if eco == "npm":
        if "package.json" in rel:
            try:
                pkg_json = json.loads(arc.read(rel["package.json"], 200_000).decode("utf8", "replace"))
            except Exception:
                pkg_json = {}
        order = _npm_priority(rel, pkg_json if isinstance(pkg_json, dict) else {})
    else:
        order = _py_priority(rel, read_small)

    files, total = [], 0
    for p in order:
        if len(files) >= limits["max_files"] or total >= limits["max_total_bytes"]:
            break
        if ext(p) in BINARY_EXT:
            continue
        cap = min(limits["max_bytes_per_file"], limits["max_total_bytes"] - total)
        raw = arc.read(rel[p], cap + 1)
        if b"\0" in raw[:2000]:
            binaries["<nul>"] = binaries.get("<nul>", 0) + 1
            continue
        trunc = len(raw) > cap
        text = raw[:cap].decode("utf8", "replace")
        files.append({"path": p, "text": text, "truncated": trunc, "orig_size": arc.sizes[rel[p]]})
        total += len(raw[:cap])
    return {"files": files, "binaries": binaries, "root": root, "n_members": len(rel),
            "size_bytes": size_bytes, "pkg_json": pkg_json if isinstance(pkg_json, dict) else {}}
