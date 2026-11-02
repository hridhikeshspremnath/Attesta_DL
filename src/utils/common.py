"""Paths, config, names, IO and hashing helpers shared by all stages."""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path = ROOT / "configs" / "config.yaml") -> dict:
    with open(path, encoding="utf8") as f:
        return yaml.safe_load(f)


def lp(path: str | Path) -> str:
    """Long-path form (\\\\?\\) so Windows handles trailing dots/spaces and >260 chars."""
    s = os.path.abspath(str(path))
    if os.name == "nt" and not s.startswith("\\\\?\\"):
        s = "\\\\?\\" + s
    return s


def data_dir(cfg: dict, key: str) -> Path:
    p = ROOT / cfg["paths"][key]
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---- names -----------------------------------------------------------------
def norm_pypi(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name.strip().lower())


def dd_npm_name(dirname: str) -> str:
    """DataDog stores scoped npm packages as '@scope@name'."""
    if dirname.startswith("@") and dirname.count("@") == 2:
        scope, name = dirname[1:].split("@", 1)
        return f"@{scope}/{name}"
    return dirname


def norm_name(eco: str, name: str) -> str:
    return norm_pypi(name) if eco == "pypi" else name.strip()


# ---- io --------------------------------------------------------------------
def write_jsonl_gz(path: Path, rows: Iterable[dict]) -> int:
    tmp = path.with_suffix(path.suffix + ".tmp")
    n = 0
    with gzip.open(tmp, "wt", encoding="utf8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    tmp.replace(path)  # atomic: a shard exists only when complete
    return n


def read_jsonl_gz(path: Path) -> Iterator[dict]:
    with gzip.open(path, "rt", encoding="utf8") as f:
        for line in f:
            yield json.loads(line)


def iter_shards(directory: Path, pattern: str = "shard-*.jsonl.gz") -> Iterator[dict]:
    for p in sorted(directory.glob(pattern)):
        yield from read_jsonl_gz(p)


# ---- hashing / text stats ---------------------------------------------------
def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf8", "replace")).hexdigest()


def content_hash(files: list[dict]) -> str:
    """Hash over selected files (path-independent of archive layout)."""
    parts = sorted(f"{f['path']}\0{sha256_text(f['text'])}" for f in files)
    return sha256_text("\n".join(parts))


def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    n = len(s)
    return -sum(c / n * math.log2(c / n) for c in Counter(s).values())


def vkey(v: str) -> tuple:
    """Loose version sort key (numeric parts first)."""
    return tuple((0, int(p)) if p.isdigit() else (1, p) for p in re.split(r"[.\-+_]", v))
