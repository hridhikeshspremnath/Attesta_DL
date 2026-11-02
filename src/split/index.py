"""Light per-record index (hash + MinHash signature) so dedup/cluster never hold full texts in memory."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.text import build_text
from src.split.cluster import signature
from src.utils.common import read_jsonl_gz

COLS = ["eco", "package", "version", "label", "source_dataset", "is_compromised", "content_hash", "size_bytes",
        "upload_date", "popularity_bucket", "pool"]


def _shard_index(args: tuple[str, int, int]) -> tuple[list[dict], list[dict]]:
    path, n_perm, k = args
    rows, errs = [], []
    for i, r in enumerate(read_jsonl_gz(Path(path))):
        if "error" in r:
            errs.append({k_: r.get(k_) for k_ in ("eco", "package", "version", "source_dataset", "error")})
            continue
        row = {c: r.get(c) for c in COLS}
        row["pool"] = r.get("pool", "main")
        row["shard"], row["line"] = path, i
        row["sig"] = signature(build_text(r["files"], 60_000), n_perm, k).tobytes()
        row["n_files"] = len(r["files"])
        rows.append(row)
    return rows, errs


def build_index(shard_dirs: list[Path], n_perm: int, k: int, workers: int = 6) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = [str(p) for d in shard_dirs for p in sorted(d.glob("shard-*.jsonl.gz"))]
    rows, errs = [], []
    with ProcessPoolExecutor(workers) as ex:
        for r, e in ex.map(_shard_index, [(p, n_perm, k) for p in paths]):
            rows += r
            errs += e
    return pd.DataFrame(rows), pd.DataFrame(errs)


def sig_matrix(df: pd.DataFrame, n_perm: int) -> np.ndarray:
    return np.vstack([np.frombuffer(b, dtype=np.uint32) for b in df.sig]).reshape(len(df), n_perm)
