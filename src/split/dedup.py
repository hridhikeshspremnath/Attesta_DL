"""Exact content-hash dedup, per-package version cap, per-cluster cap. Operates on a light index frame."""
from __future__ import annotations

import hashlib

import pandas as pd

from src.utils.common import vkey

SOURCE_PRIORITY = {"datadog": 0, "malregistry": 1, "registry": 2, "malware_bench": 3}


def _h(*parts) -> str:
    return hashlib.md5("|".join(map(str, parts)).encode()).hexdigest()


def exact_dedup(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (eco, content_hash, label); keeps the best-sourced one, counts the rest in dup_count."""
    d = df.assign(_p=df.source_dataset.map(SOURCE_PRIORITY).fillna(9)).sort_values(["_p", "package", "version"])
    g = d.groupby(["eco", "content_hash", "label"], sort=False)
    keep = g.head(1).copy()
    keep["dup_count"] = g.size().reindex(pd.MultiIndex.from_frame(keep[["eco", "content_hash", "label"]])).values
    return keep.drop(columns="_p")


def version_cap(df: pd.DataFrame, cap: int) -> pd.DataFrame:
    """Keep up to `cap` versions per (eco, package, label), evenly spread across the version order."""
    out = []
    for _, g in df.groupby(["eco", "package", "label"], sort=False):
        g = g.sort_values("version", key=lambda s: s.map(vkey)) if len(g) > 1 else g
        if len(g) > cap:
            idx = sorted({round(i * (len(g) - 1) / (cap - 1)) for i in range(cap)}) if cap > 1 else [0]
            g = g.iloc[idx]
        out.append(g)
    return pd.concat(out) if out else df.iloc[:0]


def cluster_cap(df: pd.DataFrame, cap: int, seed: int = 13) -> pd.DataFrame:
    """Keep at most `cap` rows per (cluster_id, label); deterministic hash order."""
    d = df.assign(_o=[_h(seed, a, b, c) for a, b, c in zip(df.eco, df.package, df.version)])
    d = d.sort_values("_o")
    return d.groupby(["cluster_id", "label"], sort=False).head(cap).drop(columns="_o")
