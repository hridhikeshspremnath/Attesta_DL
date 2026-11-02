"""Shared results table over a scores frame: groups all/pypi/npm/compromised, optional cluster-bootstrap CIs.

Scores frame columns: y, score, eco (0=pypi, 1=npm), is_compromised, cluster_id (one row per sample, one split).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.train.metrics import cluster_bootstrap, full_summary

SCORE_COLS = ["split", "cluster_id", "y", "eco", "is_compromised", "score"]


def save_scores(path: Path, split_frames: dict[str, dict]) -> None:
    """split_frames: {split: {y, s, eco, is_compromised, cluster_id}} -> one parquet with a `split` column."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [pd.DataFrame({"split": sp, "cluster_id": d["cluster_id"], "y": d["y"].astype(int), "eco": d["eco"],
                          "is_compromised": d["is_compromised"].astype(bool), "score": d["s"]})
            for sp, d in split_frames.items()]
    pd.concat(rows, ignore_index=True)[SCORE_COLS].to_parquet(path, index=False)


def load_scores(path: Path, split: str) -> pd.DataFrame:
    """Reads only rows of `split` (selection code passes 'val'; test is only ever requested by final_report)."""
    d = pd.read_parquet(path, filters=[("split", "==", split)])
    return d.reset_index(drop=True)


def masks(d: pd.DataFrame) -> dict[str, np.ndarray]:
    return {"all": np.ones(len(d), bool), "pypi": (d.eco == 0).to_numpy(), "npm": (d.eco == 1).to_numpy(),
            "compromised": d.is_compromised.to_numpy().astype(bool)}


def table(d: pd.DataFrame, ci: bool = False, reps: int = 1000, seed: int = 13) -> pd.DataFrame:
    rows = []
    for g, m in masks(d).items():
        y, s = d.y.to_numpy()[m], d.score.to_numpy()[m]
        r = {"group": g, **full_summary(y, s)}
        if ci:
            for k, (lo, hi) in cluster_bootstrap(y, s, d.cluster_id.to_numpy()[m], reps, seed).items():
                r[f"{k} 95% CI"] = f"[{lo:.3f}, {hi:.3f}]"
        rows.append(r)
    return pd.DataFrame(rows)


def md(t: pd.DataFrame) -> str:
    return t.to_markdown(index=False, floatfmt=".3f")
