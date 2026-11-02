"""CPU gradient-boosting baseline on the numeric features (same val/test table, scores saved).

Run: python -m src.train.run_hgb [--noshort]   (iteration count chosen on val PR-AUC)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score

from src.train.data import ECO, load_split
from src.train.report import md, save_scores, table
from src.utils.common import ROOT

SHORTCUTS = ["total_chars", "n_files", "n_binary_types", "name_len", "min_dist_top_n", "min_dist_known_mal"]


def design(root: Path, split: str, drop: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    df, f, names = load_split(root, split, drop)
    x = np.column_stack([f, df.ecosystem.map(ECO).to_numpy()])
    return df, x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--noshort", action="store_true", help="drop size/count/name shortcut features")
    ap.add_argument("--config", default="configs/m2.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / args.config).read_text())
    root, drop = Path(cfg["data"]["processed"]), (SHORTCUTS if args.noshort else [])
    name = "hgb_noshort" if args.noshort else "hgb_full"
    tr, xtr = design(root, "train", drop)
    va, xva = design(root, "val", drop)
    te, xte = design(root, "test", drop)
    clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0,
                                         early_stopping=False, random_state=13).fit(xtr, tr.label)
    ap_by_iter = [average_precision_score(va.label, p[:, 1]) for p in clf.staged_predict_proba(xva)]
    best = int(np.argmax(ap_by_iter)) + 1  # iteration count picked on val only
    print(f"{name}: best_iter={best} val_pr_auc={max(ap_by_iter):.4f}", flush=True)

    def score(x: np.ndarray) -> np.ndarray:
        for i, p in enumerate(clf.staged_predict_proba(x), 1):
            if i == best:
                return p[:, 1]
        raise RuntimeError("unreachable")

    frames = {}
    for sp, df, x in (("val", va, xva), ("test", te, xte)):
        frames[sp] = {"y": df.label.to_numpy(), "s": score(x), "eco": df.ecosystem.map(ECO).to_numpy(),
                      "is_compromised": df.is_compromised.to_numpy(), "cluster_id": df.cluster_id.to_numpy()}
    save_scores(ROOT / cfg["eval"].get("scores", "reports/scores") / f"{name}.parquet", frames)
    out = []
    for sp, d in frames.items():
        t = table(pd.DataFrame({"y": d["y"], "score": d["s"], "eco": d["eco"], "is_compromised": d["is_compromised"],
                                "cluster_id": d["cluster_id"]}))
        out.append(f"**{sp}**\n\n{md(t)}")
    text = "\n\n".join(out)
    rep = ROOT / "reports" / f"{name}.md"
    rep.parent.mkdir(exist_ok=True)
    rep.write_text(f"# {name}\n\n{text}\n")
    print(text)


if __name__ == "__main__":
    main()
