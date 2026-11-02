"""Variant selection on VAL ONLY. Run: python -m src.train.select

Rule: per variant, average val PR-AUC over its seeds (all-samples group); pick the highest. A variant within 0.002 of the
best is considered tied and the one with the higher mean val recall@1%FPR wins. Test scores are never loaded here
(`load_scores(..., "val")` filters at read time); test is read once, by `src.train.final_report`.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from src.train.metrics import full_summary
from src.train.report import load_scores
from src.utils.common import ROOT

SCORES = ROOT / "reports" / "scores"
TIE = 0.002


def variant_of(stem: str) -> tuple[str, str]:
    """(variant, run). 'q_noshort_s1' -> 'noshort'; 'm2' (the first full run, seed 13) -> 'full'; 'hgb_full' -> 'hgb_full'."""
    if stem == "m2":
        return "full", stem
    return re.sub(r"_s\d+$", "", re.sub(r"^q_", "", stem)), stem


def val_table(files: list[Path]) -> pd.DataFrame:
    rows = []
    for p in files:
        d = load_scores(p, "val")
        v, run = variant_of(p.stem)
        rows.append({"variant": v, "run": run, **{k: x for k, x in full_summary(d.y.to_numpy(), d.score.to_numpy()).items()
                                                  if k in ("pr_auc", "roc_auc", "rec@1%FPR")}})
    return pd.DataFrame(rows)


def choose(per_run: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    agg = per_run.groupby("variant").agg(runs=("run", "size"), pr_auc=("pr_auc", "mean"), pr_auc_sd=("pr_auc", "std"),
                                         roc_auc=("roc_auc", "mean"), rec1=("rec@1%FPR", "mean")).reset_index()
    top = agg.pr_auc.max()
    tied = agg[agg.pr_auc >= top - TIE]
    return agg.sort_values("pr_auc", ascending=False), str(tied.sort_values("rec1", ascending=False).variant.iloc[0])


def main() -> None:
    files = sorted(SCORES.glob("*.parquet"))
    if not files:
        raise SystemExit(f"no score files in {SCORES}")
    per_run = val_table(files)
    agg, chosen = choose(per_run)
    print(agg.to_markdown(index=False, floatfmt=".3f"))
    print(f"\nchosen (val only): {chosen}")
    runs = per_run[per_run.variant == chosen].run.tolist()
    (ROOT / "reports" / "selection.json").write_text(json.dumps({"variant": chosen, "runs": runs, "rule": __doc__.split("\n")[2]}))


if __name__ == "__main__":
    main()
