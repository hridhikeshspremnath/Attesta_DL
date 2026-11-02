"""One-shot TEST report for the variant chosen on val. Run: python -m src.train.final_report [--reps 1000]

Reads reports/selection.json (written by src.train.select). Refuses to run twice (test is read once) unless --force.
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from src.train.report import load_scores, md, table
from src.utils.common import ROOT

OUT = ROOT / "reports" / "final_test.md"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    sel_path = ROOT / "reports" / "selection.json"
    if not sel_path.exists():
        raise SystemExit("run `python -m src.train.select` first (variant must be chosen on val)")
    if OUT.exists() and not args.force:
        raise SystemExit(f"{OUT} exists: test was already read once. Use --force only to regenerate the same report.")
    sel = json.loads(sel_path.read_text())
    parts = [f"# Final test report\n\nvariant chosen on val: **{sel['variant']}**  |  rule: {sel['rule']}\n"]
    runs = sel["runs"]
    mean_scores = []
    for run in runs:
        d = load_scores(ROOT / "reports" / "scores" / f"{run}.parquet", "test")
        parts.append(f"## {run}\n\n" + md(table(d, ci=True, reps=args.reps)))
        mean_scores.append(d.score.to_numpy())
    if len(runs) > 1:  # seed-ensemble: mean score over runs of the chosen variant (same sample order in every file)
        d = load_scores(ROOT / "reports" / "scores" / f"{runs[0]}.parquet", "test")
        d["score"] = pd.DataFrame(mean_scores).mean().to_numpy()
        parts.append(f"## ensemble of {len(runs)} runs (mean score)\n\n" + md(table(d, ci=True, reps=args.reps)))
    OUT.write_text("\n\n".join(parts) + "\n")
    print("\n\n".join(parts))


if __name__ == "__main__":
    main()
