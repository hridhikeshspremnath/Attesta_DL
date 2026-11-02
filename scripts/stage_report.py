"""Stage 5: reports/dataset_summary.md + sanity checks (shortcut-leakage AUC, cluster overlap, spot samples)."""
from __future__ import annotations

import gzip
import json
import random

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from src.labels.run import LABELS
from src.utils.common import ROOT, data_dir, load_config

AUX = list(LABELS)
META = ["package", "version", "ecosystem", "label", *AUX, "is_compromised", "cluster_id", "source_dataset",
        "popularity_bucket", "size_bytes", "upload_date"]


def load(out) -> pd.DataFrame:
    parts = []
    for s in ("train", "val", "test"):
        p = out / f"{s}.parquet"
        if p.exists():
            part = pd.read_parquet(p, columns=META + ["features"]).assign(split=s)
            part["total_chars"] = part.pop("features").map(lambda d: d["total_chars"])
            parts.append(part)
    return pd.concat(parts, ignore_index=True)


def md(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    return df.to_markdown(floatfmt=floatfmt)


def baseline_auc(df: pd.DataFrame, cols: list[str]) -> float:
    X = df[cols].copy()
    for c in ("size_bytes", "total_chars"):
        if c in X:
            X[c] = np.log1p(X[c])
    X = pd.get_dummies(X)
    tr, te = (df.split == "train").values, (df.split == "test").values
    if te.sum() == 0 or df.label[te].nunique() < 2:
        return float("nan")
    m = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=0).fit(X[tr], df.label[tr])
    return float(roc_auc_score(df.label[te], m.predict_proba(X[te])[:, 1]))


def main() -> None:
    cfg = load_config()
    out, rep = data_dir(cfg, "processed"), ROOT / cfg["paths"]["reports"]
    df = load(out)
    stats = json.loads((rep / "build_stats.json").read_text())
    sec = []
    counts = df.groupby(["ecosystem", "split", "label"]).size().unstack("label", fill_value=0)
    counts.columns = ["benign", "malicious"]
    sec += ["## Class counts (ecosystem x split)", md(counts, ".0f")]
    rates = df.groupby(["ecosystem", "label"])[AUX].mean()
    rates.index = rates.index.set_levels(["benign", "malicious"], level=1)
    sec += ["## Aux label rates by class", md(rates)]
    ps = pd.DataFrame([(*k.split("|"), p, f) for k, (p, f) in stats["parse"].items()],
                      columns=["eco", "label", "aux", "ast", "regex"])
    ps = ps.groupby(["eco", "aux"])[["ast", "regex"]].sum()
    ps["ast_success"] = ps.ast / (ps.ast + ps.regex)
    sec += ["## Parse success (AST vs regex fallback, per file)", md(ps)]
    skew = df.groupby(["ecosystem", "label"]).size_bytes.describe()[["50%", "mean", "max"]]
    pop = pd.crosstab([df.ecosystem, df.label], df.popularity_bucket, normalize="index")
    sec += ["## Size skew (bytes)", md(skew, ".0f"), "## Popularity bucket share by class", md(pop)]
    comp = df[df.is_compromised].groupby(["ecosystem", "label"])[AUX].agg(["size"]).iloc[:, :1]
    comp.columns = ["n"]
    sec += ["## Compromised-package group (report separately)", md(comp, ".0f")]

    a_all = baseline_auc(df, ["size_bytes", "popularity_bucket"])
    a_size, a_pop = baseline_auc(df, ["size_bytes"]), baseline_auc(df, ["popularity_bucket"])
    a_chars = baseline_auc(df, ["total_chars"])  # near-empty benign were dropped; malicious were not
    sets = {s: set(g.cluster_id) for s, g in df.groupby("split")}
    overlaps = {f"{a}&{b}": len(sets[a] & sets[b]) for a in sets for b in sets if a < b}
    pk = {s: set(zip(g.ecosystem, g.package)) for s, g in df.groupby("split")}
    pk_over = {f"{a}&{b}": len(pk[a] & pk[b]) for a in pk for b in pk if a < b}
    sec += ["## Sanity checks",
            f"- Shortcut AUC (train->test): size+popularity **{a_all:.3f}**, size only {a_size:.3f}, popularity only {a_pop:.3f}",
            f"- total_chars only (near-empty-filter shortcut check): **{a_chars:.3f}**",
            f"- Cluster overlap across splits: {overlaps}; package overlap: {pk_over}",
            f"- Exact-hash label conflicts: {stats.get('label_conflicts')}; dropped by cluster cap: {stats.get('dropped_by_cluster_cap')}"]
    evid: dict[tuple, list] = {}
    with gzip.open(out / "evidence.jsonl.gz", "rt", encoding="utf8") as f:
        for line in f:
            e = json.loads(line)
            evid.setdefault((e["eco"], e["package"], e["version"]), []).append(f"{e['label']}:{e['detail'][:50]}")
    rng = random.Random(3)
    spot = []
    for lab in (1, 0):
        for _, r in df[df.label == lab].sample(3, random_state=rng.randint(0, 999)).iterrows():
            ev = evid.get((r.ecosystem, r.package, r.version), [])
            spot.append({"label": lab, "eco": r.ecosystem, "pkg": r.package[:24], "ver": r.version[:12],
                         **{k.replace("aux_", ""): int(r[k]) for k in AUX}, "evidence": (ev[0] if ev else "-")})
    sec += ["## Spot check (3 malicious + 3 benign)", md(pd.DataFrame(spot).set_index("label"))]
    (rep / "dataset_summary.md").write_text("# Attesta dataset summary\n\n" + "\n\n".join(sec) + "\n", encoding="utf8")
    print("\n\n".join(sec))


if __name__ == "__main__":
    main()
