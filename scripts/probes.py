"""Provenance probes (CPU): can metadata/formatting alone separate the classes? Writes reports/provenance_probes.md.

a) upload_date-only baseline + date distribution   b) path-header-only char n-gram LR (raw vs normalised paths)
c) formatting rates by class.   Train on train, AUC on test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.features.normalize import norm_path, split_files
from src.utils.common import ROOT, data_dir, load_config

COLS = ["package", "ecosystem", "label", "text", "upload_date", "source_dataset"]


def load(out) -> pd.DataFrame:
    return pd.concat([pd.read_parquet(out / f"{s}.parquet", columns=COLS).assign(split=s) for s in ("train", "val", "test")],
                     ignore_index=True)


def auc(model, xtr, ytr, xte, yte) -> float:
    return float(roc_auc_score(yte, model.fit(xtr, ytr).predict_proba(xte)[:, 1]))


def probe_dates(df: pd.DataFrame) -> tuple[str, dict]:
    year = pd.to_numeric(df.upload_date.str[:4], errors="coerce")
    dist = pd.crosstab(year.fillna(-1).astype(int).replace(-1, "NA"), [df.ecosystem, df.label.map({0: "ben", 1: "mal"})])
    x = pd.DataFrame({"year": year.fillna(-1), "eco": (df.ecosystem == "npm").astype(int), "na": year.isna().astype(int)})
    tr, te = (df.split == "train").to_numpy(), (df.split == "test").to_numpy()
    res = {"year+eco": auc(HistGradientBoostingClassifier(max_iter=100, random_state=0), x[tr], df.label[tr], x[te], df.label[te])}
    for eco in ("pypi", "npm"):
        m_tr, m_te = tr & (df.ecosystem == eco).to_numpy(), te & (df.ecosystem == eco).to_numpy()
        res[f"year ({eco} only)"] = auc(HistGradientBoostingClassifier(max_iter=100, random_state=0),
                                        x[m_tr][["year", "na"]], df.label[m_tr], x[m_te][["year", "na"]], df.label[m_te])
    return dist.to_markdown(), res


def headers(text: str, normalized: bool) -> str:
    ps = [p for p, _ in split_files(text)]
    return "\n".join(norm_path(p) for p in ps) if normalized else "\n".join(ps)


def probe_paths(df: pd.DataFrame) -> dict:
    tr, te = (df.split == "train").to_numpy(), (df.split == "test").to_numpy()
    res = {}
    for norm in (False, True):
        h = df.text.map(lambda t: headers(t, norm))
        vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=3, sublinear_tf=True, max_features=300_000)
        xtr, xte = vec.fit_transform(h[tr]), vec.transform(h[te])
        res["path headers, " + ("normalised (tmp/package/top-dir stripped)" if norm else "raw")] = auc(
            LogisticRegression(C=2.0, max_iter=400), xtr, df.label[tr], xte, df.label[te])
    return res


def fmt_rates(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (eco, lab), g in df.groupby(["ecosystem", "label"]):
        nl, nf = 0, 0
        for t in g.text:
            fs = split_files(t)
            if len(t) >= 31_900:  # last block was cut by the text cap: its newline state is unknown
                fs = fs[:-1]
            nl += sum(b.endswith("\n\n") for _, b in fs)  # body = file text + one "\n" added by build_text
            nf += len(fs)
        L = g.text.str.len()
        rows.append({"eco": eco, "class": "mal" if lab else "ben", "n": len(g),
                     "CRLF": g.text.str.contains("\r\n", regex=False).mean(), "BOM": g.text.str.contains("﻿", regex=False).mean(),
                     "tabs": g.text.str.contains("\t", regex=False).mean(), "non-ASCII": (~g.text.map(str.isascii)).mean(),
                     "file ends \\n": nl / max(nf, 1), "len median": L.median(), "len p90": L.quantile(0.9)})
    return pd.DataFrame(rows)


def main() -> None:
    cfg = load_config()
    out = data_dir(cfg, "processed")
    df = load(out)
    dist, dres = probe_dates(df)
    pres = probe_paths(df)
    src = pd.crosstab([df.ecosystem, df.label.map({0: "ben", 1: "mal"})], df.source_dataset)
    auc_t = pd.DataFrame({"probe": list(dres) + list(pres), "test AUC": list(dres.values()) + list(pres.values())})
    txt = "\n\n".join(["# Provenance probes (train -> test AUC; 0.5 = no leak)", auc_t.to_markdown(index=False, floatfmt=".3f"),
                       "## upload_date year distribution (all splits)", dist, "## Source dataset by class", src.to_markdown(),
                       "## Formatting rates by class (share of packages; 'file ends \\n' is per file)",
                       fmt_rates(df).to_markdown(index=False, floatfmt=".3f")])
    (ROOT / "reports" / "provenance_probes.md").write_text(txt + "\n", encoding="utf8")
    print(txt)


if __name__ == "__main__":
    main()
