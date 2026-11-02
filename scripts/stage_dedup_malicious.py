"""Stage 2: index malicious shards, dedup + caps -> counts used to size the benign set. Prints a short table."""
from __future__ import annotations

import pandas as pd

from src.split.cluster import cluster_ids
from src.split.dedup import cluster_cap, exact_dedup, version_cap
from src.split.index import build_index, sig_matrix
from src.utils.common import ROOT, data_dir, load_config


def main() -> None:
    cfg = load_config()
    interim, d, m = data_dir(cfg, "interim"), cfg["dedup"], cfg["dedup"]["minhash_perm"]
    df, errs = build_index([interim / "malicious"], m, d["shingle"], cfg["ingest"]["workers"])
    df.to_pickle(interim / "index_malicious_raw.pkl")
    errs.to_csv(interim / "ingest_errors.csv", index=False)
    rows = [("indexed", df.groupby("eco").size().to_dict()), ("errors", errs.groupby("eco").size().to_dict() if len(errs) else {})]
    a = exact_dedup(df)
    rows.append(("after exact hash dedup", a.groupby("eco").size().to_dict()))
    b = version_cap(a, d["version_cap"])
    rows.append((f"after {d['version_cap']} versions/pkg", b.groupby("eco").size().to_dict()))
    b = b.reset_index(drop=True)
    b["cluster_id"] = ""
    for eco, g in b.groupby("eco"):
        ids = cluster_ids(sig_matrix(g, m), list(zip(g.eco, g.package)), d["minhash_threshold"])
        b.loc[g.index, "cluster_id"] = [f"{eco}-{i}" for i in ids]
    c = cluster_cap(b, d["cluster_cap"])
    rows.append((f"after {d['cluster_cap']}/cluster cap", c.groupby("eco").size().to_dict()))
    c.to_pickle(interim / "index_malicious_dedup.pkl")
    print(pd.DataFrame(rows, columns=["step", "per_eco"]).to_string(index=False))
    print("clusters:", b.groupby("eco").cluster_id.nunique().to_dict(), "| compromised kept:",
          int(c.is_compromised.sum()), "| source:", c.source_dataset.value_counts().to_dict())
    print("hash label conflicts: n/a (malicious only) | errors by type:",
          errs.error.str.split(":").str[0].value_counts().head(3).to_dict() if len(errs) else {})


if __name__ == "__main__":
    main()
