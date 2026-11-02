"""Apply place_hard_negatives to an already-built dataset (avoids a full rebuild). Rewrites split_index.pkl + parquets."""
from __future__ import annotations

import pandas as pd

from src.split.assign import place_hard_negatives
from src.utils.common import data_dir, load_config


def main() -> None:
    cfg = load_config()
    out = data_dir(cfg, "processed")
    idx = pd.read_pickle(out / "split_index.pkl")
    new = place_hard_negatives(idx, idx["split"], seed=cfg["split"]["seed"])
    moved = new != idx["split"]
    print("rows moved:", int(moved.sum()))
    idx["split"] = new
    mapping = {(r.eco, r.package, r.version): r.split for r in idx.itertuples()}
    parts = [pd.read_parquet(out / f"{s}.parquet") for s in ("train", "val", "test")]
    df = pd.concat(parts, ignore_index=True)
    df["split"] = [mapping[(e, p, v)] for e, p, v in zip(df.ecosystem, df.package, df.version)]
    for s in ("train", "val", "test"):
        df[df.split == s].drop(columns="split").to_parquet(out / f"{s}.parquet", index=False)
    idx.to_pickle(out / "split_index.pkl")
    c = idx[idx.is_compromised.astype(bool)]
    print(c.groupby(["eco", "label", "split"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
