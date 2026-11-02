"""Stage 4: dedup+cluster+split over malicious+benign, then labels/features -> data/processed/{train,val,test}.parquet.

Streams shard by shard (texts are never all in memory). Prints a short table; stats go to reports/build_stats.json.
"""
from __future__ import annotations

import gzip
import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.features.numeric import FEATURE_NAMES
from src.features.pipeline import NAME_FEATS, featurize, load_name_indexes
from src.labels.run import LABELS
from src.split.assign import assign_splits, place_hard_negatives
from src.split.cluster import cluster_ids
from src.split.dedup import cluster_cap, exact_dedup, version_cap
from src.split.index import build_index, sig_matrix
from src.utils.common import ROOT, data_dir, load_config, read_jsonl_gz

SCHEMA = pa.schema([
    ("package", pa.string()), ("version", pa.string()), ("ecosystem", pa.string()), ("label", pa.int8()),
    ("text", pa.string()), ("features", pa.struct([(n, pa.float32()) for n in FEATURE_NAMES + NAME_FEATS])),
    *[(k, pa.int8()) for k in LABELS], ("is_compromised", pa.bool_()), ("cluster_id", pa.string()),
    ("source_dataset", pa.string()), ("popularity_bucket", pa.string()), ("size_bytes", pa.int64()),
    ("upload_date", pa.string())])

_G: dict = {}


def _init(refs_path: str, max_chars: int) -> None:
    _G["top"], _G["mal"] = load_name_indexes(Path(refs_path))
    _G["max_chars"] = max_chars


def work(args: tuple[str, dict]):
    shard, keep = args
    rows, evid, stats = [], [], defaultdict(lambda: [0, 0])
    for i, rec in enumerate(read_jsonl_gz(Path(shard))):
        if i not in keep or "error" in rec:
            continue
        cid, split = keep[i]
        out = featurize(rec, _G["top"], _G["mal"], _G["max_chars"])  # same code path as src.infer
        vals, ev, st, feats = out["aux"], out["evidence"], out["parse"], out["feats"]
        rows.append({"package": rec["package"], "version": rec["version"], "ecosystem": rec["eco"],
                     "label": rec["label"], "text": out["text"], "features": feats,
                     **vals, "is_compromised": bool(rec["is_compromised"]), "cluster_id": cid,
                     "source_dataset": rec["source_dataset"], "popularity_bucket": rec["popularity_bucket"],
                     "size_bytes": rec["size_bytes"], "upload_date": rec["upload_date"], "split": split})
        evid += ev
        for k, (p, f) in st.items():
            stats[f"{rec['eco']}|{rec['label']}|{k}"][0] += p
            stats[f"{rec['eco']}|{rec['label']}|{k}"][1] += f
    return rows, evid, dict(stats)


def build_refs(cfg: dict, interim: Path, cache: Path) -> Path:
    top_n = cfg["features"]["top_n_names"]
    from src.ingest.registry import Client
    cli = Client(cfg["benign"]["user_agent"], cfg["benign"]["rate_per_sec"])
    pypi = [r["project"] for r in cli.json(cfg["benign"]["pypi_top_url"])["rows"][:top_n]]
    npm_pool = json.loads((cache / "npm_pool_enriched.json").read_text(encoding="utf8"))
    npm = [c["package"] for c in sorted(npm_pool, key=lambda c: -(c.get("downloads") or 0))[:top_n]]
    ossf = json.loads((interim / "ossf_index.json").read_text(encoding="utf8"))
    refs = {"pypi": {"top": pypi, "mal": ossf["pypi"]["names"]}, "npm": {"top": npm, "mal": ossf["npm"]["names"]}}
    p = cache / "name_refs.json"
    p.write_text(json.dumps(refs), encoding="utf8")
    return p


def main() -> None:
    cfg = load_config()
    workers = max(1, (os.cpu_count() or 2) - 1)  # leave one core free
    interim, cache, out = data_dir(cfg, "interim"), data_dir(cfg, "cache"), data_dir(cfg, "processed")
    d, m = cfg["dedup"], cfg["dedup"]["minhash_perm"]
    df, errs = build_index([interim / "malicious", interim / "benign"], m, d["shingle"], workers)
    stats = {"indexed": df.groupby(["eco", "label"]).size().rename(str).to_dict().__repr__(),
             "errors": errs.groupby("error").size().sort_values(ascending=False).head(8).to_dict() if len(errs) else {}}
    conflicts = int((df.groupby(["eco", "content_hash"]).label.nunique() > 1).sum())
    a = version_cap(exact_dedup(df), d["version_cap"]).reset_index(drop=True)
    # main vs extra benign is decided here, on packages that survived fetch/dedup: first N (fetch order is
    # stratum-shuffled) fill the 1:1 main pool per ecosystem; hard negatives and malware_bench always count as main
    a["pool"] = "main"
    for eco, g in a[a.label == 0].groupby("eco"):
        n_mal = int(((a.eco == eco) & (a.label == 1)).sum())
        special = g.is_compromised.astype(bool) | (g.source_dataset == "malware_bench")
        rest = g[~special].sort_values(["shard", "line"])
        a.loc[rest.index[max(0, n_mal - int(special.sum())):], "pool"] = "extra"
    a["cluster_id"] = ""
    for eco, g in a.groupby("eco"):
        ids = cluster_ids(sig_matrix(g, m), list(zip(g.eco, g.package)), d["minhash_threshold"])
        a.loc[g.index, "cluster_id"] = [f"{eco}-{i}" for i in ids]
    # cluster cap never starves the 'extra' benign pool or hard negatives of their own cluster slots
    keep = cluster_cap(a, d["cluster_cap"]).reset_index(drop=True)
    keep["split"] = assign_splits(keep, cfg["split"]["fractions"], cfg["split"]["seed"])
    keep["split"] = place_hard_negatives(keep, keep["split"], seed=cfg["split"]["seed"])
    keep.to_pickle(out / "split_index.pkl")
    print("rows kept:", keep.groupby(["split", "eco", "label"]).size().unstack([1, 2], fill_value=0).to_string())
    print("exact-hash label conflicts:", conflicts, "| workers:", workers)

    by_shard: dict[str, dict] = defaultdict(dict)
    for r in keep.itertuples():
        by_shard[r.shard][r.line] = (r.cluster_id, r.split)
    refs = build_refs(cfg, interim, cache)
    writers: dict[str, pq.ParquetWriter] = {}
    evidence = gzip.open(out / "evidence.jsonl.gz", "wt", encoding="utf8")
    agg: dict[str, list] = defaultdict(lambda: [0, 0])
    items = sorted(by_shard.items())
    with ProcessPoolExecutor(workers, initializer=_init,
                             initargs=(str(refs), cfg["features"]["max_text_chars"])) as ex:
        for n, (rows, evid, st) in enumerate(ex.map(work, items), 1):
            for s in {r["split"] for r in rows}:
                part = pd.DataFrame([{k: v for k, v in r.items() if k != "split"} for r in rows if r["split"] == s])
                t = pa.Table.from_pandas(part, schema=SCHEMA, preserve_index=False)
                if s not in writers:  # setdefault would construct (and truncate the file for) a writer on every call
                    writers[s] = pq.ParquetWriter(out / f"{s}.parquet", SCHEMA)
                writers[s].write_table(t)
            for e in evid:
                evidence.write(json.dumps(e) + "\n")
            for k, (p, f) in st.items():
                agg[k][0] += p
                agg[k][1] += f
            if n % 20 == 0:
                print(f"shards {n}/{len(items)}", flush=True)
    for w in writers.values():
        w.close()
    evidence.close()
    stats.update({"label_conflicts": conflicts, "parse": dict(agg), "pre_cap_clusters": int(a.cluster_id.nunique()),
                  "dropped_by_cluster_cap": int(len(a) - len(keep))})
    (ROOT / cfg["paths"]["reports"] / "build_stats.json").write_text(json.dumps(stats, indent=1, default=str))
    print("done; dropped by cluster cap:", stats["dropped_by_cluster_cap"])


if __name__ == "__main__":
    main()
