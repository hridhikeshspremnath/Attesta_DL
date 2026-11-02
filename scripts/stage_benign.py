"""Stage 3: plan + fetch benign sources (download only). Resumable: finished shards are skipped.

Usage: python -m scripts.stage_benign [--plan-only] [--replan]
  --replan  re-sample pypi+npm picks (keeps malware_bench neutrals and hard negatives from the existing plan)
"""
from __future__ import annotations

import json
import random
import sys

import pandas as pd

from src.ingest import benign_candidates as bc
from src.ingest.benign_fetch import run
from src.ingest.hard_negatives import hard_negative_candidates
from src.ingest.ossf import is_excluded, load_unmergable
from src.ingest.registry import Client
from src.utils.common import ROOT, data_dir, load_config


def load_ossf(raw, interim) -> dict:
    """OSSF index + unmergable reports merged in (versions listed -> those versions; none -> whole name)."""
    ossf = json.loads((interim / "ossf_index.json").read_text(encoding="utf8"))
    for eco, u in load_unmergable(raw).items():
        ossf[eco]["whole"] = sorted(set(ossf[eco]["whole"]) | u["whole"])
        for name, vs in u["versions"].items():
            ossf[eco]["versions"][name] = sorted(set(ossf[eco]["versions"].get(name, [])) | vs)
    return ossf


def npm_pool(cli, cfg, cache) -> list[dict]:
    """Search pool (popular-leaning) + random draw from the full registry name list, with downloads attached."""
    enr = cache / "npm_pool_enriched.json"
    if enr.exists():
        return json.loads(enr.read_text(encoding="utf8"))
    seed = cfg["split"]["seed"]
    search = json.loads((cache / "npm_pool.json").read_text(encoding="utf8"))
    have = {c["package"] for c in search}
    extra = [c for c in bc.npm_random(cache / "npm_all_names.json", 60_000, 1200, seed) if c["package"] not in have]
    out = bc.npm_enrich(cli, search + extra, 1800, seed, cache / "npm_downloads.json")
    enr.write_text(json.dumps(out), encoding="utf8")
    return out


def plan(cfg, cli, interim, cache, old) -> list[dict]:
    b, seed = cfg["benign"], cfg["split"]["seed"]
    ossf = load_ossf(ROOT / cfg["paths"]["raw"], interim)
    mal = pd.read_pickle(interim / "index_malicious_dedup.pkl")
    mal_names = {e: set(g.package) for e, g in pd.read_pickle(interim / "index_malicious_raw.pkl").groupby("eco")}
    excl = lambda c: c["package"] in mal_names.get(c["eco"], set()) or is_excluded(ossf, c["eco"], c["package"])  # noqa: E731
    n_mal = mal.groupby("eco").size().to_dict()
    pools = {"pypi": bc.pypi_candidates(cli, cfg), "npm": npm_pool(cli, cfg, cache)}
    cands: list[dict] = []
    for eco, n in n_mal.items():
        total = int(n * (1 + b["extra_eval_factor"]))
        picked = bc.sample_strata(pools[eco], total, b["strata"], seed, excl, b["low_reserve"])
        random.Random(seed).shuffle(picked)
        cands += picked  # main/extra assignment happens at build time, on packages that survive fetch
    if old:
        cands += [c for c in old if c.get("hard_negative") or c.get("src") == "malware_bench"]
    else:
        cands += [c for c in bc.malware_bench_neutral(ROOT / cfg["paths"]["raw"]) if not excl(c)]
        cands += hard_negative_candidates(cli, mal, ossf, b["hard_negatives_per_pkg"])
    random.Random(seed).shuffle(cands)  # any prefix of the download is stratified by eco and stratum (in expectation)
    return cands


def main() -> None:
    cfg = load_config()
    b, interim, cache = cfg["benign"], data_dir(cfg, "interim"), data_dir(cfg, "cache")
    cli = Client(b["user_agent"], b["rate_per_sec"])
    plan_path = cache / "benign_plan.json"
    old = json.loads(plan_path.read_text(encoding="utf8")) if plan_path.exists() else None
    if old and "--replan" not in sys.argv:
        cands = old
    else:
        cands = plan(cfg, cli, interim, cache, old)
        plan_path.write_text(json.dumps(cands), encoding="utf8")
    t = pd.DataFrame(cands)
    t["kind"] = ["hard_neg" if c.get("hard_negative") else "bench" if c.get("src") == "malware_bench" else "sampled"
                 for c in cands]
    print(t.groupby(["eco", "kind", "stratum"]).size().unstack(fill_value=0).to_string())
    print("total candidates:", len(cands))
    if "--plan-only" in sys.argv:
        return
    ossf = load_ossf(ROOT / cfg["paths"]["raw"], interim)
    run(cands, interim / "benign", cfg, lambda eco, name, ver: is_excluded(ossf, eco, name, ver))


if __name__ == "__main__":
    main()
