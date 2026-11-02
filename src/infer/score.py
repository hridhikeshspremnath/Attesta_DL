"""Model-only inference CLI. Never installs or executes a package: archives are read in memory as text.

  python -m src.infer.score npm lodash 4.17.21
  python -m src.infer.score --sample malicious --eco pypi        # random TEST-split row (no download)
  python -m src.infer.score --verify 20                           # re-score 20 test rows vs stored predictions
Flags: --json (machine output), --cpu (force CPU), --seed N, --ckpt DIR, --policy FILE.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.features.pipeline import featurize, load_name_indexes
from src.infer.explain import evidence_rows, reason
from src.infer.fetch import download, record_from_bytes
from src.infer.model_io import Bundle, load_model, predict
from src.infer.policy import band_for, load_policy
from src.train.data import AUX
from src.utils.common import ROOT, load_config, read_jsonl_gz


@dataclass
class Context:
    cfg: dict
    policy: dict
    bundle: Bundle
    top: dict
    mal: dict


def build_context(ckpt: Path | None, policy_path: Path, cpu: bool) -> Context:
    policy = load_policy(policy_path)
    cfg = load_config()
    ckpt = ckpt or ROOT / policy["model"]["checkpoint_dir"]
    m2 = yaml.safe_load((ROOT / policy["model"]["config"]).read_text(encoding="utf8"))
    dev = torch.device("cpu" if cpu or not torch.cuda.is_available() else "cuda")
    bundle = load_model(ckpt, m2, ROOT / m2["data"]["processed"], dev)
    top, mal = load_name_indexes(ROOT / cfg["paths"]["cache"] / "name_refs.json")
    return Context(cfg, policy, bundle, top, mal)


def score_record(ctx: Context, rec: dict, source: str, truth: str | None = None) -> dict:
    """Pipeline (shared with the dataset build) -> model -> policy -> evidence/reason."""
    out = featurize(rec, ctx.top, ctx.mal, ctx.cfg["features"]["max_text_chars"])
    p, aux_p = predict(ctx.bundle, out["text"], out["feats"], rec["eco"])
    permille = int(round(p * 1000))
    band, decision = band_for(ctx.policy, permille)
    thr = ctx.policy["aux_threshold"]
    aux = {k.replace("aux_", ""): {"model_prob": round(aux_p[k], 3), "model_flag": aux_p[k] >= thr,
                                    "rules_flag": bool(out["aux"][k])} for k in AUX}
    mf = [k for k, v in aux.items() if v["model_flag"]]
    rf = [k for k, v in aux.items() if v["rules_flag"]]
    res = {"ecosystem": rec["eco"], "package": rec["package"], "version": rec["version"], "source": source,
           "risk_score": permille, "band": band, "decision": decision, "aux": aux,
           "evidence": evidence_rows(rec, out["evidence"], ctx.policy.get("max_evidence", 8)),
           "reason": reason(permille, band, decision, mf, rf), "n_files": len(rec["files"])}
    if truth:
        res["truth"] = truth
    return res


def render(r: dict) -> str:
    head = f"{r['ecosystem']}/{r['package']}@{r['version']}  [{r['source']}" + (f", truth={r['truth']}" if "truth" in r else "") + "]"
    lines = [head, f"risk_score={r['risk_score']} (permille)  band={r['band']}  decision={r['decision']}",
             f"reason: {r['reason']}", "aux            model_p  model  rules"]
    for k, v in r["aux"].items():
        lines.append(f"  {k:<13}{v['model_prob']:>7.3f}  {'Y' if v['model_flag'] else '-':<5}  {'Y' if v['rules_flag'] else '-'}")
    lines.append("evidence:" if r["evidence"] else "evidence: none")
    for e in r["evidence"]:
        lines.append(f"  [{e['aux']}] {e['file']}:{e['line'] if e['line'] else '?'}  {e['snippet'][:100]}")
    return "\n".join(lines)


# ---- test-split sampling and verification ---------------------------------------------------------------------

def _index() -> pd.DataFrame:
    return pd.read_pickle(ROOT / "data" / "processed" / "split_index.pkl")


def find_record(index: pd.DataFrame, eco: str, package: str, version: str) -> dict:
    r = index[(index.eco == eco) & (index.package == package) & (index.version == version)].iloc[0]
    for i, rec in enumerate(read_jsonl_gz(Path(r.shard))):
        if i == r.line:
            return rec
    raise LookupError(f"line {r.line} not found in {r.shard}")


def pick_sample(label: int, eco: str, seed: int | None) -> tuple[str, str, str, str]:
    t = pd.read_parquet(ROOT / "data" / "processed" / "test.parquet",
                        columns=["package", "version", "ecosystem", "label", "source_dataset"])
    t = t[(t.label == label) & (t.ecosystem == eco)]
    r = t.iloc[random.Random(seed).randrange(len(t))]
    return r.ecosystem, r.package, r.version, f"{'malicious' if label else 'benign'}/{r.source_dataset}"


def verify(ctx: Context, n: int, seed: int) -> dict:
    """Re-score n test rows through the inference path (shard record -> featurize -> model) and compare with the
    stored test predictions (reports/scores/m2.parquet), the stored text and the stored numeric features."""
    from src.train.report import load_scores
    sp = ROOT / "reports" / "scores" / f"{Path(ctx.policy['model']['checkpoint_dir']).name}.parquet"
    if not sp.exists():
        raise SystemExit(f"{sp} missing: run `python -m src.train.run_cnn --config configs/m2.yaml --score-only` first")
    stored = load_scores(sp, "test").score.to_numpy()
    t = pd.read_parquet(ROOT / "data" / "processed" / "test.parquet", columns=["package", "version", "ecosystem", "text", "features"])
    idx = _index()
    pick = np.random.default_rng(seed).choice(len(t), n, replace=False)
    d_score, d_feat, same_text = [], [], 0
    for i in pick:
        row = t.iloc[i]
        rec = find_record(idx, row.ecosystem, row.package, row.version)
        out = featurize(rec, ctx.top, ctx.mal, ctx.cfg["features"]["max_text_chars"])
        p, _ = predict(ctx.bundle, out["text"], out["feats"], rec["eco"])
        d_score.append(abs(p - stored[i]))
        same_text += out["text"] == row.text
        d_feat.append(max(abs(out["feats"][k] - row.features[k]) / max(1.0, abs(row.features[k])) for k in ctx.bundle.names))
    return {"n": n, "device": str(ctx.bundle.device), "max_abs_score_diff": float(max(d_score)),
            "mean_abs_score_diff": float(np.mean(d_score)), "text_identical": f"{same_text}/{n}",
            "max_rel_feature_diff": float(max(d_feat))}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", nargs="*", help="<npm|pypi> <name> <version|latest>")
    ap.add_argument("--sample", choices=["malicious", "benign"])
    ap.add_argument("--eco", choices=["npm", "pypi"])
    ap.add_argument("--verify", type=int, metavar="N")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--policy", type=Path, default=ROOT / "configs" / "policy.yaml")
    a = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if not (a.verify or a.sample or len(a.target) == 3):
        ap.error("give <eco> <name> <version>, or --sample, or --verify N")
    if a.sample and not a.eco:
        ap.error("--sample needs --eco")
    ctx = build_context(a.ckpt, a.policy, a.cpu)
    if a.verify:
        res = verify(ctx, a.verify, a.seed or 13)
        print(json.dumps(res) if a.json else pd.DataFrame([res]).T.rename(columns={0: "value"}).to_markdown())
        return 0
    if a.sample:
        eco, pkg, ver, truth = pick_sample(int(a.sample == "malicious"), a.eco, a.seed)
        res = score_record(ctx, find_record(_index(), eco, pkg, ver), "test-sample", truth)
    else:
        eco, name, version = a.target
        if eco not in ("npm", "pypi"):
            ap.error("eco must be npm or pypi")
        data, ver, date = download(eco, name, version, ctx.cfg)
        res = score_record(ctx, record_from_bytes(eco, name, ver, data, ctx.cfg, date), "registry")
    print(json.dumps(res) if a.json else render(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
