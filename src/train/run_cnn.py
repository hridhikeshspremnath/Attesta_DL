"""M2 trainer: byte-level multi-task CNN + numeric MLP.

Run: python -u -m src.train.run_cnn [--smoke] [--fresh] 2>&1 | Tee-Object logs\\train.log
Checkpoints every epoch (last.pt, best.pt); an existing last.pt is resumed unless --fresh.
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Subset

from src.models.malconv import MalConvFusion
from src.train.data import PackageDataset, make_loaders
from src.train.metrics import summarize
from src.train.report import md as rmd, save_scores, table as rtable
from src.utils.common import ROOT


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def subset_loaders(sets: dict, n: int, bs: int, seed: int, workers: int) -> dict[str, DataLoader]:
    """Smoke mode: n random train samples; val/test capped at n too."""
    rng = np.random.default_rng(seed)
    out = {}
    for s, ds in sets.items():
        idx = rng.choice(len(ds), min(n, len(ds)), replace=False).tolist()
        out[s] = DataLoader(Subset(ds, idx), batch_size=bs, shuffle=(s == "train"), num_workers=workers)
    return out


@torch.no_grad()
def predict(model: MalConvFusion, loader: DataLoader, dev: torch.device, amp: bool) -> dict[str, np.ndarray]:
    model.eval()
    keys = ("y", "eco", "is_compromised")
    acc: dict[str, list] = {"s": [], **{k: [] for k in keys}}
    for b in loader:
        with torch.autocast(dev.type, dtype=torch.float16, enabled=amp):
            logit, _ = model(b["bytes"].to(dev), b["x"].to(dev), b["eco"].to(dev))
        acc["s"].append(torch.sigmoid(logit.float()).cpu().numpy())
        for k in keys:
            acc[k].append(b[k].numpy())
        acc.setdefault("cluster_id", []).append(np.asarray(b["cluster_id"], dtype=object))
    return {k: np.concatenate(v) for k, v in acc.items()}


def pos_weights(ds: PackageDataset, t: dict, dev: torch.device) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    y = ds.y
    pw = t["pos_weight"]
    main = None if pw in (None, False) else torch.tensor(
        float((y == 0).sum() / (y == 1).sum().clamp(min=1)) if pw == "auto" else float(pw), device=dev)
    aux = None
    if t["aux_pos_weight"]:
        a = ds.aux
        aux = ((a == 0).sum(0) / a.sum(0).clamp(min=1)).clamp(max=10.0).to(dev)
    return main, aux


def save(path: Path, model, opt, scaler, epoch: int, best: float, bad: int) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                "epoch": epoch, "best": best, "bad": bad}, tmp)
    tmp.replace(path)  # atomic: an interrupted save never corrupts the previous checkpoint


def scores_frame(d: dict) -> pd.DataFrame:
    return pd.DataFrame({"y": d["y"], "score": d["s"], "eco": d["eco"], "is_compromised": d["is_compromised"],
                         "cluster_id": d["cluster_id"]})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="500 samples, 1 epoch, separate checkpoint dir")
    ap.add_argument("--fresh", action="store_true", help="ignore an existing last.pt")
    ap.add_argument("--score-only", action="store_true", help="skip training; score val/test with best.pt and save scores")
    ap.add_argument("--config", default="configs/m2.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / args.config).read_text())
    t, fpr = cfg["train"], cfg["eval"]["fpr"]
    if args.smoke:
        t["epochs"], t["ckpt_dir"] = cfg["smoke"]["epochs"], cfg["smoke"]["ckpt_dir"]
    seed_all(t["seed"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(t["fp16"]) and dev.type == "cuda"
    print(f"device={dev} fp16={amp} smoke={args.smoke}", flush=True)

    loaders, sets, names = make_loaders(cfg, t["batch_size"])
    workers = cfg["data"]["num_workers"]
    if args.smoke:
        loaders = subset_loaders(sets, cfg["smoke"]["samples"], t["batch_size"], t["seed"], workers)
    else:
        g = torch.Generator().manual_seed(t["seed"])
        loaders = {s: DataLoader(sets[s], batch_size=t["batch_size"], shuffle=(s == "train"), num_workers=workers,
                                 generator=g if s == "train" else None) for s in sets}
    m = cfg["model"]
    model = MalConvFusion(len(names), m["embed_dim"], m["filters"], m["kernel"], m["stride"], m["num_hidden"],
                          m["fusion_hidden"], m["dropout"], use_numeric=m.get("use_numeric", True)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    scaler = torch.amp.GradScaler(dev.type, enabled=amp)
    pw_main, pw_aux = pos_weights(sets["train"], t, dev)
    bce_main, bce_aux = nn.BCEWithLogitsLoss(pos_weight=pw_main), nn.BCEWithLogitsLoss(pos_weight=pw_aux, reduction="none")

    ck = ROOT / t["ckpt_dir"]
    ck.mkdir(parents=True, exist_ok=True)
    start, best, bad = 0, -1.0, 0
    if (ck / "last.pt").exists() and not args.fresh:
        s = torch.load(ck / "last.pt", map_location=dev)
        model.load_state_dict(s["model"])
        opt.load_state_dict(s["opt"])
        scaler.load_state_dict(s["scaler"])
        start, best, bad = s["epoch"] + 1, s["best"], s["bad"]
        print(f"resumed from epoch {start} (best val PR-AUC {best:.4f})", flush=True)
        if bad >= t["patience"]:  # that run had already early-stopped: go straight to the final eval
            start = t["epochs"]
    if args.score_only:  # no training: load best.pt, write scores + report
        assert (ck / "best.pt").exists(), f"no best.pt in {ck}"
        start = t["epochs"]

    for ep in range(start, t["epochs"]):
        t0 = time.time()
        model.train()
        opt.zero_grad(set_to_none=True)
        run, nb = 0.0, 0
        for i, b in enumerate(loaders["train"]):
            with torch.autocast(dev.type, dtype=torch.float16, enabled=amp):
                logit, aux = model(b["bytes"].to(dev), b["x"].to(dev), b["eco"].to(dev))
            y, ya = b["y"].to(dev), b["aux"].to(dev)
            loss = bce_main(logit.float(), y) + t["aux_weight"] * bce_aux(aux.float(), ya).mean(0).sum()
            scaler.scale(loss / t["accum_steps"]).backward()
            if (i + 1) % t["accum_steps"] == 0 or i + 1 == len(loaders["train"]):
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            run, nb = run + loss.item(), nb + 1
        d = predict(model, loaders["val"], dev, amp)
        torch.cuda.empty_cache()  # release eval-time cache so the next epoch doesn't fragment a 4 GB card
        v = summarize(d["y"], d["s"], fpr)
        print(f"epoch {ep + 1}/{t['epochs']} loss={run / max(nb, 1):.4f} val_pr_auc={v['pr_auc']:.4f} "
              f"val_roc_auc={v['roc_auc']:.4f} val_rec@fpr={v['rec@fpr']:.3f} time={time.time() - t0:.0f}s", flush=True)
        if v["pr_auc"] > best:
            best, bad = v["pr_auc"], 0
            save(ck / "best.pt", model, opt, scaler, ep, best, bad)
        else:
            bad += 1
        save(ck / "last.pt", model, opt, scaler, ep, best, bad)
        if bad >= t["patience"]:
            print(f"early stop at epoch {ep + 1} (no val PR-AUC gain for {t['patience']} epochs)", flush=True)
            break

    model.load_state_dict(torch.load(ck / "best.pt", map_location=dev)["model"])
    preds = {split: predict(model, loaders[split], dev, amp) for split in ("val", "test")}
    if not args.smoke:  # per-sample scores for the selection / final-report tools (no CI here: those are in final_report)
        save_scores(ROOT / cfg["eval"].get("scores", "reports/scores") / f"{ck.name}.parquet", preds)
    md = "\n\n".join(f"**{sp}**\n\n" + rmd(rtable(scores_frame(d))) for sp, d in preds.items())
    if not args.smoke:
        out = ROOT / cfg["eval"]["report"]
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(f"# M2 byte CNN + MLP\n\nbest val PR-AUC {best:.4f}\n\n{md}\n")
    print(md, flush=True)


if __name__ == "__main__":
    main()
