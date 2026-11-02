"""Train the feature-only MLP baseline (CPU) and write one results table. Run: python -m src.train.run_mlp"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn

from src.models.mlp import FeatureMLP
from src.train.data import PackageDataset, make_loaders
from src.train.metrics import summarize
from src.utils.common import ROOT


@torch.no_grad()
def predict(model: FeatureMLP, ds: PackageDataset) -> np.ndarray:
    model.eval()
    return torch.sigmoid(model(ds.x, ds.eco)[0]).numpy()


def train(cfg: dict, loaders: dict, sets: dict, n_in: int) -> FeatureMLP:
    m = cfg["mlp"]
    torch.manual_seed(m["seed"])
    model = FeatureMLP(n_in, m["hidden"], m["dropout"])
    opt = torch.optim.AdamW(model.parameters(), lr=m["lr"], weight_decay=m["weight_decay"])
    bce = nn.BCEWithLogitsLoss()
    best, best_state = -1.0, None
    for _ in range(m["epochs"]):
        model.train()
        for b in loaders["train"]:
            logit, aux = model(b["x"], b["eco"])
            loss = bce(logit, b["y"]) + m["aux_weight"] * bce(aux, b["aux"])
            opt.zero_grad()
            loss.backward()
            opt.step()
        s = summarize(sets["val"].y.numpy(), predict(model, sets["val"]), cfg["eval"]["fpr"])["pr_auc"]
        if s > best:  # checkpoint on val PR-AUC
            best, best_state = s, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model


def main() -> None:
    cfg = yaml.safe_load((ROOT / "configs" / "m1.yaml").read_text())
    loaders, sets, names = make_loaders(cfg, cfg["mlp"]["batch_size"])
    model = train(cfg, loaders, sets, len(names))
    fpr, rows = cfg["eval"]["fpr"], []
    for split in ("val", "test"):
        ds, s = sets[split], predict(model, sets[split])
        y, eco, comp = ds.y.numpy(), ds.eco.numpy(), ds.comp.numpy()
        groups = {"all": np.ones(len(y), bool), "pypi": eco == 0, "npm": eco == 1, "compromised": comp}
        for g, mask in groups.items():
            rows.append({"split": split, "group": g, **summarize(y[mask], s[mask], fpr)})
    t = pd.DataFrame(rows).rename(columns={"rec@fpr": f"recall@{fpr:.1%}FPR"})
    out = ROOT / cfg["eval"]["report"]
    out.parent.mkdir(exist_ok=True)
    md = t.to_markdown(index=False, floatfmt=".3f")
    out.write_text(f"# M1 feature-only MLP\n\nfeatures={len(names)} dropped={cfg['data']['drop_features']}\n\n{md}\n")
    print(md)


if __name__ == "__main__":
    main()
