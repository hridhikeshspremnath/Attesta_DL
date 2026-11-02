"""Checkpoint + train-set scaler loading, and single-sample prediction."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.features.normalize import normalize_text
from src.features.pipeline import ALL_FEATURES
from src.models.malconv import MalConvFusion
from src.train.data import AUX, ECO, Scaler, to_bytes


@dataclass
class Bundle:
    model: MalConvFusion
    scaler: Scaler
    names: list[str]
    cfg: dict  # the training config (configs/m2.yaml)
    device: torch.device


def feature_names(drop: list[str]) -> list[str]:
    return [n for n in ALL_FEATURES if n not in drop]


def save_scaler(path: Path, scaler: Scaler, names: list[str]) -> None:
    np.savez(path, mean=scaler.mean, std=scaler.std, names=np.array(names))


def load_scaler(path: Path) -> tuple[Scaler, list[str]]:
    z = np.load(path)
    return Scaler(z["mean"], z["std"]), [str(n) for n in z["names"]]


def fit_scaler_from_train(processed: Path, names: list[str]) -> Scaler:
    """Same fit as training (log1p + standardise on the train split's features, float32)."""
    f = pd.DataFrame(pd.read_parquet(processed / "train.parquet", columns=["features"]).features.tolist())
    return Scaler.fit(f[names].to_numpy(np.float32))


def load_model(ckpt_dir: Path, cfg: dict, processed: Path, device: torch.device) -> Bundle:
    names = feature_names(cfg["data"]["drop_features"])
    sp = Path(ckpt_dir) / "scaler.npz"
    if sp.exists():
        scaler, saved = load_scaler(sp)
        if saved != names:
            raise ValueError(f"scaler feature order {saved} != expected {names}")
    else:  # saved next to the checkpoint on first use so later runs don't need the train parquet
        scaler = fit_scaler_from_train(processed, names)
        save_scaler(sp, scaler, names)
    m = cfg["model"]
    model = MalConvFusion(len(names), m["embed_dim"], m["filters"], m["kernel"], m["stride"], m["num_hidden"],
                          m["fusion_hidden"], m["dropout"], use_numeric=m.get("use_numeric", True))
    model.load_state_dict(torch.load(Path(ckpt_dir) / "best.pt", map_location="cpu", weights_only=True)["model"])  # tensors/numbers only: no pickled code
    return Bundle(model.to(device).eval(), scaler, names, cfg, device)


@torch.no_grad()
def predict(b: Bundle, text: str, feats: dict, eco: str) -> tuple[float, dict[str, float]]:
    """-> (malicious probability, {aux_name: probability}); same input path as the training Dataset."""
    d = b.cfg["data"]
    t = normalize_text(text) if d.get("normalize", False) else text
    byt = torch.from_numpy(to_bytes(t, d["max_bytes"]))[None]
    x = torch.from_numpy(b.scaler(np.array([[feats[n] for n in b.names]], dtype=np.float32)))
    amp = b.device.type == "cuda"  # training/eval used fp16 autocast on GPU
    with torch.autocast(b.device.type, dtype=torch.float16, enabled=amp):
        logit, aux = b.model(byt.to(b.device), x.to(b.device), torch.tensor([ECO[eco]], device=b.device))
    return float(torch.sigmoid(logit.float())[0]), dict(zip(AUX, torch.sigmoid(aux.float())[0].tolist()))
