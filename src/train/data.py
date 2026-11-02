"""PyTorch dataset over data/processed/{split}.parquet: bytes, numeric features, label, aux labels, metadata."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.features.normalize import normalize_text

AUX = ["aux_install_exec", "aux_network", "aux_cred_access", "aux_obfuscation", "aux_process_spawn"]
ECO = {"pypi": 0, "npm": 1}


class Scaler:
    """log1p + standardisation fit on train only (feature columns are non-negative counts/lengths)."""

    def __init__(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.mean, self.std = mean, std

    @classmethod
    def fit(cls, x: np.ndarray) -> "Scaler":
        z = np.log1p(np.clip(x, 0, None))
        return cls(z.mean(0), z.std(0) + 1e-6)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return ((np.log1p(np.clip(x, 0, None)) - self.mean) / self.std).astype(np.float32)


def load_split(root: Path, split: str, drop: list[str]) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Returns (meta+text frame, raw feature matrix, feature names)."""
    df = pd.read_parquet(root / f"{split}.parquet")
    feats = pd.DataFrame(df.pop("features").tolist())
    feats = feats.drop(columns=[c for c in drop if c in feats])
    return df.reset_index(drop=True), feats.to_numpy(np.float32), list(feats.columns)


def to_bytes(text: str, n: int) -> np.ndarray:
    """utf-8 bytes truncated/padded to n; ids are byte+1, 0 = pad."""
    b = np.frombuffer(text.encode("utf8", "ignore")[:n], dtype=np.uint8).astype(np.int64) + 1
    out = np.zeros(n, dtype=np.int64)
    out[: len(b)] = b
    return out


class PackageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feats: np.ndarray, max_bytes: int, scaler: Scaler,
                 normalize: bool = False) -> None:
        self.max_bytes, self.normalize = max_bytes, normalize
        self.x = torch.from_numpy(scaler(feats))
        self.y = torch.tensor(df.label.to_numpy(), dtype=torch.float32)
        self.aux = torch.tensor(df[AUX].to_numpy(), dtype=torch.float32)
        self.eco = torch.tensor(df.ecosystem.map(ECO).to_numpy(), dtype=torch.long)
        self.comp = torch.tensor(df.is_compromised.to_numpy(), dtype=torch.bool)
        self.cluster = df.cluster_id.tolist()
        self.text = df.text.tolist()  # bytes are built lazily per item; a 64 KB x N matrix is never held

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int) -> dict:
        text = normalize_text(self.text[i]) if self.normalize else self.text[i]
        return {"bytes": torch.from_numpy(to_bytes(text, self.max_bytes)),
                "x": self.x[i], "y": self.y[i], "aux": self.aux[i], "eco": self.eco[i],
                "is_compromised": self.comp[i], "cluster_id": self.cluster[i]}


def make_loaders(cfg: dict, batch_size: int) -> tuple[dict[str, DataLoader], dict[str, PackageDataset], list[str]]:
    root, mb = Path(cfg["data"]["processed"]), cfg["data"]["max_bytes"]
    raw = {s: load_split(root, s, cfg["data"]["drop_features"]) for s in ("train", "val", "test")}
    scaler = Scaler.fit(raw["train"][1])
    norm = bool(cfg["data"].get("normalize", False))
    sets = {s: PackageDataset(df, f, mb, scaler, norm) for s, (df, f, _) in raw.items()}
    loaders = {s: DataLoader(d, batch_size=batch_size, shuffle=(s == "train")) for s, d in sets.items()}
    return loaders, sets, raw["train"][2]
