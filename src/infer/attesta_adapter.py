"""Plug this detector into the Attesta scoring service (ATTESTA-SPIFFE, `attesta.scoring`).

    model_factory: src.infer.attesta_adapter:build_model
    extractor_factory: src.infer.attesta_adapter:build_extractors
    model_path: <checkpoint dir>   # default checkpoints/m2_noshort; trained with configs/<dir name>.yaml

The service hands over the downloaded archive (`bundle.data`), so a package is scored through the same
Archive -> build_record -> featurize path as training and `src.infer.score`. Nothing is installed or run.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
import yaml
from attesta.scoring.errors import ArchiveError
from attesta.scoring.model import FLAG_NAMES, ExtractorResult, ModelOutput, hash_files, to_permille

from src.features.pipeline import featurize, load_name_indexes
from src.infer.fetch import record_from_bytes
from src.infer.model_io import load_model, predict
from src.labels.run import label_record
from src.utils.common import ROOT, load_config

AUX_OF = {flag: "aux_" + flag.removeprefix("flag_") for flag in FLAG_NAMES}


class Detector:
    """Both the ScoringModel and the LabelExtractor of attesta.scoring. No per-call state, so thread-safe."""

    version = "attesta-dl-rules-1"

    def __init__(self, ckpt: Path) -> None:
        train_cfg = ROOT / "configs" / f"{ckpt.name}.yaml"
        m2 = yaml.safe_load(train_cfg.read_text(encoding="utf8"))
        self.cfg = load_config()
        self.bundle = load_model(ckpt, m2, ROOT / m2["data"]["processed"], torch.device("cpu"))
        self.top, self.mal = load_name_indexes(ROOT / self.cfg["paths"]["cache"] / "name_refs.json")
        self.version_hash = hash_files([ckpt / "best.pt", ckpt / "scaler.npz", train_cfg])

    def _record(self, b: Any) -> dict:
        if not b.data:
            raise ArchiveError("bundle carries no archive data (needs attesta.scoring with SourceBundle.data)")
        try:
            return record_from_bytes(b.ecosystem, b.package, b.version, b.data, self.cfg)
        except Exception as exc:  # noqa: BLE001 - any unreadable archive is a scoring error, not a crash
            raise ArchiveError(f"detector cannot read {b.filename}: {exc}") from exc

    def predict(self, b: Any) -> ModelOutput:
        rec = self._record(b)
        out = featurize(rec, self.top, self.mal, self.cfg["features"]["max_text_chars"])
        p, aux = predict(self.bundle, out["text"], out["feats"], rec["eco"])
        return ModelOutput(to_permille(p), {flag: to_permille(aux[a]) for flag, a in AUX_OF.items()})

    def extract(self, b: Any) -> ExtractorResult:
        aux, evidence, _ = label_record(self._record(b))
        hits = {flag: tuple(sorted({f"{e['kind']}:{e['file']}" for e in evidence if e["label"] == a}))
                for flag, a in AUX_OF.items()}
        return ExtractorResult({flag: bool(aux[a]) for flag, a in AUX_OF.items()}, hits)


@lru_cache(maxsize=1)
def _detector(ckpt: Path) -> Detector:
    return Detector(ckpt)


def build_model(settings: Any) -> Detector:
    return _detector(Path(settings.model_path or ROOT / "checkpoints" / "m2_noshort"))


build_extractors = build_model  # one object serves both roles, loaded once
