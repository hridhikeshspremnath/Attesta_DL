"""The single record -> (text, numeric features, rule labels, evidence) step, shared by dataset build and inference."""
from __future__ import annotations

import json
from pathlib import Path

from src.features.numeric import FEATURE_NAMES, numeric_features
from src.features.text import build_text
from src.features.typosquat import NameIndex, name_features
from src.labels.run import label_record

NAME_FEATS = ["name_len", "min_dist_top_n", "min_dist_known_mal"]
ALL_FEATURES = FEATURE_NAMES + NAME_FEATS


def load_name_indexes(refs_path: Path) -> tuple[dict[str, NameIndex], dict[str, NameIndex]]:
    refs = json.loads(Path(refs_path).read_text(encoding="utf8"))
    return {e: NameIndex(v["top"]) for e, v in refs.items()}, {e: NameIndex(v["mal"]) for e, v in refs.items()}


def featurize(rec: dict, top: dict[str, NameIndex], mal: dict[str, NameIndex], max_chars: int) -> dict:
    """-> {text, feats, aux, evidence, parse}; `feats` keys follow ALL_FEATURES."""
    aux, evidence, parse = label_record(rec)
    feats = numeric_features(rec)
    feats.update(name_features(rec["package"], top[rec["eco"]], mal[rec["eco"]]))
    return {"text": build_text(rec["files"], max_chars), "feats": feats, "aux": aux, "evidence": evidence, "parse": parse}
