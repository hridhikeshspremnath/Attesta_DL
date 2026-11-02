"""configs/policy.yaml: score bands -> decision."""
from __future__ import annotations

from pathlib import Path

import yaml


def load_policy(path: Path) -> dict:
    p = yaml.safe_load(Path(path).read_text(encoding="utf8"))
    maxes = [b["max_permille"] for b in p["bands"]]
    if maxes != sorted(maxes) or maxes[-1] < 1000:
        raise ValueError("policy bands must ascend and the last must reach 1000")
    return p


def band_for(policy: dict, permille: int) -> tuple[str, str]:
    """(band name, decision) for an integer permille risk score."""
    for b in policy["bands"]:
        if permille <= b["max_permille"]:
            return b["name"], b["decision"]
    raise ValueError(f"score {permille} outside 0..1000")
