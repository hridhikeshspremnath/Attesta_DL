"""Name edit-distance features (typosquat signal) against same-ecosystem reference lists."""
from __future__ import annotations

from rapidfuzz import process
from rapidfuzz.distance import Levenshtein


def _short(name: str) -> str:
    return name.split("/", 1)[1] if name.startswith("@") and "/" in name else name


class NameIndex:
    def __init__(self, refs: list[str]):
        self.refs = sorted(set(refs))
        self.short = [_short(r) for r in self.refs]

    def min_dist(self, name: str) -> float:
        """Smallest Levenshtein distance to a *different* reference name (exact self-match ignored)."""
        n = _short(name)
        if not self.refs:
            return -1.0
        best = 99.0
        for hit in process.extract(n, self.short, scorer=Levenshtein.distance, limit=3):
            if hit[0] != n:  # skip self
                best = min(best, float(hit[1]))
        return best


def name_features(name: str, top: NameIndex, mal: NameIndex) -> dict:
    return {"name_len": float(len(name)), "min_dist_top_n": top.min_dist(name),
            "min_dist_known_mal": mal.min_dist(name)}
