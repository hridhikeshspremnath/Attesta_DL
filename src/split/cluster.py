"""MinHash signatures + LSH banding -> near-duplicate clusters (union-find). Pure numpy, no datasketch."""
from __future__ import annotations

import re
import zlib

import numpy as np

_P = (1 << 31) - 1
_TOK = re.compile(r"\w+|[^\w\s]")
MAX_TOKENS = 6000


def _perms(n: int, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return rng.integers(1, _P, n, dtype=np.uint64), rng.integers(0, _P, n, dtype=np.uint64)


def signature(text: str, n_perm: int = 64, k: int = 5) -> np.ndarray:
    toks = _TOK.findall(text)[:MAX_TOKENS]
    grams = [" ".join(toks[i:i + k]) for i in range(max(1, len(toks) - k + 1))]
    x = np.fromiter((zlib.crc32(g.encode("utf8", "replace")) % _P for g in grams), dtype=np.uint64, count=len(grams))
    a, b = _perms(n_perm)
    return ((a[:, None] * x[None, :] + b[:, None]) % _P).min(axis=1).astype(np.uint32)


class UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, i: int) -> int:
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def union(self, i: int, j: int) -> None:
        ri, rj = self.find(i), self.find(j)
        if ri != rj:
            self.p[max(ri, rj)] = min(ri, rj)


def cluster_ids(sigs: np.ndarray, keys: list[tuple], threshold: float = 0.8, rows: int = 8) -> list[int]:
    """sigs: (N, n_perm). keys: per-row grouping key (eco, package) -> same package always clusters together.
    Candidates come from LSH bands; a pair is merged only if est. Jaccard >= threshold (vs bucket representative)."""
    n, n_perm = sigs.shape
    uf = UnionFind(n)
    seen_pkg: dict[tuple, int] = {}
    for i, k in enumerate(keys):
        if k in seen_pkg:
            uf.union(seen_pkg[k], i)
        else:
            seen_pkg[k] = i
    for band in range(n_perm // rows):
        buckets: dict[bytes, int] = {}
        chunk = np.ascontiguousarray(sigs[:, band * rows:(band + 1) * rows])
        for i in range(n):
            h = chunk[i].tobytes()
            rep = buckets.get(h)
            if rep is None:
                buckets[h] = i
            elif (sigs[i] == sigs[rep]).mean() >= threshold:
                uf.union(rep, i)
    roots = [uf.find(i) for i in range(n)]
    remap = {r: c for c, r in enumerate(sorted(set(roots)))}
    return [remap[r] for r in roots]
