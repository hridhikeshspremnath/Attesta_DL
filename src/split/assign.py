"""Cluster-level split: no cluster in >1 split, stratified by ecosystem+label."""
from __future__ import annotations

import numpy as np
import pandas as pd


def assign_splits(df: pd.DataFrame, fractions: dict, seed: int = 13) -> pd.Series:
    """df needs: cluster_id, eco, label, pool ('main'|'extra'). Returns split name per row.

    Main clusters fill train/val/test toward `fractions` per (eco, label) stratum (by sample count).
    Extra (benign, realistic-ratio eval) clusters go to val/test only, alternating by size.
    """
    rng = np.random.default_rng(seed)
    splits = list(fractions)
    cl = df.groupby("cluster_id").agg(n=("label", "size"), eco=("eco", "first"), label=("label", lambda s: s.mode().iat[0]),
                                      extra=("pool", lambda s: (s == "extra").all())).reset_index()
    cl = cl.sample(frac=1, random_state=int(rng.integers(1 << 30))).sort_values("n", ascending=False, kind="stable")
    assign: dict[int, str] = {}
    for (eco, label), g in cl[~cl.extra].groupby(["eco", "label"]):
        total, got = g.n.sum(), dict.fromkeys(splits, 0)
        for cid, n in zip(g.cluster_id, g.n):
            # place into the split furthest below its target share
            s = max(splits, key=lambda k: fractions[k] * total - got[k])
            assign[cid] = s
            got[s] += n
    ev, held = ["val", "test"], {"val": 0, "test": 0}
    for cid, n in zip(cl[cl.extra].cluster_id, cl[cl.extra].n):
        s = min(ev, key=lambda k: held[k])
        assign[cid] = s
        held[s] += n
    return df.cluster_id.map(assign)


def place_hard_negatives(df: pd.DataFrame, split: pd.Series, per_split: int = 6, seed: int = 13) -> pd.Series:
    """Move compromised-package groups (malicious affected versions + clean neighbour versions) from train into
    val/test so the compromised group has benign rows to evaluate against. A group is a connected component of
    rows linked by shared package or shared cluster, so neither packages nor clusters are ever split."""
    hn = df[(df.label == 0) & df.is_compromised.astype(bool) & (split == "train")]
    key = list(zip(df.eco, df.package))
    by_pkg: dict = {}
    by_cl: dict = {}
    for i, (k, c) in enumerate(zip(key, df.cluster_id)):
        by_pkg.setdefault(k, []).append(i)
        by_cl.setdefault(c, []).append(i)
    seen: set[int] = set()
    comps = []
    for i in hn.index:
        if i in seen:
            continue
        stack, comp = [i], set()
        while stack:
            j = stack.pop()
            if j in comp:
                continue
            comp.add(j)
            stack += by_pkg[key[j]] + by_cl[df.cluster_id.iat[j]]
        seen |= comp
        comps.append(sorted(comp))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(comps))
    out = split.copy()
    for n, ci in enumerate(order[: 2 * per_split]):
        out.iloc[comps[ci]] = "val" if n % 2 == 0 else "test"
    return out
