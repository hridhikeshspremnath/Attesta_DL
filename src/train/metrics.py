"""PR-AUC, ROC-AUC, recall at fixed FPR, expected precision at a prevalence, cluster-bootstrap CIs."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def _threshold(neg: np.ndarray, fpr: float) -> float:
    """Lowest threshold whose benign FPR is <= fpr (strict '>' rule). With < 1/fpr benign samples it is the max benign score."""
    neg = np.sort(neg)[::-1]
    k = int(np.floor(fpr * len(neg)))  # benign samples allowed above the threshold
    return neg[k] if k < len(neg) else -np.inf


def op_point(y: np.ndarray, s: np.ndarray, fpr: float) -> tuple[float, float]:
    """(TPR, realised FPR) at the operating threshold for target `fpr`."""
    neg, pos = s[y == 0], s[y == 1]
    if len(neg) == 0 or len(pos) == 0:
        return float("nan"), float("nan")
    thr = _threshold(neg, fpr)
    return float((pos > thr).mean()), float((neg > thr).mean())


def recall_at_fpr(y: np.ndarray, s: np.ndarray, fpr: float) -> float:
    return op_point(y, s, fpr)[0]


def precision_at_prevalence(tpr: float, fpr: float, prev: float) -> float:
    """Expected precision if malicious prevalence were `prev`: TPR*p / (TPR*p + FPR*(1-p))."""
    if not fpr > 0:  # zero realised FPR (tiny benign group) makes precision a meaningless 1.0
        return float("nan")
    return float(tpr * prev / (tpr * prev + fpr * (1 - prev)))


def summarize(y: np.ndarray, s: np.ndarray, fpr: float) -> dict:
    """Light summary used for per-epoch logging / early stopping."""
    ok = len(np.unique(y)) == 2
    return {"n": len(y), "n_mal": int(y.sum()), "n_ben": int((y == 0).sum()),
            "pr_auc": average_precision_score(y, s) if ok else float("nan"),
            "roc_auc": roc_auc_score(y, s) if ok else float("nan"),
            "rec@fpr": recall_at_fpr(y, s, fpr)}


def full_summary(y: np.ndarray, s: np.ndarray) -> dict:
    ok = len(np.unique(y)) == 2
    tpr1, fpr1 = op_point(y, s, 0.01)
    return {"n": len(y), "n_mal": int(y.sum()), "n_ben": int((y == 0).sum()),
            "pr_auc": average_precision_score(y, s) if ok else float("nan"),
            "roc_auc": roc_auc_score(y, s) if ok else float("nan"),
            "rec@0.1%FPR": recall_at_fpr(y, s, 0.001), "rec@1%FPR": tpr1,
            "prec@1%prev": precision_at_prevalence(tpr1, fpr1, 0.01),  # all at the 1%-FPR operating point
            "prec@0.1%prev": precision_at_prevalence(tpr1, fpr1, 0.001)}


BOOT = {"pr_auc": lambda y, s: average_precision_score(y, s), "roc_auc": lambda y, s: roc_auc_score(y, s),
        "rec@1%FPR": lambda y, s: recall_at_fpr(y, s, 0.01)}


def cluster_bootstrap(y: np.ndarray, s: np.ndarray, clusters: np.ndarray, reps: int = 1000, seed: int = 13,
                      alpha: float = 0.05) -> dict[str, tuple[float, float]]:
    """95% percentile CIs, resampling whole clusters (cluster_id) with replacement. Single-class resamples are skipped."""
    codes = np.unique(clusters, return_inverse=True)[1]
    order = np.argsort(codes, kind="stable")
    bounds = np.cumsum(np.bincount(codes))[:-1]
    groups = np.split(order, bounds)
    rng = np.random.default_rng(seed)
    vals: dict[str, list] = {k: [] for k in BOOT}
    for _ in range(reps):
        idx = np.concatenate([groups[c] for c in rng.integers(0, len(groups), len(groups))])
        yb, sb = y[idx], s[idx]
        if len(np.unique(yb)) < 2:
            continue
        for k, f in BOOT.items():
            vals[k].append(f(yb, sb))
    return {k: (float(np.nanpercentile(v, 100 * alpha / 2)), float(np.nanpercentile(v, 100 * (1 - alpha / 2))))
            if v else (float("nan"), float("nan")) for k, v in vals.items()}
