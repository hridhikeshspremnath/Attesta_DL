"""Evidence snippets with best-effort line numbers, and the one-line reason."""
from __future__ import annotations

import re
from typing import Optional


def locate(text: str, detail: str) -> Optional[int]:
    """1-based line of an extractor snippet inside the file text, or None. The extractors store a snippet but no line,
    so this searches for it (whitespace-insensitive), then for its longest token."""
    toks = detail.split()
    if not toks:
        return None
    m = re.search(r"\s+".join(re.escape(t) for t in toks), text)
    if m:
        return text.count("\n", 0, m.start()) + 1
    for tok in sorted({t.strip(":,;()'\"[]{}") for t in toks}, key=len, reverse=True):  # longest token first
        if len(tok) >= 4 and (pos := text.find(tok)) >= 0:
            return text.count("\n", 0, pos) + 1
    return None


def evidence_rows(rec: dict, evidence: list[dict], limit: int = 8, per_label: int = 2) -> list[dict]:
    texts = {f["path"]: f["text"] for f in rec["files"]}
    rows, seen = [], {}
    for e in evidence:
        lab = e["label"].replace("aux_", "")
        if seen.get(lab, 0) >= per_label or len(rows) >= limit:
            continue
        seen[lab] = seen.get(lab, 0) + 1
        rows.append({"aux": lab, "file": e["file"], "line": locate(texts.get(e["file"], ""), e["detail"]),
                     "snippet": e["detail"], "via": e["kind"]})
    return rows


def reason(permille: int, band: str, decision: str, model_flags: list[str], rule_flags: list[str]) -> str:
    both = [f for f in model_flags if f in rule_flags]
    parts = [f"{band} risk ({permille}/1000) -> {decision}"]
    if both:
        parts.append("model and rules agree on " + ", ".join(both))
    only_m, only_r = [f for f in model_flags if f not in rule_flags], [f for f in rule_flags if f not in model_flags]
    if only_r:
        parts.append("rules only: " + ", ".join(only_r))
    if only_m:
        parts.append("model only: " + ", ".join(only_m))
    if not (model_flags or rule_flags):
        parts.append("no behaviour flags raised; score comes from overall content/feature patterns")
    return "; ".join(parts)
