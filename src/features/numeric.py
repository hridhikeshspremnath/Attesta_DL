"""Numeric feature vector over the selected text of one record."""
from __future__ import annotations

import re

from src.utils.common import shannon_entropy

B64 = re.compile(r"[A-Za-z0-9+/]{40,}={0,2}")
HEX = re.compile(r"(?:[0-9a-fA-F]{2}){20,}")
URL = re.compile(r"https?://[^\s'\"<>)]+", re.I)
IP = re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b")
DANGEROUS = {
    "eval": r"\beval\s*\(", "exec": r"\bexec\s*\(", "new_function": r"\bnew\s+Function\b|\bFunction\s*\(",
    "os_system": r"\bos\.(system|popen)\s*\(", "subprocess": r"\bsubprocess\.\w+", "child_process": r"child_process",
    "b64_decode": r"\b(b64decode|atob|decodebytes)\b|base64", "from_char_code": r"fromCharCode",
    "env_access": r"process\.env|os\.environ|getenv", "net_call": r"\b(urlopen|requests\.\w+|fetch|XMLHttpRequest)\b",
    "socket": r"\bsocket\b", "ctypes": r"\bctypes\b", "pickle_marshal": r"\b(pickle|marshal)\.loads?\b",
    "compile_import": r"\b(compile|__import__)\s*\(", "chmod": r"\bchmod\b", "curl_wget": r"\b(curl|wget)\b",
    "powershell": r"powershell|cmd\.exe", "write_file": r"\b(writeFile(Sync)?|open\([^)]*['\"][wa]b?['\"])",
}
_DANG = {k: re.compile(v) for k, v in DANGEROUS.items()}
FEATURE_NAMES = (["n_files", "n_binary_types", "total_chars", "entropy_mean", "entropy_max", "max_line_len",
                  "mean_line_len", "b64_blobs", "hex_blobs", "url_count", "ip_count", "dangerous_total"]
                 + [f"dang_{k}" for k in DANGEROUS])


def numeric_features(rec: dict) -> dict:
    files = rec["files"]
    texts = [f["text"] for f in files]
    blob = "\n".join(texts)
    ents = [shannon_entropy(t) for t in texts] or [0.0]
    lines = blob.split("\n")
    lens = [len(l) for l in lines] or [0]
    feats = {
        "n_files": len(files), "n_binary_types": len(rec.get("binaries") or {}), "total_chars": len(blob),
        "entropy_mean": sum(ents) / len(ents), "entropy_max": max(ents), "max_line_len": max(lens),
        "mean_line_len": sum(lens) / len(lens), "b64_blobs": len(B64.findall(blob)),
        "hex_blobs": len(HEX.findall(blob)), "url_count": len(URL.findall(blob)), "ip_count": len(IP.findall(blob)),
    }
    d = {f"dang_{k}": len(p.findall(blob)) for k, p in _DANG.items()}
    feats["dangerous_total"] = sum(d.values())
    feats.update(d)
    return {k: float(v) for k, v in feats.items()}
