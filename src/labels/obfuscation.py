"""aux_obfuscation: dynamic execution of decoded payloads, obfuscator signatures."""
from __future__ import annotations

import re

from src.labels.base import LabelOut, rx_hits, run_label
from src.labels.facts import Facts

PY_EXEC = {"exec", "eval", "compile", "__import__"}
PY_DECODE = {"base64.b64decode", "base64.b32decode", "base64.b16decode", "base64.decodebytes", "zlib.decompress",
             "marshal.loads", "codecs.decode", "binascii.unhexlify", "binascii.a2b_base64", "bytes.fromhex",
             "pickle.loads", "lzma.decompress", "bz2.decompress"}
JS_EXEC = {"eval", "Function", "new Function", "vm.runInNewContext", "vm.runInThisContext"}
JS_DECODE = {"atob", "unescape", "String.fromCharCode", "Buffer.from", "zlib.inflateSync", "zlib.gunzipSync"}
OBF_ID = re.compile(r"_0x[0-9a-f]{4,6}")
HEX_RUN = re.compile(r"(?:\\x[0-9a-fA-F]{2}){20,}")
RX = [re.compile(p) for p in (
    r"\b(exec|eval)\s*\([^)]{0,80}(b64decode|decompress|unhexlify|marshal|codecs\.decode)",
    r"\beval\s*\([^)]{0,80}(atob|fromCharCode|Buffer\.from)", r"_0x[0-9a-f]{4,6}.{0,200}_0x[0-9a-f]{4,6}")]


def _ast(f: Facts) -> list[str]:
    ex, dec = (PY_EXEC, PY_DECODE) if f.lang == "py" else (JS_EXEC, JS_DECODE)
    e, d = f.calls & ex, f.calls & dec
    hits = [f"{sorted(e)[0]} + {sorted(d)[0]}"] if e and d else []
    if len(OBF_ID.findall(f.text)) >= 8:
        hits.append("_0x identifiers")
    if HEX_RUN.search(f.text):
        hits.append("hex-escape run")
    return hits


def aux_obfuscation(rec: dict, facts: list[Facts]) -> LabelOut:
    return run_label(facts, _ast, lambda f: rx_hits(f.text, RX + [HEX_RUN]))
