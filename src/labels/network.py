"""aux_network: code that can open network connections."""
from __future__ import annotations

import re

from src.labels.base import LabelOut, rx_hits, run_label
from src.labels.facts import Facts

PY_MODS = {"urllib", "urllib2", "urllib3", "requests", "httpx", "aiohttp", "http.client", "httplib", "socket",
           "ftplib", "smtplib", "telnetlib", "websocket", "websockets", "pycurl", "xmlrpc.client"}
JS_MODS = {"http", "https", "net", "dns", "tls", "dgram", "http2", "axios", "node-fetch", "request", "got",
           "superagent", "ws", "undici", "needle"}
JS_CALLS = {"fetch", "new XMLHttpRequest", "new WebSocket", "navigator.sendBeacon"}
RX = [re.compile(p) for p in (
    r"\b(urlopen|urlretrieve)\s*\(", r"\brequests\.(get|post|put)\s*\(", r"\bsocket\.(socket|create_connection)\b",
    r"\bhttp\.client\b", r"require\(\s*['\"](https?|net|dns|axios|node-fetch|request|got|ws)['\"]\s*\)",
    r"\bfetch\s*\(", r"XMLHttpRequest|new\s+WebSocket")]


def _mod_hit(imports: set, mods: set) -> list[str]:
    return [f"import {i}" for i in imports if any(i == m or i.startswith(m + ".") for m in mods)]


def _ast(f: Facts) -> list[str]:
    hits = _mod_hit(f.imports, PY_MODS if f.lang == "py" else JS_MODS)
    if f.lang == "js":
        hits += [f"call {c}" for c in f.calls if c in JS_CALLS]
    return hits


def aux_network(rec: dict, facts: list[Facts]) -> LabelOut:
    return run_label(facts, _ast, lambda f: rx_hits(f.text, RX))
