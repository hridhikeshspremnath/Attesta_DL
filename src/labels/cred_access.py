"""aux_cred_access: touches credential stores / secret-bearing env vars."""
from __future__ import annotations

import re

from src.labels.base import LabelOut, rx_hits, run_label
from src.labels.facts import Facts

SENS_PATH = re.compile(
    r"(\.ssh\b|id_rsa|id_ed25519|\.aws[/\\]credentials|\.npmrc|\.pypirc|\.netrc|\.git-credentials|\.docker[/\\]config"
    r"|\.kube[/\\]config|/etc/(passwd|shadow)|Login Data|wallet\.dat|\.gnupg|\.config[/\\]gcloud|keychain"
    r"|metamask|Local Storage[/\\]leveldb)", re.I)
SENS_ENV = re.compile(r"(TOKEN|SECRET|PASSW(OR)?D|API[_-]?KEY|AWS_|CREDENTIAL|PRIVATE[_-]?KEY|NPM_CONFIG|GITHUB|GH_)", re.I)
RX = [SENS_PATH, re.compile(r"(os\.environ|process\.env)[^\n]{0,40}(TOKEN|SECRET|PASSW|API_?KEY|AWS_)", re.I),
      re.compile(r"(JSON\.stringify|dict|str)\(\s*(process\.env|os\.environ)\s*\)")]


def _ast(f: Facts) -> list[str]:
    hits = [f"path {s[:60]}" for s in f.strings if len(s) < 300 and SENS_PATH.search(s)]
    hits += [f"env {k}" for k in f.env_keys if SENS_ENV.search(k)]
    if f.env_bulk:
        hits.append("env dump")
    return hits


def aux_cred_access(rec: dict, facts: list[Facts]) -> LabelOut:
    return run_label(facts, _ast, lambda f: rx_hits(f.text, RX))
