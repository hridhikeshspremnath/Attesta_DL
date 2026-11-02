"""Compute all aux labels for one record (and parse stats)."""
from __future__ import annotations

from src.labels.cred_access import aux_cred_access
from src.labels.facts import get_facts
from src.labels.install_exec import aux_install_exec
from src.labels.network import aux_network
from src.labels.obfuscation import aux_obfuscation
from src.labels.process_spawn import aux_process_spawn

LABELS = {"aux_install_exec": aux_install_exec, "aux_network": aux_network, "aux_cred_access": aux_cred_access,
          "aux_obfuscation": aux_obfuscation, "aux_process_spawn": aux_process_spawn}


def label_record(rec: dict) -> tuple[dict, list[dict], dict]:
    """-> ({aux_*: 0/1}, evidence rows, {label: [parsed_files, fallback_files]})"""
    facts = [get_facts(f["path"], f["text"]) for f in rec["files"]]
    vals, evid, stats = {}, [], {}
    for name, fn in LABELS.items():
        o = fn(rec, facts)
        vals[name] = int(o.value)
        stats[name] = [o.parsed, o.fallback]
        evid += [{"eco": rec["eco"], "package": rec["package"], "version": rec["version"], "label": name, **e}
                 for e in o.evidence]
    return vals, evid, stats
