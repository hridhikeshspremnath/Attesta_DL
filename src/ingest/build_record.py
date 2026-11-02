"""Archive -> compact sample record (selected text + metadata). Shared by malicious and benign paths."""
from __future__ import annotations

import datetime as dt
import json
import posixpath
import re
from typing import Optional

from src.features.select_files import select_files
from src.ingest.archive_reader import Archive
from src.utils.common import content_hash


def _iso(ts: Optional[str]) -> Optional[str]:
    return ts[:10] if ts and re.match(r"\d{4}-\d{2}-\d{2}", ts) else None


def registry_date(arc: Archive, eco: str, version: str) -> Optional[str]:
    """Publish date from the registry JSON that DataDog bundles as package_info-*.json."""
    for n in arc.sizes:
        if posixpath.basename(n).startswith("package_info") and n.endswith(".json"):
            try:
                d = json.loads(arc.read(n, 2_000_000))
                if eco == "pypi":
                    rel = (d.get("releases") or {}).get(version) or d.get("urls") or []
                    return _iso(rel[0].get("upload_time_iso_8601") or rel[0].get("upload_time")) if rel else None
                return _iso((d.get("time") or {}).get(version))
            except Exception:
                return None
    return None


def build_record(arc: Archive, meta: dict, limits: dict) -> dict:
    sel = select_files(arc, meta["eco"], limits)
    date, kind = None, None
    if meta["source_dataset"] == "datadog":
        date = registry_date(arc, meta["eco"], meta["version"])
        kind = "registry_publish" if date else None
        if not date:
            m = re.search(r"(\d{4}-\d{2}-\d{2})-", meta["path"].replace("\\", "/").rsplit("/", 1)[-1])
            date, kind = (m.group(1), "datadog_discovery") if m else (None, None)
    elif meta["source_dataset"] == "malregistry" and arc.mtime:
        date, kind = dt.datetime.fromtimestamp(arc.mtime, dt.timezone.utc).strftime("%Y-%m-%d"), "tar_mtime"
    else:
        date, kind = meta.get("upload_date"), meta.get("date_kind")
    rec = {k: meta[k] for k in ("eco", "package", "version", "label", "source_dataset", "is_compromised")}
    rec.update({"upload_date": date, "date_kind": kind, "size_bytes": sel["size_bytes"],
                "n_members": sel["n_members"], "binaries": sel["binaries"], "files": sel["files"],
                "pkg_scripts": sel["pkg_json"].get("scripts") if isinstance(sel["pkg_json"].get("scripts"), dict) else {},
                "popularity_bucket": meta.get("popularity_bucket", "unknown"),
                "downloads": meta.get("downloads"), "pool": meta.get("pool", "main"), "content_hash": content_hash(sel["files"])})
    return rec
