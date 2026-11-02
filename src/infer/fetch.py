"""Registry download (in memory only) -> the same record the training pipeline builds. Nothing is installed or run."""
from __future__ import annotations

from typing import Optional

from src.ingest.archive_reader import Archive
from src.ingest.benign_fetch import fetch_npm, fetch_pypi
from src.ingest.build_record import build_record
from src.ingest.registry import Client
from src.utils.common import norm_name


class FetchError(RuntimeError):
    pass


def download(eco: str, name: str, version: str, cfg: dict, cli: Optional[Client] = None) -> tuple[bytes, str, Optional[str]]:
    """-> (archive bytes, resolved version, publish date). `version` may be 'latest'."""
    cli = cli or Client(cfg["benign"]["user_agent"], cfg["benign"]["rate_per_sec"])
    c = {"eco": eco, "package": norm_name(eco, name), "version": None if version == "latest" else version}
    got = (fetch_pypi if eco == "pypi" else fetch_npm)(cli, c, cfg["benign"]["max_download_bytes"])
    if got is None:
        raise FetchError(f"could not fetch {eco}/{name}@{version} (not found, no source archive, or too large)")
    return got


def record_from_bytes(eco: str, name: str, version: str, data: bytes, cfg: dict, date: Optional[str] = None) -> dict:
    """Same Archive -> build_record call the benign fetch uses (file selection + size guards included)."""
    a = cfg["archive"]
    arc = Archive(data, a["zip_password"], a["max_archive_bytes"], a["max_members"])
    meta = {"eco": eco, "package": norm_name(eco, name), "version": version, "label": -1, "source_dataset": "inference",
            "is_compromised": False, "upload_date": date, "date_kind": "registry_publish" if date else None, "path": "",
            "popularity_bucket": "unknown", "downloads": None, "pool": "main"}
    return build_record(arc, meta, cfg["select"])
