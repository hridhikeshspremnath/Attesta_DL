"""Download (never install) benign sources from PyPI / npm and turn them into records. Resumable."""
from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from src.ingest.archive_reader import Archive
from src.ingest.build_record import build_record
from src.ingest.registry import Client
from src.utils.common import vkey, write_jsonl_gz


def _pypi_file(files: list[dict]) -> Optional[dict]:
    live = [f for f in files if not f.get("yanked")]
    for kind in ("sdist", "bdist_wheel"):  # sdist first: same shape as the malicious tarballs
        for f in live:
            if f["packagetype"] == kind and f["size"] <= 30_000_000:
                return f
    return None


def fetch_pypi(cli: Client, c: dict, max_bytes: int) -> Optional[tuple[bytes, str, Optional[str]]]:
    ver = c.get("version")
    d = cli.json(f"https://pypi.org/pypi/{c['package']}/{ver}/json" if ver else f"https://pypi.org/pypi/{c['package']}/json")
    if not d:
        return None
    f = _pypi_file(d.get("urls") or [])
    if not f:
        return None
    r = cli.get(f["url"], max_bytes=max_bytes)
    return (r.content, d["info"]["version"], (f.get("upload_time_iso_8601") or "")[:10] or None) if r else None


def npm_versions(cli: Client, name: str) -> Optional[dict]:
    return cli.json(f"https://registry.npmjs.org/{name.replace('/', '%2F')}",
                    headers={"Accept": "application/vnd.npm.install-v1+json"})


def fetch_npm(cli: Client, c: dict, max_bytes: int) -> Optional[tuple[bytes, str, Optional[str]]]:
    doc = npm_versions(cli, c["package"])
    if not doc:
        return None
    ver = c.get("version") or (doc.get("dist-tags") or {}).get("latest")
    meta = (doc.get("versions") or {}).get(ver)
    if not meta:
        return None
    r = cli.get(meta["dist"]["tarball"], max_bytes=max_bytes)
    date = c.get("date") if not c.get("version") else None
    if date is None:  # date only needed for rows without search metadata
        full = cli.json(f"https://registry.npmjs.org/{c['package'].replace('/', '%2F')}")
        date = ((full or {}).get("time", {}).get(ver) or "")[:10] or None
    return (r.content, ver, date) if r else None


def process(cli: Client, c: dict, cfg: dict, excl) -> dict:
    try:
        fn = fetch_pypi if c["eco"] == "pypi" else fetch_npm
        got = fn(cli, c, cfg["benign"]["max_download_bytes"])
        if got is None:
            return {"error": "fetch_failed", **{k: c[k] for k in ("eco", "package")}}
        data, ver, date = got
        if excl(c["eco"], c["package"], ver):  # OSSF: whole-package or this exact version is known-malicious
            return {"error": "ossf_excluded", **{k: c[k] for k in ("eco", "package")}}
        a = cfg["archive"]
        arc = Archive(data, a["zip_password"], a["max_archive_bytes"], a["max_members"])
        meta = {"eco": c["eco"], "package": c["package"], "version": ver, "label": 0,
                "source_dataset": c.get("src", "registry"), "is_compromised": c.get("is_compromised", False),
                "upload_date": date, "date_kind": "registry_publish" if date else None, "path": "",
                "popularity_bucket": c["stratum"], "downloads": c.get("downloads"), "pool": c.get("pool", "main")}
        rec = build_record(arc, meta, cfg["select"])
        low_registry = c["stratum"] == "low" and c.get("src", "registry") == "registry" and not c.get("hard_negative")
        if low_registry and sum(len(f["text"]) for f in rec["files"]) < cfg["benign"]["near_empty_chars"]:
            return {"error": "near_empty", **{k: c[k] for k in ("eco", "package")}}
        return rec
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {str(e)[:100]}", **{k: c[k] for k in ("eco", "package")}}


def run(cands: list[dict], out_dir: Path, cfg: dict, excl, shard: int = 500) -> None:
    """One shard = 500 items = one progress line. Shards are written atomically in candidate order;
    finished shards are skipped on rerun, so an interrupt loses at most the shard in flight."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cli = Client(cfg["benign"]["user_agent"], cfg["benign"]["rate_per_sec"])  # 40s timeout, 4 retries w/ backoff
    chunks = [cands[i:i + shard] for i in range(0, len(cands), shard)]
    skipped = sum(1 for i in range(len(chunks)) if (out_dir / f"shard-{i:05d}.jsonl.gz").exists())
    print(f"{time.strftime('%H:%M:%S')} start: {len(cands)} items, {len(chunks)} shards, {skipped} already done", flush=True)
    ok = err = 0
    with ThreadPoolExecutor(8) as ex:
        for i, ch in enumerate(chunks):
            path = out_dir / f"shard-{i:05d}.jsonl.gz"
            if path.exists():
                continue
            res = list(ex.map(lambda c: process(cli, c, cfg, excl), ch))
            write_jsonl_gz(path, res)
            bad = sum("error" in r for r in res)
            ok, err = ok + len(res) - bad, err + bad
            print(f"{time.strftime('%H:%M:%S')} items {min((i + 1) * shard, len(cands))}/{len(cands)} "
                  f"ok={ok} err={err} (this session)", flush=True)
    print(f"{time.strftime('%H:%M:%S')} finished", flush=True)
