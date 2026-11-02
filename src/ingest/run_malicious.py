"""Stage 1: read malicious archives in memory -> data/interim/malicious/shard-*.jsonl.gz (resumable)."""
from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor

from src.ingest.archive_reader import Archive
from src.ingest.build_record import build_record
from src.ingest.malicious_index import build_index
from src.utils.common import ROOT, data_dir, load_config, write_jsonl_gz

_CFG = load_config()


def process(item: dict) -> dict:
    """Never raises: failures come back as {'error': ...} so one bad archive can't stop a shard."""
    try:
        a = _CFG["archive"]
        arc = Archive(item["path"], a["zip_password"], a["max_archive_bytes"], a["max_members"])
        return build_record(arc, dict(item, label=1), _CFG["select"])
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {str(e)[:100]}", "eco": item["eco"],
                "package": item["package"], "version": item["version"], "source_dataset": item["source_dataset"]}


def main() -> None:
    cfg = _CFG
    raw, interim = ROOT / cfg["paths"]["raw"], data_dir(cfg, "interim")
    out = interim / "malicious"
    out.mkdir(exist_ok=True)
    rows = build_index(raw, cfg["ingest"]["pre_cap_versions"], interim / "malicious_index.jsonl")
    n = cfg["ingest"]["shard_size"]
    shards = [rows[i:i + n] for i in range(0, len(rows), n)]
    print(f"items={len(rows)} shards={len(shards)}", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(cfg["ingest"]["workers"]) as ex:
        for i, chunk in enumerate(shards):
            path = out / f"shard-{i:05d}.jsonl.gz"
            if path.exists():
                continue
            write_jsonl_gz(path, ex.map(process, chunk, chunksize=8))
            print(f"shard {i + 1}/{len(shards)} t={int(time.time() - t0)}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
