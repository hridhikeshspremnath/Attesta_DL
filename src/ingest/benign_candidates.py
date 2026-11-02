"""Benign candidate discovery: 3 popularity strata per ecosystem (+ malware_bench neutral names)."""
from __future__ import annotations

import csv
import json
import random
import string
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.ingest.registry import Client
from src.utils.common import norm_pypi

NPM_TOP_QUERIES = ["react", "express", "lodash", "webpack", "babel", "eslint", "typescript", "jest", "axios",
                   "util", "cli", "plugin", "types", "http", "aws", "vue", "angular", "logger", "parser", "crypto"]
NPM_INSTALL_QUERIES = ["postinstall", "node-gyp", "native", "prebuild", "binding", "esbuild", "husky", "sharp"]


def pypi_candidates(cli: Client, cfg: dict) -> list[dict]:
    b = cfg["benign"]
    top = cli.json(b["pypi_top_url"])["rows"]
    ranked = [norm_pypi(r["project"]) for r in top]
    dls = {norm_pypi(r["project"]): r["download_count"] for r in top}
    out = [{"eco": "pypi", "package": n, "stratum": "top" if i < b["pypi_top_cut"] else "mid", "downloads": dls[n]}
           for i, n in enumerate(ranked)]
    simple = cli.json("https://pypi.org/simple/", headers={"Accept": "application/vnd.pypi.simple.v1+json"})
    known = set(ranked)
    out += [{"eco": "pypi", "package": norm_pypi(p["name"]), "stratum": "low", "downloads": None}
            for p in simple["projects"] if norm_pypi(p["name"]) not in known]
    return out


def _npm_stratum(downloads: float) -> str:
    """Buckets on last-month downloads."""
    return "top" if downloads >= 100_000 else "mid" if downloads >= 5_000 else "low"


def npm_enrich(cli: Client, pool: list[dict], scoped_sample: int, seed: int, cache_path: Path) -> list[dict]:
    """Attach last-month downloads. Unscoped: bulk API (128/call). Scoped: single calls on a random sample only.
    Results persist in cache_path ({name: downloads|null}) so an interrupted run resumes without repeating calls."""
    rng = random.Random(seed)
    known: dict = json.loads(cache_path.read_text(encoding="utf8")) if cache_path.exists() else {}
    plain = [c for c in pool if not c["package"].startswith("@")]
    scoped = [c for c in pool if c["package"].startswith("@")]
    rng.shuffle(scoped)
    scoped = scoped[:scoped_sample]
    todo_plain = [c for c in plain if c["package"] not in known]
    todo_scoped = [c for c in scoped if c["package"] not in known]
    url = "https://api.npmjs.org/downloads/point/last-month/"

    def bulk(chunk: list[dict]) -> None:
        d = cli.json(url + ",".join(c["package"] for c in chunk)) or {}
        if len(chunk) == 1:  # single-name responses are not keyed by package
            d = {chunk[0]["package"]: d}
        for c in chunk:
            known[c["package"]] = (d.get(c["package"]) or {}).get("downloads")

    def single(c: dict) -> None:
        known[c["package"]] = (cli.json(url + c["package"]) or {}).get("downloads")

    t0, done = time.time(), [0]

    def tick(fn):
        def w(x):
            fn(x)
            done[0] += 1
            if done[0] % 100 == 0:
                cache_path.write_text(json.dumps(known), encoding="utf8")
                print(f"{time.strftime('%H:%M:%S')} enrich calls={done[0]} ({int(time.time() - t0)}s)", flush=True)
        return w

    with ThreadPoolExecutor(24) as ex:
        list(ex.map(tick(bulk), [todo_plain[i:i + 128] for i in range(0, len(todo_plain), 128)]))
        list(ex.map(tick(single), todo_scoped))
    cache_path.write_text(json.dumps(known), encoding="utf8")
    out = []
    for c in plain + scoped:
        if known.get(c["package"]) is not None:
            out.append(dict(c, downloads=known[c["package"]], stratum=_npm_stratum(known[c["package"]])))
    return out


def npm_candidates(cli: Client, want: int, seed: int) -> list[dict]:
    rng, seen, out = random.Random(seed), set(), []
    queries = [(q, "popularity=1.0&quality=0&maintenance=0") for q in NPM_TOP_QUERIES + NPM_INSTALL_QUERIES]
    while len(out) < want:
        queries += [("".join(rng.choices(string.ascii_lowercase, k=rng.choice((2, 3)))), "") for _ in range(40)]
        for q, extra in queries:
            for frm in (0, 250):
                d = cli.json(f"https://registry.npmjs.org/-/v1/search?text={q}&size=250&from={frm}&{extra}")
                for o in (d or {}).get("objects", []):
                    p = o["package"]
                    if p["name"] in seen:
                        continue
                    seen.add(p["name"])
                    out.append({"eco": "npm", "package": p["name"], "stratum": "low",
                                "downloads": None, "date": (p.get("date") or "")[:10] or None})
            if len(out) >= want:
                break
        queries = []
    return out


def npm_random(names_path: Path, n_plain: int, n_scoped: int, seed: int) -> list[dict]:
    """Random draw from the full npm name list (all-the-package-names): unbiased w.r.t. popularity."""
    names = json.loads(names_path.read_text(encoding="utf8"))
    rng = random.Random(seed)
    plain = [n for n in names if isinstance(n, str) and not n.startswith("@") and n.strip()]
    scoped = [n for n in names if isinstance(n, str) and n.startswith("@") and "/" in n]
    picks = rng.sample(plain, min(n_plain, len(plain))) + rng.sample(scoped, min(n_scoped, len(scoped)))
    return [{"eco": "npm", "package": n, "stratum": "low", "downloads": None, "date": None} for n in picks]


def malware_bench_neutral(raw: Path) -> list[dict]:
    """Decision 2: 40 neutral names -> benign candidates at their stated version (scoped+name normalized)."""
    out = []
    with open(raw / "malware_bench" / "Package_info.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["label"] != "neutral":
                continue
            name = f"{r['scoped']}/{r['name']}" if r["scoped"] else r["name"]
            out.append({"eco": "npm", "package": name, "version": r["version"], "stratum": "low",
                        "downloads": None, "src": "malware_bench"})
    return out


def sample_strata(cands: list[dict], n: int, fracs: dict, seed: int, exclude, low_reserve: float = 0.0) -> list[dict]:
    """Pick n candidates split by stratum fractions; exclude(c)->bool drops OSSF/malicious names.
    A stratum that cannot fill its quota is backfilled from the others (low first); the shortfall is printed."""
    rng = random.Random(seed)
    by: dict[str, list[dict]] = {}
    for c in cands:
        if not exclude(c):
            by.setdefault(c["stratum"], []).append(c)
    for pool in by.values():
        rng.shuffle(pool)
    picked, short = [], 0
    for s, fr in fracs.items():
        want = round(n * fr * (1 + (low_reserve if s == "low" else 0.0)))
        take = by.get(s, [])[:want]
        by[s] = by.get(s, [])[want:]
        picked += take
        short += want - len(take)
        if want - len(take):
            print(f"  stratum '{s}' short by {want - len(take)} (quota {want})", flush=True)
    for s in ("low", "top", "mid"):  # backfill
        while short > 0 and by.get(s):
            picked.append(by[s].pop())
            short -= 1
    return picked
