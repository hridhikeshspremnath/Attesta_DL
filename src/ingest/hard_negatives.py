"""Clean neighbouring versions of compromised packages (benign hard negatives)."""
from __future__ import annotations

from src.ingest.registry import Client
from src.utils.common import vkey


def registry_versions(cli: Client, eco: str, name: str) -> list[str]:
    if eco == "pypi":
        d = cli.json(f"https://pypi.org/pypi/{name}/json")
        return [v for v, files in (d or {}).get("releases", {}).items() if files]
    d = cli.json(f"https://registry.npmjs.org/{name.replace('/', '%2F')}",
                 headers={"Accept": "application/vnd.npm.install-v1+json"})
    return list((d or {}).get("versions", {}))


def neighbours(all_versions: list[str], bad: set[str], k: int) -> list[str]:
    """Nearest clean version below and above the affected range (alternating), up to k."""
    vs = sorted(all_versions, key=vkey)
    idx = [i for i, v in enumerate(vs) if v in bad]
    if not idx:
        return []
    lo, hi = idx[0] - 1, idx[-1] + 1
    out = []
    while len(out) < k and (lo >= 0 or hi < len(vs)):
        for j in (lo, hi):
            if 0 <= j < len(vs) and vs[j] not in bad and len(out) < k:
                out.append(vs[j])
        lo, hi = lo - 1, hi + 1
    return out


def hard_negative_candidates(cli: Client, mal: "pd.DataFrame", ossf: dict, k: int) -> list[dict]:  # noqa: F821
    out = []
    comp = mal[mal.is_compromised]
    for (eco, pkg), g in comp.groupby(["eco", "package"]):
        bad = set(g.version) | set(ossf[eco]["versions"].get(pkg, []))
        for v in neighbours(registry_versions(cli, eco, pkg), bad, k):
            out.append({"eco": eco, "package": pkg, "version": v, "stratum": "unknown", "downloads": None,
                        "is_compromised": True, "src": "registry", "hard_negative": True})
    return out
