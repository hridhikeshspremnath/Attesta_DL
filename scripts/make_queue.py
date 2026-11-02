"""Generate configs/queue/*.yaml (one per run) and scripts/queue_m2.ps1 order file configs/queue/order.txt."""
from __future__ import annotations

import yaml

from src.utils.common import ROOT

SHORTCUTS = ["total_chars", "n_files", "n_binary_types", "name_len", "min_dist_top_n", "min_dist_known_mal"]
# (variant, seed) in priority order: most informative first, so an interrupted night still has the key answers
RUNS = [("noshort", 13), ("bytesonly", 13), ("full", 1), ("noshort", 1), ("bytesonly", 1),
        ("full", 2), ("noshort", 2), ("bytesonly", 2)]


def make(variant: str, seed: int) -> dict:
    cfg = yaml.safe_load((ROOT / "configs" / "m2.yaml").read_text())
    name = f"{variant}_s{seed}"
    if variant in ("noshort", "bytesonly"):
        cfg["data"]["drop_features"] = SHORTCUTS
    if variant == "bytesonly":
        cfg["model"]["use_numeric"] = False
    cfg["train"]["seed"] = seed
    cfg["train"]["ckpt_dir"] = f"checkpoints/q_{name}"
    cfg["eval"]["report"] = f"reports/queue/{name}.md"
    return name, cfg


def main() -> None:
    out = ROOT / "configs" / "queue"
    out.mkdir(exist_ok=True)
    names = []
    for variant, seed in RUNS:
        name, cfg = make(variant, seed)
        (out / f"{name}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        names.append(name)
    (out / "order.txt").write_text("\n".join(names) + "\n")
    print(names)


if __name__ == "__main__":
    main()
