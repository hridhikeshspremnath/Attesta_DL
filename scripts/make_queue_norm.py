"""Generate configs/queue_norm/*.yaml: full and bytes-only with data.normalize on, seeds 13/1/2."""
from __future__ import annotations

import yaml

from scripts.make_queue import SHORTCUTS
from src.utils.common import ROOT

RUNS = [("bytesonly_norm", 13), ("full_norm", 13), ("bytesonly_norm", 1), ("full_norm", 1),
        ("bytesonly_norm", 2), ("full_norm", 2)]


def make(variant: str, seed: int) -> tuple[str, dict]:
    cfg = yaml.safe_load((ROOT / "configs" / "m2.yaml").read_text())
    name = f"{variant}_s{seed}"
    cfg["data"]["normalize"] = True
    if variant == "bytesonly_norm":
        cfg["data"]["drop_features"] = SHORTCUTS
        cfg["model"]["use_numeric"] = False
    cfg["train"]["seed"] = seed
    cfg["train"]["ckpt_dir"] = f"checkpoints/q_{name}"
    cfg["eval"]["report"] = f"reports/queue_norm/{name}.md"
    return name, cfg


def main() -> None:
    out = ROOT / "configs" / "queue_norm"
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
