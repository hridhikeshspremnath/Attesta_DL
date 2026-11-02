"""In-memory, read-only access to zip/tar archives. Nothing is extracted or executed."""
from __future__ import annotations

import io
import tarfile
import zipfile
from pathlib import Path
from typing import Optional

import pyzipper

from src.utils.common import lp


class Archive:
    """Uniform view: names() -> {member: size}; read(member, cap) -> bytes."""

    def __init__(self, source: str | Path | bytes, password: str = "infected",
                 max_bytes: int = 200_000_000, max_members: int = 20_000):
        self.sizes: dict[str, int] = {}
        self.mtime: Optional[float] = None
        self._zip: Optional[pyzipper.AESZipFile] = None
        self._tar: Optional[tarfile.TarFile] = None
        self._members: dict[str, tarfile.TarInfo] = {}
        raw = source if isinstance(source, bytes) else None
        if raw is None:
            p = lp(source)
            import os
            if os.path.getsize(p) > max_bytes:
                raise ValueError("archive too large")
            raw = open(p, "rb").read()
        if len(raw) > max_bytes:
            raise ValueError("archive too large")
        buf = io.BytesIO(raw)
        if zipfile.is_zipfile(buf):
            self._zip = pyzipper.AESZipFile(buf)
            self._zip.setpassword(password.encode())
            for zi in self._zip.infolist():
                if not zi.is_dir():
                    self.sizes[zi.filename] = zi.file_size
        else:
            buf.seek(0)
            self._tar = tarfile.open(fileobj=buf, mode="r:*")
            for ti in self._tar.getmembers():
                if ti.isreg():  # skips symlinks/devices
                    self.sizes[ti.name] = ti.size
                    self._members[ti.name] = ti
                    if self.mtime is None:
                        self.mtime = ti.mtime
        if len(self.sizes) > max_members:
            raise ValueError("too many members")

    def read(self, name: str, cap: int = 200_000) -> bytes:
        if self._zip is not None:
            with self._zip.open(name) as f:
                return f.read(cap)
        f = self._tar.extractfile(self._members[name])
        return f.read(cap) if f else b""

    def raw_metadata_json(self, suffix: str, cap: int = 200_000) -> Optional[bytes]:
        for n in self.sizes:
            if n.endswith(suffix):
                return self.read(n, cap)
        return None
