"""Rate-limited, retrying HTTP client for PyPI / npm metadata and artifact downloads."""
from __future__ import annotations

import threading
import time
from typing import Optional

import requests


class RateLimiter:
    def __init__(self, per_sec: float):
        self.gap, self.next, self.lock = 1.0 / per_sec, 0.0, threading.Lock()

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            self.next = max(self.next, now) + self.gap
            delay = self.next - self.gap - now
        if delay > 0:
            time.sleep(delay)


class Client:
    def __init__(self, user_agent: str, per_sec: float, retries: int = 4):
        self.s = requests.Session()
        self.s.headers["User-Agent"] = user_agent
        self.rl, self.retries = RateLimiter(per_sec), retries

    def get(self, url: str, max_bytes: Optional[int] = None, **kw) -> Optional[requests.Response]:
        """GET with backoff on 429/5xx. Returns None for 404/oversize/exhausted retries."""
        for attempt in range(self.retries):
            self.rl.wait()
            try:
                r = self.s.get(url, timeout=40, stream=max_bytes is not None, **kw)
            except requests.RequestException:
                time.sleep(2 ** attempt)
                continue
            if r.status_code == 404:
                return None
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(2 ** attempt * 2)
                continue
            if r.status_code != 200:
                return None
            if max_bytes is not None:
                if int(r.headers.get("Content-Length", 0)) > max_bytes:
                    r.close()
                    return None
                buf = bytearray()
                for chunk in r.iter_content(65536):
                    buf += chunk
                    if len(buf) > max_bytes:
                        r.close()
                        return None
                r._content = bytes(buf)  # noqa: SLF001 - expose via .content
            return r
        return None

    def json(self, url: str, **kw):
        r = self.get(url, **kw)
        try:
            return r.json() if r is not None else None
        except ValueError:
            return None
