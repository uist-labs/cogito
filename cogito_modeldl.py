#!/usr/bin/env python3
"""COGITO model download engine -- resumable, stdlib-only, no dependencies.

Streams a GGUF from a URL to ``dest`` with a ``.part`` staging file, HTTP Range
resume, a progress bar, optional ``HF_TOKEN`` auth, and size/sha256 verification
before an atomic rename into place. Kept separate from the picker wizard
(cogito_modelpick) so it is unit-testable in isolation: the only side-effect
seam is an injected ``opener(url, headers) -> response`` (default urllib).

Design notes:
  * No huggingface_hub dependency -- the front door stays dependency-light. We
    implement Range-resume ourselves (~a few lines) rather than pull a library.
  * HF_TOKEN / HF_HUB_TOKEN are honored (Authorization: Bearer) so an advanced
    user can reach gated repos, but the curated catalog is all ungated so the
    token is never required.
  * Interrupted downloads leave the ``.part`` for a resume on re-run; only a
    size-overrun or sha256 mismatch is terminal (the ``.part`` is kept for
    inspection). A clean-EOF short read resumes like a dropped connection.
"""

import hashlib
import http.client
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

_CHUNK = 1 << 20  # 1 MiB
_TRANSIENT = (urllib.error.URLError, TimeoutError, ConnectionError,
              http.client.IncompleteRead)


class DownloadError(Exception):
    """A download failed in a way a retry will not fix (or retries were spent)."""


def _default_opener(url, headers):
    req = urllib.request.Request(url, headers=headers)
    return urllib.request.urlopen(req, timeout=60)


def _backoff(attempt):
    return min(2 ** (attempt - 1), 30)


def _progress(out, name, have, total, final=False):
    total = total or 1
    filled = int(24 * have / total)
    bar = "#" * filled + "-" * (24 - filled)
    end = "\n" if final else "\r"
    out.write(f"  {name}  [{bar}] {have / total * 100:5.1f}%  "
              f"{have / 1e6:8.1f}/{total / 1e6:.1f} MB{end}")
    out.flush()


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(_CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def _give_up_msg(url, exc):
    return (f"Download failed after retries ({exc}).\n"
            f"  Try manually (resumes a partial file):\n"
            f"    curl -L -C - -o <dest> {url}\n"
            f"    # or: wget -c {url}")


def download(url, dest, *, size_bytes, sha256=None, opener=None, env=None,
             out=None, token=None, retries=3, chunk=_CHUNK, sleep=None):
    """Download ``url`` to ``dest`` (resumable), verify it, atomic-rename. Return dest.

    Raises DownloadError on a terminal failure (size overrun, sha256 mismatch,
    or retries exhausted), leaving the ``.part`` file for inspection/resume.
    """
    opener = opener or _default_opener
    env = os.environ if env is None else env
    out = out or sys.stdout
    sleep = sleep or time.sleep
    dest = Path(dest)
    part = dest.with_name(dest.name + ".part")
    token = token or env.get("HF_TOKEN") or env.get("HF_HUB_TOKEN")

    if dest.exists() and dest.stat().st_size == size_bytes:
        return dest

    for attempt in range(1, retries + 1):
        have = part.stat().st_size if part.exists() else 0
        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        if have:
            headers["Range"] = f"bytes={have}-"

        try:
            with opener(url, headers) as resp:
                status = getattr(resp, "status", None) or resp.getcode()
                if have and status != 206:  # server ignored Range -> restart
                    have, mode = 0, "wb"
                else:
                    mode = "ab" if have else "wb"
                with open(part, mode) as f:
                    while True:
                        buf = resp.read(chunk)
                        if not buf:
                            break
                        f.write(buf)
                        have += len(buf)
                        _progress(out, dest.name, have, size_bytes)
            _progress(out, dest.name, have, size_bytes, final=True)
        except _TRANSIENT as exc:
            if attempt < retries:
                sleep(_backoff(attempt))
                continue
            raise DownloadError(_give_up_msg(url, exc)) from exc

        actual = part.stat().st_size
        if actual > size_bytes:
            raise DownloadError(
                f"size mismatch for {dest.name}: got {actual} bytes, expected "
                f"{size_bytes}. Kept {part.name} for inspection (delete it to retry).")
        if actual < size_bytes:  # clean EOF mid-stream -> resume
            if attempt < retries:
                sleep(_backoff(attempt))
                continue
            raise DownloadError(_give_up_msg(
                url, f"incomplete: {actual}/{size_bytes} bytes"))
        if sha256 and _sha256(part) != sha256:
            raise DownloadError(
                f"sha256 mismatch for {dest.name}; expected {sha256}. Kept "
                f"{part.name} for inspection (delete it to retry).")
        os.replace(part, dest)
        return dest

    raise DownloadError(_give_up_msg(url, "retries exhausted"))
