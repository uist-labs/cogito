#!/usr/bin/env python3
"""TDD tests for cogito_modeldl -- the resumable download engine.

No network: a fake opener serves bytes from an in-memory payload with Range
support, so every branch (full download, resume, server-ignores-range restart,
size/sha verification, token header, already-present skip, retry/give-up) is
exercised offline. Stdlib unittest.

Run with:  uv run python -m unittest tests.test_modeldl
"""

import hashlib
import io
import tempfile
import unittest
import urllib.error
from pathlib import Path

import cogito_modeldl as dl

PAYLOAD = b"COGITO-GGUF-" + bytes(range(256)) * 40  # 10252 bytes, deterministic
SHA = hashlib.sha256(PAYLOAD).hexdigest()


class FakeResponse:
    def __init__(self, data, status):
        self._buf = io.BytesIO(data)
        self.status = status
        self.headers = {"Content-Length": str(len(data))}

    def read(self, n):
        return self._buf.read(n)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def make_opener(payload=PAYLOAD, *, calls=None, force_status=None, fail_times=0):
    """A fake opener(url, headers) with Range support and optional flakiness."""
    state = {"fails": fail_times}

    def opener(url, headers):
        if calls is not None:
            calls.append(dict(headers))
        if state["fails"] > 0:
            state["fails"] -= 1
            raise urllib.error.URLError("simulated network drop")
        rng = headers.get("Range")
        start, status = 0, 200
        if rng:
            start = int(rng.split("=")[1].split("-")[0])
            status = 206
        if force_status is not None:      # server ignores Range -> full 200
            status = force_status
            if status == 200:
                start = 0
        return FakeResponse(payload[start:], status)

    return opener


class DownloadTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.dest = self.tmp / "model.gguf"
        self.part = self.tmp / "model.gguf.part"

    def tearDown(self):
        self._tmp.cleanup()

    def _dl(self, **kw):
        kw.setdefault("size_bytes", len(PAYLOAD))
        kw.setdefault("opener", make_opener())
        kw.setdefault("out", io.StringIO())
        kw.setdefault("sleep", lambda _s: None)
        return dl.download(str(self.tmp / "model.gguf") if "url" not in kw else kw.pop("url"),
                           self.dest, **kw)

    def test_full_download_writes_the_file(self):
        self._dl()
        self.assertEqual(self.dest.read_bytes(), PAYLOAD)
        self.assertFalse(self.part.exists())  # .part renamed away

    def test_resume_sends_range_and_completes(self):
        self.part.write_bytes(PAYLOAD[:4000])
        calls = []
        self._dl(opener=make_opener(calls=calls))
        self.assertEqual(calls[0].get("Range"), "bytes=4000-")
        self.assertEqual(self.dest.read_bytes(), PAYLOAD)

    def test_server_ignoring_range_restarts_cleanly(self):
        self.part.write_bytes(PAYLOAD[:4000])
        self._dl(opener=make_opener(force_status=200))
        self.assertEqual(self.dest.read_bytes(), PAYLOAD)

    def test_size_mismatch_is_terminal_and_keeps_part(self):
        with self.assertRaises(dl.DownloadError):
            self._dl(size_bytes=len(PAYLOAD) - 10)  # server returns more than expected
        self.assertTrue(self.part.exists())
        self.assertFalse(self.dest.exists())

    def test_sha256_pass(self):
        self._dl(sha256=SHA)
        self.assertEqual(self.dest.read_bytes(), PAYLOAD)

    def test_sha256_fail_is_terminal(self):
        with self.assertRaises(dl.DownloadError):
            self._dl(sha256="0" * 64)
        self.assertTrue(self.part.exists())

    def test_token_from_env_sets_authorization(self):
        calls = []
        self._dl(opener=make_opener(calls=calls), env={"HF_TOKEN": "secret"})
        self.assertEqual(calls[0].get("Authorization"), "Bearer secret")

    def test_no_token_no_authorization_header(self):
        calls = []
        self._dl(opener=make_opener(calls=calls), env={})
        self.assertNotIn("Authorization", calls[0])

    def test_already_present_skips_download(self):
        self.dest.write_bytes(PAYLOAD)

        def boom(url, headers):
            raise AssertionError("opener must not be called when file is complete")

        result = self._dl(opener=boom)
        self.assertEqual(Path(result), self.dest)

    def test_retry_then_success(self):
        calls = []
        self._dl(opener=make_opener(calls=calls, fail_times=2), retries=3)
        self.assertEqual(len(calls), 3)  # two failures + one success
        self.assertEqual(self.dest.read_bytes(), PAYLOAD)

    def test_retry_exhausted_raises_with_url(self):
        with self.assertRaises(dl.DownloadError) as ctx:
            self._dl(url="https://hf.co/acme/x.gguf",
                     opener=make_opener(fail_times=9), retries=3)
        self.assertIn("hf.co/acme/x.gguf", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
