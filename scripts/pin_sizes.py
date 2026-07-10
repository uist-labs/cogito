#!/usr/bin/env python3
"""One-off: HEAD each catalog model's HF resolve URL to pin size_bytes and
confirm the repo/file resolves ungated (HTTP 200, not 401/403/404).

Prints a table of key -> status, Content-Length. Not part of the package; run
manually when the catalog changes.  Usage: uv run python scripts/pin_sizes.py
"""
import sys
import urllib.error
import urllib.request

sys.path.insert(0, ".")
import cogito_models as m  # noqa: E402


def head(url):
    req = urllib.request.Request(url, method="HEAD")
    req.add_header("User-Agent", "cogito-pin/0.1")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, resp.headers.get("Content-Length")
    except urllib.error.HTTPError as e:
        return e.code, None
    except Exception as e:  # noqa: BLE001
        return f"ERR {type(e).__name__}", None


def main():
    print(f"{'key':<22} {'status':<8} {'bytes':>14}  {'~GB':>6}")
    for mod in m.iter_models():
        status, clen = head(m.download_url(mod))
        gb = f"{int(clen)/1e9:.2f}" if clen else "-"
        print(f"{mod.key:<22} {str(status):<8} {str(clen or '-'):>14}  {gb:>6}")


if __name__ == "__main__":
    main()
