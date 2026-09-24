"""Is the app answering? Used by the container's HEALTHCHECK and after a deploy.

    python scripts/healthcheck.py                          # http://127.0.0.1:$PORT/
    python scripts/healthcheck.py --url https://... \\
        --expect NexusRAG --retries 20 --interval 15       # smoke test a deployment

Exits 0 when the server responds without a server error (and contains ``--expect``, if
given), 1 otherwise. Standard library only, so it runs inside the slim image.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Sequence

USER_AGENT = "NexusRAG-healthcheck/1.0"


def check(url: str, timeout_s: float, expect: str | None = None) -> tuple[bool, str]:
    """(healthy, detail). A 4xx still means the server is up and routing requests."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            status, body = response.status, response.read(200_000).decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code < 500, f"HTTP {exc.code}"
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return False, f"{type(exc).__name__}: {exc}"
    if status >= 500:
        return False, f"HTTP {status}"
    if expect and expect not in body:
        return False, f"HTTP {status} but {expect!r} is missing from the page"
    return True, f"HTTP {status}"


def wait_for(
    url: str, *, timeout_s: float, expect: str | None, retries: int, interval_s: float
) -> tuple[bool, str]:
    """Check ``url`` until it is healthy or the retries run out."""
    healthy, detail = False, "not checked"
    for attempt in range(1, retries + 1):
        healthy, detail = check(url, timeout_s, expect)
        if healthy or attempt == retries:
            break
        print(f"attempt {attempt}/{retries}: {detail}", file=sys.stderr, flush=True)
        time.sleep(interval_s)
    return healthy, detail


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default=None, help="default: http://127.0.0.1:$PORT/")
    parser.add_argument("--timeout", type=float, default=5.0, help="seconds per request")
    parser.add_argument("--expect", default=None, help="text the page must contain")
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--interval", type=float, default=15.0, help="seconds between retries")
    args = parser.parse_args(argv)
    url = args.url or f"http://127.0.0.1:{os.environ.get('PORT', '8000')}/"
    healthy, detail = wait_for(
        url,
        timeout_s=args.timeout,
        expect=args.expect,
        retries=max(1, args.retries),
        interval_s=args.interval,
    )
    print(f"{'ok' if healthy else 'unhealthy'}: {url} ({detail})", file=sys.stderr)
    return 0 if healthy else 1


if __name__ == "__main__":
    raise SystemExit(main())
