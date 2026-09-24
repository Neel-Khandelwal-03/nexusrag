"""Container health check: is the Chainlit server answering?

    python scripts/healthcheck.py            # checks http://127.0.0.1:$PORT/
    python scripts/healthcheck.py --url ...  # or an explicit URL

Exits 0 when the server responds without a server error, 1 otherwise. Used by the
Dockerfile's HEALTHCHECK and by docker-compose, so it only needs the standard library.
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.error
import urllib.request
from collections.abc import Sequence


def check(url: str, timeout_s: float) -> tuple[bool, str]:
    """(healthy, detail). A 4xx still means the server is up and routing requests."""
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as response:
            return response.status < 500, f"HTTP {response.status}"
    except urllib.error.HTTPError as exc:
        return exc.code < 500, f"HTTP {exc.code}"
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return False, f"{type(exc).__name__}: {exc}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default=None, help="default: http://127.0.0.1:$PORT/")
    parser.add_argument("--timeout", type=float, default=5.0)
    args = parser.parse_args(argv)
    url = args.url or f"http://127.0.0.1:{os.environ.get('PORT', '8000')}/"
    healthy, detail = check(url, args.timeout)
    print(f"{'ok' if healthy else 'unhealthy'}: {url} ({detail})", file=sys.stderr)
    return 0 if healthy else 1


if __name__ == "__main__":
    raise SystemExit(main())
