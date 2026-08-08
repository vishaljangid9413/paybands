"""Download the public datasets this project is built on.

Run once, before anything else::

    uv run python scripts/fetch_data.py

Why this exists as a script rather than a line in the README: an instruction a
human has to follow by hand is an instruction that gets followed differently by
different people. "Download the survey and put it somewhere sensible" produces a
dozen slightly different filenames and one very confusing bug report. A script
puts the file in exactly one place, every time.

Why the file is not committed to git: it is 134MB, it is not ours, and it is
reproducible from a URL. Git is for things you wrote, not things you fetched.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

PUBLIC_DIR = Path("data/public")


@dataclass(frozen=True)
class Dataset:
    name: str
    filename: str
    url: str
    approx_mb: int
    why: str


DATASETS: tuple[Dataset, ...] = (
    Dataset(
        name="Stack Overflow Developer Survey 2025",
        filename="so_2025_raw.csv",
        # Stack Overflow publishes the raw results through their GitHub archive.
        # Pinned to the 2025 file rather than a "latest" alias on purpose: if the
        # data silently changed underneath us, every number in docs/findings.md
        # would become unreproducible without anyone noticing.
        url=(
            "https://github.com/StackExchange/Survey/raw/refs/heads/main"
            "/packages/archive/2025/results.csv"
        ),
        approx_mb=134,
        why="49,191 responses; 1,022 usable Indian salary rows after cleaning",
    ),
)


def _download(ds: Dataset, dest: Path, *, force: bool) -> bool:
    """Fetch one dataset. Returns True if it downloaded, False if already present."""
    if dest.exists() and not force:
        size_mb = dest.stat().st_size / 1_000_000
        print(f"  already present ({size_mb:.0f}MB) — skipping. Use --force to re-download.")
        return False

    print(f"  downloading ~{ds.approx_mb}MB from {ds.url.split('/')[2]} ...")
    # Download to a temporary name first. A half-downloaded file with the right
    # name is worse than no file at all: the loader would read it, fail oddly,
    # and you would go looking for the bug in the parsing code.
    tmp = dest.with_suffix(dest.suffix + ".partial")
    try:
        urllib.request.urlretrieve(ds.url, tmp)  # noqa: S310 — URL is a constant above
    except urllib.error.URLError as exc:
        tmp.unlink(missing_ok=True)
        print(f"  FAILED: {exc}", file=sys.stderr)
        print("  Check your connection, or download the URL above by hand into", dest)
        return False

    tmp.replace(dest)
    size_mb = dest.stat().st_size / 1_000_000
    digest = hashlib.sha256(dest.read_bytes()).hexdigest()[:16]
    print(f"  saved {dest} ({size_mb:.0f}MB, sha256:{digest}…)")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args()

    PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Fetching public datasets into {PUBLIC_DIR}/\n")

    ok = True
    for ds in DATASETS:
        print(f"{ds.name}")
        print(f"  {ds.why}")
        dest = PUBLIC_DIR / ds.filename
        _download(ds, dest, force=args.force)
        ok = ok and dest.exists()
        print()

    if ok:
        print("Done. Next: uv run python scripts/run_analysis.py")
        return 0
    print("Some datasets are missing — see the errors above.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
