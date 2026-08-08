"""Download the public datasets this project is built on, into this repo only.

Run once, before anything else::

    uv run python scripts/fetch_data.py          # fetch what is missing
    uv run python scripts/fetch_data.py --list   # what is registered, what is on disk
    uv run python scripts/fetch_data.py --verify # do the bytes still match the manifest
    uv run python scripts/fetch_data.py --clean  # delete every fetched file

Why this exists as a script rather than a line in the README: an instruction a
human has to follow by hand is an instruction that gets followed differently by
different people. "Download the survey and put it somewhere sensible" produces a
dozen slightly different filenames and one very confusing bug report. A script
puts the file in exactly one place, every time.

**Everything lands under `data/` inside this repository.** Nothing is written to
a home directory, a system cache, `/tmp`, or anywhere else on the machine. One
folder holds every byte this project fetched, so `--clean` really does clean and
nothing is left behind to go stale in a corner you forgot about.

Why the files are not committed to git: they are large, they are not ours, and
they are reproducible from a URL. Git is for things you wrote, not things you
fetched.

**But the manifest IS committed.** `data/MANIFEST.json` records, for every file
ever fetched, where it came from, under what licence, how big it was, its SHA-256
and the date it was pulled. That is the part that matters for tracing: a number
in `docs/findings.md` can be tied to an exact set of bytes even though those
bytes are not in the history. If the upstream file changes silently — which it
can, these are live URLs — `--verify` catches it instead of everyone quietly
computing different results from the "same" dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import date
from pathlib import Path

#: Repo root, derived from this file's location rather than the working
#: directory — running the script from anywhere must still write here and
#: nowhere else, which is the whole point.
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
PUBLIC_DIR = DATA_DIR / "public"
MANIFEST = DATA_DIR / "MANIFEST.json"


@dataclass(frozen=True)
class Dataset:
    name: str
    filename: str
    url: str
    approx_mb: int
    licence: str
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
        licence="ODbL 1.0",
        why="49,191 responses; 1,022 usable Indian salary rows after cleaning",
    ),
    # ── Earlier years, for one reason: 1,022 rows is why the bands are 240% wide.
    #
    # Pooling 2019-2024 adds roughly 15,000 more usable Indian rows. It does NOT
    # fix the missing city/employer/level columns — those were checked and do not
    # exist in any year — so this buys sample size, not new features.
    #
    # It also brings a hazard worth naming before anyone pools these naively: the
    # median clean Indian salary in this survey runs 7.2L (2019) -> 15L (2024).
    # That is real wage growth, and treating six years as one undifferentiated
    # pool would fold it into the spread and make the bands WIDER, not narrower.
    # Survey year has to be handled explicitly.
    #
    # These use the `media.githubusercontent.com/media/` host on purpose. The
    # ordinary `raw.githubusercontent.com` path returns a 134-byte Git-LFS
    # pointer that looks like a successful download and fails much later, during
    # parsing, a long way from the cause.
    *(
        Dataset(
            name=f"Stack Overflow Developer Survey {year}",
            filename=f"so_{year}_raw.csv",
            url=(
                "https://media.githubusercontent.com/media/StackExchange/Survey"
                f"/main/packages/archive/{year}/results.csv"
            ),
            approx_mb=mb,
            licence="ODbL 1.0",
            why=why,
        )
        for year, mb, why in (
            (2019, 196, "88,883 responses; ~3,730 usable Indian salary rows"),
            (2020, 95, "64,461 responses; ~2,497 usable Indian salary rows"),
            (2021, 81, "83,439 responses; ~3,642 usable Indian salary rows"),
            (2022, 109, "73,268 responses; ~2,045 usable Indian salary rows"),
            (2023, 159, "89,184 responses; ~1,799 usable Indian salary rows"),
            (2024, 160, "65,437 responses; ~1,446 usable Indian salary rows"),
        )
    ),
)


def _sha256(path: Path) -> str:
    """Full-file digest, streamed so a 134MB CSV does not become 134MB of RAM."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest() -> dict[str, dict]:
    if not MANIFEST.exists():
        return {}
    return {entry["filename"]: entry for entry in json.loads(MANIFEST.read_text())["datasets"]}


def _save_manifest(entries: dict[str, dict]) -> None:
    MANIFEST.write_text(
        json.dumps(
            {
                "note": (
                    "Provenance for files under data/. The files themselves are "
                    "gitignored (large, not ours, reproducible); this record is "
                    "committed so any published number can be traced to exact bytes. "
                    "Regenerate with: uv run python scripts/fetch_data.py"
                ),
                "datasets": [entries[k] for k in sorted(entries)],
            },
            indent=2,
        )
        + "\n"
    )


def _record(ds: Dataset, dest: Path, entries: dict[str, dict]) -> None:
    """Write this file's provenance into the manifest.

    Re-hashes rather than trusting the previous entry: the point of the record
    is to describe the bytes that are actually there now.
    """
    previous = entries.get(ds.filename, {})
    entries[ds.filename] = {
        "name": ds.name,
        "filename": ds.filename,
        "path": str(dest.relative_to(REPO_ROOT)),
        "url": ds.url,
        "licence": ds.licence,
        "why": ds.why,
        "bytes": dest.stat().st_size,
        "sha256": _sha256(dest),
        # Keep the ORIGINAL fetch date across re-records. Overwriting it on every
        # run would make every dataset look freshly pulled, which is exactly the
        # fact you want when judging whether a result is stale.
        "first_fetched": previous.get("first_fetched", date.today().isoformat()),
        "last_checked": date.today().isoformat(),
    }


def _download(ds: Dataset, dest: Path, *, force: bool) -> bool:
    """Fetch one dataset. Returns True if the file is present afterwards."""
    if dest.exists() and not force:
        print(f"  already present ({dest.stat().st_size / 1_000_000:.0f}MB) — skipping.")
        return True

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
    print(f"  saved {dest.relative_to(REPO_ROOT)} ({dest.stat().st_size / 1_000_000:.0f}MB)")
    return True


def _cmd_list(entries: dict[str, dict]) -> int:
    print(f"Registered datasets (all paths are inside {REPO_ROOT.name}/):\n")
    for ds in DATASETS:
        dest = PUBLIC_DIR / ds.filename
        entry = entries.get(ds.filename)
        state = "on disk" if dest.exists() else "NOT FETCHED"
        print(f"  {ds.name}")
        print(f"    {dest.relative_to(REPO_ROOT)}  [{state}]  licence: {ds.licence}")
        if entry:
            print(
                f"    sha256:{entry['sha256'][:16]}…  {entry['bytes'] / 1_000_000:.0f}MB  "
                f"first fetched {entry['first_fetched']}"
            )
        print()
    orphans = set(entries) - {d.filename for d in DATASETS}
    if orphans:
        print("In the manifest but no longer registered in this script:")
        for name in sorted(orphans):
            print(f"  {name} — kept for provenance; delete the entry by hand if truly retired")
    return 0


def _cmd_verify(entries: dict[str, dict]) -> int:
    """Do the bytes on disk still match what the manifest says?

    Upstream URLs are live and can change under you. Without this, a silently
    updated file means today's numbers stop matching yesterday's write-up and
    nobody can tell whether the model changed or the data did.
    """
    problems = 0
    for ds in DATASETS:
        dest = PUBLIC_DIR / ds.filename
        entry = entries.get(ds.filename)
        if not dest.exists():
            print(f"  MISSING   {ds.filename} — run without --verify to fetch it")
            problems += 1
        elif not entry:
            print(f"  UNRECORDED {ds.filename} — on disk but not in the manifest")
            problems += 1
        elif (actual := _sha256(dest)) != entry["sha256"]:
            print(f"  CHANGED   {ds.filename}")
            print(f"            manifest sha256:{entry['sha256'][:16]}…")
            print(f"            on disk  sha256:{actual[:16]}…")
            problems += 1
        else:
            print(f"  ok        {ds.filename}  sha256:{entry['sha256'][:16]}…")

    if problems:
        print(f"\n{problems} problem(s). Results computed from these files are not reproducible.")
        return 1
    print("\nEvery file matches its manifest entry.")
    return 0


def _cmd_clean() -> int:
    """Delete every fetched file. The manifest survives, deliberately.

    Wiping the provenance along with the bytes would mean a published number
    could no longer be traced to the data that produced it — and re-fetching
    then looks identical whether or not upstream changed in the meantime.
    """
    removed = 0
    for pattern in ("*.csv", "*.zip", "*.partial", "*.json"):
        for path in PUBLIC_DIR.glob(pattern):
            print(f"  removing {path.relative_to(REPO_ROOT)} ({path.stat().st_size / 1e6:.0f}MB)")
            path.unlink()
            removed += 1
    print(f"\n{removed} file(s) removed. {MANIFEST.name} kept — that is the audit trail.")
    print("Re-fetch with: uv run python scripts/fetch_data.py")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    parser.add_argument("--list", action="store_true", help="show what is registered and on disk")
    parser.add_argument("--verify", action="store_true", help="check files against the manifest")
    parser.add_argument("--clean", action="store_true", help="delete every fetched file")
    args = parser.parse_args()

    PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
    entries = _load_manifest()

    if args.list:
        return _cmd_list(entries)
    if args.verify:
        return _cmd_verify(entries)
    if args.clean:
        return _cmd_clean()

    print(f"Fetching public datasets into {PUBLIC_DIR.relative_to(REPO_ROOT)}/\n")
    ok = True
    for ds in DATASETS:
        print(f"{ds.name}")
        print(f"  {ds.why}")
        print(f"  licence: {ds.licence}")
        dest = PUBLIC_DIR / ds.filename
        if _download(ds, dest, force=args.force):
            _record(ds, dest, entries)
        else:
            ok = False
        print()

    _save_manifest(entries)
    print(f"Provenance written to {MANIFEST.relative_to(REPO_ROOT)}")

    if ok:
        print("Done. Next: uv run python scripts/run_analysis.py")
        return 0
    print("Some datasets are missing — see the errors above.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
