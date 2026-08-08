# data/

**Every byte this project fetches lives here.** Nothing is written to a home
directory, a system cache, `/tmp`, or anywhere else on the machine. Delete this
folder and the project has no data; nothing is left behind in a corner to go
stale unnoticed.

```
data/
  MANIFEST.json    provenance for every fetched file        COMMITTED
  public/          openly-licensed survey data              gitignored
  company/         real payroll data                        gitignored, never commit
  synthetic/       generated locally from code              gitignored
```

## Why the data is gitignored but the manifest is not

The files are large, not ours to redistribute, and reproducible from a URL. Git
is for things you wrote, not things you downloaded.

But "reproducible from a URL" is only true if the URL still returns the same
bytes, and these are live upstream files that can change without warning. So
`MANIFEST.json` is committed and records, per file: source URL, licence, size,
SHA-256, and the date first fetched.

That is what makes a published number traceable. A figure in `docs/findings.md`
can be tied to an exact set of bytes even though those bytes were never in the
history — and if upstream quietly changes, `--verify` says so instead of
everyone computing different results from what they all call "the same dataset".

## Commands

```bash
uv run python scripts/fetch_data.py            # fetch what is missing, record provenance
uv run python scripts/fetch_data.py --list     # what is registered, what is on disk
uv run python scripts/fetch_data.py --verify   # do the bytes still match the manifest
uv run python scripts/fetch_data.py --clean    # delete every fetched file
```

`--clean` keeps `MANIFEST.json` on purpose. Wiping the provenance along with the
bytes would mean a published number could no longer be traced to the data that
produced it, and a later re-fetch would look identical whether or not upstream
had changed in between.

## Adding a new source

Add a `Dataset(...)` entry to `scripts/fetch_data.py`. It needs a `licence`
field — that is not paperwork, it is the check that stops unlicensed scraped
data entering a public repository. If you cannot name the licence, the dataset
does not go in.

## data/company/

Gitignored before it ever had anything in it, deliberately. Real payroll data
is not something to remember to exclude later.
