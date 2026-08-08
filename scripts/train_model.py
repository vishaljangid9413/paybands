"""Train the model the API serves, and write it to `models/band.pkl`.

    uv run python scripts/train_model.py              # all survey years found
    uv run python scripts/train_model.py --years 2025 # one year, for comparison

Why this exists: the served artifact used to be produced by hand, in a shell,
from whatever was convenient at the time. That makes the single most important
question about a deployed model — *what was it trained on?* — unanswerable from
the repository. One command, checked in, answers it.

**Everything is read from and written to this repo.** Inputs come from
`data/public/`, output goes to `models/band.pkl`, and nothing touches a path
outside the project tree.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from paybands.api import service  # noqa: E402
from paybands.data.stackoverflow import load_years  # noqa: E402

PUBLIC_DIR = REPO_ROOT / "data" / "public"
MODEL_PATH = REPO_ROOT / "models" / "band.pkl"


def survey_files(years: list[int] | None) -> list[Path]:
    found = sorted(PUBLIC_DIR.glob("so_*_raw.csv"))
    if not found:
        raise SystemExit(
            f"No survey files in {PUBLIC_DIR.relative_to(REPO_ROOT)}/.\n"
            "Run: uv run python scripts/fetch_data.py"
        )
    if years is None:
        return found
    wanted = {f"so_{y}_raw.csv" for y in years}
    chosen = [p for p in found if p.name in wanted]
    missing = wanted - {p.name for p in chosen}
    if missing:
        raise SystemExit(f"Not fetched: {', '.join(sorted(missing))}")
    return chosen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", type=int, nargs="+", help="survey years (default: all present)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    paths = survey_files(args.years)
    print(f"Training on {len(paths)} survey year(s)\n")
    df, _ = load_years(paths, verbose=True)

    years = sorted(df["survey_year"].unique())
    span = f"{years[0]}-{years[-1]}" if len(years) > 1 else str(years[0])
    label = (
        f"Stack Overflow India survey {span} ({len(df):,} rows) — REAL market data"
        # Naming the span, not just the count, because "trained on 16,040 rows"
        # invites the reader to assume they are all current. Half of them are
        # from a market where the median salary was less than half today's.
    )

    print(f"\nFitting … (seed {args.seed})")
    bundle = service.train_bundle(df, seed=args.seed, label=label)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    service.save_bundle(bundle, MODEL_PATH)

    print(f"\nWrote {MODEL_PATH.relative_to(REPO_ROOT)}")
    print(f"  trained on          {bundle.trained_on}")
    print(f"  usable fields       {', '.join(bundle.usable_fields)}")
    print(f"  holdout coverage    {bundle.holdout_coverage:.1%} against an 80% promise")
    print(f"  holdout width       {bundle.holdout_relative_width:.0%} of midpoint")
    if bundle.holdout_relative_width > 0.5:
        print(
            "\n  NOT DECISION-GRADE. A band has to sit inside 50% of its midpoint\n"
            "  to be quotable. Serving this is fine; quoting it in an offer is not."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
