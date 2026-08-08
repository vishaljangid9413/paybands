"""Making the band's promise true — conformalised quantile regression.

`band.py` trains three models and calls the gap between the 0.1 and the 0.9 a
"80% band". Read that sentence again. It says 80% because 0.9 − 0.1 = 0.8, which
is arithmetic about the *loss functions we asked for*, not a measurement of what
the model does.

Measure it and you typically find something else. On this project's survey data
the raw quantile band covers 74.9%, not 80% (mean over 10 splits, range
70.1%-84.3%). Somewhere between "we asked
for the 90th percentile" and "the model found the 90th percentile" sits a
finite training set, a model with 7 leaves, and a test set the model has never
seen — and the promise leaks out through all three.

**A band that promises 80% and delivers 74.9% is lying to a recruiter.** They will
make offers at the edge of it and be wrong more often than they were told. The
fix is this file.

---

## The fix — CQR in four lines

Conformalized Quantile Regression (Romano, Patterson & Candès, 2019). Take a
**calibration set** the model has never seen. For each row in it, ask how badly
the band missed:

    score = max(lower − y,  y − upper)

Read that as: how far *outside* the band did the truth fall? If ``y`` sits below
the band, ``lower − y`` is how far below. If it sits above, ``y − upper`` is how
far above. If it sits comfortably inside, both terms are negative and the score
is negative — a *credit*, measuring how much room to spare there was.

Now take the ``⌈(n+1)(1−α)⌉``-th smallest of those scores; call it ``q̂``. Widen
every band by ``q̂`` on both sides. Done.

The result is a **finite-sample coverage guarantee**: the widened band covers at
least ``1 − α`` of new rows. Not asymptotically, not on average over many
datasets — for this dataset, at this size, right now. The only assumption is
**exchangeability**: the calibration rows and the future rows have to be drawn
from the same pool, in no particular order. (Which is exactly the assumption a
random split makes and a temporal split does not — see `split.temporal_split`.)

Three things worth noticing:

* **The odd-looking ``(n+1)``** is not a typo or a fudge factor. The guarantee is
  proved by treating the new row as one more member of the calibration set and
  asking where its score would rank among ``n + 1`` scores. The ``+1`` is that
  new row. It also means the guarantee is impossible with too few calibration
  rows — at α = 0.2 you need at least 4, and at that size the "guarantee" is
  covering 100% by making the band enormous.

* **``q̂`` can be negative**, and then CQR *narrows* the band. That is not a bug.
  If the quantile models were over-cautious — covering 92% when 80% was asked
  for — the honest correction is a tighter band, and the guarantee still holds.
  Conformal prediction is a calibration step in both directions.

* **We do all of this in log space.** ``q̂`` is a constant, and a constant in log
  space is a constant *percentage* in rupees: every band grows by, say, 9%. A
  constant in rupees would add the same ₹3L to the fresher's band and the CTO's,
  which is precisely the fixed-width mistake `band.py` was built to avoid.

---

## The part people get wrong: you need THREE sets, not two

This is the loudest warning in the file.

    ┌──────────────┬──────────────┬──────────────┐
    │    TRAIN     │ CALIBRATION  │     TEST     │
    │  fit trees   │  compute q̂  │  report      │
    └──────────────┴──────────────┴──────────────┘

**Calibrating on the training data destroys the guarantee.** The model has
already seen those rows; it fits them far better than it fits anything new. The
scores come out small, ``q̂`` comes out small, the band barely widens, and you
publish a coverage guarantee that is simply false. The failure is silent — the
code runs, the numbers look *better* than the honest ones, and the flaw only
shows up in production.

**Calibrating on the test data is leakage**, the same mistake in a different
coat. You would have tuned the band to the very rows you then use to claim it
works, and the coverage you report would be a fact about your test set rather
than a prediction about anyone's future.

So this module ships :func:`three_way_split`, which makes the correct cut in one
call, and :meth:`ConformalBand.calibrate`, which **refuses** to run if the
calibration rows overlap the rows the model was fitted on. It checks index
labels — which `split.random_split` preserves precisely so that a row's identity
survives being moved around. If you build the split some other way and lose the
index, the check cannot protect you, and the docstring says so rather than
pretending otherwise.

Cost of the third set: on 1,000 rows, roughly 200 rows the trees never get to
learn from. That is a real price. It buys a coverage number you can put in
writing, which for a tool that quotes salaries is worth more than the marginal
accuracy those rows would have bought.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from .band import BandPrediction, BandReport, band_report
from .metrics import from_log, to_log
from .split import DEFAULT_SEED, random_split

#: Default miss rate. α = 0.2 means "at least 80% coverage", which pairs with
#: the 0.1 / 0.9 quantile models in `band.py`.
DEFAULT_ALPHA = 0.2


# ─────────────────────────────────────────────── what we can wrap


@runtime_checkable
class QuantileModel(Protocol):
    """Anything with three quantile predictions in log space can be conformalised.

    Deliberately a `Protocol` — structural typing, "if it has this method it
    qualifies" — rather than a base class. CQR knows nothing about LightGBM. Any
    model that produces a low / middle / high triple in log space can be handed
    to :class:`ConformalBand`, including a linear quantile regression or a
    hand-written lookup table, which makes this module testable without training
    anything.
    """

    def predict_quantiles_log(self, X: pd.DataFrame) -> NDArray[np.float64]:
        """Sorted quantiles in log space, shape ``(n_rows, 3)``."""
        ...


# ─────────────────────────────────────────────── the three-way split


@dataclass(frozen=True)
class ThreeWaySplit:
    """Train / calibration / test, cut once and carried together.

    Handing the three frames around as one object is the point. Three loose
    DataFrames named ``train``, ``cal`` and ``test`` are three chances to pass
    the wrong one, and the mistake produces a *better* number, so nothing about
    the output tells you it happened.

    The constructor verifies the three index sets are pairwise disjoint. If that
    check ever fires, something upstream re-indexed the frames and the guarantee
    is void.
    """

    train: pd.DataFrame
    calibration: pd.DataFrame
    test: pd.DataFrame
    description: str

    def __post_init__(self) -> None:
        train, cal, test = set(self.train.index), set(self.calibration.index), set(self.test.index)
        for name, a, b in (
            ("train ∩ calibration", train, cal),
            ("train ∩ test", train, test),
            ("calibration ∩ test", cal, test),
        ):
            shared = a & b
            if shared:
                raise ValueError(
                    f"{name} share {len(shared)} rows — the conformal guarantee "
                    "requires the three sets to be disjoint"
                )

    @property
    def sizes(self) -> tuple[int, int, int]:
        return len(self.train), len(self.calibration), len(self.test)

    def __str__(self) -> str:
        n_train, n_cal, n_test = self.sizes
        total = n_train + n_cal + n_test
        return (
            f"three-way split — {n_train:,} train / {n_cal:,} calibration / {n_test:,} test "
            f"({total:,} rows)\n  {self.description}"
        )


def three_way_split(
    df: pd.DataFrame,
    *,
    calibration_size: float = 0.2,
    test_size: float = 0.2,
    seed: int = DEFAULT_SEED,
) -> ThreeWaySplit:
    """Cut a dataset into the three sets conformal prediction needs.

    Args:
        df: Any common-schema frame. The index is preserved, so every row can be
            traced back — and so the overlap checks have something to check.
        calibration_size: Share of the **whole** dataset used to compute ``q̂``.
        test_size: Share of the whole dataset held out for reporting.
        seed: Same seed, same three sets, forever.

    Both fractions are of the original total, not of what is left after the
    previous cut. Ask for 0.2 and 0.2 and you get 60/20/20, which is what
    everybody means and not what nested splitting naturally gives you.
    """
    if not 0 < calibration_size < 1 or not 0 < test_size < 1:
        raise ValueError(
            f"calibration_size and test_size must each be in (0, 1), "
            f"got {calibration_size} and {test_size}"
        )
    if calibration_size + test_size >= 1:
        raise ValueError(
            f"calibration_size + test_size = {calibration_size + test_size:.2f} leaves "
            "nothing to train on"
        )

    # Two independent child seeds derived from the one the caller gave us.
    # Reusing `seed` for both cuts would work by accident here (the frames have
    # different lengths, so the permutations differ), and "works by accident" is
    # how a reproducibility bug gets planted. SeedSequence is the supported way
    # to split one seed into several that are guaranteed not to correlate.
    outer, inner = (
        int(child.generate_state(1)[0]) for child in np.random.SeedSequence(seed).spawn(2)
    )

    first = random_split(df, test_size=test_size, seed=outer)
    # `calibration_size` was a share of the ORIGINAL frame; rescale it to a share
    # of what survived the first cut.
    inner_fraction = calibration_size / (1.0 - test_size)
    second = random_split(first.train, test_size=inner_fraction, seed=inner)

    return ThreeWaySplit(
        train=second.train,
        calibration=second.test,
        test=first.test,
        description=(
            f"seed={seed}: {1 - calibration_size - test_size:.0%} train / "
            f"{calibration_size:.0%} calibration / {test_size:.0%} test; "
            "calibration is disjoint from training, which is what makes the "
            "coverage guarantee real"
        ),
    )


# ─────────────────────────────────────────────── the calibration wrapper


class ConformalBand:
    """Wraps a quantile model and makes its coverage promise true.

        split = three_way_split(df, seed=42)
        model = SalaryBandModel(seed=42).fit(split.train, split.train["salary_annual"])
        band = ConformalBand(model).calibrate(
            split.calibration, split.calibration["salary_annual"]
        )
        print(band.evaluate(split.test, split.test["salary_annual"]))

    Fitted attributes:

    * ``qhat_log_`` — the widening, in log points. Multiply out with ``exp`` to
      read it as a percentage; ``widening_factor`` does that for you.
    * ``scores_`` — the calibration conformity scores, sorted. Worth plotting:
      the shape tells you whether the raw band was uniformly too narrow or fine
      for most people and hopeless for a few.
    * ``n_calibration_`` — how many rows produced ``q̂``. Small ``n`` means a
      noisy ``q̂``; the guarantee still holds, but the band will be wider than it
      needs to be.
    """

    def __init__(self, model: QuantileModel, *, alpha: float | None = None) -> None:
        """
        Args:
            model: Anything satisfying :class:`QuantileModel`.
            alpha: Allowed miss rate. ``None`` (the default) reads it off the
                model's own ``nominal_coverage`` — so a model built on the 0.1
                and 0.9 quantiles is calibrated to 80%, and the two numbers
                cannot drift apart because somebody edited one of them.
        """
        if alpha is None:
            nominal = getattr(model, "nominal_coverage", None)
            alpha = DEFAULT_ALPHA if nominal is None else 1.0 - float(nominal)
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")

        self.model = model
        self.alpha = alpha

        self.qhat_log_: float | None = None
        self.scores_: NDArray[np.float64] = np.empty(0)
        self.n_calibration_: int = 0

    # ── properties ─────────────────────────────────────────────────────

    @property
    def target_coverage(self) -> float:
        """The promise: ``1 − α``."""
        return 1.0 - self.alpha

    @property
    def is_calibrated(self) -> bool:
        return self.qhat_log_ is not None

    @property
    def widening_factor(self) -> float:
        """``q̂`` read as a rupee multiplier — 1.09 means "each edge moved 9%"."""
        if self.qhat_log_ is None:
            raise RuntimeError("call calibrate() before reading the widening")
        return float(np.exp(self.qhat_log_))

    # ── calibrate ──────────────────────────────────────────────────────

    def calibrate(self, X_cal: pd.DataFrame, y_cal: ArrayLike) -> ConformalBand:
        """Compute ``q̂`` from rows the model has never seen.

        Args:
            X_cal: Calibration rows, in the common schema.
            y_cal: Their true salaries **in rupees**.

        Raises:
            ValueError: if the calibration rows overlap the model's training
                rows, or if there are too few of them for the guarantee to mean
                anything.

        **On the overlap check.** It compares index labels against the model's
        ``train_index_``, which `SalaryBandModel` records at fit time and
        `split.random_split` preserves. It is a genuine guard, not decoration —
        calibrating on training rows is the single mistake that turns this
        module from a guarantee into a decoration, and it leaves no trace in the
        output. If you build your split some other way and reset the index, this
        check goes quiet: it cannot see rows it cannot identify. Use
        :func:`three_way_split`.
        """
        salaries = np.asarray(y_cal, dtype=float)
        if salaries.size != len(X_cal):
            raise ValueError(f"X_cal has {len(X_cal)} rows but y_cal has {salaries.size}")

        n = len(X_cal)
        # ⌈(n+1)(1−α)⌉ must be a rank that exists among n scores. Below that,
        # the honest q̂ is infinity: with 3 calibration rows you cannot certify
        # 80% coverage, and pretending otherwise is worse than refusing.
        minimum = math.ceil(1.0 / self.alpha) - 1
        if n < minimum:
            raise ValueError(
                f"{n} calibration rows cannot certify {self.target_coverage:.0%} coverage; "
                f"need at least {minimum}. (⌈(n+1)(1−α)⌉ has to be a rank that exists.)"
            )

        train_index = getattr(self.model, "train_index_", None)
        # `is not None`, not a truthiness test. An empty frozenset is falsy, so
        # `if train_index:` would silently SKIP the leakage check for a model
        # trained on zero rows — exactly the degenerate case where you most want
        # a loud failure. A guard that disables itself in the edge case is worse
        # than no guard, because you believe you are protected.
        if train_index is not None:
            shared = train_index & frozenset(X_cal.index)
            if shared:
                raise ValueError(
                    f"{len(shared)} calibration rows were also used to train the model. "
                    "The model already fits those rows better than it fits new ones, so "
                    "the conformity scores would be too small and the coverage guarantee "
                    "would be false. Use three_way_split()."
                )

        q = self.model.predict_quantiles_log(X_cal)
        y_log = to_log(salaries)
        lower, upper = q[:, 0], q[:, 2]

        # How far outside the band the truth fell. Negative = comfortably inside,
        # and those credits are what let q̂ come out negative and TIGHTEN a band
        # that was too cautious.
        scores = np.maximum(lower - y_log, y_log - upper)
        self.scores_ = np.sort(scores)

        rank = math.ceil((n + 1) * self.target_coverage)
        # Guarded above, but an off-by-one here would silently degrade the
        # guarantee rather than crash, so it is worth being explicit.
        rank = min(rank, n)
        self.qhat_log_ = float(self.scores_[rank - 1])
        self.n_calibration_ = n
        return self

    # ── predict ────────────────────────────────────────────────────────

    def predict_quantiles_log(self, X: pd.DataFrame) -> NDArray[np.float64]:
        """The widened quantiles in log space — so a `ConformalBand` is itself a
        :class:`QuantileModel` and could, in principle, be wrapped again."""
        if self.qhat_log_ is None:
            raise RuntimeError("call calibrate() before predicting")
        q = self.model.predict_quantiles_log(X).copy()
        q[:, 0] -= self.qhat_log_
        q[:, 2] += self.qhat_log_
        # A negative q̂ narrows the band, and if it narrowed past the median the
        # band would invert. Clip rather than let that happen: the median is a
        # prediction, not an edge, and a band that excludes its own midpoint is
        # nonsense to show anybody.
        q[:, 0] = np.minimum(q[:, 0], q[:, 1])
        q[:, 2] = np.maximum(q[:, 2], q[:, 1])
        return q

    def predict_band(self, X: pd.DataFrame) -> BandPrediction:
        """The calibrated band, in rupees.

        The midpoint is untouched. CQR adjusts the *edges* — it says nothing
        about whether the median model is any good, which is what MAE is for.
        """
        q = self.predict_quantiles_log(X)
        return BandPrediction(
            lower=from_log(q[:, 0]),
            median=from_log(q[:, 1]),
            upper=from_log(q[:, 2]),
            # The nominal coverage is now α-driven rather than quantile-driven:
            # this band promises 1 − α, and that promise is the calibrated one.
            quantiles=(self.alpha / 2, 0.5, 1.0 - self.alpha / 2),
            n_crossed=getattr(self.model, "n_crossed_", 0),
        )

    # ── reporting ──────────────────────────────────────────────────────

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: ArrayLike,
        *,
        label: str = "conformal band",
    ) -> BandReport:
        """Coverage and width on the test set — the honest numbers.

        Report both, always. Narrower is better **only if coverage holds**: a
        band that shrank and lost coverage did not improve, it started lying
        more confidently.
        """
        return band_report(y_test, self.predict_band(X_test), label=label)

    def describe(self) -> str:
        if self.qhat_log_ is None:
            return "ConformalBand(uncalibrated)"
        direction = "widened" if self.qhat_log_ >= 0 else "NARROWED (the raw band was too cautious)"
        return "\n".join(
            [
                f"ConformalBand — target coverage {self.target_coverage:.0%} (α = {self.alpha:g})",
                f"  calibrated on      {self.n_calibration_:,} rows the model never saw",
                f"  q̂ (log points)     {self.qhat_log_:>+8.4f}",
                f"  each edge {direction} by ×{self.widening_factor:.3f} "
                f"({abs(self.widening_factor - 1):.1%})",
                f"  scores ran         [{self.scores_[0]:+.3f}, {self.scores_[-1]:+.3f}] "
                "log points (negative = the truth sat inside, with room to spare)",
            ]
        )
