# 02 — Features and the model

This is the second part of the code reading guide. It covers ten files, in the order
they are meant to be read:

| # | file | what it is |
|---|---|---|
| 1 | `src/paybands/features/experience.py` | years of experience → several useful shapes |
| 2 | `src/paybands/features/skills.py` | a semicolon-separated string → binary flags |
| 3 | `src/paybands/features/location.py` | Indian city names → tiers 1/2/3 |
| 4 | `src/paybands/features/builder.py` | all three, composed into one table |
| 5 | `src/paybands/model/split.py` | cutting the data into train and test |
| 6 | `src/paybands/model/metrics.py` | how a prediction is scored |
| 7 | `src/paybands/model/baseline.py` | the two deliberately stupid models |
| 8 | `src/paybands/model/band.py` | **the core model** — three quantile models that draw a band |
| 9 | `src/paybands/model/conformal.py` | making the band's confidence promise true |
| 10 | `tests/test_features.py` | including the leakage test and its deliberate contrast |

**Read `01-payroll-and-data.md` first.** It covers the payroll calculator, the common
schema (in particular the column names used everywhere below, and `TARGET`, which is the
string `"salary_annual"`), and where the 1,022 rows of Stack Overflow survey data come
from. This document assumes you know what a row looks like.

Two terms used constantly from here on:

- A **feature** is one input column the model reads. Turning raw data into good features
  is *feature engineering*.
- **Fit** means "learn something from data and remember it"; **transform** means "apply
  what you remembered". Every transformer in this project has both methods, even the ones
  where `fit` learns nothing at all. That symmetry is deliberate and is explained below.

You do not need to have used scikit-learn. Where a scikit-learn convention shows up — most
often the trailing underscore on attribute names, as in `vocabulary_` — it is explained
where it first appears.

---

### `src/paybands/features/experience.py`

> Turns one `years_experience` number into four columns, because rupees do not rise in a
> straight line with years.

**Read time:** 10 minutes · **Difficulty:** easy
**Read it when:** you know what a pandas `Series` is. Nothing else is assumed.

#### What problem it solves

Experience is the strongest signal in this dataset — `docs/findings.md` §4.1 measures it
at 53.7% of the model's gain, and roughly four fifths of its held-out importance. But the
relationship between years and pay bends sharply. These are the real medians from the
survey, and they are printed at the top of the file:

```
   0–1 yrs   ₹5.4L        ┐
   2–3 yrs   ₹7.0L        │  steep: about ₹1.6L per extra year
   4–5 yrs   ₹12.0L       │
   6–8 yrs   ₹22.0L       ┘
   9–12 yrs  ₹30.0L       ┐
  13–20 yrs  ₹40.0L       │  flattening: about ₹1.0L per year, then about ₹0
    20+ yrs  ₹40.0L       ┘
```

Between year 3 and year 7 pay roughly triples. Between year 15 and year 25 it barely
moves at all. **The same "+1 year" is worth wildly different money depending on where you
already are.**

Why that matters: a model fitting a single number of rupees-per-year has to compromise
between the steep start and the flat end. Say it settles on ₹1.7L per year. It then
predicts about ₹5L for a fresher (roughly right, by luck), about ₹22L at year 10 (too
low), and about ₹42L at year 25 (too high) — and it keeps climbing forever, extrapolating
a 40-year veteran to ₹68L, which is not how the market works.

Gradient-boosted trees (the algorithm used in `band.py`) are not linear, so they *can*
learn a curve. But they learn it by spending splits on discovering it, and only where they
have enough rows. Above 15 years the survey is thin. So this file hands the model the
shape instead of making it rediscover it from 1,022 rows.

#### Module constants

| name | value | meaning |
|---|---|---|
| `MAX_PLAUSIBLE_YEARS` | `50.0` | anything above this is a data-entry error, not a career |
| `BUCKET_EDGES` | `(0.0, 2.0, 4.0, 6.0, 9.0, 13.0, 21.0, inf)` | cut points, left-closed |
| `BUCKET_LABELS` | `("0-1", "2-3", "4-5", "6-8", "9-12", "13-20", "20+")` | the seven band names |
| `BUCKET_DTYPE` | `pd.CategoricalDtype(categories=BUCKET_LABELS, ordered=True)` | the declared category type |

The edges are cut where the *slope* changes, not at round numbers. `BUCKET_DTYPE` is
declared up front rather than inferred, so a test batch containing three juniors still
produces all seven categories — the columns then line up with what the model was trained
on. If pandas were left to infer the categories from whatever happened to be in the frame,
a small batch would produce a narrower column and the model would be reading the wrong
thing.

#### Functions

```python
def clean_years(values: pd.Series | list[float]) -> pd.Series
```

Coerces to numbers, then blanks out anything impossible. Non-numeric text, negatives, and
anything above `MAX_PLAUSIBLE_YEARS` all become `NaN` ("not a number", pandas' marker for
a missing value).

Why `NaN` and not `0`: **a zero is a claim.** Writing `0` says "this person is a fresher".
If someone typed `-3` or `2005`, we have no idea what they meant, and answering "fresher"
on their behalf inserts a confidently wrong row into training. `NaN` says "unknown", and
LightGBM knows what to do with unknown — at every split it tries sending the missing rows
left and right and keeps whichever fits better. This is the same argument that recurs in
`location.py` and `builder.py`, and it is one of the two or three most important ideas in
the whole feature layer.

```python
def log_years(years: pd.Series) -> pd.Series
```

Returns `log1p(years)`, which is `log(1 + years)`. Plain `log` is used everywhere in
statistics to turn multiplicative relationships into additive ones — but `log(0)` is
negative infinity, and freshers have 0 years. `log1p(0)` is exactly `0`. `NaN` in, `NaN`
out.

On a log scale the early steep years get stretched and the late flat years get squashed,
which is precisely the shape in the table above. One linear slope on `log1p` means "each
extra year is worth a *percentage*", which is how raises actually work — nobody describes
their year as "a ₹2.4 lakh raise", they say "a 30% raise".

```python
def sqrt_years(years: pd.Series) -> pd.Series
```

A gentler curve than the log: still concave (rising but flattening), just less
aggressively. Which of log or square root fits the Indian market better is an empirical
question, so the project ships both and lets feature importance settle it. (It did:
`experience_log` earns 12.5% of the model's gain, `experience_sqrt` 3.9%.)

```python
def bucket_years(years: pd.Series) -> pd.Series
```

Cuts years into the seven bands as an *ordered category*. Uses `pd.cut(..., right=False)`,
which makes the bands left-closed: exactly 2.0 years lands in `"2-3"`, not `"0-1"`, which
is the boundary behaviour the labels imply. The result is then cast to `BUCKET_DTYPE`, so
all seven categories are present whatever the input contained.

A bucket lets a tree isolate "6–8 years" in a single split, rather than having to find two
separate thresholds.

#### Classes

**`ExperienceFeatures`** — the transformer wrapper. It represents "the experience block of
the feature matrix" and exists so that `builder.py` can treat experience, skills and
location identically.

```python
ExperienceFeatures(column: str = "years_experience", *, include_sqrt: bool = True)
```

| field | meaning |
|---|---|
| `column` | which input column to read |
| `include_sqrt` | whether to emit `experience_sqrt` |
| `feature_names_` | the exact output columns, in order — set by `fit` |

The trailing underscore on `feature_names_` is the scikit-learn convention for "this was
set by fitting, not by you". In this project it doubles as a warning label: **anything
with a trailing underscore is a thing that could, in principle, leak.**

- `fit(df: pd.DataFrame) -> ExperienceFeatures` — **learns nothing.** `df` is deliberately
  unused. All it does is populate `feature_names_` with
  `["years_experience", "experience_log", "experience_sqrt", "experience_bucket"]`
  (the `experience_sqrt` entry is inserted at position 2, and omitted when
  `include_sqrt=False`). It returns `self` so calls can be chained.
- `transform(df: pd.DataFrame) -> pd.DataFrame` — returns those four columns, indexed like
  the input. Raises `RuntimeError("call fit() before transform()")` if `fit` was never
  called. If `df` genuinely lacks the column, it emits an all-`NaN` column rather than
  raising, because the *shape* the model expects must not depend on which data source the
  rows came from.

#### The one thing to understand here

`fit` doing nothing is the point of the file's design, not an oversight.

`log1p(7)` is `log1p(7)` whether the 7 came from a training row or a test row. Nothing
about the training data is baked into the transform, so **nothing can leak**. Compare
`skills.py`, where `fit` genuinely learns a vocabulary from the training rows and therefore
must never be shown test rows. Keeping the same `fit`/`transform` interface everywhere
makes that difference the thing you notice, instead of hiding it. The package's
`features/__init__.py` prints the same distinction as a table.

#### Surprises and gotchas

- **The raw `years_experience` column is kept too**, alongside all three derived shapes.
  The file's reasoning: trees are happy to ignore what they do not need, and dropping a
  column to prove a point is a worse habit than keeping a redundant one. It was the right
  call — `years_experience` turned out to be the single most important feature in the
  model, at 36.4% of gain.
- **`transform` does not call `log_years` or `sqrt_years`.** It calls `np.log1p` and
  `np.sqrt` directly on the already-cleaned Series. The results are identical (the helpers
  are just `clean_years` plus the same numpy call), but if you edit one you must remember
  to edit the other. The module-level functions exist mainly so tests can exercise them
  directly, which `tests/test_features.py` does.
- **A 120-year career becomes `NaN`, not 50.** Clipping to a maximum would invent a
  confident claim out of a typo. There is a parametrised test for exactly this
  (`test_absurd_experience_becomes_nan_not_clipped`).
- **These bucket edges are not the only bucket edges in the project.** `baseline.py`
  defines its own `EXPERIENCE_BUCKETS` with the same seven labels but upper-bound-inclusive
  boundaries. For whole numbers of years the two agree exactly; for fractional years they
  do not (1.5 years is `"0-1"` here and `"2-3"` there). The survey's `WorkExp` column is
  whole years, so this never bites in practice — but it is a duplication that could.

---

### `src/paybands/features/skills.py`

> Turns `"Python;SQL;JavaScript"` into binary columns — and is the clearest example of
> leakage in the project.

**Read time:** 12 minutes · **Difficulty:** medium (the concept, not the code)
**Read it when:** you have read `experience.py` and understand why `fit` there is empty.

#### What problem it solves

The survey stores skills as one semicolon-separated string per person. A model cannot read
that. The standard fix is *binary flags*: one column per skill, `1` if the person listed
it, `0` if not.

But there are hundreds of distinct skills and most appear a handful of times. A column that
is `1` for four people out of a thousand teaches a tree nothing. So this file keeps only
the **top N most common** skills — 25 by default — and ignores the long tail.

And that is where the interesting problem is, because *"which skills are most common?"* is
a fact learned from data.

#### Leakage — the whole reason this file has a long docstring

**Leakage means information from your test data reaching the model during training.**

Why it is the most expensive mistake in applied machine learning: it does not look like a
bug. Your validation score goes *up*. You feel clever. You ship the model confidently, and
it fails quietly in production — where the future genuinely is unknown — and by then your
own measurements are telling you everything is fine, so you have no idea where to look.

Here is the concrete version. Suppose you compute the top-25 skills over the *whole*
dataset, and only afterwards split into train and test. If `rust` clears the cut-off only
because of six Rust developers who all happen to land in the **test** set, then the model's
input columns were chosen with knowledge of the test rows. Your test score is no longer an
estimate of performance on unseen data, because those rows already shaped the model's
input space.

The fix is one line of discipline:

- `fit(train)` counts skills over the training rows only and stores `vocabulary_`.
- `transform(anything)` only ever emits columns from that stored vocabulary.

The size of the cheat in *this* case is genuinely small — a couple of columns either way.
**The habit is what matters.** The same mistake made with target encoding, feature scaling,
or median imputation is catastrophic, and it is the identical mistake. This file is where
the discipline is built while the stakes are low.

#### Why an unseen skill is ignored, and that is correct

A test-set person who lists `zig` gets no `skill_zig` column. This is not a bug. Three
reasons, in ascending order of importance:

1. The model was never trained on such a column, so there is no learned weight to apply
   to it. There is nothing honest to do with it.
2. Silently adding a column at prediction time changes the input shape, which either
   crashes the model or — worse, with some libraries — shifts every subsequent column by
   one position and produces confidently wrong answers with no error.
3. In production this happens constantly. New frameworks appear every year. The honest
   answer is "this model does not know about Zig yet; retrain it", and retraining is a
   decision a human makes, not something a `transform` call should quietly do on your
   behalf.

The unseen skill still leaves one trace: `n_skills` counts *everything* the person listed,
in-vocabulary or not. That count is computed row by row with no fitted state, so it cannot
leak.

#### Module constants

- `DEFAULT_TOP_N = 25` — a judgement call, and the file says so. The rationale: at rank 25
  a language still appears in a few hundred CVs, which is enough rows for a tree to split
  on; below that you are mostly modelling noise.
- `_UNSAFE_CHARS = set('[]{}":, ')` — characters LightGBM rejects in feature names, because
  it serialises models to JSON. Note what is deliberately *not* in the set: `+`, `#`, `/`
  and `.` are all legal, so `c++` stays recognisably `skill_c++` rather than becoming an
  unreadable `skill_c__`.

#### Functions

```python
def parse_skills(value: object) -> list[str]
```

Splits one survey cell into a clean list of lowercase names. Handles the three things real
data does: missing values (which arrive as `NaN`, not a string, hence the
`isinstance(value, str)` check), empty strings, and stray whitespace around the semicolons.
Lowercasing means `"Python"` and `"python"` are one skill, which they obviously are but
which a naive `str.split` would treat as two.

```python
def _safe_name(skill: str) -> str
```

Private helper. Prefixes `skill_` and replaces every character in `_UNSAFE_CHARS` with an
underscore.

#### Classes

**`SkillFeatures`** — learns a vocabulary from training rows, then emits flags.

```python
SkillFeatures(column: str = "skills", *, top_n: int = DEFAULT_TOP_N)
```

| field | meaning |
|---|---|
| `column` | which input column holds the semicolon-separated string |
| `top_n` | how many skills to keep |
| `vocabulary_` | **learned** — the chosen skills, most common first |
| `skill_counts_` | **learned** — how many training rows listed each of them |
| `feature_names_` | **learned** — the columns `transform` will produce |

`skill_counts_` is kept purely for inspection. When a feature turns out to be useless, the
first question is always "how many rows even had it?" — see the caveat about `skill_php` in
`docs/findings.md` §4.1, which is exactly that question.

- `_column_or_blank(df) -> pd.Series` — private. Returns the column if present, otherwise
  an all-`None` object Series of the right length. This is what lets a source with no
  skills column still produce the right-shaped output.

- `fit(df: pd.DataFrame) -> SkillFeatures` — counts skills across **these rows only**.
  Note `counts.update(set(parse_skills(cell)))`: the `set()` means one vote per person, so
  someone who somehow lists Python twice does not count twice. Ranking is
  `sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:top_n]` — by count descending,
  then alphabetically. **The alphabetical tiebreak matters**: without it, two skills with
  identical counts could swap places between runs and silently reorder the model's columns.

- `transform(df: pd.DataFrame) -> pd.DataFrame` — one `int8` column per vocabulary skill
  plus an `int16` `n_skills`. `int8` rather than `bool` because LightGBM wants numbers, and
  one byte per cell keeps 25 skill columns cheap on a laptop. Raises `RuntimeError` if
  `fit` has not been called.

A worked example, using the training fixture from `tests/test_features.py` (six people,
whose skill strings contain Python four times, SQL four times, JavaScript twice, Java
twice, Go once). With `top_n=3`:

```
vocabulary_    = ["python", "sql", "java"]
skill_counts_  = {"python": 4, "sql": 4, "java": 2}
feature_names_ = ["skill_python", "skill_sql", "skill_java", "n_skills"]
```

`python` beats `sql` on the alphabetical tiebreak at four each; `java` beats `javascript`
on the same tiebreak at two each. That is deterministic, and there is a test asserting
exactly this list.

#### The one thing to understand here

`fit` and `transform` are two different security levels. `fit` must see training rows and
nothing else, ever. `transform` may be called on anything. Once you internalise that split,
the whole feature layer reads as one idea repeated four times.

#### Surprises and gotchas

- **`transform` returns `out[self.feature_names_]`**, which reorders the frame to the fitted
  order. Column order matters to LightGBM, which identifies features by position as well as
  by name.
- **A skill in the vocabulary that nobody in the current batch has still gets a column**, of
  all zeros. That is right: the shape must not depend on the batch.
- **`n_skills` is a rough proxy for breadth, and it is not very good.** `docs/findings.md`
  §4.1 uses it as its worked example of a misleading importance measure: it reaches 12.3%
  of LightGBM's *split count* — second place overall — because it is an integer running
  0–20 and a tree can split it at many thresholds. Shuffle it on held-out rows and the
  model loses ₹3,318, about a quarter of one percent of its error. Split-count importance
  measures how many questions a feature *could* answer, not how useful the answers were.

---

### `src/paybands/features/location.py`

> Maps free-text Indian city names to tier 1, 2 or 3 — and is written for data the project
> does not have yet.

**Read time:** 8 minutes · **Difficulty:** easy
**Read it when:** any time. It is self-contained.

#### What problem it solves

Raw city is a *high-cardinality* column: hundreds of distinct values, most appearing once
or twice. A model given "Bhilai" with three rows learns noise, and at prediction time it
meets cities it has never seen at all.

What actually drives Indian tech pay is not the city's identity but its **tier** — the size
of its tech market and its cost of living. Bangalore, Hyderabad and Pune behave alike;
Indore and Coimbatore behave alike. A tier is three values instead of three hundred.

**Important: the Stack Overflow survey has no city column.** It records country, and the
loader already filters to India. So on today's data this module produces a column of
nothing but `NaN`. It is written and tested now because of the pipeline promise in
`docs/design.md`: all three data sources produce the same table shape, so when company data
finally arrives with a real `city` column, the feature that consumes it already exists and
already has tests. The alternative is writing brand-new untested code on the same day you
first touch real salary data, which is the worst possible day to be debugging.

#### Module constants

| name | contents |
|---|---|
| `TIER_1_CITIES` | 8 metros: bangalore, mumbai, delhi, gurgaon, noida, hyderabad, pune, chennai |
| `TIER_2_CITIES` | 25 cities with real tech presence but lower pay and cost of living |
| `DEFAULT_TIER` | `3` — everything else |
| `CITY_ALIASES` | ~40 spelling variants, renamings, colloquialisms and airport codes |
| `_NON_LETTERS` | `re.compile(r"[^a-z ]+")` — strips punctuation and digits after lowercasing |

`CITY_ALIASES` is the whole reason the module exists. A real HR export contains
`"Bengaluru"`, `"BLR"`, `"bangalore "` and `"Bangalore, KA"` in the same column, and every
one of them is the same city. Left unhandled, three of those four silently become tier 3
and **the model learns that Bangalore is cheap**. The map also collapses the Delhi-NCR
municipalities (Faridabad and Ghaziabad map to `delhi`, Gurugram to `gurgaon`) because they
are one labour market.

#### Functions

```python
def normalise_city(value: object) -> str | None
```

Reduces free text to a canonical lowercase name. In order: reject non-strings; keep the
part before the first comma (`"Gurugram, Haryana"` → `"Gurugram"`); lowercase; replace
every non-letter with a space; collapse runs of whitespace; return `None` if nothing is
left; finally look the result up in `CITY_ALIASES`, falling back to the cleaned string
itself.

```
normalise_city("  Bengaluru ")      -> "bangalore"
normalise_city("Delhi-NCR")         -> "delhi"
normalise_city("Gurugram, Haryana") -> "gurgaon"
normalise_city("Pune - 411001")     -> "pune"
```

```python
def city_tier(value: object) -> float
```

Returns `1.0`, `2.0`, `3.0`, or `float("nan")`. The return type is float rather than int
precisely so that `NaN` is representable.

**The distinction between "missing" and "tier 3" is the point of the function:**

- `"Bhilai"` → **3**. We recognise it as a real place that is not a tech metro. That is a
  genuine claim about the city.
- `None`, `""`, `NaN`, `"   "` → **`NaN`**. We were not told where this person works.
  Calling that tier 3 would invent a fact, and specifically it would tell the model "this
  person is in a low-paying city", which drags the predicted band *down* for everyone whose
  record merely happened to be incomplete.

That second bullet is the same argument as `clean_years`, and it is worth noticing that it
has a *direction*: filling missing values with the pessimistic default is not a rounding
error, it systematically underpays people with incomplete records.

#### Classes

**`LocationFeatures`** — derives one `location_tier` column.

```python
LocationFeatures(city_column: str = "city", tier_column: str = "location_tier")
```

| field | meaning |
|---|---|
| `city_column` | input column of free-text city names |
| `tier_column` | output column name, also used as an alternative input |
| `feature_names_` | set by `fit` to `[tier_column]` |

- `fit(df) -> LocationFeatures` — a no-op on purpose. It even writes `del df` with a comment
  saying nothing is learned and the argument exists for interface symmetry.
- `transform(df) -> pd.DataFrame` — three branches, in this priority order:
  1. `city_column` present → map every value through `city_tier`.
  2. otherwise `tier_column` present → use it as-is via `pd.to_numeric(..., errors="coerce")`.
     A company export or the synthetic generator may already store tiers directly.
  3. neither present → an all-`NaN` column. **This is the Stack Overflow case.** The
     matrix keeps the same width for every source, so a model trained on synthetic data can
     still score survey rows.

  The result is cast to `float64` unconditionally.

#### The one thing to understand here

Nothing in this file is learned from data, so nothing in it can leak. The tier lists are
hand-written domain knowledge. They do not change when the training split changes.

The file is honest about the trade this makes: the lists encode *2026 opinions*. If
Jaipur's tech market grows, this file needs a human to edit it — it will not update itself.
Hand-written rules are auditable and stable, and they go stale silently. Knowing at a glance
which of your transformers learn and which do not is exactly the audit you want to be able
to run, and `features/__init__.py` prints that table.

#### Surprises and gotchas

- **The city column wins over an existing tier column.** If a frame somehow has both, the
  city is re-derived and any supplied `location_tier` is ignored. That is probably the right
  precedence, but it is silent.
- **One alias is a dead end.** `"pondicherry"` maps to `"puducherry"`, which appears in
  neither tier set, so both spellings end up at tier 3. Harmless — Puducherry would be tier
  3 anyway — but the alias buys nothing.
- **Tier 3 is not a value judgement.** The comment on `DEFAULT_TIER` says so explicitly: it
  means "we have no reason to think its tech salaries track the metros", not "this is a bad
  place".
- **On today's data this file contributes a column of pure `NaN`**, and
  `tests/test_features.py::test_round_trip_on_the_real_survey` asserts exactly that, stating
  it as a fact about the source rather than a defect.

---

### `src/paybands/features/builder.py`

> Composes the three feature blocks into one model-ready table, and makes four decisions
> worth arguing about.

**Read time:** 15 minutes · **Difficulty:** medium
**Read it when:** you have read all three of `experience.py`, `skills.py` and `location.py`.

#### What problem it solves

Something has to assemble the blocks in a fixed order, add the straightforward categorical
and numeric columns, guarantee that the training matrix and the test matrix have exactly
the same columns in exactly the same order, and refuse to let certain columns through at
all. That is this file.

Its four documented decisions are the substance. They are reproduced here because each one
is a general lesson, not a detail of this project.

#### Decision 1 — categories stay `category`; no one-hot encoding

The reflex from tutorials is `pd.get_dummies`. **One-hot encoding** turns a column with K
distinct values into K binary columns. For `education` with about eight values that is
fine. For job titles, org sizes, or (when company data lands) internal grades running to
dozens or hundreds, it produces a wide, mostly-zero matrix, and that hurts *trees*
specifically:

- **Each split asks one yes/no question.** With one-hot, a tree isolating a group inside a
  40-value column needs up to 39 separate splits. With a native categorical column,
  LightGBM can split "these 6 values left, the rest right" **in a single split**, because
  it sorts the categories by their gradient statistics and finds the best partition
  directly.
- **Rare dummies get starved.** A column that is `1` for 20 rows out of 10,000 almost never
  wins a split, so its information is effectively thrown away while still costing memory
  and training time.

So the columns are left as pandas `category` dtype, and `band.py` passes
`categorical_feature=` to LightGBM at fit time. One-hot is the right answer for linear
models and neural networks, neither of which this project uses.

#### Decision 2 — categories are frozen at `fit` time; unseen values become `NaN`

`fit` records the exact category list from the *training* rows. `transform` casts to that
same `CategoricalDtype`. A value that appears only in test — a new role, a job family
nobody had before — becomes `NaN`.

That is the same argument as unseen skills. But there is a second, sharper reason: freezing
the categories also freezes their **integer codes**. Internally a categorical column is
stored as integers pointing into the category list. If those codes were re-derived per
frame, `"Backend"` could be code 2 during training and code 5 at prediction time, and the
model would confidently answer a different question than the one you asked. **That failure
is silent** — no exception, just quietly wrong salaries.

#### Decision 3 — missing values are never filled with 0

LightGBM handles `NaN` natively: at each split it tries sending the missing rows left and
right and keeps whichever reduces the loss. It *learns what missingness means* rather than
being told.

`fillna(0)` is not neutral. **Zero is a claim.** Zero years of experience means fresher.
Zero previous salary means unemployed. A missing value is the *absence* of a claim.
Conflating the two puts confident, wrong rows into training — and because it always pushes
numbers downward, it systematically drags predicted bands down for people whose records
happened to be incomplete. That is not a rounding error; it is a fairness problem with a
specific direction.

The one thing the builder does fill is column *presence*: if a source lacks a column
entirely it emits an all-`NaN` column so the matrix is the same width everywhere. That is
shape, not content.

#### Decision 4 — what is deliberately left out

`EXCLUDED_COLUMNS` is `frozenset({TARGET, "source", "gender", "age_band", "event_date"})`,
and `include_prev_salary` defaults to `False`.

- **`prev_salary`** — off by default. See the previous-salary trap in `docs/design.md`
  §4.1: anchoring an offer to last drawn salary makes an early underpayment follow someone
  for their whole career, and a model fed that column learns to do it automatically, at
  scale. The flag exists so the planned experiment (train both versions, compare) is a
  one-line change rather than a rewrite.
- **`gender`, `age_band`** — never features. They are carried through the schema for the
  fairness *audit*, which measures whether predictions differ across groups. Feeding them
  in would be both a legal problem and a circular one. Note that dropping them is
  *necessary but not sufficient*: `docs/findings.md` §5.3 shows 38.4% of an injected gap
  surviving deletion of the gender column, because proxies survive.
- **`source`** — which loader produced the row. Pure dataset identity, not a property of a
  person. Left in, the model would learn "synthetic rows pay ₹X", which tells you nothing
  about a real candidate.
- **`salary_annual`** (the target) — obvious, but the assertion that it never reaches the
  feature matrix is a real test, because "the target leaked into the features" is a mistake
  people genuinely ship, and it produces a suspiciously perfect model.

#### Module constants

```python
CATEGORICAL_COLUMNS = ("role", "education", "org_size", "remote",
                       "employment_type", "level", "institute_tier", "prev_company_type")
PASSTHROUGH_NUMERIC = ("performance_rating",)
EXCLUDED_COLUMNS    = frozenset({TARGET, "source", "gender", "age_band", "event_date"})
```

`PASSTHROUGH_NUMERIC` is deliberately short: most numeric signal in this schema arrives
through the experience block. The last three `CATEGORICAL_COLUMNS` — `level`,
`institute_tier` and `prev_company_type` — plus `performance_rating` are company-data
columns the survey does not have at all (`docs/findings.md` §1 lists them among the seven
missing drivers). They are named here so they are picked up automatically the day they
appear, without a code change.

#### Classes

**`FeatureBuilder`**

```python
FeatureBuilder(*, top_skills: int = DEFAULT_TOP_N,
                  city_column: str = "city",
                  include_sqrt_experience: bool = True,
                  include_prev_salary: bool = False)
```

| field | meaning |
|---|---|
| `include_prev_salary` | whether `prev_salary` may become a feature |
| `experience` / `skills` / `location` | the three sub-transformers, kept public so you can inspect e.g. `builder.skills.vocabulary_` after fitting |
| `feature_names_` | **learned** — the exact columns, in order, that `transform` produces |
| `categorical_features_` | **learned** — the subset LightGBM should treat as categorical |
| `category_levels_` | **learned** — `{column: sorted list of training values}` |
| `numeric_features_` | **learned** — passthrough numeric columns actually present |

Methods:

- `fit(df) -> FeatureBuilder` — fits the three sub-transformers, then builds
  `category_levels_` from `sorted({str(v) for v in df[col].dropna().unique()})` for each
  categorical column present. **The sort matters**: an unsorted `unique()` depends on row
  order, and a model whose feature encoding depends on row order is a reproducibility bug
  waiting to be discovered at the worst possible moment. Finally it fixes the column order:

  ```
  experience block → location → categoricals → other numerics → skills
  ```

  and sets `categorical_features_ = ["experience_bucket", *self.category_levels_]`
  (iterating a dict yields its keys, so this is the bucket plus every categorical column
  that survived).

- `transform(df) -> pd.DataFrame` — calls all five blocks, concatenates them, asserts the
  row count is unchanged, and returns `out[self.feature_names_]`. The row-count check is
  explicit and raises `RuntimeError`, because **dropping rows inside a transform silently
  desynchronises `X` from `y`**, and the resulting misalignment is very hard to spot after
  the fact.

- `_categorical_block(df)` — private. For each frozen column it builds
  `[str(v) if not pd.isna(v) and str(v) in known else None for v in df[col]]`, then casts
  to the frozen `CategoricalDtype`. The `str()` first means `3` and `"3"` do not become two
  different categories. Mapping unknown values to `None` *before* the cast, rather than
  letting pandas silently drop them, keeps the behaviour explicit and deliberate.

- `_numeric_block(df)` — private. `pd.to_numeric(..., errors="coerce")` to `float64`, or an
  all-`NaN` column when absent.

- `fit_transform(df) -> pd.DataFrame` — `self.fit(df).transform(df)`. The docstring says
  "Only ever valid on the training split. Named to make misuse visible."

- `describe() -> str` — a short human summary: feature count, the categorical list, the
  first ten skills, and whether `prev_salary` is included. When it is not, the string reads
  `EXCLUDED (see design.md §4.1)`, which is loud on purpose.

#### The one thing to understand here

The class docstring states the rule that everything in the feature layer serves:

> Never call `fit` on the full dataset before splitting. Every learned thing in this
> pipeline — the skill vocabulary, the category lists — would then have been chosen with
> knowledge of the test rows, and the test score would stop measuring what you think it
> measures.

`band.py` enforces this structurally by owning its own `FeatureBuilder` and fitting it
inside `SalaryBandModel.fit`, so a caller cannot get it wrong by accident.

#### Surprises and gotchas

- **`city` never becomes a feature.** Only the derived `location_tier` does. On survey data
  that column is entirely `NaN`, so `FeatureBuilder` currently emits one column that
  carries no information at all — kept because it costs almost nothing and guarantees the
  matrix width is source-independent.
- **The `col not in EXCLUDED_COLUMNS` checks inside `fit` are belt and braces.** None of
  `CATEGORICAL_COLUMNS` or `PASSTHROUGH_NUMERIC` overlaps `EXCLUDED_COLUMNS`, so those
  conditions can never currently be false. They guard against a future edit to the tuples.
- **A category value that is `NaN` in training simply does not become a level**, because of
  the `.dropna()`. Missing stays missing at transform time too, so the two agree.
- **`describe()` slices `vocabulary_[:10]` and always appends `"..."`,** even when the
  vocabulary is shorter than ten. Cosmetic.
- **Feature count on the real data is 37 distinct features**, of which
  `docs/findings.md` §4.1 finds **16 carry no measurable held-out signal at all** and four
  actively hurt. The builder is not tuned; it emits everything and lets measurement sort it
  out. Whether pruning would help is listed as an open question, not as a finding.

---

### `src/paybands/model/split.py`

> Cuts the data into a training half and a testing half, and records how the cut was made.

**Read time:** 8 minutes · **Difficulty:** easy
**Read it when:** before anything in `model/`. It is 200 lines and mostly docstring.

#### What problem it solves

The whole point of a test set is to be a **stand-in for the future**. Hide some rows from
the model, then ask "how wrong were you about rows you had never seen?" That number is the
only honest estimate of how the model will behave on a candidate who walks in next month.

That framing decides *how* you cut, and picking wrong is the most common serious mistake in
applied ML.

#### Classes

**`Split`** — a frozen dataclass holding a train/test pair plus a record of how it was made.

| field | meaning |
|---|---|
| `train` | `pd.DataFrame` of training rows |
| `test` | `pd.DataFrame` of held-out rows |
| `strategy` | `"random"` or `"temporal"` |
| `description` | plain-English account of the cut, including the seed or cutoff |

Properties `n_train` and `n_test`; `__str__` prints e.g.
`random split — 818 train / 204 test (20% held out)` followed by the description.

`description` is not decoration. Six weeks from now you will have a results table with
numbers you cannot account for, and the first question will be "was that split random or
temporal, and what cutoff?". Carrying the answer alongside the data means you never have to
reconstruct it from memory.

#### Functions

```python
def random_split(df: pd.DataFrame, *, test_size: float = 0.2,
                 seed: int = DEFAULT_SEED) -> Split
```

Shuffles the rows and holds out `test_size` of them. `DEFAULT_SEED` is `42`. Validates
`0 < test_size < 1` and at least two rows, then:

- Uses `np.random.default_rng(seed)`, a **local** generator, not `np.random.seed()`. The
  global random state is shared by every library in the process, so anything else drawing a
  random number would silently change your "reproducible" split. This generator belongs to
  this call and nothing else can touch it.
- `n_test = max(1, round(len(df) * test_size))`, then clamped to `len(df) - 1` so the
  training set is never left empty.
- Selects with `.iloc`, which takes **positions**, not index labels. That distinction is
  load-bearing here: we shuffled positions, and `.loc` would try to look those numbers up
  as labels and quietly return the wrong rows.
- The index is preserved, so a test row can always be traced back to the row it came from.
  `conformal.py` depends on this — its overlap check compares index labels.

```python
def temporal_split(df: pd.DataFrame, date_col: str,
                   cutoff: str | pd.Timestamp) -> Split
```

Trains on everything strictly before `cutoff`, tests on everything from it onward. Parses
dates with `errors="coerce"`, **counts** the rows with unparseable or missing dates and
says so in the description rather than dropping them silently, and raises if either side
comes out empty (an empty side is nearly always a wrong cutoff or a date format pandas read
differently than you expected).

This function is currently unused — the survey has no dates. It exists now so that when
company data arrives there is no moment where a random split is the convenient thing lying
around.

#### The one thing to understand here — why the choice of split *is* a leakage question

A random split is fine for this survey because every respondent answered within the same
few weeks. There is no "past" and "future" inside the file, so shuffling loses nothing.

The instant rows carry dates, that stops being true. Suppose your company's data covers
2023–2026 and salaries rose 15% over that stretch. You shuffle it randomly. The training set
now contains 2026 hires, so the model learns "salaries around here are high". Then you score
it on test rows that are *also* from 2026, and it looks excellent.

Deploy it, and it is asked about 2027 — a year it has seen nothing from. The excellent score
never described that situation. It described a situation where the answer had already been
shown to the model.

That is **leakage**, exactly the same failure as the skills vocabulary but with a much
bigger effect size. It does not announce itself. It shows up as a model that scores well and
then disappoints in production, three months after the split decision that caused it.

A temporal split reproduces the real task exactly: you only ever know the past, and you are
always asked about the future. **The score it gives you will be worse than the random-split
score. That lower number is the true one.** `docs/findings.md` §6 says the same thing about
what to expect when company data arrives.

#### Surprises and gotchas

- **`temporal_split` takes its arguments positionally**, unlike `random_split` and every
  other split-like function in the project, which are keyword-only after the frame.
- **`random_split` rounds before clamping**, so on very small frames the realised test
  fraction can differ noticeably from what you asked for. On 5 rows with `test_size=0.2`
  you get exactly 1 test row (20%); on 3 rows you also get 1 (33%).
- The exchangeability assumption behind conformal prediction (see `conformal.py`) is exactly
  the assumption a random split makes and a temporal split does not. The module docstring in
  `conformal.py` notes this explicitly; it is an open design question rather than a solved
  one.

---

### `src/paybands/model/metrics.py`

> Every way this project scores a prediction — always in rupees, never in log space.

**Read time:** 15 minutes · **Difficulty:** medium
**Read it when:** before `baseline.py`. You will re-read `pinball_loss` and `coverage` when
you get to `band.py`.

#### What problem it solves

The model is trained on `log(salary)`, because salaries have a long right tail. But a score
of "MSE 0.083" in log space means nothing to anyone. "Typically off by ₹7,73,600" means
something to a recruiter, to your manager, and to you.

So there is one rule, and everything else in the file follows from it:

> **Train in log space if you like. Report in rupees, always.**

The file goes further than stating the rule: it deliberately provides no
`evaluate_in_log_space` function. If scoring in logs were one import away, someone (you, at
1am, in week 3) would report "MSE 0.083" and nobody reading it would know whether that was
good.

#### Functions

```python
def format_rupees(amount: float) -> str
```

Formats the way an Indian payslip does: `₹21,34,500`. Indian digit grouping is not the
Western one — the last three digits group together, then every **two** digits after that.
Ten lakh is `₹10,00,000`, not `₹1,000,000`. Eight lines of code, and it is the difference
between a tool that looks local and one that looks imported.

```python
def to_log(salary: ArrayLike) -> NDArray[np.float64]
def from_log(log_salary: ArrayLike) -> NDArray[np.float64]
```

`to_log` raises `ValueError` on any non-positive salary rather than producing `-inf`
quietly. `from_log` is `np.exp`.

`from_log` carries the subtlety that matters most in this file. `exp(log(x))` is `x` for a
single number, but **`exp` and `log` do not cancel across an average**. Take three
salaries, ₹10L, ₹20L and ₹40L:

- their ordinary (arithmetic) mean is **₹23.3L**;
- average their logs and exponentiate and you get **₹20L** — the **geometric mean**, which
  is always lower whenever the numbers differ at all.

So a model trained on logs, whose predictions you exponentiate, is not estimating the
average salary of people like this candidate. It is estimating something closer to their
*typical* salary — the middle of the pack, unmoved by the one founder in the group who
wrote down their equity.

For salary bands that is exactly what you want. But if you ever need a genuine expected
value — "what will these 40 hires cost us in total?" — exponentiating is biased low, and
the standard fix is Duan's smearing estimator. The file says: know it exists; we do not
need it here.

```python
def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float
```

**Mean absolute error.** The average miss, in rupees. "On average we are off by this much."
The most interpretable single number in the project, which is why it leads every report. It
is the mean of `|truth − prediction|`.

```python
def median_absolute_error(y_true: ArrayLike, y_pred: ArrayLike) -> float
```

The *typical* miss. Report it next to MAE and compare the two. MAE is dragged upward by a
handful of catastrophic misses; the median ignores them. If MAE is ₹6L and the median miss
is ₹2.5L, that gap is telling you the model is usually fine and occasionally very wrong —
a completely different problem from being uniformly mediocre, and it needs a different fix.

```python
def mape(y_true: ArrayLike, y_pred: ArrayLike) -> float
```

**Mean absolute percentage error**, returned as a percentage (already multiplied by 100).
Being ₹3L wrong about a ₹40L salary is a small mistake; being ₹3L wrong about a ₹6L salary
is a disaster. MAE cannot tell those apart; MAPE can.

Its known flaw, documented in the docstring: it punishes over-prediction more harshly than
under-prediction, because the error is divided by the **true** value.

```
predict ₹20L for someone on ₹10L  ->  |10−20| / 10  = 100%
predict ₹10L for someone on ₹20L  ->  |20−10| / 20  =  50%
```

The same factor-of-two error, scored twice as badly in one direction. This is why MAPE
never appears alone here — and why `docs/findings.md` §4.2 reports a MAPE of 227% for the
20+ experience bucket and immediately says it is "arithmetically correct and rhetorically
useless". Raises if any true value is zero, since the division would be infinite.

```python
def r2(y_true: ArrayLike, y_pred: ArrayLike) -> float
```

**R²** ("R squared"). Read it as a comparison: *how much better than always guessing the
mean?* 1.0 is perfect, 0.0 is no better than the mean, and **negative means worse than the
mean**. Computed as `1 − (sum of squared residuals) / (sum of squared deviations from the
mean)`; raises if every true value is identical, since the denominator would be zero.

**Why R² goes negative here, and why that is not a bug.** The global-median baseline scores
R² = −0.131 on one measured split. R² measures *squared* error, and squared error is
minimised by the **mean**. The baseline deliberately predicts the **median**. On a
right-skewed salary distribution those two numbers are far apart — in this data the mean is
₹24L and the median is ₹15L — so a predictor optimised for one loses badly on the other.

The baseline is still the better model for our purpose, because we care about the typical
candidate rather than about minimising squares. **A metric can say "worse" when the answer
is better.** This is what "the metric encodes a choice" means in practice, and the code
says so in the docstring rather than leaving you to discover it.

The file also warns that R² is a secondary number: it is unitless, so it cannot tell a
recruiter anything, and a bare R² with nothing to compare against is the classic portfolio
mistake. "We beat the group-median baseline by 19.4%" is a claim; "R² = 0.87" is not.

```python
def pinball_loss(y_true: ArrayLike, y_pred: ArrayLike, quantile: float) -> float
```

Scores a **quantile** prediction, in rupees, lower is better. A quantile is a cut point:
the 90th percentile is the number 90% of people earn less than.

**Why an ordinary error metric cannot score a quantile.** Suppose you are predicting the
90th percentile — a number that *should* sit above 90% of real salaries. MAE would score
that as wrong nine times out of ten and push it down toward the middle, destroying the
thing you asked for.

Pinball loss charges asymmetrically. The implementation is one line:

```python
np.mean(np.maximum(quantile * error, (quantile - 1) * error))   # error = truth − prediction
```

Where you under-predicted (`error > 0`) it charges `quantile` per rupee; where you
over-predicted it charges `1 − quantile`. With numbers, at `quantile=0.9`:

```
truth ₹20L, predicted ₹15L (under by ₹5L)  ->  max(0.9×5L, −0.1×5L) = ₹4.5L
truth ₹20L, predicted ₹25L (over  by ₹5L)  ->  max(−4.5L,   0.5L)   = ₹0.5L
```

Under-predicting costs nine times as much. So the cheapest possible prediction is one that
lands above the truth exactly 90% of the time — which is the *definition* of the 90th
percentile. **The metric and the goal are the same thing.** At `quantile=0.5` it reduces to
MAE, halved. Raises unless `0 < quantile < 1`.

```python
def coverage(y_true: ArrayLike, lower: ArrayLike, upper: ArrayLike) -> float
```

The fraction of true salaries that actually landed inside their predicted band. **This is
the honesty check, and it is about fifteen lines of code that almost nobody writes.**

If the model says "80% confident the salary is between ₹18L and ₹24L", then across a
hundred people, roughly eighty of their real salaries should fall inside their own band.
Run this and find out. If it returns 0.55, the model is overconfident — the bands are too
narrow and it is lying to the recruiter. If it returns 0.97, the bands are so wide they are
useless advice.

Note what it does *not* measure: **width**. A band of ₹0 to ₹5Cr scores perfect coverage
and helps nobody. Always report coverage and typical width together — coverage says the band
is honest, width says it is useful.

Validates that all three arrays have the same shape, and raises if any lower bound sits
above its upper bound (an inverted band). Both ends are inclusive, which with continuous
rupee amounts almost never matters but which prevents a flaky test later.

#### Classes

**`MetricSet`** — a frozen dataclass holding every score for one model on one dataset.

| field | meaning |
|---|---|
| `n` | number of rows scored |
| `mae` | average miss, rupees |
| `median_absolute_error` | typical miss, rupees |
| `mape` | average miss as a share of salary, percent |
| `r2` | share of variation explained |
| `label` | defaults to `"model"` |

Frozen because a results object you can edit after the fact is a results object you cannot
trust. `__str__` prints all four as an aligned table with a one-line gloss on each.

#### The two entry points

```python
def evaluate(y_true: ArrayLike, y_pred: ArrayLike, *, label: str = "model") -> MetricSet
def evaluate_log_predictions(y_true_rupees: ArrayLike, y_pred_log: ArrayLike,
                             *, label: str = "model") -> MetricSet
```

`evaluate` scores predictions that are already in rupees. `evaluate_log_predictions` is the
one a log-space model uses: it takes truth in rupees and predictions in logs,
back-transforms the predictions with `from_log`, and scores in rupees. Neither ever shows
you a log-space number.

`_as_pair` is the private helper both use: it coerces to float arrays, checks the shapes
match (a length mismatch is almost always a split gone wrong — predictions from one set
scored against another's labels, which numpy would broadcast into a plausible-looking wrong
answer), rejects an empty set, and rejects `NaN` predictions.

#### The one thing to understand here

Every metric encodes a preference, and reporting one number hides which preference you
picked. MAE cares about rupees. MAPE cares about proportions and punishes over-prediction.
R² cares about squares and therefore about the mean. Pinball loss cares about a specific
quantile. Coverage cares about honesty and says nothing about usefulness.

That is why the file offers several and the reports print several. The negative R² on the
median baseline is not an embarrassment to be hidden; it is the clearest possible
demonstration of the point.

#### Surprises and gotchas

- **`_as_pair` rejects `NaN` in the predictions but not in the truth.** A `NaN` truth would
  quietly propagate into MAE. In practice the loaders drop rows without a salary, so this
  has never bitten.
- **`format_rupees` rounds to whole rupees** (`f"{abs(amount):.0f}"`) and handles negatives
  with a leading `-` before the `₹`.
- **`coverage` returning 0.97 is a warning, not a success.** The docstring is explicit about
  this, and `band.py`'s `BandReport.__str__` encodes it: a coverage gap above +0.05 is
  labelled `conservative (wider than it needs to be)`.

---

### `src/paybands/model/baseline.py`

> The two deliberately stupid models that the real model has to beat.

**Read time:** 12 minutes · **Difficulty:** easy
**Read it when:** after `metrics.py`. This is the last file before the real model.

#### What problem it solves

"R² = 0.87" is not a result. It is a number with nothing behind it.

A result is *"the gradient boosting model is off by ₹12.5L where a `groupby` is off by
₹14.1L"* — an 11.9% improvement, stated against something a sceptical person can reproduce
in four lines of pandas.

And sometimes the honest answer comes out the other way. **If a thousand lines of LightGBM
only ties the lookup table, the model has not earned its complexity**, and the right
engineering call is to ship the lookup table: it trains instantly, never drifts in a way you
cannot see, and any HR person can read it. Knowing that requires having built the lookup
table first, which is why this file exists and why it exists *before* `band.py`.

Both classes follow scikit-learn's `fit`/`predict` shape so that swapping in the real model
is a one-line change and every evaluation script keeps working. They do not *inherit* from
scikit-learn: it is an optional dependency here, and a baseline that cannot run because a
40MB library is missing defeats its own purpose.

#### Module constants

```python
EXPERIENCE_BUCKETS = ((1, "0-1"), (3, "2-3"), (5, "4-5"), (8, "6-8"),
                      (12, "9-12"), (20, "13-20"), (float("inf"), "20+"))
UNKNOWN_BUCKET = "unknown"
DEFAULT_MIN_GROUP_SIZE = 10
```

Each pair is `(upper bound in years, label)`, and a person falls in the first bucket whose
upper bound they do not exceed. The widths are unequal on purpose: going from 1 to 3 years
moves your salary far more than going from 15 to 17, so equal five-year buckets would lump
the steep part of the curve together and split the flat part apart, which is exactly
backwards.

`UNKNOWN_BUCKET` is where rows with a missing experience value go. They get their own bucket
rather than being dropped or filled with a guess — "we do not know" is a real state, and
pretending it is "0 years" would drag that bucket's median down.

`DEFAULT_MIN_GROUP_SIZE = 10` is the threshold below which a cell's median is not trusted.
**Why a minimum at all:** the median of two people is not a market rate, it is those two
people. If one of them is a founder who wrote down their equity, every future Security hire
gets quoted a fantasy number. Below the threshold the group is deliberately thrown away in
favour of a coarser, better-populated estimate — a slightly less specific answer built on
real evidence beats a very specific answer built on noise.

#### Functions

```python
def experience_bucket(years: float | None) -> str
def experience_buckets(years: pd.Series) -> pd.Series
```

Scalar and vectorised versions. The scalar one checks `years is None or pd.isna(years)`
first: missing values arrive here as `None`, numpy `NaN`, or pandas' own `pd.NA` depending
on the column's dtype, and only `pd.isna` recognises all three.

#### Classes

**`GlobalMedianBaseline`** — predicts the same number, the overall median, for everybody.

This is the floor. A model that cannot beat "everyone earns the median" has learned nothing
at all, and that is a genuinely useful thing to be able to check in one line.

**Median, not mean.** Salaries have a long right tail. In this survey the mean is ₹24L and
the median is ₹15L — the mean sits above roughly two-thirds of respondents, so it describes
almost nobody. The median is the better single guess, and choosing it *is* the whole model.

| field | meaning |
|---|---|
| `median_` | **learned** — the one number, or `None` before fitting |

- `fit(X: pd.DataFrame, y: ArrayLike) -> GlobalMedianBaseline` — `X` is accepted and
  ignored on purpose. Taking an unused `X` is what makes this interchangeable with every
  other model in the project: the evaluation loop does not need to know which model it is
  holding. Raises on an empty target.
- `predict(X: pd.DataFrame) -> NDArray[np.float64]` — `np.full(len(X), self.median_)`.
  Raises `RuntimeError` if unfitted.

**`GroupMedianBaseline`** — a lookup table: the median salary per (role × experience
bucket).

Four lines of pandas, no learning, no parameters to tune, and typically the number that a
gradient boosting model beats by less than you would hope. On this data the model beats it
by 11.9% — real, consistent across 10 of 10 splits, and nothing like the 3x that a bare R²
would let you imply.

```python
GroupMedianBaseline(*, role_col: str = "role",
                       experience_col: str = "years_experience",
                       min_group_size: int = DEFAULT_MIN_GROUP_SIZE)
```

| field | meaning |
|---|---|
| `role_col`, `experience_col` | which input columns to group by |
| `min_group_size` | minimum people in a cell before its median is trusted |
| `group_medians_` | **learned** — `{(role, bucket): median}` for cells that cleared the threshold |
| `bucket_medians_` | **learned** — `{bucket: median}`, no threshold applied |
| `global_median_` | **learned** — the answer of last resort |
| `fallback_counts_` | a `Counter` recording which rung answered, reset on every `predict` |

**The fallback hierarchy** is the part that takes actual thought. A lookup table has one
failure mode: you look something up and it is not there. A Security engineer with 30 years'
experience walks in, and that cell is empty or has three people in it. So predictions
cascade:

1. `(role, experience bucket)` — the specific answer, used only if that group has at least
   `min_group_size` people behind it. Reported as `"role+experience"`.
2. `experience bucket` alone — drops the role. Much better populated, and experience is the
   strongest single driver of salary anyway, so this is a genuinely reasonable second-best
   rather than a token gesture. Reported as `"experience only"`.
3. the global median — always available. Reported as `"global median"`.

Each step trades precision for evidence, deliberately and in that order.

Methods:

- `fit(X, y)` — validates the columns exist and that `len(X) == len(y)`, then rebuilds
  everything into a plain three-column frame (`role`, `bucket`, `salary`) built from
  `.to_numpy()` calls. That rebuild is not decoration: **silent index misalignment between
  a DataFrame and a Series is one of pandas' sharpest edges, and it produces `NaN`s rather
  than an error.** Then it computes level 2 first (bucket medians, with *no* minimum size —
  this **is** the fallback, so if it could also fall back the cascade would have a hole in
  it), and level 1 second, keeping only cells where `size >= min_group_size`.
- `predict(X)` — resets `fallback_counts_`, walks the rows, and records which rung answered
  each one.
- `_lookup(role, bucket) -> tuple[float, str]` — private. Walks the cascade and returns both
  the value and the name of the rung.
- `explain(role: str, years: float | None) -> str` — e.g.
  `Backend, 6-8 years → ₹22,00,000  [matched on: role+experience]`. "A number a recruiter
  cannot argue with is a number a recruiter should not be given." This is the baseline's
  version of the SHAP explanations the real model gets later.
- `coverage_report() -> str` — how often each rung fired in the last `predict`.

#### The one thing to understand here

Read `coverage_report()` every time. A table answering "global median" for 30% of your
candidates is not a lookup table, it is a constant with extra steps — and the aggregate MAE
will not tell you that.

On this data the measured cascade is: **72.5%** matched a full (role × bucket) cell, **27.5%**
fell back to experience alone, and **0%** reached the global median. So more than a quarter
of candidates are quoted a number that ignores their role entirely. The headline MAE looks
identical either way. This is the pattern that recurs throughout `findings.md`: aggregate
scores hide where the model is actually working.

#### Surprises and gotchas

- **`GroupMedianBaseline` gives the same predictions fitted on rupees or on log rupees**
  (after transforming back), because the median of a list of logs is the log of the median —
  taking logs reorders nothing and the median only cares about order. That is **not** true
  of the mean, which is a large part of why this file uses medians throughout.
- **`fallback_counts_` is state mutated by `predict`.** Calling `predict` twice and then
  reading `coverage_report()` describes only the second call.
- **This file's `EXPERIENCE_BUCKETS` are not `features/experience.py`'s `BUCKET_EDGES`.**
  Same seven labels, boundaries defined the other way round (upper-bound-inclusive here,
  left-closed there). They agree exactly on whole years, which is all the survey contains.
- **`GlobalMedianBaseline` scores a negative R²**, and that is the worked example in
  `metrics.r2`. Do not go looking for the bug; there is not one.

---

### `src/paybands/model/band.py`

> **The core file.** Three gradient boosting models that together draw a salary range
> instead of a single number.

**Read time:** 30 minutes · **Difficulty:** hard
**Read it when:** you have read `metrics.py` (especially `pinball_loss` and `coverage`),
`baseline.py`, and `features/builder.py`. This file assumes all three.

#### What problem it solves

Everything before this predicted *one number*. That is fake precision: nobody knows a
candidate's salary to the rupee, and quoting a single figure implies you do. This file
predicts a **band** — ₹18L–₹24L with a middle at ₹21L — which is the only shape of answer a
recruiter can actually use.

The algorithm is **LightGBM**, a gradient boosting library. *Gradient boosting* means:
build many small decision trees, where each new tree tries to fix the mistakes of the ones
before it. For **tabular data** — rows and columns, like a spreadsheet — it is the best
general-purpose choice available, beating both linear models and neural networks in almost
all cases. Neural networks would be the wrong tool for 1,022 rows.

The file is organised around four ideas, and its module docstring states them explicitly.

#### Idea 1 — train in log space, report in rupees

Salary is right-skewed, and more importantly salary differences are **multiplicative**.
Nobody says "I got a ₹2.4 lakh raise" as a way of describing how well they did; they say "I
got a 30% raise", because 30% means the same thing to a fresher and to a principal engineer
while ₹2.4L means wildly different things to the two.

So the model learns `log(salary)`, where multiplying becomes adding and "30% more" is a
constant step at every salary level. Predictions come back through `exp` before anyone sees
them. The geometric-mean subtlety from `metrics.from_log` applies here in full and the file
repeats it.

#### Idea 2 — three models, not one model with an error bar

The obvious way to build a band is: train one model, measure how wrong it usually is, and
put ±that around every prediction. **Do not.**

That band has one width, so it is **the same width for a fresher and for a CTO**, which is
wrong in both directions at once:

- Junior salaries genuinely cluster. The market for a 1-year backend engineer in Bangalore
  is narrow, and a ±₹6L band around it is uselessly vague.
- Senior and leadership salaries genuinely spread. Two CTOs can be a factor of three apart,
  and a ±₹6L band there is confidently too narrow — which is worse than vague, because
  someone will believe it.

So instead: three separate LightGBM models, each with `objective="quantile"`.

| `alpha` | what it learns | role in the band |
|---|---|---|
| 0.1 | the 10th percentile | the low edge |
| 0.5 | the 50th percentile (the median) | the middle |
| 0.9 | the 90th percentile | the high edge |

Each is fitted with **pinball loss** (see `metrics.pinball_loss`). For the 0.9 model,
under-predicting costs nine times as much as over-predicting, so the cheapest prediction it
can make is one that sits above the truth 90% of the time. The metric *is* the goal.

**The gap between the 0.1 and 0.9 predictions is the band.** Because all three are full
models with access to experience, role and everything else, that gap is free to be narrow
for the fresher and wide for the CTO. It is learned, not assumed — and it worked:
`docs/findings.md` §2.4 measures band width in rupees growing **5.0x** from the bottom
decile of predicted salary to the top.

Nominal coverage is `0.9 − 0.1 = 80%`. Whether the band *delivers* 80% is a completely
separate question, and the answer turned out to be no. That is what `conformal.py` exists
to fix.

#### Idea 3 — quantile crossing, and why it is counted rather than hidden

Nothing ties the three models together. They are fitted independently, on the same rows,
with three different loss functions. So on some rows the 0.1 model predicts *above* the 0.9
model and the band comes out inside-out. This is called **quantile crossing**.

It is not a bug in LightGBM; it is the direct consequence of not constraining the models
jointly, and it happens most on rows where the data is thin — exactly the rows you were
least sure about anyway.

The repair is embarrassingly simple: sort the three numbers per row. That is provably no
worse than the originals under pinball loss (a special case of the rearrangement result of
Chernozhukov, Fernández-Val and Galichon), so there is no reason not to do it.

**The count is the interesting part, and this module reports it.** A crossing rate of 1%
means the three models mostly agree about the shape of the world. A crossing rate of 15%
means they do not, and your band edges are being determined by noise. That is a diagnostic
worth putting in a report, not a wart to hide. Measured here: **0.2%** of test rows, ranging
0% to 1.5%.

#### Idea 4 — small data needs a small model

This dataset is about 1,000 rows. LightGBM's defaults (31 leaves, unlimited depth, no
regularisation) were chosen for datasets a thousand times larger, and on 1,000 rows they
will happily grow a tree with one row per leaf and memorise the training set perfectly. You
would then see a beautiful training score and a test score worse than the `groupby`
baseline. That is **overfitting**.

> Model capacity should be matched to the amount of evidence, not to the size of the library
> you imported.

`SMALL_DATA_PARAMS` turns every dial towards "less model":

| parameter | value | why |
|---|---|---|
| `objective` | `"quantile"` | pinball loss — the whole point of the file |
| `n_estimators` | `800` | an upper bound only; early stopping picks the real number |
| `learning_rate` | `0.03` | small steps, so early stopping has a fine-grained choice |
| `num_leaves` | `7` | a depth-3 tree at most. The default is 31, far too expressive here |
| `max_depth` | `3` | belt and braces: caps interaction depth as well as leaf count |
| `min_child_samples` | `25` | a leaf must describe 25 real people, not 3 outliers |
| `subsample` | `0.8` | each tree sees 80% of rows... |
| `subsample_freq` | `1` | ...and this is what actually switches that on |
| `colsample_bytree` | `0.7` | ...and 70% of columns, so no one feature dominates |
| `reg_lambda` | `5.0` | L2 on leaf values: pulls extreme predictions inward |
| `verbose` | `-1` | LightGBM is chatty about tiny data; we print our own numbers |

These are hand-chosen, not tuned, and the code says so. Three related constants:

- `NO_EARLY_STOPPING_TREES = 300` — how many trees to build when early stopping is off.
- `MIN_VALIDATION_ROWS = 40` — below this, early stopping is disabled entirely. Early
  stopping asks "did the validation score improve?"; on 15 rows that question is answered by
  noise, and you would stop at whatever iteration happened to get lucky. Fewer trees chosen
  blindly beats a stopping point chosen by coin flip.
- `EARLY_STOPPING_ROUNDS = 50` — rounds without improvement before stopping.

There is also `_FIT_TAKES_EVAL_XY`, a module-level boolean computed by inspecting
`lgb.LGBMRegressor.fit`'s signature. LightGBM 4.7 renamed `fit(eval_set=[(X, y)])` to
`fit(eval_X=..., eval_y=...)`, and `pyproject.toml` allows 4.5 and up, so the code asks the
installed version which spelling it speaks rather than picking one and either emitting a
deprecation warning on every fit or crashing on an older install.

#### Compa-ratio

```python
COMPA_BELOW_BAND = 0.90
COMPA_ABOVE_BAND = 1.10

def compa_label(ratio: float) -> str      # "below band" / "in band" / "above band"
```

**Compa-ratio** is standard HR vocabulary: an employee's actual salary divided by the
midpoint of their band. 0.95 means they are paid 5% under the middle of the market for their
role and level.

The thresholds are **policy, not measurement** — a company can and does move them — so they
sit as named constants rather than being written into a comparison somewhere.

- below **0.90** → paid below band. A genuine flight risk, and the pay-equity list.
- 0.90 to 1.10 → in band.
- above **1.10** → above band, so a smaller increment this cycle.

Both of the tool's headline use cases *are* this number. **Pay equity** is everyone below
0.90. **Increments** are "more to the low ratios, less to the high ones, within budget". No
second model is needed for either.

#### Classes

**`BandPrediction`** — a frozen dataclass holding one band per row, in rupees, plus the
honesty diagnostics. Frozen because a predictions object you can edit after the fact is a
predictions object nobody can audit.

| field | meaning |
|---|---|
| `lower` | low edge in rupees, one per row |
| `median` | middle in rupees |
| `upper` | high edge in rupees |
| `quantiles` | the three levels these came from, so a reader never has to guess whether "the band" meant 80% or 50% |
| `n_crossed` | how many rows were out of order before sorting (default 0) |

Properties:

- `n` — number of rows.
- `nominal_coverage` — `quantiles[2] − quantiles[0]`. What the band **claims** to contain.
- `width` — `upper − lower`, in rupees, per row.
- `relative_width` — `width / median`. The comparable number across salary levels: a ₹6L
  band is enormous at ₹8L and tight at ₹60L; 0.4 is 0.4 everywhere.
- `crossing_rate` — `n_crossed / n`.

Methods: `to_frame(index=None)` returns a three-column DataFrame; `explain(i=0)` writes one
row the way it would be read aloud in a hiring meeting —
`₹4,91,660 – ₹39,84,224  (midpoint ₹13,95,079, 80% band)`.

**`BandReport`** — a frozen dataclass holding how good a band is, which takes exactly two
numbers, never one.

| field | meaning |
|---|---|
| `n` | rows scored |
| `label` | name for the report |
| `target_coverage` | what was promised |
| `coverage` | what was delivered |
| `mean_width`, `median_width` | band width in rupees |
| `mean_relative_width` | width ÷ midpoint |
| `crossing_rate` | share of rows repaired, before sorting |

`coverage_gap` is `coverage − target_coverage`; negative means overconfident. `__str__`
prints a verdict: `overconfident` when the gap is below −0.02, `honest` otherwise, and
`conservative (wider than it needs to be)` when the gap exceeds +0.05.

> **Coverage** says whether the band is honest. **Width** says whether it is useful. Read
> them together and in that order. **Narrower is better only once coverage holds** — a model
> that reports a tighter band and worse coverage has not improved, it has started lying more
> confidently.

**`SalaryBandModel`** — the model itself.

```python
SalaryBandModel(*, quantiles: tuple[float, float, float] = DEFAULT_QUANTILES,
                   seed: int = DEFAULT_SEED,
                   validation_size: float = 0.2,
                   params: dict[str, object] | None = None,
                   top_skills: int | None = None,
                   include_prev_salary: bool = False)
```

Validates `0 < lo < mid < hi < 1` and `0 < validation_size < 1`. `params` is merged *over*
`SMALL_DATA_PARAMS`, so a caller can override one dial without restating the rest.

| field | meaning |
|---|---|
| `builder` | its **own** `FeatureBuilder`, created in `__init__` |
| `models_` | **learned** — the three fitted LightGBM regressors, low to high |
| `best_iterations_` | **learned** — how many trees each one actually kept |
| `feature_names_` | **learned** — copied from the builder |
| `train_index_` | **learned** — a `frozenset` of the index labels used for training |
| `n_train_`, `n_validation_` | **learned** — row counts |
| `n_crossed_`, `crossing_rate_` | from the most recent prediction |

Two of those deserve a note. **`best_iterations_`** is worth reading after every fit: if all
three sit at `n_estimators` the model never stopped improving and you capped it too low; if
they sit at 20 there was almost no signal. **`train_index_`** exists purely so
`conformal.ConformalBand` can *refuse* to calibrate on rows the model has already seen.

Properties: `nominal_coverage` (0.9 − 0.1 = 80%, "a claim, not yet a fact") and `is_fitted`.

Methods:

- `fit(X: pd.DataFrame, y: ArrayLike) -> SalaryBandModel`

  `X` is a **common-schema frame**, not a feature matrix. The model owns its
  `FeatureBuilder` and fits it here, on the training rows only. That is not a convenience:
  a builder fitted on all the data has chosen its skill vocabulary and category lists with
  knowledge of the test rows, which is leakage, and it makes the test score a flattering
  lie. **Keeping the builder inside the model makes that mistake impossible to make by
  accident.**

  `y` is annual salaries **in rupees**. They are converted with `to_log` inside the method,
  so no caller has to remember to do it and nobody can accidentally fit one model on logs
  and another on rupees.

  It then carves a validation slice using a local `np.random.default_rng(self.seed)` — never
  `np.random.seed()`, for the same reason as `split.random_split`. The slice stays *inside*
  your training data; the test set is untouched, so this costs nothing you were entitled to.
  The `FeatureBuilder` is fitted on *all* the rows given here, validation slice included,
  because that slice is training data too.

  Then, for each of the three alphas, it constructs an `LGBMRegressor` and fits it with
  `categorical_feature=self.builder.categorical_features_`. When there are at least
  `MIN_VALIDATION_ROWS` validation rows it adds early stopping.

  **There is deliberately no `eval_metric` argument.** LightGBM then scores the validation
  set with the objective itself — pinball loss at this very alpha. Early-stopping on the
  metric you are actually optimising is the point; stopping on, say, RMSE would drag the 0.9
  model back towards the middle, undoing its whole job.

- `predict_quantiles_log(X) -> NDArray[np.float64]`

  The three quantiles in **log space**, sorted, shape `(n_rows, 3)`. This is the primitive
  everything else is built on, and it is the method `conformal.ConformalBand` requires of
  anything it wraps.

  **Why log space is the interface:** conformal calibration widens a band by a constant
  amount. A constant in log space is a constant *percentage* in rupees — every band grows by
  the same 12%. A constant in rupees would add ₹3L to the fresher's band and ₹3L to the
  CTO's, which is the fixed-width mistake from Idea 2 sneaking back in through the
  calibration step.

  The `np.sort(raw, axis=1)` is the crossing repair. The count uses exact float comparison
  (`np.any(raw != repaired, axis=1)`), which is correct here because sorting permutes values
  and never alters them, so a row is unchanged if and only if it was already in order.

- `predict_band(X) -> BandPrediction` — the same three numbers back through `from_log`,
  wrapped up with the quantile levels and the crossing count. This is the method the API
  calls.

- `predict(X) -> NDArray[np.float64]` — the midpoint alone, so the model is interchangeable
  with the baselines in any point-scoring loop. The docstring says to prefer `predict_band`:
  "a single number is exactly the fake precision this project set out to avoid".

- `compa_ratio(actual_salary: ArrayLike, X: pd.DataFrame) -> NDArray[np.float64]` —
  `actual ÷ predicted median`. One division, and the centre of the tool. Note **the
  denominator is the median, not the midpoint of lower and upper.** On a skewed distribution
  those differ, and the median is what HR means by "the middle of the band" — the salary
  half the market is below.

- `describe() -> str` — quantiles, nominal coverage, training rows, feature count, whether
  early stopping ran, trees kept per quantile, and the crossing rate of the last prediction.

```python
def band_report(y_true_rupees: ArrayLike, band: BandPrediction,
                *, label: str = "band") -> BandReport
```

Scores a band against salaries people actually earn. Checks the shapes match, then fills in
every field of `BandReport` using `metrics.coverage` for the coverage figure.

#### The one thing to understand here

Three models beat one prediction plus a fixed error bar because **the width of the band is
itself something to be learned.** Uncertainty about a junior's salary is genuinely smaller
than uncertainty about a senior architect's, and a single error bar cannot express that.

And then: even with three models, the "80%" on the tin is arithmetic about the loss
functions you requested, not a measurement of what the model delivers. It delivered 74.9%.
Everything in `conformal.py` follows from taking that gap seriously.

#### Surprises and gotchas

- **`predict_quantiles_log` mutates the model.** It writes `n_crossed_` and
  `crossing_rate_` on every call. So `describe()` reports the crossing rate of whichever
  prediction ran most recently, and the class is not safe to share between threads.
- **`compa_ratio` runs a full prediction internally.** Calling it after `predict_band` costs
  a second forward pass through all three models, and it overwrites the crossing counters.
- **`train_index_` is a `frozenset` of index labels**, so the conformal overlap check only
  works if you preserve the index. `split.random_split` does. If you build a split some
  other way and reset the index, the check goes quiet — it cannot see rows it cannot
  identify. The docstrings in both files say so rather than pretending otherwise.
- **The `alpha` parameter here is LightGBM's quantile level**, unrelated to the `alpha` in
  `conformal.py`, which is a miss rate. Same word, two meanings, one file apart.
- **A 0.2% crossing rate is a good sign, not a small bug.** It is evidence the three models
  agree about the shape of the world.
- **`best_iterations_` uses `model.best_iteration_ or model.n_estimators_`.** When early
  stopping is off, `best_iteration_` is falsy and the fallback reports the configured tree
  count.

---

### `src/paybands/model/conformal.py`

> Takes a band that promises 80% and delivers 75%, and makes the promise true.

**Read time:** 25 minutes · **Difficulty:** hard
**Read it when:** immediately after `band.py`. It is meaningless on its own.

#### What problem it solves

`band.py` trains three models and calls the gap between the 0.1 and the 0.9 an "80% band".
Read that again. It says 80% because 0.9 − 0.1 = 0.8, which is **arithmetic about the loss
functions we asked for**, not a measurement of what the model does.

Measured on this project's data, the raw quantile band covers **74.9%**, not 80%. Somewhere
between "we asked for the 90th percentile" and "the model found the 90th percentile" sit a
finite training set, a model with 7 leaves, and a test set the model has never seen — and
the promise leaks out through all three. **Nothing about the output announces this.** You
only find out by counting.

A band that promises 80% and delivers 75% is lying to a recruiter. They will make offers at
the edge of it and be wrong more often than they were told.

The fix, measured across 10 splits: **82.3%** coverage (range 79.4%–87.3%). And checked at
five confidence levels rather than one — because one level checked is a spot check, five is
a calibration curve:

| promised | raw delivered | conformal delivered |
|---|---|---|
| 50% | 43.0% | 48.5% |
| 60% | 53.1% | 59.8% |
| 70% | 63.1% | 67.3% |
| 80% | 72.9% | 81.7% |
| 90% | 86.8% | 90.4% |

The raw band under-covers at 5 of 5 levels, by a strikingly consistent ~7 points. The
calibrated band's worst miss is −2.7 points.

#### The method, in four lines

**Conformalised Quantile Regression** (Romano, Patterson and Candès, 2019). Take a
**calibration set** the model has never seen. For each row in it, ask how badly the band
missed:

```
score = max(lower − y,  y − upper)
```

Read that as *how far outside the band did the truth fall?* If `y` sits below the band,
`lower − y` is how far below. If it sits above, `y − upper` is how far above. If it sits
comfortably inside, both terms are negative and the score is negative — **a credit**,
measuring how much room to spare there was.

Now take the `⌈(n+1)(1−α)⌉`-th smallest of those scores; call it `q̂` (q-hat). Widen every
band by `q̂` on both sides. Done.

The result is a **finite-sample coverage guarantee**: the widened band covers at least
`1 − α` of new rows. Not asymptotically, not on average over many datasets — for this
dataset, at this size, right now. The only assumption is **exchangeability**: the calibration
rows and the future rows have to be drawn from the same pool, in no particular order.
(Which is exactly the assumption a random split makes and a temporal split does not.)

Three things worth noticing, all of them stated in the module docstring:

- **The odd-looking `(n+1)` is not a fudge factor.** The guarantee is proved by treating the
  new row as one more member of the calibration set and asking where its score would rank
  among `n + 1` scores. The `+1` is that new row.
- **`q̂` can be negative**, and then CQR *narrows* the band. That is not a bug. If the
  quantile models were over-cautious — covering 92% when 80% was asked for — the honest
  correction is a tighter band, and the guarantee still holds. Conformal prediction
  calibrates in both directions.
- **All of it happens in log space**, for the same reason `predict_quantiles_log` is the
  interface: a constant in log space is a constant percentage in rupees.

Measured here, `q̂` averaged **0.1573 log points**, which is each band edge moving by
**17.2%** on average.

#### The part people get wrong: you need THREE sets, not two

This is the loudest warning in the file, drawn as a diagram in the docstring:

```
┌──────────────┬──────────────┬──────────────┐
│    TRAIN     │ CALIBRATION  │     TEST     │
│  fit trees   │  compute q̂  │  report      │
└──────────────┴──────────────┴──────────────┘
```

**Calibrating on the training data destroys the guarantee.** The model has already seen
those rows; it fits them far better than it fits anything new. The scores come out small,
`q̂` comes out small, the band barely widens, and you publish a coverage guarantee that is
simply false.

**Calibrating on the test data is leakage**, the same mistake in a different coat. You would
have tuned the band to the very rows you then use to claim it works, and the reported
coverage would be a fact about your test set rather than a prediction about anyone's future.

Both mistakes are silent, and — this is the dangerous part — **both produce numbers that
look better than the honest ones.** The code runs. Nothing warns you. The flaw shows up in
production.

So the module ships `three_way_split`, which makes the correct cut in one call, and
`ConformalBand.calibrate`, which **refuses** to run if the calibration rows overlap the
model's training rows.

Cost of the third set: on 1,000 rows, roughly 200 rows the trees never get to learn from.
That is a real price. `docs/findings.md` §2.2 pays it explicitly and then runs the fair
fight anyway — comparing the band model on 60% of the data against a lookup table built on
80% — and the model still wins on 10 of 10 splits, by 11.0%.

#### Classes

**`QuantileModel`** — a `@runtime_checkable` `Protocol` declaring one method,
`predict_quantiles_log(X) -> NDArray[np.float64]`.

A `Protocol` is structural typing: "if it has this method it qualifies", with no inheritance
required. CQR knows nothing about LightGBM. Any model producing a low/middle/high triple in
log space can be handed to `ConformalBand`, including a linear quantile regression or a
hand-written lookup table — which is what makes this module testable without training
anything.

**`ThreeWaySplit`** — a frozen dataclass carrying `train`, `calibration`, `test` and a
`description`.

Handing the three frames around as one object is the point. Three loose DataFrames named
`train`, `cal` and `test` are three chances to pass the wrong one, and the mistake produces
a *better* number, so nothing about the output tells you it happened.

`__post_init__` verifies the three index sets are pairwise disjoint and raises with the
count of shared rows if not. `sizes` returns the three lengths; `__str__` prints them.

**`ConformalBand`** — wraps a quantile model and makes its coverage promise true.

```python
ConformalBand(model: QuantileModel, *, alpha: float | None = None)
```

`alpha` is the allowed **miss rate**, so `alpha=0.2` means "at least 80% coverage".
`DEFAULT_ALPHA` is `0.2`. Passing `None` (the default) reads it off the model's own
`nominal_coverage`, so a model built on the 0.1 and 0.9 quantiles is automatically
calibrated to 80% and the two numbers cannot drift apart because somebody edited one of
them.

| field | meaning |
|---|---|
| `model` | the wrapped quantile model |
| `alpha` | allowed miss rate |
| `qhat_log_` | **learned** — the widening, in log points |
| `scores_` | **learned** — the sorted calibration conformity scores |
| `n_calibration_` | **learned** — how many rows produced `q̂` |

`scores_` is worth plotting: its shape tells you whether the raw band was uniformly too
narrow, or fine for most people and hopeless for a few.

Properties: `target_coverage` (`1 − α`), `is_calibrated`, and `widening_factor`
(`exp(qhat_log_)`, so 1.09 means "each edge moved 9%").

Methods:

- `calibrate(X_cal: pd.DataFrame, y_cal: ArrayLike) -> ConformalBand`

  Three guards before any arithmetic: shapes match; there are enough calibration rows
  (`minimum = math.ceil(1.0 / alpha) - 1`, which is **4** at α = 0.2 — below that, the
  honest `q̂` is infinity, and pretending otherwise is worse than refusing); and the
  calibration index does not intersect the model's `train_index_`.

  The overlap check is a genuine guard, not decoration, and its error message says why:

  > "The model already fits those rows better than it fits new ones, so the conformity
  > scores would be too small and the coverage guarantee would be false. Use
  > three_way_split()."

  Then it predicts, computes `scores = np.maximum(lower − y_log, y_log − upper)`, sorts them,
  and picks `rank = ceil((n + 1) * target_coverage)` clipped to `n`, taking
  `scores_[rank - 1]` as `qhat_log_`.

  A worked example with round numbers: 200 calibration rows at 80% target.
  `ceil(201 × 0.8) = ceil(160.8) = 161`, so `q̂` is the 161st smallest score — meaning the
  band widens by enough to have covered 161 of the 200 calibration rows, which is 80.5%.
  The extra half-row is the `+1` doing its job.

- `predict_quantiles_log(X)` — subtracts `q̂` from the low edge and adds it to the high
  edge, on a `.copy()` so the wrapped model's output is untouched. Then clips: the lower
  edge cannot exceed the median, and the upper edge cannot fall below it. That clip only
  matters when `q̂` is negative and large enough to invert the band — the median is a
  prediction, not an edge, and a band that excludes its own midpoint is nonsense to show
  anybody.

  Because this method exists, a `ConformalBand` is itself a `QuantileModel` and could in
  principle be wrapped again.

- `predict_band(X) -> BandPrediction` — the calibrated band in rupees. **The midpoint is
  untouched.** CQR adjusts the *edges*; it says nothing about whether the median model is
  any good, which is what MAE is for.

- `evaluate(X_test, y_test, *, label="conformal band") -> BandReport` — coverage and width
  on the test set, via `band_report`.

- `describe()` — target coverage, calibration rows, `q̂` in log points, the direction and
  size of the move, and the range the scores ran over. It prints
  `NARROWED (the raw band was too cautious)` when `q̂` is negative, which is the case the
  file wants you to know is possible.

#### Functions

```python
def three_way_split(df: pd.DataFrame, *, calibration_size: float = 0.2,
                    test_size: float = 0.2, seed: int = DEFAULT_SEED) -> ThreeWaySplit
```

Cuts a dataset into the three sets conformal prediction needs. **Both fractions are of the
original total**, not of what is left after the previous cut. Ask for 0.2 and 0.2 and you
get 60/20/20 — which is what everybody means and not what nested splitting naturally gives
you. The rescaling is one line: `inner_fraction = calibration_size / (1.0 - test_size)`.

The seed handling is worth noting. It derives **two independent child seeds** from the
caller's one, via `np.random.SeedSequence(seed).spawn(2)`. Reusing `seed` for both cuts
would work by accident here (the frames have different lengths, so the permutations differ),
and "works by accident" is how a reproducibility bug gets planted.

#### The one thing to understand here

Conformal prediction does not make the model better. It makes the model's *claim about
itself* true, by measuring the miss on data the model has never seen and widening by exactly
that much.

And it costs something. `docs/findings.md` §3 is blunt about it:

| | width ÷ midpoint |
|---|---|
| raw quantile band | median **1.94x** |
| after conformal calibration | median **2.40x** |

**Honesty cost 23.4% more width.** The raw band is narrower and lies about its confidence;
the calibrated band is wider and tells the truth. That is the correct trade — but it means
the honest version is the less usable one. A typical calibrated answer is
*"₹4,91,660 – ₹39,84,224"*, and nobody can make an offer from that. The project's headline
negative result is that the band it built honestly is too wide to quote from, and the
evidence says that is a data limitation rather than a modelling failure.

#### Surprises and gotchas

- **The overlap check is `if train_index:` — an empty `frozenset` is falsy.** A wrapped model
  that does not record a `train_index_` (a hand-written test double, say) skips the check
  entirely, silently. That is by design so the `Protocol` stays minimal, but it means the
  guard protects you only when you use `SalaryBandModel`.
- **`predict_band` reports `quantiles=(alpha/2, 0.5, 1 - alpha/2)`.** For the default α = 0.2
  that is exactly (0.1, 0.5, 0.9) and matches the underlying models. It assumes a symmetric
  two-sided band; if you wrapped a model built on, say, (0.05, 0.5, 0.9), the reported
  quantile labels would not describe the models that produced the numbers. `nominal_coverage`
  would still be right.
- **The `n_crossed` on a calibrated `BandPrediction` comes from the wrapped model** via
  `getattr(self.model, "n_crossed_", 0)`, so it reflects the raw models' crossing count, not
  anything conformal did.
- **`α` here is a miss rate; `alpha` in `band.py` is a quantile level.** Two different
  meanings of the same word, one file apart.
- **The consistent −7 point raw shortfall is unexplained.** `docs/findings.md` §3 flags it
  honestly: four of five levels miss by between −6.9 and −7.1 points, which is too neat to be
  noise. The guess is pinball loss on 613 rows with `min_child_samples=25` pulling the
  extreme quantiles inward. The symptom was fixed without diagnosing the cause — good
  engineering, unfinished science, and it is listed as an open question rather than glossed.

---

### `tests/test_features.py`

> 472 lines of tests for the feature layer, built around one test that guards the discipline
> rather than the behaviour.

**Read time:** 15 minutes · **Difficulty:** easy to read, medium to appreciate
**Read it when:** after `builder.py`. Read it before you trust anything in `features/`.

#### What problem it solves

The module docstring says it plainly:

> The most important test in this file is `test_fit_on_train_does_not_see_test_skills`.
> Everything else checks the transforms behave sensibly; that one checks the **discipline**.
> If that test ever starts failing, do not relax it. It is guarding the mistake that makes a
> model look excellent in validation and fail in production.

#### The fixtures

Two hand-written frames, and the design of the *test* frame is the interesting part.

`train_df` — six people, with roles Backend/Data Science/Management/QA, cities Bengaluru,
Pune, Hyderabad, Indore, Mumbai, Bhilai, and skills drawn from Python, SQL, JavaScript,
Java, Go.

`test_df` — three people, deliberately containing four things the training split never had:

| planted in the test split | what it exercises |
|---|---|
| skills `zig`, `rust`, `elixir` | unseen skills must not enter the vocabulary |
| role `Security`, education `Doctorate` | unseen categories must become `NaN`, not crash |
| city `Kota`, city `None`, city `"Bangalore "` (trailing space) | tier 3 vs `NaN`, and alias normalisation |
| `years_experience` of `np.nan` | missing must stay missing through every derived column |

#### The leakage test

```python
def test_fit_on_train_does_not_see_test_skills(train_df, test_df):
    skills = SkillFeatures(top_n=10).fit(train_df)

    test_only = {"zig", "rust", "elixir"}
    assert test_only.isdisjoint(skills.vocabulary_)

    before = list(skills.vocabulary_)
    out = skills.transform(test_df)
    assert skills.vocabulary_ == before
    assert "skill_zig" not in out.columns
```

Three assertions, and each one closes a different door:

1. **The vocabulary fitted on train contains no test-only skill.** If `zig` had turned up in
   `vocabulary_`, the fitted state would have been influenced by rows the model is supposed
   to be *evaluated* on, and the evaluation would be measuring memory rather than
   generalisation.
2. **Transforming the test set does not retroactively grow the vocabulary.** This is the
   subtle one. A transformer that quietly learned during `transform` would pass the first
   assertion and still leak. `before` is captured, `transform` is called, and equality is
   asserted.
3. **No `skill_zig` column appears in the output.** The matrix stays the shape the model was
   trained on.

`test_builder_fit_on_train_freezes_categories` is the same guarantee one level up: `Security`
is not in `builder.category_levels_["role"]`, `Doctorate` is not in the education levels, and
the transformed test frame's category list is *identical* to the frozen one — which is what
keeps the integer codes stable.

#### The deliberate contrast test — the best test in the file

```python
def test_fitting_on_everything_would_leak(train_df, test_df):
    """A demonstration, not just an assertion."""
    leaky = SkillFeatures(top_n=30).fit(pd.concat([train_df, test_df], ignore_index=True))
    assert "zig" in leaky.vocabulary_  # ← information from the test set, in the model's inputs
```

This test **asserts the bug appears**.

It takes the tempting shortcut — fit on train and test concatenated, which is what you do if
you have not thought about it — and demands that `zig` shows up in the vocabulary. If some
future refactor made this test fail, that would mean the leak had become impossible for a
different reason, and someone should find out why before deleting the test.

Why this is worth more than the passing test next to it: **the passing test documents the
rule; this one documents the failure mode.** Anyone reading the file learns what leakage
actually looks like when it happens, in six lines, rather than being told to avoid something
abstract. Almost no codebase does this.

Two honest observations. First, on these fixtures the contrast is entirely about *which rows
`fit` saw*, not about `top_n` — there are only eight distinct skills across both frames, so
`zig` would enter the vocabulary at `top_n=10` just as it does at `top_n=30`. The differing
`top_n` between the two tests is cosmetic. Second, the test asserts the bug exists but does
not measure its size. On this data it is two or three columns; the docstring in `skills.py`
is the thing that tells you the same mistake with target encoding would be catastrophic.

#### The rest of the file, grouped

**Other leakage-adjacent tests.**
`test_target_never_becomes_a_feature` asserts `salary_annual` is not in the feature matrix
("if the target leaks into X, the model scores near-perfectly and is useless").
`test_fairness_columns_never_become_features` asserts `gender` and `source` are absent.
`test_prev_salary_excluded_by_default_and_opt_in` asserts both halves of the flag.

**Unseen values must not crash.**
`test_unseen_category_becomes_nan_not_crash` checks that the `Security` row is `NaN` **and
that the Backend row in the same frame is still mapped correctly** — a transform that fell
over on one bad value and corrupted its neighbours would pass a weaker test.
`test_unseen_skill_is_ignored_but_still_counted` asserts `n_skills == [2, 3, 1]` while
`skill_python == [1, 0, 0]`: the person who listed Zig, Rust and Elixir gets no columns for
any of them, but the count knows they listed three.
`test_transform_works_on_a_frame_missing_columns_entirely` drops `city` and `skills` and
asserts the column list is unchanged, `location_tier` is all `NaN`, and `n_skills` is all
zero.

**Experience.** Monotonicity (`test_log_is_monotonic_in_years`, which insists on *strictly*
increasing — ties would make two different levels of experience indistinguishable to the
model); the `log1p(0) == 0` fresher case; and the one that encodes the whole reason for the
transform:

```python
def test_log_compresses_late_years_more_than_early_ones():
    v = log_years(pd.Series([1.0, 3.0, 18.0, 20.0])).tolist()
    assert (v[1] - v[0]) > 5 * (v[3] - v[2])
```

Year 1→3 must move the feature more than five times as much as year 18→20 does. That is the
salary curve from the top of `experience.py`, asserted in code.

Plus: absurd years become `NaN` rather than being clipped (parametrised over −1, −30, 51,
120, 2005); missing stays missing through all four derived columns; the bucket boundaries
are pinned one by one, including `(2, "2-3")` with the comment "left-closed: exactly 2
belongs to the second band"; and three juniors still produce all seven ordered categories.

**Skills parsing.** `parse_skills` against the mess real data contains — stray whitespace,
empty segments, `""`, `np.nan`, `None`. Vocabulary ordering and the alphabetical tiebreak.
And `skill_c++` surviving as a column name, because `+` and `#` are legal for LightGBM.

**Location.** Seven Bangalore spellings all mapping to tier 1, with the comment that real HR
exports contain every one of them in one column. Fifteen more city/tier pairs. And
`test_missing_city_is_nan_not_tier_3`, which is the test that encodes the honest-missingness
argument.

**Builder contract.** Row count and index preserved (including on a two-row slice);
`feature_names_` matching the actual output columns for both frames; missing values not
filled with zero; categoricals keeping `CategoricalDtype` and no `role_` dummy columns
appearing; every output column being numeric or categorical, so LightGBM never meets an
object dtype; the expected feature names all present with no duplicates; and `describe()`
containing the string `"EXCLUDED"` so the `prev_salary` warning stays visible.

**The real data.** `test_round_trip_on_the_real_survey` is decorated with
`@pytest.mark.skipif(not SURVEY_CSV.exists(), ...)`, so it is skipped when the 140MB CSV has
not been downloaded. It loads, shuffles, splits 80/20, fits and transforms, and asserts the
columns line up, the target is absent, `location_tier` is entirely `NaN` (stated as a fact
of this source, not a defect), the vocabulary contains `python`, `javascript` and `sql`, and
that experience is present for more than half the rows with finite logs where present.

Its docstring is the reason it exists: *"Hand-written fixtures agree with your assumptions by
construction; only real data disagrees with you."*

#### The one thing to understand here

A test suite can assert two different kinds of thing: that the code does what it does, and
that the *engineer* did what they should have. Most of this file is the first kind. Three
tests — the leakage test, the freeze test, and the deliberate contrast — are the second kind,
and they are the ones that would catch a well-meaning refactor six months from now.

#### Surprises and gotchas

- **`test_round_trip_on_the_real_survey` shuffles before splitting**, with an explicit
  comment that survey rows are not in random order. It uses a positional 80/20 cut rather
  than `split.random_split` — the features package does not import from `model`.
- **The fixtures contain a `gender` column** purely so the exclusion test has something to
  exclude.
- **`test_transform_before_fit_raises`** loops over all four unfitted transformers and
  requires the error message to contain `"fit"`, which is why all four raise
  `RuntimeError("call fit() before transform()")` rather than something more varied.
- **The count is 65 tests across the whole feature layer**, per `docs/JOURNEY.md` Part 7;
  this file holds most of them.

---

## Where to go next

`03-` and `04-` cover the remaining layers. If you want to see every number in this document
regenerated from scratch:

```bash
uv run python scripts/run_analysis.py
```

It is seeded, runs top to bottom in about half a minute, prints every figure in
`docs/findings.md`, and writes the charts to `reports/`. No number in that document appears
without the script printing it, which is the standard worth holding this code to.
