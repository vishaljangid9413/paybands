# 03 — Fairness, policy, and explanation

This part of the guide covers six files, in this reading order:

1. `src/paybands/fairness/audit.py` — measuring a pay gap, properly
2. `src/paybands/fairness/proxy.py` — why deleting the gender column does not work
3. `src/paybands/policy/increment.py` — the layer with no machine learning in it
4. `configs/policy/increment_fy_2026_27.yaml` — the policy those rules apply
5. `src/paybands/explain/shapley.py` — why *this* candidate got *this* number
6. `tests/test_fairness.py` — how we know the audit is not lying

**Read `01-payroll-and-data.md` and `02-features-and-model.md` first.** This part assumes you
already know what the synthetic data generator produces, what a `FeatureBuilder` is, and what
`SalaryBandModel` predicts. It will not re-explain them.

One orientation note before you start. The project has three layers, set out in `docs/design.md`
§2: Layer 1 is the model (learned), Layer 2 is the payroll calculator (arithmetic), Layer 3 is
decision rules (policy). The fairness module sits beside all three rather than inside any of them
— it takes plain arrays and audits whatever produced them. `increment.py` *is* Layer 3.
`shapley.py` explains Layer 1.

---

### `src/paybands/fairness/audit.py`

> Measures whether pay, or a model's predictions, treats one group worse than another — and
> proves that the measurement itself is trustworthy.

**Read time:** 45 minutes · **Difficulty:** hard
**Read it when:** you have read `data/synthetic.py` and understand that it injects a pay gap of a
size we choose. You do not need any statistics background; every statistical idea used here is
explained in the file's own comments, and again below.

#### What problem it solves

A model learns patterns from history. If a company has historically paid one group less for the
same work, the model learns that as a pattern and applies it to every new candidate. The output
then looks objective — it is a number, from a computer — but it is the old unfairness laundered,
and now harder to argue with because "the model said so".

So the project measures it. The hard part is not the arithmetic. The hard part is that there are
two completely different numbers people call "the pay gap", and confusing them is the most common
serious error in this whole subject.

#### The two gaps — read this before anything else in the file

**Raw gap.** Take everyone in group A, average their pay. Take everyone in group B, average
theirs. Report the difference as a percentage. That is all.

**A raw gap is not evidence of discrimination.** Groups genuinely differ in the things that pay
differently. If one group averages four years of experience and the other nine, they will earn
noticeably different amounts *at a perfectly fair employer*, and `raw_gap` will dutifully report a
large number. It has measured a real difference in **who the people are**, not in **how they are
treated**.

The test `test_raw_and_adjusted_gaps_disagree_when_the_groups_differ` builds exactly that world by
hand: everyone paid by the identical formula, one group averaging four years of experience and the
other nine. Raw gap: over 25%. Adjusted gap: zero. Publishing the first number as proof of bias
would be a false accusation about a demonstrably fair employer.

**Adjusted gap.** Same experience, same role, same city, same everything the company can
legitimately pay differently for — is the pay *still* different? That is the number that supports
a claim, and it is what `adjusted_gap` computes.

The raw gap is still worth computing, for two reasons. It is the number the outside world uses
(national "gender pay gap" headlines are raw gaps), so a company needs to know its own. And the
*distance between the two* is itself the finding: "of the 14.6% overall difference, 8.1 points
survive controlling for experience and role, and the rest is explained by the two groups doing
different jobs" is far more useful than either number alone.

#### Why the module is deliberately model-agnostic

Every function here takes plain arrays — actual salaries, predicted salaries, group labels, a
DataFrame of controls. Nothing in the file imports the band model. Even
`validate_on_synthetic` imports the data generator *inside the function body*, with a comment
explaining that the dependency must not exist at module level.

The reason is practical: the same audit has to score the lookup baseline, the gradient booster, a
consultant's spreadsheet, and a human recruiter's past offers. An audit that only works against
one model is an audit nobody will run on the thing that actually needs auditing.

#### Module-level constants

These are at the top of the file, in plain sight, because they are judgement calls rather than
facts.

| Constant | Value | What it is |
|---|---|---|
| `LEGITIMATE_CONTROLS` | 9 column names | Things the project accepts as a defensible reason for two people to be paid differently: `years_experience`, `role`, `education`, `org_size`, `remote`, `employment_type`, `location_tier`, `institute_tier`, `prev_company_type`. |
| `CONTESTED_CONTROLS` | `("career_gap_months",)` | Features that predict pay honestly *and* leak group membership. Whether to control for one is an argument, not a technical question. |
| `DEFAULT_MAX_AUTO_LEVELS` | `40` | A numeric control with at most this many distinct values becomes one dummy variable per value instead of entering the regression as a straight line. |

The comment above `LEGITIMATE_CONTROLS` is the most important prose in the file. The trap runs in
both directions:

- Control for too little and you report a gap that is really "one group has fewer years of
  experience". A false accusation.
- Control for too much and you explain the unfairness away. If one group is systematically pushed
  into lower-paying roles, then controlling for role subtracts exactly the discrimination you were
  looking for, and reports zero. A false all-clear — and the more dangerous error, because it is
  the comfortable one.

There is no formula that settles this. The code's answer is to state the control list out loud in
every report it prints and let people argue with it.

`DEFAULT_MAX_AUTO_LEVELS` deserves a word too. `years_experience` is a number, but pay does not
rise in a straight line with it — the first five years are worth far more than years 15 to 20.
Forcing a straight line through a curve leaves a pattern in what the regression could not explain,
and if that pattern correlates at all with group membership it lands on the group coefficient.
Measured on this project's synthetic data, that single mistake shifts the recovered gap by about
+0.3 percentage points. One dummy per value assumes nothing about the shape.

#### Classes

Every class here is a frozen dataclass — a plain container of numbers that cannot be modified after
it is created. Each one carries a `__str__` that turns itself into a readable report, so the
interpretation travels with the numbers.

##### `RawGap`

What two groups earn, before accounting for anything.

| Field | Type | Meaning |
|---|---|---|
| `disadvantaged` | `str` | The group under study. |
| `reference` | `str` | The group it is compared against. |
| `n_disadvantaged`, `n_reference` | `int` | How many people in each. |
| `mean_disadvantaged`, `mean_reference` | `float` | Average pay in each, rupees. |
| `median_disadvantaged`, `median_reference` | `float` | Median pay in each, rupees. |
| `mean_gap` | `float` | `1 - mean_disadvantaged / mean_reference`. `0.146` reads as "earns 14.6% less". Positive means the disadvantaged group earns less. |
| `median_gap` | `float` | The same thing on medians. |

Both mean and median are reported because salaries have a long right tail. A handful of very large
salaries drag the mean around; if they are unevenly spread across groups, the mean gap moves with
them and the median does not. If the two disagree sharply, that disagreement is itself the story.

`__str__` prints both, plus the group sizes, plus the line
`NOT evidence of unfair pay on its own — see the docstring.`

##### `AdjustedGap`

The like-for-like gap, with an honest interval around it.

| Field | Type | Meaning |
|---|---|---|
| `disadvantaged`, `reference` | `str` | As above. |
| `n` | `int` | Rows used in the fit (people in either of the two groups). |
| `n_disadvantaged` | `int` | Rows in the group under study. |
| `gap` | `float` | `1 - exp(coefficient)`. Positive = paid less. |
| `ci_low`, `ci_high` | `float` | The ends of the confidence interval, in the same units. |
| `confidence` | `float` | Which interval it is — `0.95` by default. |
| `coefficient` | `float` | The regression's raw output, in log units. Kept so the arithmetic can be checked. |
| `standard_error` | `float` | How much that coefficient would wobble if the world were re-run. |
| `p_value` | `float` | Two-sided, from Student's t. |
| `controls` | `tuple[str, ...]` | Exactly which columns were held fixed. |
| `n_parameters` | `int` | How many columns the design matrix had. |
| `r_squared` | `float` | Share of variation in `log(salary)` the fit explains. |

Properties: `is_significant` (does the interval exclude zero?) and `ci_width` (`ci_high - ci_low`).

The docstring on `is_significant` is worth quoting because it corrects a near-universal
misreading. "Significant" means only that the data would be surprising if the true gap were
exactly zero. It says nothing about whether the gap is *large*. A statistically significant 0.4%
gap is not a scandal, and a non-significant 9% gap on 60 employees is not an all-clear — it is a
sample too small to tell. Read the size first, then the interval.

##### `GroupResidual` and `ResidualCheck`

A **residual** is what the model got wrong for one person: actual minus predicted.

`GroupResidual` summarises the residuals for one group:

| Field | Meaning |
|---|---|
| `group` | The group name. |
| `n` | How many people. |
| `mean_log_residual` | Mean of `log(actual) − log(predicted)`. Positive means the model predicts **less** than these people actually earn. |
| `relative_bias` | The same thing as a percentage: `exp(mean_log_residual) − 1`. |
| `mean_rupee_error`, `median_rupee_error` | Mean and median of `actual − predicted` in rupees. |

`ResidualCheck` holds two of those plus the comparison between them: `difference` (as a
percentage), `ci_low`, `ci_high`, `confidence`, `p_value`, and an `is_significant` property.

##### `FairnessReport`

Everything the audit found, written for someone who does not do maths.

| Field | Meaning |
|---|---|
| `group_column` | Name of the protected attribute, for the header only. |
| `raw` | A `RawGap`. |
| `adjusted` | An `AdjustedGap`. |
| `residual` | A `ResidualCheck`, or `None` if no predictions were supplied. |
| `notes` | Extra caveats to print — e.g. which contested controls you chose to include, and why. |

The one property, `explained_by_controls`, is `raw.mean_gap - adjusted.gap`: how much of the
headline gap the legitimate controls account for.

`__str__` produces a numbered report with four parts: the headline gap (with the words "This
number on its own proves nothing"), the like-for-like gap with its interval and its control list,
optionally the model's own errors, and a closing "WHAT THIS IS NOT" section. That last section is
not decoration — `test_the_report_says_what_a_reader_needs_to_know` asserts the caveat strings are
present, because a fairness result handed over without its caveats gets repeated without them.

#### Functions

##### Plumbing

`_as_labels(group: ArrayLike) -> NDArray[np.str_]` — turns group labels into a plain string array,
so `1`, `1.0` and `"1"` cannot end up as three different groups.

`_two_groups(labels, salary, disadvantaged, reference) -> tuple[str, str]` — works out which two
values are being compared and in which order. Order matters because it fixes the *sign*: a
positive gap always means the disadvantaged group is paid less. If the caller names one and there
is exactly one other, the other is inferred. If the caller names neither and there are exactly
two, the **lower-paid group is chosen as disadvantaged**. That last behaviour is convenient and
dangerous — see the gotchas.

`_design_matrix(controls, *, categorical=None, max_auto_levels=DEFAULT_MAX_AUTO_LEVELS) ->
pd.DataFrame` — turns a frame of controls into the numeric matrix a regression can take. Text
columns become **dummy variables**: one 0/1 column per value, with the first value dropped.
Dropping one is not optional — keep them all and they sum to the intercept column, the matrix stops
being invertible, and the fit becomes arbitrary. The dropped value is the baseline every other
value is measured against. Numeric columns with at most `max_auto_levels` distinct values are also
dummied, for the curve reason above. An `intercept` column of `1.0` is inserted first. Missing
values are refused with an error rather than silently dropped, because dropping rows quietly
changes who the audit is about.

`_Fit` — a frozen dataclass holding `coefficients`, `standard_errors`, `n`, `n_parameters`,
`df_residual`, `r_squared`.

`_ols(design, y) -> _Fit` — ordinary least squares, written out in about fifteen lines rather than
imported from `statsmodels`. Four points to notice:

- Degrees of freedom use `matrix_rank(x)`, not the column count. If two controls happen to be
  perfectly collinear, the matrix is short of independent columns and using the count would
  understate the noise.
- If that leaves zero or fewer degrees of freedom, it raises. A fit with no residual left is
  perfect and meaningless.
- `np.linalg.pinv` rather than `inv`, so a rank-deficient design degrades instead of exploding.
- The **standard error** is the whole reason this is written out. It answers "if we re-ran the
  world, how much would this coefficient wobble?" It shrinks like `1/sqrt(n)`, which is why a gap
  measured on 200 people is a rumour and the same gap on 20,000 is a finding.

`_to_salary_array(salary) -> NDArray[np.float64]` — rejects empty input, NaN, infinity, and any
value at or below zero. The last one is not fussiness: the gap is measured on `log(salary)`, and
`log(0)` has no answer.

##### The three measurements

```python
raw_gap(
    salary: ArrayLike,
    group: ArrayLike,
    *,
    disadvantaged: str | None = None,
    reference: str | None = None,
) -> RawGap
```

In: one salary and one group label per person. Out: a `RawGap`. It computes four averages and two
divisions. The docstring is five times the length of the code, and that ratio is correct — the
danger in this function is entirely in how the answer is read.

```python
adjusted_gap(
    salary: ArrayLike,
    group: ArrayLike,
    controls: pd.DataFrame,
    *,
    disadvantaged: str | None = None,
    reference: str | None = None,
    confidence: float = 0.95,
    categorical: Sequence[str] | None = None,
    max_auto_levels: int = DEFAULT_MAX_AUTO_LEVELS,
) -> AdjustedGap
```

**The number.** It fits one regression:

```
log(salary) = intercept
            + β × [is in the disadvantaged group]
            + (a coefficient for every control)
            + unexplained leftover
```

and reads off `β`. Because the controls are in the same equation, `β` is what is left *after*
experience, role, location and the rest have been given every chance to explain the difference.

Three details that are easy to get wrong, all of them in the docstring:

**Logs, not rupees.** Pay differences are multiplicative. People say "8% less", never "₹1.7 lakh
less", because ₹1.7 lakh means very different things at ₹6L and at ₹60L. Taking logs turns one
consistent percentage into one consistent number, which is the only form a single coefficient can
take.

**`1 - exp(β)`, not `-β`.** A coefficient of −0.0834 is an 8.0% cut, not an 8.3% one. Check it:
`exp(-0.0834) = 0.9200`, and `1 - 0.9200 = 0.0800`. At small gaps the two conventions nearly agree,
which is exactly why the mistake survives unnoticed until somebody checks against a known answer.
(The synthetic generator uses `log1p(-pay_gap)` on the way in, so the two match by construction.)

**An interval, not a point.** On 300 employees the estimate wobbles by several percentage points
from sample to sample. There is deliberately no option to get the point estimate without the
interval.

Notice also what the function refuses to do:

- If any control column is *identical* to the group labels, it raises. Controlling for the
  protected attribute removes the very comparison you are trying to make, and it is one typo away
  in any pipeline that builds a column list programmatically.
- If adding the group indicator does not increase the matrix rank, it raises. That means the
  controls already determine group membership — no two people are alike in every control but
  different in group — so "same job, same experience, different pay" has no examples. Without this
  check the linear algebra still returns a number, picked arbitrarily from infinitely many equally
  good answers. On the test's example that number was **−76%**, and somebody would have reported
  it.

`stats.t.ppf` (Student's t) rather than the normal distribution: with few rows the extra width in
the tails is exactly the humility a small sample deserves, and the two converge above a few hundred
rows so it costs nothing when data is plentiful.

One subtlety in the return: because `gap = 1 - exp(β)` flips the ordering, the *interval ends swap
over*. `ci_low` is built from `beta_high` and vice versa. The code says so in a comment.

```python
residual_analysis(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    group: ArrayLike,
    *,
    disadvantaged: str | None = None,
    reference: str | None = None,
    confidence: float = 0.95,
) -> ResidualCheck
```

Does the model miss in a *different direction* for one group? Residuals should scatter evenly
around zero. If they average positive for one group and negative for another, the model is
systematically cheaper about one kind of person.

Worked in log space for the same reason the gap is: a ₹2L miss is a disaster at ₹6L and a rounding
error at ₹60L, so an average of rupee misses is dominated by whoever earns most.

It uses **Welch's t-test**, which does *not* assume the two groups have the same spread. They
rarely do — a model is usually more certain about the group it has most examples of — and the
equal-variance version would report a falsely narrow interval exactly when the groups are most
lopsided.

**This test and `adjusted_gap` catch different things, and you need both.** A model trained
faithfully on unfair data passes *this* test perfectly: it learned the unfairness, so it predicts
the unfair salaries accurately, so its residuals are beautifully balanced — while it quietly
recommends lower offers for one group forever. Residual balance is not fairness. What this catches
instead is bias the *model* added on top: too few rows from a group to learn their pattern, a loss
function dominated by the majority, a proxy leaned on too hard.

`tests/test_fairness.py` states that trap as executable code in
`test_a_model_that_learned_the_bias_perfectly_passes_the_residual_test`.

##### The convenience wrapper

```python
audit(
    salary: ArrayLike,
    group: ArrayLike,
    controls: pd.DataFrame,
    *,
    y_pred: ArrayLike | None = None,
    group_column: str = "gender",
    disadvantaged: str | None = None,
    reference: str | None = None,
    confidence: float = 0.95,
    categorical: Sequence[str] | None = None,
    notes: Sequence[str] = (),
) -> FairnessReport
```

Runs all three checks and packages them. One detail worth noticing: it computes `raw_gap` first and
then **pins** `raw.disadvantaged` and `raw.reference` into the other two calls, so all three
sections talk about the same group in the same order. Without that, a sign flip between sections
would read as a contradiction.

If `y_pred` is `None`, no residual section is computed and the report simply omits it. Silence
beats a placeholder.

##### The function that makes the rest trustworthy

```python
validate_on_synthetic(
    injected_gaps: Sequence[float] = (0.0, 0.03, 0.08, 0.15, 0.25),
    *,
    n: int = 10_000,
    seed: int = 42,
    control_for_proxy: bool = True,
    confidence: float = 0.95,
    generator: Callable[..., tuple[pd.DataFrame, object]] | None = None,
) -> pd.DataFrame
```

Everything above is a measuring instrument, and **an unvalidated measuring instrument is just an
opinion with decimal places**.

Here is why this matters more than it might sound. On real company data you *cannot* validate a
bias detector. Nobody knows the true pay gap. If the audit reports 3%, you have no way to tell
whether the gap really is 3%, or the gap is 9% and the audit is broken. Both look identical from
the outside. A wrong answer and a right answer are indistinguishable.

So the project generates a world where it wrote the rules — `paybands.data.synthetic.generate`
pays one group exactly `X` less, by construction — runs the audit against it, and demands `X`
back. That is calibrating a thermometer in boiling water before trusting it on a patient.

The result, from `docs/findings.md` §5.2:

| injected | recovered | 95% interval | truth inside? |
|---|---|---|---|
| **0%** | **−0.34%** | −1.39% to +0.71% | yes |
| 3% | 2.67% | 1.65% – 3.68% | yes |
| 8% | 7.69% | 6.72% – 8.65% | yes |
| 15% | 14.71% | 13.81% – 15.60% | yes |
| 25% | 24.75% | 23.95% – 25.53% | yes |

**The most important row is the first.** With zero real bias injected, the audit reports −0.34% —
it does not invent bias in fair data. An audit that cries wolf is worse than no audit at all,
because it is confidently wrong in the direction that gets somebody accused.

Note that it reads about 0.25 to 0.34 percentage points *low* across all five rows. That is a
small, consistent, honest bias, and the test suite bounds it (`error.abs().max() < 0.005`) rather
than pretending it is zero.

The returned DataFrame has one row per injected gap with columns `injected`, `recovered`,
`ci_low`, `ci_high`, `error`, `truth_inside_ci`, `n`. The real pass/fail column is
`truth_inside_ci` — a correct interval contains the true value about `confidence` of the time.

The `control_for_proxy` switch is the second lesson, and it leads directly into the next file.
Career breaks in the synthetic data both depress pay *and* signal group membership. Control for
them (`True`) and the audit recovers the injected gap. Leave them out (`False`) and every estimate
comes back 2.2 to 2.9 points too high, with **0 of 5** intervals containing the truth — because
part of the unfairness is flowing through the career-break channel and is being attributed to
gender directly. Neither answer is wrong; they answer different questions. Controlling for career
breaks says *"time out of the workforce is a legitimate reason to pay less"*. Not controlling says
*"who takes that time off is itself the unfairness"*. Both are arguable positions and the project
publishes both tables.

Note the `disadvantaged=truth.disadvantaged_group` argument inside the loop, with its comment about
"reading the answer sheet". This is essential. If validation let the audit auto-detect the
disadvantaged group, the zero-gap row would report a positive gap **by construction** and the most
important test in the file could not fail.

#### The one thing to understand here

A raw gap and an adjusted gap are different quantities and only the second one supports a claim
about fairness. And an unvalidated bias detector tells you nothing at all, because on real data a
broken detector and a working one produce output that looks identical — which is why
`validate_on_synthetic` exists and why it is the function to run first.

#### Surprises and gotchas

- **Auto-detection makes the gap positive by construction.** If you call `raw_gap` or
  `adjusted_gap` without naming `disadvantaged`, the lower-paid group is chosen automatically. The
  result can tell you a gap's *size*, but never that a gap *exists* — even on perfectly fair data
  it returns a positive number. `test_auto_detection_picks_the_lower_paid_group` pins this down
  deliberately. Name the group whenever you plan to quote the result.
- **`explained_by_controls` can go negative.** It is `raw.mean_gap - adjusted.gap`, and the
  adjusted gap can exceed the raw gap when a proxy is left out of the controls. The report will
  then print a negative "Accounted for by the controls" figure. That is arithmetically honest but
  it reads oddly, and nothing in the code flags it.
- **`audit()` accepts `categorical` but has no `max_auto_levels`.** The wrapper forwards
  `categorical` to `adjusted_gap` and leaves `max_auto_levels` at its default. If you need to
  override the 40-level threshold, call `adjusted_gap` directly.
- **The controls list is a judgement call, not a constant.** `LEGITIMATE_CONTROLS` looks like
  configuration but it is an argument the project is making. Change it and every number changes.
- **A significant result is not a large result.** See the `is_significant` docstring. This is the
  single most-misread output in the file.
- **A regression coefficient is not a cause.** The docstring says it plainly: the honest output is
  "X% unexplained by what we recorded", which is a reason to go and look, not a verdict.
- **In production, this audit has nothing to point at.** JOURNEY Part 9 records that all 172
  columns of the Stack Overflow survey were checked for `gender`, `ethnic`, `race`, `sexuality`,
  `disability`, `nationality`, `religion`, `caste` and more. Every one returns zero — the questions
  are no longer asked. So the audit is provably correct and currently unusable on real data. The
  only demographic available is `Age`, where the raw gap is 43.9% and the adjusted gap is −5.22%
  with an interval from −33.6% to +17.1%. The honest reading of a 51-point interval is "we cannot
  tell", which is a different sentence from "there is no gap".

---

### `src/paybands/fairness/proxy.py`

> Demonstrates, with a controlled experiment, that removing the gender column from training data
> does not remove the bias from the model.

**Read time:** 30 minutes · **Difficulty:** medium
**Read it when:** immediately after `audit.py` — it imports six things from it, including two
private helpers.

#### What problem it solves

There is a claim you will hear in almost every discussion of fair machine learning, and in real
compliance documents at real companies:

> *"We removed gender from the training data, so the model cannot be biased."*

It is wrong, it is widespread, and it is comfortable — which is the worst combination a false
belief can have, because nobody goes looking for the evidence against it.

Here is why it fails. A model does not need the column it was denied. It needs *anything
correlated with it*. Career breaks, part-time history, a gap between graduation year and first
job, which schools, which job titles, which neighbourhood a postcode implies. Each is an
innocent-looking feature that carries information about the protected attribute, and a model with
enough of them reassembles the pattern from the pieces. Those features are called **proxies**.

Worse: removing the protected column makes the bias *harder to find*, not smaller. With gender in
the model you can read its coefficient. Without it, the same effect is smeared across six features,
none of which looks suspicious on its own.

**This file is the most important demonstration in the project.** Not because the code is clever —
it is about 200 lines of real logic — but because it converts an assertion into evidence.

#### The result

Same model, fitted twice on synthetic data carrying an 8% injected gap. Once **with** the gender
column, once **without**. The gap is then measured on held-out predictions, controlling only for
the legitimate controls. From `docs/findings.md` §5.3:

| seed | gap in the data | predictions **with** gender | predictions **without** | surviving | accuracy cost |
|---|---|---|---|---|---|
| 0 | 10.8% | 10.5% | 3.6% | 34.3% | +2.3% |
| 1 | 10.0% | 9.4% | 3.8% | 41.0% | +0.7% |
| 2 | 9.3% | 11.4% | 4.5% | 39.8% | +0.8% |

**Mean 38.4% of the gap survives deleting the protected column (range 34.3% – 41.0%).**

And the accuracy cost of removing gender was **1.3% on average**. That last figure is what makes
the result worse rather than better. If blinding the model had wrecked its accuracy, at least there
would be a trade-off to argue about. There isn't one. The proxies carry the same information, so
the model keeps its predictive power *and* the disadvantage. **Nothing meaningful was even traded
away for the harm that remains.**

#### Why it survives

The bias travels through `career_gap_months`. In the synthetic world, career breaks are 45% likely
in one group and 6% in the other, and each year out costs 7% of pay. So the column both predicts
pay honestly *and* smuggles group membership.

The correlation scan finds it before any model is trained:

| feature | kind | association | strength |
|---|---|---|---|
| `career_gap_months` | numeric | **0.367** | strong |
| `salary_annual` | numeric | 0.103 | weak |
| `prev_company_type` | categorical | 0.025 | negligible |
| everything else | | 0.004 or less | negligible |

One channel, in a dataset built with exactly one channel. **A real dataset has several and nobody
labels them.** That is precisely why this scan should be the first thing pointed at company data,
before a line of modelling.

#### The negative control — what makes this an experiment rather than an anecdote

`test_switching_the_proxy_off_removes_the_survival` runs the same comparison on data where both
groups have the same 6% chance of a career break:

```python
df, truth = generate(N, seed=4, pay_gap=0.08, career_gap_prob_disadvantaged=0.06)
assert not truth.has_proxy

result = compare_with_without(df, "gender", disadvantaged=truth.disadvantaged_group)
assert result.gap_with_protected > 0.06
assert abs(result.gap_without_protected) < 0.02
```

With the proxy channel switched off, the surviving gap **collapses to roughly zero**. The model
with no gender column has nothing left to rebuild the gap from.

This is the step most demonstrations skip, and it is what turns "the gap survived" into "the gap
survived *because of the proxy*". Without it, the headline test could be passing for some entirely
unrelated reason and nobody would know.

There is a second control too: `test_the_proxy_result_is_not_an_artefact_of_a_linear_model` reruns
the whole thing with LightGBM instead of the plain least-squares default. Same conclusion. A reader
could reasonably suspect a result from a linear model is an artefact of linear algebra; gradient
boosting reconstructs the gap just as happily, because the information is in the data rather than
in the algorithm.

#### Classes

##### `ProxyScan`

The output of the reconnaissance step.

| Field | Meaning |
|---|---|
| `protected_column` | Which attribute everything was scored against. |
| `n` | Rows in the frame that was scanned. |
| `table` | A DataFrame with columns `feature`, `kind`, `association`, `strength`, `n_unique`, sorted by association, strongest first. |

Methods: `strongest(k=5)` returns the top `k` rows; `leaking(threshold=0.15)` returns the feature
*names* at or above a threshold; `__str__` prints the whole table with the legend "0 = tells you
nothing about the group, 1 = tells you exactly".

##### `_Regressor` (a `Protocol`)

Not a class you instantiate — it is a type declaration saying "anything with `fit(x, y)` and
`predict(x)`". Kept so `compare_with_without` can be handed an `LGBMRegressor`, a `Ridge`, or the
plain fit below without caring which. The demonstration should not depend on the model, and saying
so in the type system is cheaper than saying it in a comment.

##### `LeastSquares`

A linear model in six lines, and the deliberate default.

| Field | Meaning |
|---|---|
| `coefficients` | `None` until `fit` is called, then the solved weights. |

`fit(x, y)` calls `np.linalg.lstsq` and returns `self`. `predict(x)` returns `x @ coefficients`,
raising `RuntimeError("fit the model before predicting")` if called first.

Why the default is the plainest possible model: if the proxy demonstration only worked with
gradient boosting, a reader could reasonably suspect the result was an artefact of complexity.

##### `ProxyComparison`

The headline result.

| Field | Meaning |
|---|---|
| `protected_column` | The column removed in model B. |
| `proxy_columns` | Which proxy features were included in both models' inputs. |
| `n_train`, `n_test` | Split sizes. |
| `gap_in_data` | An `AdjustedGap` — the gap actually present in the held-out salaries, with an interval. |
| `gap_with_protected` | The gap in **model A's predictions**, a bare float. |
| `gap_without_protected` | The gap in **model B's predictions**, a bare float. |
| `mae_with_protected`, `mae_without_protected` | Mean absolute rupee error of each model. |

Properties: `surviving_share` = `gap_without_protected / gap_with_protected` (the headline
fraction), and `accuracy_cost` = `mae_without_protected / mae_with_protected - 1.0`.

Note the deliberate asymmetry: only the gap in the *real* salaries gets a confidence interval. The
other two are gaps in numbers a model produced, so an interval there would describe sampling noise
in a deterministic function — a real number with no real meaning. Point estimates only, on purpose.

`__str__` prints the whole comparison and then six lines of prose ending "…'We removed the
protected column, so the model is fair' is not a weak claim; it is a false one, and this is the
evidence."

#### Functions

##### Measuring association

The scan has to put numbers and categories in the same column of the same table, so it uses two
different measures, both scaled 0 to 1.

`_correlation_ratio(values, labels) -> float` — for numeric features. Known as **eta** (η). Split
people by group and ask what share of the spread in this feature is the gap *between* the group
averages, rather than the spread *within* each group. 0 means the groups look identical on this
feature; 1 means the feature tells you the group exactly. For two groups this is the same number as
the absolute correlation with a 0/1 indicator, but written this way it also works for three groups
or ten, so the audit needs no separate code path for a non-binary attribute.

`_chi_squared(table) -> float` — Pearson's chi-squared statistic, written out in four lines because
the idea fits in one sentence: compare what you observed against what independence would have
produced, square the differences, and scale each by how big it was expected to be.

`_cramers_v(feature, labels) -> float` — for categorical features. Built on chi-squared, scaled by
the table's size so a 2×2 result and a 9×3 result can sit in the same column. It applies the
**Bergsma–Wicher bias correction**, because chi-squared drifts upward as the table gets bigger —
without it, a job-title column with 200 values would score higher than a genuine proxy purely for
being sprawling.

`_strength_label(association) -> str` — maps a number to `strong` (≥ 0.30), `moderate` (≥ 0.15),
`weak` (≥ 0.05) or `negligible`. These bands are judgement calls, printed rather than hidden so a
reader can disagree with the wording instead of misreading a bare 0.31.

##### The scan

```python
proxy_correlation(
    df: pd.DataFrame,
    protected_col: str,
    candidate_cols: Sequence[str] | None = None,
    *,
    max_auto_levels: int = 40,
) -> ProxyScan
```

In: a DataFrame and the name of the protected column. Out: a `ProxyScan` ranking every other
column by how much it reveals about that attribute.

This is the reconnaissance step and it costs one line. Run it *before* training, on whatever data
you have. Every feature that scores high is a route the bias can travel down after the protected
column is dropped — which is why "we don't collect gender" is not a fairness strategy but a
measurement problem: you have removed your ability to *check*, not the effect.

**What a high score does and does not mean.** It means the feature carries information about the
group. It does *not* mean the feature is illegitimate. Experience carries information about age,
and experience genuinely belongs in a pay model. The scan tells you where to look, not what to
delete. Deleting every correlated feature usually leaves a model that cannot predict pay at all.

The docstring is honest about a purist objection: η and Cramér's V are not the same quantity, and
ranking them against each other is not strictly defensible. They are close enough in spirit to sort
a shortlist, and a shortlist is what this is for.

##### The demonstration

```python
compare_with_without(
    df: pd.DataFrame,
    protected_col: str = "gender",
    *,
    target_col: str = "salary_annual",
    legitimate_controls: Sequence[str] | None = None,
    proxy_cols: Sequence[str] | None = None,
    test_size: float = 0.3,
    seed: int = 0,
    model_factory: Callable[[], _Regressor] | None = None,
    disadvantaged: str | None = None,
    reference: str | None = None,
) -> ProxyComparison
```

The procedure, and every step matters:

1. **Split** into training and held-out parts. The gap is measured on rows the models never saw, so
   nobody can say the effect is memorisation.
2. **Fit model A** on the legitimate controls, the proxy columns, *and* the protected column.
3. **Fit model B** on exactly the same thing minus the protected column — the "we removed gender,
   so we're fine" model.
4. **Convert both sets of predictions back into rupees** and run `adjusted_gap` over them,
   controlling for the **legitimate controls only**.

**Step 4 is the one people get wrong.** We do not control for the proxy when measuring the gap. If
we did, we would be subtracting the very channel we are trying to expose, both models would report
a clean zero, and the audit would confirm the false claim instead of breaking it. That is the same
mistake in a different costume: a fairness number adjusted until it is comfortable.

Two implementation details worth noticing in the body:

- Both feature matrices are built from the **whole frame** and then split by row. Encoding train
  and test separately is a classic silent bug: a category appearing only in the test set would
  produce a different set of dummy columns there, the columns would stop lining up with the fitted
  coefficients, and the predictions would be confidently meaningless.
- Both models are fitted on `log(salary)` and their predictions are exponentiated back to rupees
  before the gap is measured — consistent with everything else in the project.

**Is this fair to the model?** The docstring asks the right question and answers it honestly.
Career breaks really do reduce pay in this data, so part of what model B reproduces is a genuine
effect, not laundered gender. That is precisely why this is hard, and precisely why "delete the
column" is the wrong tool: **the harmful and the legitimate arrive through the same wire**. The
answer is not a cleverer feature list. It is to measure the outcome, and then make a human decision
about a career-break penalty that lands on one group nine times out of ten.

#### The one thing to understand here

Removing a protected column does not remove the disadvantage; it removes your ability to see it.
The model rebuilds most of the gap from correlated features, at almost no cost to its accuracy, and
this file proves it with a positive result, a negative control, and a second algorithm.

#### Surprises and gotchas

- **Model A does not get the raw gender column.** It gets a 0/1 indicator, `(labels == low)`,
  appended to the design matrix. Equivalent for two groups, but not the same thing as one-hot
  encoding a multi-valued column.
- **`proxy_correlation` scans the target column too.** Called without `candidate_cols`, it scores
  every column except the protected one — including `salary_annual`, which shows up at 0.103. That
  is the raw gap wearing a different hat, not a proxy. Pass an explicit `candidate_cols` list when
  you want a clean feature scan (the tests do).
- **`max_auto_levels` is duplicated.** `proxy_correlation` hard-codes `40` rather than importing
  `DEFAULT_MAX_AUTO_LEVELS` from `audit.py`, even though its docstring explicitly says it is
  matching that function's encoding. Change one and the other silently disagrees.
- **`surviving_share` returns `nan`** when `gap_with_protected` is exactly zero. Only reachable in
  contrived data, but it is a float and not an error.
- **Proxy columns are excluded if they are already controls.** `proxies` is built with
  `c not in controls`, so if you pass `legitimate_controls` that includes `career_gap_months`, the
  proxy list quietly empties and the demonstration measures nothing.
- **This module imports private helpers** (`_as_labels`, `_design_matrix`, `_two_groups`) from
  `audit.py`. That is a deliberate coupling within one package, but it means changes to those
  helpers can break this file without any public API changing.

---

### `src/paybands/policy/increment.py`

> Turns a predicted band, a current salary and a performance rating into a recommended increment
> with a sentence explaining it — using rules from a config file, and no machine learning at all.

**Read time:** 40 minutes · **Difficulty:** medium
**Read it when:** after you understand what `SalaryBandModel` produces. You do not need to have
read the fairness module first; this file is independent of it.

#### What problem it solves

Once you can predict what the market pays for a job, the obvious next question is: what should we
actually *do* about the people we already employ? Who is underpaid? How big should this year's
raise be? What happens when the recommendations cost more than the budget?

**There is no machine learning in this file, and that is the point.**

It is tempting to build a second model for increments: feed it last year's raises, let it learn
what a raise looks like. Don't. An increment is not a pattern hiding in data — it is a *decision*.
How big is the pool? What is a rating of 4 worth in rupees? Do we fix somebody's underpayment this
year or over three?

Those are choices a company makes. A model trained on last year's choices would do nothing but
repeat them, including the ones that were wrong. Worse, it would repeat them *unarguably*. A rule
says "8% because your rating was 3, plus 5% because you sit at 0.86 of band". A model says "13%".
Only one of those can be challenged by the person it is about, and being challengeable is the whole
job here.

**Knowing when *not* to use machine learning is part of being good at machine learning.**

#### Compa-ratio — learn this term

```
compa-ratio = actual salary ÷ midpoint of the predicted band
```

That is it. One division. "Compa" is short for "comparative"; it is standard compensation
vocabulary and using it correctly makes the tool sound like it was built by somebody who
understands compensation rather than somebody who understands pandas.

| compa-ratio | meaning | what this module does |
|---|---|---|
| below 0.90 | paid below band, a genuine flight risk | equity correction on top of merit |
| 0.90 – 1.10 | in band | merit only |
| above 1.10 | paid above band | merit damped (halved) |
| above 1.25 | well above band | held — no increase this year |

A compa-ratio of 0.86 means you are paid 86% of the going rate for your job.

This one number drives both of the project's use cases. **Pay equity** is `equity_review`:
everyone below 0.90, worst first, with a price tag. **Increments** are `recommend`: merit from the
rating, plus equity from the compa-ratio, capped, then squeezed into the budget by
`apply_budget_constraint`.

#### Classes — the config models

The top third of the file is pydantic models mirroring the YAML exactly. They all inherit from
`Frozen`, which sets `frozen=True` (cannot be modified after loading) and `extra="forbid"` (an
unrecognised key in the YAML is a loud error at load time, not a silently ignored line that costs
somebody their raise).

`BudgetPriority` is a `Literal["equity_first", "merit_first", "pro_rata"]` — a type that only
accepts one of three exact strings.

| Model | Fields | Validation |
|---|---|---|
| `BudgetRules` | `pct_of_payroll` (0–1), `priority` | — |
| `MeritRules` | `by_rating: dict[int, float]`, `on_missing_rating` (`"assume_default"` or `"no_merit"`), `default_rating` | `_check_table`: table must be non-empty, no negative percentages, and `default_rating` must exist in the table if it will be used. |
| `CompaRules` | `below_band`, `above_band`, `hold_above`, `above_band_merit_multiplier` (0–1) | `_check_order`: `below_band < above_band <= hold_above`. |
| `EquityRules` | `target_compa_ratio`, `close_gap_over_years` (≥1), `max_single_year_correction_pct` | — |
| `PromotionRules` | `uplift_pct` | — |
| `LimitRules` | `max_total_increase_pct`, `round_down_to_nearest` (default 100) | — |
| `IncrementRules` | `financial_year`, the six blocks above, plus `source`, `approved_by`, `approved_on` | `_check_target_is_in_band`: the equity target must not be below `below_band`. |

That last validator earns its place. Correcting somebody to a target that is *still* below the band
would be busywork — they stay on the equity list forever, at increasing cost.

The three provenance fields (`source`, `approved_by`, `approved_on`) are never used in any
calculation. They exist because a policy that has forgotten who approved it is a policy nobody can
defend when it is questioned.

`IncrementRules` has two methods:

- `from_yaml(path: str | Path) -> IncrementRules` — loads and validates a config file.
- `budget_from_payroll(total_annual_payroll: float) -> float` — the pool in rupees. The config
  stores a *percentage* rather than an amount so the number survives headcount changes: 10% of
  payroll is still 10% of payroll after twenty people join.

#### Classes — the outputs

##### `IncrementRecommendation`

One person's recommended increment, and why.

| Field | Meaning |
|---|---|
| `current_salary` | Annual base today, rupees. |
| `band_midpoint` | The midpoint of the predicted band for this person's job. |
| `merit_amount` | The merit component, **in rupees**. |
| `equity_amount` | The equity correction, in rupees. |
| `promotion_amount` | The promotion uplift, in rupees. |
| `rating` | The rating as supplied — `None` means none was on file. |
| `rating_used` | The rating after the missing-rating policy was applied. |
| `rules` | The `IncrementRules` this was computed under. Marked `repr=False`. |
| `employee_id` | Carried through, never used in the maths. |
| `rating_assumed` | Flag: a default rating was substituted. |
| `merit_damped` | Flag: merit was multiplied down because pay is above band. |
| `held` | Flag: no increase at all, pay is above `hold_above`. |
| `capped_by_total_limit` | Flag: the single-year total cap bit. |
| `capped_by_correction_limit` | Flag: the equity slice hit its ceiling. |
| `budget_reduced` | Flag: `apply_budget_constraint` took money away. |

**Only three numbers are stored: the three component amounts.** Everything else is a
`@property` computed on demand:

| Property | Formula |
|---|---|
| `recommended_amount` | `merit + equity + promotion` |
| `recommended_pct` | `recommended_amount / current_salary` |
| `new_salary` | `current_salary + recommended_amount` |
| `compa_ratio_before` | `current_salary / band_midpoint` |
| `compa_ratio_after` | `new_salary / band_midpoint` |
| `band_label` | `band_label(compa_ratio_before, rules)` — the words "below band" / "in band" / "above band" |
| `reason` | The English sentence |

##### The design detail worth stealing: `reason` is derived, not stored

This is the single most transferable idea in the file, so it is worth being explicit about it.

The obvious design is to compute the amount and the explanation together and store both. Then
`apply_budget_constraint` trims the amount — and the explanation, sitting in a string field, keeps
saying whatever it said before. The classic bug is a recommendation reading "12%" next to an amount
that is actually 9% after a budget pass.

Here, `reason` is a property. It reads the three component amounts every time it is called and
rebuilds the sentence from them. When the budget pass replaces `merit_amount`, the sentence rewrites
itself automatically. **Text and figures cannot drift apart, structurally.**

The mechanism that makes it safe is that the dataclass is `frozen=True`, so nobody can assign to
`merit_amount` without going through `dataclasses.replace`, which builds a whole new object. There
is nowhere for a stale string to hide.

`as_row()` returns a flat dict — including `reason`, freshly computed — for building a DataFrame or
an API response. `__str__` prefixes the reason with the employee id if there is one.

##### `EquityReview`

| Field | Meaning |
|---|---|
| `frame` | A DataFrame of only the people below the threshold, worst compa-ratio first. |
| `n_reviewed` | How many people were looked at in total. |
| `threshold` | The compa-ratio below which somebody counts as underpaid (0.90). |
| `target` | Where the policy aims to land them (0.95). |
| `total_cost_to_target` | What it would cost to bring everyone to target, all at once. The honest total — the size of the problem. |
| `total_cost_this_year` | What this year's slice costs under the phasing and cap rules. The number that has to fit in an actual budget. |

Properties `n_below`, `share_below`, `worst_compa_ratio`. `summary()` produces the paragraph that
goes at the top of a slide, and handles the empty case ("Nothing to correct") separately.

The docstring is right about why this output matters: a model that predicts bands is interesting; a
list saying "these eleven people are underpaid, here is who is worst, and it costs ₹34.2L to fix"
is a thing a Head of People takes into a budget meeting.

##### `BudgetAllocation`

| Field | Meaning |
|---|---|
| `recommendations` | The (possibly trimmed) recommendations, as a tuple. |
| `budget` | The pool, in rupees. |
| `requested` | What the untrimmed recommendations added up to. |
| `allocated` | What was actually handed out. |
| `priority` | Which allocation rule was used. |

Properties `shortfall` (`max(0, requested - allocated)`), `n_reduced` (how many recommendations
carry `budget_reduced=True`), and `utilisation` (`allocated / budget`).

#### Functions

##### Small helpers

`_lakhs(amount: float) -> str` — rupees in the unit Indians actually speak: `₹1.6L`, `₹1.24Cr`,
`₹48,000`. Deliberately *not* `model.metrics.format_rupees`, which spells the number out in full
(`₹1,60,000`). That is right for a table; this is right for a sentence read at speed. Below one
lakh, Indian and Western digit grouping agree, so plain formatting is safe there.

`_pct(fraction: float) -> str` — `0.08` → `"8%"`, `0.054` → `"5.4%"`. Drops the decimal when it is
a clean whole number, because "8.0% merit" reads like a machine wrote it.

`_floor_to(amount: float, step: float) -> float` — rounds **down** to a multiple of `step`. Down,
never nearest: rounding up can push a set of recommendations past a budget they had just fitted
inside.

`_band_midpoint(band: Any) -> float` — pulls one midpoint out of whatever the caller had to hand.
Accepts a plain number, or anything with a `.median` attribute (which is what
`model.band.BandPrediction` carries). Duck-typed on purpose: this module must not import Layer 1,
both because the dependency runs the wrong way (policy should not need LightGBM installed) and
because a consultant's spreadsheet of midpoints is a perfectly valid input.

`_midpoints(bands: Any, n: int) -> list[float]` — the same idea for a whole population, with a
length check.

`band_label(compa_ratio: float, rules: IncrementRules) -> str` — the public one-liner turning a
ratio into "below band" / "in band" / "above band", reading thresholds from the config. Note the
existence of `model.band.compa_label`, which says the same thing against module constants — this
one reads the config, and the config is the authority for anything that spends money.

`_resolve_rating(rating, rules: MeritRules) -> tuple[int | None, float, bool]` — returns
`(rating_used, merit_pct, was_assumed)`. A rating that is *present but not in the table* is a data
error, not a missing value, so it raises rather than guessing. Guessing would silently pay somebody
the wrong amount.

##### The rule itself

```python
recommend(
    current_salary: float,
    band: Any,
    performance_rating: int | None,
    rules: IncrementRules,
    *,
    is_promotion: bool = False,
    employee_id: object | None = None,
) -> IncrementRecommendation
```

The order of operations is itself policy, and the code follows it in five numbered blocks:

1. **Merit** from the rating. If `compa > hold_above` (1.25), merit is zeroed and `held` is set.
   Otherwise, if `compa > above_band` (1.10), merit is multiplied by
   `above_band_merit_multiplier` (0.5). Rewarding performance is fine; paying 18% more to somebody
   already 25% over the market rate is how a band stops meaning anything.
2. **Equity**, only if `compa < below_band` (0.90). Compute the distance to the target salary as a
   percentage of current pay, divide by `close_gap_over_years`, and cap at
   `max_single_year_correction_pct`.
3. **Promotion** uplift if `is_promotion`.
4. **Cap the total** at `max_total_increase_pct`. When the cap bites, the room is handed out
   equity first, then promotion, then merit — so merit is trimmed first and the equity correction
   is protected last. Same reasoning as the budget priority: a mistake the company made outranks a
   reward it is choosing to give.
5. **Round down** every component to a multiple of `round_down_to_nearest`, so the total can never
   creep upward.

**Worked example — below band.** ₹12,00,000 against a ₹14,00,000 midpoint, rating 3.

```
compa            = 1,200,000 / 1,400,000 = 0.857          → below band
merit_pct        = 0.08                                    (rating 3)
target_salary    = 0.95 × 1,400,000     = 1,330,000
full_gap_pct     = (1,330,000 − 1,200,000) / 1,200,000 = 0.10833
equity_pct       = 0.10833 / 2          = 0.05417          (2-year phasing)
total            = 0.08 + 0.05417       = 0.13417          (< 0.35 cap)
merit_amount     = floor(1,200,000 × 0.08)     = ₹96,000
equity_amount    = floor(1,200,000 × 0.05417)  = ₹65,000
recommended      = ₹1,61,000  =  13.4%
compa_after      = 1,361,000 / 1,400,000 = 0.972  ≥ 0.95 target
```

And `reason` produces:

> ₹1.6L (13.4%): 8% merit for a rating of 3, plus 5.4% equity correction — at 0.86 of band
> midpoint (below band); this brings pay up to the 0.95 target.

**Worked example — above band, same rating.** ₹24,00,000 against a ₹20,00,000 midpoint, rating 3.

```
compa        = 24,00,000 / 20,00,000 = 1.20      → above band, not held
merit_pct    = 0.08 × 0.5 = 0.04                 (damped)
equity_pct   = 0                                 (not below band)
merit_amount = floor(2,400,000 × 0.04) = ₹96,000 = 4%
```

> ₹96,000 (4%): 4% merit for a rating of 3, cut to 50% of normal because pay is above band — at
> 1.20 of band midpoint (above band).

Same performance rating, very different outcome — because position in band matters as much as
performance. Both sentences are readable by somebody with no technical background, which is the
actual acceptance criterion. If a recruiter cannot read the sentence and say "no, hang on, her
rating was 4", the tool has failed even when the arithmetic is perfect.

```python
recommend_batch(
    employees: pd.DataFrame,
    bands: Any,
    rules: IncrementRules,
    *,
    salary_col: str = "salary_annual",
    rating_col: str | None = "performance_rating",
    promotion_col: str | None = None,
    id_col: str | None = None,
) -> list[IncrementRecommendation]
```

`recommend` over a whole population, and the input `apply_budget_constraint` expects. `bands` is
anything with one midpoint per row. A `NaN` in the rating column becomes `None`, which routes into
the missing-rating policy.

##### Pay equity

```python
equity_review(
    employees: pd.DataFrame,
    bands: Any,
    rules: IncrementRules,
    *,
    salary_col: str = "salary_annual",
    id_col: str | None = None,
    keep_cols: Iterable[str] = (),
) -> EquityReview
```

Walks every employee, computes the compa-ratio, skips anyone at or above the threshold, and builds
a row for the rest with columns `employee_id`, `current_salary`, `band_midpoint`, `compa_ratio`,
`shortfall_to_band` (cost to reach 0.90), `cost_to_target` (cost to reach 0.95), `cost_this_year`,
plus whatever `keep_cols` names.

Two decisions worth understanding:

**Sorted by compa-ratio, not by rupees.** Somebody at 0.78 is more wronged than somebody at 0.88,
whatever their salaries are — the ratio is how unfair it is *to that person*. The rupee costs are
in the frame too, so re-sort by `cost_to_target` when the question is "what can we afford", which
is a different question and deserves to be asked separately.

**`keep_cols` exists for one reason.** The docstring says it: `["gender", "location"]` is the usual
choice, because the second question after "who is underpaid" is always "is it the same group every
time". That is the bridge from this module back to `fairness/audit.py`.

The `cost_this_year` calculation deliberately duplicates the arithmetic in `recommend`'s equity
block, with a comment saying so, so the two can never disagree about the bill.

##### The budget

```python
apply_budget_constraint(
    recommendations: Sequence[IncrementRecommendation],
    budget: float,
    *,
    priority: BudgetPriority | None = None,
) -> BudgetAllocation
```

The realistic case: the rules recommend ₹1.4Cr and the CFO has approved ₹1.1Cr. Something has to
give, and *what* gives is a policy choice — so it is a named, switchable one rather than whatever
the code happened to do.

If the recommendations already fit, the originals are returned untouched rather than re-derived
(re-deriving would only pick up rounding).

Otherwise, two inner functions do the work:

`fund_equity(remaining)` — sorts by `compa_ratio_before` ascending and funds each equity correction
**in full** until the money runs out. Deliberately *not* pro rata across the underpaid.
Half-correcting everybody leaves everybody still below band and still on the list next year; fully
correcting the worst cases takes real people off it. **When you cannot fix everything, fix
something completely.**

`fund_merit(remaining)` — scales merit and promotion together by one common factor. Pro rata rather
than best-ratings-first: cutting the bottom half of the performance range to zero to protect the
top would make the rating table mean something different at budget time than it does on paper.

The three priorities:

- `equity_first` → `fund_merit(fund_equity(budget))`
- `merit_first` → `fund_equity(fund_merit(budget))`
- `pro_rata` → one scale factor applied to every component of every recommendation

**Worked example.** Budget ₹2,00,000, priority `equity_first`, three employees:

| | compa | equity | merit | requested |
|---|---|---|---|---|
| A | 0.80 | ₹1,20,000 | ₹0 | ₹1,20,000 |
| B | 0.88 | ₹60,000 | ₹40,000 | ₹1,00,000 |
| C | 1.00 | ₹0 | ₹80,000 | ₹80,000 |
| | | | | **₹3,00,000** |

`fund_equity(200,000)`: A is worst, so A gets its full ₹1,20,000 (₹80,000 left). B gets its full
₹60,000 (₹20,000 left). C wants none.

`fund_merit(20,000)`: the merit-plus-promotion pool is ₹1,20,000, so the scale is
`20,000 / 120,000 = 0.1667`. B's merit becomes `floor(40,000 × 0.1667)` = ₹6,600; C's becomes
`floor(80,000 × 0.1667)` = ₹13,300.

Allocated: ₹1,99,900. Shortfall: ₹1,00,100. Two people carry `budget_reduced=True` — B and C. A
does not, because A lost nothing.

And every affected `reason` string now ends with " Trimmed to fit the increment budget." without
anybody having written that line at trim time.

#### The one thing to understand here

Budget size, what a rating is worth in rupees, and whether an equity gap closes in one year or
three are **decisions a company makes**, not patterns in data — so they live in a YAML file, and
the explanation of every recommendation is derived from its component amounts rather than stored
alongside them, so the words can never drift from the numbers.

#### Surprises and gotchas

- **The band-label boundaries are strict on one side.** `band_label` uses `< below_band` and
  `> above_band`, so exactly 0.90 and exactly 1.10 both read "in band". Consistent with `recommend`,
  which uses the same comparisons, but worth knowing before you write a test at a boundary.
- **`priority` defaults to the *first* recommendation's rules.** If you pass a list built under two
  different policy files, row 0 silently decides for everybody.
- **`allocated` can come in under budget by up to `step` per person.** Every component is floored,
  so the utilisation figure is usually just below 100% rather than exactly at it. That is the
  intended direction of the error.
- **The "trimmed from X by the single-year cap" clause reads the undamped table value.** In
  `reason`, that clause looks up `rules.merit.by_rating[rating_used]`, which is the full percentage
  before any above-band damping. With the shipped config the two caps cannot bite together (an
  above-band person gets no equity, so the total cannot reach 35%), but a different config could
  make that sentence overstate what was trimmed.
- **`equity_review` and `recommend` share arithmetic by duplication, not by a shared function.** The
  comment acknowledges it. It works, and it is one edit away from diverging.
- **Nothing here can produce a negative increment.** Every component is clamped at zero and rounded
  down, so the worst outcome for anybody is ₹0.
- **`rules` is a required positional field on `IncrementRecommendation`.** It carries `repr=False`
  so it does not flood debug output, but it has no default, so you cannot construct one of these
  by hand without a full rules object.

---

### `configs/policy/increment_fy_2026_27.yaml`

> The actual increment policy for one financial year — every number the rules apply, with the
> reasoning for each written beside it.

**Read time:** 10 minutes · **Difficulty:** easy
**Read it when:** immediately after `increment.py`. Read them side by side; the file is mostly
comments and they explain the code you just read.

#### What problem it solves

A policy hard-coded into Python is a policy that needs an engineer to change — and the people who
own compensation policy are not engineers. Everything arguable lives here, in a file an HR partner
can open, read, disagree with, and edit.

The header comment states the versioning convention and it is worth internalising:

> Next year you add `increment_fy_2027_28.yaml`. You do **not** edit this one. Six months after the
> cycle closes, somebody will ask "why did she get 13%?" The only honest answer is "here is the
> rule book we ran her through" — and that answer disappears the moment you edit last year's
> numbers.

Same convention as `configs/payroll/fy_YYYY_YY.yaml`, same reason.

#### What is in it

| Key | Value | Why |
|---|---|---|
| `financial_year` | `"2026-27"` | Matches the filename. |
| `source` | `"ILLUSTRATIVE. Not approved by anyone. Replace with your own numbers."` | Honesty about what this is. |
| `approved_by`, `approved_on` | `null` | Provenance slots, deliberately empty. |
| `budget.pct_of_payroll` | `0.10` | 10% of payroll. Plausible for Indian tech in 2026-27 (market averages have been running 8–10%); a real CFO sets the real one. |
| `budget.priority` | `equity_first` | The most contested line in the file. |
| `merit.by_rating` | `1:0, 2:0.04, 3:0.08, 4:0.12, 5:0.18` | A 1–5 scale, percentages of current salary. |
| `merit.on_missing_rating` | `assume_default` | What to do about a new joiner or an unsubmitted form. |
| `merit.default_rating` | `3` | |
| `compa.below_band` | `0.90` | Under this, the equity correction applies. |
| `compa.above_band` | `1.10` | Over this, merit is damped. |
| `compa.hold_above` | `1.25` | Over this, no increase at all. |
| `compa.above_band_merit_multiplier` | `0.5` | Half the normal merit above band. |
| `equity.target_compa_ratio` | `0.95` | Where corrections aim. |
| `equity.close_gap_over_years` | `2` | Equal slices over two years. |
| `equity.max_single_year_correction_pct` | `0.15` | Hard ceiling on one year's equity slice. |
| `promotion.uplift_pct` | `0.12` | Added on top of merit. |
| `limits.max_total_increase_pct` | `0.35` | No single person exceeds this, however the parts add up. |
| `limits.round_down_to_nearest` | `100` | Every rupee figure rounds down to ₹100. |

#### The arguments, as the file makes them

Each of these is a genuine judgement call, and the file spends more lines defending them than
stating them.

**`priority: equity_first`.** Underpayment is a mistake the company already made; merit is a reward
it is choosing to give; fixing a mistake outranks giving a reward. It is also the cheapest order
over time, because the people below band are the ones most likely to resign. `merit_first` is the
mirror image and genuinely defensible — some companies correct underpayment off-cycle rather than
out of the annual pool. `pro_rata` is included specifically so that "obviously fair" is an option
that was rejected on purpose, rather than one nobody considered.

**The merit spread, not the merit level.** The comment says it: 0% to 18% says performance is worth
arguing about; 7% to 10% says the rating is theatre.

**Rating 1 is 0%, by design.** Not a bug, and not a rounding.

**Missing ratings still get equity.** Whichever `on_missing_rating` you pick, the equity correction
applies regardless — being underpaid is a fact about the band, not about the rating, and
withholding it because a form is missing punishes the wrong person.

**Target 0.95, not 1.00.** The goal is to get people *inside* the band, not to move everyone to the
middle of it, which would cost a fortune and erase every legitimate difference within the band.

**Two years, not one.** Somebody at 0.70 needs a 36% raise to reach 0.95. No compensation committee
approves 36% in one line, so a rule that recommends it produces a recommendation that gets ignored
— and ignored recommendations correct nobody. Two slices of 18% get approved and actually fix the
problem.

**`above_band_merit_multiplier: 0.5`, not zero.** A strong performer who happens to sit high in the
band has still done the work, and zeroing them is how you lose them. The band catches up.

**Round down, never nearest.** Rounding up can push a set of recommendations past a budget they had
just fitted inside, and "we broke the budget by rounding" is not a sentence anyone wants to say out
loud.

#### The one thing to understand here

Every number that spends money is in this file with its reasoning beside it, and none of it is in
the Python. That is what makes the policy arguable — and being arguable is the entire justification
for Layer 3 not being a model.

#### Surprises and gotchas

- **The compa thresholds are duplicated.** The file says so explicitly: they also exist as
  constants in `src/paybands/model/band.py` for labelling predictions. If you move them, move them
  in both places. This file is the authority for anything that spends money; `band.py`'s copy is
  for display only.
- **The numbers are illustrative, and the file says so in `source`.** Nothing here has been
  approved by any compensation committee. Treat every figure as a placeholder demonstrating the
  shape of a policy, not as a recommendation.
- **`extra="forbid"` means a typo is fatal at load time.** Adding a key the models do not know
  about raises immediately. That is the intended behaviour — a silently ignored config line costs
  somebody their raise.

---

### `src/paybands/explain/shapley.py`

> Splits one prediction into per-feature contributions, converts them out of log space honestly,
> and turns the result into a sentence a recruiter can argue with.

**Read time:** 45 minutes · **Difficulty:** hard
**Read it when:** after `model/band.py` and `model/conformal.py`. You must already know that the
model trains on `log(salary)` and that the band has three quantile models — otherwise most of this
file will not land.

#### What problem it solves

Everything before this file produced a band. This one produces the **argument** behind it.

A recruiter must be able to argue with the model. A salary number with no reasoning attached cannot
be challenged, and an unchallengeable salary number is exactly the kind that causes harm — it
launders a guess into an authority. The same explanation is also what makes a decision defensible
six months later, when somebody asks why this candidate was offered less than that one.

The deliverable is one sentence:

> Predicted ₹17,67,000 — 8 years of experience added ₹2,60,000, a tier-1 (metro) location added
> ₹1,93,000, company size not given subtracted ₹1,01,000.

Everything else in the module is the machinery that lets that sentence be true.

#### Idea 1 — what SHAP is, and why `TreeExplainer` specifically

**SHAP** (SHapley Additive exPlanations) answers one question: starting from what the model
predicts for a *typical* candidate, how much did each feature of *this* candidate push the answer
up or down? Those pushes are **SHAP values**, and their defining property is that they add up
exactly:

```
prediction = baseline + sum(shap value for every feature)
```

That is not an approximation and not a fit — it is guaranteed by construction. It is what separates
SHAP from "feature importance", which tells you what matters *on average* and says nothing at all
about the person in front of you.

`shap.TreeExplainer` computes those values **exactly** for tree ensembles, in milliseconds, by
walking the trees. The general-purpose explainers (`KernelExplainer`, `PermutationExplainer`)
*sample* — orders of magnitude slower, and a different answer each run unless you pin a seed. A
sampled explanation of a salary is a bad thing to hand somebody who wants to argue with it.

The **median** model of the three gets explained. The band's edges are the honesty; the midpoint is
what people act on, so the midpoint is what needs a defence.

#### Idea 2 — the log-space trap, which is the whole difficulty

`model/band.py` trains on `log(salary)`. So SHAP values come back **in log units**. A raw SHAP value
of `0.18` is not a salary, not a percentage, and not anything a recruiter has ever seen. Handing it
over unconverted is the easiest way to make an explanation feel rigorous while communicating
nothing.

There are exactly two honest conversions and this module reports **both**, because they answer
different questions.

**(a) Multipliers — exact.** `exp(0.18) = 1.197`, so "eight years of experience multiplies the
prediction by about 1.20×". This is exact and complete:

```
prediction = baseline × multiplier₁ × multiplier₂ × ... × multiplierₖ
```

because multiplying is what adding in log space *is*. Nothing is lost, nothing is approximated, and
it is the scale the model actually learned on. **This is the number to trust.**

**(b) Rupees — a leave-one-out counterfactual, and it does not sum.** Recruiters think in rupees,
so the module also answers "what is this feature worth, in money?" using:

```
rupees = prediction − prediction ÷ multiplier
```

Read as: "take this candidate's contribution away and the prediction would fall from ₹23.6L to
₹20.4L, so it is worth about ₹3.2L." **That statement is exactly true for any one feature on its
own.** It stops being true the moment you add two of them together.

Here is why, with real numbers. Suppose the baseline is ₹10,00,000 and two features each contribute
`0.20` in log units (`multiplier = 1.2214` each):

```
prediction        = 1,000,000 × exp(0.40)          = ₹14,91,825
prediction − baseline                              = ₹4,91,825

feature 1 rupees  = 1,491,825 − 1,491,825 / 1.2214 = ₹2,70,422
feature 2 rupees  = 1,491,825 − 1,491,825 / 1.2214 = ₹2,70,422
sum of the parts                                   = ₹5,40,844

residual = 491,825 − 540,844                       = −₹49,019
```

The multipliers, meanwhile, are exact: `1,000,000 × exp(0.20) × exp(0.20) = ₹14,91,825`, which is
the prediction to the rupee. (Written out to four decimal places the multiplier is 1.2214, and
`1,000,000 × 1.2214 × 1.2214` comes to ₹14,91,820 — that ₹5 is the *display* rounding, not the
method.)

The parts over-count because each is measured against the *full* prediction, so when several
features push the same way they each get credit for ground the others also covered. `exp` is not
linear: the same 20% is ₹1L at ₹5L and ₹10L at ₹50L.

**The residual is published rather than fudged.** `Explanation.rupee_residual` reports the leftover
and `Explanation.approximation_note` says it in words, as a *required* field the API ships.

The tempting alternative is to rescale every rupee figure by
`(prediction − baseline) ÷ sum(rupees)` so the column sums correctly. **Don't.** That produces
numbers that are individually wrong, sum to something true, and carry no warning — the worst
combination available. A number that visibly does not tie out invites the question "why not?", and
the answer to that question is real information about the model.

On the project's own worked example (JOURNEY Part 13) the parts totalled ₹1,66,000 against a real
change of ₹1,93,000. The ₹27,000 residual is printed. Nudging those five numbers so they sum neatly
would take two lines and nobody would ever notice. **The tidy total would have been the dishonest
one.**

#### Idea 3 — four experience columns are one fact wearing four hats

`features/experience.py` builds `years_experience`, `experience_log`, `experience_sqrt` and
`experience_bucket` out of one input number, so the model does not have to rediscover the shape of
the experience-to-pay curve. Together they are 53.7% of the model's gain and about four fifths of
its permutation importance.

Shown to a recruiter unmerged, that becomes four separate rows — "experience added ₹3.2L,
experience (log scale) added ₹1.1L, the 6–8 band subtracted ₹0.2L, experience (square-root scale)
added ₹0.1L" — which reads like four independent findings that partly contradict each other. It is
one finding.

So `DEFAULT_GROUPS` folds them into a single `experience` row, and that fold is **exact**: SHAP
values add, so summing four of them is summing, not averaging.

The trade-off is stated rather than hidden. Merged, you can no longer see *which* encoding the
trees leaned on, and that is a genuine modelling diagnostic. Pass `groups={}` for the raw view.
Grouping is right for recruiter-facing output and wrong for model debugging, so it is a default
rather than a rule.

Skill flags are deliberately **not** grouped. "Knowing Go" is something a candidate can point at
and a recruiter can weigh; a merged `skills` row is not actionable by anyone.

#### Idea 4 — a missing input is an answer, not a blank

If a caller does not send `employment_type`, `FeatureBuilder` emits `NaN`, LightGBM routes `NaN`
down whichever branch fits the training data best, and that routing has a SHAP value like any other.
You will see rows such as "employment type not given — ×1.04". That looks odd and it is correct: the
model learned a response to missingness, and concealing it would misrepresent what the model
actually did.

#### Idea 5 — rupees round to ₹1,000

Reporting "experience added ₹3,22,135" claims a precision the band's own width flatly disproves.
`RUPEE_ROUNDING` matches `api/service.RUPEE_ROUNDING`, so an explanation and the band it explains
round the same way and cannot disagree. **Multipliers and log contributions are not rounded** —
those are the exact quantities, and rounding them would break the additivity a reader might want to
check.

#### Module-level constants

| Constant | Value | Meaning |
|---|---|---|
| `RUPEE_ROUNDING` | `1_000` | Every money figure this module reports rounds to this. |
| `DEFAULT_SENTENCE_TOP_N` | `3` | How many contributions a sentence mentions. Three is the number of reasons a person repeats after hearing them once. |
| `NEGLIGIBLE_LOG_CONTRIBUTION` | `0.005` | Below this in absolute log units, a contribution is noise, not a reason — half a percent, smaller than the rupee rounding on most bands. |
| `EXPERIENCE_FEATURES` | 4 column names | The four experience encodings. |
| `DEFAULT_GROUPS` | `{"experience": EXPERIENCE_FEATURES}` | The only grouping applied by default. |
| `FEATURE_LABELS` | 17 entries | Short human labels, e.g. `org_size` → "company size". |
| `TIER_PHRASES` | `{1: "a tier-1 (metro) location", 2: "a tier-2 city", 3: "a tier-3 (non-metro) location"}` | Because "tier 3" alone tells a reader nothing — and tier 3 is also the catch-all for a city the alias table did not recognise. |

#### Classes

##### `Contribution`

One feature's push on one prediction, in three scales at once.

| Field | Meaning |
|---|---|
| `feature` | Display name — a column name, or a group name like `experience`. |
| `members` | Which raw columns were folded in. Length 1 unless this row is a group. |
| `value` | The candidate's value as the model saw it. `nan` when the field was not supplied — which is itself a contribution. |
| `log_contribution` | The raw SHAP value. Exact, additive, unreadable. |
| `multiplier` | `exp(log_contribution)`. **Exact.** |
| `rupees` | The leave-one-out figure, rounded to ₹1,000. Approximate, and the one a recruiter will quote. |
| `phrase` | The English noun phrase, e.g. "8 years of experience". |

Properties: `direction` (`"increased"` / `"decreased"` / `"no effect"`, using
`NEGLIGIBLE_LOG_CONTRIBUTION` as the cut-off), `percent` (`multiplier - 1.0`), and `verb`
(`"added"` / `"subtracted"` / `"changed"` — money words, not maths words). `describe()` renders one
table line.

Read the fields in the order the docstring gives: log contribution, then multiplier, then rupees.

##### `Explanation`

Every contribution behind one predicted midpoint, ranked by absolute size.

| Field | Meaning |
|---|---|
| `contributions` | A tuple of `Contribution`, biggest push first. |
| `baseline` | What the model predicts before it knows anything about this person, in rupees. |
| `prediction` | The median model's prediction for this candidate, in rupees. |
| `baseline_log`, `prediction_log` | The same two numbers in log units, **unrounded**. Additivity is checked here. |
| `quantile` | Which quantile was explained, so nobody has to guess whether an explanation belongs to the midpoint or to a band edge. Defaults to `0.5`. |

The invariant, which `tests/test_explain.py` asserts:

```
baseline_log + sum(c.log_contribution for c in contributions) == prediction_log
```

exactly, to floating-point tolerance. It survives grouping because grouping is addition.

Note the honest reading of `baseline`, from the docstring: it is "a typical candidate in the
training data", **not** "the market median". The training data is whatever we happened to collect.

Properties and methods:

- `n_features` — how many contribution rows.
- `rupee_residual` — `(prediction − baseline) − sum(rupees)`. **Should not be zero.** This is the
  honesty gauge for Idea 2.
- `approximation_note` — the sentence that stops the rupee column being misread. Required output,
  never optional, for the same reason `api/service.build_caveat` is: JSON strips uncertainty by
  default, and a column of rupee figures that a UI renders in a neat stack will be read as a sum
  unless something explicitly says it is not one.
- `top(n=3)` — the `n` largest contributions with negligible ones dropped.
- `sentence(n=3, *, include_note=False)` — the deliverable. Handles the case where nothing moved
  much with a different sentence entirely. The API sets `include_note=False` and ships the note as
  its own field so a UI can style it.
- `to_frame()` — the contributions as a DataFrame with columns `feature`, `value`, `phrase`,
  `log_contribution`, `multiplier`, `rupees`.
- `describe(n=8)` — a block of text for the findings log.

##### `BatchExplanation`

Explanations for many rows at once — the *global* view, built bottom-up.

| Field | Meaning |
|---|---|
| `log_contributions` | One row per explained candidate, one column per (grouped) feature, in log units. The raw material. |
| `baseline_log` | The typical training prediction, in log units. |
| `quantile` | Which model was explained. |

`to_frame()` ranks features by mean **absolute** push and returns four columns:

- `mean_abs_log` — the ranking column: typical size of the push, regardless of sign.
- `mean_swing` — the same as a percentage (`exp(mean_abs_log) - 1`), which is what a log unit
  *means*: 0.12 is "this feature typically moves a prediction by about 12%".
- `mean_log` — the **signed** average. Near zero with a large `mean_abs_log` is the interesting
  case: the feature matters a lot and pushes both ways, which is a real feature. Large and
  one-signed means the feature mostly shifts everybody the same direction, which a constant would
  have done.
- `share` — each feature's share of the total absolute push.

Why bother when LightGBM already prints a feature importance chart? Because split-count importance
rewards high-cardinality columns for being *splittable* rather than useful. `docs/findings.md` §4.1
records the textbook case: `n_skills` reaches 12.3% of split count while shuffling it costs the
model only 0.25% of its error, because it is an integer running 0–20 and a tree can split on it at
many thresholds.

**What this still is not.** Every number here is measured on the model, not on reality. A feature
can dominate SHAP because the model leans on it and be worthless on held-out rows — `education` in
this project is exactly that. For "does this feature earn its place?", permutation importance on
held-out rows is the measure; for "what is the model doing?", this one is.

##### `BandExplainer`

A `shap.TreeExplainer` for the median model, built once and reused.

```python
explainer = BandExplainer(bundle.band)
print(explainer.explain(candidate_frame).sentence())
```

| Attribute | Meaning |
|---|---|
| `regressor` | The LightGBM model for the median quantile. |
| `builder` | Its `FeatureBuilder`, used to transform incoming rows. |
| `quantile` | Which quantile is being explained. |
| `explainer` | The `shap.TreeExplainer`. |
| `feature_names` | The builder's column list, in order. |
| `baseline_log` | The explainer's expected value, in log units. |
| `skill_names` | `{"skill_typescript": "typescript"}` — so casing can be restored. |

Property `baseline` returns `exp(baseline_log)` in rupees.

**Why this is a class and not just a function.** Constructing a `TreeExplainer` parses the whole
booster; explaining a row afterwards takes about a millisecond. Rebuilding it per request would make
the cheap part pay for the expensive part on every call, so the API holds one for the process
lifetime.

**`shap` is imported inside the constructor**, not at module scope. `shap` pulls in `numba`, which
takes seconds to import; doing that at module level would tax every `import paybands.api`, every
test collection, and every process that never asks for an explanation.

Two robustness details in `__init__` worth noticing. `np.ravel(...)[0]` on `expected_value`,
because `shap` has shipped that as both a scalar and a length-1 array across versions. And
`zip(..., strict=False)` when pairing skill feature names to vocabulary entries, because the skills
block appends `n_skills` to its own `feature_names_` — the non-strict zip is what drops that
trailing extra.

Methods:

- `_shap_matrix(X) -> tuple[NDArray, pd.DataFrame]` — transforms the frame through the builder,
  calls `shap_values`, reshapes a flat single-row result, and returns both the values and the
  feature matrix they describe.
- `explain(X_row, *, groups=None) -> Explanation` — one candidate. Raises if the frame is not
  exactly one row.
- `explain_many(X, *, groups=None) -> BatchExplanation` — a whole frame. Grouping is done by
  transposing, `groupby(keys, sort=False).sum()`, and transposing back, because summing columns is
  the whole of grouping.

#### Functions

`_is_missing(value) -> bool` — `True` for NaN / None / NaT, with a `try/except` for arrays and
exotic objects.

`_pretty_skill(raw) -> str` — `"typescript"` → `"TypeScript"`. `features/skills.py` lowercases every
skill so "Python" and "python" cannot become two features; that is right for the model and wrong
for a sentence. Fourteen known cases, then `.title()` as a fallback — which is imperfect
(`"node.js"` → `"Node.Js"`) and still far better than shouting "PYTHON" or whispering "python" at a
hiring manager.

`_phrase(name, value, *, skill_names=None) -> str` — a noun phrase that slots into "… added ₹6L".
The target grammar is a **thing**, never a column name: "a services company", not
`prev_company_type = services`. Skill flags become "knowing Go" or "no Go"; a missing value becomes
"company size not given"; a tier becomes one of the `TIER_PHRASES`; anything unrecognised falls
back to `label = value`, which is the right failure mode for a column somebody added yesterday.

`_round_rupees(amount) -> float` — nearest `RUPEE_ROUNDING`, sign preserved.

`_group(names, values, row, groups) -> list[tuple[str, tuple[str, ...], Any, float]]` — folds
grouped columns into one entry each, keeping the builder's column order. Returns
`(display name, member columns, headline value, summed log value)`. The headline value for a group
is the value of its **first listed member** — for experience, `years_experience`, the one a human
actually said.

```python
median_quantile_model(model: object) -> tuple[Any, Any, float]
```

Digs the LightGBM regressor for the median quantile out of whatever the caller handed over — a
`ConformalBand`, a `SalaryBandModel`, or something shaped like one in a test. It unwraps `.model`
up to four times (a depth cap, so a self-referential object cannot hang the process) looking for an
object with `models_`.

**Unwrapping the conformal layer is not a shortcut, it is correct.** Conformal calibration moves the
band's *edges* by a constant in log space and leaves the median untouched. There is nothing about
the midpoint for it to explain.

It picks the model **nearest to 0.5** rather than index 1, so a band built on unusual quantiles
still gets its most-central model explained instead of whatever sits in the middle slot.

```python
explain_prediction(model, X_row, *, groups=None) -> Explanation
explain_batch(model, X, *, groups=None) -> BatchExplanation
sentence(explanation, n=DEFAULT_SENTENCE_TOP_N) -> str
```

The two functions to call, plus a module-level alias for `Explanation.sentence` that makes pipelines
read better.

`X_row` must be a **one-row common-schema frame** — the same thing you would pass to `predict_band`,
not a feature matrix. The model's own `FeatureBuilder` does the transform, so an explanation can
never describe a differently-built matrix from the prediction it explains.

Both convenience functions build a fresh `BandExplainer` per call. Fine for a script, wasteful in a
loop or a server — hold a `BandExplainer` there instead. The docstrings say so.

`explain_batch` prefers held-out rows. Run on training rows it measures what the model memorised as
much as what it learned, which is a different and less interesting question.

#### The one thing to understand here

The model works in log space, so SHAP values arrive as log contributions. Multipliers (`exp` of
those values) are **exact** and multiply out to the prediction; rupee figures are a leave-one-out
counterfactual that is true one feature at a time and **does not sum**, because `exp` is not
linear. The residual is published rather than fudged, because a number that visibly does not tie
out invites the right question.

#### Surprises and gotchas

- **The prediction is reconstructed from SHAP, not read from the model.**
  `prediction_log = baseline_log + values[0].sum()`. `TreeExplainer` guarantees this equals the
  model's own output, but it means the explanation never calls `predict` and the two paths are only
  as consistent as that guarantee.
- **`Explanation.baseline` is rounded; `BandExplainer.baseline` is not.** The former is
  `_round_rupees(exp(baseline_log))`, the latter is raw. Both are correct for their purposes and
  they will differ by up to ₹500.
- **`rupee_residual` uses the rounded baseline and prediction.** So a small part of it is rounding
  rather than non-linearity. The note says "out by about", which covers it.
- **The residual has a sign.** When features push the same way it is negative (the parts
  over-count, as in the worked example above); when they push in opposite directions the arithmetic
  can go the other way. `approximation_note` prints `abs(...)` and says "out by about", so the
  direction is not surfaced.
- **`explain_many` returns no values or phrases.** `BatchExplanation` holds only log contributions,
  so you cannot get English sentences out of the batch path — that is by design, since the batch
  view answers a global question.
- **The unknown-feature fallback is `f"{label} = {value}"`.** A column added to the builder without
  a `FEATURE_LABELS` entry will appear in a recruiter-facing sentence as `new_column = 3`. Ugly,
  but visible, which is the right failure mode.
- **`Contribution.describe()` truncates the phrase to 42 characters.** Only affects the table
  rendering, not `sentence()`.
- **`groups={}` and `groups=None` mean different things.** `None` means "use `DEFAULT_GROUPS`"; an
  empty dict means "no grouping at all". Easy to conflate.

---

### `tests/test_fairness.py`

> Tests *of* the fairness audit, not merely tests *for* it — a calibration suite that checks a
> measuring instrument against a known standard.

**Read time:** 30 minutes · **Difficulty:** medium
**Read it when:** last, after `audit.py` and `proxy.py`. It is the best summary of both.

#### What problem it solves

Most test files check that a function returns the right type and handles an empty list. This one
mostly does something else.

It generates data in which one group is paid exactly 8% less *because the test wrote that rule
itself*, runs the audit, and demands 8% back. If the audit says 2% or 15%, it is broken — and an
audit broken in the confident direction gets somebody accused of discrimination they did not commit,
while one broken the other way tells a company everything is fine when it is not.

**You cannot do this on real data.** Nobody knows the true pay gap at a real company, so a wrong
answer and a right answer look identical. The synthetic generator exists so that here, in this
file, they do not.

If you read only one test, read `test_recovers_the_injected_gap`.

#### Shared setup

| Name | Value | Purpose |
|---|---|---|
| `N` | `10_000` | Large enough that sampling noise is smaller than the effects being tested. |
| `TOLERANCE` | `0.01` | At `N`, the standard error on the gap coefficient is about 0.005 in log points, so this is roughly two standard errors — tight enough to fail a broken audit, loose enough not to fail on luck. |
| `ALL_CONTROLS` | `[*LEGITIMATE_CONTROLS, *CONTESTED_CONTROLS]` | Everything, including the career-break column. Including that one is a choice, not a default. |

`gap_of(df, truth, *, controls=None) -> float` — runs `adjusted_gap` on a generated frame and
returns the gap. It passes `disadvantaged=truth.disadvantaged_group` — reading the answer sheet
rather than letting the audit auto-detect. Auto-detection picks the lower-paid group, which would
make the gap positive by construction and turn the zero-gap control test into a test that cannot
fail.

Two module-scoped fixtures: `biased` is `generate(N, seed=42)` (the default world: an 8% gap plus a
proxy that partly encodes it) and `fair` is the same with `pay_gap=0.0`.

`experience_gap_data(n=4_000, seed=0) -> pd.DataFrame` is a hand-built world with **no unfair pay
and a large raw gap**: everyone paid by exactly the same rule, but one group averaging four years of
experience and the other nine. It is built by hand rather than taken from the generator because the
generator deliberately gives both groups the same experience distribution — and to show that the raw
and adjusted gaps can disagree completely, you need a world where they do. The two experience ranges
deliberately **overlap**; if they did not, `adjusted_gap` would correctly refuse to answer.

#### The tests, by section

##### The headline — does the audit read true?

**`test_recovers_the_injected_gap`** — parametrised over five injected gaps (0%, 3%, 8%, 15%, 25%)
and two seeds (42, 7), so ten runs. Each must recover the injected gap to within one percentage
point. This is the test that licenses every other fairness number in the project. Ten combinations
because an audit that only works at 8% on seed 42 is a coincidence dressed as a measurement.

**`test_zero_gap_stays_zero`** — the control case, and the one that matters most for not doing harm.
Note what "fair" does and does not mean here: `pay_gap=0.0` means two people alike in every recorded
respect are paid the same. The two groups still do *not* earn the same on average, because one takes
more career breaks and breaks cost money. Separating "explained" from "unexplained" is the entire
job of the audit, and this test is where that separation is checked.

**`test_the_confidence_interval_contains_the_truth`** — ten seeds at n=3,000; allows at most two
misses. Ten seeds is far too few to measure a 95% coverage rate properly, and the docstring says so.
But it is plenty to catch an interval that is wrong *by a factor* — a missing square root, a standard
error on the wrong degrees of freedom. Those bugs fail everywhere, not occasionally.

**`test_validate_on_synthetic_reports_a_clean_calibration_table`** — the shipped self-check must
pass on the values it ships with, since it is what a reader runs first. It also bounds the
systematic error at 0.005, with a comment noting the audit reads about 0.2–0.3 percentage points low
across the board.

##### Raw versus adjusted

**`test_raw_and_adjusted_gaps_disagree_when_the_groups_differ`** — the single most important
distinction in the module, as an assertion. Raw gap above 25%, adjusted gap zero, and all of the
difference explained by experience.

**`test_the_raw_gap_reports_both_mean_and_median`** — two averages, because the mean of a skewed
distribution misleads.

**`test_the_raw_gap_is_bigger_than_the_adjusted_gap`** — the controls should explain *some* of the
difference, never all of it.

##### Confidence intervals

**`test_intervals_widen_on_smaller_samples`** — n = 400, 1,600, 6,400. Each step quadruples the
sample, so each width should fall by roughly half. The bounds are loose (1.4 to 3.0) because this is
checking the shape of a relationship, not its constant.

**`test_a_small_sample_cannot_pin_down_a_real_gap`** — this is the honest, uncomfortable one. Ten
samples of 150 employees from the same world with the same injected 8% gap. The point estimates
swing across a range wider than the gap itself, every interval is over 10 points wide, and at least
half cannot rule out zero. A department of 150 people simply cannot answer this question, and the
right conclusion is "we cannot tell" — a different sentence from "there is no gap". A tool printing
only a point estimate lets a reader confuse the two, and that confusion runs in whichever direction
the reader already preferred.

**`test_a_wider_confidence_level_gives_a_wider_interval`** — 80% versus 99%; the point estimate is
identical and only the width changes.

##### The model's own bias

**`test_residuals_are_balanced_when_the_model_is_even_handed`** — the false-positive case. A model
that misses randomly must not be reported as biased. Noise is not bias, and a residual test that
cannot tell them apart is useless, because every model has residuals.

**`test_residuals_catch_a_model_that_lowballs_one_group`** — the true-positive case. Predictions are
shaved 10% for one group only, so `actual/predicted ≈ 1/0.9` and the check should read about 11%
low. It must find it, get the size roughly right, and get the **direction** right — a bias detector
that reports the wrong group is worse than no detector.

**`test_a_model_that_learned_the_bias_perfectly_passes_the_residual_test`** — the trap, stated as
code. A model that reproduces the salaries, unfairness included, has beautifully balanced residuals
(`not check.is_significant`) *and* still hands every future candidate an 8% gap
(`gap_of(df, truth) ≈ truth.pay_gap`). Both assertions in one test. This is why `adjusted_gap` and
`residual_analysis` are separate functions and why the report prints both.

##### The proxy problem

**`test_the_proxy_scan_finds_the_planted_proxy`** — `career_gap_months` must come out top with
association above 0.25, must be the only entry `leaking()` returns, and every legitimate control
must score below 0.05. If the scan cannot find a proxy planted on purpose, it will not find the ones
a real dataset hides.

**`test_the_proxy_inflates_the_gap_when_it_is_left_out`** — stop controlling for career breaks and
the measured gap grows by more than two points. Neither number is wrong; they answer different
questions, and which one you quote is a judgement about whether time out of the workforce is a
legitimate reason to pay less.

**`test_the_gap_survives_deleting_the_protected_column`** — **the proxy demonstration.** Asserts
`gap_with_protected > 0.09`, `gap_without_protected > 0.03`, and `surviving_share > 0.30`. The
comment in the test is the point: "Not 'smaller' — SURVIVING."

**`test_deleting_the_protected_column_costs_almost_no_accuracy`** — `accuracy_cost < 0.05`. The
removal bought nothing, which makes it worse rather than better.

**`test_the_proxy_result_is_not_an_artefact_of_a_linear_model`** — the same demonstration with
LightGBM, guarded by `pytest.importorskip`.

**`test_switching_the_proxy_off_removes_the_survival`** — the negative control, discussed above.
Without it, the headline test could be passing for some unrelated reason and nobody would know.

##### The report

**`test_the_report_says_what_a_reader_needs_to_know`** — asserts the printed report contains
"HEADLINE PAY GAP", "LIKE-FOR-LIKE", "proves nothing", "not a finding of", the control list
(`years_experience` appears by name), the caller's custom note, and the model section — and that
every line is 100 characters or fewer. A fairness number that travels without its caveats gets
repeated without them, so the string itself has to push back, which makes it testable.

**`test_the_report_omits_the_model_section_without_predictions`** — no predictions, no claims about
a model. Silence beats a placeholder.

##### Guard rails

**`test_the_audit_refuses_to_control_for_the_protected_attribute`** — it sounds absurd written down
and it is one column-name typo away in any pipeline that builds a control list programmatically. The
failure is otherwise silent: the audit would report a clean zero.

**`test_the_audit_refuses_when_the_groups_never_overlap`** — the most instructive guard rail. One
group has 0–5 years of experience and the other 6–12, so there is not a single comparable pair. Left
unchecked, the linear algebra still returns a number — and on this data that number came out at
**−76%**, which somebody would have reported.

**`test_missing_control_values_are_refused_rather_than_guessed`** — a blank is not a category, and
silently dropping rows changes the answer.

**`test_auto_detection_picks_the_lower_paid_group`** — pins the documented catch. Even on a world
with no injected gap, auto-detection still returns a positive number. It can tell you a gap's size
but never that a gap exists.

**`test_more_than_two_groups_must_be_named`** — three groups is an ambiguous request, not a
computable one. Named explicitly it works, and quietly ignores the third group rather than folding
it into one of the two (`measured.n == N - 100`).

**`test_bad_arguments_fail_loudly`** — five failures in one test: non-positive salary, shape
mismatch, confidence outside (0, 1), mismatched control row count, and zero degrees of freedom. The
last one has the best comment: four rows and four independent columns means the fit passes exactly
through every point, leaves no residual to measure noise with, and would report a confidence
interval of width zero around a made-up number.

**`test_the_least_squares_default_refuses_to_predict_before_fitting`** — one line, for
`proxy.LeastSquares`.

#### The one thing to understand here

A bias detector cannot be validated on real data, because a broken one and a working one produce
output that looks identical. This file is the only place in the project where the true answer is
known, so it is the only place the audit can be shown to work — which makes it a calibration
certificate rather than a regression suite.

#### Surprises and gotchas

- **28 test functions, 37 test cases.** `test_recovers_the_injected_gap` is parametrised twice
  (5 gaps × 2 seeds), so it counts as ten. That matches the "37 tests" figure in JOURNEY Part 9.
- **Two fixtures are `scope="module"`.** `biased` and `fair` each generate 10,000 rows once and
  share the frames across every test that asks for them. They are never mutated — tests that need a
  variant call `generate` directly.
- **Tolerances are asymmetric and deliberate.** `TOLERANCE = 0.01` on the gap, but
  `abs=0.02` for the hand-built experience world and `abs=0.03` for the residual direction check.
  Each is sized to its own noise, not copied.
- **The LightGBM test is skipped, not failed, if LightGBM is absent.** `pytest.importorskip` means
  a lean environment silently loses one of the two robustness controls.
- **`test_more_than_two_groups_must_be_named` mutates a copy.** `df["gender"].to_numpy(object).copy()`
  — without the copy, the module-scoped `biased` fixture would be corrupted for every later test.
