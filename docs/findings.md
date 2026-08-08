# Findings

What we measured, what worked, what did not, and what the data cannot answer at all.

Everything here comes from one script:

```bash
uv run python scripts/run_analysis.py
```

It is seeded, it runs top to bottom in about half a minute, and it prints every number
below and writes every chart to `reports/`. **No number appears in this document that the
script does not print.** If you disagree with a claim, run it and argue with the output.

---

## The short version

| | |
|---|---|
| **Predicting a salary at all is easy-ish.** | A four-line `groupby` beats "everyone earns the median" by **19.4%** MAE. Gradient boosting beats the `groupby` by another **11.9%**, on 10 of 10 splits. |
| **Predicting an *honest* band is harder, and we did it.** | The raw quantile band promises 80% and delivers **74.9%** — it lies. Conformal calibration lifts that to **82.3%**. |
| **The honest band is too wide to use.** | Median band width is **2.4x the midpoint**. A typical answer is *"₹4.9L – ₹39.8L"*. That is not a salary band, it is the industry. |
| **One thing drives salary, and everything else is decoration.** | Experience is **53.7%** of the model's gain and **four fifths** of its permutation importance. **16 of 37** features carry no measurable held-out signal. Education is worse than useless: shuffling it *improves* held-out error. |
| **The error is concentrated, not spread.** | Roles with fewer than 5 test rows per split are one-sided by **40.8%** on average, against **10.3%** for the rest. |
| **The fairness audit works and has nothing to audit.** | It recovers injected gaps of 0/3/8/15/25% to within **0.34 percentage points**. The 2025 survey has **zero** protected-attribute columns out of 172. |

The last line is the most important one in the document, and §5 is about it.

---

## 1 · What was measured, and on what

**Source:** Stack Overflow Developer Survey 2025, the public CSV. 49,191 responses,
172 columns.

**Filtering, and what each step cost:**

```
survey responses            49,191
from India                   2,547
reporting salary in INR      2,463
  − no salary reported        1,235  (50.1%)
  − below ₹100,000              200  (8.1%)
  − above ₹20,000,000             6  (0.2%)
usable rows                  1,022  (41% of INR rows)
```

Two things to notice before anything else.

**Half the Indian respondents did not answer the salary question.** 1,235 of 2,463 —
50.1%. This is not a cleaning problem, it is **selection bias**, and no rule removes it.
We know nothing at all about the half who declined, and "declined to state a salary" is
very unlikely to be independent of the salary. Everything downstream describes the people
who were willing to tell a public survey what they earn.

**Six people were dropped for being implausibly rich, and one of them broke the mean.**
The largest entry in the file is `1.0000e+23` rupees — 23 digits, a run of nines typed
into a text box. It dragged the mean of the raw INR salaries to **₹8.14 × 10^19**, roughly
81 quintillion rupees. The median of the same column was **₹11,00,000**, entirely
unmoved.

> One typo moved the mean by eighteen orders of magnitude and the median not at all. That
> is the whole argument for medians in one line, and it is why every baseline in this
> project predicts one.

**After cleaning:** median ₹15,00,000, mean ₹24,23,787. The mean sits above roughly
two-thirds of respondents, which is what a long right tail does.

**Skew, before and after logging:** **+2.86** in rupees, **−0.23** in log rupees. Anyone
can assert "salaries are skewed, so model log(salary)". Those two numbers are the
evidence, and `reports/dist_raw_vs_log.png` is the picture: same 1,022 people, nothing
trimmed, only the axis respaced.

**Who the 1,022 are:**

| role | n | share |
|---|---|---|
| Fullstack | 330 | 32.3% |
| Other | 210 | 20.5% |
| Backend | 201 | 19.7% |
| Frontend | 72 | 7.0% |
| Mobile | 53 | 5.2% |
| DevOps | 30 | 2.9% |
| Management | 28 | 2.7% |
| Data Engineering | 26 | 2.5% |
| Data Science | 18 | 1.8% |
| Student | 16 | 1.6% |
| Embedded | 15 | 1.5% |
| Analytics | 10 | 1.0% |
| QA | 8 | 0.8% |
| Security | 5 | 0.5% |

The three largest roles are **72.5%** of the sample. Six roles have fewer than 20
respondents each: Data Science (18), Student (16), Embedded (15), Analytics (10), QA (8),
Security (5). Everything this project says about those rests on a handful of people, and
§4 shows exactly what that does.

Median experience is 6 years; 6 rows have none recorded.

### The drivers the survey simply does not have

`docs/design.md` §4.1 lists what actually sets an Indian tech salary. The script checks
the loaded frame for each one:

| driver | present? | why it matters |
|---|---|---|
| `location_tier` | no | Bangalore vs a tier-2 city — one of the largest gaps there is |
| `prev_company_type` | no | product vs services — design.md calls this a make-or-break driver |
| `institute_tier` | no | the IIT/NIT/BITS premium, and how it fades |
| `level` | no | internal grade |
| `prev_salary` | no | the anchor we specifically wanted to test against |
| `performance_rating` | no | needed for the increment layer |
| `event_date` | no | no date means no temporal split is possible |

Seven for seven. **The model that follows is not a weak model of Indian salaries. It is an
honest model of the four or five things a public survey happens to ask about.** That
distinction is the difference between "this needs better hyperparameters" and "this needs
different data", and it is the finding the rest of the document keeps running into.

---

## 2 · What worked

Every comparison below is run across **10 independent train/test splits**, seeds 0–9, and
reported as a mean *with the full range*. A single split is an anecdote: on ~204 test rows
the headline MAE moves by more than 10% depending on which 204 you hold out. Where the
claim is "A beats B", the script counts the splits on which that was actually true.

### 2.1 The lookup table beats the constant, by a lot

| | MAE |
|---|---|
| Baseline 0 — everyone earns the median | mean **₹17,00,680** (₹15,13,097 – ₹19,93,813) |
| Baseline 1 — median of (role × experience) | mean **₹13,72,133** (₹11,74,150 – ₹15,85,261) |
| **improvement** | **mean 19.4%** (13.1% – 26.0%), on **10/10** splits |

Baseline 1 is four lines of pandas. It is the number everything else has to beat.

Two details worth reading:

**The lookup table is coarser than it looks.** Its fallback cascade reports where each
prediction actually came from. On the last split: **72.5%** matched a (role × experience)
cell with at least 10 people in it, **27.5%** fell back to experience alone, and 0% fell
all the way to the global median. So more than a quarter of candidates are being quoted a
number that ignores their role entirely.

**R2 disagrees with MAE, and MAE is right here.** On the last split the global-median
baseline scores **R2 = −0.131** — *worse than guessing*. That is not a bug. R2 measures
squared error, squared error is minimised by the **mean**, and the baseline deliberately
predicts the **median**. On a right-skewed target those are far apart. The baseline is
still the better model for our purpose, because we care about the typical candidate rather
than about minimising squares. The metric encodes a choice; ours is stated.

### 2.2 The band model earns its complexity

Three LightGBM quantile models (0.1 / 0.5 / 0.9), trained on log(salary), on a 60/20/20
train / calibration / test split.

| | MAE |
|---|---|
| Lookup table, *identical* training rows | mean **₹14,12,145** (₹10,96,574 – ₹16,03,946) |
| Band model midpoint | mean **₹12,47,471** (₹9,21,042 – ₹14,32,699) |
| **improvement** | **mean 11.9%** (7.4% – 17.5%), on **10/10** splits |

And the fair fight for an actual shipping decision — the lookup table needs no calibration
set, so in production it would be built on 80% of the data while the band model gets only
60%:

| | MAE |
|---|---|
| Lookup table, all 80% | mean **₹13,98,447** |
| **improvement** | **mean 11.0%** (6.8% – 15.7%), on **10/10** splits |

Even handicapped, the model wins on every split. It earns its place — but note the size:
**11%**, not the 3x that a bare R2 would let you imply. The lookup table is genuinely hard
to beat, which is precisely why we built it first.

Quantile crossing — the three independently fitted models predicting out of order — happens
on **0.2%** of test rows (0% – 1.5%). That is low enough to say the three models agree
about the shape of the world; we sort each row anyway, and we count it rather than hide it.

### 2.3 The band promised 80% and delivered 75%. Conformal prediction fixed it.

This is the section the project exists for.

| | coverage of the "80%" band |
|---|---|
| Raw quantile band | **74.9%** (70.1% – 84.3%) |
| After conformal calibration | **82.3%** (79.4% – 87.3%) |

The raw band under-covers on essentially every split. Asking LightGBM for the 90th
percentile is not the same as getting it: between the loss function and the answer sit a
finite training set, a 7-leaf model, and a test set nobody has seen. The promise leaks out
through all three, and *nothing about the output announces it*. You only find out by
measuring.

Conformalised quantile regression (Romano, Patterson & Candes 2019) computes one number
qhat from a calibration set the model has never seen and moves both edges by it. Here
qhat averaged **0.1573 log points** (0.0469 – 0.2508), which is each band edge moving by
**17.2%** on average (4.8% – 28.5%).

Checked at five confidence levels rather than one — because one level checked is a spot
check, five is a calibration curve:

| promised | raw delivered | shortfall | conformal delivered | shortfall |
|---|---|---|---|---|
| 50% | 43.0% | −7.0 pts | 48.5% | −1.5 pts |
| 60% | 53.1% | −6.9 pts | 59.8% | −0.2 pts |
| 70% | 63.1% | −6.9 pts | 67.3% | −2.7 pts |
| 80% | 72.9% | −7.1 pts | 81.7% | **+1.7 pts** |
| 90% | 86.8% | −3.2 pts | 90.4% | **+0.4 pts** |

The raw band under-covers at **5 of 5** levels, by a strikingly consistent ~7 points. The
calibrated band's worst miss is **−2.7 points**. See `reports/calibration_raw.png` and
`reports/calibration_conformal.png` — the raw curve sits below the diagonal everywhere,
the calibrated one sits on it.

**This part of the project works.** Measuring coverage takes about fifteen lines and
almost nobody does it; fixing it takes about twenty more.

### 2.4 The band is the right *shape*

Training three separate quantile models — rather than putting one error bar around one
prediction — was supposed to let the band be narrow for a fresher and wide for a CTO. It
did:

| decile of predicted salary | median midpoint | median band width |
|---|---|---|
| bottom | ₹4,48,385 | ₹14,66,924 |
| top | ₹45,19,620 | ₹74,06,481 |

Width in rupees grows **5.0x** from the bottom decile to the top. That is learned from the
data, not assumed. Good.

---

## 3 · What did NOT work

### The band is too wide to quote an offer from. This is the headline negative result.

| | width / midpoint |
|---|---|
| Raw quantile band | median **1.94x** (1.69 – 3.07) |
| **After conformal calibration** | median **2.40x** (2.05 – 3.44) |

Pooled over all 10 splits: median band width **₹36,64,920** around a median midpoint of
**₹13,95,758**. A typical calibrated answer looks like this:

> **₹4,91,660 – ₹39,84,224**  (midpoint ₹13,95,079, 80% band)

Nobody can make an offer from that. It is not a salary band; it is the salary range of the
whole industry, returned with a straight face.

**And notice the trade.** The *raw* band is 1.94x wide and lies about its confidence. The
*calibrated* band is 2.40x wide and tells the truth. Honesty cost us **23.4% more width**.
That is the correct trade and we would make it again — but it means the honest version is
the less usable one, and pretending otherwise would be exactly the failure mode this
project was built to avoid.

**Why this is a data limitation, not a modelling failure.** Three independent lines of
evidence say so:

1. **The model has almost nothing to condition on.** §1 showed seven of the strongest known
   Indian pay drivers are absent from the file — location tier, product-vs-services, and
   institute tier chief among them. Two candidates who are identical on everything the
   survey records can legitimately be 2–3x apart on the things it does not record. A band
   that is honest about that uncertainty *has to be* wide. Narrowing it without adding
   information would just move us back to the overconfident raw band.
2. **The signal that does exist is already extracted.** §4 shows one strong feature and
   two weak ones. There is no unexploited structure sitting in the file waiting for a
   better hyperparameter sweep.
3. **The width is worst exactly where the data is thinnest, and best where it is densest** —
   see the next table. That is the signature of a sample-size problem, not an
   architecture problem.

### The band is proportionally worst where a company hires most

| decile | median midpoint | width / midpoint | coverage |
|---|---|---|---|
| 0 | ₹4,48,385 | **3.44x** | 85.3% |
| 1 | ₹6,01,150 | 3.07x | 85.3% |
| 2 | ₹7,61,280 | 2.87x | 83.8% |
| 3 | ₹9,61,249 | 2.70x | 79.9% |
| 4 | ₹12,35,247 | 2.67x | 73.0% |
| 5 | ₹15,94,769 | 2.36x | 85.3% |
| 6 | ₹21,87,458 | 2.13x | 80.9% |
| 7 | ₹29,13,085 | 1.91x | 85.8% |
| 8 | ₹35,70,381 | 1.71x | 77.9% |
| 9 | ₹45,19,620 | **1.64x** | 85.3% |

Relative width falls monotonically from **3.44x** at the bottom to **1.64x** at the top.
So the band is at its most useless for junior and mid-level candidates — the population a
company hires most of, and the population the tool was meant to serve. See
`reports/band_width_relative.png`.

### A raw quantile band was over-confident by a suspiciously constant amount

Not a failure so much as an unresolved question. The raw shortfall in §2.3 is −7.0, −6.9,
−6.9, −7.1, −3.2 points across five levels. Four of those five are within 0.2 points of
each other. A constant offset like that smells like something systematic — pinball loss on
613 rows with `min_child_samples=25`, most likely — rather than like noise. We fixed the
symptom with conformal calibration without diagnosing the cause. That is listed under open
questions.

---

## 4 · What actually drives salary, and where the model is worst

### 4.1 Three importance measures, which disagree — and the disagreement is the finding

Averaged over 3 splits. Permutation importance is measured on **held-out** rows, in rupees
of added MAE, with 5 shuffles each.

| feature | gain % | split % | permutation (₹ MAE added) | sd |
|---|---|---|---|---|
| `years_experience` | **36.4** | 23.1 | **+3,07,623** | 18,596 |
| `experience_log` | 12.5 | 7.3 | +61,947 | 20,603 |
| `skill_php` | 3.3 | 3.4 | +44,390 | 16,807 |
| `org_size` | 10.0 | 7.8 | +36,634 | 16,810 |
| `remote` | 3.8 | 3.9 | +23,123 | 7,209 |
| `skill_typescript` | 2.5 | 4.1 | +14,455 | 7,418 |
| `experience_sqrt` | 3.9 | 2.2 | +14,182 | 11,531 |
| `skill_go` | 1.9 | 2.5 | +14,070 | 6,490 |
| `role` | 3.4 | 3.4 | +13,306 | 14,161 |
| `n_skills` | 5.8 | **12.3** | **+3,318** | 4,655 |
| `education` | 0.6 | 0.8 | **−2,675** | 3,540 |
| `skill_java` | 1.5 | 3.3 | **−2,756** | 7,504 |

Grouped, because four encodings of experience are one signal wearing four hats and 25 skill
flags are one idea spread thin:

| block | features | gain % | split % | permutation (₹) |
|---|---|---|---|---|
| experience (4 encodings) | 4 | **53.7** | 33.7 | **+3,83,223** |
| individual skill flags | 26 | 22.7 | 38.1 | +1,03,388 |
| categorical columns | 5 | 17.8 | 15.9 | +70,390 |
| `n_skills` | 1 | 5.8 | 12.3 | +3,318 |

**Finding 1 — experience is the model, and everything else is trim.** It is 53.7% of the
gain and about four fifths of the total permutation cost. Twenty-six skill flags together
buy roughly a quarter of what experience buys, and most of that quarter comes from three of
them. `docs/design.md` called experience "the single strongest signal"; on this data it is
close to the *only* signal.

**Finding 2 — split-count importance is biased towards high-cardinality features, and
`n_skills` proves it.** `n_skills` is an integer running 0–20, so a tree can split on it at
many different thresholds, and it climbs to **12.3%** of split count — second place, ahead
of `org_size`. Shuffle it on held-out rows and the model loses **₹3,318**, roughly a
quarter of one percent of its error (**+0.25% of MAE**, against **+23.13%** for
`years_experience`). Split count measures how *many* questions a feature could answer, not
how *useful* the answers were. Anyone reporting LightGBM's default importance is reporting
cardinality as much as signal; this row is why the script computes all three.

**Finding 3 — education is worse than useless.** Shuffling it *improves* held-out error by
₹2,675 (**−0.20% of MAE**). `skill_java` likewise (−₹2,756), as do `experience_bucket`
and `skill_rust`. `design.md` predicted education "usually weaker than people expect
once experience is accounted for". It is weaker than that: on this data the model is
leaning on noise in the column, and taking the noise away helps.

**Finding 4 — most of the feature matrix is dead weight.** Unshuffled held-out MAE is
**₹13,29,995**. Of 37 distinct features, **16 carry no measurable held-out signal at all**,
and only **9** move MAE by more than 1% of it (about ₹13,300). That is 36 features per
fitted model, 9 of them earning their place, on 613 training rows.

**A caveat on `skill_php`, which ranks third and should not be trusted.** PHP plausibly
marks lower-paid services work, which would be a real and interesting effect. But it rests
on however many PHP developers happen to land in a ~200-row test set, and its seed-to-seed
standard deviation is ₹16,807 on a mean of ₹44,390. Treat it as a hypothesis to test on
more data, not as a finding. The same caution applies to every skill flag in the table.

`reports/feature_importance.png` has the top 12.

### 4.2 Where the model is worst — by experience

Pooled across the 10 splits. `bias` is `exp(mean log residual) − 1`: **positive** means the
model predicts *less* than these people actually earn; **negative** means it predicts
*more*.

| bucket | pooled n | per split | MAE | MAPE | coverage | width/mid | bias |
|---|---|---|---|---|---|---|---|
| 0–1 | 215 | 21.5 | ₹6,00,040 | 70% | 79.1% | 3.15x | +4.1% |
| 2–3 | 439 | 43.9 | ₹6,67,383 | 65% | 82.9% | 2.93x | +16.6% |
| 4–5 | 351 | 35.1 | ₹9,98,161 | 80% | 83.2% | 2.55x | +5.5% |
| 6–8 | 332 | 33.2 | ₹11,88,124 | 116% | 83.7% | 2.14x | −7.6% |
| 9–12 | 345 | 34.5 | ₹17,91,690 | 84% | 82.6% | 2.01x | +1.2% |
| 13–20 | 277 | 27.7 | ₹19,41,934 | 171% | 83.4% | 1.74x | −11.0% |
| **20+** | **69** | **6.9** | **₹31,63,258** | **227%** | **73.9%** | 2.00x | **−21.1%** |

*(Rows recur across seeds, so "pooled n" is "times scored", not an independent sample size.
The per-split column is the honest one.)*

**The 20+ bucket is where everything goes wrong at once.** Highest MAE, worst MAPE at
227%, the **only** slice where the calibrated band breaks its 80% promise (73.9%), and a
systematic **−21%** bias — the model consistently tells you a 25-year veteran earns more
than they do. All four symptoms have one cause: about **7 people per split**.

Error rising with experience *in rupees* is expected and fine — senior salaries are bigger
and genuinely more variable. The coverage collapse is not fine, and it is invisible in the
headline 82.3%.

### 4.3 Where the model is worst — by role

| role | pooled n | per split | MAE | MAPE | coverage | bias |
|---|---|---|---|---|---|---|
| QA | 18 | 1.8 | ₹16,72,731 | **583%** | **55.6%** | **−64.2%** |
| Student | 28 | 2.8 | ₹10,39,370 | 278% | **57.1%** | −36.1% |
| Analytics | 19 | 1.9 | ₹6,20,830 | 236% | 89.5% | −26.9% |
| Other | 399 | 39.9 | ₹16,39,485 | 109% | 80.5% | +4.1% |
| Mobile | 100 | 10.0 | ₹8,02,018 | 99% | 86.0% | −2.8% |
| Fullstack | 682 | 68.2 | ₹10,56,793 | 97% | 82.6% | −2.8% |
| Backend | 399 | 39.9 | ₹12,83,745 | 93% | 82.5% | +2.0% |
| Frontend | 151 | 15.1 | ₹7,73,011 | 79% | 89.4% | +0.2% |
| Management | 62 | 6.2 | **₹30,53,366** | 64% | **74.2%** | **+30.6%** |
| Embedded | 24 | 2.4 | ₹11,90,062 | 50% | 83.3% | +43.0% |
| Data Science | 31 | 3.1 | ₹10,22,348 | 49% | 83.9% | +18.3% |
| Data Engineering | 51 | 5.1 | ₹11,01,361 | 46% | 86.3% | +26.9% |
| DevOps | 67 | 6.7 | ₹11,52,459 | 44% | 86.6% | +12.8% |
| Security | 9 | 0.9 | ₹7,62,169 | 38% | 77.8% | **+56.1%** |

**Mean absolute bias in roles with fewer than 5 test rows per split: 40.8%. In the rest:
10.3%.**

Read that as the summary of the whole section. The error is not spread evenly across a
mediocre model — it is a decent model on Fullstack, Backend and Frontend (biases of −2.8%,
+2.0%, +0.2%) sitting next to a catastrophic one on everything else.

The worst single cell is **QA**: 56% coverage against an 80% promise, and a **−64%** bias
meaning the model routinely predicts nearly three times what a QA engineer actually earns.
The survey contains 8 QA respondents. **Security** is the mirror image at **+56%** — the
model under-predicts them by more than half — on 5 respondents.

**And this is the part that makes it actionable rather than academic.** The roles the model
is worst at are the roles a company would most want a band for, because they are the ones
nobody in the room has an intuition about. Everyone knows roughly what a backend engineer
costs. Nobody is sure about a security engineer, which is exactly where the model is
+56% wrong and 78% covered.

Concentrated error is fixable — collect 200 QA rows. Average error is not.
`reports/error_by_role.png` and `reports/error_by_experience.png`.

---

## 5 · What the data cannot answer at all

### The fairness audit works. There is nothing in the public data to point it at.

This is the strongest single argument in the project for getting company data, so it gets
stated precisely.

**The 2025 Stack Overflow survey has 172 columns and zero protected-attribute columns.**
The script checks the real header for `gender`, `ethnic`, `race`, `sexuality`,
`orientation`, `transgender`, `disability`, `accessibility`, `nationality`, `religion`,
`caste`. Every one returns **0 columns**. The only demographic field of any kind is `Age`,
recorded as a band.

This is not "the column has missing values". **The question is not in the survey.** There
is no cleaning step, no imputation, no clever join that recovers it.

So: the audit in `src/paybands/fairness/` — which §5.2 and §5.3 show works to within a
third of a percentage point — **cannot be run on the public data at all.** That is not a
flaw in the code. It is a fact about the dataset, and it means a whole category of question
about this model is unanswerable until different data arrives.

> "We don't collect it" is not a fairness strategy. It removes your ability to *check*, not
> the effect. §5.3 shows the effect surviving perfectly well without the column.

### 5.1 The one demographic we do have, and what it can't tell us

`Age` is present, so we ran the audit on it: 25–34 year olds against 35–44 year olds, 702
people with recorded experience.

| | |
|---|---|
| Raw gap (mean pay) | **43.9%** — ₹23,74,814 vs ₹42,31,796 |
| Adjusted gap, controlling for experience, role, education, org size, remote, employment type | **−5.22%** |
| 95% confidence interval | **−33.6% to +17.1%** |
| n | 702 (533 / 169), 62 parameters, R2 = 0.451 |

Two lessons in one number.

**First: a raw gap is a fact about who the groups *are*, not about how they are treated.**
A 43.9% headline gap becomes a −5.2% adjusted gap. Of the 43.9 points, **49.1 points** are
accounted for by the controls — the 35–44s simply have more experience. Reporting the raw
number as evidence of age discrimination would have been straightforwardly wrong, and it is
the single most common mistake in this area.

**Second: 702 people cannot answer this question.** The interval spans **51 percentage
points**. It contains −30% and it contains +15%. The honest reading is *"we cannot tell"*,
not *"no gap"*, and those are different sentences.

### 5.2 The audit is calibrated against a known truth

You cannot validate a bias detector on real data, because nobody knows the true gap. So we
generate a world where we wrote the rules, inject a gap we chose, and demand it back.

| injected | recovered | 95% CI | error | truth inside CI? |
|---|---|---|---|---|
| **0%** | **−0.34%** | −1.39% to +0.71% | −0.34 pts | yes |
| 3% | 2.67% | 1.65% – 3.68% | −0.33 pts | yes |
| 8% | 7.69% | 6.72% – 8.65% | −0.31 pts | yes |
| 15% | 14.71% | 13.81% – 15.60% | −0.29 pts | yes |
| 25% | 24.75% | 23.95% – 25.53% | −0.25 pts | yes |

Largest error across all five: **0.34 percentage points**. Truth inside its interval:
**5 of 5**.

The row that matters most is the first. At an injected gap of **zero**, the audit reports
**−0.34%** — it does not invent bias in fair data. An audit that cries wolf is worse than
no audit, because it is confidently wrong in the direction that gets someone accused.

`reports/fairness_validation.png` shows the 8% case: injected 8.0%, raw gap 10.9%,
recovered 7.69%.

**And the same run with the proxy channel left out of the controls:**

| injected | recovered | error | truth inside CI? |
|---|---|---|---|
| 0% | 2.91% | +2.91 pts | no |
| 3% | 5.82% | +2.82 pts | no |
| 8% | 10.68% | +2.68 pts | no |
| 15% | 17.47% | +2.47 pts | no |
| 25% | 27.18% | +2.18 pts | no |

Every estimate too high by 2.2 to 2.9 points; **0 of 5** intervals contain the truth.

Neither table is "wrong" — they answer different questions. Controlling for career breaks
says *"time out of the workforce is a legitimate reason to pay less."* Not controlling for
it says *"who takes that time off is itself the unfairness."* Both positions are arguable,
they give different answers, and the honest thing is to publish both numbers rather than
quietly pick the comfortable one.

### 5.3 Deleting the gender column does not delete the bias

The claim this demolishes: *"we removed gender from the training data, so the model can't
be biased."*

Same model fitted twice on synthetic data with an 8% injected gap, once with the gender
column and once without, gap measured on held-out rows:

| seed | gap in the data | predictions **with** gender | predictions **without** | surviving | accuracy cost |
|---|---|---|---|---|---|
| 0 | 10.8% | 10.5% | 3.6% | 34.3% | +2.3% |
| 1 | 10.0% | 9.4% | 3.8% | 41.0% | +0.7% |
| 2 | 9.3% | 11.4% | 4.5% | 39.8% | +0.8% |

**Gap surviving deletion: mean 38.4% (34.3% – 41.0%). Accuracy given up for it: mean 1.3%.**

Deleting the protected column removed about three fifths of the modelled gap and left the
rest — delivered by career breaks, a feature that both predicts pay honestly *and* smuggles
group membership. And it did so at essentially no accuracy cost, so nothing was even traded
away for the harm that remains.

The proxy is visible before any model is trained. A scan of every column against gender:

| feature | kind | association | strength |
|---|---|---|---|
| `career_gap_months` | numeric | **0.367** | strong |
| `salary_annual` | numeric | 0.103 | weak |
| `prev_company_type` | categorical | 0.025 | negligible |
| everything else | | 0.004 or less | negligible |

One channel, in a dataset we built with exactly one channel. **A real dataset has several
and nobody labels them** — which is why this scan should be the first thing run on company
data, before a line of modelling.

---

## 6 · What would change with company data

Quantified where the evidence supports it, flagged as a guess where it does not.

| what changes | evidence from above | expected effect |
|---|---|---|
| **The fairness audit becomes runnable at all.** | §5: 0 protected columns in 172. | This is binary, not incremental. Today the answer is "unanswerable"; with a `gender` column it is a number with a confidence interval. Nothing else on this list matters as much. |
| **Location tier enters the model.** | §1: absent. design.md calls the tier-1/tier-2 gap one of the largest in Indian pay. | Should take a real bite out of the 2.40x band width — a chunk of what the model currently calls "irreducible uncertainty" is Bangalore-vs-Indore. Size unknown; this is the single most valuable missing column after the protected attribute. |
| **Product vs services enters the model.** | §1: absent. design.md: "two people with identical experience can be 35% apart on this alone." | If that 35% figure is even roughly right, this alone explains a large slice of the current residual spread. |
| **Thin roles get populated.** | §4.3: mean absolute bias 40.8% in roles with under 5 test rows/split vs 10.3% elsewhere. | The concentrated error is a sample-size problem with a known fix. A company with 200 QA engineers gets a QA band; the survey's 8 respondents never will. |
| **Non-response disappears.** | §1: 50.1% skipped the salary question. | Company payroll has no non-response and no self-reporting. Both of the largest caveats in §7 simply cease to apply. |
| **A temporal split becomes possible.** | §1: no `event_date`. | Company hire dates enable `split.temporal_split`, which is *mandatory* once rows have dates. Expect the honest score to be **worse** than the numbers here — that lower number is the true one. |
| **The `prev_salary` experiment becomes possible.** | §1: absent. | design.md §4.1 promises training with and without last-drawn CTC and comparing. Cannot be done on public data at all. |
| **The band might become decision-grade.** | §3. | This is the bet. The band is 2.40x wide because the model is conditioning on four or five columns. Add five strong ones and it should narrow substantially — but *how much* is unknown, and it is possible that Indian salary variance at fixed observables is simply large. That would itself be worth knowing. |

`docs/company-data-request.md` should be cut down to the fields that earned their place
here: experience above all, then org size, remote, role — plus the ones we could not test
but have strong prior reason to want (location tier, previous company type, level) and the
protected attribute the audit needs. Education can be dropped; §4.1 shows it actively hurts.

---

## 7 · Caveats that apply to every number above

Stated here rather than buried, because a result handed over without its caveats gets
repeated without its caveats.

1. **Self-reported.** Nobody audited these salaries. People round up, some report CTC where
   others report base, and one person typed 23 nines. We cleaned the last kind; we cannot
   clean the first two.

2. **50.1% non-response — and this is selection bias, not missing data.** Half the Indian
   respondents who got as far as the currency question declined to state a salary. If the
   people who declined differ systematically from those who answered — and "I'd rather not
   say" is very unlikely to be independent of the number — then every figure here describes
   a self-selected subset. No cleaning rule fixes this. It is the largest single caveat on
   the project.

3. **Stack Overflow respondents are not a random sample of Indian developers.** They skew
   towards English-speaking, internationally-oriented, senior developers who participate in
   a Western developer community. Median experience in the surviving sample is 6 years.
   **Salaries here are probably overstated relative to the actual Indian market**, and a
   band built from them will quote high. We cannot measure by how much.

4. **Three roles are 72.5% of the data.** Anything the model says about the other eleven
   rests on single-digit or low-double-digit counts, and §4.3 shows what that produces.

5. **Pooled test rows are not independent.** The breakdowns in §4 pool across 10 splits so
   thin slices have enough rows to describe at all. A row appears in roughly 2 of the 10
   test sets, so a "pooled n" of 69 is not 69 independent observations. The per-split
   column is the honest sample size.

6. **A random split is only valid because this data has no time order.** Everyone answered
   within the same few weeks. The moment company data with hire dates arrives, a random
   split becomes leakage and `temporal_split` is mandatory.

7. **The synthetic results validate the *audit*, not the *world*.** §5.2 and §5.3 prove our
   measuring instruments work on data whose rules we wrote. They say nothing about the size
   of any real pay gap anywhere.

8. **The control list in the fairness audit is a judgement call, and it is contestable.**
   Control for too little and you report a gap that is really an experience difference.
   Control for too much — including a factor that is itself unfairly distributed, like who
   gets put in which role — and you subtract the discrimination from the answer. There is
   no formula that settles this. Ours is printed with every result so people can argue with it.

---

## 8 · Open questions, and what I would do next

**In order of what I would actually do first.**

1. **Get the company data, and get a protected attribute with it.** Everything in §5 is
   blocked on this, and it is blocked absolutely rather than partially. Nothing else on
   this list changes the project as much.

2. **Diagnose the constant −7 point coverage shortfall.** §2.3: the raw band under-covers
   by −7.0, −6.9, −6.9, −7.1 points at four of five levels. That consistency is too neat to
   be noise. My guess is pinball loss on 613 rows with `min_child_samples=25` systematically
   pulling the extreme quantiles inward. Testable: sweep `min_child_samples` and see whether
   the shortfall tracks it. We fixed the symptom without understanding the cause, which is
   fine engineering and unfinished science.

3. **Prune the feature matrix and re-measure.** §4.1: 16 of 37 features carry no held-out
   signal and four actively hurt. Refit with the 9 that earn their place and see whether MAE
   and band width improve. On 613 rows, capacity spent on noise is capacity not spent on
   signal — but this is a hypothesis, and the point of this document is not to state
   hypotheses as findings.

4. **Ask whether the band should be conditional on confidence.** Right now every candidate
   gets a band, including the 20+ veteran the model over-predicts by 21% with 74% coverage.
   A tool that returned *"we don't have enough data about people like this"* for thin slices
   would be more useful than one that returns a confident-looking ₹5L–₹40L. The lookup
   table already reports which rung of its fallback cascade fired; the band model should
   report something equivalent.

5. **Run the `prev_salary` experiment as soon as the column exists.** design.md §4.1 sets
   it up: train with and without last-drawn CTC, compare accuracy and fairness. The
   prediction there is that the version *without* is nearly as accurate while being far
   fairer. It is currently untested in either direction, and it would be the strongest
   section in the README if it holds.

6. **Investigate `skill_php`.** Third by permutation importance (+₹44,390) with a
   standard deviation of ₹16,807. Either PHP is a genuine marker of lower-paid services work
   in the Indian market — which would be a real finding — or it is an artefact of a
   200-row test set. Currently we cannot tell.

7. **Test whether MAPE is the right metric here at all.** §4.2 reports MAPE of 227% for the
   20+ bucket, which is arithmetically correct and rhetorically useless. MAPE punishes
   over-prediction far more harshly than under-prediction, and the model over-predicts
   exactly where MAPE is worst. A symmetric alternative (log-space error, or SMAPE) might
   describe the same failure less misleadingly.

---

*Regenerate everything in this document:* `uv run python scripts/run_analysis.py`
*Charts land in `reports/` (gitignored — they are derived artefacts, rebuilt from code).*
