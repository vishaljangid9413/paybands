# paybands — Salary Band Prediction

> **This is a learning project.** Every phase below explains *what* we build, *why* we build it
> that way, and *what you learn*. Plain language. Terms get defined the first time they appear.

**Started:** 2026-08-07 · **Target:** 5–6 weeks · **Machine:** M1 / 8GB — no GPU needed

---

## Part 1 — What we're building, in plain words

A model that answers: **"What should this person be paid?"**

But notice the trap in that sentence. If the model answers "₹21,34,500", it sounds precise and
scientific — and it's fake precision. Nobody can predict a salary to the rupee. Two equally good
engineers at the same company can legitimately be paid 20% apart.

So our model answers differently:

> **"₹18L – ₹24L, and I'm reasonably confident about that."**

That's a **band**. It's honest about what we don't know, and it's what a recruiter can actually
use in a negotiation.

### Indian salaries, and what exactly we predict

This model is built for the Indian market, and specifically for your company's structure:

```
  Base salary          ← the model predicts THIS, and only this
− PF deduction
− Insurance deduction
− Income tax (TDS)
= Net take-home        ← calculated by formula, never predicted
```

**Read [`docs/design.md`](docs/design.md) before any code.** It explains why the model predicts base
salary and never take-home — short version: PF, insurance and tax are *arithmetic*, and making a
model learn arithmetic is both less accurate and actively harmful (it confuses "the market moved"
with "the Budget changed taxes"). The project is three layers: a **learned** model, a **calculated**
payroll layer, and a **policy** rules layer for increments.

### Two things a company would use it for

**1. Making an offer.** A candidate walks in. Recruiter enters role, experience, location, skills.
Model returns a band. Recruiter makes an offer inside that band. No more "what did we pay the last
guy?" guesswork.

**2. Finding underpaid employees.** Run every current employee through the model. Anyone whose
actual salary sits *below* their predicted band is potentially underpaid. HR gets a list, and a
number: "fixing all 14 of these costs ₹32L a year." This is the use case that makes companies
actually want it — it turns a vague fairness worry into a budget line.

Your increment use case falls out of this second one for free: if someone is below band, the
increment that brings them to band is the recommendation. **No second model needed** — it's a
rules layer on top of the band. (We'll build that layer, but it's policy written as code, not
machine learning, and knowing the difference matters.)

The number that ties both use cases together is the **compa-ratio**:

```
compa-ratio = actual salary ÷ midpoint of predicted band
```

Below 0.90 → underpaid. 0.90–1.10 → in band. Above 1.10 → above band. That single division powers
the pay-equity list *and* the increment recommendation. It's standard HR vocabulary — using it
correctly makes the tool sound like it was built by someone who understands compensation.

---

## Part 2 — Why this isn't just "train a regression model"

Three things separate this from the thousands of salary-prediction notebooks on GitHub. Each one
is a real skill, and each gets its own phase.

### 2.1 The model must say how sure it is

Most models output one number. Ours outputs a range, and — critically — we **check whether the
range is honest.**

Here's what that means. If we say "80% confident the true salary is in this range," then across
100 people, roughly 80 of their real salaries should land inside their predicted range. If only
55 do, our model is overconfident and lying to the recruiter.

Measuring this is called **checking coverage**. It takes about twenty lines of code. Almost nobody
does it. Doing it — and reporting the result honestly, even if it's bad the first time — is a
genuinely senior habit.

### 2.2 The model will learn the company's past unfairness unless we stop it

This is the important one. Read it slowly.

A model learns patterns from history. If your company has historically paid one group less than
another for the same work, **the model learns that as a pattern** and applies it to new hires.

The output then looks objective — it's a number, from a computer, with maths behind it. But it's
just the old bias, laundered. And now it's harder to challenge, because "the model said so."

In a hiring context, that's not only wrong, it's a legal risk in many countries.

**So we audit for it.** We measure whether the model's predictions differ across groups for people
with the same experience, role, and location. And here's the part most people get wrong: **simply
deleting the gender column doesn't fix it.** The model finds proxies — career gaps, job title
patterns, part-time history — and reconstructs the bias anyway. We'll *demonstrate* this, which is
a far better README section than claiming we "removed bias."

### 2.3 We build fake data on purpose — and that's the smartest part

Here's a problem: how do you know your bias audit works?

On real data, you can't. You don't know the true size of the pay gap, so you can't tell whether
your audit found the right answer, missed it, or invented one.

**Solution: build our own dataset where we decide the truth.** We write a generator that creates
employees and pays them according to rules *we* write — including a deliberate 8% gap for one
group.

Then we run the audit. If it reports ≈8%, the audit works. If it reports 2% or 15%, our audit is
broken and we fix it *before* pointing it at real people.

This is called **validating your measurement against known ground truth**, and it's how serious
engineering works. You calibrate the thermometer in boiling water before trusting it on a patient.

---

## Part 3 — The data plan

Three sources, in order. We're never blocked waiting for anyone.

| Stage | Source | Why |
|---|---|---|
| **1. Public** | Stack Overflow Developer Survey | Free, large, real, has salary + experience + role + country + education. Gets the pipeline working today. |
| **2. Synthetic** | Our own generator | We control the truth, so we can validate the fairness audit and the interval calibration. |
| **3. Company** | Your employer, later | The real prize. Requested formally once we know which fields actually matter. |

**Honest limits of the public data**, to state in the README rather than hide: it's self-reported
(people round up), developer-skewed (not the whole company), and global (so we filter to India,
and maybe keep the US as a comparison).

The formal data request for stage 3 lives in
[`docs/company-data-request.md`](docs/company-data-request.md) — a draft is ready now, because
company data requests take weeks to get approved and you should start that conversation early.

---

## Part 4 — The phases

### Phase 0 · Scaffold — week 1

**What:** repo, environment, config system, a place to store results, tests, CI.

**Why:** so that in week 4 you can answer "which settings produced this number?" Without this,
you end up with `model_final_v3_REAL.ipynb` and no idea what's in it.

**Learn:** project structure, reproducibility, why experiments need IDs.

---

### Phase 1 · Public data + the baselines — week 1–2

**What:** download the survey, clean it, explore it, and build two deliberately dumb models.

**Why baselines matter — this is a big idea:**

Before building anything clever, build something stupid:

- **Baseline 0:** predict the same median salary for everybody. Terrible, but it's the floor.
- **Baseline 1:** a **lookup table** — the median salary of everyone with the same (role, level,
  country). No machine learning at all, just `groupby`.

Baseline 1 is usually surprisingly good, and it's the number your fancy model has to beat. If
gradient boosting only ties the lookup table, **the model isn't earning its complexity** and the
right engineering decision is to ship the lookup table.

Most portfolios skip this and report "R² = 0.87!" with nothing to compare against, which means
nothing. Beating a real baseline is a claim; a bare R² is not.

**Also in this phase — two things that will bite you:**

*Skewed target.* Salaries have a long right tail: most people cluster low, a few earn enormously.
This wrecks averages. We predict **log(salary)** instead, which turns "20% raise" into a constant
step and makes the maths behave. You'll see the before/after distribution and understand why.

*Messy categories.* "Job title" has hundreds of values, most appearing once or twice ("Senior
Backend Engineer II", "Sr. Backend Eng"). We'll cover how to encode these without **leakage** —
leakage means accidentally letting information from the test set into training, which makes your
model look brilliant in testing and fail in production. It is the single most common serious
mistake in applied ML.

**Learn:** EDA, skewed targets and log transforms, high-cardinality categorical encoding,
leakage, and the discipline of baselines.

---

### Phase 2 · The synthetic data generator — week 2

**What:** write code that invents realistic employees and pays them by rules we define — including
a deliberate, known pay gap.

**Why:** covered in 2.3. This is our test rig for everything that follows.

**Learn:** thinking explicitly about how data is *generated* rather than just consuming it — which
is what separates someone who understands their model from someone who feeds it CSVs.

---

### Phase 3 · The model and the bands — week 3

**What:** gradient boosting (LightGBM or CatBoost), then turn point predictions into intervals.

**How the intervals work, simply:** instead of training one model to predict the middle, we train
models to predict the **10th percentile** and the **90th percentile** — the low end and high end.
That gap is the band. Then we apply **conformal prediction**, a technique that mathematically
adjusts the band so the coverage promise actually holds.

Then we check coverage (from 2.1) and plot it. If our 80% band only covers 62%, we say so and fix it.

**Learn:** gradient boosting and how to tune it, quantile regression, conformal prediction,
calibration checking, proper metrics (MAE in rupees — interpretable to a human — plus pinball loss
for quantiles).

---

### Phase 4 · The fairness audit — week 4

**What:** measure whether predictions differ across groups at equal experience, role and location.
Validate the audit against the synthetic data's known gap. Then demonstrate the proxy problem:
delete the gender column and show the gap survives.

**Learn:** fairness metrics for regression, residual analysis by group, the proxy-variable problem,
and how to write about a sensitive result carefully and honestly.

---

### Phase 5 · Payroll calculator and increment rules — week 5

**What:** Layers 2 and 3 from `docs/design.md`. No machine learning in this phase at all.

*The calculator:* base salary → PF (12% of basic, with the ceiling question settled), insurance,
income tax by the current year's slabs, professional tax if applicable → net take-home. All values
in `configs/tax/fy_YYYY_YY.yaml`, one file per financial year, **never edited once written** — so
you can always recompute what someone's take-home *was* in a past year.

*The increment rules:* current salary + compa-ratio + performance rating + budget → recommended
raise, with the reasoning spelled out in words a recruiter can read.

**The test that matters:** feed a real payslip's base salary in, and check every deduction matches
to the rupee. If it doesn't, the calculator is wrong — and unlike the model, there's no "close
enough" here.

**Learn:** Indian payroll mechanics, config-driven business rules, and the judgement to recognise
when a problem is arithmetic rather than machine learning. That judgement is worth more than
another model.

---

### Phase 6 · Explanations and serving — week 6

**What:** **SHAP** — a method that explains a single prediction by showing how much each feature
pushed it up or down: *"Predicted ₹22L — 8 years experience added ₹6L, Bangalore added ₹3L, coming
from a services company subtracted ₹2L."*

Then a FastAPI service — your home turf. Input a candidate, get back the band, the explanation, the
computed take-home, and a confidence flag.

**Why explanations are non-negotiable here:** a recruiter must be able to *argue with* the model. A
salary number with no reasoning attached can't be challenged — and an unchallengeable salary number
is exactly the thing that causes harm. It's also what makes the tool defensible if anyone ever
questions a decision.

**Learn:** SHAP, model serving, and designing an API whose output a non-technical person can act on.

---

### Phase 7 · Company data, monitoring, write-up — week 6–7

**What:** cut the data request down to the fields that actually earned their place. Add drift
monitoring — salary markets move, and a 2026 model will quietly rot by 2028. Write the README.

**Learn:** turning a model into a proposal a business will approve, monitoring, and technical
writing.

---

## Part 5 — The vocabulary, in one place

Terms you'll meet, in plain words. Come back here when one stops making sense.

| Term | Plain meaning |
|---|---|
| **Target** | The thing we're predicting. Here: salary. |
| **Feature** | An input the model uses. Experience, role, city. |
| **Baseline** | A deliberately simple model your real model must beat to justify itself. |
| **Skewed / long tail** | Most values are small, a few are huge. Breaks averages. |
| **Log transform** | Predicting log(salary) instead of salary, so percentage changes become even steps. |
| **Leakage** | Test-set information sneaking into training. Makes results look great and production fail. |
| **High cardinality** | A category column with very many distinct values (job title). |
| **Quantile** | A cut point in a distribution. The 90th percentile = 90% of people earn less than this. |
| **Conformal prediction** | A method that makes an interval's confidence promise mathematically honest. |
| **Coverage** | Do your 80% intervals actually contain the truth 80% of the time? |
| **Calibration** | Whether stated confidence matches reality. |
| **SHAP** | Explains one prediction by attributing it across features. |
| **Proxy variable** | A feature that secretly encodes a protected one (career gap ≈ gender). |
| **Drift** | The world changes, the model silently gets worse. |
| **MAE** | Average error in rupees. Interpretable. |
| **Gradient boosting** | Many small decision trees, each fixing the last one's mistakes. Best-in-class for tabular data. |

---

## Part 6 — Done means

- [ ] Beats the group-median lookup baseline, with the margin stated
- [ ] Intervals with measured coverage, plotted — not assumed
- [ ] Fairness audit validated against synthetic data with a known injected gap
- [ ] The proxy demonstration: dropping gender does not remove the gap
- [ ] Per-prediction SHAP explanations
- [ ] FastAPI service returning band + explanation + confidence
- [ ] A data-request document your company could actually act on
- [ ] README leading with **findings**, including what surprised you and what you got wrong

---

## Part 7 — Findings log

Keep this from day one. It becomes the README and your interview material. Negative results count.

| Date | Finding | Evidence |
|---|---|---|
| | | |
