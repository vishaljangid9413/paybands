# Design — Indian salary structure, and what the model actually predicts

Plain language. Read this before writing any code.

---

## 1. The most important design decision in the project

Your company's salary has four parts:

```
  Base salary                    ← the number you negotiate
− PF deduction                   ← employee provident fund
− Insurance deduction
− Income tax (TDS)               ← per Indian government rules
─────────────────────────────
= Net take-home
```

Now the question that decides the whole architecture: **which of these does the model predict?**

**The model predicts base salary. Nothing else.** Everything below the first line is *calculated*,
not learned.

### Why — the rule to remember

> **Never make a model learn something you can compute.**

PF is a percentage. Insurance is a fixed amount or a rate. Income tax is a published formula from
the government. These are **arithmetic**. They are exactly known.

If you make a model learn arithmetic, three bad things happen:

1. **It gets it slightly wrong.** A model approximates. A formula is exact. You'd be introducing
   error into the one part of the system that has no uncertainty in it.
2. **It breaks every February.** The Union Budget changes tax slabs most years. A model would need
   retraining and revalidating. A formula in a config file needs one line edited.
3. **It confuses two different things.** This is the subtle one. Suppose take-home salaries drop
   next year because tax slabs changed. A model trained on take-home sees the drop and concludes
   *"engineers are worth less now."* They aren't — the government just took more. The model has
   mixed up **market value** with **tax policy**, and every prediction afterwards is poisoned.

So we split the system in two:

| Part | How it works | Changes when |
|---|---|---|
| **Salary band model** | Learned from data | The job market moves |
| **Deduction calculator** | Plain arithmetic from a config file | The Budget changes |

This separation is a genuinely senior instinct. Most beginners try to make one model do everything,
then can't explain why the numbers drift. Being able to say *"I split the learned part from the
computed part, and here's why"* is a strong interview moment.

---

## 2. The three layers

```
┌─────────────────────────────────────────────────────────────┐
│  LAYER 1 — MODEL (learned)                                  │
│  candidate details  →  base salary band                     │
│                        ₹18,00,000 – ₹24,00,000              │
│  This is the only machine learning in the project.          │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│  LAYER 2 — CALCULATOR (arithmetic, config-driven)           │
│  base salary  →  PF, insurance, income tax  →  take-home    │
│  Exact. Testable against real payslips.                     │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│  LAYER 3 — DECISION RULES (policy, not ML)                  │
│  current salary + band + performance + budget               │
│                       →  recommended increment              │
│  This is HR policy written as code. It should be readable   │
│  by an HR person, and arguable by them.                     │
└─────────────────────────────────────────────────────────────┘
```

Layer 3 deserves a note. It is tempting to build a second model for increments. **Don't.**
Increments are a *policy decision* — how much budget exists, how performance maps to money, whether
to correct underpayment this year or next. Those are choices your company makes, not patterns
hiding in data. Writing them as explicit rules means HR can read them, disagree with them, and
change them. A model would hide those choices inside numbers nobody can argue with.

**Knowing when *not* to use machine learning is part of being good at machine learning.**

---

## 3. Deductions — how each one works

Every number here goes in a **config file**, never in code. Rules change; code shouldn't have to.

### 3.1 Provident Fund (PF)

The employee contributes **12% of basic salary**. The employer contributes 12% too, but that's the
employer's cost — it isn't deducted from the employee, so it doesn't appear in your take-home
calculation.

One wrinkle: there's a statutory wage ceiling (₹15,000/month) that caps the *mandatory*
contribution. Many companies apply the 12% to actual basic instead, which is more generous. **Ask
your HR which your company does** — it changes the number meaningfully, and it's a one-line config
setting either way.

### 3.2 Insurance

Usually a fixed premium, sometimes varying by grade or family size. Config value. Simple.

### 3.3 Income tax

The complicated one, and the one that changes most.

India currently has two regimes — the **new regime** (default, lower rates, almost no deductions)
and the **old regime** (higher rates, but you can subtract investments, HRA, and so on). Employees
choose. That choice changes their tax, which changes their take-home — **without changing their
market value at all.**

That's one more reason the model predicts base salary. Take-home depends on personal financial
choices; base salary doesn't.

> ⚠️ **Verify the current slabs yourself before using them.** Tax rules change in the Union Budget
> every year, and I won't hardcode numbers you'd then trust blindly. Get them from
> `incometax.gov.in` for the current financial year and put them in
> `configs/tax/fy_2026_27.yaml`.
>
> The config file is versioned by financial year on purpose. Next year you add a new file — you
> never edit history. That means you can always recompute what someone's take-home *was* in 2025,
> which matters for auditing.

### 3.4 Professional tax

You didn't mention it, but it exists and it's state-specific (Karnataka, Maharashtra, West Bengal
and others levy it; some states don't). Small — a few hundred rupees a month. I've left it in the
config as optional, defaulting to zero. Turn it on if your payroll deducts it.

**How we'll know the calculator is right:** test it against a real payslip. Feed in a known base
salary, check that PF, tax and net match the payslip to the rupee. That's a proper test, and it's
the kind of grounding that makes a project believable.

---

## 4. What goes into the model — "all the reasons"

You asked to cover every reason a salary is set or raised. Here they are, split by the two decisions.

### 4.1 Reasons a *hire* salary is what it is

Some of these are India-specific and matter a great deal here.

| Driver | Why it matters | Notes |
|---|---|---|
| **Total experience** | The single strongest signal | Not linear — the first 5 years pay off far more steeply than years 15–20 |
| **Relevant experience** | 8 years total but 2 in this stack ≠ 8 relevant years | Often stronger than total experience |
| **Role & job family** | Backend ≠ QA ≠ Data | |
| **Level / grade** | Internal ladder position | |
| **Location tier** | Bangalore, Mumbai, Delhi-NCR, Hyderabad, Pune, Chennai pay well above tier-2 cities | Group into tiers rather than using raw city names — too many rare values otherwise |
| **Skills** | Certain skills carry a real premium | This premium moves year to year, which is a drift problem we'll monitor |
| **Education level** | Degree | Usually weaker than people expect once experience is accounted for |
| **Institute tier** | IIT / NIT / BITS / IIIT carry a measurable premium in India | Fades with experience — worth *showing* this in the analysis, it's a satisfying finding |
| **Previous company type** | **Product company vs services company is one of the biggest pay gaps in Indian tech** | Must be included. Ignoring it will make your model badly wrong. |
| **Previous salary** | Offers routinely anchor to last drawn CTC | ⚠️ See the warning below — this one is dangerous |
| **Notice period** | Immediate joiners often get a premium | |
| **Hire source** | Referral / agency / campus / direct all price differently | |
| **Competing offers** | The strongest short-term lever there is | Rarely recorded, but ask |
| **Interview scores** | If your process records them | |
| **Hiring urgency** | Backfilling a critical role costs more | Usually not recorded — flag as a known blind spot |
| **Market timing** | A 2021 hire and a 2023 hire are not comparable | Handled by including the date and validating across time |

> ### ⚠️ The previous-salary trap — read this twice
>
> Anchoring an offer to someone's last salary is standard practice in India. It is also the single
> most effective way to make unfairness permanent.
>
> Think it through. Someone is underpaid at their first job — maybe they negotiated poorly, maybe
> they came from a less-known college, maybe something worse. They switch jobs. The new offer is
> "last drawn + 30%". Still underpaid, now with a bigger number. Next switch, same. **The original
> unfairness follows them for their entire career, compounding.**
>
> If we feed `previous_salary` into the model, it learns to do exactly this, at scale, automatically.
>
> So we'll do something specific: **train two models, one with previous salary and one without,
> and compare them.** My expectation is that the version *without* it is nearly as accurate — because
> experience, role and location already explain most of the variation — while being far fairer.
>
> If that turns out to be true, it's a genuinely valuable finding and the strongest single section
> in your README. If it turns out to be false, that's worth knowing too, and worth reporting
> honestly.

### 4.2 Reasons an *increment* is what it is

| Driver | Why it matters |
|---|---|
| **Performance rating** | The official reason |
| **Compa-ratio** | The real reason — see below |
| **Time since last increment** | 18 months without one usually means a bigger correction |
| **Tenure** | Long-tenure employees often fall behind new hires — "salary compression" |
| **Promotion vs merit** | A level change is a different sized jump from an annual raise |
| **Company budget** | There's a fixed pool. Everything is constrained by it. |
| **Market movement** | If market rates rose 10% and you gave 5%, you fell behind even while "giving a raise" |
| **Retention risk** | A counter-offer situation prices differently |
| **Internal equity** | Raising one person can leave a peer visibly behind |
| **Scope change** | More responsibility without a title change still deserves money |

#### Compa-ratio — learn this term, it's the centre of the whole thing

```
compa-ratio = actual salary ÷ midpoint of predicted band
```

That's it. One division.

| Value | Meaning | Action |
|---|---|---|
| below 0.90 | Paid below band — underpaid | Correct it; genuine flight risk |
| 0.90 – 1.10 | Within band | Normal increment |
| above 1.10 | Paid above band | Smaller increment, or hold |

This one number is what makes your two use cases work. **Pay equity** = list everyone below 0.90.
**Increment** = give more to low compa-ratios, less to high ones, within budget.

It's real HR vocabulary. Using it correctly makes the tool sound like it was built by someone who
understands compensation, not just someone who understands pandas.

---

## 5. Folder layout

Company and public data sit at the same level, as you asked — they're processed by the same
pipeline, so the code doesn't care which is which.

```
paybands/
├── configs/
│   ├── model/                    experiment settings
│   └── tax/
│       ├── fy_2025_26.yaml       one file per financial year, never edited once written
│       └── fy_2026_27.yaml
├── data/
│   ├── public/                   Stack Overflow survey etc.        — committed
│   ├── synthetic/                our generator's output            — regenerated, not committed
│   └── company/                  real company data                 — GITIGNORED, never committed
├── docs/
│   ├── design.md                 this file
│   └── company-data-request.md
├── src/paybands/
│   ├── data/                     loaders — one per source, same output shape
│   ├── features/                 feature engineering
│   ├── model/                    band model + calibration
│   ├── payroll/                  the deduction calculator (Layer 2)
│   ├── policy/                   increment rules (Layer 3)
│   ├── fairness/                 the bias audit
│   └── api/                      FastAPI service
└── tests/
```

Two things about `data/`:

**`company/` is gitignored, hard.** Not "we'll remember not to commit it" — the ignore rule goes in
before the folder exists. Real salary data leaking into a public GitHub repo is the kind of mistake
that ends badly, and the fix is one line written in advance.

**All three loaders output the same shape.** Public, synthetic and company data get converted into
one common table format. That's why they sit at the same level. When company data arrives, you drop
it in and everything downstream — features, model, fairness audit — works unchanged. **You'll have
already tested the whole pipeline before the real data exists.**
