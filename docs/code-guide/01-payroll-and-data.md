# 01 — Payroll and data

This is the first part of the code reading guide, and the right place to start reading the project.

It covers six files, in this order:

| # | File | Why here |
|---|---|---|
| 1 | `src/paybands/payroll/calculator.py` | Pure arithmetic. No machine learning at all. |
| 2 | `src/paybands/data/schema.py` | The one table shape everything else agrees on. |
| 3 | `src/paybands/data/stackoverflow.py` | Turning a real, messy survey into that shape. |
| 4 | `src/paybands/data/synthetic.py` | Inventing data where we know the true answer. |
| 5 | `configs/payroll/fy_2025_26.yaml` | The numbers the calculator runs on. |
| 6 | `tests/test_payroll.py` | What a good test file looks like. |

**Why start here rather than with the model.** The project is built in three layers (see
`docs/design.md`): a learned model that predicts base salary, a calculator that converts base
salary into take-home, and a set of hand-written policy rules for increments. Only the first layer
involves machine learning.

The calculator is the layer with a *known right answer*. Indian tax law is published. A real
payslip exists to check against. You can be completely certain whether the code is correct — which
is not true of anything else in the repository. So it is the one file you can read without
learning any machine learning first, and it teaches you the project's central habit: work out what
you can compute exactly, and only then reach for a model.

Two conventions used throughout, in case they are unfamiliar:

- **Lakh and crore.** Indian numbers are grouped differently. One lakh is 100,000 (written
  ₹1,00,000). One crore is 10,000,000 (₹1,00,00,000). "₹12.2L" means ₹12,20,000.
- **CTC, gross, take-home.** These are three different numbers and confusing them is the most
  common error in Indian salary data. *CTC* (cost to company) is everything the company spends on
  you, including its own provident fund contribution — money that never reaches you. *Gross* is the
  salary before deductions. *Take-home* (or *in-hand*, or *net*) is what actually lands in the bank
  account. On the payslip this project is verified against they are ₹51,800, ₹50,000 and ₹47,900 a
  month: just over an 8% spread between the largest and the smallest. A dataset that mixes them
  carries a silent few-percent error through every model trained on it.
- **Basic.** One named component of gross — here ₹26,000 of the ₹50,000, or 52% — and several
  Indian deductions are calculated on basic rather than on gross.

> **A note on the payslip figures in this guide.** Every rupee amount quoted from "the payslip" is
> **anonymised**: it has the shape of a real payslip — the same component ratios, the same binding
> ceiling, the same nil tax outcome — with the amounts changed. The validation the project actually
> performed used genuine documents, and those documents are not in this repository. The configs,
> the tests and this guide all agree with each other on the published figures, so every teaching
> point below can be checked by running the code.

---

### `src/paybands/payroll/calculator.py`

> Converts an annual gross salary into a complete, explainable Indian payslip — provident fund,
> insurance, professional tax, income tax, cost-to-company and take-home — using rates loaded from
> a config file.

**Read time:** 20 minutes · **Difficulty:** easy
**Read it when:** you can read Python. Nothing else is needed. Start here.

#### What problem it solves

An Indian salary is not one number. You negotiate a gross figure, and then several deductions come
off it before anything reaches your bank: a retirement contribution, an insurance premium, possibly
a state tax, and income tax. Each of those is defined by a published rule — a percentage, a flat
amount, a government slab table.

Because they are rules, they can be computed exactly. This file computes them. There is no
estimation, no fitting, no uncertainty. Feed it ₹6,00,000 a year and it tells you that ₹47,900 a
month lands in the bank — the figure printed on the payslip the project is checked against — and
shows every step of how it got there.

The reason this matters for the project as a whole: a machine learning model that tried to *learn*
these deductions would get them slightly wrong, would break whenever the annual Budget changed the
rates, and — worst of all — would confuse "the government took more tax this year" with "engineers
are worth less this year". Splitting the computed part from the learned part avoids all three
problems.

#### Classes

The file has two groups of classes. The first group describes **the rules** (loaded from YAML). The
second group describes **the result** (produced by the calculation). They are all built with
Pydantic — a library that validates data as it is loaded, so a typo in the config file becomes a
clear error at load time instead of a strange number three files later.

##### `Frozen`

```python
class Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
```

A one-line base class that every other class here inherits from. It sets two policies:

- `frozen=True` — once an object is created, its fields cannot be reassigned. Tax rules should not
  change halfway through a calculation.
- `extra="forbid"` — if the YAML file contains a key the class does not declare, loading *fails*.
  Without this, a misspelled `cess_rte:` in the config would be silently ignored and you would
  quietly compute tax with no cess at all.

##### Rule classes

| Class | Represents | Fields |
|---|---|---|
| `ProvidentFundRules` | How the retirement deduction is worked out | `employee_rate`, `basic_as_fraction_of_gross`, `apply_statutory_ceiling`, `statutory_ceiling_monthly`, `employer_matches` (default `True`) |
| `InsuranceRules` | The insurance premium | `monthly_premium` |
| `ProfessionalTaxRules` | An optional state-level tax | `enabled` (default `False`), `monthly_amount` (default `0`) |
| `TaxSlab` | One row of the income tax table | `upto`, `rate` |
| `Rebate87A` | The rebate that zeroes tax for lower earners | `taxable_income_threshold`, `max_rebate` |
| `IncomeTaxRules` | The whole income tax calculation | `standard_deduction`, `slabs`, `rebate_87a`, `cess_rate` |
| `PayrollRules` | Everything, for one financial year | `financial_year`, `regime`, `provident_fund`, `insurance`, `professional_tax`, `income_tax`, `source`, `verified_on`, `verified_against` |

Field meanings that are not obvious from the name:

| Field | Plain English |
|---|---|
| `ProvidentFundRules.employee_rate` | The fraction of basic pay the employee contributes. 0.12 = 12%. |
| `ProvidentFundRules.basic_as_fraction_of_gross` | What share of gross salary counts as "basic". 0.52 = 52%, measured off the payslip. |
| `ProvidentFundRules.apply_statutory_ceiling` | Whether to compute the contribution on only the first ₹15,000 of monthly basic (the legal minimum) or on the whole of it (more generous). `true` in the current config — the payslip shows this employer applies the ceiling. |
| `ProvidentFundRules.employer_matches` | Whether the employer contributes the same rate again. It is **not** deducted from the employee, so it never touches take-home — but it is part of CTC, which is why CTC exceeds gross by exactly this amount. |
| `TaxSlab.upto` | The top of this income band. `None` means "no upper limit" — the final slab. |
| `TaxSlab.rate` | The rate charged on income *inside this band only*. |
| `IncomeTaxRules.standard_deduction` | A flat amount subtracted from gross before tax is computed. |
| `IncomeTaxRules.cess_rate` | An extra levy charged **on the tax**, not on income. 0.04 = 4% of the tax bill. |
| `PayrollRules.regime` | Which of India's two tax systems this file describes — `"new"` or `"old"`. It is a free-text string; nothing validates it. |
| `PayrollRules.source` / `verified_on` / `verified_against` | Provenance. Never used in a calculation. They exist so that a config nobody wrote notes on cannot pretend to be authoritative. |

`PayrollRules` has one method:

```python
@classmethod
def from_yaml(cls, path: str | Path) -> PayrollRules
```

Opens the file, parses it with `yaml.safe_load`, and validates it into a `PayrollRules`. That is
the only way rules enter the system.

##### `SlabCharge`

One line of the tax working — "on the slice of income from ₹8,00,000 to ₹12,00,000, at 10%, you
paid ₹40,000".

| Field | Meaning |
|---|---|
| `band_from` | Bottom of this band. |
| `band_to` | Top of this band, or `None` for the open-ended top slab. |
| `rate` | Rate applied to this band. |
| `taxable_in_band` | How many rupees of *this person's* income fell inside the band. |
| `tax` | `taxable_in_band × rate`. |

It exists purely so the answer can be explained. A take-home figure that nobody can check is a
figure nobody should trust.

##### `Payslip`

The full result. Annual figures are the fields; monthly figures are computed properties, because
monthly is what people actually recognise from their own payslip.

| Field | Meaning |
|---|---|
| `annual_gross` | The input salary. |
| `annual_basic` | Basic pay for the year. Note: this is the **uncapped** basic, even when the PF ceiling was applied. |
| `annual_pf` | Employee provident fund contribution — the part that is deducted from pay. |
| `annual_employer_pf` | The employer's matching contribution. Never deducted, so it does not affect take-home; it is what makes CTC larger than gross. Zero if `employer_matches` is off. |
| `annual_insurance` | Insurance premium for the year. |
| `annual_professional_tax` | State tax; zero when disabled. |
| `taxable_income` | Gross minus the standard deduction, floored at zero. |
| `income_tax_before_rebate` | Tax from the slab table, before Section 87A. |
| `rebate_applied` | How much of that tax was wiped out by the rebate. |
| `cess` | 4% charged on the tax that survives the rebate. |
| `annual_income_tax` | The final tax actually deducted, cess included. |
| `annual_net` | Take-home for the year. |
| `slab_working` | Tuple of `SlabCharge` rows — the audit trail. |

Properties: `monthly_gross`, `monthly_pf`, `monthly_insurance`, `monthly_income_tax`,
`monthly_net` (each simply the annual figure ÷ 12), `total_annual_deductions`
(`annual_gross − annual_net`), and two that are worth singling out:

- `annual_ctc` — `annual_gross + annual_employer_pf`. Cost to company: always *larger* than gross,
  which is itself larger than take-home.
- `monthly_ctc` — `annual_ctc ÷ 12`.

**Why `annual_ctc` earns its place.** When a salary dataset says "salary", the first job is finding
out which of the three numbers it means. Recruiters quote CTC, payslips show gross, and people
describing their own pay usually mean take-home. They can differ by 10% or more. Having all three
available from one object makes the distinction impossible to forget.

Method: `explain() -> str` — formats a small monthly breakdown as text, for printing during a
payslip check or returning from the API. For ₹6,00,000 a year it produces:

```
Monthly CTC                  51,800   (incl. employer PF)
Monthly gross                50,000
  - Provident Fund            1,800
  - Insurance                   300
  - Income tax                    0
────────────────────────────────────
Monthly take-home            47,900
```

Three numbers, top to bottom, in descending order — which is the whole point of putting the CTC
line first. The professional tax line only appears if the deduction is non-zero.

#### Functions

##### `_income_tax(taxable: float, rules: IncomeTaxRules) -> tuple[float, float, float, list[SlabCharge]]`

Private helper. Returns four things, in this order: `(tax_before_rebate, rebate, cess, working)`.

**Goes in:** a taxable income and the tax rules. **Comes out:** the tax before any rebate, the
rebate amount, the cess, and the list of per-band charges.

The loop is short, and it is the heart of the file:

```python
for slab in rules.slabs:
    upper = slab.upto if slab.upto is not None else float("inf")
    in_band = max(0.0, min(taxable, upper) - lower)
    ...
    lower = upper
    if taxable <= upper:
        break
```

**Why it works this way — marginal slabs.** Each rate applies *only* to the slice of income that
falls inside its own band. It does not apply to your whole salary. `min(taxable, upper) - lower` is
exactly "how much of my income sits between the bottom and the top of this band", and `max(0.0, …)`
turns a negative into zero for bands you never reached.

This is the single most misunderstood thing about Indian income tax. People genuinely turn down
pay rises because they believe crossing into the 30% bracket taxes everything at 30%. It does not.

Worked example — taxable income of ₹29,25,000, which is what a ₹30,00,000 gross produces after the
₹75,000 standard deduction:

| Band | Rate | Income in band | Tax |
|---|---|---|---|
| ₹0 – ₹4,00,000 | 0% | ₹4,00,000 | ₹0 |
| ₹4,00,000 – ₹8,00,000 | 5% | ₹4,00,000 | ₹20,000 |
| ₹8,00,000 – ₹12,00,000 | 10% | ₹4,00,000 | ₹40,000 |
| ₹12,00,000 – ₹16,00,000 | 15% | ₹4,00,000 | ₹60,000 |
| ₹16,00,000 – ₹20,00,000 | 20% | ₹4,00,000 | ₹80,000 |
| ₹20,00,000 – ₹24,00,000 | 25% | ₹4,00,000 | ₹1,00,000 |
| ₹24,00,000 and above | 30% | ₹5,25,000 | ₹1,57,500 |
| | | **Total** | **₹4,57,500** |

₹4,57,500 is 15.6% of the taxable income, not 30%. A flat reading of the table would have said
₹8,77,500 — nearly twice as much.

**Then the rebate.** Section 87A of the Income Tax Act says: if taxable income is at or below a
threshold (₹12,00,000 in this config), the tax bill is cancelled, up to a cap (₹60,000):

```python
if taxable <= rules.rebate_87a.taxable_income_threshold:
    rebate = min(tax, rules.rebate_87a.max_rebate)
tax_after_rebate = max(0.0, tax - rebate)
```

This is why the payslip shows no income tax at all. Its own annual projection is ₹6,00,000 gross,
less the ₹75,000 standard deduction gives ₹5,25,000 taxable, the slab table charges ₹6,250 on the
slice above ₹4,00,000, and the rebate wipes out all ₹6,250. The same happens far higher up: at
₹12,20,000 gross the slab table charges ₹54,500 and the rebate cancels every rupee of it.

**Then the cess.** Cess is an extra 4% levy charged on *the tax*, not on income — and it is
computed after the rebate, so if the rebate reduced the tax to zero, the cess is zero too. That
ordering is not arbitrary; get it backwards and low earners would pay a cess on tax they never
owed.

##### `compute_payslip(annual_gross: float, rules: PayrollRules) -> Payslip`

The public entry point, and the only function most callers ever use.

**Goes in:** an annual gross base salary (the number the model predicts) and a loaded
`PayrollRules`. **Comes out:** a fully populated `Payslip`.

It runs in four steps.

**Step 0 — reject a non-positive salary.**

```python
if annual_gross <= 0:
    raise ValueError(
        f"annual_gross must be positive, got {annual_gross}. Fixed deductions "
        "(insurance) would exceed pay and produce a negative take-home."
    )
```

Read the comment above it in the source; this is one of the most instructive four lines in the
project. Insurance is a *flat* premium, not a percentage — ₹300 a month, ₹3,600 a year. So a gross
of ₹0 produces a take-home of −₹3,600. That is arithmetically correct and completely meaningless.
(The comment beside this code still quotes an older premium from before the payslip arrived. The
argument is unaffected; only the number is stale.)

There were two possible fixes. Clamp the output to zero — which makes the test pass and hides the
fact that nobody ever decided what a valid input is. Or reject the input, and say plainly that a
salary band model has no answer for a zero salary. The project chose the second. The error message
carries the reasoning, so the next person to hit it understands rather than "fixes" it.

**Step 1 — basic pay and provident fund.**

```python
monthly_basic = (annual_gross / 12) * pf_rules.basic_as_fraction_of_gross
if pf_rules.apply_statutory_ceiling:
    monthly_basic_for_pf = min(monthly_basic, pf_rules.statutory_ceiling_monthly)
else:
    monthly_basic_for_pf = monthly_basic
annual_pf = monthly_basic_for_pf * pf_rules.employee_rate * 12
# Same rate again from the employer. Not a deduction — a company cost.
annual_employer_pf = annual_pf if pf_rules.employer_matches else 0.0
```

**Why PF is computed on basic and not on gross.** In an Indian salary structure, "gross" is a sum
of components: basic pay, house rent allowance, special allowance, and so on. The Employees'
Provident Fund contribution is legally defined as 12% of *basic* — not of the whole package. The
verified payslip splits ₹50,000 of gross into basic ₹26,000, house rent allowance ₹13,000 (exactly
half of basic, the usual convention) and special allowance ₹11,000.

Worked example, ₹6,00,000 a year: monthly gross ₹50,000, basic = 50,000 × 0.52 = ₹26,000.
The ceiling is on, so the figure PF is computed from is `min(26,000, 15,000)` = ₹15,000, and
provident fund = 15,000 × 0.12 = **₹1,800** a month. The payslip says ₹1,800.

**That single number is the most informative figure on the payslip.** Had the employer applied PF
to full basic, the deduction would have been 26,000 × 0.12 = ₹3,120. It is ₹1,800 instead, which is
12% of the statutory ₹15,000 and nothing else. One line on a document settled a policy question
that no amount of reasoning about the rest of the payslip could have.

It has a consequence that matters for the project: **capped PF does not grow with salary.** Double
this salary and the deduction is still ₹1,800. Any modelling that assumes deductions scale with pay
is wrong for every employee above ₹15,000 of monthly basic, which is nearly all of them.

Note that `basic_as_fraction_of_gross` is a ratio *read off one document*, not a law. At the
payslip's own gross it reproduces the ₹26,000 basic to the rupee, but it is a rounded policy figure,
so at other salaries the computed basic is only approximately what a real structure would show. That
approximation never reaches the PF figure, because the ceiling truncates the basic long before it
could matter; it shows up only in the reported `annual_basic`, which is why the tests assert basic
with a tolerance — `abs=2` for year 1, `abs=5` for year 2 — rather than demanding equality.

Note also that `annual_basic` in the returned `Payslip` is `monthly_basic * 12` — the *uncapped*
figure. The ceiling is a rule about the provident fund, not about what your basic pay is.

The employer's matching contribution is computed on the same line and stored separately. It is not
subtracted anywhere in the arithmetic below, which is exactly right: it is a company cost, not a
deduction. It exists in the `Payslip` only so `annual_ctc` can be honest.

**Step 2 — taxable income.**

```python
taxable = max(0.0, annual_gross - rules.income_tax.standard_deduction)
```

Under the new regime the standard deduction applies and the provident fund contribution is **not**
deductible. Under the old regime it would be, via Section 80C. That difference is one reason the
two regimes live in separate config files rather than being switched with a flag.

**Step 3 — assemble.** Tax is computed, cess added, and everything subtracted:

```python
annual_net = annual_gross - annual_pf - annual_insurance - annual_ptax - annual_income_tax
```

#### The one thing to understand here

Tax slabs are **marginal**: each rate touches only the slice of income inside its own band, and
never the whole salary. Everything else in the file follows from getting that one loop right.

The close second is that this file deals in three different salary numbers — CTC, gross and
take-home — and keeps them apart on purpose. ₹51,800, ₹50,000 and ₹47,900 all describe the same
month.

#### Surprises and gotchas

- **`annual_employer_pf` is a field on the `Payslip` that is subtracted from nothing.** It appears
  in no deduction line and does not affect `annual_net`. That is correct — it is the company's
  money, not yours — and it is there solely so that CTC can be reported without anybody having to
  reconstruct it.
- **The PF ceiling makes PF insensitive to salary.** With `apply_statutory_ceiling: true`, everyone
  earning more than about ₹28,850 a month gross (₹15,000 of basic at 52%) has an identical ₹1,800 PF
  deduction. A test asserts this explicitly, because it looks like a bug the first time you see it.
- **`Payslip` has no `monthly_basic`, `monthly_employer_pf` or `monthly_professional_tax`
  property**, although it has six other monthly views (`monthly_ctc`, `monthly_gross`,
  `monthly_pf`, `monthly_insurance`, `monthly_income_tax`, `monthly_net`). `explain()` divides
  `annual_professional_tax` by 12 inline, and `tests/test_payroll.py` divides `annual_basic` by 12
  by hand. Harmless, but inconsistent.
- **`annual_income_tax` is recomputed in `compute_payslip`** as
  `max(0.0, tax_before - rebate) + cess`, even though `_income_tax` already computed the same
  intermediate internally. The two agree; it just means the same expression exists in two places.
- **The rebate is a genuine cliff, not a taper.** At ₹12,74,000 gross the tax is exactly ₹0. At
  ₹12,80,000 it is ₹63,180. A ₹6,000 raise costs ₹63,180. That is a real feature of Indian tax law,
  not a bug in this code, and there is a test (`test_rebate_cliff_is_real`) whose whole purpose is
  to stop a future reader "fixing" it.
- **`extra="forbid"` makes the config strict in a way that can surprise you.** Adding a helpful new
  key to the YAML without also adding the field to the model will make loading fail outright.
  That is the intended trade: loud failure beats silent omission.
- **Nothing validates `regime`.** It is a plain string. Loading an old-regime file and running it
  through this new-regime arithmetic would produce confident nonsense. The only defence at present
  is that there is one config file.

---

### `src/paybands/data/schema.py`

> Defines the single table shape that every data source is converted into, plus the two salary
> plausibility thresholds used when cleaning.

**Read time:** 5 minutes · **Difficulty:** easy
**Read it when:** immediately after the calculator, and before either loader.

#### What problem it solves

The project has three sources of salary data: a public developer survey, a synthetic generator, and
(eventually) real company data. On the outside they look nothing alike — different column names,
different categories, different missing fields.

You could teach every downstream piece of code to handle all three shapes. This file takes the
other route: each loader converts its own source into *one* agreed shape, and everything downstream
only ever sees that shape.

The payoff is concrete. When the real company data finally arrives, you write one loader and
nothing else changes — the feature code, the model, the fairness audit and the API all keep
working, because they were already tested against two other sources producing exactly these
columns. You will have tested the entire pipeline before the real data exists.

#### Classes

##### `Source`

```python
class Source(StrEnum):
    STACKOVERFLOW = "stackoverflow"
    SYNTHETIC = "synthetic"
    COMPANY = "company"
```

A `StrEnum` is an enumeration whose members *are* strings, so `Source.SYNTHETIC == "synthetic"` is
`True` and it can be written straight into a pandas column. Every row carries the name of the
loader that produced it, which means a combined dataset can always be split back apart, and a
surprising result can be traced to a source.

#### Module-level constants

There are no functions in this file. It is entirely constants, and that is the point.

| Constant | Value | What it is |
|---|---|---|
| `TARGET` | `"salary_annual"` | The name of the column being predicted. Annual **gross base** salary in rupees — never take-home, never CTC. |
| `CORE_COLUMNS` | 8-tuple | Columns every loader must produce. |
| `OPTIONAL_COLUMNS` | 10-tuple | Columns only some sources have. The model uses them when present. |
| `MIN_PLAUSIBLE_SALARY` | `100_000` | Rows below ₹1 lakh a year are dropped. |
| `MAX_PLAUSIBLE_SALARY` | `20_000_000` | Rows above ₹2 crore a year are dropped. |

`CORE_COLUMNS` is, exactly:

```python
(TARGET, "years_experience", "role", "education", "org_size",
 "remote", "employment_type", "source")
```

`OPTIONAL_COLUMNS` is:

```python
("location_tier", "skills", "level", "institute_tier", "prev_company_type",
 "prev_salary", "performance_rating", "event_date", "gender", "age_band")
```

Two of those carry a comment worth repeating. `prev_company_type` distinguishes product companies
from services companies, which is one of the largest single pay gaps in Indian tech. And `gender`
and `age_band` are marked **fairness audit only** — they exist to *measure* bias, not to be fed to
the model as predictors.

Note the docstring on `CORE_COLUMNS`: missing values are allowed, as `NaN`. A source that genuinely
lacks a field should say "unknown", not invent a value.

#### The one thing to understand here

Three different sources, one table shape. Agreeing on the shape *before* the interesting data
arrives is what lets the rest of the pipeline be written and tested early.

#### Surprises and gotchas

- **The two salary thresholds are judgement calls, not facts.** The file says so, at length. ₹1
  lakh a year is ₹8,300 a month, which is not a plausible full-time developer salary in India —
  those rows are students, unemployed respondents, people who typed a monthly figure into an annual
  box, or blanks entered as 0. Above ₹2 crore is possible but vanishingly rare, and in this dataset
  those rows are mostly joke entries and confusion between salary and total compensation.
- **Why the thresholds live here rather than in the loader.** They are debatable. Putting them in
  one visible place, with the reasoning written beside them, means a reviewer can disagree in one
  line instead of hunting through code for a hardcoded `100000`. Both loaders import them from
  here, so a single edit changes both consistently. This is a small file doing an important
  organisational job.
- **`CORE_COLUMNS` is not enforced anywhere.** Nothing in this file checks that a loader actually
  produced these columns; it is a convention that the loaders honour. `synthetic.generate` uses the
  tuple to reorder its output, which is the closest thing to enforcement in the project.

---

### `src/paybands/data/stackoverflow.py`

> Loads the Stack Overflow Developer Survey 2025, filters it down to Indian respondents reporting
> rupee salaries, converts it to the common schema, and reports every row it dropped and why.

**Read time:** 12 minutes · **Difficulty:** easy
**Read it when:** after `schema.py`. Some familiarity with pandas helps but is not required — the
pandas here is basic filtering and column construction.

#### What problem it solves

Real survey data is messy in specific, predictable ways. Of 49,191 responses worldwide, only 2,547
are from India, and of those only 2,463 report a salary in rupees. Half of *those* left the salary
question blank. A handful typed nonsense — one person entered twenty-two nines, which dragged the
worldwide mean salary to about ₹81 quintillion.

This file does the narrowing and cleaning. More importantly, it counts and prints everything it
removed.

#### Classes

##### `LoadReport`

A **dataclass** — a Python class whose main job is to hold data, where the `@dataclass` decorator
writes the constructor and the comparison methods for you. This one records what the loader kept
and what it threw away.

| Field | Type | Meaning |
|---|---|---|
| `total_responses` | `int` | Rows in the raw CSV, all countries. |
| `india_rows` | `int` | Rows where `Country == "India"`. |
| `inr_rows` | `int` | Of those, rows whose reported currency starts with `INR`. |
| `dropped` | `dict[str, int]` | Reason → number of rows removed for that reason. Insertion-ordered, so printing it walks the cleaning steps in the order they ran. |
| `final_rows` | `int` | Rows surviving into the output. |

Its only method is `__str__`, which formats the report. Running the loader on the real file prints:

```
Stack Overflow 2025 → India
  survey responses            49,191
  from India                   2,547
  reporting salary in INR      2,463
    − no salary reported        1,235  (50.1%)
    − below ₹100,000              200  (8.1%)
    − above ₹20,000,000             6  (0.2%)
  usable rows                  1,022  (41% of INR rows)
```

**Why this pattern matters, and it matters more than it looks.** Every cleaning rule quietly
changes what your dataset represents. Drop the zeros and you have excluded the unemployed. Drop the
top 0.5% and you have excluded the highest earners. Neither is necessarily wrong — but if you do
not count them, you cannot tell whether you cleaned a dataset or replaced it with a different one.

A loader that silently filters is dangerous precisely because it *looks* like it worked. You get a
DataFrame, the code runs, the numbers seem reasonable, and you never learn that 59% of the rows are
gone.

Look at that first line — 1,235 rows, 50.1%. Half the Indian respondents skipped the salary
question. Now ask whether that half is random. It almost certainly is not: people who feel
underpaid, or are between jobs, or find the question intrusive are more likely to skip it. That is
**selection bias**, and no amount of cleaning fixes it, because there is no data to recover. You
can only know about it and say so. The report is what makes it visible.

#### Functions

##### `_map_role(devtype: object) -> str`

Private helper. **Goes in:** the raw `DevType` value for one respondent (which may not be a string
at all — hence the `object` type and the `isinstance` check). **Comes out:** one of fourteen
job-family names, or `"unknown"` for a non-string, or `"Other"` for a string that matches nothing.

It lowercases the text and walks a fixed list of `(keyword, family)` pairs, returning the first
family whose keyword appears anywhere in the string.

**Why collapse at all.** The raw field has dozens of distinct values. A model given hundreds of
rare categories learns noise rather than signal — with two or three examples of a category, any
pattern it finds is chance. Grouping them into a handful of families gives each category enough
rows to mean something.

The docstring is honest that the mapping is "a judgement call and deliberately crude — it's a
starting point to improve, not a truth". On the real data it leaves 210 rows out of 1,022 in
`"Other"`, which is about a fifth.

##### `load(path: str | Path, *, verbose: bool = True) -> tuple[pd.DataFrame, LoadReport]`

The entry point. **Goes in:** a path to the survey CSV. **Comes out:** a `(DataFrame, LoadReport)`
pair. Prints the report unless `verbose=False`.

The sequence:

1. `pd.read_csv(path, usecols=_RAW_COLUMNS, low_memory=False)` — reads only the twelve columns it
   needs. The full file is 172 columns and 134 MB; reading all of it would use over a gigabyte of
   memory for nothing.
2. Filter to `Country == "India"`.
3. Filter to currencies starting with `"INR"`. **Why filter on currency too:** about 70 Indian
   respondents report in US dollars. Those are usually people working remotely for foreign
   employers — a genuinely different market. Mixing them in would inflate the band for local roles.
4. Three cleaning steps, each recording `before - len(inr)` into `report.dropped`: missing salary,
   below `MIN_PLAUSIBLE_SALARY`, above `MAX_PLAUSIBLE_SALARY`.
5. Build the output DataFrame in the common schema.

One line in step 5 deserves attention:

```python
"years_experience": pd.to_numeric(inr["WorkExp"], errors="coerce").fillna(
    pd.to_numeric(inr["YearsCode"], errors="coerce")
),
```

The survey has two experience fields. `WorkExp` is professional experience. `YearsCode` counts from
the first line of code you ever wrote, school included — a much weaker signal for pay. So the code
prefers `WorkExp` and falls back to `YearsCode` only where `WorkExp` is missing.
`errors="coerce"` turns unparseable text into `NaN` rather than raising, and `fillna` with a Series
fills row by row. Six rows end up with no experience figure at all, which the schema permits.

The remaining columns are straightforward renames with `.fillna("unknown")`, plus
`"source": Source.STACKOVERFLOW.value` stamped on every row.

#### The one thing to understand here

A loader must tell you what it removed. Returning only the surviving rows makes cleaning invisible,
and invisible cleaning is how a dataset silently becomes something other than what you think you
are analysing.

#### Surprises and gotchas

- **The output column order does not match `CORE_COLUMNS`.** The dict places `skills` and
  `age_band` before `source`, so `source` ends up last rather than eighth. `synthetic.py`
  explicitly reorders its output; this file does not. Nothing downstream selects by position, so it
  is harmless — but it is an inconsistency between the two loaders.
- **The drop-reason keys use Western digit grouping** (`"below ₹100,000"`), while `docs/JOURNEY.md`
  quotes the same figures in Indian grouping (`₹1,00,000`). Same numbers, different formatting.
- **`_map_role` is first-match-wins on the keyword list, not on the text.** The docstring notes
  that respondents can select several roles. If someone's `DevType` string contains both
  "back-end" and "data scientist", they are classified as Data Science, because that keyword comes
  earlier in the list — regardless of which appeared first in their answer. The comment "Order
  matters: the first match wins" is about the list, and that is worth reading carefully.
- **`LoadReport` is mutable**, unlike everything in the calculator. It is filled in step by step as
  the load progresses, which is the natural way to accumulate the counts.
- **`report.dropped` percentages are shares of `inr_rows`, not of the rows remaining at that step.**
  So the three percentages are directly comparable to each other, but they are not "8.1% of what
  was left", they are "8.1% of the INR rows we started with".

---

### `src/paybands/data/synthetic.py`

> Invents a population of employees and pays them according to rules we write down, including a pay
> gap of a size we choose — so that a fairness audit can be checked against a known answer.

**Read time:** 35 minutes · **Difficulty:** hard
**Read it when:** after both `schema.py` and `stackoverflow.py`. This is the most demanding file in
this part of the guide. Read the module docstring twice before reading the code.

#### What problem it solves

Later in the project there is a **fairness audit** — code that answers "does this model underpay
one group?".

Here is the question that stops you: how do you know the audit works?

On real data, you cannot know. Nobody knows the true size of the real pay gap. So if your audit
reports 3%, you have no way to tell whether the gap is 3%, or whether the gap is 9% and your audit
is broken.

So this file builds a world where we decide the truth. We declare that one group is paid exactly 8%
less, generate people accordingly, and then run the audit on them. If it comes back with roughly
8%, the audit works. If it says 2% or 15%, the audit is wrong — and we find that out *before*
pointing it at real people.

It is calibrating a thermometer in boiling water before trusting it on a patient. That is the whole
reason the file exists.

#### The data-generating process

Every salary is built as a product of multipliers, computed as a sum in log space:

```
log(salary) = log(600,000)                                      base
            + log(experience curve)
            + log(role multiplier)
            + log(location tier multiplier)
            + log(institute premium, faded by experience)
            + log(previous company type multiplier)
            + log(education multiplier)
            + log(org size multiplier)
            + log(remote multiplier)
            + (career gap months / 12) × log(0.93)              the proxy channel
            + is_disadvantaged × log(1 − pay_gap)               THE INJECTED GAP
            + noise ~ Normal(0, 0.22)
```

**Why multiplicative, and why logs.** People talk about pay in percentages: "a 30% raise", never "a
₹2.4 lakh raise". A 30% raise means the same thing to a fresher and to a principal engineer; ₹2.4
lakh means completely different things to the two. So the honest model is a product of factors.

And a product of factors *is* a sum of logs — `log(a × b) = log(a) + log(b)`. Adding the log terms
and exponentiating at the end gives exactly the product, but written as a list of readable lines,
one per reason a salary is what it is.

Two useful consequences fall out of working in logs. Salary can never go negative, because
`exp(anything)` is positive. And normally-distributed noise in log space becomes **lognormal**
noise in rupees: a long right tail, and a spread that scales with the salary. A wobble of ±₹2 lakh
is enormous at ₹6 lakh and trivial at ₹60 lakh; ±20% is neither.

Worked example — a fresher (0 years), Backend, tier-1 city, non-premium college, from a services
company, bachelor's degree, 101–1000 employees, in person, no career break, not in the
disadvantaged group, no noise. Every multiplier is 1.00, so the salary is ₹6,00,000. That is what
`BASE_SALARY` means: the baseline person, not the average person.

Now the same fresher at a product company (×1.35) from a tier-1 institute (×1.20 on day one) in a
tier-2 city (×0.78): 600,000 × 1.35 × 1.20 × 0.78 = ₹7,58,160.

#### Module-level constants

These are "the physics of our invented world". They are written as **multipliers** rather than log
coefficients on purpose: `1.35` reads as "35% more" to anybody, whereas `0.3001` reads as nothing to
anyone. Every one of them is a judgement call informed by the Indian market, not a measured fact,
and they sit in one visible block so they can be argued with.

| Constant | Value | Meaning |
|---|---|---|
| `BASE_SALARY` | `600_000.0` | The baseline fresher's salary. |
| `EXPERIENCE_GROWTH` | `0.85` | The exponent in the experience curve. |
| `EXPERIENCE_KNEE_YEARS` | `3.0` | The scale at which the curve starts to flatten. |
| `INSTITUTE_FADE_YEARS` | `8.0` | How fast the elite-college premium decays. |
| `ROLE_MULTIPLIERS` | 9 roles | Backend is the reference at 1.00; Management 1.30, QA 0.78. |
| `LOCATION_TIER_MULTIPLIERS` | `{1: 1.00, 2: 0.78, 3: 0.64}` | Tier 1 is Bangalore, Mumbai, Delhi-NCR, Hyderabad, Pune, Chennai. |
| `INSTITUTE_TIER_MULTIPLIERS` | `{"tier1": 1.20, "tier2": 1.08, "other": 1.00}` | The **day-one** premium, before fading. |
| `PREV_COMPANY_MULTIPLIERS` | services 1.00, startup 1.12, gcc 1.25, product 1.35 | "GCC" is a global capability centre — the India arm of a foreign firm. |
| `EDUCATION_MULTIPLIERS` | 0.90 to 1.15 | Deliberately weak; education matters less than people expect once experience is in the model. |
| `ORG_SIZE_MULTIPLIERS` | 0.88 to 1.15 | Bigger company, more pay. |
| `REMOTE_MULTIPLIERS` | in-person 1.00, hybrid 1.03, remote 1.06 | |
| `EMPLOYMENT_TYPES` | both 1.00 | Contract work is generated but given **no** pay effect, because modelling it properly needs a day-rate versus annualised distinction the project does not have. Leaving it flat is more honest than inventing one. |

Alongside them sit the population shares — `_ROLE_SHARES`, `_LOCATION_SHARES` and so on — so that
how common each value is, is as visible as what each value pays. Each shares tuple sums to 1.0.

Experience is drawn from a **gamma distribution** with shape 2.2 and scale 3.0, capped at 30 years.
A gamma is a right-skewed distribution — lots of people early in their careers, a thin tail of
veterans — with a mean here of about 6.6 years. Career break lengths use the same family, shape 2.0
and scale 7.0, mean about 14 months, capped at 60.

`_LEVEL_BANDS` maps years onto ladder labels: under 2 is Junior, under 5 Mid, under 9 Senior, under
14 Lead, under 20 Principal, and 20 or more is `_TOP_LEVEL`, `"Director"`. The comment is
important: level is a *label* for experience, not an independent pay driver. Giving it its own
multiplier would pay a person twice for the same years. It is generated only because real company
data will have the column and the pipeline must handle it.

#### Functions

##### `experience_multiplier(years, *, growth=EXPERIENCE_GROWTH, knee=EXPERIENCE_KNEE_YEARS) -> np.ndarray | float`

```python
return (1.0 + np.asarray(years) / knee) ** growth
```

**Goes in:** years of experience, either a single number or a NumPy array. **Comes out:** the pay
multiplier from experience alone.

**Why a power law rather than a straight line.** The real medians from the survey (see Part 4 of
`docs/JOURNEY.md`) climb steeply for the first several years and then flatten hard. Years 0 to 5
matter far more than years 15 to 20. A straight line would overpay veterans, underpay juniors, and
make the finding "experience is non-linear" impossible to demonstrate.

The curve has two required properties: it is **increasing** (more experience never pays less) and
**decelerating** (each extra year is worth less than the one before). Real values with the
defaults:

| Years | Multiplier |
|---|---|
| 0 | 1.00 |
| 3 | 1.80 |
| 5 | 2.30 |
| 10 | 3.48 |
| 20 | 5.65 |
| 30 | 7.68 |

Going from 0 to 5 years multiplies pay by 2.30 — a gain of 130%. Going from 15 to 20 years takes
you from 4.59 to 5.65 — a gain of 23%. Same five years, very different money.

**One correction to the source.** The docstring claims "years 0→5 add ~90%; years 15→20 add ~17%".
The real figures with the current constants are 130% and 23%. The shape of the claim is right — the
early years are worth several times the later ones — but the two percentages in that docstring do
not match the code. Trust the table above.

The other line in the docstring *is* exact: every doubling of `(1 + years/3)` multiplies pay by
`2 ** 0.85 ≈ 1.80`.

##### `institute_multiplier(years, day_one_premium, *, fade_years=INSTITUTE_FADE_YEARS) -> np.ndarray | float`

```python
decay = np.exp(-np.asarray(years) / fade_years)
return 1.0 + (np.asarray(day_one_premium) - 1.0) * decay
```

**Goes in:** years of experience and the day-one premium for that person's institute tier (1.20,
1.08 or 1.00). **Comes out:** the premium as it stands after that many years.

The `- 1.0` and `+ 1.0` are doing the work: they strip the multiplier down to just its *excess over
1*, shrink that excess exponentially, and add the 1 back. So the premium decays towards 1.00 (no
effect) rather than towards 0 (no salary).

For a tier-1 graduate: 1.200 at year 0, 1.074 at year 8, 1.027 at year 16, 1.016 at year 20. After
`fade_years` the excess has shrunk to about 37% of its original size (that is `exp(-1)`), and after
twice as long to about 14%.

This interaction is written out explicitly because it is a finding the later analysis is meant to
*recover*: an elite degree is worth about 20% to a fresher and almost nothing to someone with 20
years of track record. By then, the track record is the evidence.

##### `_pick(rng, values, shares, n) -> np.ndarray`

Private. Draws `n` values from `values` with probabilities `shares`, using
`rng.choice(np.array(values, dtype=object), size=n, p=shares)`. The `dtype=object` keeps mixed
types (the location tiers are integers, everything else strings) working through the same helper.

##### `_multipliers(values: np.ndarray, table: dict) -> np.ndarray`

Private. Looks up each row's multiplier: `np.array([table[v] for v in values], dtype=float)`.

Note what it deliberately does *not* do: `table.get(v, 1.0)`. A missing category would then silently
get a multiplier of 1.0, and a typo would come out the other end as a plausible-looking salary
rather than an error. `KeyError` is the right behaviour here.

##### `_level_for(years: np.ndarray) -> np.ndarray`

Private. Starts everyone at `"Director"` and then walks `_LEVEL_BANDS` from the top down,
overwriting `level[years < cutoff] = name` each time. Because it goes downwards, the lowest
matching band writes last and therefore wins. Someone with 5 years ends as `"Senior"` (not under 5,
so Mid does not apply; under 9, so Senior does).

##### `GroundTruth`

A **frozen dataclass** — a data-holding class whose attributes cannot be reassigned after
construction. It is the answer sheet.

| Field | Meaning |
|---|---|
| `n`, `seed` | How many rows, and the random seed used. |
| `pay_gap` | The injected unfairness. 0.08 means the disadvantaged group is paid 8% less than an otherwise identical person. Multiplicative, so it is the same 8% at ₹6 lakh and at ₹60 lakh. |
| `disadvantaged_group` | Which value of `gender` takes the cut. |
| `share_disadvantaged` | That group's share of the population. |
| `noise_sigma_log` | Spread of the lognormal noise, in log points. 0.22 means a typical person sits about ±22% from what the rules alone would pay. |
| `base_salary`, `experience_growth`, `experience_knee_years`, `institute_fade_years` | Copies of the module constants, so the record is self-contained. |
| `role_multipliers`, `location_tier_multipliers`, `institute_tier_multipliers`, `prev_company_multipliers`, `education_multipliers`, `org_size_multipliers`, `remote_multipliers` | Copies of the multiplier tables. |
| `career_gap_penalty_per_year` | Pay multiplier per year of career break. Default 0.93 — a 7% cut per year out. |
| `career_gap_prob_disadvantaged`, `career_gap_prob_other` | How likely each group is to have had a break. Defaults 0.45 and 0.06. |

Properties:

- `log_pay_gap -> float` returns `float(np.log1p(-self.pay_gap))`. This is a subtle and valuable
  detail. A regression of `log(salary)` on a group indicator recovers `log(1 − pay_gap)`, **not**
  `−pay_gap`. For an 8% gap those are −0.0834 and −0.0800. Close enough to be mistaken for each
  other, far enough apart to fail a tight test. Having the ground truth expose the number in the
  units the audit will actually measure removes an entire class of confusion.
- `has_proxy -> bool` is simply whether the two career-break probabilities differ. Set them equal
  and the proxy channel disappears, leaving the injected gap as the only difference between groups
  — which is exactly what you want for a control experiment.

`__str__` prints a readable summary, including the log coefficient labelled "← what a regression
should recover".

##### `generate(...) -> tuple[pd.DataFrame, GroundTruth]`

The full signature:

```python
def generate(
    n: int = 5000,
    seed: int = 42,
    pay_gap: float = 0.08,
    *,
    disadvantaged_group: str = "female",
    share_disadvantaged: float = 0.28,
    noise_sigma_log: float = 0.22,
    career_gap_penalty_per_year: float = 0.93,
    career_gap_prob_disadvantaged: float = 0.45,
    career_gap_prob_other: float = 0.06,
) -> tuple[pd.DataFrame, GroundTruth]
```

The `*` means everything after it must be passed by keyword, so nobody can accidentally swap two
probabilities by position.

**Goes in:** the size and shape of the world to invent. **Comes out:** a DataFrame in the common
schema, and the `GroundTruth` describing exactly how it was built.

It runs in five stages.

**Validation.** `n` must be positive. `pay_gap` must be in `[0, 1)` — the error message explains
why: it is a fractional cut, so 0.08 means "paid 8% less" and 1.0 would mean "paid nothing".

**The random generator.**

```python
rng = np.random.default_rng(seed)
```

A *local* generator, never `np.random.seed()`. Global random state is shared by every library in
the process, so anything else that draws a number — a scikit-learn split, another module shuffling
something at import time — would silently change your data. A local generator cannot be disturbed
from outside, which is what makes "same seed, same bytes" actually true.

**Who these people are.** Experience from the gamma, then each categorical attribute via `_pick`.
Group membership is `rng.random(n) < share_disadvantaged`, and `gender` is the corresponding label.

**The proxy channel.**

```python
gap_prob = np.where(is_disadvantaged, career_gap_prob_disadvantaged, career_gap_prob_other)
has_gap = rng.random(n) < gap_prob
gap_length = rng.gamma(_CAREER_GAP_SHAPE, _CAREER_GAP_SCALE, n).round()
max_gap = np.minimum(years * 6.0, _MAX_CAREER_GAP_MONTHS)
career_gap_months = np.where(has_gap, np.clip(gap_length, 1.0, max_gap), 0.0)
career_gap_months = np.where(max_gap < 1.0, 0.0, career_gap_months)
```

The cap `years * 6.0` says a break cannot be longer than half a career — someone with two years of
experience has not taken a five-year break. Freshers therefore have none at all, which is what the
final line cleans up.

**What the proxy is for, and why it is the most valuable idea in the project.** `career_gap_months`
is correlated with gender (45% of one group have a break, 6% of the other) *and* it independently
reduces pay (×0.93 per year out). Neither of those facts is unusual or sinister on its own.

The point comes later. Everyone's first instinct on hearing "your model might be biased" is "then
delete the gender column". This dataset is how the project *proves* that is wrong rather than
merely asserting it: delete the gender column, retrain, re-run the audit, and the gap survives —
because the model rebuilds group membership out of career gaps, a column that looks completely
innocent.

A **proxy variable** is exactly that: a feature that stands in for a protected attribute without
naming it. Postcode is the classic real-world example.

The docstring is honest about the simplification: nothing *else* correlates with gender here. Real
life is messier — women are unevenly distributed across roles and cities too — but one clean proxy
channel makes the demonstration readable.

**The salary.** The big `log_salary = ...` expression, one line per term. Two lines deserve
pointing at:

```python
+ (career_gap_months / 12.0) * np.log(career_gap_penalty_per_year)
```

Log-linear in months, so two years out costs `0.93 ** 2` (a 13.5% cut), not `2 × 7%`.

```python
+ is_disadvantaged * np.log1p(-pay_gap)
```

**That single line is the injected unfairness.** Nothing about this person's work differs. It is
the term the fairness audit has to find. `is_disadvantaged` is a boolean array, and multiplying a
boolean by a float gives the float where `True` and zero where `False`. `np.log1p(-x)` computes
`log(1 - x)` with better precision for small `x` than writing it out.

Then:

```python
salary = np.round(np.exp(log_salary), -3)
salary = np.clip(salary, MIN_PLAUSIBLE_SALARY, MAX_PLAUSIBLE_SALARY)
```

`np.round(x, -3)` rounds to the nearest thousand, because real offers come out in round numbers. At
₹1,000 the rounding is under 0.2% of even the smallest salary here — visible in the data, invisible
to any measurement made on it.

The clip is a safety net, not a modelling step, and it reuses the same plausibility bounds that
clean the real survey. At the default settings it binds for essentially nobody — a run of 8,000
rows at seed 7 produced a minimum of ₹2,73,000 and a maximum of ₹1,03,70,000, comfortably inside
the bounds. If it starts binding often, the parameters have drifted somewhere unrealistic and you
want to know.

**Assembly.** The DataFrame is built, then reordered so that the core columns come first in schema
order:

```python
out = out[list(CORE_COLUMNS) + [c for c in out.columns if c not in CORE_COLUMNS]]
```

Finally the `GroundTruth` is constructed and returned alongside.

A quick check of the output, `generate(n=8000, seed=7)`: 28.2% of rows are female (target 28%),
43.9% of them have a career break (target 45%) against 5.9% of the men (target 6%), median salary
₹15,47,000. The mechanism behaves as described.

#### The one thing to understand here

You cannot validate a bias detector on real data, because nobody knows the true answer. So you
build a world where you *are* the answer — inject a gap of known size, and check the audit recovers
it.

#### Surprises and gotchas

- **The `experience_multiplier` docstring's percentages are wrong.** It says years 0→5 add ~90% and
  years 15→20 add ~17%; the actual values are 130% and 23%. Presumably left over from an earlier
  choice of `EXPERIENCE_GROWTH`. The code is fine; the comment is stale.
- **`_pick` couples the shares tuples to dictionary insertion order.** `_pick(rng,
  tuple(ROLE_MULTIPLIERS), _ROLE_SHARES, n)` relies on `tuple(dict)` giving the keys in the order
  they were written, matched one-for-one against `_ROLE_SHARES`. Insert a new role in the middle of
  `ROLE_MULTIPLIERS` without editing `_ROLE_SHARES` in the same place and every subsequent role
  silently gets the wrong population share. Nothing checks this. (Both currently have nine entries
  and the shares sum to 1.0, so it is correct today.)
- **`GroundTruth` is frozen, but its dictionaries are not.** `@dataclass(frozen=True)` prevents
  reassigning an attribute; it does not stop you mutating a dict *inside* one. `truth.role_multipliers["QA"] = 2.0`
  succeeds. The `default_factory=lambda: dict(ROLE_MULTIPLIERS)` at least means each `GroundTruth`
  gets its own copy, so mutating one cannot corrupt the module-level table.
- **Mean salary is not the "rules" salary.** Because the noise is symmetric in *log* space, it is
  asymmetric in rupees: the mean of the generated salaries sits about `exp(0.22² / 2) ≈ 2.5%` above
  what the multipliers alone would produce. The *median* matches the rules exactly. This is a
  general property of lognormal data and it is why the project prefers medians.
- **`GroundTruth` records the module constants, not overrides.** `generate` always calls
  `experience_multiplier` and `institute_multiplier` with their default `growth`, `knee` and
  `fade_years`, so the copies stored in `GroundTruth` are always accurate. But those two functions
  *do* accept overrides, and if a future caller used them the ground truth would no longer describe
  the data.
- **`synthetic` is not re-exported from `paybands/data/__init__.py`.** That file imports `schema`
  and `stackoverflow` only. `from paybands.data import synthetic` still works — Python imports the
  submodule — but plain `import paybands.data` followed by `paybands.data.synthetic` does not.
- **`pay_gap = 0.0` is the control case**, and it is meant to be used. With zero injected gap the
  data has no direct gender effect at all, and an audit that still reports a non-zero gap is
  telling you about the proxy, not about direct discrimination.

---

### `configs/payroll/fy_2025_26.yaml`

> The tax and deduction rates for Indian financial year 2025-26, in one file, with its own
> verification status written into it.

**Read time:** 8 minutes · **Difficulty:** easy
**Read it when:** after `calculator.py`. Read them side by side if you can — every key here maps to
a field there.

#### What problem it solves

Rates change. The Union Budget adjusts tax slabs most years; insurance premiums and company PF
policy change too. If those numbers live in Python, every change is a code change, a code review
and a release. If they live in a config file, a change is one line.

Equally important: the file records *how much it should be trusted*, which is unusual and worth
copying.

#### Structure

The top-level keys map exactly onto `PayrollRules`:

```yaml
financial_year: "2025-26"
regime: new
source: "https://incometax.gov.in — verify before use"
verified_on: "2026-08-08"
verified_against: "payslip PDF, year 1: gross 50,000 = 26,000 basic + 13,000 HRA + 11,000 special;
                   PF 1,800; medical 300; net 47,900; CTC 51,800"

provident_fund:
  employee_rate: 0.12
  basic_as_fraction_of_gross: 0.52
  apply_statutory_ceiling: true
  statutory_ceiling_monthly: 15000
  employer_matches: true

insurance:
  monthly_premium: 300

professional_tax:
  enabled: false
  monthly_amount: 200

income_tax:
  standard_deduction: 75000
  slabs: [...]
  rebate_87a:
    taxable_income_threshold: 1200000
    max_rebate: 60000
  cess_rate: 0.04
```

The slab table, in full:

| Up to | Rate |
|---|---|
| ₹4,00,000 | 0% |
| ₹8,00,000 | 5% |
| ₹12,00,000 | 10% |
| ₹16,00,000 | 15% |
| ₹20,00,000 | 20% |
| ₹24,00,000 | 25% |
| `null` (no limit) | 30% |

`null` in YAML becomes `None` in Python, which `TaxSlab.upto` accepts and `_income_tax` turns into
`float("inf")`.

#### Where the numbers came from

Four of them are read straight off a real payslip PDF, and the working is written into the
comments. This is worth reading closely, because it is a nice demonstration of how much a single
document can tell you.

The payslip's salary structure, for year 1:

| Line | Monthly |
|---|---|
| Basic | ₹26,000 |
| House rent allowance | ₹13,000 |
| Special allowance | ₹11,000 |
| **Gross** | **₹50,000** |
| − Provident fund | ₹1,800 |
| − Medical | ₹300 |
| **Net** | **₹47,900** |
| Employer provident fund (separate line, not deducted) | ₹1,800 |
| **CTC** | **₹51,800** |

- Basic is stated outright, so no inference is needed: 26,000 ÷ 50,000 = 52% of gross. Hence
  `basic_as_fraction_of_gross: 0.52`.
- **The ceiling question, settled by one number.** 12% of the actual ₹26,000 basic would be ₹3,120.
  12% of the statutory ₹15,000 ceiling is ₹1,800. The payslip says ₹1,800. So this employer applies
  the ceiling. Hence `apply_statutory_ceiling: true`. That was a question the project was about to
  email HR about; one line on a document answered it.
- The medical deduction is ₹300, and the year-to-date figure of ₹1,200 over four months confirms it
  is a flat monthly premium rather than a one-off. Hence `insurance.monthly_premium: 300`.
- Employer provident fund appears on its own line at ₹1,800 and is *not* subtracted from pay. It is
  the whole of the difference between CTC ₹51,800 and gross ₹50,000. Hence
  `provident_fund.employer_matches: true`.
- 50,000 − 1,800 − 300 = 47,900, exactly the net figure. Nothing is left over, so there is no
  professional tax and no income tax. Hence `professional_tax.enabled: false`.
- The payslip's own TDS block projects the year: ₹6,00,000 gross, ₹75,000 standard deduction,
  ₹5,25,000 taxable. Reproducing that is a second, independent check — of the tax path rather than
  the deduction path.

**What this file used to say, and why it matters.** An earlier version of this config had three of
these values wrong: basic at 33.33%, the ceiling off, and the premium at ₹250. They had been
reverse-engineered from figures reported verbally from memory — "PF is about ₹3,600, insurance
about ₹250" — before any payslip existed. The ₹3,600 was almost certainly the employee's ₹1,800 and
the employer's ₹1,800 read together off a payslip that lists them on separate lines. Every
downstream number was then derived from that single misreading, and each derivation looked sound.
Numbers read off a document beat numbers recalled from one.

#### The one thing to understand here

One file per financial year, **never edited once written**. `fy_2026_27.yaml` sits beside this file
rather than replacing anything in it. That is what lets you recompute what someone's take-home *was*
in 2025-26 — a question HR will absolutely ask when an employee notices their in-hand changed.
Editing history destroys the ability to answer it, and there is a test
(`test_last_years_payslip_still_reconciles_against_last_years_config`) whose only job is to fail if
somebody tries.

#### Surprises and gotchas

- **The file distinguishes between what is verified and what is not, and this is the best thing
  about it.** PF, insurance and the salary structure are verified against a real payslip. The income
  tax slabs are marked **still unverified**. The comment explains why the payslip does *not*
  validate them: the Section 87A rebate zeroes the tax at this salary regardless of what the slabs
  say, so a matching net figure tells you nothing about whether the slab table is right. Being
  explicit about which parts of a passing test actually got tested is a genuinely mature habit.
- **The comments record what the file used to say and why that was wrong.** The current PF block
  does not just assert `apply_statutory_ceiling: true`; it states the ₹1,800-versus-₹3,120
  arithmetic that proves it, and notes what the previous setting had been inferred from. A config
  that explains a reversal is a config the next reader will not quietly reverse back.
- **The file is deliberately out of date, and says so.** These are FY 2025-26 figures; the
  `verified_on` date falls in FY 2026-27. That is not a bug to fix here — the fix was to check
  `incometax.gov.in` and add `fy_2026_27.yaml` alongside, which is what happened. This file stays
  frozen at what was true in its own year.
- **`basic_as_fraction_of_gross: 0.52` is a ratio measured from one document, not a rule.** It
  reproduces the ₹26,000 basic exactly at the payslip's own gross, but it is a company policy that
  happens to land near half, not a legal proportion, so it is only approximate at any other salary.
  It cannot affect the PF figure either way, because the ₹15,000 ceiling truncates the basic long
  before the ratio matters; `tests/test_payroll.py` still asserts basic with a tolerance (`abs=2`)
  rather than exact equality, which is the honest way to state a measured constant.
- **`professional_tax.monthly_amount: 200` is set even though `enabled: false`.** The calculator
  ignores the amount entirely when disabled, so this is a sensible default sitting ready rather
  than a bug.
- **The warning at the top of the file is not decoration.** Nothing in this repository checks these
  slabs against the government's published table. They must be verified by a human before anyone
  relies on the output for a real decision.

---

### `tests/test_payroll.py`

> Twenty-four tests for the payroll calculator — twenty-six cases, because one is parametrised
> three times — four of which check it against actual payslip PDFs from two different years.

**Read time:** 15 minutes · **Difficulty:** easy
**Read it when:** last in this part. You will get the most from it if `calculator.py` and the YAML
config are still fresh.

#### What problem it solves

The calculator produces numbers. Numbers look convincing whether or not they are right. These tests
are what make the difference.

The file is worth reading as a *model of how to write tests*, not just as verification. Every test
has a docstring saying what it checks and, more usefully, *why anyone should care*.

#### Read the module docstring first — it is the most valuable thing in this part of the guide

The docstring at the top of the file records how the suite got here, and it is worth quoting the
substance of it:

> The first version of this file was built on figures reported from memory ("PF is about 3,600,
> insurance about 200"). The config was then tuned until the tests passed. Everything was green,
> and three separate things were wrong: the basic fraction, the PF ceiling rule, and the insurance
> premium.

Sit with that for a moment, because it is the most useful lesson in this document.

The tests passed. They passed for months. They passed because the config had been bent until they
did — and both the config and the tests came from the same mistaken belief about what the payslip
said. **A test and a config derived from the same source agree with each other perfectly and tell
you nothing.** The green tick was measuring the consistency of one belief with itself.

No amount of extra testing would have caught it. More assertions against the same wrong figures
would only have made the suite more confidently wrong. Nor would code review: the arithmetic was
correct throughout, and each inference followed sensibly from the one before. The chain was sound
and its first link was broken.

Only a primary document could settle it. When the payslip PDF arrived, three config values changed
in one sitting.

The habit worth taking from this: when a test encodes a fact about the outside world, ask where
that fact came from. If the answer is "from the same place the code came from", the test is
checking your arithmetic, not your beliefs. Getting the actual document is usually a five-minute
job, and it is worth more than a week of additional tests.

#### The fixture

```python
@pytest.fixture
def rules() -> PayrollRules:
    return PayrollRules.from_yaml(RULES_PATH)
```

A pytest **fixture** is a named piece of setup. Any test that declares a parameter called `rules`
gets a freshly loaded `PayrollRules`. It means the config is loaded from the real file, not mocked
— so a broken config breaks the tests, which is exactly what you want.

There is also a plain function, `rules_fixture()`, which loads the same file. It exists because the
four year-1 ground-truth tests take no arguments — a pytest fixture can only be injected into a test
that declares it as a parameter, and those tests are written without one.

A second fixture, `rules_2026_27`, loads `configs/payroll/fy_2026_27.yaml`. It appears in the final
block of the file, and one test declares *both* fixtures so that the two financial years can be
compared side by side inside a single test.

Note that `RULES_PATH` is a relative path, so the suite must be run from the repository root.

#### The tests

##### `test_against_real_payslip_year_1` — the one that matters

```python
slip = compute_payslip(50_000 * 12, rules_fixture())

assert slip.monthly_gross == pytest.approx(50_000)
assert slip.annual_basic / 12 == pytest.approx(26_000, abs=2)
assert slip.monthly_pf == pytest.approx(1_800, abs=1)
assert slip.monthly_insurance == pytest.approx(300)
assert slip.annual_income_tax == pytest.approx(0)
assert slip.monthly_net == pytest.approx(47_900, abs=2)
assert slip.monthly_ctc == pytest.approx(51_800, abs=2)
```

**This is the most valuable test in the file, and the reason is worth stating carefully.**

Every other test in this file checks that the code is *internally consistent* — that its own logic
holds together, that doubling an input doubles an output, that a boundary behaves as the same code
says it should. Those tests are useful. They catch regressions. But they can all pass while the
calculator is confidently wrong, because they only ever compare the code against itself. The suite's
own history proves it: they did exactly that, for three separate config values at once.

This test compares the code against **a document**. ₹47,900 is not a number this project computed
and then wrote down. It is what payroll printed. If the calculator agrees with it, the calculator
models something real.

The distinction, stated plainly:

> A calculator that agrees with itself is worthless. A calculator that agrees with payroll is
> trustworthy.

Note that the last assertion checks `monthly_ctc`, not just gross and net. All three of the numbers
people confuse are pinned in one test.

`pytest.approx` compares floating-point numbers with a tolerance instead of demanding exact
equality. `abs=2` on the basic figure allows for the fact that the basic fraction is a measured
ratio rather than an exact rule.

This test replaced an earlier one built on verbally reported numbers. That test also passed.

##### `test_payslip_own_tax_projection_reproduces`

```python
slip = compute_payslip(6_00_000, rules_fixture())
assert slip.taxable_income == pytest.approx(5_25_000)
assert slip.annual_income_tax == pytest.approx(0)
```

A **second, independent** check against the same document. The payslip's TDS block does its own
annual projection — ₹6,00,000 gross, ₹75,000 standard deduction, ₹5,25,000 taxable — and this test
reproduces it.

Why bother, when the test above already matches the payslip? Because the two exercise different
code paths. The first checks the deduction path (basic, PF, insurance, net). This one checks the
tax path. A single ground-truth test can be satisfied by an implementation that happens to be wrong
in two compensating ways; two ground-truth tests hitting different code have to be right for
different reasons.

##### `test_pf_is_capped_by_the_ceiling_not_by_basic`

```python
at_current = compute_payslip(50_000 * 12, rules_fixture()).monthly_pf
at_double = compute_payslip(1_00_000 * 12, rules_fixture()).monthly_pf
assert at_current == pytest.approx(1_800, abs=1)
assert at_double == pytest.approx(1_800, abs=1), "capped PF must not grow with salary"
```

This pins the single most informative number on the payslip. 12% of the actual ₹26,000 basic would
be ₹3,120; the payslip says ₹1,800, which is 12% of the ₹15,000 statutory ceiling exactly. One
figure decided a policy question that the rest of the document could not.

The second assertion states the consequence: **capped PF does not grow with salary.** Double the
pay and the deduction is unchanged. It looks like a bug and is not, so it is asserted rather than
left to be rediscovered.

##### `test_ctc_exceeds_gross_by_employer_pf`

```python
assert slip.annual_ctc > slip.annual_gross > slip.annual_net
assert slip.annual_ctc - slip.annual_gross == pytest.approx(slip.annual_employer_pf)
assert slip.annual_employer_pf == pytest.approx(slip.annual_pf)
```

The three-numbers problem, written as a test. The first line asserts the ordering — CTC above
gross above take-home, always, in that order. The second asserts that the gap between the top two
is exactly the employer's contribution and nothing else. The third asserts the employer matches the
employee rate.

This is the test to point at when somebody asks what the difference between CTC and salary actually
is.

##### `test_slabs_are_marginal_not_flat`

```python
just_below = compute_payslip(16_70_000, rules).annual_income_tax
just_above = compute_payslip(16_80_000, rules).annual_income_tax

assert just_above > just_below, "more income must not mean less tax"
assert just_above - just_below < 5_000, ...
```

Crossing a bracket boundary must not tax the whole salary at the higher rate. If the code were flat
rather than marginal, a small raise across a boundary would cost tens of thousands.

**Picking the two numbers is the whole test, and the earlier version of it got them wrong.** Slabs
apply to *taxable* income, which is gross minus the ₹75,000 standard deduction — so to straddle the
₹16,00,000 taxable boundary you need a gross either side of ₹16,75,000, not either side of
₹16,00,000.

The original compared ₹15,99,000 and ₹16,01,000 gross. Those give taxable incomes of ₹15,24,000 and
₹15,26,000 — both comfortably inside the same 15% band. No boundary was crossed, so a flat-tax
implementation would have passed too. The test was green and proved nothing.

The current constants do straddle it: ₹16,70,000 gross gives ₹15,95,000 taxable, ₹16,80,000 gives
₹16,05,000. The real tax figures are ₹1,24,020 and ₹1,25,840 — a difference of ₹1,820, which is the
₹10,000 that crossed taxed at 15% and 20% respectively, plus 4% cess. A flat implementation would
tax the whole ₹16,05,000 at 20% instead of 15% and jump by about ₹81,750. The ₹5,000 threshold sits
between the two outcomes by a factor of sixteen in either direction, so the test cannot be satisfied
by accident.

This is the kind of thing worth checking yourself rather than trusting a passing green tick — and
it is the second illustration in this file of a test that was green while proving nothing.

##### `test_high_earner_tax_working`

The strongest of the logic tests, because the docstring contains the whole calculation done by hand,
slab by slab, and the assertions check each stage of it:

```python
assert slip.taxable_income == pytest.approx(29_25_000)
assert slip.income_tax_before_rebate == pytest.approx(4_57_500)
assert slip.rebate_applied == 0
assert slip.cess == pytest.approx(18_300)
assert slip.annual_income_tax == pytest.approx(4_75_800)
```

Checking intermediate values, not just the final one, means a failure tells you *where* the logic
broke rather than only that it did.

##### `test_rebate_wipes_out_tax_below_threshold`

At ₹11,00,000 gross, tax is computed (`income_tax_before_rebate > 0`), then rebated away
(`rebate_applied > 0`), leaving both the tax and the cess at zero. The cess assertion is the
interesting one: it verifies the ordering. Cess is charged on tax, so no tax means no cess.

##### `test_rebate_cliff_is_real`

Confirms that ₹12,74,000 pays nothing and ₹12,80,000 pays something. As the docstring says, this is
a genuine cliff in Indian tax law, not a bug — and the test exists so that nobody "fixes" it later.
Writing a test to protect surprising-but-correct behaviour is a habit worth stealing. (The real
figures: ₹0 versus ₹63,180.)

##### `test_statutory_ceiling_caps_pf`

Checks that PF is ₹1,800 a month on a ₹50 lakh salary. It needs no setup at all, because the
ceiling is on in the real config — the payslip showed this employer applies it.

**This test used to be the awkward one and is now the plain one**, which is a small but telling
consequence of the config correction. It previously had to build a modified copy of the rules to
turn the ceiling *on*, because the config said it was off. This test and the one below have
swapped roles: the setup moved from here to there when the payslip arrived.

##### `test_pf_scales_with_salary_when_uncapped`

```python
uncapped = rules.model_copy(
    update={
        "provident_fund": rules.provident_fund.model_copy(
            update={"apply_statutory_ceiling": False}
        )
    }
)
```

Now *this* is the test that needs a modified copy, to turn the ceiling off. `model_copy` is the
Pydantic way to "change" a frozen object — it makes a new one, it does not mutate the original.
Because the structure is nested, it does this twice: once for `provident_fund`, once for `rules`.

Then doubling the salary doubles the PF. A simple property test — it checks a relationship rather
than a specific number.

It is a good example of testing a code path the real config does not exercise. This employer applies
the ceiling, but more generous employers apply PF to full basic, the code supports both, and the
branch that is not currently in use is exactly the one that will silently rot if untested.

##### `test_deductions_never_exceed_gross`

Loops over four realistic salaries and asserts `0 < annual_net < annual_gross`. **This is the test
that failed** and produced the zero-salary guard, back when the loop also included `0`.

##### `test_non_positive_salary_rejected`

```python
@pytest.mark.parametrize("bad", [0, -1, -10_00_000])
def test_non_positive_salary_rejected(rules, bad):
    with pytest.raises(ValueError, match="must be positive"):
        compute_payslip(bad, rules)
```

`@pytest.mark.parametrize` runs the same test body once per value, so this is really three tests
reported separately. `pytest.raises` asserts that an exception *is* thrown, and `match` checks the
message — so a `ValueError` raised for some unrelated reason would not satisfy it.

The docstring records the history: an earlier version of the suite passed `0` in with the realistic
salaries and failed, and the right fix was to reject the input rather than clamp the output. That
paragraph is arguably more valuable than the assertion, because it stops the next reader from
undoing the decision.

##### `test_explain_is_readable`

```python
text = compute_payslip(50_000 * 12, rules).explain()
assert "Monthly take-home" in text
assert "47,900" in text
assert "51,800" in text
```

A small test of the presentation layer, tying the human-readable output back to the same payslip.
The third assertion is the interesting one: the CTC figure must appear, so that a reader of the
output cannot mistake gross for cost-to-company. A formatting change that quietly dropped the CTC
line would fail here.

##### The surcharge block — six tests

Salaries above ₹50 lakh of taxable income attract a *surcharge*: an extra percentage charged on the
tax, not on the income. Six tests cover it, and they are worth skimming even though nobody on the
payslip goes anywhere near the threshold.

| Test | What it pins |
|---|---|
| `test_no_surcharge_below_fifty_lakh` | At exactly ₹50,00,000 taxable, the surcharge is zero. |
| `test_surcharge_rates_match_the_official_table` | 10% above ₹50L and 15% above ₹1Cr, checked well clear of the thresholds so that marginal relief is not interfering. |
| `test_marginal_relief_kills_the_surcharge_cliff` | Crossing the threshold must cost roughly the extra income, not a lakh of extra tax. |
| `test_marginal_relief_can_be_switched_off` | Turning the config flag off must *reproduce* the cliff — otherwise the flag is decorative and you would not know the relief was doing the work. |
| `test_cess_is_charged_on_tax_plus_surcharge` | Cess is 4% of tax **plus** surcharge, never of income. |
| `test_ordinary_salaries_are_untouched_by_surcharge` | A regression guard: the whole surcharge path stays dormant across the salary range the model actually predicts. |

The fourth of those is the one to copy. It is easy to write a test that passes because the feature
under test is irrelevant; asserting that *disabling* the feature changes the answer is what proves
the feature is what produced it.

##### The two-year block — five tests

The last section of the file loads a second config, `fy_2026_27.yaml`, and runs the same kind of
checks against a second payslip.

`test_against_real_payslip_year_2` is the year-2 twin of the first test: gross ₹1,00,000 a month
splitting into basic ₹51,000, HRA ₹25,500 and special ₹23,500, less PF ₹1,800 and medical ₹350,
giving net ₹97,850 and CTC ₹1,01,800. Its docstring makes a point the year-1 test does not:

> NOTE THE INPUT: gross, not CTC. CTC is ₹1,01,800/month here, and feeding that in where gross
> belongs would overstate take-home by ₹1,800/month.

That is the CTC-versus-gross lesson stated as an *input* error rather than a reporting one, and it
is the form the mistake actually takes in code.

`test_current_payslip_own_tax_projection_reproduces` does for year 2 what the TDS test did for year
1 — reproduces the payslip's own annual projection, and confirms the tax comes out nil, so the tax
path is independently checked in both years.

`test_pf_deducted_is_half_of_what_lands_in_the_pf_account` is the most instructive of the five.
₹1,800 leaves the salary; ₹3,600 arrives in the provident fund account, because the employer matches
it. Both numbers are real and they answer different questions — ₹1,800 determines take-home,
₹3,600 is what the retirement balance grows by — and neither of them is "the PF". Reading the pair
as a single ₹3,600 deduction is exactly the mistake this project made early on, and it briefly drove
the statutory ceiling in the config to ₹30,000 to accommodate it.

`test_the_same_salary_gives_different_pf_in_different_years` is the argument for one config file per
financial year, written as an assertion. Feed the *same* gross salary through both configs and you
get two different, both-correct answers: PF is identical, because the ₹15,000 ceiling binds in both
years, but the medical premium rose from ₹300 to ₹350, so take-home differs. Had the 2025-26 file
simply been edited when the new rates arrived, there would be no honest answer to "what was my
take-home last year?".

`test_last_years_payslip_still_reconciles_against_last_years_config` is the guard that makes the
previous test's promise stick. It re-runs the year-1 reconciliation, so if anyone ever edits
`fy_2025_26.yaml` to match a newer rule, the suite fails and says why. A test whose only job is to
protect a file from being edited is an unusual thing to write, and here it is the right one.

#### The one thing to understand here

There are two kinds of test, and they are not equally valuable. Most tests check that code agrees
with itself — useful for catching regressions, but they can all pass while the whole thing is
wrong. This file is the proof: it was entirely green while three config values were wrong, because
the tests and the config had been derived from the same mistaken belief.

Four tests here check the code against **documents that the project did not write** — a full
reconciliation and a tax projection for each of two years. That is what turns "the code runs" into
"the code is correct".

#### Surprises and gotchas

- **The suite's own history is written into the module docstring**, and it should be read before
  any of the tests. Being green is not the same as being right, and this file is a worked example
  of the difference.
- **`50_000 * 12` rather than `6_00_000`.** The ground-truth tests write the annual figure as the
  monthly one times twelve, because the payslip is a monthly document and the multiplication is the
  only step between it and the function's input. Writing the product would hide where it came from.
- **`50_00_000` is Python's numeric underscore separator used in the Indian grouping.** Python
  ignores underscores in numeric literals entirely, so `50_00_000` is just 5,000,000. It reads
  naturally to an Indian reader and is initially baffling to anyone else.
- **The module docstring names a test that no longer exists.** It calls out
  `test_against_real_payslip_july_2025`; the function is now `test_against_real_payslip_year_1`. The
  substance of the docstring is unaffected, but the name is stale — a small reminder that docstrings
  are not checked by anything.
- **All twenty-four tests pass** — twenty-six cases once the parametrised one is expanded — and they
  run in well under a second, because none of them touch a network, a database or a large file.
- **The suite depends on the real config files.** That is a deliberate choice — it means a mistake
  in either YAML is caught by the tests — but it does mean these are not pure unit tests, and they
  must be run from the repository root for the relative `RULES_PATH` and `CURRENT_RULES_PATH` to
  resolve.
- **There is no test for `total_annual_deductions`, `monthly_gross` on odd inputs, or a professional
  tax that is switched on.** The professional tax path in `compute_payslip` and the corresponding
  branch in `explain()` are the only lines in the calculator that no test exercises.

---

## Where to go next

You now understand the layer with a known right answer, the shape all data is converted into, both
of the loaders that produce it, and what a well-argued test file looks like.

The later parts of this guide pick up where the material stops being arithmetic and starts being
machine learning.
