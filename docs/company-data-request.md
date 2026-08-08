# Data Request — Salary Band Model

**Status: DRAFT v0** — do not send yet.

Written early because data approvals take weeks. The list will get **shorter** before you send it:
after Phases 1–5 we'll know which fields actually improved the model, and we cut the rest. Asking
for less data makes approval easier and shows you know what you need.

---

## Part A — For the person approving this

*(Written for HR / Finance, not engineers. Keep it in this register.)*

### What we want to build

A tool that estimates a **fair base salary range** for a role, using our own pay history.

1. **Hiring** — a defensible offer range, instead of relying on memory or on what one similar
   person happened to negotiate.
2. **Pay equity** — identify current employees paid below the range for their role and experience,
   and cost out what correcting it would take.
3. **Increments** — recommend raises that account for both performance and how far someone sits
   below their range.

### What it will not do

- It will **not** set salaries automatically. It gives a range; a human decides within it.
- It will **not** be used to justify paying someone less.
- It will **not** name individuals in anything shared outside HR.

### Why we need salary *history*, not today's snapshot

Two reasons:

- To test the tool honestly, we train it on older records and check it against newer ones. That's
  the only way to know it works on people it has never seen. A snapshot makes this impossible.
- Pay moves with the market. A tool that can't see change can't stay accurate.

### Privacy

- **Names and employee IDs are not needed.** Replace each ID with a random code first. We only need
  to know which rows belong to the same person, never who that person is.
- Demographic fields (Section 3) are used **only** to check the tool for bias, only in aggregate.
  Without them we cannot check for bias — which means we cannot promise the tool is fair.
- Data stays on a company machine. Nothing is uploaded anywhere.

---

## Part B — The fields

**One row = one salary event.** A person appears several times: once at hire, once per increment or
promotion.

### Section 1 — Required

Without these there is no model.

| Field | Example | Why |
|---|---|---|
| `person_code` | `EMP_0481` | Random code, not the real ID. Links one person's rows together. |
| `event_date` | `2025-04-01` | When this salary took effect. **The most important field** — it's what makes honest testing possible. |
| `event_type` | `hire` / `increment` / `promotion` | These behave very differently. |
| `base_salary_annual` | `1800000` | The negotiated annual base. **This is what the model predicts.** |
| `job_title` | `Senior Backend Engineer` | Raw text, exactly as written. Don't tidy it — the messiness is information. |
| `job_family` | `Engineering` | Broad grouping. |
| `level` | `L3` / `Senior` | Internal grade, however you name it. |
| `location` | `Bangalore` | City. We group into tiers ourselves. |
| `employment_type` | `full_time` | |
| `years_experience_total` | `7.5` | Total career experience **at the time of this event**. |
| `years_at_company` | `2.0` | Tenure at the time of this event. |

> ### ⚠️ WHICH salary number? This is the most important line in the request.
>
> A single payslip from this company shows three different figures for the same person in the
> same month:
>
> | | | |
> |---|---|---|
> | **CTC** | ₹51,800 | gross **plus** the employer's PF contribution |
> | **Gross** | ₹50,000 | basic + HRA + special allowance — **this is what we want** |
> | **Net / in-hand** | ₹47,900 | after PF, medical and tax |
>
> They differ by about 8% end to end. If half the export is CTC and half is gross, that error
> runs silently through every prediction the tool ever makes, and nothing in the data announces
> it.
>
> **Please send gross annual salary — basic + all allowances, before any deduction, excluding
> employer PF and gratuity.** If your HRMS can only export CTC, that is fine: say so explicitly
> and send it consistently, and we will convert. What we cannot recover from is a mixture.
>
> **We do not need the deductions.** PF, medical and income tax are calculated by formula from
> gross — the tool computes them itself and has been checked against a real payslip line by line.
>
> One sample payslip has already been checked against the calculator, and it reproduces every
> line exactly — gross, basic, PF, medical, net, CTC, and the payslip's own annual tax
> projection. So the deduction side is settled and needs nothing further from you.
>
> That check also **corrected three things we had wrong**: the basic-to-gross ratio, whether the
> PF ceiling applies, and the insurance premium. All three had been taken from figures quoted
> from memory, and all three were wrong. It is the reason this request asks for exported records
> rather than summarised ones.

### Section 2 — Valuable, include if your HRMS exports it easily

Each is a hypothesis about what drives pay. We'll test which ones actually do.

| Field | Example | Why |
|---|---|---|
| `performance_rating` | `4` | The rating at the time of the event. Central to increment logic. |
| `previous_salary` | `1500000` | Salary immediately before this event. See the warning below. |
| `skills` | `Python;AWS;Kubernetes` | Semicolon-separated. Often a strong signal. |
| `education_level` | `B.Tech` | |
| `institute_name` | `NIT Trichy` | We derive a tier from this; we don't use the raw name. |
| `previous_company` | `Infosys` | For hires. We derive **product vs services** from it — one of the largest pay differences in Indian tech, and ignoring it would make the model badly wrong. |
| `department` | `Platform` | |
| `manager_level` | `2` | Levels below the CEO — a rough proxy for scope. |
| `hire_source` | `referral` | Referral / agency / campus / direct. These price differently. |
| `notice_period_days` | `60` | Immediate joiners often command a premium. |
| `offered_salary` | `1750000` | For hires — what we offered before negotiation. Very useful. |
| `competing_offers` | `2` | The strongest short-term pricing lever there is. Rarely recorded — ask anyway. |
| `increment_budget_pct` | `8` | The company-wide increment pool that year. Explains why a good performer got a small raise. |

> ### ⚠️ A note on `previous_salary`
>
> Anchoring offers to last-drawn salary is normal practice — and it's how past underpayment becomes
> permanent. Someone underpaid early gets "last drawn + 30%" forever, still behind, compounding.
>
> We're requesting it so we can **measure** this, not blindly use it. The plan is to train the model
> both with and without it and compare. If accuracy barely drops without it, we ship the fairer
> version — and that comparison becomes one of the most useful things the project produces.

### Section 3 — Sensitive, for the bias check only

Handle separately, access-controlled, used only in aggregate.

| Field | Example | Note |
|---|---|---|
| `gender` | `F` | |
| `age_band` | `30-34` | **Bands, not date of birth.** |

> If these can't be shared, the model still works — but the fairness report becomes impossible, and
> the documentation will have to say so plainly. That's a real loss: a salary tool with no bias
> check is precisely the kind that causes quiet harm.
>
> Do **not** include caste, religion, marital status, or health data. We won't use them, several
> carry legal restrictions, and asking damages trust in the project.

---

## Part C — How much data

| Amount | What it supports |
|---|---|
| **< 300 rows** | Not enough to train. Still useful to sanity-check a public-data model. |
| **500–1,000** | Minimum workable. Expect wide bands and a shaky fairness check. |
| **2,000–5,000** | Comfortable. The realistic target. |
| **5,000+** | Enough for per-department models. |

**How that adds up:** roughly one hire event plus one increment per person per year. A 300-person
company with 4 years of history yields about **1,200–1,500 rows** — already enough. Smaller company?
Ask for a longer history rather than more people.

**Timespan matters more than row count.** 1,000 rows across 4 years beats 1,000 rows from one year,
because only the first lets us test on unseen time.

---

## Part D — Format

- **CSV or Excel**, one row per salary event
- Missing values left **blank** — never `0` or `N/A`. Blank means "unknown"; zero means "zero
  rupees", and the model will believe you.
- Dates as `YYYY-MM-DD`
- Amounts as plain numbers — no `₹`, no commas, no "18L"
- Annual figures, not monthly (or tell us which, and stay consistent)
- No merged cells, no rows above the header, no totals row at the bottom

A template with exact column names ships in Phase 6, so nobody has to guess.

---

## Part E — Settle these before sending

- [ ] Which Section 2 fields does our HRMS actually export? Don't request anything needing manual entry.
- [ ] Is Section 3 approvable? Ask early — likely needs separate sign-off.
- [ ] How far back does clean salary history go?
- [x] ~~Is PF calculated on the ₹15,000/month statutory ceiling, or on actual basic?~~
      **Answered by the payslip: the ceiling applies.** PF is ₹1,800 = 12% × ₹15,000, where 12%
      of the actual basic would have been ₹2,700. So PF does not rise with salary.
- [x] ~~Does our payroll deduct professional tax?~~ **No** — the payslip's professional tax line
      is blank.
- [ ] Does the PF rule still hold at higher salaries? One current payslip would confirm whether
      the ceiling is applied at every level or only below a threshold.
- [ ] Who owns this data and who signs off on its use?
- [ ] **Does an anonymised version go in a public portfolio, or stay internal forever?**
      Answer this *before* touching company data. It changes how the repo is structured, and
      restructuring afterwards is painful.
