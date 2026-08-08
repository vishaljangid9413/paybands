# The Journey — how this project was built, from nothing to finished

**Read this if you want to understand not just *what* this project is, but *how* it came to
exist** — every decision, every mistake, every thing that broke and how it got fixed.

It is written for someone who knows how to program but has never done machine learning. No prior
ML knowledge is assumed. Every term is explained the first time it appears.

It is also written to be **self-sufficient**. If you read this document top to bottom, you should
be able to pick up the project and keep going without asking anyone anything.

---

## Contents

- [Part 0 — What this project is](#part-0--what-this-project-is)
- [Part 1 — Choosing what to build (and abandoning the first attempt)](#part-1--choosing-what-to-build)
- [Part 2 — One question that decided the whole architecture](#part-2--one-question-that-decided-the-whole-architecture)
- [Part 3 — Building the easy layer first](#part-3--building-the-easy-layer-first)
- [Part 4 — Getting real data, and what it looked like](#part-4--getting-real-data)
- [Part 5 — Building something deliberately stupid](#part-5--building-something-deliberately-stupid)
- [Part 6 — Inventing data on purpose](#part-6--inventing-data-on-purpose)
- [Part 7 — Features, and the most expensive mistake in ML](#part-7--features-and-leakage)
- [Part 8 — The actual model, and honest uncertainty](#part-8--the-actual-model)
- [Part 9 — The fairness audit](#part-9--the-fairness-audit)
- [Part 10 — Charts that are evidence, not decoration](#part-10--charts-that-are-evidence)
- [Part 11 — The layer with no machine learning in it](#part-11--the-layer-with-no-ml-in-it)
- [Part 12 — The API](#part-12--the-api)
- [Part 13 — Explaining a prediction](#part-13--explaining-a-prediction)
- [Part 14 — Measuring everything properly](#part-14--measuring-everything-properly)
- [Part 15 — Every mistake, collected in one place](#part-15--every-mistake)
- [Part 16 — Every file, explained](#part-16--every-file-explained)
- [Part 17 — How to continue from here](#part-17--how-to-continue)
- [Part 18 — Glossary](#part-18--glossary)

---

# Part 0 — What this project is

## The problem in one sentence

**A company needs to decide what to pay someone, and currently guesses.**

When a candidate is hired, somebody picks a number. That number usually comes from memory
("what did we pay the last backend person?"), from the candidate's previous salary, or from
whoever negotiates harder. None of those are good reasons.

The same problem shows up again every year at appraisal time.

## What we built

A tool that answers: **"What should this person be paid?"**

But notice the trap in that question. If a tool answers `₹21,34,500`, it sounds precise and
scientific — and it is fake precision. Nobody can predict a salary to the rupee. Two equally
good engineers at the same company can legitimately be paid 20% apart, for reasons no dataset
records.

So this tool answers differently:

> **"₹18,00,000 – ₹24,00,000, and here is how confident I am."**

That is called a **band**. It is honest about what it does not know, and it is the shape a
recruiter can actually use in a negotiation.

## The two things a company does with it

**1. Making an offer.** A candidate applies. You enter their experience, role, location. The tool
returns a band. You make an offer inside it. No more "what did we pay the last guy?".

**2. Finding underpaid employees.** Run every current employee through the tool. Anyone whose
actual salary sits *below* their band may be underpaid. HR gets a list and a number: *"correcting
all 14 of these costs ₹32 lakh a year."*

That second one is what makes companies actually want it — it turns a vague worry into a budget
line.

## The honest headline

**The tool works, and it is not yet usable for real offers.**

Both of those are true, and understanding why is the most valuable thing in this project. We will
get there in Part 8. The short version: the public data we could obtain is too weak to support a
narrow band, and the tool says so rather than pretending otherwise.

---

# Part 1 — Choosing what to build

## We built the wrong project first

This is the honest starting point.

The first project chosen was an **LLM evaluation harness** — infrastructure for measuring whether
an AI question-answering system was getting better or worse. It was scoped, planned in detail, and
built through its first phase: project scaffolding, configuration system, a results database, a
caching layer, a command-line interface, continuous integration, and 26 tests.

Then it was deleted.

**Why:** the goal was to learn **model building** — fitting models to data, tuning them, measuring
them. The evaluation harness was mostly plumbing around someone else's model. Good engineering,
wrong subject.

**The lesson:** it is much cheaper to abandon a project after one week than after six. The one
week of work was not wasted — several patterns from it (caching expensive results, tracking
experiments, treating configuration as typed objects) reappeared in this project.

**What it cost:** one day. **What it would have cost if noticed in week six:** six weeks.

## How the replacement was chosen

Three candidate projects were considered, all requiring real model building:

| Idea | What it predicts | Why it was interesting |
|---|---|---|
| **Salary bands** | What a person should be paid | Own idea; real deployment target at the owner's employer |
| Attrition timing | *When* an employee will leave | Uses survival analysis, a rare and deep technique |
| Receivables risk | Which invoices get paid late | Highest immediate business value; cash flow |

**Salary bands won** for three reasons that are worth copying when you choose your own project:

1. **The owner had domain access.** He works at a company with this exact problem and could
   eventually get real data.
2. **Public data existed**, so work could start immediately and never be blocked waiting for
   permission.
3. **It had a hard problem inside it** that most versions of the project ignore — see below.

## The hard problem hiding inside "predict a salary"

A model learns patterns from history. That is all it does.

So if a company has historically paid one group less than another for the same work, **the model
learns that as a pattern** and applies it to every future candidate.

The output then looks objective. It is a number, from a computer, with mathematics behind it. But
it is just the old unfairness, laundered — and now much harder to challenge, because "the model
said so."

In hiring, that is not only wrong, it is a legal risk in many countries.

**This turned a routine regression project into an interesting one.** Instead of just predicting
salaries, the project had to:

- predict a **range**, not a number, so it cannot pretend to precision it does not have
- **measure** whether that range is honest
- **audit** itself for exactly the bias described above
- prove the audit works, on data where the right answer is known

Those four things are the spine of everything that follows.

---

# Part 2 — One question that decided the whole architecture

Before any code, one question had to be answered, and it turned out to shape the entire project.

## The salary structure

The owner's company pays salaries like this:

```
  Base salary                    ← the number you negotiate
− Provident Fund (PF) deduction  ← retirement savings, mandatory in India
− Medical / insurance deduction
− Income tax (TDS)               ← per Indian government rules
──────────────────────────────
= Net take-home                  ← what actually lands in the bank
```

**The question: which of these does the model predict?**

## The answer, and the rule behind it

**The model predicts base salary. Nothing else.** Everything below the first line is *calculated*,
not learned.

The rule to remember:

> ### Never make a model learn something you can compute.

PF is a percentage. Insurance is a fixed amount. Income tax is a published government formula.
These are **arithmetic** — exactly known, written down by law.

If you make a model *learn* arithmetic, three bad things happen:

**1. It gets it slightly wrong.** A model approximates; a formula is exact. You would be adding
error to the one part of the system that has no uncertainty in it at all.

**2. It breaks every February.** India's Union Budget changes tax rates most years. A model would
need retraining and re-validating. A formula in a configuration file needs one line edited.

**3. It confuses two completely different things.** This is the subtle one, and it is worth
reading twice.

Suppose next year's Budget raises taxes, so everyone's take-home pay drops. A model trained on
take-home sees the drop and concludes: *"engineers are worth less now."*

They are not. The government just took more. The model has mixed up **market value** with **tax
policy**, and every prediction it makes afterwards is poisoned.

## The three layers

That single decision produced the architecture:

```
┌──────────────────────────────────────────────────────────────┐
│  LAYER 1 — MODEL          (learned from data)                │
│  candidate details  →  base salary band                      │
│  This is the only machine learning in the project.           │
│  Changes when: the job market moves.                         │
└────────────────────────────┬─────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────┐
│  LAYER 2 — CALCULATOR     (arithmetic, from a config file)    │
│  base salary  →  PF, insurance, tax  →  take-home            │
│  Exact. Testable against a real payslip.                     │
│  Changes when: the Union Budget changes.                     │
└────────────────────────────┬─────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────┐
│  LAYER 3 — POLICY         (rules, written by humans)          │
│  band + performance + budget  →  recommended increment       │
│  Changes when: HR decides differently.                       │
└──────────────────────────────────────────────────────────────┘
```

**Layer 3 deserves a note.** It is tempting to build a *second model* for increments. Do not.
How much budget exists, how a performance rating converts to money, whether to fix underpayment
this year or over three — those are **decisions the company makes**, not patterns hiding in data.

Written as rules, HR can read them, disagree with them, and change them. Buried inside a model,
nobody can.

> **Knowing when *not* to use machine learning is part of being good at machine learning.**

---

# Part 3 — Building the easy layer first

## Why start in the middle

The obvious place to start is the model. We started with **Layer 2**, the payroll calculator.

**Why:** it is the part with a *known right answer*. Tax law is published. A payslip exists. You
can be 100% certain whether your code is correct.

Everything else in the project involves uncertainty. Starting with the certain part means that
when things later go wrong, you know it is not the calculator.

> **Nail down what you can be certain about, then attack what you cannot.**

## A note on the figures in this part

Every rupee amount below that comes off a payslip has been changed. The **shape** is real — the
same components, the same ratio between them, the same statutory ceiling binding in the same
place, the same arithmetic reconciling to the rupee — but the amounts are invented, because this
repository is public and a payslip belongs to a person.

The validation described here was done against genuine documents. Those documents are not in this
repository and will not be. Every teaching point survives the substitution; nothing else does.

## The numbers we started from — reported from memory

The owner supplied his own figures. Not from a document; from recollection:

```
Annual salary   ₹12,20,000
PF              about ₹3,600 per month
Insurance       about ₹250 per month
Everything else is take-home
```

Four numbers. Watch how much they *appear* to reveal — and then watch what happens when an actual
payslip turns up.

### Step 1 — reverse-engineering the structure

Monthly gross is `₹12,20,000 ÷ 12 ≈ ₹1,01,700`.

PF in India is **12% of *basic* salary** — and "basic" is a component of your salary, not the whole
of it. So the reasoning ran:

```
basic = ₹3,600 ÷ 0.12 = ₹30,000 per month
₹30,000 ÷ ₹1,01,700   ≈ 30% of gross
```

A very standard-looking Indian structure. Companies often do keep basic near a third, because PF
and gratuity are calculated on it. It looked entirely reasonable.

### Step 2 — a policy question, answered without asking

There is a legal ceiling: employers only *have* to pay PF on the first ₹15,000/month of basic.
12% of ₹15,000 is **₹1,800**.

The reported deduction was **₹3,600** — twice that. So, the reasoning continued, this company must
apply PF to the **full basic**, the more generous option.

**That was a question we had been about to email HR.** One number appeared to have answered it.

At the time this was written up as a small triumph. It was not. Both steps were wrong, and the
next section is why. Read them again first, because the reasoning is fine — it is the *input* that
was rotten, and that is the harder failure to spot.

### Step 3 — the missing deduction

The owner listed PF and insurance but **no income tax** — and that turned out to be correct, not
an oversight.

India's "new tax regime" gives salaried people a standard deduction, and a rebate (Section 87A)
that wipes out tax entirely below a threshold.

Here it is on the figures from the current payslip's own tax block — annual gross ₹12,00,000, not
the ₹12,20,000 that was recollected, because that figure was the CTC (₹12,21,600 a year), rounded
in the telling:

```
Gross                    ₹12,00,000
− standard deduction        ₹75,000
= taxable income         ₹11,25,000

  ₹0 – 4L        at  0%  →         ₹0
  ₹4L – 8L       at  5%  →    ₹20,000
  ₹8L – 11.25L   at 10%  →    ₹32,500
                             ─────────
  tax                          ₹52,500
− 87A rebate                   ₹52,500   ← wipes it out completely
= income tax                        ₹0
```

The payslip's Total Tax line is blank, which agrees.

### The most misunderstood thing about income tax

Look at that middle block again.

**This person is not "in the 10% bracket."** They pay 0% on the first ₹4 lakh, 5% on the next ₹4
lakh, and 10% only on the last ₹3.25 lakh.

That is what **marginal** means: each rate applies only to the slice of income inside its own band.

Almost everyone gets this wrong. People genuinely refuse pay rises because they believe crossing a
bracket taxes their *whole* salary at the higher rate. It does not.

### The slabs were verified against the government, and that found a gap

Everything above was written from tax figures that had been *marked unverified in the config* —
deliberately, because guessing at tax numbers and presenting them confidently is worse than
admitting they need checking.

They were later checked against the Income Tax Department's official page for AY 2026-27. **Every
slab, the rebate threshold, the rebate cap and the cess rate matched exactly.**

But the official page listed something the config did not have at all: **surcharge**.

Surcharge is an extra percentage charged **on the tax**, not on income, once taxable income passes
₹50 lakh — 10% above ₹50L, 15% above ₹1 crore. Since this project treats salaries up to ₹2 crore
as plausible, two surcharge bands sit inside the range it predicts. Without them the calculator
would understate tax for high earners.

And surcharge has a nasty shape: it is a **cliff**. At ₹50,00,001 of taxable income a 10%
surcharge lands on the *whole* tax bill, so one extra rupee of income could cost over a lakh in
tax. The law prevents that with **marginal relief** — tax plus surcharge may not exceed the tax at
the threshold plus the income earned above it.

Measured, with relief working:

```
  taxable ₹50,00,000   surcharge      ₹0   total tax ₹11,23,200
  taxable ₹50,05,000   surcharge  ₹3,500   total tax ₹11,28,400
```

₹5,000 more income costs ₹5,200 more tax. Without marginal relief the surcharge would have been
₹1,08,150 and that step would have cost over a lakh.

**The lesson: a verification against an authoritative source is not only about checking what you
have. It is about noticing what you are missing.** The slabs were all correct. The absence was the
finding.

There is a test in the project (`test_slabs_are_marginal_not_flat`) that specifically checks a
₹2,000 raise across a bracket boundary does not cause a tax jump — so nobody can ever break this by
accident.

### Step 4 — the result we believed

```
Monthly gross       ₹1,01,700
  − Provident Fund      ₹3,600
  − Insurance             ₹250
  − Income tax               ₹0
────────────────────────────────
Monthly take-home    ₹97,850
```

The config was set to match: basic at 30% of gross, the statutory ceiling switched **off**,
insurance at ₹250. The tests were written against that config. Everything passed.

That is where this section used to end, and it ended with the phrase *"validated against a real
payslip"*. It was not. It had been validated against somebody's memory of one.

---

## The payslip that actually arrived

Months later the owner produced the PDF: the earlier payslip, before his increment. Here is the
structure it shows, with every identifying detail left out — a payslip carries a name, an employee
number, a PAN, a bank account, a PF number and more, and none of that belongs in a repository.

```
EARNINGS                          DEDUCTIONS
  Basic          ₹26,000            Provident Fund   ₹1,800
  HRA            ₹13,000            Medical            ₹300
  Special        ₹11,000          ─────────────────────────
─────────────────────            Total deductions   ₹2,100
  Gross          ₹50,000

  Net pay        ₹47,900           ( 50,000 − 2,100 )

  PFEmpr          ₹1,800   ← employer's own contribution, shown separately
  CTC            ₹51,800   ← 50,000 gross + 1,800 employer PF
```

That is a smaller salary than the ₹1,01,700 discussed above, because it predates the increment. The
*structure* is what matters, and the structure is what the project had guessed at.

**Three of the numbers the calculator had been built on were wrong.**

| What the project believed | What the payslip says |
|---|---|
| basic is **30%** of gross | basic is **52%** of gross (₹26,000 ÷ ₹50,000) |
| the employer applies PF to the **full basic** | the employer applies the **statutory ceiling** |
| insurance is **₹250**, called insurance | it is **₹300**, and the line is called **Medical** |

The ceiling one is worth doing the arithmetic on, because it is unambiguous:

```
12% × ₹15,000 (the ceiling)      = ₹1,800   ← the payslip's PF line, exactly
12% × ₹26,000 (the actual basic) = ₹3,120   ← what full-basic PF would have been
```

There is no rounding to hide behind. The company caps PF at the statutory floor, which is the
opposite of what Step 2 concluded.

### Where the ₹3,600 came from

Almost certainly this: **₹1,800 employee + ₹1,800 employer**. The payslip prints them on two
separate lines — the employee's deduction among the deductions, the employer's under a heading
called `PFEmpr` — and reading them as a single ₹3,600 is a very easy thing to do from memory.

Every later conclusion followed from that one misreading. Halve the input and Step 1 gives
`₹1,800 ÷ 0.12 = ₹15,000` — which is the ceiling itself, and would have pointed straight at the
right answer.

> **Clever reasoning from a wrong premise is worse than no reasoning at all.** No reasoning leaves
> you with an open question. The reverse-engineering above produced three confident numbers, a
> config tuned to match them, and a written claim of real-world validation. It manufactured
> certainty out of a guess.

### The part that matters most: the tests all passed

This is the reason this mistake is in the document at all.

Thirteen payroll tests were green throughout. They had been written against the config, and the
config had been tuned until the tests agreed with the reverse-engineered story. So:

```
a belief  →  a config that encodes the belief
          →  a test that asserts what the config computes
          →  green
```

**A test and a config derived from the same mistaken belief agree with each other perfectly and
tell you nothing.** They are not two witnesses; they are one witness repeated. No amount of
additional testing in that loop could ever have caught this, because every new test would have been
written from the same wrong premise.

The only thing that could break the loop was **a primary document** — something generated outside
the project, by a party that had no idea the project existed.

> **Green tests prove your code does what you told it to. They cannot tell you whether what you
> told it is true.** For anything the outside world decides — tax rates, company policy, a salary
> structure — you need a document, not a test.

### CTC, gross and take-home — three different numbers

The payslip makes a distinction concrete that people confuse constantly, including in job offers:

```
CTC          ₹51,800   what the employee costs the company, incl. its own PF contribution
Gross        ₹50,000   what is credited to the employee before deductions
Take-home    ₹47,900   what actually reaches the bank
```

The employer's ₹1,800 is real money, but the employee never sees it as pay — it goes into their PF
account. It is part of **CTC** and it is *not* a deduction from gross. Treating it as one is
exactly the error that produced the ₹3,600.

This is why an offer of "₹51,800 CTC" and a bank credit of ₹47,900 are both honest descriptions of
the same job, and why candidates who compare a CTC against a take-home reach nonsense conclusions.

The calculator now models all three. `Payslip` gained `annual_employer_pf` and the properties
`annual_ctc` and `monthly_ctc`, `ProvidentFundRules` gained `employer_matches`, and `explain()`
prints CTC on the line above gross so the two can never be read as the same thing.

### What the corrected calculator says

Against the earlier payslip, to the rupee:

```
Monthly gross          ₹50,000
  − Provident Fund      ₹1,800     (12% of the ₹15,000 ceiling, not of basic)
  − Medical               ₹300
  − Income tax               ₹0
────────────────────────────────
Monthly take-home     ₹47,900     ← matches the payslip exactly
```

The payslip also carries its own tax projection, and the calculator reproduces that independently:
annual gross ₹6,00,000, minus the ₹75,000 standard deduction, gives taxable income ₹5,25,000 and a
tax of nil. That is a **second** check, on different logic, from the same document. The payslip is
also marked *"New Tax Regime Opted"*, confirming the regime the config assumes, and it carries no
professional-tax line, confirming that setting too.

### What is still not settled

At his current ₹1,01,700/month the owner reported PF of ₹3,600 — and with the ceiling applied, PF
stays at ₹1,800 no matter how high the salary goes. So one of two things is true:

- **(a)** the ₹3,600 was again employee ₹1,800 plus employer ₹1,800 read together, or
- **(b)** the company moved to full-basic PF at some point after the increment.

There is no way to choose between them from here. **A current payslip would settle it in one
glance**, and until one exists this stays written down as an open question rather than quietly
assumed away.

### How it was settled — by a second document, not by an argument

The current payslip arrived, and it disagreed with **both** readings.

```
BASIC 51,000 + HRA 25,500 + SPECIAL 23,500 = gross  1,00,000
− PF 1,800 − Medical 350                   = net      97,850
CTC 1,01,800 = gross 1,00,000 + employer PF 1,800
```

Three things fall out, and two of them had been got wrong.

**PF deducted is ₹1,800, not ₹3,600.** The slip prints `PF 1,800` under Deductions and
`PFEmpr 1,800` on a separate line below. **₹3,600 is what lands in the PF account; ₹1,800 is what
leaves the salary.** Both numbers are real and they answer different questions — ₹1,800 sets
take-home, ₹3,600 grows the retirement balance. The ₹15,000 ceiling is still in force and still
binding: 12% of the actual ₹51,000 basic would have been ₹6,120.

**The ₹12.2 lakh salary is the CTC, not the gross.** Gross is ₹12,00,000/year — ₹1,00,000/month —
and the payslip's own tax block confirms it. Feeding the CTC in where gross belongs overstates
take-home by ₹1,800 a month, which is the employer's PF counted as if it were pay.

**The medical premium rose** from ₹300 to ₹350 between the two years.

Real take-home is **₹97,850**.

### The mistake worth studying here

The first payslip supported an inference: ₹3,600 was probably employee ₹1,800 plus employer
₹1,800, because the slip printed them as two lines. That inference was correct.

It was then contradicted from memory — "₹3,600 is deducted, and it is up to the company how they
show it" — and rather than asking for another document, the model was **rebuilt to accommodate the
assertion**: the ceiling was moved to ₹30,000, since 12% × ₹30,000 is exactly ₹3,600. The
reasoning was tidy, the arithmetic was right, and a config file was written with a confident
explanation of why a ceiling change was more likely than two simultaneous changes.

All of it was wrong, and elaborately so.

> **When a document-based inference is contradicted from memory, the answer is another document —
> not a cleverer model that accommodates the memory.**

The tell was there: the explanation required inventing a ceiling change nobody had reported, to
explain a number that already had a simpler explanation sitting on the original payslip. Effort
spent making a theory fit is usually effort that should have been spent checking the theory.

### The one-file-per-year rule still earned its keep

The two years genuinely differ — the medical premium changed, and the basic fraction moved from
52% to 51% — so `fy_2025_26.yaml` and `fy_2026_27.yaml` both exist and both reconcile
against their own payslip. A test asserts the earlier slip still balances against its own year's
config, so if anyone edits history the suite says so.

That part of the design held up. It was only the contents of the new file that had to be rewritten.

## The tax-free threshold, stated exactly

The owner's understanding was that income up to "12 LPA" is tax-free. That is nearly right, and the
gap between nearly and exactly is ₹75,000.

The Section 87A rebate is written against **taxable** income, not gross. Taxable income is gross
minus the ₹75,000 standard deduction. So:

```
rebate applies to taxable income up to   ₹12,00,000
standard deduction                     +   ₹75,000
──────────────────────────────────────────────────
tax-free GROSS salary                    ₹12,75,000
```

At ₹12,75,000 gross the tax is **nil**. At ₹12,80,000 gross it is **₹63,180**.

Five thousand rupees more salary, sixty-three thousand rupees of tax. That is not a bug in the
calculator — it is a genuine cliff in the law, because the rebate is all-or-nothing rather than
tapered. There is a test (`test_rebate_cliff_is_real`) that pins it, precisely so nobody later
"fixes" the calculator into smoothing it out.

> **Round numbers people quote — "12 lakh is tax-free", "PF is 12%" — are usually approximations
> of a rule with an exact boundary.** Find the boundary. It is where the money is.

## Design decisions in the calculator

**Every number lives in a config file, never in code.** `configs/payroll/fy_2025_26.yaml` holds
the tax slabs, PF rates, insurance premium. Rates change yearly; code should not have to.

**One config file per financial year, never edited once written.** Next year you *add*
`fy_2026_27.yaml`. You do not edit the old one. That way you can always recompute what someone's
take-home *was* in 2025 — a question HR will absolutely ask.

**A deliberate gap left open.** The tax slabs in the config are from FY 2025-26 and are marked
**unverified**. Verifying them requires checking `incometax.gov.in`, which the reader should do.
Guessing at tax numbers and presenting them confidently would be worse than admitting they need
checking.

**The config carries its own provenance.** Every value that came off the payslip now sits next to a
comment saying which line of which document it came from — and, next to it, what the file used to
say and why that was wrong. A config file that records *where its numbers came from* is the only
kind you can audit later.

## The first failure

A test was written saying *"take-home should never be negative."* It failed — on a salary of ₹0.

Why? **The medical premium is a flat monthly amount, not a percentage** — ₹300/month in the
corrected config. So zero salary minus a year of it = a negative take-home. The arithmetic was
correct. The situation was meaningless.

There were two ways to fix it:

- **Clamp the output** — force it to zero. This *hides* the problem.
- **Reject the input** — state plainly that a salary model has no answer for a zero salary.

**We chose the second.**

> **When a test fails, the bug is not always in the formula.** Sometimes the test is telling you
> that you never decided what a valid input is. Clamping would have made the test pass while
> leaving the confusion in place.

The code now raises an error explaining exactly this, and the test file records *why* — so in six
months nobody "fixes" it back.

**Result: 15 payroll tests, all passing.** Four of them are checked line by line against the payslip
PDF: `test_against_real_payslip_year_1` (the full earnings-to-net reconciliation),
`test_payslip_own_tax_projection_reproduces` (the document's own annual tax block),
`test_pf_is_capped_by_the_ceiling_not_by_basic`, and `test_ctc_exceeds_gross_by_employer_pf`.

That first test used to be named after the person whose payslip it was. It was renamed, partly to
keep a personal name out of the test suite and partly because a test asserting facts from a document
should say **which** document — the year 1 structure may not be next year's, and it was not.

---

# Part 4 — Getting real data

## The source

The **Stack Overflow Developer Survey 2025** — a public survey of developers worldwide, released
as a CSV file. Free, real, and containing salary, experience, role, education, and country.

It is 134 megabytes and 172 columns.

`scripts/fetch_data.py` downloads it. Two details in that script are worth copying:

**It downloads to a temporary filename first**, then renames. A half-downloaded file wearing the
correct name is worse than no file — the loader would read it, fail strangely, and you would go
hunting for the bug in your parsing code.

**It pins the 2025 file** rather than a "latest" link. If the data changed underneath us, every
number in our results would silently stop being reproducible.

## Narrowing to India

```
49,191   total survey responses
 2,547   from India
 2,463   reporting salary in Indian rupees
```

Why filter on currency too? About 70 Indian respondents report in US dollars. Those are usually
people working remotely for foreign employers — a genuinely different market. Mixing them in would
inflate the band for local roles.

## Then we looked at the raw salaries

```
mean salary   = ₹81,433,224,755,703,136,256
median salary = ₹11,00,000
```

Read that mean again. **Eighty-one quintillion rupees.**

**One person typed 22 nines into the salary box.** That single row dragged the mean up by an
amount larger than the world economy. The median did not move at all.

This is the clearest possible demonstration of why **median beats mean on real data**:

- The **mean** asks every value to vote, and lets the loudest one win.
- The **median** asks "who is in the middle?" and ignores the shouting.

> When you run `df.describe()` and the mean and median are far apart, something is either genuinely
> skewed or genuinely broken. Find out which before doing anything else.

## Cleaning, and counting what we removed

```
reporting salary in INR      2,463
  − no salary reported        1,235  (50.1%)
  − below ₹1,00,000             200   (8.1%)
  − above ₹2,00,00,000            6   (0.2%)
usable rows                  1,022
```

The cleaning rules (keep ₹1 lakh to ₹2 crore) are **judgement calls, not facts**, so they live in
one visible place — `src/paybands/data/schema.py` — as named constants with their reasoning written
beside them. Anyone reviewing the project can disagree in one line instead of hunting through code.

The loader **reports every row it drops, with a reason**. Silent filtering is how a dataset quietly
becomes a different dataset than you think you have.

## The dangerous line is the first one

The junk entries are easy — you spot them and remove them.

**Half the Indian respondents skipped the salary question entirely.**

Now ask: *is that half random?*

Almost certainly not. People who feel underpaid, or are between jobs, or find the question
intrusive are more likely to skip it. So the 1,022 who answered are probably **not
representative** of the 2,463 who were asked.

That is called **selection bias**, and you cannot fix it by cleaning. No amount of careful
filtering recovers data from people who never typed anything. You can only *know about it* and say
so.

> **The rows that are not there are often more important than the rows that are.** Junk data is
> loud and easy. Missing data is silent and shapes your conclusions anyway.

## What the clean data says

```
MEDIAN SALARY BY EXPERIENCE (India, n=976)

  0–1 yr      ₹5,40,000   (n=105)
  2–3 yr      ₹7,00,000   (n=221)     +30%
  4–5 yr     ₹12,00,000   (n=172)     +71%
  6–8 yr     ₹22,00,000   (n=162)     +83%
  9–12 yr    ₹30,00,000   (n=183)     +36%
  13–20 yr   ₹40,00,000   (n=136)     +33%
  20+ yr     ₹40,00,000   (n=36)        +0%
```

Two things to notice.

**The curve is steepest between years 4 and 8.** Salaries nearly quadruple from year 3 to year 8,
then the increases slow sharply. This is why **experience must not enter the model as a straight
line** — one more year is worth wildly different amounts at year 3 versus year 15.

**The flat 20+ bracket is probably noise, not truth.** Only 36 people. With samples that small the
median bounces around. Being able to say *"that flat line is a sample-size artefact, not a
finding"* is exactly the instinct this project is built around.

## The skew, measured

```
skew of raw salary   =  2.87    ← badly lopsided, long tail to the right
skew of log(salary)  = -0.23    ← nearly symmetric
```

**Skew** measures lopsidedness. Zero means symmetric. 2.87 is very lopsided.

Taking the **logarithm** of salary makes the distribution nearly symmetric. This matters because
salary differences are **multiplicative**: people say *"a 30% raise"*, never *"a ₹2.4 lakh raise"*.
Working in logs matches how salaries actually behave.

So the model trains on `log(salary)` and converts back to rupees afterwards.

## The three biases in one dataset

Worth listing plainly, because every real dataset has problems like these:

1. **Joke entries** — someone typed 22 nines
2. **Non-response** — half skipped the salary question (selection bias, uncleanable)
3. **Sampling** — Stack Overflow respondents skew senior, English-speaking, and
   globally-oriented, so salaries here are probably **higher** than the true Indian market

> The difference between a junior and a senior analyst is not that the senior finds clean data.
> It is that the senior knows what is wrong with theirs and says so out loud.

---

# Part 5 — Building something deliberately stupid

## Why build a bad model on purpose

Before building anything clever, we built two deliberately simple models:

**Baseline 0 — the global median.** Predict the same number for everybody: the median salary. It
ignores every input. It is terrible. It is the floor.

**Baseline 1 — a lookup table.** For each combination of (role, experience bucket), compute the
median salary of people in the training data who match. To predict, look up the matching cell.

Baseline 1 is **four lines of pandas**. No machine learning at all.

## Why this matters more than it sounds

Baseline 1 is usually surprisingly good. And it is the number your sophisticated model has to beat.

If gradient boosting only ties the lookup table, **the model is not earning its complexity**, and
the correct engineering decision is to ship the lookup table — it is faster, simpler, and anyone
can debug it.

Most portfolio projects skip this and report *"R² = 0.87!"* with nothing to compare against, which
means nothing at all. **Beating a real baseline is a claim. A bare accuracy score is not.**

## A detail that matters: the fallback hierarchy

What if a candidate's (role, experience) combination never appeared in training? Or appeared
twice?

The lookup table falls back in order:

```
(role, experience bucket)   →   experience bucket only   →   global median
```

And it requires a **minimum group size of 10** before trusting a cell. A median computed from two
people is noise, not a signal.

In practice: 75.5% of predictions matched a full (role, bucket) cell, 24.5% fell back to
bucket-only, and 0% needed the global median.

## Results

```
                    global median      role × experience     gain
  MAE               ₹15,42,363    →    ₹11,95,730          22.5%
  Median error      ₹10,68,000    →    ₹ 6,76,498          36.7%
```

**MAE** means "mean absolute error" — on average, how many rupees was the prediction off by.

## The first big lesson about measuring

That 22.5% came from **one** train/test split. Re-running on four different random splits gave:

```
  seed 1        +17.5%
  seed 555       +8.2%
  seed 31337    +22.1%
  seed 8        +20.4%
```

The original 22.5% sat at the **top** of the range. Not wrong. Lucky.

> ### One train/test split is an anecdote.

The final measured figure, across 10 splits, is **19.4% (range 13.1%–26.0%)**.

This single lesson recurs throughout the project, and later in this document you will see the
author of this project make exactly this mistake — twice.

## Two findings worth keeping

**The global baseline has a *negative* R².** R² is a standard score where 1.0 is perfect and 0.0
means "no better than predicting the average". Negative means *worse* than predicting the average.

Not a bug. R² measures against the **mean**; the baseline predicts the **median**. On right-skewed
salary data those are far apart, so predicting the median scores badly on a mean-based metric —
even though the median is the more sensible prediction.

> **A metric can say "worse" when the answer is better.** Know what your metric rewards.

**MAPE is useless here.** MAPE (mean absolute percentage error) came out at 150–170%. Percentage
error explodes when the true value is small: predicting ₹15 lakh for someone earning ₹1 lakh is a
1,400% error. MAPE was dropped as a headline metric, and the reason was written down rather than
quietly forgotten.

## Metrics are always in rupees

The model trains on `log(salary)`. But every number reported is converted back to rupees, because:

- *"off by ₹7,73,600"* is a number a human can judge
- *"MSE 0.083 in log space"* is not

There is one subtlety documented in the code: `exp(mean of logs)` is the **geometric mean**, not
the arithmetic mean. They differ, and pretending otherwise introduces quiet bias.

## The honest reading

```
  median salary          ₹15,00,000
  typical error          ₹ 7,73,600   ← 52% of the median
```

**The best baseline is off by half a salary.** That is genuinely bad — and it is supposed to be.
It is the floor. Now we know exactly how much any real model improves on it.

---

# Part 6 — Inventing data on purpose

## The problem

We wanted to build a fairness audit — code that measures whether a model underpays one group.

But here is the question that stops you: **how do you know your audit works?**

On real data, you cannot. Nobody knows the true size of a real pay gap. So you cannot tell whether
your audit found the right answer, missed it, or invented one.

## The solution

**Build a dataset where we decide the truth.**

`src/paybands/data/synthetic.py` invents employees and pays them according to rules *we* write —
including a deliberate pay gap of a size *we* choose.

Then run the audit. If we inject 8% and the audit reports ≈8%, the audit works. If it reports 2% or
15%, the audit is broken and we fix it **before** pointing it at real people.

> **This is calibrating a thermometer in boiling water before trusting it on a patient.**

## How the fake salaries are built

The formula is explicit and documented — a reader can see exactly how each salary is constructed:

```
log(salary) = log(600,000)                                   base
            + log((1 + years/3) ** 0.85)                     experience, flattening
            + log(role multiplier)
            + log(location tier multiplier)
            + log(1 + (institute premium − 1) × exp(−years/8))   college premium, fading
            + log(previous company type multiplier)
            + log(education multiplier)
            + log(org size multiplier) + log(remote multiplier)
            + (career gap months / 12) × log(0.93)           the proxy channel
            + is_disadvantaged × log(1 − pay_gap)            ← THE INJECTED GAP
            + random noise
```

Several deliberate choices here:

**Built in log space**, so effects multiply rather than add — matching how salaries really work.

**Experience flattens**, matching the real curve from Part 4. A straight line would be wrong.

**The college premium fades with experience.** An IIT degree matters at year 1 and matters much
less at year 15. That interaction is explicit.

**Product versus services company** is included, because in Indian tech it is one of the largest
pay differences there is.

## The two critical design requirements

### 1. The injected gap

One group is paid `pay_gap` less for otherwise identical work. Default 8%. Configurable.

Setting `pay_gap = 0.0` must produce genuinely unbiased data — that is the **control case**, and
there is a test for it.

### 2. The proxy variable

This one is subtle and it is the most valuable idea in the project.

The generator also creates `career_gap_months` — time out of the workforce — which:

- is **correlated with gender** (one group is more likely to have career breaks)
- **independently reduces salary** a little

Why build this in? So that later we can demonstrate that **deleting the gender column does not
remove the bias**. The model rebuilds it from the career-gap column, which looks completely
innocent.

## Verification

The agent that wrote this reported the injected 8% gap recovered as 0.0769. That claim was checked
independently, on seeds the author had never used:

```
INJECTED 8% GAP
  seed 7     recovered 0.0790
  seed 99    recovered 0.0802
  seed 2024  recovered 0.0805

CONTROL — inject 0%
  seed 7     recovered −0.0011
  seed 99    recovered  0.0002

DIFFERENT SIZES
  injected 0.03 → recovered 0.0290
  injected 0.15 → recovered 0.1491
  injected 0.25 → recovered 0.2492
```

It tracks almost perfectly across the whole range. The generator does what it claims.

## A mistake worth showing you

The **first** verification attempt returned **exactly zero** where the agent claimed 8%.

Two possibilities: the agent was wrong, or the check was wrong.

The check was wrong. It filtered on `df.gender == 'F'`, but the actual values in the data are
`'female'` and `'male'`. **The filter matched nothing**, so of course the measured effect was zero.

> ### When your verification disagrees with a claim, suspect your own check first.
> It is wrong more often than the thing you are checking, and assuming otherwise wastes hours.

**Result: 25 tests.** The headline one recovers the injected gap with a simple regression — proving
the generator does what it says.

---

# Part 7 — Features and leakage

**Features** are the inputs a model uses: experience, role, city, skills. **Feature engineering**
is turning raw data into inputs a model can use well.

## The most expensive mistake in machine learning

**Leakage** means information from your test data sneaking into training.

Why it is so dangerous: it makes your model look **excellent** in testing and fail in production.
You ship it confidently. It breaks quietly. And by the time you find out, you have no idea why,
because your own measurements told you it was fine.

### A concrete example

Our skills feature works by finding the 25 most common programming languages and creating a
yes/no column for each.

Suppose you compute "the 25 most common skills" using **all** your data, then split into train and
test. You have leaked. The vocabulary was chosen with knowledge of the test set. Your test score is
now optimistic, and you will never know by how much.

The fix: compute the vocabulary on **training data only**, then apply it to test data unchanged.

### The test that proves it

There is a test where three languages (`zig`, `rust`, `elixir`) appear **only** in the test split.
It asserts they never enter the fitted vocabulary.

And — this is the good part — there is a **deliberate contrast test** that fits on train and test
combined and **asserts the bug appears**. It documents the failure mode rather than just avoiding
it. Anyone reading the test file learns what leakage looks like.

## What the feature code does

| File | What it does | Key decision |
|---|---|---|
| `experience.py` | Turns years into several forms: `log1p`, buckets, square root | The relationship is not linear, so a raw years column would mislead the model |
| `skills.py` | Semicolon-separated skills → top-25 yes/no flags + a count | Vocabulary fitted on train only |
| `location.py` | Indian city names → tiers 1/2/3 | Handles spelling variants: `bengaluru`, `Banglore`, `BLR`, trailing spaces |
| `builder.py` | Combines everything into a model-ready table | Freezes category levels at fit time |

## Two decisions worth understanding

**Categories are left as categories, not one-hot encoded.** One-hot encoding turns a column with
200 job titles into 200 columns of mostly zeros. Decision trees split badly on those. LightGBM
handles categorical columns natively and better. This is documented in the code with the reasoning.

**Missing values are left missing.** They are *not* filled with 0.

> A zero is a **claim that the value is zero**. If you do not know someone's experience and write
> `0`, you have told the model they are a fresher. That is a lie. LightGBM handles missing values
> natively and learns what "unknown" means.

Similarly: an unrecognised city becomes tier 3 (a real claim — probably a smaller town), but a
*missing* city becomes `NaN` (we were not told). "We do not know" and "small town" are different
statements.

**Result: 65 tests**, including the leakage test and its deliberate contrast.

---

# Part 8 — The actual model

## Choosing an algorithm

**LightGBM**, a gradient boosting library.

**Gradient boosting** means: build many small decision trees, where each new tree tries to fix the
mistakes of the ones before it. For **tabular data** — data in rows and columns, like a spreadsheet
— it is the best general-purpose choice available, beating both linear models and neural networks
in almost all cases.

Note what we did *not* use: scikit-learn's own models are supporting tools here, not the main
event, and neural networks would be the wrong tool for 1,022 rows of tabular data.

## The band: three models, not one

Here is the key idea.

Instead of training one model to predict "the salary", we train **three**:

- one predicting the **10th percentile** (the low edge)
- one predicting the **50th percentile** (the middle)
- one predicting the **90th percentile** (the high edge)

The gap between the outer two **is the band**.

### Why not just one prediction with a ± error bar?

Because a fixed ± range would be **equally wide for a fresher and a CTO**. That is nonsense.
Uncertainty about a junior's salary is genuinely smaller than about a senior architect's.

Three models learn that difference from the data. Measured: band width in rupees grows **5.0×**
from the lowest-predicted candidates to the highest. The band is learned, not assumed.

## A failure mode that catches people out

The three models are trained independently. **Nothing stops the 10th-percentile model predicting
above the 90th-percentile model** on some rows.

That is called **quantile crossing**. The code detects it, repairs it by sorting, and — importantly
— **counts how often it happened** and reports the count. It occurred on 0–1.5% of rows.

Quietly sorting and never mentioning it would hide a real signal about model quality.

## Small data needs a small model

With 1,022 rows, default LightGBM settings would **overfit** badly — memorising the training data
instead of learning patterns.

So: shallow trees (depth 3), few leaves (7), a minimum of 25 samples per leaf, strong
regularisation. These are hand-chosen, not tuned, and that is written down.

## Did it beat the baseline?

Verified across four seeds the authoring agent never used:

```
 seed    model MAE      baseline MAE    beats?
   11   ₹12,43,398     ₹13,99,644       YES
   77   ₹13,41,140     ₹15,01,370       YES
  909   ₹11,04,950     ₹12,17,458       YES
 5150   ₹12,72,230     ₹13,90,950       YES
```

Combined with the agent's six seeds: **10 out of 10 splits.** Final measured figure across 10
seeds: **11.9% better MAE (range 7.4%–17.5%)**.

Consistency across ten splits is what makes this a real result rather than a lucky one.

## Now the important part — is the band honest?

We ask the model for an "80% band". **Does it actually contain 80% of real salaries?**

Almost nobody checks. Here is what we found:

```
  raw quantile band:  74.9% coverage   ← promised 80%
```

The band was catching only three-quarters of real salaries while advertising 80%.

**Asking LightGBM for the 90th percentile is not the same as getting it.** Between the request and
the answer sit a finite training set, a small model, and a test set nobody has seen. The promise
leaks out through all three. Nothing in the output announces this. You only find out by counting.

## The fix: conformal prediction

**Conformalised Quantile Regression** works like this:

1. Hold out a third slice of data — the **calibration set** — that the model has never seen
2. Measure how badly the band misses on that slice
3. Widen the band by exactly that much

This gives a mathematical coverage guarantee. Result:

```
  raw:        74.9%
  conformal:  82.3%   (range 79.4% – 87.3%)
```

Measured at five confidence levels rather than one — because one level checked is a spot check,
five is a calibration curve:

| promised | raw delivered | conformal delivered |
|---|---|---|
| 50% | 43.0% | 48.5% |
| 60% | 53.1% | 59.8% |
| 70% | 63.1% | 67.3% |
| 80% | 72.9% | 81.7% |
| 90% | 86.8% | 90.4% |

The raw band under-covers at **5 of 5** levels. The calibrated band's worst miss is −2.7 points.

## The three-way split, and why it is mandatory

Conformal prediction requires **train / calibration / test**:

- Calibrating on **training** data destroys the guarantee — the model has already seen it
- Calibrating on **test** data is leakage — your evaluation is now meaningless

Both mistakes produce impressive numbers that are entirely false. The code enforces the three-way
split structurally so it cannot be got wrong by accident.

## And now the bad news

```
  raw band width:         1.94× the midpoint
  calibrated band width:  2.40× the midpoint
```

For a predicted midpoint of about ₹14 lakh, the honest answer is roughly **₹4,91,660 to
₹39,84,224**.

**No recruiter can make an offer from that.**

### But the model is not broken

Look at what it has to work with. Self-reported survey answers, where half the respondents skipped
the salary question, with **no city, no seniority level, no company tier, no performance rating,
and no previous salary**. Seven of the strongest known drivers of an Indian tech salary are simply
absent.

Two people identical on everything the survey *does* record can legitimately earn ₹8 lakh and ₹35
lakh.

**A model that produced a narrow band from this data would be lying.** The wide band is the model
correctly reporting that the data is too weak.

> ### A well-built model on weak data gives you an honest "I do not know."
> ### A badly-built model on weak data gives you a confident wrong answer.
> Most people never find out which one they built, because they never measure coverage.

Note also: honesty **cost** 23.4% more width. The narrower 1.94× band is the one that lies about
its own confidence.

**Result: 50 tests** across the band model and conformal calibration.

---

# Part 9 — The fairness audit

## Two very different numbers

**Raw gap** — the plain difference between groups, with no adjustments.

The audit code has a warning in its own docstring: **this is not evidence of discrimination.**
Groups genuinely differ in experience, seniority, and role. Reporting a raw gap as proof of bias is
a common and serious error.

**Adjusted gap** — same experience, same role, same location: *is the pay still different?*

That is the meaningful number. It comes with a **confidence interval**, because a bare point
estimate on a small sample invites overclaiming.

## Calibrating against known truth

Pointed at synthetic data with gaps we injected deliberately:

| injected | recovered | 95% confidence interval | truth inside? |
|---|---|---|---|
| **0%** | **−0.34%** | −1.39% to +0.71% | yes |
| 3% | 2.67% | 1.65% – 3.68% | yes |
| 8% | 7.69% | 6.72% – 8.65% | yes |
| 15% | 14.71% | 13.81% – 15.60% | yes |
| 25% | 24.75% | 23.95% – 25.53% | yes |

**The most important row is the first.** With zero real bias, the audit reports −0.34% — it does
**not invent bias in fair data**.

An audit that cries wolf is worse than no audit at all, because it is confidently wrong in the
direction that gets someone accused.

## The demonstration that makes this project distinctive

There is a claim you will hear in real companies, in real compliance documents:

> *"We removed gender from the training data, so the model cannot be biased."*

**This is false, and here is the controlled refutation.**

The same model, fitted twice on synthetic data carrying an 8% injected gap — once **with** the
gender column, once **without** — with the gap then measured on held-out predictions:

| seed | gap with gender | gap without | surviving |
|---|---|---|---|
| 0 | 10.5% | 3.6% | 34.3% |
| 1 | 9.4% | 3.8% | 41.0% |
| 2 | 11.4% | 4.5% | 39.8% |

Verified independently on three further unseen seeds: **40%, 35%, 39%**.

**Mean 38.4% of the gap survives deleting the protected column.**

### Why it survives

The bias travels through `career_gap_months` — a feature that predicts pay honestly *and* carries
group membership (association 0.367 with gender, versus 0.004 or less for nearly every other
column).

The model cannot see gender any more. It can see career gaps. It rebuilds most of the same
prediction.

And note: the accuracy cost of removing gender was only 1.3%. **Nothing meaningful was even traded
away for the harm that remains.**

### Two controls that make this an experiment rather than an anecdote

- **Negative control:** switch the proxy off in the generator, and survival collapses to ~0
- **Reproduced with LightGBM**, not just linear regression — so it is not an artefact of one
  algorithm

The correlation scan that finds proxies runs **before any model is trained**. That is why it should
be the first thing pointed at company data. Our synthetic world has exactly one proxy channel
because we built it with one. **A real dataset has several, and nobody labels them.**

## The audit that could not run

We then tried to run the fairness audit on the real survey data.

**All 172 columns were checked for**: `gender`, `ethnic`, `race`, `sexuality`, `orientation`,
`transgender`, `disability`, `accessibility`, `nationality`, `religion`, `caste`.

**Every one returns zero.** Stack Overflow no longer publishes demographic fields.

This is not missing values. The question is not in the survey, and no imputation or join recovers
it.

So two things are now true at once:

1. The audit is **provably correct** — it recovers injected gaps almost exactly
2. It has **nothing real to point at**, until company data arrives

The only demographic present is `Age`. On the 702 people it covers, the raw gap is 43.9% — which
sounds enormous — and the adjusted gap is **−5.22%, with a 95% interval from −33.6% to +17.1%**.

The honest reading of a 51-point interval is **"we cannot tell"**. That is a different sentence
from "there is no gap", and the project says the first one.

**Result: 37 tests.**

---

# Part 10 — Charts that are evidence

Plots here are not decoration. Each one proves a specific claim.

## The rule that matters most

**Rupees are formatted in lakhs and crores, not millions.** A chart reading "₹1.5M" is unreadable
to the Indian audience it is for. This is a small detail that separates a tool built *for* users
from one built *at* them.

## The two most useful charts

**Distribution, raw versus log** (`reports/dist_raw_vs_log.png`). The same salaries on two axes.
Raw shows skew +2.86 and a long tail out to ₹2 crore. Log shows −0.23 and near-symmetry. Median
₹15L and mean ₹24.2L are both marked, so you can *see* the mean being dragged right by the tail.

That single image settles the "why do we log-transform?" question without a paragraph of text.

**Salary by experience** (`reports/salary_by_experience.png`). The steep-then-flat curve, with two
things most charts omit:

- the **sample size printed under every bucket**
- a **hollow marker** on any bucket with fewer than 50 people

The 20+ bucket shows `n=36`, a hollow marker, and a confidence band spanning roughly ₹25L to ₹52L.
The flat line that might look like a finding now visibly announces itself as noise.

> A chart that hides sample size invites exactly the misreading it should prevent.

## The most important chart in the project

**The calibration curve** (`reports/calibration_conformal.png`). Promised confidence on one axis,
actual measured coverage on the other, with a diagonal line for perfect.

Points below the line mean the model is over-confident.

If you put one image in a portfolio README, it is this one — because it is visual proof that your
uncertainty estimates mean something.

## Technical choices

- Non-interactive backend, so charts render on a server with no display
- Every function takes an optional axis and returns the figure, so charts compose
- **Nothing writes files as a side effect** — there is one explicit `save()` helper
- Colourblind-safe palette, validated across all pairs

**Result: 37 tests**, including that the rupee formatter produces `₹15L` and `₹1.2Cr` correctly,
and that coverage calculations match a hand-computed case.

---

# Part 11 — The layer with no ML in it

## Compa-ratio: one division that runs everything

```
compa-ratio = actual salary ÷ midpoint of predicted band
```

| Value | Meaning | Action |
|---|---|---|
| below 0.90 | paid below band | correct it; genuine flight risk |
| 0.90 – 1.10 | within band | normal increment |
| above 1.10 | paid above band | smaller increment, or hold |

That single number drives **both** use cases. Pay equity is "list everyone below 0.90". Increments
weight compa-ratio against performance.

It is real HR vocabulary. Using it correctly makes the tool sound like it was built by someone who
understands compensation, not just someone who understands pandas.

## Worked examples, from the real code

**Below band** — ₹12L against a ₹14L midpoint, performance rating 3:

> *₹1.6L (13.4%): 8% merit for a rating of 3, plus 5.4% equity correction — at 0.86 of band
> midpoint (below band); this brings pay up to the 0.95 target.*

**Above band** — ₹24L against a ₹20L midpoint, **same rating 3**:

> *₹96,000 (4%): 4% merit for a rating of 3, cut to 50% of normal because pay is above band — at
> 1.20 of band midpoint (above band).*

Same performance, very different outcome — because position in band matters as much as performance.
And both explanations are readable by someone with no technical background.

## Why no second model

It is tempting to train a model to predict increments. **Do not.**

How big is the budget? How does a rating convert to money? Do you close an equity gap in one year
or three? Those are **company decisions**, not patterns in data. As rules, HR can read and argue
with them. Inside a model, nobody can.

## A design detail worth stealing

**The explanation sentence is *derived* from the numbers, not stored alongside them.**

So when a budget cut trims the amount, the sentence rewrites itself automatically. Text and figures
**cannot** drift apart.

The classic bug here — a recommendation saying "12%" next to an amount that is actually 9% after a
budget pass — is structurally impossible.

## Policy choices made explicit

All in `configs/policy/increment_fy_2026_27.yaml`, all arguable, all documented:

- Equity gaps close over **2 years**, not one — a 40% correction in a single year is rarely approvable
- **15% single-year ceiling** on any correction
- When budget runs short, **equity is funded before merit**, worst-compa-ratio first, in full
  rather than pro rata — fixing some people completely beats half-fixing everyone
- Rounding is always **down**, so rounding can never breach a budget

**Result: 46 tests.**

---

# Part 12 — The API

Three endpoints, built with FastAPI:

| Endpoint | Returns |
|---|---|
| `POST /predict-band` | band, take-home, confidence, explanation, caveat |
| `POST /compa-ratio` | ratio and position label for an actual salary |
| `GET /health` | model loaded, version, payroll config year |

## The caveat is mandatory, and it fires

Given the band is 2.40× its own midpoint, an endpoint returning clean JSON numbers without saying
so would be **actively misleading** whoever calls it.

Every response carries the band width relative to midpoint and a plain-English warning:

> *"This band spans 68% of its own midpoint (₹8,39,000 to ₹15,33,000 around ₹10,15,000).
> NOT DECISION-GRADE: that is wider than the 50%-of-midpoint limit a band has to meet to be
> quotable (published HR bands typically run 80%–120% of midpoint). Use it as rough market
> orientation only — do not put these numbers in an offer..."*

**The threshold is set from HR practice, not from what the model achieves.** That distinction is
important: a threshold tuned to your model's current performance always passes, which makes it
decoration. This one fails on every request today.

> **A quality bar you set after seeing your results is not a quality bar.**

## Strict input validation caught a real mistake

The request schema uses `extra="forbid"` — unknown fields are rejected.

While testing, the author sent `{"years_experience": 3, "role": "Backend"}` when the schema wants
those nested under `candidate`. The API returned **422 with all three wrong fields named**.

Now consider the alternative. Pydantic's *default* is to silently **ignore** unknown fields. That
request would have returned **200 OK with a perfectly plausible band** — computed from no inputs at
all, because everything sent was discarded.

> **A strict schema turns a silent wrong answer into a loud error.** The dangerous failure is not a
> crash; it is a confident answer to a question you never actually asked.

## Other decisions

- The model loads **once at startup**, not per request
- If no trained model exists, the API **starts and reports unhealthy** rather than crashing — so it
  is deployable before a model artifact exists
- `/health` returns **503** when unloaded, so a load balancer reads the status code and a human
  reads the message
- The served model trains on **synthetic data by default** and labels itself
  `"NOT real market data"` — so nobody mistakes a plumbing demo for a market estimate

**Result: 66 tests.**

---

# Part 13 — Explaining a prediction

## Why explanations are not optional

**A recruiter must be able to argue with the model.**

A salary number with no reasoning attached cannot be challenged — and an unchallengeable salary
number is exactly the kind that causes harm. It is also what makes a decision defensible if anyone
questions it later.

## What it produces

> *Predicted ₹17,67,000 — 8 years of experience added ₹2,60,000, a tier-1 (metro) location added
> ₹1,93,000, company size not given subtracted ₹1,01,000.*

This uses **SHAP**, a method that splits a prediction into per-feature contributions.

## The honesty problem, and how it was solved

The model trains on `log(salary)`, so SHAP returns values like `0.159`. Meaningless to a recruiter.

Converting to rupees has a catch: **log contributions add up exactly; rupee contributions do not**,
because `exp()` is not linear.

Both conversions are published and neither is fudged:

**Multipliers** — `exp(shap)`. These are exact. Baseline × all multipliers = the prediction.
Verified: 11 log contributions summed to within 0.0001 of `log(prediction) − log(baseline)`.

**Rupees** — a leave-one-out figure: "remove this factor and the prediction falls by this much".
Exactly true one factor at a time, and it does **not** sum. On one example the parts totalled
₹1,66,000 against a real change of ₹1,93,000.

**The ₹27,000 residual is published**, and `approximation_note` is a **required** response field
saying so in words.

> Nudging those five numbers so they sum neatly takes two lines and nobody would ever notice.
> **The tidy total would have been the dishonest one.**

## Two more honest choices

**Missing inputs are explained, not hidden** — *"company size not given, ×0.946"*. It looks odd. It
is correct: the model learned a response to missingness, and concealing that would misrepresent
what it actually did.

**The waterfall chart is drawn on a log axis**, so the steps multiply and land exactly on the
prediction. A rupee-space waterfall would need a fudge bar to close.

**Result: 25 tests**, including that SHAP values reconstruct the prediction in log space.

---

# Part 14 — Measuring everything properly

The final piece: `scripts/run_analysis.py` — **one script that regenerates every number and every
chart** in the project's findings.

Seeded, deterministic, ~30 seconds, verified byte-identical across runs. All 467 numeric values in
`docs/findings.md` were checked against what the script prints. **No number appears in the findings
document that the script does not produce.**

Every headline comparison runs across **10 splits** and reports mean *and* range.

## New findings from that analysis

**Experience is nearly the whole model.** `years_experience` alone is 36.4% of importance and
₹3,07,623 of permutation error (23.1% of the model's total error). All four experience encodings
together are about four-fifths of the model's predictive power.

**16 of 37 features carry no signal at all.** Only 9 move the error by more than 1%.

**A textbook demonstration, unplanned.** `n_skills` ranks **#2** on the importance chart LightGBM
prints by default (12.3% of split count, ahead of `org_size`) — but shuffling it costs only ₹3,318,
about 0.25% of error.

Why? It is an integer running 0–20, so a tree can split on it at many thresholds.

> **Split count measures how many questions a feature *could* answer, not how useful the answers
> were.** Anyone reporting LightGBM's default importance is reporting cardinality as much as
> signal.

The script therefore computes **three** importance measures and prints the disagreement.

**`education` is negative** — permutation importance −₹2,675. Shuffling it *improves* the model. On
613 training rows the model is fitting noise in that column. Education went onto the list of fields
to **drop** from the company data request.

**The error is concentrated, not spread.** Mean absolute bias in roles with fewer than 5 test rows
per split is **40.8%**; elsewhere it is **10.3%**. The worst cell is QA — 55.6% coverage against an
80% promise, and −64.2% bias, on 8 respondents. Security is the mirror image at +56.1%, on 5.
Meanwhile Fullstack, Backend and Frontend sit at −2.8%, +2.0% and +0.2%.

> This is not a mediocre model. It is a decent model next to a catastrophic one — and the
> catastrophic half is exactly the roles a company would most want a band for, because they are the
> ones nobody in the room has an intuition about.
>
> **Concentrated error is fixable by collecting 200 QA rows. Average error is not.**

## One problem left open, deliberately

The raw band's coverage shortfall is **−7.0, −6.9, −6.9, −7.1** points at four of five confidence
levels.

Four numbers within 0.2 points of each other is **not noise**. Something systematic is happening.

Best guess: pinball loss on 613 rows with `min_child_samples=25` is pulling the extreme quantiles
inward. Testable by sweeping that parameter.

Conformal calibration fixes the *symptom*. The *cause* is undiagnosed.

> **Fine engineering, unfinished science.** Writing that down — with a hypothesis and the experiment
> that would settle it — is more credible than pretending everything is resolved.

---

# Part 15 — Every mistake

Collected in one place, because these are the most useful part of the document. Every one is real.

## Mistakes about the project itself

**1. Built the wrong project first.** An LLM evaluation harness was scoped, planned, and built
through Phase 0 — then deleted, because the goal was model building and that project was mostly
plumbing. *Cost: one day. Cost if noticed in week six: six weeks.*

**2. Invented an example that looked like a measurement.** The band width was described as
*"a predicted ₹19L means ₹5L to ₹42L"*. That illustration was made up from the 1.9× figure — never
measured. It was replaced with the real pooled example (₹4,91,660 – ₹39,84,224 around ₹13,95,079).
*Lesson: an illustration that looks like a measurement should be a measurement.*

## Mistakes about measuring

**3. Reported coverage from four seeds.** Conformal coverage was reported as 70.6% → 77.2%. Across
ten seeds it is actually **74.9% → 82.3%**. The four seeds happened to land near the bottom of the
range.

This is exactly the mistake that had been *taught* two sections earlier — "one split is an
anecdote" — committed with four splits. Four is better than one. It still was not enough.

**4. Quoted the wrong band width.** The figure 1.94× was quoted while describing conformal results.
But 1.94× is the width **before** calibration — the version that promises 80% and delivers 75%. The
calibrated band is **2.40×**. *Quoting 1.94× means quoting the band that lies about itself.*

**5. Used the wrong kind of average.** Two defensible ways to average the same bands:

```
  mean(width) ÷ mean(midpoint)    2.15    ← weights by salary size
  mean(width ÷ midpoint)          2.61    ← per candidate
```

Same data, different question. The first asks *"across the whole payroll, how wide?"*; the second
asks *"for the person in front of me, how wide?"*. Since the tool answers questions about
individuals, the second is correct. The first had been used, and it flattered the result.

> **Pick the average that matches the decision, not the one that reads better.**

**6. Baseline improvement was reported as ~17%; the ten-seed figure is 19.4%.**

## Mistakes in verification

**7. Filtered on the wrong value and got a false zero.** Checking the synthetic generator, the
filter was `df.gender == 'F'`, but the data contains `'female'`. **The filter matched nothing**, so
the measured gap was exactly zero — appearing to disprove a correct claim.

**8. Multiplied a truncated list.** Verifying SHAP additivity, the top 5 contributions returned by
the API were multiplied together. There are **11**; the API truncates for readability. The
mismatch looked like a bug in the code.

**9. Read a rounding artefact as a defect.** With all 11 contributions the result was still off by
**₹191** on ₹1.77M. That looked real — until checking in log space, where additivity holds, showed
the sum correct to 1e-4. The ₹191 is the ₹1,000 display rounding, and it is *under half a rounding
step*, which is precisely what rounding must produce.

> Three verification mistakes in a row, all the checker's fault, none the code's.
> **When your verification disagrees with a claim, suspect your own check first.**

## The mistake that no test could have caught

**10. Built the payroll calculator on three wrong numbers, and proved it correct.** This is the most
instructive entry in the list, so it gets its own heading.

The salary structure was reverse-engineered from figures the owner reported **from memory**: gross
about ₹1,01,700/month, PF ₹3,600, insurance "about ₹250". From the PF figure the project inferred a
basic salary of `₹3,600 ÷ 0.12 = ₹30,000`, i.e. about 30% of gross, and concluded that the employer
must therefore ignore the ₹15,000 statutory PF ceiling — because the ceiling would have produced
₹1,800.

Then an actual payslip PDF arrived. Three of those values were wrong:

| Believed | Actual |
|---|---|
| basic = 30% of gross | **52%** |
| employer ignores the PF ceiling | employer **applies** it (₹1,800 = 12% × ₹15,000) |
| insurance ₹250 | **₹300**, and the line is called *Medical* |

Every one of those followed from the first, and the first came from a single misreading: the
₹3,600 was almost certainly the employee's ₹1,800 **plus** the employer's ₹1,800, which a payslip
prints on two separate lines. The reasoning was sound. The premise was not, and sound reasoning
turns a bad premise into confident, specific, wrong numbers.

**And the tests all passed.** Thirteen of them, throughout. That is the point. The config had been
tuned until they did, so the test and the config were both expressions of the same belief — one
witness, asked twice. No quantity of extra tests inside that loop could ever have exposed it.

The published take-home figure was ₹97,850. The correct projection, if the year 1 structure carries
forward, is **₹99,600** — and even that is a projection rather than a verified number, because the
payslip in hand predates the increment.

> **Tests check your code against your beliefs. Only a document checks your beliefs against the
> world.** For anything decided outside your project — tax law, company policy, a salary structure —
> get the document. A verbal figure is a hypothesis, and it should be labelled as one until paper
> arrives.

Corollary, and it was expensive here: **the phrase "validated against a real payslip" was in this
document for months, and it was false.** Nobody had seen the payslip. Claims of external validation
deserve the same scrutiny as the numbers themselves — write down which document, and when.

## Mistakes in the code

**11. Broke every payroll test with a comment.** A `verified_against` note was added to the payroll
YAML. `PayrollRules` uses `extra="forbid"`, so it rejected the whole file and 13 tests errored.

The fix was to add `verified_against` as a **real field**, not to loosen the validation.
`extra="forbid"` did its job and should keep doing it.

It then did it a **second** time, months later: adding `employer_matches` to the provident-fund
block was rejected on first write for exactly the same reason, until the field was declared on
`ProvidentFundRules`. Twice now, a strict schema has refused a config key that the code would not
have read. A permissive schema would have accepted both silently and left the setting doing
nothing — the failure mode where everything looks configured and nothing is.

> **A guard that has caught you twice is not being annoying. It is being useful twice.**

**12. Guessed API field names and got a 422.** Sent flat fields when the schema wants them nested.
The strict schema named all three wrong fields. Had the schema been permissive, the request would
have returned **200 OK with a plausible band computed from nothing.**

**13. A test caught an undefined input.** "Take-home should never be negative" failed on a salary
of ₹0, because the medical premium is a flat ₹300/month. The fix was **rejecting the input**, not clamping
the output — clamping would have hidden the fact that "valid input" had never been defined.

**14. A claim that was never true.** `data/public/README.md` said the survey was "downloaded by
`paybands fetch`". No such command existed — a placeholder written on day one. Rather than document
a manual step, the command was written.

## Mistakes in tooling

**15. `shap` broke the environment, twice.** It pulled in `llvmlite`/`numba` versions supporting
only Python 3.6–3.9. First response: quarantine it in its own optional group so a Phase 6
dependency could not block Phase 3 work. Later fix: pin `numba>=0.60` to force a modern resolution.

The error message was worth reading properly — it named a *Python version range*, and the hint line
named `numba`, not `shap`.

> **The broken dependency is rarely the one you asked for.**

**16. LightGBM would not import.** `Library not loaded: @rpath/libomp.dylib`. LightGBM is compiled
against OpenMP, which Apple does not ship. `pip install` reported success; the failure only appeared
at import.

> **When an error does not look like Python, stop reading Python documentation.** That was a linker
> error from the operating system. `brew install libomp` fixed it.

## Mistakes in process

**17. Parallel agents reformatted each other's files.** One ran `ruff format` across the whole tree
and touched another's in-flight files. Whitespace only, nothing broke — but the fix is to scope
formatters to each worker's own paths.

**18. `git add -A` swept in unfinished work.** A commit intended for two workstreams silently
included a third that had just landed, producing an inaccurate commit message. It was amended
(unpushed, so safe).

> **`git add -A` while background work is running commits code you have never tested.** Stage
> explicit paths instead.

---

# Part 16 — Every file, explained

Roughly **20,000 lines** across source, tests, configs and docs — of which about 2,050 are this
document. Here is what each part does.

## `src/paybands/payroll/` — Layer 2, the calculator

| File | Lines | What it does |
|---|---|---|
| `calculator.py` | 301 | Base salary → PF, medical, income tax → take-home, plus CTC (gross + employer PF). Marginal tax slabs, Section 87A rebate, statutory PF ceiling. Config-driven; no rate hardcoded. |

**Start reading here if you are new to the project.** It is the only part with a known right
answer, and it is validated against an actual payslip PDF — reconciled to the rupee, not to a
remembered figure. Part 15, mistake 10 explains why that distinction cost the project three wrong
numbers.

## `src/paybands/data/` — getting data in

| File | Lines | What it does |
|---|---|---|
| `schema.py` | 72 | The one table shape every source converts into. Also holds the cleaning thresholds as named constants with reasoning. |
| `stackoverflow.py` | 157 | Loads the survey, filters to India + INR, reports every dropped row with a reason. |
| `synthetic.py` | 491 | Invents employees from an explicit formula, with a **known injected pay gap** and a **proxy variable**. Returns the data *and* the ground truth. |

**Why `schema.py` matters:** all three sources produce the same columns. When company data arrives,
you write one loader and everything downstream already works — already tested against two other
sources.

## `src/paybands/features/` — preparing inputs

| File | Lines | What it does |
|---|---|---|
| `experience.py` | 180 | Non-linear transforms of years: log, buckets, sqrt. |
| `skills.py` | 165 | Semicolon string → top-25 flags. **Vocabulary fitted on train only.** |
| `location.py` | 249 | Indian city → tier 1/2/3, handling spelling variants. |
| `builder.py` | 266 | Combines everything; freezes category levels at fit time. |

**The whole package exists to prevent leakage.** Anything that *learns* from data is fitted on
training data only.

## `src/paybands/model/` — Layer 1, the machine learning

| File | Lines | What it does |
|---|---|---|
| `split.py` | 202 | Train/test splitting. Random for the survey; **temporal** for company data, where a random split would be leakage. |
| `metrics.py` | 319 | MAE, median error, MAPE, R², pinball loss, coverage. Always reported in rupees. |
| `baseline.py` | 289 | The two deliberately simple models the real one must beat, with a fallback hierarchy and a minimum group size. |
| `band.py` | 607 | Three LightGBM quantile models → the band. Crossing detection and repair. Compa-ratio. |
| `conformal.py` | 444 | Conformalised quantile regression, and the enforced three-way split. |

## `src/paybands/fairness/` — the audit

| File | Lines | What it does |
|---|---|---|
| `audit.py` | 1,028 | Raw gap, adjusted gap with confidence intervals, residual analysis. Model-agnostic — takes plain arrays. |
| `proxy.py` | 510 | Finds proxy variables; the with-versus-without demonstration. |

**Deliberately model-agnostic**, so it works against the baselines, the band model, or anything
built later.

## `src/paybands/policy/` — Layer 3, the rules

| File | Lines | What it does |
|---|---|---|
| `increment.py` | 914 | Merit + equity correction, budget allocation, the pay-equity list. **No machine learning.** |

## `src/paybands/api/` — serving

| File | Lines | What it does |
|---|---|---|
| `models.py` | 414 | Request/response schemas. `extra="forbid"` on requests. |
| `service.py` | 863 | Loads the model once; builds caveats; orchestrates prediction. |
| `app.py` | 110 | The three routes. |

## `src/paybands/explain/` and `plots/`

| File | Lines | What it does |
|---|---|---|
| `explain/shapley.py` | 861 | SHAP contributions, log→rupee conversion, plain-English sentences. |
| `plots/style.py` | 325 | Shared look; rupee formatting in lakhs and crores. |
| `plots/distributions.py` | 385 | Raw vs log distribution; salary by experience with sample sizes. |
| `plots/calibration.py` | 388 | **The coverage and calibration charts** — the most important plots here. |
| `plots/fairness.py` | 323 | Residuals by group; raw versus adjusted gap. |
| `plots/explain.py` | 213 | Contribution waterfall on a log axis. |

## `scripts/`

| File | Lines | What it does |
|---|---|---|
| `fetch_data.py` | 109 | Downloads the survey. Idempotent; temp-file-then-rename. |
| `run_analysis.py` | 1,069 | **Regenerates every number and chart.** ~30 seconds. |

## `configs/`

| File | What it does |
|---|---|
| `payroll/fy_2025_26.yaml` | Tax slabs, PF (with the statutory ceiling and employer match), medical premium. Every value annotated with the payslip line it came from. One file per financial year, never edited once written. |
| `policy/increment_fy_2026_27.yaml` | Budget, rating→merit table, compa thresholds, caps. |

## `docs/`

| File | Lines | What it does |
|---|---|---|
| `design.md` | 266 | The architecture and the reasoning. **Read before writing code.** |
| `findings.md` | 685 | Every measured result with caveats. |
| `company-data-request.md` | 168 | The document to hand to HR. |
| `JOURNEY.md` | this file | How it was all built. |

## `tests/` — 441 tests

| File | Tests | Covers |
|---|---|---|
| `test_api.py` | 66 | Endpoints, caveats, validation |
| `test_features.py` | 65 | **Including the leakage test and its deliberate contrast** |
| `test_policy.py` | 46 | Increment rules, budget allocation |
| `test_fairness.py` | 37 | Gap recovery against known truth |
| `test_plots.py` | 37 | Charts render; rupee formatting |
| `test_band.py` | 28 | Quantile models, crossing repair |
| `test_baseline.py` | 26 | Fallback hierarchy |
| `test_explain.py` | 25 | SHAP additivity |
| `test_synthetic.py` | 25 | **Injected gap is recoverable** |
| `test_metrics.py` | 23 | Coverage, pinball loss |
| `test_conformal.py` | 22 | Calibration, three-way split |
| `test_split.py` | 15 | Reproducibility, no overlap |
| `test_payroll.py` | 26 | **Against a real payslip PDF** — earnings-to-net reconciliation, the payslip's own tax projection, the PF ceiling, CTC versus gross, surcharge and marginal relief |

---

# Part 17 — How to continue

## Getting it running

```bash
brew install libomp                    # macOS only — LightGBM needs OpenMP
uv sync --extra dev --extra data --extra model --extra api --extra explain
uv run python scripts/fetch_data.py    # 134MB survey CSV
uv run pytest                          # 441 tests
uv run python scripts/run_analysis.py  # ~30s, regenerates every number and chart
```

If tests pass, everything in this document is reproducible on your machine.

## The five things to do next, in order

### 1. Send the data request to HR

`docs/company-data-request.md` is written and ready. Everything built so far is the argument for
it, and it is a strong one:

- the tool is built and verified
- the band is too wide to use, because seven known pay drivers are missing from public data
- the fairness audit **cannot run at all**, because no public dataset contains the column

Trim the request first. After the analysis, `education` is known to be worthless (negative
importance) and should be dropped. **Asking for less data makes approval easier and shows you know
what you need.**

Settle one question *before* receiving any data: does an anonymised version go in a public
portfolio, or stay internal forever? That changes how the repository is structured, and
restructuring afterwards is painful.

### 2. Verify the tax slabs

`configs/payroll/fy_2025_26.yaml` is marked **unverified** for income tax. Check
`incometax.gov.in` for the current financial year, then create `fy_2026_27.yaml`.

**Do not edit the old file.** New year, new file — so you can always recompute what someone's
take-home *was* in a past year.

Note what the payslip does and does not prove. It confirms the PF ceiling, the employer match, the
medical premium, the regime, the absence of professional tax, and the standard-deduction arithmetic
(its own annual projection of ₹6,00,000 → ₹5,25,000 taxable reproduces exactly). It does **not**
test the slabs, because the 87A rebate zeroes the tax at these income levels whatever the slabs say.

### 3. Get a current payslip and settle the PF question

One open question is left over from the correction in Part 15, and it takes one document to close.

At about ₹1,01,700/month the owner reported PF of ₹3,600. With the statutory ceiling applied, PF
stays at ₹1,800 however high the salary goes. So either the ₹3,600 was again employee ₹1,800 plus
employer ₹1,800 read together, or the company switched to full-basic PF after the increment.

RESOLVED: PF is ₹3,600, confirmed directly, and `fy_2026_27.yaml` now models it. The old text
below is kept because the reasoning is still worth reading — the projected take-home of ₹99,600
was exactly that, a
projection, resting on the assumption that the year 1 structure carried forward. **Ask for the
latest payslip PDF before quoting it as fact.** That is the whole lesson of mistake 10, and it would
be a poor showing to repeat it.

### 4. Chase the −7 point mystery

The raw band under-covers by 7.0, 6.9, 6.9, 7.1 points across four confidence levels. Four numbers
that close together are not noise.

Hypothesis: pinball loss on 613 rows with `min_child_samples=25` pulls the extreme quantiles inward.

Experiment: sweep `min_child_samples` and watch whether the shortfall moves. If it does, the cause
is confirmed and the raw band can be fixed rather than merely patched.

### 5. Fix the concentrated error

QA sits at 55.6% coverage and −64.2% bias on 8 respondents; Security at +56.1% on 5. Fullstack,
Backend and Frontend are within ±3%.

Concentrated error is fixable by collecting more rows in those roles. Average error is not.

## What company data will change

| Now | With company data |
|---|---|
| Band 2.40× midpoint, unusable | Should tighten substantially — seven missing drivers become available |
| Fairness audit cannot run | Becomes the headline capability |
| Random train/test split | **Must switch to temporal split** — `split.py` already supports it |
| One salary definition, self-reported | One consistent definition, from payroll |

**The temporal split is not optional.** With dated company records, training on the past and testing
on the future is the only honest evaluation. A random split there is leakage, and it is the single
most common serious mistake made with this kind of data.

## Rules to keep

1. **Every headline number carries its range and split count.** `spread()` in the analysis script
   physically cannot print a mean without its range — that guardrail exists because it was needed.
2. **Every claim traces to something the analysis script prints.** No number in the findings
   document that the script does not produce.
3. **Cleaning rules are named constants with reasoning**, in one place, so a reviewer can disagree
   in one line.
4. **Loaders report what they drop.** Silent filtering changes your dataset without telling you.
5. **Company data never gets committed.** The ignore rule was written before the folder existed and
   was tested with a fake file.
6. **One config file per financial year, never edited once written.**

---

# Part 18 — Glossary

Every term used in this document, in plain words.

## Basic ideas

| Term | Meaning |
|---|---|
| **Target** | The thing being predicted. Here: annual base salary. |
| **Feature** | An input the model uses. Experience, role, city. |
| **Row / sample** | One person's record. |
| **Training data** | Rows the model learns from. |
| **Test data** | Rows held back, used to check whether it learned anything real. |
| **Baseline** | A deliberately simple model your real model must beat to justify itself. |

## Data problems

| Term | Meaning |
|---|---|
| **Skew** | Lopsidedness. Salary is right-skewed: most people cluster low, a few earn far more. |
| **Log transform** | Predicting `log(salary)` instead of salary, so percentage changes become even steps. |
| **Leakage** | Test information sneaking into training. Makes results look great and production fail. |
| **Selection bias** | The people in your data differ systematically from the people you care about. Cannot be cleaned away. |
| **High cardinality** | A category column with very many distinct values, like job title. |
| **Censored data** | You know something has not happened *yet*, not that it never will. |

## Models

| Term | Meaning |
|---|---|
| **Gradient boosting** | Many small decision trees, each fixing the previous ones' mistakes. Best default for tabular data. |
| **LightGBM** | A fast, widely used gradient boosting library. |
| **Overfitting** | Memorising training data instead of learning patterns. Looks great in training, fails on new data. |
| **Regularisation** | Deliberately limiting a model so it cannot overfit. |
| **Hyperparameter** | A setting you choose (tree depth, leaf count) rather than something learned. |

## Uncertainty

| Term | Meaning |
|---|---|
| **Quantile** | A cut point. The 90th percentile is the value 90% of people fall below. |
| **Quantile regression** | Training a model to predict a percentile rather than an average. |
| **Quantile crossing** | When the low-percentile model predicts *above* the high-percentile one. Must be detected. |
| **Conformal prediction** | A method that makes an interval's confidence promise mathematically honest. |
| **Coverage** | Do your 80% intervals actually contain the truth 80% of the time? |
| **Calibration** | Whether stated confidence matches reality. |
| **Confidence interval** | A range the true value probably sits in. Wide interval = "we cannot tell". |
| **Bootstrap** | Re-sampling your data many times to estimate how uncertain a number is. |

## Measuring

| Term | Meaning |
|---|---|
| **MAE** | Mean absolute error — average miss, in rupees. Interpretable. |
| **MAPE** | Mean absolute percentage error. Explodes when true values are small; useless here. |
| **R²** | 1.0 is perfect, 0.0 is no better than the average, negative is worse than the average. Measures against the *mean*. |
| **Pinball loss** | The scoring rule for quantile predictions. |
| **Permutation importance** | Shuffle one feature; see how much error rises. Measures what a feature is *worth*. |
| **Split-count importance** | How often trees split on a feature. Biased toward high-cardinality features — measures opportunity, not value. |
| **Seed** | A number fixing the random draws, so results reproduce exactly. |

## Fairness

| Term | Meaning |
|---|---|
| **Protected attribute** | A characteristic it is unlawful or unethical to discriminate on. |
| **Raw gap** | Plain difference between groups. **Not evidence of discrimination on its own.** |
| **Adjusted gap** | The difference remaining after controlling for legitimate factors. The meaningful number. |
| **Proxy variable** | A feature that secretly carries a protected attribute. Career gaps ≈ gender. |
| **Ground truth** | The real answer. Known in synthetic data; unknown in real data. |
| **SHAP** | Splits one prediction into per-feature contributions. |

## Compensation

| Term | Meaning |
|---|---|
| **Base salary** | The negotiated annual figure. What this model predicts. |
| **Basic** | One *component* of gross pay, sitting alongside HRA, special allowance and the rest. PF and gratuity are calculated on it. Often quoted as "about a third of gross", but that is a habit, not a rule — in the payslip behind this project it is **52%**. Read it off the document; do not assume. |
| **Gross** | The total of all earnings components, before any deduction. Basic + HRA + special + anything else. |
| **CTC** | Cost to company: gross **plus** what the employer pays on your behalf, chiefly its own PF contribution. Always larger than gross. It is what an offer letter usually quotes and it is **not** money that reaches you. |
| **Take-home / in-hand** | What reaches the bank: gross minus PF, medical and income tax. **Calculated, never predicted.** |
| **PF** | Provident Fund. A compulsory retirement saving. The employee contributes 12%, and the employer contributes the same again — but only the employee's half is a deduction from pay. The employer's half sits in CTC. |
| **PF ceiling** | Employers are only obliged to compute PF on the first **₹15,000/month of basic**, which caps the contribution at ₹1,800. Some pay on full basic instead, which is more generous. Which one a company does is a policy choice you cannot deduce — you must read a payslip. |
| **HRA** | House Rent Allowance. An earnings component, commonly set at 50% of basic. |
| **Standard deduction** | A flat ₹75,000 subtracted from gross before tax is computed, for salaried people under the new regime. |
| **Taxable income** | Gross minus the standard deduction. Every tax threshold, including the 87A rebate, is written against *this*, not against gross. |
| **Section 87A rebate** | Wipes out income tax entirely when taxable income is at or below ₹12,00,000 — so the tax-free **gross** salary is ₹12,75,000. All-or-nothing, not tapered, so it produces a genuine cliff just above the threshold. |
| **Marginal tax** | Each rate applies only to the income inside its own band — not to your whole salary. |
| **Compa-ratio** | Actual salary ÷ band midpoint. Below 0.90 = underpaid. |
| **Band** | A salary range rather than a single number. |

---

## The one thing to take away

Six months from now the code will have changed. This will not:

> **A well-built model on weak data gives you an honest "I do not know."**
> **A badly-built model on weak data gives you a confident wrong answer.**
>
> The only difference between the two is whether somebody measured.

Everything in this project — the bands instead of numbers, the coverage check, the fairness audit
calibrated against an injected gap, the mandatory caveat in every API response, the corrections in
Part 15 — is one idea repeated: **find out whether your numbers mean anything, and say so plainly
when they do not.**
