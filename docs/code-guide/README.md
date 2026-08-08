# Code Reading Guide

A file-by-file tour of this codebase, written for someone who can program but has never done
machine learning. Every class and function is explained in plain language, with the reasoning
behind it.

**You do not need anyone's help to use this.** Every term is defined the first time it appears.

---

## Before you start

Read these two first. They are short and everything else assumes them:

| Document | Why | Time |
|---|---|---|
| [`../design.md`](../design.md) | The three-layer architecture and the single rule behind it | 15 min |
| [`../JOURNEY.md`](../JOURNEY.md) Parts 0–2 | What the project is and the one decision that shaped it | 20 min |

If you only have five minutes, read this much:

> The project predicts a salary **band** — a range, not a number — because nobody can predict a
> salary to the rupee, and a tool that pretends otherwise lies to whoever uses it.
>
> It has **three layers**: a model that *learns* base salary from data, a calculator that
> *computes* PF, insurance and tax by formula, and a set of policy *rules* for increments. The rule
> behind that split: **never make a model learn something you can compute.**

---

## The four parts

Read them in this order. Each builds on the one before.

| Part | Covers | Files | Time |
|---|---|---|---|
| **[01 — Payroll and data](01-payroll-and-data.md)** | The calculator, the data schema, both loaders | 6 | ~60 min |
| **[02 — Features and model](02-features-and-model.md)** | Feature engineering, splitting, metrics, baselines, the band model | 10 | ~2 hrs |
| **[03 — Fairness, policy, explanations](03-fairness-policy-explain.md)** | The bias audit, the proxy demonstration, increment rules, SHAP | 6 | ~2 hrs |
| **[04 — API, plots, scripts](04-api-plots-and-scripts.md)** | Serving, charts, the analysis script | 10 | ~90 min |

**Total: about six hours** to read properly. Not one sitting.

---

## Why this order

It is not alphabetical and it is not by folder. It goes **from certain to uncertain**.

**Part 01 starts with `payroll/calculator.py`** — the only file in the project with a *known right
answer*. Tax law is published. A real payslip exists to check against. You can be completely
certain whether the code is correct.

That matters when you are learning: you get to understand the shape of the project (typed configs,
tests that check against reality, decisions written down as comments) on a file where nothing is
ambiguous.

Then it moves outward into the parts where the right answer is unknown, and where the interesting
question stops being *"is this correct?"* and becomes *"how would I even tell?"*

---

## Three ways to use this guide

### The quick tour — 45 minutes

You want to know what the project does without reading it all.

1. `../README.md` — the results and the honest limitation
2. Part 01, the section on `payroll/calculator.py` — see the house style
3. Part 02, the section on `model/band.py` — the core idea, three quantile models
4. Part 03, the section on `fairness/proxy.py` — the most interesting result in the project

### The full read — six hours

Parts 01 → 04 in order, with the actual source file open beside the guide. The guide tells you what
you are looking at; the code shows you how it is done.

Do one part per sitting.

### "I need to change something"

| I want to... | Read |
|---|---|
| Add company data | Part 01 (`schema.py`, `stackoverflow.py` as the template) |
| Change how salary is predicted | Part 02 (`band.py`, `conformal.py`) |
| Add a feature (a new input column) | Part 02 (`features/builder.py`) |
| Change increment rules | Part 03 (`policy/increment.py`) — and note it is config-driven, so you may only need the YAML |
| Update tax slabs | Part 01 (`configs/payroll/`) — and **create a new file, never edit the old one** |
| Add an endpoint | Part 04 (`api/`) |
| Add a chart | Part 04 (`plots/style.py` first — it sets the conventions) |

---

## How to read code you did not write

Some habits that make this much easier, if you have not done a lot of it:

**Read the module docstring first.** In this project every file opens with several paragraphs
explaining *why* it exists. That is deliberate. It is usually more useful than the code below it.

**Read the tests before the implementation.** A test file tells you what the code is *supposed* to
do, in small examples. `tests/test_payroll.py` is the friendliest place to start.

**Do not read top to bottom.** Find the main entry point — usually the one public function — and
follow it. Ignore the helpers until you reach them.

**Run it.** Open a Python shell, import the thing, call it with made-up numbers, print the result.
Ten minutes of that beats an hour of reading.

```bash
uv run python
>>> from paybands.payroll import PayrollRules, compute_payslip
>>> rules = PayrollRules.from_yaml("configs/payroll/fy_2025_26.yaml")
>>> print(compute_payslip(1_080_000, rules).explain())
```

**When something looks wrong, check the comments before assuming it is a bug.** Several things in
this codebase look odd and are deliberate — rejecting a zero salary, publishing a residual that
does not add up, counting quantile crossings rather than silently fixing them. Each has a comment
explaining why.

---

## Conventions used throughout the codebase

Recognising these will save you time in every file.

| Convention | What it means |
|---|---|
| **Config over constants** | Anything a business might argue about — tax rates, increment budgets, thresholds — lives in a YAML file, not in code. |
| **One config file per financial year, never edited** | So you can always recompute what a past year's numbers *were*. New year, new file. |
| **`extra="forbid"` on typed models** | Unknown fields are rejected loudly rather than silently ignored. A typo becomes an error, not a wrong answer. |
| **Fit on train, transform on test** | Anything that *learns* from data is fitted on training data only. This prevents leakage — see Part 02. |
| **Report what you drop** | Loaders count and print every row they filter out, with a reason. Silent filtering changes your dataset without telling you. |
| **Rupees, not log units, in anything a human reads** | The model works in log space; every number reported is converted back. |
| **Judgement calls are named constants with reasons** | So a reviewer can disagree in one line instead of hunting through code. |

---

## What the guide does not cover

**Test files**, except three that are worth reading as examples:

- `tests/test_payroll.py` — what it looks like to test against reality rather than against yourself
- `tests/test_features.py` — the leakage test, plus a deliberate contrast test that asserts the bug
  *does* appear when you do it wrong
- `tests/test_fairness.py` — validating a measurement tool against a known answer

The other ten test files follow the same patterns.

**`__init__.py` files** — they only re-export names.

---

## If you get stuck

1. The module docstring of the file you are reading
2. The relevant Part of [`../JOURNEY.md`](../JOURNEY.md) — it explains why the file exists at all
3. [`../JOURNEY.md`](../JOURNEY.md) Part 18 — a glossary of every term used
4. Run the thing in a shell with small inputs and print what comes out
