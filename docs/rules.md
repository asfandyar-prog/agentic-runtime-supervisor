# Rules — how this codebase gets built

## Style

- Minimal and readable over clever. No premature abstraction: no plugin systems, no registries, no base-class hierarchies for things that occur once.
- Every constant lives in `config.py` with a name and a one-line rationale. No magic numbers in logic.
- Dependencies: `torch`, `torchvision`, `medmnist`, `langgraph`, `numpy` — nothing else without an entry in memory.md justifying it. No matplotlib (deliverable is a table, not plots). No requirements.txt (uv + pyproject.toml only).
- Type hints on public functions; docstrings only where the name doesn't carry the meaning.
- Deterministic: seed 0 set once, threaded everywhere (dataloaders, stream schedule, ε-exploration, TTA).

## Comments: the WHY rule

The policy logic (`policy.py`, and the benefit/cost definitions) gets **why-comments**, because it is defended to reviewers on a deep-dive call. Specifically comment: why running frequency estimates and not thresholds; why the drift-only bucket with a set-size gate; why ABSTAIN/ESCALATE share a benefit definition and why the escalation info bonus exists; why counterfactual replay is needed under gated feedback; why ε-exploration is mandatory here. Elsewhere, comment only what the code cannot say.

## What NOT to build

- No learned/RL policy, no learned cost model (thesis-scale future work).
- No real clinical data; DermaMNIST only, one dataset.
- No production infra, no auth, no serving, no scaling, no Docker.
- No multi-dataset, no multi-seed sweeps, no hyperparameter searches (single exception: the fixed-threshold monitor's grid search, which is part of the experiment itself).
- No continual TTA, no per-sample selective prediction, no ACI (RQ4 is thesis work).
- No plots, dashboards, or web UI. One table + one ablation table.
- No LLM calls inside the graph.

## Honesty rules

- Single seed is acceptable; the variance limitation is **disclosed in RESULTS.md**, never hidden.
- Every simplification (adapt-then-recalibrate, counterfactual replay, escalation bonus, test-image reuse, oracle-tuned monitor baseline) is flagged where it lives in code and listed in RESULTS.md.
- A negative result is reported as a result. No metric shopping: the primary metric (net calibration benefit) is fixed in design.md §4 before any run.
- memory.md is updated **at the time a decision is made**, not retrofitted at the end.
- **Label-path separation is airtight in code.** The harness plays two roles: metric reporter (full labels — fine) and supplier of the policy's replayed learning signal (must stay gated: escalation + monitoring batches only). These paths never share code or data structures: policy/bandit updates consume a feedback object that is constructible *only* from gated sources; reporting metrics never flow into memory. This invariant is defended on the review call; any leak is a release blocker.
