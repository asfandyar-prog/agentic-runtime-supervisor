# Project Requirements Document — agentic-runtime-supervisor

**Status:** approved 2026-07-15 with two review resolutions (§6 rows 4–5; memory.md D16–D18). Implementation in progress.
**Authoritative spec:** the research exposé ("Agentic Runtime Supervision for Deployed Medical Vision Models Under Distribution Shift"). Where the build prompt and the exposé differ, the exposé wins; all detected differences are flagged in §6.

## 1. Goal, restated

Build a ~2-day proof-of-concept demonstrating a **runtime supervisor** that sits in front of a frozen medical image classifier (a small ViT on DermaMNIST) under synthetic distribution shift. Per stream batch, the supervisor reads three **label-free** signals (embedding drift vs a clean-calibration reference, conformal prediction set size, expected-benefit-minus-cost), selects one of four actions (**ACCEPT / TTA / ABSTAIN / ESCALATE**), observes **delayed, action-gated** outcome feedback, and updates a simple online estimate of each action's benefit. The PoC's purpose is to back the exposé's RQ1–RQ3 with one head-to-head experiment:

- **RQ1:** does the multi-signal, outcome-updating supervisor beat well-tuned static policies (static classifier, always-TTA, never-TTA, fixed-threshold monitor) on calibration improvement net of intervention cost? Supporting signal ablation included. A null/negative result is a legitimate, reportable finding.
- **RQ2:** does intervention history/memory add value over a memoryless contextual bandit seeing the same signals? The bandit comparator is **required**.
- **RQ3:** does adapt-then-recalibrate preserve empirical conformal coverage (95% target) under episodic TTA? Measured directly.
- **RQ4** (theory: conformal validity under a changing score function) is **out of PoC scope**; named as thesis work only.

## 2. In scope

- One dataset: DermaMNIST (medmnist, 28×28, 7 classes), single seed.
- One small **ViT** classifier trained on clean data (ViT required: LayerNorm-only TTA needs LayerNorm; a plain small CNN has none). Accuracy need not be high.
- Split-conformal calibration (LAC score, 95% target coverage) on a held-out clean calibration set.
- **Episodic** LayerNorm-only TTA (entropy minimization, γ/β only), reverting to frozen weights after each batch. Continual TTA = future work.
- **Adapt-then-recalibrate**: after TTA, the frozen labeled calibration set is rerun through the adapted model to refresh the conformal threshold. Flagged simplification, not a solution to exchangeability.
- Supervisor as a LangGraph state machine (`SignalCollection → RiskAssessment → PolicyDecision → Action → OutcomeLogging`), **no LLM calls anywhere** — nodes are plain Python.
- Policy = online outcome estimator (running frequency estimate of realized ECE reduction per action per drift bucket), **not** a weighted-signal threshold rule and **not** a learned/RL policy.
- Feedback model: **delayed + action-gated** — escalation returns ground truth; a fixed ~10% labeled monitoring subset of stream batches returns ground truth; all other batches yield no labels to the policy. (User-approved; see §6.)
- Synthetic shift: gaussian noise, gaussian blur, color jitter at graded severities, arranged in temporal blocks over a ~200-batch stream.
- Five comparators vs the supervisor: (1) static classifier no-intervention (raw, no conformal), (2) never-TTA (conformal-wrapped, always accept), (3) always-TTA, (4) fixed-threshold monitor (grid-tuned; tuning on the eval stream, disclosed as favoring the baseline), (5) memoryless contextual bandit. Static and never-TTA are **distinct** (user-approved; see §6).
- Signal-ablation run mode (drop each of the three signals in turn) + a recalibration-on/off contrast for RQ3.
- Outputs: JSONL decision trace per run; one summary table; generated RESULTS.md; README with ASCII architecture diagram and single-command reproduction.

## 3. Explicitly out of scope

Multiple datasets · learned/RL policy · online policy learning at scale · continual (non-episodic) TTA · per-sample selective prediction · production infra · auth · scaling · real clinical data · multi-seed sweeps · learned cost models · RQ4 theory · plots/dashboards.

## 4. Constraints

- ~2 days total effort; phases and checkpoints in `phases.md`.
- CPU-only target (user-confirmed); CUDA used opportunistically if present, never required.
- Python + uv (`pyproject.toml` only; no requirements.txt). Minimal deps: `torch`, `torchvision`, `medmnist`, `langgraph`, `numpy`.
- Single seed (0), disclosed as a variance limitation in RESULTS.md.
- Readable over clever; the WHY of the policy logic is commented in code (it will be defended to reviewers).

## 5. Success criteria — what "done" looks like

1. `uv run <single command>` reproduces the full pipeline: train (or load) ViT → calibrate → stream run for all six methods → ablations → RESULTS.md.
2. **The PoC produces exactly one head-to-head comparison table plus supporting ablation, nothing more.** Table rows: static, never-TTA, always-TTA, fixed-threshold monitor, contextual bandit, supervisor. Columns: accuracy (issued), ECE (issued, pooled), empirical coverage vs 95%, mean set size, #TTA, #escalations, abstention rate, total intervention cost, net calibration benefit (primary metric). Ablation table: three signal ablations (RQ1) + always-TTA with/without recalibration (RQ3) + escalation-bonus ON vs OFF (VoI finding, outside RQ2).
3. Decision trace (JSONL) logs, per batch: signals, bucket, per-action estimates, chosen action, reasoning string, realized outcome when labels arrived.
4. RESULTS.md states the single-seed limitation and every disclosed simplification (§6, and design.md §10). The oracle-label simplification note is **not** needed — the policy never sees oracle labels (only the evaluation harness does, which is standard).
5. Supervisor's policy is demonstrably the outcome-estimator described in design.md (estimates visible in the trace), not a disguised threshold rule.
6. A negative result (a tuned static baseline matching the supervisor) still counts as done — it is reported, not hidden.

## 6. Prompt-vs-exposé differences and resolutions

| # | Difference | Resolution | Status |
|---|---|---|---|
| 1 | Exposé: "only escalation returns ground truth." Prompt adds a small labeled monitoring subset. | Delayed-gated + ~10% monitoring subset; documented as a deliberate extension (without it the estimator can ~never observe TTA/accept outcomes at PoC scale). | **User-approved** |
| 2 | Exposé reads signals "on each input"; prompt specifies batch-level actions. | Batch-level. Drift statistics and entropy-minimization TTA are only well-defined over a batch. Per-sample selective prediction = future work. | **User-approved** |
| 3 | Exposé names both "static classifier, no intervention" and "never-TTA" as baselines. | Kept distinct: static = raw classifier, no conformal (coverage n/a); never-TTA = conformal-wrapped, always accept. | **User-approved** |
| 4 | Exposé: the policy "observes the outcome of **that** action" (on-policy only). PoC adds **counterfactual outcome replay**: on the rare labeled batches, the realized benefit of all four actions is computed (TTA's by running episodic adapt-then-revert offline on that batch) and all four (action, bucket) cells are updated. | Needed for a functioning estimator at 200-batch scale under gated feedback (on-policy-only would leave TTA's estimate near-empty). Uses only legitimately obtained labels. Applied identically to the bandit for fairness. Stated scope limit on the page (design.md §6): the PoC demonstrates the runtime-supervision architecture, not online sample-efficiency under gated feedback. | **User-approved** |
| 5 | Under the PoC's benefit definition, ABSTAIN and ESCALATE have identical raw benefit (avoided miscalibration), so the cheaper one always dominates and ESCALATE would never be chosen deliberately. PoC adds a small **escalation information bonus** that decays with the per-bucket labeled-sample count: escalation is how the policy buys ground truth when its estimates for the current shift regime are data-poor. | Simple (one term), principled (value of information). Ablatable config flag, **OFF by default**: the RQ2 headline is supervisor-with-bonus-OFF vs bandit (isolates memory representation); the bonus-ON run is reported separately as a VoI finding, outside RQ2. Details in design.md §5. | **User-approved as ablatable; OFF in RQ2 headline** |
| 6 | Exposé's non-stationarity argument for the agentic controller ("TTA mutates the supervised model") is muted in the PoC because episodic TTA reverts after every batch. | Disclosed honestly in design.md §8 and RESULTS.md: the PoC setting is *kinder to the bandit* than the thesis setting; this makes a supervisor win more meaningful and a null result expected/acceptable. | Disclosed |

## 7. Assumptions

- Evaluation harness uses full ground truth to *report* metrics for all methods; the gating constraint applies only to what the *policy* learns from. This is standard experimental practice, not the oracle-label simplification the prompt warns about.
- Fixed per-action costs are constants in ECE units (design.md §4); their ordering, not magnitude, carries the argument. Sensitivity analysis is out of scope and disclosed.
- Stream images are drawn with reuse from the DermaMNIST test split to reach ~200 batches; disclosed.
