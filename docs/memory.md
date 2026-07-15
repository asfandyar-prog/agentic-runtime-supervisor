# Memory — running log of decisions, alternatives, deviations

Newest entries at the bottom. Every entry: the decision, the rejected alternative(s), and the reasoning — this log backs the deep-dive defense. Updated at decision time, not retrofitted.

---

## 2026-07-15 — Planning (pre-code)

### D1. Feedback model: delayed-gated + ~10% labeled monitoring subset — **user-approved**
Exposé says only escalation returns ground truth; the build prompt added a monitoring subset. Flagged to the author; they chose the monitoring-subset version over (a) strict escalation-only gating and (b) oracle labels on the full stream. Reasoning: strict gating gives the estimator ~zero TTA/accept outcomes at 200-batch scale (degenerate policy); oracle labels erase the bandit's structural disadvantage and gut the RQ2 claim. The monitoring subset is a realistic deployment practice (periodic labeled audits) and is disclosed as an extension of the exposé's gating.

### D2. Batch-level decisions, not per-input — **user-approved**
Exposé reads signals "on each input"; prompt specifies batch-level. Chose batch-level: embedding-drift statistics and entropy-minimization TTA are only well-defined over a batch; single-image LN adaptation is noisy/ill-posed and would blow the 2-day budget. Per-sample selective prediction recorded as future work.

### D3. Static ≠ never-TTA — **user-approved**
Kept the exposé's four named static baselines distinct: static = raw classifier without conformal wrapping (deployment status quo, coverage n/a); never-TTA = conformal-wrapped, always accept. The delta between those two rows isolates the value of conformal wrapping alone. Rejected alternative: merging them into one column (would drop an exposé-named baseline).

### D4. CPU-only, tiny ViT — **user-approved**
Patch-4/dim-96/depth-4 ViT (~0.5M params) on 28×28 DermaMNIST, ~15 epochs. ViT is mandatory (LayerNorm-only TTA needs LN; a small CNN has none). Accuracy is a non-goal beyond "well above chance and calibratable."

### D5. Bucket = drift quantile only; set size as an ACCEPT gate — my call
Exposé: estimates are "conditioned on a coarse bucket of current shift magnitude (embedding-drift quantile)". Rejected alternative: 2-D buckets (drift × set-size band) — doubles cells to 24, starving the estimator under gated feedback at 200 batches. The gate keeps set size a real decision-time signal (ablatable for RQ1) at zero sample-efficiency cost. Bucket edges q95_clean and 3·q95_clean: the 3× is arbitrary-but-fixed a priori; disclosed rather than tuned.

### D6. ⚠ Counterfactual outcome replay on labeled batches — my call, **flagged, awaiting review** (PRD §6.4)
Exposé implies on-policy updates ("observes the outcome of that action"). Under gated feedback, on-policy-only yields ~2–5 TTA outcomes per run → nonfunctional estimator. On labeled batches (escalation/monitoring) the harness computes all four actions' realized benefits (TTA's via offline episodic adapt-then-revert) and updates all four cells of that bucket. Uses only legitimately obtained labels; granted identically to the bandit so RQ2 compares representations, not data access. Rejected alternatives: bigger monitoring fraction (20% still too thin, and dilutes the gating claim); fewer buckets (loses the shift-conditioning that IS the policy).

### D7. ⚠ Escalation information bonus β/√(n_labeled(bucket)+1), β=0.05 — my call, **flagged, awaiting review** (PRD §6.5)
Found during design: ABSTAIN and ESCALATE share the benefit definition (avoided ECE), so cheaper ABSTAIN strictly dominates and ESCALATE would fire only via ε-noise. But escalation's real second return is ground truth. A count-decaying value-of-information bonus makes "escalate to learn a new regime" expressible. *(Original claim that this "sharpens the RQ2 contrast" was reversed at review — see D17: a history-defined term the bandit is structurally denied would build the RQ2 answer in by construction.)* Rejected alternatives: a fixed resolution bonus for escalation (any constant makes one of abstain/escalate dominate the other *always* — checked algebraically); pricing information into the benefit definition (conflates calibration units with information units).

### D8. Benefit definitions (design.md §4) — my call
ACCEPT ≡ 0 (do-nothing reference); TTA = ECE_frozen − ECE_adapted (signable — TTA harming a fine model must be learnable); ABSTAIN/ESCALATE = ECE_frozen (avoided miscalibration). Known consequence: the net-benefit metric credits avoidance, so always-abstain could look good on a mostly-corrupted stream → mitigated by ~40% clean blocks + abstention cost 0.06, and exposed by abstention-rate/accuracy columns. Disclosed, not hidden.

### D9. Costs {0, 0.02, 0.06, 0.15} in ECE units — my call
Ordering (free < compute < throughput loss < human time) is the defensible content; magnitudes are disclosed knobs. No sensitivity sweep (out of scope per prompt). Cost asymmetry mirrors the exposé's medical framing.

### D10. Plain running mean, no EMA/decay — my call
Exposé says "simple frequency estimate." Episodic TTA + frozen weights make the outcome mapping ~stationary within a run, so decay would be unmotivated machinery. Honest consequence, disclosed in design.md §8: episodic reversion also mutes the exposé's non-stationarity argument *for* the agentic controller — the PoC setting is kinder to the bandit than the thesis setting; null result acceptable.

### D11. Bandit spec: ε-greedy incremental ridge over [1, drift, set_size] — my call
Strong enough to be a fair RQ2 comparator (learns online from the same gated + replayed rewards), simple enough for the budget. "Memoryless" = decision is a fixed parametric function of current signals only; no buckets, no label counts, no bonus. Rejected: LinUCB (confidence machinery duplicates the escalation bonus's role and muddies the comparison), and a no-learning bandit (would strawman RQ2).

### D12. Fixed-threshold monitor tuned by grid search on the evaluation stream — my call
Deliberately oracle-tuned so the static baseline is an upper bound ("best-tuned static intervention policy," exposé RQ1). Makes RQ1 *harder* for the supervisor; disclosed. Rejected: tuning on a separate stream (weaker baseline, weaker claim).

### D13. Block-structured stream (10-batch regime blocks, ~40% clean) — my call
Real drift is temporally correlated; also gives memoryful policies something to exploit. Both faces disclosed (design.md §10.4). Test images reused across batches to reach 200 batches from a 2,005-image pool; disclosed.

### D14. Evaluation labels ≠ policy labels
Harness reports metrics with full ground truth for all methods; the gating constraint binds only the policy's learning signal. Standard practice; the prompt's "oracle simplification" warning therefore does not apply to reporting — decisions never see oracle labels.

### D15. No LLM in the graph
LangGraph is the state-machine substrate (matches exposé's reuse claim); nodes are plain Python. The exposé itself notes the argument is for a stateful outcome-updating decision process, "not for a large-language-model agent specifically."

---

## 2026-07-15 — Doc review outcomes (author)

### D16. Counterfactual replay approved; scope limit stated on the page
Approved as designed. Added to design.md §6 at the author's direction: replay makes the PoC a demonstration of the runtime-supervision *architecture*, not a test of online sample-efficiency under gated feedback (continual TTA, true delayed on-policy learning = thesis scope). The limit is on the page, not just understood.

### D17. Escalation bonus: ablatable, OFF in the RQ2 headline — reverses D7's framing
Author's correction, accepted: RQ2 asks whether bucketed outcome memory beats a memoryless linear surface. A history-defined bonus the bandit is structurally denied would make a supervisor win true *by construction* — assuming RQ2, not testing it. Resolution: config flag `escalation_bonus`, default OFF; headline = bonus-off supervisor vs bandit; bonus-ON reported as a separate VoI finding (design.md §9d). This also partially reverses D11's LinUCB rejection: a bonus-ON headline would require a matching count-based exploration term for the bandit; the bonus-off headline (path taken) makes that unnecessary.

### D18. Label-path separation is a defended code invariant
The harness is both metric reporter (full labels) and supplier of the policy's replayed learning signal (gated only). The author defends this separation on the review call → treated as a code invariant: policy/bandit updates consume a feedback object constructible *only* from gated sources (escalation/monitoring batches); reporting metrics never flow into memory. rules.md updated; Phase 4 checkpoint verifies it explicitly.

---

*(Implementation entries follow here as phases execute.)*
