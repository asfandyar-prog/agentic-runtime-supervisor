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

## 2026-07-15 — Implementation

### D19. Per-batch precompute cache; supervisor still runs TTA live
Episodic TTA is deterministic given (frozen weights, batch): entropy minimization has no random sampling. So each batch's adapted outcome (probs, recalibrated q_hat, ECE) is computed ONCE in `runner.precompute` and reused for baselines, monitor grid tuning, and counterfactual replay — otherwise always-TTA, the monitor grid, and replay would each rerun identical adaptations. The supervisor's LangGraph Action node still runs TTA live and `assert`s equality with the cache, which doubles as the determinism proof that justifies both the cache and replay. Episodic reversion is asserted bit-identical on every precompute batch (Phase 3 checkpoint). Rejected alternative: cache-only supervisor (would demote the graph to a lookup and weaken the "deployed artifact" demonstration).

### D20. Label-path separation, as built (D18 discharge)
`runner.gated_feedback` is the only constructor of `PolicyFeedback` (which asserts source ∈ {escalation, monitoring}); ground-truth labels never enter the LangGraph state (`SupState` has no `y` field); the graph obtains feedback only via the injected gated provider; evaluation metrics live in `run_method` and never touch memory. Supervisor runs pass `learner=None` because feedback is consumed inside the OutcomeLogging node — no second path.

### D21. Monitor grid = drift-threshold pairs incl. ∞
(τ_tta, τ_esc) over 19 stream-drift quantiles plus ∞ (∞ lets the grid collapse to one-threshold or never-escalate variants, so the exposé's "single threshold" monitor is inside the search space). Tuned on the eval stream per D12, maximizing the same net-benefit metric the table reports.

### D22. Temperature scaling added to the deployed model — deviation, flagged to author
First full run exposed a degenerate regime: the tiny ViT's clean per-batch ECE (~0.10) exceeds the abstention cost (0.06), so abstaining is net-positive *everywhere* and every learning policy (bandit and supervisor alike) correctly collapses to near-always-abstain — the arbitration the PoC exists to study never engages. Root causes: (a) the tiny ViT is genuinely miscalibrated on clean data (pooled clean ECE ~0.09), violating the exposé's premise of a well-calibrated deployed model (thesis: clean ECE 0.014); (b) per-batch ECE on 64 samples has an upward small-sample bias that inflates ABSTAIN/ESCALATE's measured benefit. Fix for (a): temperature scaling fitted by NLL on the existing clean calibration set — standard deployment practice, no new data, no new labels. (b) is measured (bootstrap clean floor) and disclosed in RESULTS.md with an always-abstain reference number, not corrected — debiasing batch ECE is beyond PoC scope, and correcting the benefit definition post-hoc was rejected as adjacent to metric shopping (the author locked the benefit definition at review). **Flagged for author review since it post-dates doc approval.** *Approved 2026-07-15 (D27).*

### D23. Monitor grid extended to choose its intervention actions — deviation, flagged to author
Same first run showed the approved accept/TTA/escalate monitor is NOT the "oracle-tuned upper bound for static policies" the design promised: it was structurally denied ABSTAIN, the action that wins on this stream, and scored near zero while the learned policies scored ~+0.05. The approved *property* (upper bound) contradicts the approved *mechanism* (fixed action mapping); the property is the point of D12, so the grid now also selects the mid/high-region actions from {TTA, ABSTAIN, ESCALATE}. This makes RQ1 strictly harder for the supervisor. **Flagged for author review since it post-dates doc approval.** *Approved 2026-07-15 (D27).*

### D24. Withhold benefit debiased by the measured clean batch-ECE floor — deviation, flagged to author
Post-temperature-scaling run showed the pooled clean ECE is 0.023 but the **per-batch** ECE floor is ≈0.11 at batch size 64 — pure small-sample estimator bias. Since benefit(ABSTAIN/ESCALATE) = batch ECE, every policy was being paid ≈ floor − cost ≈ +0.05 for withholding *perfectly calibrated* batches, and all learning policies (bandit and supervisor alike) correctly collapsed to near-always-abstain — the arbitration the PoC studies never engages, and the winner is decided by an artifact. Fix: benefit(withhold) = ECE_frozen − floor, where floor is bootstrap-measured **a priori on the clean calibration set** (no stream peeking; signable, so the estimate stays unbiased rather than re-biased by clamping). TTA's benefit is a same-batch difference where the bias largely cancels — untouched. Rejected alternatives: raising the abstain cost (conflates cost semantics with measurement error, and is hand-tuned where the floor is measured); fewer/adaptive ECE bins (shrinks but does not remove the floor); reporting the artifact-dominated table as the result (the metric would be measuring its own estimator bias, not the RQ). **Changes the approved design.md §4 benefit definition — flagged for author sign-off.** *Approved 2026-07-15 (D27).*

### D25. Corruption severities retuned on a 500-image probe — deviation, flagged to author
Original severities barely damaged calibration (corrupted pooled ECE ≈ clean), so no intervention could ever be worth its cost — the stream did not contain the exposé's failure mode. Retuned once against a probe (never against method rankings): noise σ→[0.15, 0.25, 0.40] (sev2 halves accuracy: 0.46), jitter strengthened (sev2 ECE 0.16); blur kept benign **deliberately** — it moves embeddings without hurting calibration, giving drift-threshold rules a false-positive regime that outcome-conditioned policies can learn to ignore. Note the tiny ViT under shift fails with *low* confidence (entropy rises), so corrupted ECE reaches ~0.13–0.16, not the thesis's 0.889 "confidently wrong" regime — the PoC's shift is milder than the exposé's motivating case; disclosed. *Approved 2026-07-15 (D27).*

### D26. Live-vs-cache asserts relaxed to float tolerance
Empirical finding: after autograd has run once in the process, CPU (oneDNN) kernel-path selection changes and identical forwards differ by ~1e-6 (parameters verified bit-identical; repeated adaptation in the same context bit-identical). The graph's live-TTA and live-signal asserts now use atol 1e-4/1e-5 instead of exact equality; the cache-equivalence claim (D19) holds at float tolerance, which is immaterial to ECE-scale outcomes. *Approved 2026-07-15 (D27).*

### Phase checkpoints (seed 0, final run)
- **Phase 0** ✓ deps resolve; `import ars` ok (torch 2.13 CPU).
- **Phase 1** ✓ tiny ViT (~0.5M params), 15 epochs CPU ≈ 2 min; clean test acc **0.724** (target ≥0.65; majority class 0.669; 7-class chance 0.143). Fix en route: medmnist needs `root` dir created before download.
- **Phase 2** ✓ T=1.072; q_hat=0.9393; **clean coverage 0.9551** vs 0.95 target; pooled clean test ECE 0.0226; clean batch-ECE floor 0.1096 (the D24 measurement); drift q95_clean=0.1716, bucket edges [0.17, 0.51].
- **Phase 3** ✓ episodic revert asserted bit-identical on all 200 precompute batches; drift separates clean 0.133 vs corrupted 0.634; corrupted mean batch ECE 0.134 vs clean 0.101. Finding: **TTA worsens ECE on average** (0.134→0.143) — entropy minimization sharpens confidence; under the exposé's calibration-only benefit this makes TTA rarely worth its cost, and the supervisor must learn *restraint*.
- **Phase 4** ✓ trace invariants verified from the JSONL alone (scratch checker): memory frozen on all 176 unlabeled batches (D18/D20 label gating); every feedback row updates exactly 4 cells × 1 bucket (replay); escalation ⇔ ESCALATE; monitoring only on monitored rows; 5 escalation + 19 monitoring labeled batches.
- **Phase 5** ✓ all 11 runs from one command. Headline nets: static/never-TTA/monitor 0.0000 (the oracle-tuned monitor **collapsed to "never intervene"** — on this stream no intervention clears its cost bar), always-TTA −0.0263, bandit −0.0064, supervisor −0.0074, no-benefit ablation −0.0601. **RQ1: negative** (supervisor does not beat the oracle static, structurally impossible when optimal = do-nothing; but removing the outcome loop costs −0.053 — the ablation carries the positive finding). **RQ2: null** (supervisor ≈ bandit, Δ=0.001 within single-seed noise). **RQ3**: recal 89.2% vs norecal 89.0% coverage (corrupted-only 86.8% vs 86.5%) — both under the 95% target; the gap is driven by the shift itself, and 3-step LN-only TTA barely moves the score function, so recalibration has little to correct. VoI bonus changed nothing (β=0.05 never exceeds the escalate−abstain cost gap; kept as the a-priori constant per no-tuning rule, reported as-is).
- **Phase 6** ✓ RESULTS.md generated by report.py; README with ASCII diagram + single command. Negative/null headline results reported per the exposé's explicit design ("a null or negative result is an acceptable answer"); no constants were retuned after method rankings were seen (rules.md).

---

## 2026-07-15 — Post-audit review outcomes (author)

### D27. D22–D26 all accepted; Run C committed as the accepted result
Author verdict after the three-run freeze audit (A original / B crashed-at-assert / C current): all five deviations approved — "none shopped a result; collectively they removed the small-sample-ECE-floor artifact that was inflating Run A, which is the honest direction." Audit findings that drove the verdict: TTA worsens ECE under shift in all three runs; the RQ2 supervisor-vs-bandit tie is stable (|Δnet| < 0.005 in A and C; B unmeasured); Run A's apparent RQ1 win (+0.0496 vs +0.0028) was attributed by evidence to the floor bias (D24) plus the ABSTAIN-less monitor (D23) — Run B's tuned-monitor line (+0.0543 with ABSTAIN available, pre-debias) proves D23 alone would have overturned it. Actions taken: (1) frozen tree committed as Run C (350ea42); (2) per-experiment archiving enabled — reproduce.py copies runs/ (JSONL traces + summary.json) to committed `results/seed<seed>/`; (3) RESULTS.md now headlines the two stable findings, carries the RQ1-flip attribution framed as a credibility-strengthening correction, and discloses the milder-shift regime (low-confidence failure, corrupted ECE ≈0.13–0.16, not the thesis's confidently-wrong 0.889). No numbers, costs, severities, or model changed. Open anomaly on record: Run A's supervisor logged 0 TTA picks (p ≈ 0.1–0.4% under ε=10%); A's traces were overwritten before archiving existed, so it is unauditable — the archiving rule exists precisely so this cannot recur. **Multi-seed RQ2 spread (3–5 seeds) is under author consideration — explicitly NOT started, awaiting go.**

### D28. Multi-seed RQ2 robustness run (author-ordered scope)
Go received with exact scope: 5 seeds (0–4) varying ONLY stream realization (corruption schedule, batch draws, monitoring subset) and policy RNGs (ε-draws); trained ViT, temperature, calibration split, q_hat, drift reference, and ECE floor held fixed at base config — isolating "given the same classifier, does bucketed memory beat the memoryless bandit across stream realizations." Supervisor + bandit only; no cost/severity/model/benefit changes (only code addition: `ars/multiseed.py` + a `runs_dir` passthrough on `run_supervisor`). Safeguards: seed 0 byte-compared against the committed Run C archive (identical — determinism re-confirmed at the artifact level, never overwritten); trace invariants (label gating, 4-cell replay) re-verified on every seed's supervisor trace. **Result: the tie holds.** Supervisor −0.0091 ± 0.0020 vs bandit −0.0077 ± 0.0031 (mean ± sample std, n=5); per-seed Δ(sup−bandit): −0.0011, +0.0025, −0.0026, −0.0029, −0.0025 — sign flips (1/5 favor supervisor), |Δ| ≤ 0.003 everywhere, mean Δ −0.0013 ± 0.0022 within one std of zero. Both policies are consistently slightly negative on every seed — exploration overhead in streams where doing nothing is optimal. RESULTS.md gained a generated "RQ2 robustness (5 seeds)" subsection sourced from committed `results/multiseed_rq2.json`.

*(end of build log)*
