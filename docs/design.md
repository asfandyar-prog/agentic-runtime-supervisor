# Design — the supervisor policy (graded core)

This document is the precise specification of the decision policy, its signals, its feedback model, and the comparators. Everything here is defended to reviewers; every constant is a named default in `config.py`.

## 1. What the policy is — and is not

The policy is **not** a weighted-signal threshold rule. It is an **online outcome estimator**, exactly as the exposé specifies:

> For each of the four actions, maintain a running frequency estimate of ECE reduction conditioned on a coarse bucket of current shift magnitude (embedding-drift quantile). Select the action maximizing estimated benefit (ECE reduction) minus a fixed per-action intervention cost.

Formally, per batch *t* with drift bucket *b*:

```
choose  a* = argmax over allowed a of  [ Q̂(a, b) − cost(a) + bonus(a, b) ]
```

where `Q̂(a, b)` is the running mean of *realized* ECE-reduction outcomes observed so far for action *a* in bucket *b*, `cost(a)` is a fixed constant, and `bonus` is nonzero only for ESCALATE (§5). With probability ε = 0.1 a uniformly random allowed action is taken instead (exploration; without it, gated feedback plus greedy selection would never populate the estimates). A learned policy (RL, learned cost model) is thesis-scale future work, deliberately not built.

## 2. The three decision-time signals (all label-free)

1. **Embedding drift** vs the clean-calibration reference. Reference = per-dimension mean μ_ref and std σ_ref of frozen-model embeddings (mean-pooled final-layer tokens) over the clean calibration set (n=500). Drift value = ‖(mean batch embedding − μ_ref)/σ_ref‖₂ / √d. Clean-batch drift distribution: bootstrap 200 size-64 batches from the calibration set → quantiles. **Bucketing:** b0 (in-distribution): drift ≤ q95_clean; b1 (moderate shift): q95 < drift ≤ 3·q95; b2 (severe shift): drift > 3·q95. Coarse, fixed a priori, logged. The 3·q95 edge is arbitrary-but-fixed (recorded in memory.md).
2. **Conformal set size** of the batch under the frozen model's q_hat. Enters as a **safety gate**: if mean set size > 1.5 × the clean-calibration mean set size, ACCEPT is disallowed for this batch (the policy must pick among TTA/ABSTAIN/ESCALATE). Rationale: inflated sets mean the calibration guarantee is already visibly strained; shipping as-if-normal is the one clearly indefensible act. The bucket itself stays drift-only, matching the exposé's wording ("a coarse bucket of current shift magnitude (embedding-drift quantile)") and keeping the estimator at 4×3 = 12 cells a 200-batch stream can actually populate.
3. **Expected-benefit-minus-cost** — the `Q̂(a,b) − cost(a)` quantities themselves, computed from memory. Label-free at decision time: they are functions of *past* (delayed, gated) outcomes only.

Decision-time inputs contain no labels anywhere. The corruption type/severity of a batch is logged for analysis but never exposed to any policy.

## 3. The four actions — exact semantics

| Action | Semantics | Predictions shipped? | Labels returned to policy? |
|---|---|---|---|
| ACCEPT | Trust the frozen model as-is | Yes (frozen preds + sets) | Only if monitoring batch |
| TTA | Episodic LayerNorm entropy minimization on this batch, adapt-then-recalibrate, then revert | Yes (adapted preds + recalibrated sets) | Only if monitoring batch |
| ABSTAIN | Withhold predictions for the whole batch (batch-level; per-sample selective prediction = future work) | No | Only if monitoring batch |
| ESCALATE | Withhold predictions, flag batch for human review, incur high fixed cost | No | **Yes, always** (the human resolves the batch) |

Condition for each being optimal (given the benefit definitions in §4): ACCEPT when estimated frozen-model miscalibration in this bucket is below every intervention's net gain (typ. bucket b0). TTA when adaptation is estimated to recover most of the miscalibration at small cost (typ. b1). ABSTAIN when miscalibration is high and TTA doesn't recover it (typ. b2). ESCALATE when, additionally, the policy's estimates for this bucket are data-poor and buying ground truth is worth the premium (§5).

## 4. Realized benefit, costs, and the objective

"Expected benefit" is **calibration improvement, explicitly not accuracy** (exposé). For a labeled batch, with `ECE_frozen` = ECE of the frozen model's predictions on that batch:

| Action | Realized benefit |
|---|---|
| ACCEPT | 0 by definition (it *is* the do-nothing reference) |
| TTA | `ECE_frozen − ECE_adapted` (can be negative: TTA can make a well-behaved model worse — that is the point of supervising it; same-batch difference, so small-sample ECE bias largely cancels) |
| ABSTAIN | `ECE_frozen − floor` (miscalibration avoided by not shipping, **debiased**: `floor` = clean batch-ECE noise floor measured a priori on the calibration set — see §10.8/D24) |
| ESCALATE | `ECE_frozen − floor` (same avoidance; the human handles the batch) |

Fixed costs (ECE units): `ACCEPT 0.00, TTA 0.02, ABSTAIN 0.06, ESCALATE 0.15`. Only the *ordering* (free < cheap compute < lost throughput < human time) carries the argument; magnitudes are disclosed knobs, sensitivity analysis out of scope.

**Primary comparison metric** (same formula for every method, computed by the harness with full labels):

```
net_calibration_benefit = mean over batches of realized_benefit(action_t) − mean over batches of cost(action_t)
```

Note the metric fully credits avoided ECE, which structurally favors abstention on heavily corrupted batches; the ~40%-clean stream mix and the abstention cost keep always-abstain from being trivially optimal, and abstention rate + issued-accuracy columns in the table expose any degenerate policy. Disclosed in RESULTS.md. Empirical coverage vs the 95% target is reported alongside (RQ3) rather than folded into the scalar.

## 5. Escalation information bonus (approved as ablatable; OFF in the RQ2 headline)

Under §4, ABSTAIN and ESCALATE have identical raw benefit, so the cheaper ABSTAIN always dominates and ESCALATE would only ever fire through ε-exploration. But escalation has a real second return: it is the only action that *always* buys ground truth. The PoC makes that explicit with a value-of-information term:

```
bonus(ESCALATE, b) = β / sqrt(n_labeled(b) + 1),    β = 0.05
```

where `n_labeled(b)` counts labeled outcomes observed in bucket *b*. Early, or when a new shift regime opens an unexplored bucket, escalation is worth its premium as label acquisition; once the bucket's estimates are populated the bonus fades and ABSTAIN dominates again.

**Review-resolved status (PRD §6.5):** the bonus is a config-flagged, ablatable component (`escalation_bonus: bool`, **default OFF**). Its role is to make ESCALATE reachable as a deliberate choice; it sits explicitly **outside** the memory-vs-memoryless comparison. The RQ2 headline is supervisor-with-bonus-OFF vs bandit; the bonus-ON run is reported separately as its own finding (§9d), never as the RQ2 result. Running an RQ2 headline with the bonus ON would be structurally unfair — the bandit is denied history-defined terms by construction, so a supervisor win would be built in rather than tested; if that comparison were ever wanted, the bandit would need a matching count-based exploration term (not built; the bonus-off headline makes it unnecessary).

## 6. Feedback model: delayed, action-gated, with counterfactual replay

**Gating (user-approved):** at decision time the policy sees only the label-free signals of §2. Labels reach the policy only from (a) escalated batches — always, and (b) monitoring batches — every 10th batch (~10% of the stream), fixed schedule. Accepted/abstained/TTA'd batches outside the monitoring schedule return nothing. Labels arrive *after* the action executes; memory updates take effect from the next batch (delay = one step; longer delays add nothing at PoC scale).

**Counterfactual replay (approved, PRD §6.4):** when a batch's labels do arrive, the harness computes the realized benefit of *all four* actions on that batch — ACCEPT is 0 by definition, ABSTAIN/ESCALATE equal `ECE_frozen`, and TTA's is obtained by running the episodic adapt-then-revert procedure offline on that batch — and updates all four `(action, bucket=b)` cells. Why: strict on-policy updates under gated feedback would give ~2–5 TTA outcomes across the whole stream — a nonfunctional estimator. Replay uses only legitimately obtained labels, is cheap (TTA is episodic), and is granted **identically to the bandit**, so RQ2 stays a fair fight about representation and history, not data access. **Stated limit:** counterfactual replay makes this PoC a demonstration of the runtime-supervision *architecture*, not a test of online sample-efficiency under gated feedback; the latter (continual TTA, true delayed on-policy learning) is thesis scope.

**Update rule** (plain running mean — the exposé's "simple frequency estimate"):

```
n(a,b) += 1;   Q̂(a,b) += (benefit − Q̂(a,b)) / n(a,b)
```

Unobserved cells start at Q̂ = 0, n = 0 (i.e., "no better than doing nothing"), which is conservative for interventions since they still pay their cost until proven useful; ε-exploration plus replay populates them.

## 7. Memory: storage and update across batches

Memory is a plain dict serialized inside the LangGraph state and mirrored to the JSONL trace every batch (so any decision can be audited post hoc):

- `estimates[(action, bucket)] = {mean_benefit, n}` — the core outcome memory (12 cells).
- `labeled_count_per_bucket[b]` — drives the escalation bonus.
- `action_counts`, `last = {action, benefit}` — bookkeeping surfaced in the reasoning string.

Update path: OutcomeLogging node, only when labels arrived, per §6. Nothing else mutates memory. No decay/EMA: the frozen model + episodic TTA make the outcome mapping approximately stationary within a run, so a plain mean matches both the exposé's wording and the setting (recorded in memory.md).

## 8. RQ2 comparator: the memoryless contextual bandit (required)

- ε-greedy (same ε = 0.1) with a per-action **linear** reward model over the raw continuous signals `x = [1, drift_value, mean_set_size]`, fit online by incremental ridge regression. Reward = the same realized `benefit − cost`, same gated labels, same counterfactual replay.
- **Memoryless** operationalized as: the decision depends only on the current batch's signal vector through a fixed parametric form assuming a stationary reward surface — no drift buckets, no per-bucket label counts, no information bonus, no history features.

**Honest framing, stated up front:** the supervisor and the bandit are deliberately close cousins — both are ε-greedy online estimators of benefit-minus-cost over the same signals and the same gated data. The **RQ2 headline comparison runs the supervisor with the escalation bonus OFF**, so the implemented difference is exactly one thing: **bucketed outcome memory vs a stationary linear reward surface**. The escalation bonus never enters the headline — it is a history-defined term the bandit is structurally denied, so including it would assume RQ2's answer rather than test it; the bonus-ON run is reported separately (§9d). Moreover, episodic TTA reverts the model each batch, so the exposé's strongest pro-agentic argument (TTA-induced non-stationarity of the reward mapping) is largely absent at PoC scale — the PoC is *kinder to the bandit* than the thesis setting would be. Consequence: a supervisor win here is meaningful evidence; a null result is expected to be possible, legitimate, and will be reported as such (exposé: "the justification for the agentic controller… collapses" is an acceptable answer).

## 9. Static baselines and ablations

- **Static, no intervention:** raw frozen classifier, no conformal machinery. Accuracy/ECE only; coverage n/a; zero cost.
- **Never-TTA:** conformal-wrapped frozen classifier, always ACCEPT. Isolates the value of conformal wrapping.
- **Always-TTA:** TTA + adapt-then-recalibrate on every batch. Also run once **without** recalibration → the RQ3 coverage contrast.
- **Fixed-threshold monitor:** drift thresholds (τ1 ≤ τ2): ACCEPT below τ1; between τ1/τ2 and above τ2, one intervention each, with the grid choosing those two actions from {TTA, ABSTAIN, ESCALATE} as well as the thresholds (∞ thresholds keep the exposé's single-threshold monitor inside the search space). Grid-searched on the **evaluation stream itself**, best net-benefit rule reported — an oracle-tuned upper bound for static policies. Without the action choice the monitor is denied the action that may win and stops being an upper bound (memory.md D23). Disclosed: this deliberately favors the baseline, making RQ1 *harder* for the supervisor.
- **Signal ablations (RQ1),** supervisor variants: (a) **no-drift** — single bucket, estimator unconditioned on shift; (b) **no-set-size** — ACCEPT gate removed; (c) **no-benefit/cost** — memory off; fixed mapping b0→ACCEPT, b1→TTA, b2→ESCALATE (+gate), i.e., the policy collapses to a two-signal threshold monitor, which is precisely the exposé's point about what removing the outcome loop leaves behind. RQ1's ablation claim is tested against the *strongest* single-signal variant.
- **(d) Escalation-bonus ON (additional finding, not RQ2):** the headline supervisor re-run with the VoI bonus enabled (design.md §5). Reported alongside — never inside — the RQ2 comparison, as "VoI-driven escalation changes X."

## 10. Disclosed weaknesses (carried into RESULTS.md)

1. Single seed; batch-level ECE on 64 samples is noisy — the policy learns from a noisy but honest outcome signal; table-level ECE is pooled over all issued predictions.
2. Costs and stream mix jointly locate the arbitration boundaries; constants are disclosed, not tuned via sweeps.
3. Adapt-then-recalibrate does not restore exchangeability (clean calibration set vs shifted batch); it is a flagged simplification measured by RQ3, not a fix.
4. Block-structured (temporally correlated) drift is realistic but favors memoryful policies; disclosed.
5. Counterfactual replay (§6) and escalation bonus (§5) are PoC additions beyond the exposé's literal policy — both review-approved. The bonus is OFF in the headline and reported as its own ablation (§9d); replay's scope limit is stated in §6.
6. Stream reuses test images across batches to reach ~200 batches.
7. The deployed model is the frozen ViT **plus temperature scaling fitted on the calibration set** (standard deployment practice; restores the exposé's well-calibrated-deployment premise — without it the tiny ViT's clean ECE (~0.10) exceeds the abstention cost and every policy degenerates to always-abstain; memory.md D22).
8. Per-batch ECE on 64 samples has an upward small-sample bias (measured clean floor ≈0.11 even for the temperature-scaled model whose pooled clean ECE is ≈0.02); left uncorrected it pays withholding for estimator bias rather than avoided miscalibration, and every policy degenerates to always-abstain. The withhold benefit therefore subtracts the floor (measured a priori on the calibration set — §4). The floor is an approximation under shift (bias depends on the confidence distribution); disclosed, with an always-abstain reference number in RESULTS.md. Post-approval deviation, flagged: memory.md D24.
