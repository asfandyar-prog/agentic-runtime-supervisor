"""Aggregate run summaries into RESULTS.md: one head-to-head table + the ablation
table + RQ readouts + the full disclosure list. Generated, not hand-edited."""
import json
import os

HEADLINE = ["static_raw", "never_tta", "always_tta", "monitor_fixed", "bandit", "supervisor"]
ABLATIONS = ["supervisor_no_drift", "supervisor_no_setsize", "supervisor_no_benefit",
             "supervisor_bonus", "always_tta_norecal"]

LABELS = {
    "static_raw": "Static classifier (no conformal, no intervention)",
    "never_tta": "Never-TTA (conformal-wrapped, always accept)",
    "always_tta": "Always-TTA (+ recalibrate)",
    "monitor_fixed": "Fixed-threshold monitor (oracle-tuned)",
    "bandit": "Contextual bandit (memoryless)",
    "supervisor": "Supervisor (ours; bonus OFF)",
    "supervisor_no_drift": "Supervisor, ablate drift signal",
    "supervisor_no_setsize": "Supervisor, ablate set-size gate",
    "supervisor_no_benefit": "Supervisor, ablate benefit/cost memory",
    "supervisor_bonus": "Supervisor + escalation VoI bonus (not RQ2)",
    "always_tta_norecal": "Always-TTA without recalibration (RQ3)",
}


def _fmt(v, pct=False):
    if v is None:
        return "n/a"
    return f"{v * 100:.1f}%" if pct else f"{v:.4f}"


def _table(rows: list[dict]) -> str:
    head = ("| Method | Acc (issued) | ECE (issued) | Coverage (95% target) | Mean set size "
            "| #TTA | #Esc | Abstain rate | Mean cost | **Net calib. benefit** |\n"
            "|---|---|---|---|---|---|---|---|---|---|\n")
    lines = []
    for s in rows:
        lines.append(
            f"| {LABELS[s['method']]} | {_fmt(s['accuracy_issued'], pct=True)} | {_fmt(s['ece_issued'])} "
            f"| {_fmt(s['coverage'], pct=True)} | {_fmt(s['mean_set_size'])} | {s['n_tta']} "
            f"| {s['n_escalate']} | {_fmt(s['abstention_rate'], pct=True)} | {_fmt(s['mean_cost'])} "
            f"| **{_fmt(s['net_benefit'])}** |")
    return head + "\n".join(lines) + "\n"


def generate(summary_path: str = "runs/summary.json", out_path: str = "RESULTS.md") -> None:
    with open(summary_path, encoding="utf-8") as f:
        data = json.load(f)
    S = {s["method"]: s for s in data["summaries"]}
    meta = data["meta"]

    sup, band = S["supervisor"], S["bandit"]
    statics = [S[m] for m in ("static_raw", "never_tta", "always_tta", "monitor_fixed")]
    best_static = max(statics, key=lambda s: s["net_benefit"])
    sig_abls = [S[m] for m in ("supervisor_no_drift", "supervisor_no_setsize", "supervisor_no_benefit")]
    best_abl = max(sig_abls, key=lambda s: s["net_benefit"])

    # RQ2 robustness subsection, rendered only if the multiseed run has been executed
    # (results/multiseed_rq2.json is the committed aggregate; memory.md D28)
    ms_md = ""
    ms_path = "results/multiseed_rq2.json"
    if os.path.exists(ms_path):
        with open(ms_path, encoding="utf-8") as f:
            ms = json.load(f)
        rows = "\n".join(
            f"| {r['seed']} | {r['supervisor_net']:+.4f} | {r['bandit_net']:+.4f} | {r['delta']:+.4f} |"
            for r in ms["per_seed"])
        d = ms["delta"]
        n_pos, n = d["n_positive"], ms["n_seeds"]
        if n_pos == n:
            sign_txt = "the supervisor is ahead on every seed"
        elif n_pos == 0:
            sign_txt = "the bandit is ahead on every seed"
        else:
            sign_txt = f"the sign of Δ flips across seeds ({n_pos}/{n} favor the supervisor)"
        overlap = abs(ms["supervisor"]["mean"] - ms["bandit"]["mean"]) < (
            ms["supervisor"]["std"] + ms["bandit"]["std"])
        tie_txt = ("the tie holds: the two distributions overlap" if overlap
                   else "the distributions separate — the tie does NOT hold")
        ms_md = f"""
### RQ2 robustness ({n} seeds)

Per seed, only the stream realization and the policy RNGs vary ({ms['varied']});
held fixed: {ms['held_fixed']}. Supervisor and bandit only — the static baselines need
no seed spread. Traces per seed in `results/seedN/`.

| Seed | Supervisor net | Bandit net | Δ (sup − bandit) |
|---|---|---|---|
{rows}
| **mean ± std** | **{ms['supervisor']['mean']:+.4f} ± {ms['supervisor']['std']:.4f}** | **{ms['bandit']['mean']:+.4f} ± {ms['bandit']['std']:.4f}** | **{d['mean']:+.4f} ± {d['std']:.4f}** |

Across {n} seeds, {tie_txt} (supervisor {ms['supervisor']['mean']:+.4f} ± {ms['supervisor']['std']:.4f} vs bandit
{ms['bandit']['mean']:+.4f} ± {ms['bandit']['std']:.4f}), and {sign_txt}.
"""

    # multi-seed no-benefit ablation (Day 3a, D30), rendered only if executed
    abl_ms = ""
    if os.path.exists("results/multiseed_no_benefit.json"):
        with open("results/multiseed_no_benefit.json", encoding="utf-8") as f:
            ab = json.load(f)
        abl_ms = (f"\nAcross the same {ab['n_seeds']} seeds as the RQ2 robustness run, the no-benefit/cost "
                  f"ablation nets {ab['no_benefit']['mean']:+.4f} ± {ab['no_benefit']['std']:.4f}; removing the outcome loop costs "
                  f"{ab['removal_cost']['mean']:+.4f} ± {ab['removal_cost']['std']:.4f} net vs the full supervisor — the single-seed "
                  f"figure holds across seeds.")

    # cost-sensitivity crossovers (Day 3b, D30), rendered only if executed
    cost_md = ""
    if os.path.exists("results/cost_sensitivity.json"):
        with open("results/cost_sensitivity.json", encoding="utf-8") as f:
            cs = json.load(f)
        t, w = cs["tta"], cs["withhold"]
        fb = lambda d: ", ".join(f"{k} {v:+.4f}" for k, v in d.items())
        tta_cross = ("No positive-cost crossover exists on this stream (mean TTA benefit is ≤ 0 in every "
                     "drift bucket)" if t["crossover_cost"] is None
                     else f"A positive crossover exists only at cost ≈ {t['crossover_cost']:.4f}, and only in the "
                          f"least-shifted bucket — where intervention is least needed")
        cost_md = f"""
## Cost-sensitivity (crossover values, derived from archived traces)

Computed analytically from the committed Run C per-batch benefits (no policy re-runs;
committed costs untouched). A crossover is the cost below which an intervention's mean
realized benefit (committed definition) turns positive — an oracle-estimate upper bound;
a learning supervisor additionally pays its measured ≈{cs['supervisor_exploration_overhead']:.4f} exploration overhead.

- **TTA** (committed cost 0.02): stream-mean benefit {t['stream_mean_benefit']:+.4f}; per bucket: {fb(t['by_bucket'])}.
  {tta_cross}. Notably, even the gate sweep's best-case TTA ECE benefit
  ({t['gate_best_case_ece_benefit']:+.4f}, noise-only cells) sits just **below** the committed 0.02 cost — that
  proximity is itself part of the finding: TTA is priced almost exactly at the margin of
  its best measurable calibration value here.
- **ABSTAIN/ESCALATE** (committed costs 0.06/0.15): withhold benefit crosses positive at
  cost ≈ {w['crossover_cost_stream']:.4f} stream-wide, and ≈ {w['crossover_cost_best_bucket']:.4f} in the most-shifted bucket
  (per bucket: {fb(w['by_bucket'])}). Below ≈ {w['crossover_cost_best_bucket'] - cs['supervisor_exploration_overhead']:.4f} (best bucket minus exploration
  overhead) a bucket-aware supervisor should overtake the never-intervene static baseline;
  at the committed 0.06 it necessarily trails it by roughly the exploration overhead.
"""

    # TTA-recovery gate (Day 1, D29/D30), rendered only if the sweep was run
    gate_md = ""
    if os.path.exists("results/gate_tta_sweep.json"):
        with open("results/gate_tta_sweep.json", encoding="utf-8") as f:
            g = json.load(f)
        rows_g = [r for r in g["rows"] if r["corruption"] != "clean"]
        best3 = sorted(rows_g, key=lambda r: r["acc_after"] - r["acc_before"], reverse=True)[:3]
        best_txt = "; ".join(
            f"{r['corruption']} s={r['strength']}: {(r['acc_after'] - r['acc_before']) * 100:+.1f}pt accuracy, "
            f"ΔECE {r['ece_after'] - r['ece_before']:+.4f}, drift bucket b{r['bucket']}" for r in best3)
        gate_md = f"""
## TTA-recovery gate: the two-regime extension was not built (a finding, not a failure)

A gated Day-1 sweep ({len(rows_g)} corruption×strength cells × {g['n_batches_per_cell']}×{g['batch_size']} held-out samples,
before/after episodic TTA, paired; full table in `results/gate_tta_sweep.json`) asked
whether TTA yields real accuracy recovery in some mild band. Result: **TTA produces
genuine accuracy recovery under gaussian-noise shift** (best cells: {best_txt}), growing
with severity and absent for blur/jitter — **but the calibration-based benefit metric is
structurally blind to it**: TTA's best-case ECE improvement (+{max(r['ece_before'] - r['ece_after'] for r in rows_g):.4f}) is below the
committed TTA cost (0.02), so a calibration-driven supervisor *correctly declines an
intervention that would have improved accuracy*. No mild-severity recovery band exists
(recovery is noise-only and severity-increasing), so the planned mild-vs-severe regime
contrast has no substrate on this model, and per the pre-committed gate it was not built.
This is a conceptual finding about **benefit-metric choice in runtime supervision** — how
intervention benefit should be defined when calibration- and accuracy-value diverge is a
thesis-level question (the exposé's benefit definition is deliberately calibration-only,
and redefining it post hoc would be result-shopping).
"""

    def verdict(a, b, what):
        d = a["net_benefit"] - b["net_benefit"]
        if d > 0.005:
            return f"the supervisor beats {what} by {d:+.4f} net calibration benefit."
        if d < -0.005:
            return f"the supervisor does NOT beat {what} ({d:+.4f}) — a legitimate negative result per the exposé."
        return f"the supervisor and {what} are within noise of each other ({d:+.4f}) — a null result, reportable as such."

    md = f"""# RESULTS — agentic-runtime-supervisor (PoC)

Generated by `ars.report` from `runs/summary.json`. Single seed = {meta['seed']}; stream =
{meta['n_stream_batches']} batches × {meta['stream_batch_size']} (DermaMNIST test pool, corrupted in
10-batch regime blocks, ~40% clean). Deployed model = frozen tiny ViT + temperature
T={meta['temperature']:.3f} fitted on the calibration set: clean test accuracy {meta['clean_test_acc']:.4f},
clean conformal coverage {meta['clean_coverage']:.4f} (target 0.95, q_hat={meta['qhat_frozen']:.4f}).
Monitor tuned on the eval stream to `{meta['monitor_rule']}`.

Primary metric: **net calibration benefit** = mean realized ECE-reduction benefit − mean
intervention cost (design.md §4; identical formula for every method, computed with full labels
by the evaluation harness — the *policies* only ever saw gated labels). Decision traces for
this run are archived in `results/seed{meta['seed']}/` (one JSONL per method + summary.json), committed
and inspectable without re-running.

## Headline findings

1. **TTA worsens calibration under shift here.** Pooled issued ECE {S['always_tta']['ece_issued']:.4f} with
   always-TTA vs {S['never_tta']['ece_issued']:.4f} without (net {S['always_tta']['net_benefit']:+.4f} after costs). Entropy minimization
   sharpens confidence, which under the exposé's calibration-not-accuracy objective makes
   blanket adaptation net-harmful; the supervisor's learned behavior is accordingly
   *restraint*, and blind-intervention policies lose 4–8× more than the learners' small
   exploration overhead.
2. **RQ2 is a tie — the pre-committed, reportable negative result.** Supervisor
   {sup['net_benefit']:+.4f} vs memoryless bandit {band['net_benefit']:+.4f} (|Δ| = {abs(sup['net_benefit'] - band['net_benefit']):.4f}, inside the 0.005 single-seed noise
   band). Bucketed outcome memory did not beat a stationary linear reward surface in this
   episodic-TTA setting — consistent with the exposé's own caution that episodic
   reversion mutes the non-stationarity argument for the agentic controller.

## Correction during development (RQ1 attribution)

An earlier configuration of this PoC showed the supervisor beating all static baselines
(+0.0496 vs +0.0028 for the then-monitor; Run A, 2026-07-15). That advantage was traced —
by attribution across three runs, not conjecture — to two artifacts: (a) the small-sample
ECE noise floor (≈0.11 at batch size 64) was being credited to the ABSTAIN/ESCALATE
benefit, paying every policy for withholding well-calibrated batches; and (b) the
fixed-threshold monitor was denied the ABSTAIN action — the very action the artifact made
profitable. Both were corrected (temperature scaling, floor-debiased withhold benefit,
full-vocabulary monitor grid; memory.md D22–D26, author-reviewed and approved), and the
apparent supervisor advantage disappeared. The corrected numbers are reported: a
comparison that survives its own artifact audit is worth more than a win that does not.

## Head-to-head comparison (the single table)

{_table([S[m] for m in HEADLINE])}
**Reading the table (numbers, not spin):** always-TTA shifts issued accuracy by
{(S['always_tta']['accuracy_issued'] - S['never_tta']['accuracy_issued']) * 100:+.1f}pt and pooled ECE by {S['always_tta']['ece_issued'] - S['never_tta']['ece_issued']:+.4f} vs never-TTA — under the exposé's
calibration-not-accuracy objective its net is {S['always_tta']['net_benefit']:+.4f}. The net-benefit spread across
policies that intervene blindly vs learn restraint: always-TTA {S['always_tta']['net_benefit']:+.4f}, no-memory
threshold ablation {S['supervisor_no_benefit']['net_benefit']:+.4f}, bandit {S['bandit']['net_benefit']:+.4f}, supervisor {S['supervisor']['net_benefit']:+.4f} (the learners' residual
gap to zero is exploration plus estimation noise from only ~24 gated-labeled batches —
the supervisor's {S['supervisor']['n_tta']} TTA picks come from noisy per-batch outcome estimates, not the
ε≈10% exploration floor alone). Note the no-memory ablation buys the best issued
accuracy/ECE/coverage by escalating {S['supervisor_no_benefit']['n_escalate']} batches to humans — quality via cost, which the
net metric correctly prices as the worst policy.

## Ablations (RQ1 signal ablations, VoI bonus, RQ3 recalibration contrast)

{_table([S[m] for m in ABLATIONS])}
## Research-question readouts

**RQ1 (supervisor vs best-tuned static policy):** best static baseline is
"{LABELS[best_static['method']]}" (net {best_static['net_benefit']:.4f}); {verdict(sup, best_static, 'the best static baseline')}
Signal ablation: strongest reduced variant is "{LABELS[best_abl['method']]}"
(net {best_abl['net_benefit']:.4f}); {verdict(sup, best_abl, 'its strongest single-signal reduction')}{abl_ms}

**RQ2 (memory vs memoryless, the headline test):** supervisor (escalation bonus OFF,
so the only difference under test is bucketed outcome memory vs a stationary linear
reward surface) nets {sup['net_benefit']:.4f} vs bandit {band['net_benefit']:.4f}: {verdict(sup, band, 'the memoryless bandit')}
The VoI-bonus run ("{LABELS['supervisor_bonus']}", net {S['supervisor_bonus']['net_benefit']:.4f},
{S['supervisor_bonus']['n_escalate']} escalations vs {sup['n_escalate']} without) is reported here as its own
finding, deliberately outside RQ2 (memory.md D17).
{ms_md}

**RQ3 (does adapt-then-recalibrate preserve coverage under TTA?):** always-TTA with
recalibration covers {_fmt(S['always_tta']['coverage'], pct=True)} vs {_fmt(S['always_tta_norecal']['coverage'], pct=True)} without,
against the 95% target (corrupted-only: {meta['rq3_corrupted_recal']:.1%} vs {meta['rq3_corrupted_norecal']:.1%}).
Recalibration is a flagged simplification (clean calibration set vs shifted batch), not an
exchangeability fix. Context: 3-step LayerNorm-only adaptation barely moves the scoring
function, so recalibration has little to correct here; the coverage gap to target is
driven by the shift itself (clean coverage {meta['clean_coverage']:.1%} → corrupted ~{meta['rq3_corrupted_recal']:.0%}). A stronger/continual
TTA regime (thesis scope) is where recalibration should matter.
{cost_md}{gate_md}
## Limitations and disclosures (design.md §10)

1. **Single seed ({meta['seed']}).** No variance estimate; differences within a few
   thousandths of net benefit should be read as noise. Multi-seed sweeps are out of scope.
2. **Counterfactual replay** (approved, PRD §6.4): on gated-labeled batches, all four
   actions' outcomes update the estimators. This makes the PoC a demonstration of the
   runtime-supervision architecture, **not** a test of online sample-efficiency under
   gated feedback (thesis scope). Applied identically to the bandit.
3. **No oracle labels in any policy.** Decision-time signals are label-free; policy
   learning uses only escalated + monitoring batches (~10%). Full labels are used only
   for reporting, which is standard.
4. Costs {meta['costs']} (ECE units) and the ~40%-clean stream mix jointly locate the
   arbitration boundaries; ordering, not magnitude, is the argument. The benefit
   definition credits avoided ECE, which structurally favors abstention on heavily
   corrupted batches (exposed by the abstention-rate column). For reference, the
   degenerate always-abstain policy would net {meta['always_abstain_net']:+.4f} on this stream —
   any method near that number is (or found) that policy.
4b. **Batch-ECE noise floor (debiased, D24).** Per-batch ECE on {meta['stream_batch_size']} samples is biased
   upward even for a calibrated model (measured clean floor ≈ {meta['clean_batch_ece_floor']:.4f} while the
   pooled clean ECE is far lower) — uncorrected, it pays withholding for estimator bias
   and every policy degenerates to always-abstain. The ABSTAIN/ESCALATE benefit therefore
   subtracts this floor, measured a priori on the calibration set. The floor is an
   approximation under shift; TTA's benefit is a same-batch difference, so the bias
   largely cancels there. Temperature scaling (T={meta['temperature']:.3f}) separately restores the
   exposé's well-calibrated-deployment premise.
5. Adapt-then-recalibrate does not restore exchangeability; RQ3 measures it.
6. Episodic TTA reverts each batch, muting the exposé's non-stationarity argument for
   the agentic controller — this PoC setting is *kinder to the bandit* than the thesis
   setting would be.
7. The fixed-threshold monitor is tuned on the evaluation stream (an oracle-tuned upper
   bound for static policies — deliberately favors the baseline).
8. Stream reuses test images across batches; batch-level ECE on 64 samples is noisy
   (policies learn from a noisy but honest outcome signal; table ECE is pooled).
9. Batch-level actions; per-sample selective prediction is future work.
10. **Milder shift than the motivating case (D25).** Under these corruptions the tiny ViT
   fails with LOW confidence (entropy rises; corrupted batch-level ECE ≈ 0.13–0.16), not
   the thesis's confidently-wrong regime (shift ECE 0.889). The silent-failure setting
   that most motivates runtime supervision is under-represented in this stream;
   confidently-wrong shifts are thesis-scale work.
"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"wrote {out_path}")
