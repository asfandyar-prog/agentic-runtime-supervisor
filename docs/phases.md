# Phases — ~2-day build plan (≈16h budget, CPU-only)

Sequential; each phase ends with a **review checkpoint** (a runnable sanity check + a memory.md entry). No phase may expand beyond the exposé's stated scope — if a phase wants to grow, the growth goes to "future work," not code.

## Phase 0 — Repo scaffold (0.5h)
`git init`, pyproject.toml (uv), package skeleton, config.py with all constants from design.md, .gitignore (checkpoints, runs/, data cache).
**Checkpoint:** `uv run python -c "import ars"` works; deps resolve.

## Phase 1 — Data + frozen ViT classifier (3h)
medmnist DermaMNIST download + splits per architecture.md §3; corruption transforms (gaussian noise, gaussian blur, color jitter; severities 1–3); stream builder (200 batches × 64, block schedule, monitoring mask); small ViT (patch 4, dim 96, depth 4, heads 4); training script (~15 epochs, CPU-feasible in minutes).
**Checkpoint:** clean test accuracy ≥ ~65% (accuracy is secondary; it must merely be well above the 7-class chance and clean-calibratable); corrupted accuracy visibly degraded; checkpoint saved.

## Phase 2 — Split-conformal calibration (1.5h)
LAC scores on the 500-image clean calibration set; q_hat at 95%; prediction sets; empirical coverage on clean test; drift reference stats (μ_ref, σ_ref, bootstrap clean-batch drift quantiles) — computed here because they share the calibration set.
**Checkpoint:** clean coverage ≈ 0.95 (±2pts); corrupted coverage visibly broken; drift value separates clean vs corrupted batches.

## Phase 3 — Episodic LayerNorm TTA + adapt-then-recalibrate (2.5h)
Tent-style entropy minimization, LN γ/β only, 3 steps, snapshot/restore (episodic — revert verified by test); recalibration pass (calibration set through adapted model → fresh q_hat for the batch).
**Checkpoint:** on moderately corrupted batches, TTA reduces batch ECE on average; after revert, frozen-model outputs are bit-identical to pre-TTA; recalibrated coverage ≥ non-recalibrated coverage on TTA'd batches (RQ3 preview).

## Phase 4 — Supervisor graph + online outcome-estimator policy (4h)
The graded core. LangGraph nodes per architecture.md §4; policy.py implementing design.md §§1–7 exactly (buckets, gate, costs, escalation bonus, ε-greedy, gated + delayed feedback, counterfactual replay, running-mean updates); JSONL trace with reasoning strings and estimate snapshots every batch.
**Checkpoint:** a full supervisor stream run completes; trace shows (a) label-free decisions, (b) memory updating only on labeled batches — the label-path separation invariant (rules.md, memory.md D18) verified explicitly, (c) action mix varying by drift regime; every decision auditable from the trace alone.

## Phase 5 — Baselines + comparison + ablation run (3h)
static, never-TTA, always-TTA (±recalibration), grid-tuned threshold monitor, contextual bandit (design.md §8); signal-ablation modes (design.md §9); runner executes all methods on the identical stream (same seed, same batches); report.py aggregates traces → the single head-to-head table + ablation table.
**Checkpoint:** all runs complete from one command; table columns match PRD §5.2; bandit and supervisor consumed identical label budgets.

## Phase 6 — RESULTS.md + README (1.5h)
report.py emits RESULTS.md: both tables, RQ1–RQ3 readouts, single-seed limitation, full disclosure list (design.md §10). README: motivation, ASCII architecture diagram, single-command reproduce (`uv run ars-reproduce` or equivalent), honest summary of what the PoC does and does not show.
**Checkpoint:** fresh-clone reproduction path verified; docs/memory.md current.

**Total: ~16h.** Buffer lives inside Phases 4–5 estimates. If time runs short, the cut order is: ablation variants first (keep at minimum no-benefit/cost, the strongest reduction), never the bandit (RQ2 comparator is required), never the head-to-head table.
