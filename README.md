# agentic-runtime-supervisor

Proof-of-concept for **agentic runtime supervision of a deployed medical vision model
under distribution shift** (research exposé, ETH Agentic Systems Lab application).

## Motivation

A deployed medical classifier facing input drift has four options per batch: trust the
prediction, adapt at test time, abstain, or escalate to a human — and each is wrong most
of the time. Blind trust ships confident errors (a ViT on DermaMNIST can stay *highly
confident while systematically wrong* under shift); always-adapt wastes compute, can
degrade a healthy model, and silently breaks conformal coverage; always-defer destroys
throughput. Fixed thresholds can't weigh context, asymmetric costs, or what past
interventions actually achieved.

This PoC puts a **runtime supervisor** in front of a frozen tiny ViT (temperature-scaled
on its calibration set, per standard deployment practice; DermaMNIST, synthetic
corruption stream). Per batch it reads three **label-free** signals — embedding
drift vs a clean-calibration reference, conformal prediction set size, and
expected-benefit-minus-cost — picks one of **ACCEPT / TTA / ABSTAIN / ESCALATE**, then
updates a running per-action, per-drift-bucket estimate of realized ECE reduction from
**delayed, action-gated** feedback (labels only from escalations and a fixed ~10%
labeled monitoring subset). The policy is the exposé's online outcome estimator — a
simple frequency estimate, deliberately not a learned/RL policy.

## Architecture

```
                     ┌──────────── OFFLINE (clean data) ────────────┐
 DermaMNIST train ─► │ train tiny ViT ─► frozen checkpoint          │
 DermaMNIST val ───► │ split-conformal: q_hat (95%) + drift ref     │
                     └──────────────────────┬───────────────────────┘
                                            ▼
 corrupted stream ──► ┌──────── SUPERVISOR (LangGraph, per batch) ────────┐
 (200×64; noise/blur/ │ SignalCollection ─► RiskAssessment ─► Policy      │
  jitter in blocks)   │  drift, set size     bucket, gate     Decision    │
                      │                                        │ ε-greedy │
                      │        argmax_a  Q̂(a,bucket) − cost(a) ▼          │
                      │ OutcomeLogging ◄──────────────────── Action       │
                      │  JSONL trace;      ACCEPT | TTA(+recalibrate,     │
                      │  gated labels ─►   then revert) | ABSTAIN |       │
                      │  update memory     ESCALATE (returns labels)      │
                      └───────────────┬───────────────────────────────────┘
                    persistent outcome memory (conditions the next batch)
```

No LLM calls anywhere — LangGraph is the state-machine substrate; nodes are plain Python.

## Reproduce (single command)

```bash
uv sync
uv run ars-reproduce
```

Trains the ViT if no checkpoint exists (~3 min CPU), calibrates, streams 200 corrupted
batches through **six methods** (static raw, never-TTA, always-TTA, oracle-tuned
fixed-threshold monitor, memoryless contextual bandit, supervisor) plus ablations
(three signal ablations, escalation-VoI bonus, recalibration on/off), writes per-batch
decision traces to `runs/*.jsonl`, and generates [RESULTS.md](RESULTS.md).

Everything is seeded (seed 0); the whole pipeline is CPU-only and finishes in minutes.
Each run archives its decision traces (one JSONL per method + summary.json) to the
committed `results/seed<seed>/` folder, so the submitted experiment's per-batch decisions
are inspectable without re-running.

## What the comparison shows

One head-to-head table (accuracy, ECE, empirical coverage vs 95%, interventions,
abstention rate, and **net calibration benefit** = realized ECE reduction minus fixed
intervention cost) answering, per the exposé:

- **RQ1** — does the multi-signal outcome-updating supervisor beat the best-tuned static
  policy (and its strongest single-signal ablation)?
- **RQ2** — does bucketed outcome memory beat a memoryless contextual bandit seeing the
  same signals, same gated labels? (Escalation VoI bonus OFF in this headline.)
- **RQ3** — does adapt-then-recalibrate preserve empirical conformal coverage under
  episodic TTA?

A null or negative result is a legitimate, reported answer — and that is what this run
produced: on this stream no intervention clears its cost bar (LayerNorm-entropy TTA
*worsens* calibration by sharpening confidence), so the oracle-tuned static monitor
collapses to "never intervene" and the supervisor cannot beat it (RQ1 negative); the
supervisor matches the memoryless bandit within single-seed noise (RQ2 null). The
positive finding sits in the ablation: removing the outcome-estimator loop (keeping
only threshold logic) costs −0.053 net — learned restraint is what the outcome memory
buys. See [RESULTS.md](RESULTS.md) for the table and the full limitations list (single
seed, counterfactual replay, batch-ECE floor debiasing, cost constants, etc.).

## Repo map

- `docs/` — planning documents (requirements, architecture, design, rules, phases,
  decision log). `docs/design.md` specifies the policy precisely; `docs/memory.md`
  logs every design decision with rejected alternatives.
- `src/ars/` — `policy.py` (outcome-estimator policy — the core), `graph.py` (LangGraph
  nodes), `tta.py` (episodic LayerNorm-only TTA + adapt-then-recalibrate),
  `conformal.py`, `signals.py`, `baselines.py`, `runner.py` (label-path-separated
  harness), `report.py`, `reproduce.py`.
