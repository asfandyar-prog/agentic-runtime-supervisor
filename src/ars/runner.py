"""Stream harness: per-batch precompute cache, method runners, JSONL traces, summaries.

Two strictly separated label paths (rules.md invariant, memory.md D18):
  1. REPORTING -- run_method() computes table metrics with full ground truth for every
     method. Standard practice; never flows into any policy.
  2. POLICY LEARNING -- gated_feedback() is the ONLY constructor of PolicyFeedback and
     returns it exclusively for escalated batches and monitoring batches.

Precompute (memory.md D19): episodic TTA is deterministic given (frozen weights, batch)
-- no random sampling anywhere in adaptation -- so each batch's adapted outcome is
computed ONCE and reused for baselines, monitor tuning, and counterfactual replay.
The supervisor's Action node still runs TTA live and asserts equality with the cache,
which doubles as a determinism proof. Episodic reversion is asserted bit-identical here.
"""
import json
import os
from dataclasses import dataclass

import numpy as np
import torch

from .config import CFG
from .conformal import covered, ece, prediction_sets, set_sizes, softmax_probs
from .data import StreamBatch
from .policy import ACTIONS, PolicyFeedback
from .signals import DriftReference
from .tta import _ln_params, adapt_batch


@dataclass
class BatchRecord:
    batch: StreamBatch
    drift: float
    bucket: int
    mean_set_size: float          # frozen model, frozen q_hat
    inflated: bool
    probs_frozen: np.ndarray
    ece_frozen: float
    probs_adapted: np.ndarray     # episodic TTA outcome (deterministic)
    qhat_adapted: float           # adapt-then-recalibrate threshold
    ece_adapted: float
    ece_floor: float              # clean batch-ECE noise floor, measured a priori (D24)


def benefit(action: str, rec: BatchRecord) -> float:
    """Realized ECE-reduction benefit (design.md #4). ACCEPT is the do-nothing
    reference. TTA is a same-batch DIFFERENCE, so the small-sample ECE bias largely
    cancels. ABSTAIN/ESCALATE avoid shipping the frozen model's miscalibration, but
    batch ECE on 64 samples is biased upward even for a calibrated model (~0.11
    measured), which would pay withholding for estimator bias rather than avoided
    miscalibration -- so the clean floor, measured a priori on the calibration set,
    is subtracted (signable: keeps the estimate unbiased; memory.md D24)."""
    if action == "ACCEPT":
        return 0.0
    if action == "TTA":
        return rec.ece_frozen - rec.ece_adapted
    return rec.ece_frozen - rec.ece_floor  # ABSTAIN / ESCALATE


def precompute(model, stream: list[StreamBatch], cal_x, cal_y, qhat_frozen: float,
               ref: DriftReference, ece_floor: float) -> list[BatchRecord]:
    records = []
    ln_before = torch.cat([p.detach().flatten() for p in _ln_params(model)]).clone()
    for sb in stream:
        with torch.no_grad():
            emb = model.forward_features(sb.x).numpy().astype(np.float64)
            probs_f = softmax_probs(model(sb.x))
        drift = ref.drift_value(emb)
        sizes = set_sizes(prediction_sets(probs_f, qhat_frozen))
        y = sb.y.numpy()

        probs_a, qhat_a = adapt_batch(model, sb.x, cal_x, cal_y)
        ln_after = torch.cat([p.detach().flatten() for p in _ln_params(model)])
        assert torch.equal(ln_before, ln_after), "episodic TTA failed to revert bit-identically"

        records.append(BatchRecord(
            batch=sb, drift=drift, bucket=ref.bucket(drift),
            mean_set_size=float(sizes.mean()), inflated=ref.setsize_inflated(float(sizes.mean())),
            probs_frozen=probs_f, ece_frozen=ece(probs_f, y),
            probs_adapted=probs_a, qhat_adapted=qhat_a, ece_adapted=ece(probs_a, y),
            ece_floor=ece_floor,
        ))
    return records


def gated_feedback(rec: BatchRecord, action: str) -> PolicyFeedback | None:
    """THE label-path boundary. Labels reach a policy only if (a) it escalated -- the
    human resolves the batch and returns ground truth -- or (b) this is a monitoring
    batch (fixed labeled ~10% subset, user-approved PRD #6.1). Benefits include all
    four actions' counterfactuals (replay, design.md #6) -- legitimate because the
    labels themselves arrived through the gate."""
    if action == "ESCALATE":
        source = "escalation"
    elif rec.batch.monitored:
        source = "monitoring"
    else:
        return None
    return PolicyFeedback(bucket=rec.bucket, source=source,
                          benefits={a: benefit(a, rec) for a in ACTIONS})


def run_method(name: str, records: list[BatchRecord], decider, qhat_frozen: float,
               use_conformal: bool = True, recalibrate: bool = True,
               learner=None, runs_dir: str = CFG.runs_dir) -> dict:
    """Stream all batches through one method. decider(rec) -> (action, extra_trace_dict).
    learner(fb, rec), if given, receives ONLY gated feedback."""
    os.makedirs(runs_dir, exist_ok=True)
    trace_path = os.path.join(runs_dir, f"{name}.jsonl")
    pooled_probs, pooled_y, pooled_cov, pooled_sizes = [], [], [], []
    n_act = {a: 0 for a in ACTIONS}
    total_cost = total_benefit = 0.0

    with open(trace_path, "w", encoding="utf-8") as f:
        for rec in records:
            action, extra = decider(rec)
            n_act[action] += 1
            y = rec.batch.y.numpy()
            cost = CFG.cost[action]
            ben = benefit(action, rec)   # reporting path: full labels, never enters memory
            total_cost += cost
            total_benefit += ben

            row = {
                "batch_id": rec.batch.batch_id, "corruption": rec.batch.corruption,
                "monitored": rec.batch.monitored, "drift": round(rec.drift, 4),
                "bucket": rec.bucket, "mean_set_size": round(rec.mean_set_size, 3),
                "inflated": rec.inflated, "action": action, "cost": cost,
                "ece_frozen": round(rec.ece_frozen, 4), "benefit": round(ben, 4),
            }
            issued = action in ("ACCEPT", "TTA")
            if issued:
                probs = rec.probs_frozen if action == "ACCEPT" else rec.probs_adapted
                qhat = qhat_frozen if (action == "ACCEPT" or not recalibrate) else rec.qhat_adapted
                pooled_probs.append(probs)
                pooled_y.append(y)
                row["acc_batch"] = round(float((probs.argmax(1) == y).mean()), 4)
                row["ece_realized"] = round(ece(probs, y), 4)
                if use_conformal:
                    sets = prediction_sets(probs, qhat)
                    cov, sz = covered(sets, y), set_sizes(sets)
                    pooled_cov.append(cov)
                    pooled_sizes.append(sz)
                    row["coverage_batch"] = round(float(cov.mean()), 4)

            fb = gated_feedback(rec, action)
            row["feedback_source"] = fb.source if fb else None
            if learner is not None and fb is not None:
                learner(fb, rec)
            row.update(extra)
            f.write(json.dumps(row) + "\n")

    n = len(records)
    if pooled_probs:
        P, Y = np.concatenate(pooled_probs), np.concatenate(pooled_y)
        acc, pooled_ece = float((P.argmax(1) == Y).mean()), ece(P, Y)
    else:
        acc = pooled_ece = None
    summary = {
        "method": name, "n_batches": n,
        "accuracy_issued": acc, "ece_issued": pooled_ece,
        "coverage": float(np.concatenate(pooled_cov).mean()) if pooled_cov else None,
        "mean_set_size": float(np.concatenate(pooled_sizes).mean()) if pooled_sizes else None,
        "n_tta": n_act["TTA"], "n_escalate": n_act["ESCALATE"], "n_abstain": n_act["ABSTAIN"],
        "abstention_rate": n_act["ABSTAIN"] / n,
        "mean_cost": total_cost / n, "mean_benefit": total_benefit / n,
        "net_benefit": (total_benefit - total_cost) / n,
    }
    return summary
