"""Single-command reproduction: train (if needed) -> calibrate -> stream all methods ->
ablations -> RESULTS.md.   Usage: uv run ars-reproduce
"""
import json
import os

import numpy as np
import torch

from . import report, train
from .baselines import RidgeBandit, accept_decider, always_tta_decider, monitor_decider, tune_monitor
from .config import CFG
from .conformal import conformal_qhat, covered, ece, prediction_sets, set_sizes, softmax_probs
from .data import build_stream, calibration_split, load_splits, normalize
from .graph import GraphContext, build_graph
from .model import load_frozen
from .policy import SupervisorPolicy
from .runner import gated_feedback, precompute, run_method
from .signals import fit_drift_reference


def run_supervisor(name, records, model, cal_x, cal_y, qhat_frozen, ref, seed_offset, **flags):
    policy = SupervisorPolicy(np.random.default_rng(CFG.seed + seed_offset), **flags)
    by_id = {r.batch.batch_id: r for r in records}
    ctx = GraphContext(
        model=model, cal_x=cal_x, cal_y=cal_y, qhat_frozen=qhat_frozen, ref=ref, policy=policy,
        feedback_provider=lambda bid, action: gated_feedback(by_id[bid], action),
        cache_lookup=lambda bid: by_id[bid],
    )
    app = build_graph(ctx)

    def decide(rec):
        out = app.invoke({"batch_id": rec.batch.batch_id, "x": rec.batch.x})
        assert out["bucket"] == rec.bucket and abs(out["drift"] - rec.drift) < 1e-9, \
            "live signals diverged from precompute"
        return out["action"], {"utilities": out["utilities"], "explored": out["explored"],
                               "reasoning": out["reasoning"], "memory": out["memory"]}

    # learner=None: gated feedback is consumed INSIDE the graph's OutcomeLogging node
    return run_method(name, records, decide, qhat_frozen)


def main():
    torch.manual_seed(CFG.seed)
    np.random.seed(CFG.seed)

    print("== data ==")
    splits = load_splits()
    te_x, te_y = splits["test"]
    cal_x, cal_y = calibration_split(*splits["val"])
    cal_x, te_x_n = normalize(cal_x), normalize(te_x)

    if not os.path.exists(CFG.checkpoint_path):
        print("== training (no checkpoint found) ==")
        train.main()
    model = load_frozen()

    print("== phase 2: split-conformal calibration ==")
    with torch.no_grad():
        clean_acc = float((model(te_x_n).argmax(1) == te_y).float().mean())
        cal_probs = softmax_probs(model(cal_x))
        test_probs = softmax_probs(model(te_x_n))
    qhat_frozen = conformal_qhat(cal_probs, cal_y.numpy())
    clean_cov = float(covered(prediction_sets(test_probs, qhat_frozen), te_y.numpy()).mean())
    clean_mss = float(set_sizes(prediction_sets(cal_probs, qhat_frozen)).mean())
    print(f"clean test acc={clean_acc:.4f}  qhat={qhat_frozen:.4f}  "
          f"clean coverage={clean_cov:.4f} (target {1 - CFG.alpha})  clean mean set size={clean_mss:.2f}")

    ref = fit_drift_reference(model, cal_x, clean_mss)
    print(f"drift reference: q95_clean={ref.q95_clean:.4f}, bucket edges "
          f"[{ref.q95_clean:.4f}, {CFG.drift_edge_factor * ref.q95_clean:.4f}]")

    print("== phase 3: stream + precompute (episodic TTA asserted per batch) ==")
    stream = build_stream(te_x, te_y)
    records = precompute(model, stream, cal_x, cal_y, qhat_frozen, ref)
    corrupted = [r for r in records if r.batch.corruption != "clean"]
    clean = [r for r in records if r.batch.corruption == "clean"]
    print(f"{len(records)} batches ({len(clean)} clean); "
          f"mean ECE frozen clean={np.mean([r.ece_frozen for r in clean]):.4f} "
          f"corrupted={np.mean([r.ece_frozen for r in corrupted]):.4f} "
          f"corrupted after TTA={np.mean([r.ece_adapted for r in corrupted]):.4f}; "
          f"drift clean={np.mean([r.drift for r in clean]):.3f} "
          f"corrupted={np.mean([r.drift for r in corrupted]):.3f}")

    print("== phases 4-5: methods ==")
    summaries = []
    summaries.append(run_method("static_raw", records, accept_decider, qhat_frozen, use_conformal=False))
    summaries.append(run_method("never_tta", records, accept_decider, qhat_frozen))
    summaries.append(run_method("always_tta", records, always_tta_decider, qhat_frozen))
    summaries.append(run_method("always_tta_norecal", records, always_tta_decider, qhat_frozen,
                                recalibrate=False))
    tau_tta, tau_esc, tuned_net = tune_monitor(records)
    print(f"monitor tuned on eval stream: tau_tta={tau_tta:.3f} tau_esc={tau_esc:.3f} net={tuned_net:.4f}")
    summaries.append(run_method("monitor_fixed", records, monitor_decider(tau_tta, tau_esc), qhat_frozen))

    bandit = RidgeBandit(np.random.default_rng(CFG.seed + 20))
    summaries.append(run_method("bandit", records, bandit.decide, qhat_frozen, learner=bandit.learn))

    sup_args = dict(records=records, model=model, cal_x=cal_x, cal_y=cal_y,
                    qhat_frozen=qhat_frozen, ref=ref)
    summaries.append(run_supervisor("supervisor", seed_offset=10, **sup_args))
    summaries.append(run_supervisor("supervisor_bonus", seed_offset=10, escalation_bonus=True, **sup_args))
    summaries.append(run_supervisor("supervisor_no_drift", seed_offset=11, use_drift=False, **sup_args))
    summaries.append(run_supervisor("supervisor_no_setsize", seed_offset=12, use_setsize=False, **sup_args))
    summaries.append(run_supervisor("supervisor_no_benefit", seed_offset=13, use_benefit=False, **sup_args))

    for s in summaries:
        print(f"  {s['method']:24s} net={s['net_benefit']:+.4f} tta={s['n_tta']:3d} "
              f"esc={s['n_escalate']:3d} abstain={s['n_abstain']:3d}")

    # RQ3 corrupted-only coverage of the adapted model, with vs without recalibration
    def cov(rec_list, qhat_fn):
        c = [covered(prediction_sets(r.probs_adapted, qhat_fn(r)), r.batch.y.numpy()) for r in rec_list]
        return float(np.concatenate(c).mean())

    meta = {
        "seed": CFG.seed, "n_stream_batches": CFG.n_stream_batches,
        "stream_batch_size": CFG.stream_batch_size, "clean_test_acc": clean_acc,
        "clean_coverage": clean_cov, "qhat_frozen": qhat_frozen,
        "tau_tta": tau_tta, "tau_esc": tau_esc, "costs": CFG.cost,
        "rq3_corrupted_recal": cov(corrupted, lambda r: r.qhat_adapted),
        "rq3_corrupted_norecal": cov(corrupted, lambda r: qhat_frozen),
    }
    os.makedirs(CFG.runs_dir, exist_ok=True)
    with open(os.path.join(CFG.runs_dir, "summary.json"), "w") as f:
        json.dump({"summaries": summaries, "meta": meta}, f, indent=1)

    print("== phase 6: report ==")
    report.generate()


if __name__ == "__main__":
    main()
