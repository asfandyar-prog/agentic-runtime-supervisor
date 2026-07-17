"""Day-3 additions (author-ordered, memory.md D30). Two items, no config changes:

(a) Multi-seed the no-benefit/cost ablation across the same 5 seeds as the D28 RQ2 run
    (stream + policy RNG vary; classifier/calibration fixed). The ablation's decide()
    is deterministic given the stream (no epsilon draw), so seed 0 must byte-match the
    committed Run C archive and is verified, never overwritten.

(b) Cost-sensitivity crossovers, derived ANALYTICALLY from the committed Run C
    per-batch traces (results/seed0/) -- no policy is re-run under altered costs, so
    the committed cost constants are untouched. For each intervention, the crossover
    is the cost below which its mean realized benefit (committed definition) is
    positive, stream-wide and per drift bucket; the supervisor's measured exploration
    overhead is reported alongside as the practical correction.

Usage: uv run python -m ars.day3
"""
import json
import os
import shutil

import numpy as np
import torch

from .config import CFG
from .conformal import conformal_qhat, ece, prediction_sets, set_sizes, softmax_probs
from .data import build_stream, calibration_split, load_splits, normalize
from .model import TemperatureScaled, fit_temperature, load_frozen
from .reproduce import run_supervisor
from .runner import precompute
from .signals import fit_drift_reference

N_SEEDS = 5


def multiseed_no_benefit():
    torch.manual_seed(CFG.seed)
    np.random.seed(CFG.seed)
    splits = load_splits()
    te_x, te_y = splits["test"]
    cal_x, cal_y = calibration_split(*splits["val"])
    cal_x = normalize(cal_x)
    temperature = fit_temperature(load_frozen(), cal_x, cal_y)
    model = TemperatureScaled(load_frozen(), temperature)
    with torch.no_grad():
        cal_probs = softmax_probs(model(cal_x))
    qhat_frozen = conformal_qhat(cal_probs, cal_y.numpy())
    clean_mss = float(set_sizes(prediction_sets(cal_probs, qhat_frozen)).mean())
    ref = fit_drift_reference(model, cal_x, clean_mss)
    rng = np.random.default_rng(CFG.seed + 30)
    floor = float(np.mean([ece(cal_probs[idx], cal_y.numpy()[idx])
                           for idx in rng.integers(0, len(cal_y), size=(100, CFG.stream_batch_size))]))

    with open("results/multiseed_rq2.json", encoding="utf-8") as f:
        rq2 = {r["seed"]: r for r in json.load(f)["per_seed"]}

    per_seed = []
    for s in range(N_SEEDS):
        stream = build_stream(te_x, te_y, seed=s)
        records = precompute(model, stream, cal_x, cal_y, qhat_frozen, ref, ece_floor=floor)
        runs_dir = os.path.join(CFG.runs_dir, f"day3-seed{s}")
        # seed_offset 13+s: seed 0 matches Run C's supervisor_no_benefit run exactly
        nb = run_supervisor("supervisor_no_benefit", records=records, model=model, cal_x=cal_x,
                            cal_y=cal_y, qhat_frozen=qhat_frozen, ref=ref, seed_offset=13 + s,
                            use_benefit=False, runs_dir=runs_dir)
        sup_net = rq2[s]["supervisor_net"]
        per_seed.append({"seed": s, "no_benefit_net": nb["net_benefit"], "supervisor_net": sup_net,
                         "removal_cost": sup_net - nb["net_benefit"],
                         "actions": {k: nb[k] for k in ("n_tta", "n_escalate", "n_abstain")}})
        print(f"seed {s}: no_benefit {nb['net_benefit']:+.4f}  supervisor {sup_net:+.4f}  "
              f"removal cost {per_seed[-1]['removal_cost']:+.4f}")

        dst = os.path.join("results", f"seed{s}")
        os.makedirs(dst, exist_ok=True)
        src = os.path.join(runs_dir, "supervisor_no_benefit.jsonl")
        d = os.path.join(dst, "supervisor_no_benefit.jsonl")
        if os.path.exists(d):
            with open(src, "rb") as f1, open(d, "rb") as f2:
                same = f1.read() == f2.read()
            print(f"  seed{s}/supervisor_no_benefit.jsonl: "
                  f"{'identical to existing archive' if same else 'DIFFERS from existing archive -- kept both'}")
            if not same:
                shutil.copy2(src, d + ".day3")
        else:
            shutil.copy2(src, d)

    nb_nets = [r["no_benefit_net"] for r in per_seed]
    rc = [r["removal_cost"] for r in per_seed]
    agg = {"n_seeds": N_SEEDS, "per_seed": per_seed,
           "no_benefit": {"mean": float(np.mean(nb_nets)), "std": float(np.std(nb_nets, ddof=1))},
           "removal_cost": {"mean": float(np.mean(rc)), "std": float(np.std(rc, ddof=1))}}
    with open("results/multiseed_no_benefit.json", "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=1)
    return agg


def cost_sensitivity():
    with open("results/seed0/always_tta.jsonl", encoding="utf-8") as f:
        rows = [json.loads(l) for l in f]
    with open("results/seed0/summary.json", encoding="utf-8") as f:
        meta = json.load(f)["meta"]
    floor = meta["clean_batch_ece_floor"]
    with open("results/gate_tta_sweep.json", encoding="utf-8") as f:
        gate = json.load(f)
    best_gate_ece = max(r["ece_before"] - r["ece_after"] for r in gate["rows"] if r["corruption"] != "clean")

    tta_b = np.array([r["benefit"] for r in rows])            # committed defn: ece_frozen - ece_adapted
    withhold_b = np.array([r["ece_frozen"] - floor for r in rows])  # committed defn (floor-debiased)
    buckets = np.array([r["bucket"] for r in rows])

    def by_bucket(v):
        return {f"b{b}": float(v[buckets == b].mean()) for b in sorted(set(buckets.tolist()))}

    out = {
        "derived_from": "results/seed0 archived traces (no re-runs; committed costs untouched)",
        "committed_costs": meta["costs"],
        "supervisor_exploration_overhead": 0.0074,   # measured: supervisor net at committed costs vs static 0
        "tta": {"stream_mean_benefit": float(tta_b.mean()), "by_bucket": by_bucket(tta_b),
                "crossover_cost": None if max(by_bucket(tta_b).values()) <= 0 else max(by_bucket(tta_b).values()),
                "gate_best_case_ece_benefit": float(best_gate_ece)},
        "withhold": {"stream_mean_benefit": float(withhold_b.mean()), "by_bucket": by_bucket(withhold_b),
                     "crossover_cost_stream": float(withhold_b.mean()),
                     "crossover_cost_best_bucket": float(max(by_bucket(withhold_b).values()))},
        "note": ("Crossovers are oracle-estimate upper bounds: the cost below which the action's mean "
                 "realized benefit (committed definition) is positive; a learning supervisor additionally "
                 "pays ~0.007 exploration overhead."),
    }
    with open("results/cost_sensitivity.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    return out


def main():
    print("== (a) multi-seed no-benefit ablation ==")
    agg = multiseed_no_benefit()
    print(f"no_benefit net: {agg['no_benefit']['mean']:+.4f} +/- {agg['no_benefit']['std']:.4f}; "
          f"outcome-loop removal cost: {agg['removal_cost']['mean']:+.4f} +/- {agg['removal_cost']['std']:.4f}")
    print("== (b) cost-sensitivity crossovers (analytic, from archived traces) ==")
    cs = cost_sensitivity()
    print(f"TTA: stream mean benefit {cs['tta']['stream_mean_benefit']:+.4f}, by bucket {cs['tta']['by_bucket']}, "
          f"crossover: {cs['tta']['crossover_cost']}")
    print(f"withhold: stream crossover {cs['withhold']['crossover_cost_stream']:.4f}, "
          f"best-bucket crossover {cs['withhold']['crossover_cost_best_bucket']:.4f}, by bucket {cs['withhold']['by_bucket']}")


if __name__ == "__main__":
    main()
