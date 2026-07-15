"""Multi-seed RQ2 robustness run (author-ordered scope, memory.md D28).

5 seeds vary ONLY the stream realization (corruption schedule, batch draws, monitoring
subset) and the policy RNGs (epsilon-exploration draws). The trained ViT, temperature,
calibration split, conformal threshold, drift reference, and ECE floor are held fixed
at the base configuration -- isolating the claim "given the same classifier, does
bucketed outcome memory beat the memoryless bandit across stream realizations?".
Methods: supervisor (bonus OFF headline config) and bandit ONLY. No cost, severity,
model, or benefit-definition changes. Seed 0 must reproduce Run C exactly and is
byte-compared against the committed results/seed0/ archive rather than overwritten.

Usage: uv run python -m ars.multiseed
"""
import json
import os
import shutil

import numpy as np
import torch

from .baselines import RidgeBandit
from .config import CFG
from .conformal import conformal_qhat, ece, prediction_sets, set_sizes, softmax_probs
from .data import build_stream, calibration_split, load_splits, normalize
from .model import TemperatureScaled, fit_temperature, load_frozen
from .reproduce import run_supervisor
from .runner import precompute, run_method
from .signals import fit_drift_reference

N_SEEDS = 5


def main():
    torch.manual_seed(CFG.seed)
    np.random.seed(CFG.seed)
    splits = load_splits()
    te_x, te_y = splits["test"]

    # deployed system fixed across seeds: calibration split, temperature, q_hat,
    # drift reference, ECE floor -- all from the base seed, identical to Run C
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
    print(f"fixed deployed system: T={temperature:.3f} qhat={qhat_frozen:.4f} floor={floor:.4f}")

    per_seed = []
    for s in range(N_SEEDS):
        stream = build_stream(te_x, te_y, seed=s)
        records = precompute(model, stream, cal_x, cal_y, qhat_frozen, ref, ece_floor=floor)
        runs_dir = os.path.join(CFG.runs_dir, f"multiseed-{s}")
        # rng offsets chosen so seed 0 reproduces Run C exactly (supervisor 10, bandit 20)
        sup = run_supervisor("supervisor", records=records, model=model, cal_x=cal_x, cal_y=cal_y,
                             qhat_frozen=qhat_frozen, ref=ref, seed_offset=10 + s, runs_dir=runs_dir)
        band = RidgeBandit(np.random.default_rng(20 + s))
        bnd = run_method("bandit", records, band.decide, qhat_frozen, learner=band.learn,
                         runs_dir=runs_dir)
        row = {"seed": s, "supervisor_net": sup["net_benefit"], "bandit_net": bnd["net_benefit"],
               "delta": sup["net_benefit"] - bnd["net_benefit"],
               "supervisor_actions": {k: sup[k] for k in ("n_tta", "n_escalate", "n_abstain")},
               "bandit_actions": {k: bnd[k] for k in ("n_tta", "n_escalate", "n_abstain")}}
        per_seed.append(row)
        print(f"seed {s}: supervisor {row['supervisor_net']:+.4f}  bandit {row['bandit_net']:+.4f}  "
              f"delta {row['delta']:+.4f}")

        # archive traces; never overwrite the committed Run C seed-0 archive -- verify instead
        dst = os.path.join("results", f"seed{s}")
        os.makedirs(dst, exist_ok=True)
        for fn in os.listdir(runs_dir):
            src, d = os.path.join(runs_dir, fn), os.path.join(dst, fn)
            if os.path.exists(d):
                with open(src, "rb") as f1, open(d, "rb") as f2:
                    same = f1.read() == f2.read()
                print(f"  seed{s}/{fn}: {'identical to existing archive' if same else 'DIFFERS from existing archive -- kept both'}")
                if not same:
                    shutil.copy2(src, d + ".multiseed")
            else:
                shutil.copy2(src, d)
        if s != 0:
            with open(os.path.join(dst, "summary.json"), "w", encoding="utf-8") as f:
                json.dump({"summaries": [sup, bnd],
                           "meta": {"seed": s, "note": "multiseed RQ2 run: stream + policy RNG vary; "
                                    "classifier/calibration fixed at base config"}}, f, indent=1)

    sup_nets = [r["supervisor_net"] for r in per_seed]
    bnd_nets = [r["bandit_net"] for r in per_seed]
    deltas = [r["delta"] for r in per_seed]
    agg = {
        "n_seeds": N_SEEDS,
        "held_fixed": "trained ViT, temperature, calibration split, q_hat, drift reference, ECE floor",
        "varied": "stream realization (corruption schedule, batch draws, monitoring subset) + policy RNGs",
        "per_seed": per_seed,
        "supervisor": {"mean": float(np.mean(sup_nets)), "std": float(np.std(sup_nets, ddof=1))},
        "bandit": {"mean": float(np.mean(bnd_nets)), "std": float(np.std(bnd_nets, ddof=1))},
        "delta": {"mean": float(np.mean(deltas)), "std": float(np.std(deltas, ddof=1)),
                  "n_positive": int(sum(d > 0 for d in deltas))},
    }
    with open(os.path.join("results", "multiseed_rq2.json"), "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=1)

    print("\nseed  supervisor_net  bandit_net  delta(sup-bandit)")
    for r in per_seed:
        print(f"{r['seed']:>4}  {r['supervisor_net']:>+13.4f}  {r['bandit_net']:>+10.4f}  {r['delta']:>+17.4f}")
    print(f"mean  {agg['supervisor']['mean']:>+13.4f}  {agg['bandit']['mean']:>+10.4f}  {agg['delta']['mean']:>+17.4f}")
    print(f" std  {agg['supervisor']['std']:>13.4f}  {agg['bandit']['std']:>10.4f}  {agg['delta']['std']:>17.4f}")


if __name__ == "__main__":
    main()
