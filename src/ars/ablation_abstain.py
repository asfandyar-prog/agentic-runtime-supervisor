"""Post hoc b2->ABSTAIN variant of the no-benefit/cost ablation (memory.md D31).

Without the estimator loop, ESCALATE is strictly dominated by ABSTAIN (same benefit,
higher cost), so the pre-registered fixed map (b2->ESCALATE) may overstate what removing
the outcome loop costs. This runs the b2->ABSTAIN fixed map over the same 5 seeds, same
streams, same seed offsets as Day 3a (D30). The specified variant is kept as
pre-registered: it is re-run here only into a scratch dir and byte-compared against the
committed archive, never overwritten.

Usage: uv run python -m ars.ablation_abstain
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
NAME = "supervisor_no_benefit_abstain"


def main():
    # identical setup to day3.multiseed_no_benefit
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
    with open("results/multiseed_no_benefit.json", encoding="utf-8") as f:
        spec = {r["seed"]: r for r in json.load(f)["per_seed"]}

    per_seed = []
    for s in range(N_SEEDS):
        stream = build_stream(te_x, te_y, seed=s)
        records = precompute(model, stream, cal_x, cal_y, qhat_frozen, ref, ece_floor=floor)
        runs_dir = os.path.join(CFG.runs_dir, f"d31-seed{s}")
        args = dict(records=records, model=model, cal_x=cal_x, cal_y=cal_y,
                    qhat_frozen=qhat_frozen, ref=ref, seed_offset=13 + s, use_benefit=False,
                    runs_dir=runs_dir)

        # sanity: the pre-registered variant is unchanged by the new flag
        run_supervisor("supervisor_no_benefit", **args)
        with open(os.path.join(runs_dir, "supervisor_no_benefit.jsonl"), "rb") as f1, \
                open(os.path.join("results", f"seed{s}", "supervisor_no_benefit.jsonl"), "rb") as f2:
            assert f1.read() == f2.read(), f"seed {s}: specified ablation no longer matches archive"

        ab = run_supervisor(NAME, no_benefit_b2="ABSTAIN", **args)
        sup_net = rq2[s]["supervisor_net"]
        per_seed.append({"seed": s, "abstain_variant_net": ab["net_benefit"], "supervisor_net": sup_net,
                         "removal_cost": sup_net - ab["net_benefit"],
                         "specified_net": spec[s]["no_benefit_net"],
                         "specified_removal_cost": spec[s]["removal_cost"],
                         "actions": {k: ab[k] for k in ("n_tta", "n_escalate", "n_abstain")}})
        print(f"seed {s}: specified {spec[s]['no_benefit_net']:+.4f} (gap {spec[s]['removal_cost']:+.4f})  "
              f"abstain-variant {ab['net_benefit']:+.4f} (gap {per_seed[-1]['removal_cost']:+.4f})  "
              f"specified archive byte-identical")
        dst = os.path.join("results", f"seed{s}")
        shutil.copy2(os.path.join(runs_dir, f"{NAME}.jsonl"), os.path.join(dst, f"{NAME}.jsonl"))

    nets = [r["abstain_variant_net"] for r in per_seed]
    rc = [r["removal_cost"] for r in per_seed]
    agg = {"n_seeds": N_SEEDS, "variant": "no-benefit fixed map with b2->ABSTAIN (post hoc, D31)",
           "per_seed": per_seed,
           "abstain_variant": {"mean": float(np.mean(nets)), "std": float(np.std(nets, ddof=1))},
           "removal_cost": {"mean": float(np.mean(rc)), "std": float(np.std(rc, ddof=1))}}
    with open("results/multiseed_no_benefit_abstain.json", "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=1)
    print(f"abstain variant net {agg['abstain_variant']['mean']:+.4f} +/- {agg['abstain_variant']['std']:.4f}; "
          f"gap vs supervisor {agg['removal_cost']['mean']:+.4f} +/- {agg['removal_cost']['std']:.4f}")


if __name__ == "__main__":
    main()
