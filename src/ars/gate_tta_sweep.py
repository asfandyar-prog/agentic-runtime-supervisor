"""Day-1 gate for the two-regime extension (author-ordered, memory.md D29).

Question: does episodic LayerNorm-entropy TTA produce REAL accuracy recovery (not just
ECE change) on this ViT in some mild corruption band? Gate passes ONLY if a mild band
shows clear positive delta-accuracy (>= +2-3 points). ECE-only improvement does not pass.

Measurement only. The committed corruption config, costs, model, and benefit definition
are untouched: this script mirrors data.corrupt's formulas at a finer strength grid
(including strengths milder than the committed severity 1) purely to measure the
before/after-TTA accuracy and ECE surface. Also records mean embedding drift per cell,
because a passing band that lands in drift bucket 0 would be indistinguishable from
clean to the supervisor -- a Day-2 feasibility fact.

Usage: uv run python -m ars.gate_tta_sweep
"""
import json
import os

import numpy as np
import torch
import torchvision.transforms.functional as TF

from .config import CFG
from .conformal import conformal_qhat, ece, prediction_sets, set_sizes, softmax_probs
from .data import calibration_split, load_splits, normalize
from .model import TemperatureScaled, fit_temperature, load_frozen
from .signals import fit_drift_reference
from .tta import adapt_batch

# sweep-only strength grids; the three committed severities per corruption are included
# as anchors (noise 0.15/0.25/0.40, blur 0.8/1.5/2.5, jitter s=1/2/3)
GRID = {
    "gaussian_noise": [0.02, 0.05, 0.08, 0.12, 0.15, 0.25, 0.40],
    "gaussian_blur": [0.3, 0.5, 0.8, 1.2, 1.5, 2.0, 2.5],
    "color_jitter": [0.25, 0.5, 1.0, 1.5, 2.0, 3.0],
}
N_BATCHES = 16  # 16 x 64 = 1024 held-out samples per cell


def apply_corruption(x: torch.Tensor, kind: str, strength: float, gen: torch.Generator) -> torch.Tensor:
    """Mirror of data.corrupt with continuous strength (measurement only)."""
    if kind == "clean":
        return x
    if kind == "gaussian_noise":
        return (x + torch.randn(x.shape, generator=gen) * strength).clamp(0.0, 1.0)
    if kind == "gaussian_blur":
        return TF.gaussian_blur(x, kernel_size=5, sigma=strength)
    if kind == "color_jitter":
        s = strength
        x = TF.adjust_brightness(x, 1.0 + 0.30 * s)
        x = TF.adjust_contrast(x, 1.0 - 0.22 * s)
        x = TF.adjust_saturation(x, 1.0 + 0.50 * s)
        return TF.adjust_hue(x, min(0.05 * s, 0.5))
    raise ValueError(kind)


def main():
    torch.manual_seed(CFG.seed)
    splits = load_splits()
    te_x, te_y = splits["test"]
    cal_x, cal_y = calibration_split(*splits["val"])
    cal_x = normalize(cal_x)

    # deployed model exactly as the pipeline builds it (frozen ViT + fitted temperature)
    temperature = fit_temperature(load_frozen(), cal_x, cal_y)
    model = TemperatureScaled(load_frozen(), temperature)
    with torch.no_grad():
        cal_probs = softmax_probs(model(cal_x))
    qhat = conformal_qhat(cal_probs, cal_y.numpy())
    clean_mss = float(set_sizes(prediction_sets(cal_probs, qhat)).mean())
    ref = fit_drift_reference(model, cal_x, clean_mss)
    edges = (ref.q95_clean, CFG.drift_edge_factor * ref.q95_clean)
    print(f"deployed model: T={temperature:.3f}; drift bucket edges b0<={edges[0]:.4f}<b1<={edges[1]:.4f}<b2")

    rng = np.random.default_rng(42)
    gen = torch.Generator().manual_seed(42)

    def eval_cell(kind: str, strength: float) -> dict:
        accs_b, accs_a, probs_b, probs_a, ys, drifts = [], [], [], [], [], []
        for _ in range(N_BATCHES):
            idx = torch.from_numpy(rng.integers(0, len(te_x), size=CFG.stream_batch_size))
            x = normalize(apply_corruption(te_x[idx], kind, strength, gen))
            y = te_y[idx].numpy()
            with torch.no_grad():
                emb = model.forward_features(x).numpy().astype(np.float64)
                pb = softmax_probs(model(x))
            pa, _ = adapt_batch(model, x, cal_x, cal_y)  # episodic; reverts internally
            drifts.append(ref.drift_value(emb))
            probs_b.append(pb)
            probs_a.append(pa)
            ys.append(y)
        P_b, P_a, Y = np.concatenate(probs_b), np.concatenate(probs_a), np.concatenate(ys)
        drift = float(np.mean(drifts))
        return {
            "corruption": kind, "strength": strength, "n": len(Y),
            "acc_before": float((P_b.argmax(1) == Y).mean()),
            "acc_after": float((P_a.argmax(1) == Y).mean()),
            "ece_before": ece(P_b, Y), "ece_after": ece(P_a, Y),
            "drift": drift, "bucket": ref.bucket(drift),
        }

    rows = [eval_cell("clean", 0.0)]
    for kind, strengths in GRID.items():
        for s in strengths:
            rows.append(eval_cell(kind, s))
            r = rows[-1]
            print(f"{kind:15s} s={s:<5} acc {r['acc_before']:.3f}->{r['acc_after']:.3f} "
                  f"({(r['acc_after'] - r['acc_before']) * 100:+.1f}pt)  "
                  f"ece {r['ece_before']:.3f}->{r['ece_after']:.3f}  drift {r['drift']:.3f} b{r['bucket']}")

    os.makedirs("results", exist_ok=True)
    out = {"n_batches_per_cell": N_BATCHES, "batch_size": CFG.stream_batch_size,
           "temperature": temperature, "bucket_edges": list(edges), "rows": rows,
           "note": "Day-1 gate sweep (D29): measurement only; committed config untouched"}
    with open("results/gate_tta_sweep.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)

    print("\ncorruption      strength  n     acc_before acc_after  d_acc(pt)  ece_before ece_after  d_ece    drift  bucket")
    for r in rows:
        print(f"{r['corruption']:15s} {r['strength']:<8} {r['n']:<5} "
              f"{r['acc_before']:<10.4f} {r['acc_after']:<10.4f} {(r['acc_after'] - r['acc_before']) * 100:>+8.1f}  "
              f"{r['ece_before']:<10.4f} {r['ece_after']:<10.4f} {r['ece_after'] - r['ece_before']:>+7.4f}  "
              f"{r['drift']:<6.3f} b{r['bucket']}")


if __name__ == "__main__":
    main()
