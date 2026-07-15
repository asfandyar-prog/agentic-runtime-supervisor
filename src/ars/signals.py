"""Label-free decision-time signals (design.md #2).

Signal 1 -- embedding drift vs the clean calibration reference:
  reference = per-dim mean/std of frozen-model embeddings over the clean calibration set;
  drift(batch) = || (mean batch embedding - mu_ref) / sigma_ref ||_2 / sqrt(d).
  Buckets: b0 <= q95_clean < b1 <= 3*q95_clean < b2, where q95_clean comes from
  bootstrapped clean calibration batches. Coarse and fixed a priori (memory.md D5).

Signal 2 -- conformal set size, used as an ACCEPT gate (design.md #2.2): inflated sets
mean the coverage guarantee is already visibly strained, so shipping as-if-normal is
the one clearly indefensible act. The bucket stays drift-only to keep the estimator at
4 actions x 3 buckets = 12 cells that a 200-batch stream can actually populate.

Signal 3 (expected benefit minus cost) lives in policy.py: it is a function of past
gated outcomes only, so it is label-free at decision time.
"""
from dataclasses import dataclass

import numpy as np
import torch

from .config import CFG


@dataclass
class DriftReference:
    mu: np.ndarray
    sigma: np.ndarray
    q95_clean: float
    clean_mean_set_size: float

    def drift_value(self, emb: np.ndarray) -> float:
        z = (emb.mean(axis=0) - self.mu) / self.sigma
        return float(np.linalg.norm(z) / np.sqrt(len(z)))

    def bucket(self, drift: float) -> int:
        if drift <= self.q95_clean:
            return 0
        if drift <= CFG.drift_edge_factor * self.q95_clean:
            return 1
        return 2

    def setsize_inflated(self, mean_set_size: float) -> bool:
        return mean_set_size > CFG.setsize_gate_factor * self.clean_mean_set_size


def fit_drift_reference(model, cal_x: torch.Tensor, clean_mean_set_size: float,
                        seed: int = CFG.seed) -> DriftReference:
    with torch.no_grad():
        emb = model.forward_features(cal_x).numpy().astype(np.float64)
    mu, sigma = emb.mean(axis=0), emb.std(axis=0) + 1e-8

    rng = np.random.default_rng(seed + 3)
    drifts = []
    for _ in range(CFG.drift_bootstrap_batches):
        idx = rng.integers(0, len(emb), size=CFG.stream_batch_size)
        z = (emb[idx].mean(axis=0) - mu) / sigma
        drifts.append(np.linalg.norm(z) / np.sqrt(emb.shape[1]))
    return DriftReference(mu=mu, sigma=sigma, q95_clean=float(np.quantile(drifts, 0.95)),
                          clean_mean_set_size=clean_mean_set_size)
