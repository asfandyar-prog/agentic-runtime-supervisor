"""Split-conformal prediction with the LAC score, plus ECE.

LAC nonconformity: s(x, y) = 1 - p_y(x). Prediction set = {y : p_y(x) >= 1 - q_hat}.
q_hat is the ceil((n+1)(1-alpha))/n empirical quantile of calibration scores.
Assumes exchangeability under a FIXED score function -- which TTA breaks; the
adapt-then-recalibrate step (tta.py) recomputes q_hat under the adapted model and is
a flagged simplification, not a fix (design.md #10.3, RQ3).
"""
import numpy as np

from .config import CFG


def softmax_probs(logits) -> np.ndarray:
    z = logits.detach().cpu().numpy().astype(np.float64)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def conformal_qhat(cal_probs: np.ndarray, cal_y: np.ndarray, alpha: float = CFG.alpha) -> float:
    scores = 1.0 - cal_probs[np.arange(len(cal_y)), cal_y]
    n = len(scores)
    level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(scores, level, method="higher"))


def prediction_sets(probs: np.ndarray, qhat: float) -> np.ndarray:
    """Boolean membership mask (N, C). Sets may be empty (raw LAC; disclosed)."""
    return probs >= 1.0 - qhat


def set_sizes(sets: np.ndarray) -> np.ndarray:
    return sets.sum(axis=1)


def covered(sets: np.ndarray, y: np.ndarray) -> np.ndarray:
    return sets[np.arange(len(y)), y]


def ece(probs: np.ndarray, y: np.ndarray, n_bins: int = CFG.ece_bins) -> float:
    """Expected calibration error, equal-width confidence bins."""
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out, n = 0.0, len(y)
    for i in range(n_bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1]) if i > 0 else (conf <= edges[1])
        if m.any():
            out += (m.sum() / n) * abs(correct[m].mean() - conf[m].mean())
    return float(out)
