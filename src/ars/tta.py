"""Episodic LayerNorm-only test-time adaptation (Tent-style) + adapt-then-recalibrate.

EPISODIC: LayerNorm affine params (gamma/beta) are snapshotted, adapted on the current
batch by entropy minimization, used for that batch's predictions, then RESTORED before
returning. The deployed model never drifts across batches; continual TTA is future work
(docs/architecture.md #6). Restoration is asserted bit-identical by the caller in
runner.precompute (Phase 3 checkpoint).

Adapt-then-recalibrate: while adapted, the frozen labeled calibration set is rerun to
recompute LAC scores and a fresh q_hat used only for this batch's prediction sets.
The calibration set is clean while the batch is shifted, so exchangeability is NOT
restored -- flagged simplification, measured by RQ3, not a fix.

Adaptation is deterministic given (frozen weights, batch): entropy minimization has no
random sampling. This is what makes counterfactual replay and the precompute cache
exactly equivalent to a live run (memory.md D19).
"""
import numpy as np
import torch
import torch.nn as nn

from .config import CFG
from .conformal import conformal_qhat, softmax_probs


def _ln_params(model) -> list[torch.nn.Parameter]:
    out = []
    for m in model.modules():
        if isinstance(m, nn.LayerNorm):
            out += [m.weight, m.bias]
    return out


def adapt_batch(model, x: torch.Tensor, cal_x: torch.Tensor, cal_y: torch.Tensor,
                ) -> tuple[np.ndarray, float]:
    """Run episodic TTA on one batch. Returns (adapted softmax probs for x,
    q_hat recalibrated under the adapted model). Model state is restored on exit."""
    params = _ln_params(model)
    snapshot = [p.detach().clone() for p in params]
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.Adam(params, lr=CFG.tta_lr)

    was_training = model.training
    model.eval()  # LayerNorm has no batch stats; eval() just guards any future dropout
    for _ in range(CFG.tta_steps):
        opt.zero_grad()
        logp = torch.log_softmax(model(x), dim=1)
        entropy = -(logp.exp() * logp).sum(dim=1).mean()
        entropy.backward()
        opt.step()

    with torch.no_grad():
        probs = softmax_probs(model(x))
        cal_probs = softmax_probs(model(cal_x))
    qhat_adapted = conformal_qhat(cal_probs, cal_y.numpy())

    # revert: episodic by construction
    with torch.no_grad():
        for p, s in zip(params, snapshot):
            p.copy_(s)
    for p in params:
        p.requires_grad_(False)
    model.train(was_training)
    return probs, qhat_adapted
