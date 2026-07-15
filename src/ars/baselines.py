"""The five comparators (expose RQ1 + RQ2) and the supervisor's run adapters.

- static_raw:   raw frozen classifier, no conformal machinery (deployment status quo)
- never_tta:    conformal-wrapped frozen classifier, always ACCEPT
- always_tta:   TTA + adapt-then-recalibrate every batch (norecal variant = RQ3 contrast)
- monitor:      fixed drift-threshold rule, grid-tuned ON THE EVAL STREAM -- a
                deliberately oracle-tuned upper bound for static policies (design.md #9);
                this favors the baseline and makes RQ1 harder for the supervisor.
- bandit:       memoryless contextual bandit (RQ2 comparator, required): epsilon-greedy,
                per-action linear reward model over the raw current signals
                [1, drift, set_size], incremental ridge, SAME gated feedback and SAME
                counterfactual replay as the supervisor -- the only difference under
                test is bucketed outcome memory vs a stationary linear surface.
"""
import numpy as np

from .config import CFG
from .policy import ACTIONS, PolicyFeedback
from .runner import BatchRecord, benefit


def accept_decider(rec: BatchRecord):
    return "ACCEPT", {}


def always_tta_decider(rec: BatchRecord):
    return "TTA", {}


def tune_monitor(records: list[BatchRecord]) -> tuple[float, float, float]:
    """Grid-search (tau_tta, tau_esc) on the evaluation stream, maximizing the same
    net-benefit metric the table reports. Oracle-tuned by design; disclosed."""
    drifts = np.array([r.drift for r in records])
    cands = list(np.unique(np.quantile(drifts, np.linspace(0.05, 0.95, 19)))) + [np.inf]
    best = (np.inf, np.inf, -np.inf)
    for t1 in cands:
        for t2 in cands:
            if t2 < t1:
                continue
            net = 0.0
            for r in records:
                a = "ACCEPT" if r.drift <= t1 else ("TTA" if r.drift <= t2 else "ESCALATE")
                net += benefit(a, r) - CFG.cost[a]
            if net / len(records) > best[2]:
                best = (float(t1), float(t2), net / len(records))
    return best


def monitor_decider(tau_tta: float, tau_esc: float):
    def decide(rec: BatchRecord):
        a = "ACCEPT" if rec.drift <= tau_tta else ("TTA" if rec.drift <= tau_esc else "ESCALATE")
        return a, {"tau_tta": round(tau_tta, 4), "tau_esc": round(tau_esc, 4)}
    return decide


class RidgeBandit:
    """Memoryless: the decision is a fixed parametric function of the CURRENT batch's
    signals only -- no buckets, no per-bucket label counts, no history features.
    (It still learns online; any bandit does. What it lacks is intervention memory.)"""

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        d = 3
        self.A = {a: CFG.ridge_lambda * np.eye(d) for a in ACTIONS}
        self.bvec = {a: np.zeros(d) for a in ACTIONS}

    @staticmethod
    def features(rec: BatchRecord) -> np.ndarray:
        return np.array([1.0, rec.drift, rec.mean_set_size])

    def decide(self, rec: BatchRecord):
        x = self.features(rec)
        pred = {a: float(x @ np.linalg.solve(self.A[a], self.bvec[a])) for a in ACTIONS}
        explored = bool(self.rng.random() < CFG.epsilon)
        if explored:
            action = ACTIONS[int(self.rng.integers(len(ACTIONS)))]
        else:
            action = max(ACTIONS, key=lambda a: (pred[a], -CFG.cost[a]))
        return action, {"predicted_reward": {a: round(v, 4) for a, v in pred.items()},
                        "explored": explored}

    def learn(self, fb: PolicyFeedback, rec: BatchRecord) -> None:
        # same gated feedback + counterfactual replay granted to the supervisor;
        # reward is net (benefit - cost), matching what decide() maximizes
        x = self.features(rec)
        for a, ben in fb.benefits.items():
            self.A[a] += np.outer(x, x)
            self.bvec[a] += (ben - CFG.cost[a]) * x
