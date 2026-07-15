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


def tune_monitor(records: list[BatchRecord]) -> tuple[float, float, str, str, float]:
    """Grid-search a static drift-threshold rule on the evaluation stream, maximizing the
    same net-benefit metric the table reports. Oracle-tuned by design; disclosed.
    The grid picks thresholds (tau1 <= tau2) AND the mid/high actions from the full
    intervention vocabulary {TTA, ABSTAIN, ESCALATE} -- otherwise the monitor is denied
    the action that may win and stops being an upper bound for static policies
    (memory.md D23). Low region is always ACCEPT; inf thresholds allow one-threshold
    and never-intervene variants, so the expose's single-threshold monitor is inside
    the search space."""
    drifts = np.array([r.drift for r in records])
    cands = list(np.unique(np.quantile(drifts, np.linspace(0.05, 0.95, 19)))) + [np.inf]
    interventions = ("TTA", "ABSTAIN", "ESCALATE")
    best = (np.inf, np.inf, "TTA", "ESCALATE", -np.inf)
    for t1 in cands:
        for t2 in cands:
            if t2 < t1:
                continue
            for mid in interventions:
                for high in interventions:
                    net = 0.0
                    for r in records:
                        a = "ACCEPT" if r.drift <= t1 else (mid if r.drift <= t2 else high)
                        net += benefit(a, r) - CFG.cost[a]
                    if net / len(records) > best[4]:
                        best = (float(t1), float(t2), mid, high, net / len(records))
    return best


def monitor_decider(tau1: float, tau2: float, mid: str, high: str):
    def decide(rec: BatchRecord):
        a = "ACCEPT" if rec.drift <= tau1 else (mid if rec.drift <= tau2 else high)
        return a, {"rule": f"ACCEPT<={tau1:.3f}<{mid}<={tau2:.3f}<{high}"}
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
