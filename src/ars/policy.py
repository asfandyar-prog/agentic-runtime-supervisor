"""The supervisor policy: an online outcome estimator (docs/design.md, the graded core).

WHY this design, point by point (defended to reviewers):

* NOT a weighted-signal threshold rule. A fixed rule's decision boundary cannot
  re-estimate the benefit of intervening from realized outcomes -- that inability is
  exactly why static rules are the RQ1 baselines rather than the system. The policy
  here maintains, for each of the four actions, a RUNNING FREQUENCY ESTIMATE of
  realized ECE reduction conditioned on a coarse drift bucket, and picks
      argmax_a  Q(a, bucket) - cost(a) [+ optional escalation VoI bonus]
  which is the expose's specification verbatim. A learned policy is thesis work.

* Buckets are drift-only (3 of them), keeping 4x3=12 cells a 200-batch stream can
  populate under gated feedback. Conformal set size instead acts as an ACCEPT gate:
  inflated sets mean the coverage guarantee is visibly strained, so "ship as-if-normal"
  is the one indefensible action (memory.md D5).

* Benefit definitions (design.md #4): ACCEPT = 0 (it IS the do-nothing reference);
  TTA = ECE_frozen - ECE_adapted, signable, because TTA can make a fine model worse
  and the policy must be able to learn that; ABSTAIN = ESCALATE = ECE_frozen minus the
  clean batch-ECE noise floor (miscalibration avoided by not shipping predictions,
  debiased so withholding is not paid for small-sample estimator bias -- D24).

* Because ABSTAIN and ESCALATE share a benefit, the cheaper ABSTAIN always weakly
  dominates -- so deliberate escalation only exists via the VoI bonus
  beta/sqrt(n_labeled(bucket)+1): escalation is how the policy BUYS ground truth when
  its estimates for the current regime are data-poor. The bonus is OFF by default:
  the RQ2 headline must not hand the supervisor a history-defined term the bandit is
  structurally denied (that would assume RQ2's answer, not test it -- memory.md D17).

* epsilon-greedy exploration is mandatory here, not a nicety: feedback is gated, so a
  greedy policy that never tries interventions never receives evidence about them.

* Feedback is DELAYED and ACTION-GATED. PolicyFeedback is the ONLY path into memory,
  and the runner constructs it exclusively from gated sources (escalated batches +
  the fixed labeled monitoring subset). Evaluation metrics never touch memory
  (rules.md label-path invariant, memory.md D18).
"""
from dataclasses import dataclass, field

import numpy as np

from .config import CFG

ACTIONS = ("ACCEPT", "TTA", "ABSTAIN", "ESCALATE")


@dataclass(frozen=True)
class PolicyFeedback:
    """Delayed, gated outcome of one labeled batch. Constructible only from gated label
    sources ('escalation' | 'monitoring'); this dataclass IS the label-path boundary."""
    bucket: int
    source: str
    # realized ECE-reduction benefit per action on this batch, incl. counterfactuals
    # (replay, design.md #6): labels arrived legitimately, so the benefit each action
    # WOULD have had is computable; without replay the estimator gets ~2-5 TTA
    # outcomes per 200-batch stream and cannot function.
    benefits: dict

    def __post_init__(self):
        assert self.source in ("escalation", "monitoring"), \
            f"gated feedback only; got source={self.source!r}"


@dataclass
class OutcomeMemory:
    """The intervention memory that conditions future decisions (expose Fig. 1)."""
    mean: dict = field(default_factory=lambda: {(a, b): 0.0 for a in ACTIONS for b in range(CFG.n_buckets)})
    count: dict = field(default_factory=lambda: {(a, b): 0 for a in ACTIONS for b in range(CFG.n_buckets)})
    labeled_per_bucket: dict = field(default_factory=lambda: {b: 0 for b in range(CFG.n_buckets)})
    action_counts: dict = field(default_factory=lambda: {a: 0 for a in ACTIONS})

    def update(self, fb: PolicyFeedback) -> None:
        # plain running mean -- the expose's "simple frequency estimate". No decay:
        # frozen weights + episodic TTA keep the outcome mapping ~stationary in-run.
        for a, benefit in fb.benefits.items():
            key = (a, fb.bucket)
            self.count[key] += 1
            self.mean[key] += (benefit - self.mean[key]) / self.count[key]
        self.labeled_per_bucket[fb.bucket] += 1

    def snapshot(self) -> dict:
        return {
            "estimates": {f"{a}/b{b}": {"mean": round(self.mean[(a, b)], 4), "n": self.count[(a, b)]}
                          for a in ACTIONS for b in range(CFG.n_buckets)},
            "labeled_per_bucket": dict(self.labeled_per_bucket),
            "action_counts": dict(self.action_counts),
        }


@dataclass
class Decision:
    action: str
    explored: bool
    utilities: dict
    reasoning: str


class SupervisorPolicy:
    """Ablatable via flags: use_drift (bucket conditioning), use_setsize (ACCEPT gate),
    use_benefit (outcome memory; off collapses to the fixed threshold mapping),
    escalation_bonus (VoI term, off in the RQ2 headline)."""

    def __init__(self, rng: np.random.Generator, use_drift: bool = True, use_setsize: bool = True,
                 use_benefit: bool = True, escalation_bonus: bool = CFG.escalation_bonus):
        self.rng = rng
        self.memory = OutcomeMemory()
        self.use_drift = use_drift
        self.use_setsize = use_setsize
        self.use_benefit = use_benefit
        self.escalation_bonus = escalation_bonus

    def _bucket(self, raw_bucket: int) -> int:
        # no-drift ablation: estimator loses shift conditioning (single bucket)
        return raw_bucket if self.use_drift else 0

    def decide(self, raw_bucket: int, inflated: bool, drift: float) -> Decision:
        b = self._bucket(raw_bucket)
        gated = self.use_setsize and inflated
        allowed = [a for a in ACTIONS if not (gated and a == "ACCEPT")]

        if not self.use_benefit:
            # no-benefit/cost ablation: memory off -> the policy collapses to exactly
            # the two-signal fixed rule (what removing the outcome loop leaves behind)
            fixed = {0: "ACCEPT", 1: "TTA", 2: "ESCALATE"}[raw_bucket]
            if fixed == "ACCEPT" and gated:
                fixed = "TTA"
            return Decision(fixed, False, {}, f"ablation no-benefit: fixed map bucket={raw_bucket}"
                                              f"{' +gate' if gated else ''} -> {fixed}")

        utilities = {}
        for a in allowed:
            u = self.memory.mean[(a, b)] - CFG.cost[a]
            if a == "ESCALATE" and self.escalation_bonus:
                u += CFG.bonus_beta / np.sqrt(self.memory.labeled_per_bucket[b] + 1)
            utilities[a] = round(float(u), 4)

        explored = bool(self.rng.random() < CFG.epsilon)
        if explored:
            action = allowed[int(self.rng.integers(len(allowed)))]
        else:
            # ties break toward the cheaper action (ACTIONS is cost-ascending)
            action = max(allowed, key=lambda a: (utilities[a], -CFG.cost[a]))

        self.memory.action_counts[action] += 1
        reason = (f"bucket={b} (drift={drift:.3f}); "
                  + (f"sets inflated -> ACCEPT gated; " if gated else "")
                  + f"utilities={utilities}; "
                  + (f"explored -> {action}" if explored else f"greedy -> {action}"))
        return Decision(action, explored, utilities, reason)

    def learn(self, fb: PolicyFeedback) -> None:
        fb = PolicyFeedback(bucket=self._bucket(fb.bucket), source=fb.source, benefits=fb.benefits)
        self.memory.update(fb)
