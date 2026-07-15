"""LangGraph wiring of the supervisor: SignalCollection -> RiskAssessment ->
PolicyDecision -> Action -> OutcomeLogging, invoked once per stream batch.

No LLM anywhere: nodes are plain Python (the expose argues for a stateful,
outcome-updating decision process, "not for a large-language-model agent").

Label hygiene: ground-truth labels never enter the graph state. The OutcomeLogging
node obtains feedback only through the gated provider (runner.gated_feedback), which
is the single label-path boundary (rules.md invariant).

The Action node runs TTA LIVE (this graph is the deployed artifact being demonstrated)
and asserts equality with the precompute cache -- doubling as the determinism proof
that justifies counterfactual replay (memory.md D19).
"""
from dataclasses import dataclass
from typing import Any, TypedDict

import numpy as np
import torch
from langgraph.graph import END, START, StateGraph

from .conformal import prediction_sets, set_sizes, softmax_probs
from .policy import SupervisorPolicy
from .signals import DriftReference
from .tta import adapt_batch


class SupState(TypedDict, total=False):
    batch_id: int
    x: Any                    # torch.Tensor; labels are deliberately NOT in the state
    # SignalCollection
    drift: float
    mean_set_size: float
    probs_frozen: Any
    # RiskAssessment
    bucket: int
    inflated: bool
    # PolicyDecision
    action: str
    explored: bool
    utilities: dict
    reasoning: str
    # Action
    probs_issued: Any         # None when predictions are withheld
    qhat_issued: float | None
    # OutcomeLogging
    feedback_source: str | None
    memory: dict


@dataclass
class GraphContext:
    model: Any
    cal_x: torch.Tensor
    cal_y: torch.Tensor
    qhat_frozen: float
    ref: DriftReference
    policy: SupervisorPolicy
    feedback_provider: Any    # (batch_id, action) -> PolicyFeedback | None (gated)
    cache_lookup: Any         # batch_id -> BatchRecord, for the determinism assert


def build_graph(ctx: GraphContext):
    def signal_collection(state: SupState) -> dict:
        with torch.no_grad():
            emb = ctx.model.forward_features(state["x"]).numpy().astype(np.float64)
            probs = softmax_probs(ctx.model(state["x"]))
        sizes = set_sizes(prediction_sets(probs, ctx.qhat_frozen))
        return {"drift": ctx.ref.drift_value(emb), "mean_set_size": float(sizes.mean()),
                "probs_frozen": probs}

    def risk_assessment(state: SupState) -> dict:
        return {"bucket": ctx.ref.bucket(state["drift"]),
                "inflated": ctx.ref.setsize_inflated(state["mean_set_size"])}

    def policy_decision(state: SupState) -> dict:
        d = ctx.policy.decide(state["bucket"], state["inflated"], state["drift"])
        return {"action": d.action, "explored": d.explored,
                "utilities": d.utilities, "reasoning": d.reasoning}

    def action(state: SupState) -> dict:
        a = state["action"]
        if a == "ACCEPT":
            return {"probs_issued": state["probs_frozen"], "qhat_issued": ctx.qhat_frozen}
        if a == "TTA":
            probs, qhat = adapt_batch(ctx.model, state["x"], ctx.cal_x, ctx.cal_y)
            rec = ctx.cache_lookup(state["batch_id"])
            assert np.allclose(probs, rec.probs_adapted, atol=1e-8) and qhat == rec.qhat_adapted, \
                "live TTA diverged from precompute cache (determinism broken)"
            return {"probs_issued": probs, "qhat_issued": qhat}
        return {"probs_issued": None, "qhat_issued": None}  # ABSTAIN / ESCALATE withhold

    def outcome_logging(state: SupState) -> dict:
        fb = ctx.feedback_provider(state["batch_id"], state["action"])
        if fb is not None:
            ctx.policy.learn(fb)  # delayed: takes effect from the next batch
        return {"feedback_source": fb.source if fb else None,
                "memory": ctx.policy.memory.snapshot()}

    g = StateGraph(SupState)
    g.add_node("SignalCollection", signal_collection)
    g.add_node("RiskAssessment", risk_assessment)
    g.add_node("PolicyDecision", policy_decision)
    g.add_node("Action", action)
    g.add_node("OutcomeLogging", outcome_logging)
    g.add_edge(START, "SignalCollection")
    g.add_edge("SignalCollection", "RiskAssessment")
    g.add_edge("RiskAssessment", "PolicyDecision")
    g.add_edge("PolicyDecision", "Action")
    g.add_edge("Action", "OutcomeLogging")
    g.add_edge("OutcomeLogging", END)
    return g.compile()
