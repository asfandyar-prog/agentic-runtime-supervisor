"""All experiment constants in one place. Every value is a disclosed default (docs/design.md).

Nothing in here is tuned by sweeps; the ordering of the action costs, not their
magnitudes, carries the argument (design.md D9).
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Config:
    seed: int = 0

    # paths
    data_root: str = "data"
    checkpoint_path: str = "checkpoints/vit.pt"
    runs_dir: str = "runs"

    # model: tiny ViT. Must be a ViT: LayerNorm-only TTA needs LayerNorm layers.
    img_size: int = 28
    patch_size: int = 4
    embed_dim: int = 96
    depth: int = 4
    n_heads: int = 4
    mlp_ratio: float = 2.0
    n_classes: int = 7

    # training (accuracy is a non-goal beyond "well above chance and calibratable")
    epochs: int = 15
    train_batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 1e-4

    # split-conformal calibration (LAC score), 95% target coverage
    n_cal: int = 500
    alpha: float = 0.05
    ece_bins: int = 10

    # drift reference: bootstrap clean calibration batches -> clean drift quantiles.
    # Bucket edges q95 and 3*q95: the 3x factor is arbitrary-but-fixed a priori (memory.md D5).
    drift_bootstrap_batches: int = 200
    drift_edge_factor: float = 3.0
    n_buckets: int = 3

    # conformal set-size gate: inflated sets mean the calibration guarantee is
    # visibly strained -> ACCEPT is disallowed for the batch (design.md #2.2)
    setsize_gate_factor: float = 1.5

    # episodic LayerNorm-only TTA (Tent-style; gamma/beta only, then revert)
    tta_lr: float = 1e-3
    tta_steps: int = 3

    # stream: 200 batches x 64, 10-batch regime blocks, ~40% clean blocks
    n_stream_batches: int = 200
    stream_batch_size: int = 64
    block_size: int = 10
    p_clean_block: float = 0.4
    monitoring_frac: float = 0.10  # fixed labeled monitoring subset (user-approved, PRD #6.1)

    # policy: fixed per-action intervention costs in ECE units.
    # Ordering is the content: free < cheap compute < lost throughput < human time.
    cost: dict = field(default_factory=lambda: {
        "ACCEPT": 0.00, "TTA": 0.02, "ABSTAIN": 0.06, "ESCALATE": 0.15,
    })
    epsilon: float = 0.1  # exploration; without it, gated feedback + greedy never populates estimates

    # escalation VoI bonus: OFF by default -- the RQ2 headline is bonus-off vs bandit
    # (design.md #5; memory.md D17). Bonus-ON is its own reported ablation.
    escalation_bonus: bool = False
    bonus_beta: float = 0.05

    # bandit comparator (RQ2): epsilon-greedy incremental ridge over [1, drift, set_size]
    ridge_lambda: float = 1.0


CFG = Config()
