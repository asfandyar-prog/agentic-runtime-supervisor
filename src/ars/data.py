"""DermaMNIST splits, synthetic corruptions, and the evaluation stream.

Splits (docs/architecture.md #3):
  train (7,007)  -> ViT training only
  val   (1,003)  -> first n_cal=500 (seed-shuffled) = clean calibration set,
                    which doubles as the drift reference (per spec); rest unused
  test  (2,005)  -> stream pool; images drawn WITH reuse to build ~200 batches (disclosed)
"""
import os
from dataclasses import dataclass

import numpy as np
import torch
import torchvision.transforms.functional as TF
from medmnist import DermaMNIST

from .config import CFG

CORRUPTIONS = ("gaussian_noise", "gaussian_blur", "color_jitter")


def _to_float(imgs: np.ndarray) -> torch.Tensor:
    """(N,28,28,3) uint8 -> (N,3,28,28) float in [0,1]."""
    return torch.from_numpy(imgs).permute(0, 3, 1, 2).float() / 255.0


def normalize(x: torch.Tensor) -> torch.Tensor:
    return (x - 0.5) / 0.5


def load_splits(root: str = CFG.data_root):
    os.makedirs(root, exist_ok=True)
    tr = DermaMNIST(split="train", download=True, root=root)
    va = DermaMNIST(split="val", download=True, root=root)
    te = DermaMNIST(split="test", download=True, root=root)
    out = {}
    for name, ds in (("train", tr), ("val", va), ("test", te)):
        out[name] = (_to_float(ds.imgs), torch.from_numpy(ds.labels).long().squeeze(1))
    return out


def calibration_split(val_x: torch.Tensor, val_y: torch.Tensor, seed: int = CFG.seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(val_x))[: CFG.n_cal]
    return val_x[idx], val_y[idx]


def corrupt(x: torch.Tensor, name: str, severity: int, gen: torch.Generator) -> torch.Tensor:
    """Apply one corruption at severity 1-3 to a [0,1] image batch. Deterministic given gen."""
    s = severity
    if name == "clean":
        return x
    # severities tuned once on a 500-image probe so shift actually damages calibration
    # (noise sev2 halves accuracy; jitter sev2 triples ECE); blur is deliberately kept
    # benign -- it moves embeddings without hurting calibration, giving drift-threshold
    # rules a false-positive regime (memory.md D25)
    if name == "gaussian_noise":
        sigma = [0.15, 0.25, 0.40][s - 1]
        noise = torch.randn(x.shape, generator=gen) * sigma
        return (x + noise).clamp(0.0, 1.0)
    if name == "gaussian_blur":
        sigma = [0.8, 1.5, 2.5][s - 1]
        return TF.gaussian_blur(x, kernel_size=5, sigma=sigma)
    if name == "color_jitter":
        # deterministic per severity (no random sampling): shift brightness/contrast/sat/hue
        x = TF.adjust_brightness(x, 1.0 + 0.30 * s)
        x = TF.adjust_contrast(x, 1.0 - 0.22 * s)
        x = TF.adjust_saturation(x, 1.0 + 0.50 * s)
        return TF.adjust_hue(x, 0.05 * s)
    raise ValueError(name)


@dataclass
class StreamBatch:
    batch_id: int
    x: torch.Tensor          # normalized, ready for the model
    y: torch.Tensor          # ground truth: harness-only; the policy sees it only via gating
    corruption: str          # e.g. "gaussian_noise:2" or "clean"; logged, never shown to policies
    monitored: bool          # True -> labels revealed to the policy AFTER the action executes


def build_stream(test_x: torch.Tensor, test_y: torch.Tensor, seed: int = CFG.seed) -> list[StreamBatch]:
    """~200 batches in 10-batch regime blocks (~40% clean). Temporally-correlated drift is
    realistic and favors memoryful policies -- both faces disclosed (design.md #10.4)."""
    rng = np.random.default_rng(seed + 1)
    gen = torch.Generator().manual_seed(seed + 2)
    n_blocks = CFG.n_stream_batches // CFG.block_size

    regimes = []
    for _ in range(n_blocks):
        if rng.random() < CFG.p_clean_block:
            regimes.append(("clean", 0))
        else:
            regimes.append((CORRUPTIONS[rng.integers(len(CORRUPTIONS))], int(rng.integers(1, 4))))

    # monitoring batches: fixed seeded 10% subset, NOT block-aligned (avoids the artifact of
    # always monitoring a block's first batch)
    n_mon = int(CFG.n_stream_batches * CFG.monitoring_frac)
    monitored = set(rng.choice(CFG.n_stream_batches, size=n_mon, replace=False).tolist())

    batches = []
    for b in range(CFG.n_stream_batches):
        name, sev = regimes[b // CFG.block_size]
        idx = torch.from_numpy(rng.integers(0, len(test_x), size=CFG.stream_batch_size))
        x = corrupt(test_x[idx], name, sev, gen)
        batches.append(StreamBatch(
            batch_id=b,
            x=normalize(x),
            y=test_y[idx],
            corruption="clean" if name == "clean" else f"{name}:{sev}",
            monitored=b in monitored,
        ))
    return batches
