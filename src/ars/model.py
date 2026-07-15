"""Tiny ViT for 28x28 DermaMNIST (~0.5M params).

Must be a ViT: LayerNorm-only TTA adapts LayerNorm affine params, which a small CNN
lacks. Accuracy is a non-goal beyond "well above 7-class chance and calibratable".
"""
import torch
import torch.nn as nn

from .config import CFG


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


class TinyViT(nn.Module):
    def __init__(self, cfg=CFG):
        super().__init__()
        n_patches = (cfg.img_size // cfg.patch_size) ** 2
        self.patch = nn.Conv2d(3, cfg.embed_dim, kernel_size=cfg.patch_size, stride=cfg.patch_size)
        self.pos = nn.Parameter(torch.zeros(1, n_patches, cfg.embed_dim))
        self.blocks = nn.ModuleList(Block(cfg.embed_dim, cfg.n_heads, cfg.mlp_ratio) for _ in range(cfg.depth))
        self.norm = nn.LayerNorm(cfg.embed_dim)
        self.head = nn.Linear(cfg.embed_dim, cfg.n_classes)
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Mean-pooled final-layer tokens -- also the embedding used for the drift signal."""
        x = self.patch(x).flatten(2).transpose(1, 2) + self.pos
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x).mean(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))


class TemperatureScaled(nn.Module):
    """The deployed model = frozen ViT + temperature fitted on the calibration set.
    Standard deployment practice, and the expose's premise is a well-calibrated deployed
    model (clean ECE 0.014 in the thesis work); without it the tiny ViT's clean ECE
    (~0.10) exceeds the abstention cost and every policy degenerates to always-abstain
    (memory.md D22)."""

    def __init__(self, inner: nn.Module, temperature: float):
        super().__init__()
        self.inner = inner
        self.temperature = temperature

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner.forward_features(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner(x) / self.temperature


def fit_temperature(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    """1-D NLL grid search on the clean calibration set (labels legitimately available)."""
    with torch.no_grad():
        logits = model(x)
    grid = torch.logspace(-0.7, 0.7, 141)
    nll = [nn.functional.cross_entropy(logits / t, y).item() for t in grid]
    return float(grid[int(torch.tensor(nll).argmin())])


def load_frozen(path: str = CFG.checkpoint_path) -> TinyViT:
    model = TinyViT()
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model
