"""Quick training script for the tiny ViT on clean DermaMNIST. CPU-feasible in minutes."""
import os

import torch
import torch.nn as nn

from .config import CFG
from .data import load_splits, normalize
from .model import TinyViT


def main():
    torch.manual_seed(CFG.seed)
    splits = load_splits()
    tr_x, tr_y = splits["train"]
    te_x, te_y = splits["test"]
    tr_x, te_x = normalize(tr_x), normalize(te_x)

    model = TinyViT()
    opt = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    n = len(tr_x)

    for epoch in range(CFG.epochs):
        model.train()
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, CFG.train_batch_size):
            idx = perm[i : i + CFG.train_batch_size]
            opt.zero_grad()
            loss = loss_fn(model(tr_x[idx]), tr_y[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            acc = (model(te_x).argmax(1) == te_y).float().mean().item()
        print(f"epoch {epoch + 1:2d}/{CFG.epochs}  loss {total / n:.4f}  clean test acc {acc:.4f}")

    os.makedirs(os.path.dirname(CFG.checkpoint_path), exist_ok=True)
    torch.save(model.state_dict(), CFG.checkpoint_path)
    print(f"saved {CFG.checkpoint_path}")


if __name__ == "__main__":
    main()
