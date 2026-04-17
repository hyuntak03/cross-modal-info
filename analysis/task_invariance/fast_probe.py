"""
GPU-based fast linear probe helper.
"""
import numpy as np
import torch
import torch.nn as nn


def gpu_probe(X, y, n_classes, seed=42, test_ratio=0.3, epochs=200, lr=1e-2):
    """Train linear probe on GPU. Returns test accuracy."""
    with torch.enable_grad():
        X = torch.from_numpy(X).float().cuda()
        y = torch.from_numpy(y).long().cuda()
        n = X.shape[0]
        torch.manual_seed(seed)
        perm = torch.randperm(n)
        n_tr = int(n * (1 - test_ratio))
        tr, te = perm[:n_tr], perm[n_tr:]

        # Normalize (per-feature)
        mean, std = X[tr].mean(0), X[tr].std(0) + 1e-6
        X_n = (X - mean) / std

        model = nn.Linear(X.shape[1], n_classes).cuda()
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-2)
        loss_fn = nn.CrossEntropyLoss()

        for e in range(epochs):
            model.train()
            logits = model(X_n[tr])
            loss = loss_fn(logits, y[tr])
            opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            preds = model(X_n[te]).argmax(dim=1)
            acc = (preds == y[te]).float().mean().item()
    return float(acc)
