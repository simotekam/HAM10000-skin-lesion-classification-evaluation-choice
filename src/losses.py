"""Focal loss and weighted cross-entropy for the four loss configurations."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)."""

    def __init__(self, alpha: torch.Tensor, gamma: float = 2.0):
        super().__init__()
        self.register_buffer("alpha", alpha)
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_p = F.log_softmax(logits, dim=1)
        log_pt = log_p.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()
        alpha_t = self.alpha[targets]
        loss = -alpha_t * (1 - pt) ** self.gamma * log_pt
        return loss.mean()


def class_weights_from_counts(counts: torch.Tensor) -> torch.Tensor:
    """Inverse frequency, normalised to mean 1. Fixed, not hand-tuned (our choice).

    Counts are clamped to at least 1: real splits always have every class represented
    (see subset tables), but a subsampled smoke test can otherwise hit a 1/0.
    """
    weights = 1.0 / counts.float().clamp(min=1)
    return weights * (len(weights) / weights.sum())


def build_loss(config: str, class_weights: torch.Tensor, gamma: float = 2.0) -> nn.Module:
    n_classes = len(class_weights)
    if config == "full":
        return FocalLoss(alpha=class_weights, gamma=gamma)
    if config == "no_fl":
        return nn.CrossEntropyLoss(weight=class_weights)
    if config == "no_cw":
        return FocalLoss(alpha=torch.ones(n_classes), gamma=gamma)
    if config == "plain":
        return nn.CrossEntropyLoss()
    raise ValueError(f"unknown config: {config}")
