"""Method 1: repair scattered per-head attention using the segmentation mask.

Runs BEFORE head averaging. Heads whose visual attention is too scattered
(high-entropy / leaked outside the mask) get their pre-softmax logits nudged
toward the mask; well-focused heads are left untouched. Everything stays in
logit space, scaled by each head's own magnitude, so the image-vs-text balance
is perturbed no more than the existing head-guide correction.
"""

import torch

from seg_attention_utils import (
    EPS,
    compute_attention_entropy,
    connected_components,
    mass_outside_mask,
)


def scatter_score(p, a, m, metric):
    """Per-head scatter score in ~[0, 1]; higher = more scattered. (B, H)."""
    if metric == "entropy":
        return compute_attention_entropy(p)
    if metric == "leakage":
        return mass_outside_mask(p, m)
    if metric == "variance":
        # Peaky heads have high peak mass; scattered heads have low peak mass.
        return 1.0 - p.amax(dim=-1)
    if metric == "components":
        comp = connected_components(a)
        if torch.isnan(comp).any():
            return compute_attention_entropy(p)  # scipy missing -> fall back
        return torch.clamp(comp / 10.0, 0.0, 1.0)
    raise ValueError(f"unknown repair_scatter_metric: {metric}")


def repair_attention(v, p, m, scale, cfg):
    """Return repaired logits (B, H, N); only scattered heads are modified.

    Strategies (all additive in units of each head's mean |logit| ``scale``):
      * suppress     - lower logits outside the mask.
      * redistribute - raise inside / lower outside, proportional to leaked mass.
      * blend        - suppress, then blend with the original by repair_blend_alpha.
    """
    B, H, N = v.shape
    m_ = m.view(1, 1, -1)
    a = p / (p.amax(dim=-1, keepdim=True) + EPS)
    scattered = (scatter_score(p, a, m, cfg.repair_scatter_metric) > cfg.repair_threshold)
    scattered = scattered.view(B, H, 1).to(v.dtype)

    gamma = cfg.repair_gamma
    if cfg.repair_strategy == "suppress":
        repaired = v - gamma * (1.0 - m_) * scale
    elif cfg.repair_strategy == "redistribute":
        leaked = (p * (1.0 - m_)).sum(dim=-1, keepdim=True)  # (B, H, 1)
        repaired = v + gamma * leaked * scale * (m_ - (1.0 - m_))
    elif cfg.repair_strategy == "blend":
        suppressed = v - gamma * (1.0 - m_) * scale
        repaired = cfg.repair_blend_alpha * suppressed + (1.0 - cfg.repair_blend_alpha) * v
    else:
        raise ValueError(f"unknown repair_strategy: {cfg.repair_strategy}")

    return v * (1.0 - scattered) + repaired * scattered


class SegmentationAttentionRepair:
    """Thin OO wrapper around ``repair_attention`` for a clean interface."""

    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, v, p, m, scale):
        return repair_attention(v, p, m, scale, self.cfg)
