"""Method 3: segmentation-guided attention alignment.

Scores every head by how visually grounded it is (agreement with the mask), then
intervenes on the aggregation by gating, scaling, or top-k selecting heads. The
scores are also returned for logging/visualisation. Produces a per-head
multiplicative modifier in [0, 1]+ that the orchestrator folds into the head
weights, so it composes with Method 2.
"""

import torch

from seg_attention_utils import (
    compute_attention_dice,
    compute_attention_iou,
    compute_attention_kl,
    normalize_attention,
)


def head_alignment_scores(p, m, metric):
    """Per-head alignment score, higher = better grounded. (B, H).

    KL is a divergence (lower = better), so it is converted to exp(-KL) to keep
    the "higher is better" convention consistent across metrics.
    """
    if metric == "iou":
        return compute_attention_iou(normalize_attention(p, "max"), m)
    if metric == "dice":
        return compute_attention_dice(normalize_attention(p, "max"), m)
    if metric == "kl":
        return torch.exp(-compute_attention_kl(p, m))
    raise ValueError(f"unknown align_metric: {metric}")


def alignment_modifier(scores, cfg):
    """Per-head multiplicative weight modifier from alignment scores. (B, H).

      * threshold - 1 for heads at/above align_threshold, else 0.
      * scale     - the score itself (grounded heads contribute more).
      * topk      - 1 for the align_topk best heads, else 0.
    """
    if cfg.align_intervention == "threshold":
        return (scores >= cfg.align_threshold).to(scores.dtype)
    if cfg.align_intervention == "scale":
        return scores.clamp(min=0.0)
    if cfg.align_intervention == "topk":
        k = min(cfg.align_topk, scores.shape[1])
        idx = scores.topk(k, dim=1).indices
        mod = torch.zeros_like(scores)
        mod.scatter_(1, idx, 1.0)
        return mod
    raise ValueError(f"unknown align_intervention: {cfg.align_intervention}")
