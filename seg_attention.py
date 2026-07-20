"""Orchestrator for the segmentation-guided attention-correction methods.

Pipeline applied to the last query token's visual attention logits:

    attention
      -> Method 1 repair       (optional, per-head, before averaging)
      -> Method 2/3 weighting  (optional, mask-guided head weights / gating)
      -> head averaging        (weighted; equal weights == the original mean)
      -> existing correction    v_rep + alpha * aggregate

Every stage collapses to the original head-guide behaviour when its flag is off:
with all methods disabled the write-back is exactly ``v + alpha * mean(|v|)``.
"""

from dataclasses import dataclass

import torch

from attention_alignment import alignment_modifier, head_alignment_scores
from attention_repair import repair_attention
from head_weighting import head_grounding_scores
from seg_attention_utils import (
    attention_distribution,
    normalize_weights,
    weighted_head_average,
)


@dataclass
class SegAttentionConfig:
    # Method 1: scattered-attention repair
    use_repair: bool = False
    repair_scatter_metric: str = "entropy"      # entropy | variance | leakage | components
    repair_threshold: float = 0.5
    repair_strategy: str = "suppress"           # suppress | redistribute | blend
    repair_gamma: float = 1.0
    repair_blend_alpha: float = 0.5

    # Method 2: mask-guided head weighting
    use_weighted_average: bool = False
    head_weight_metric: str = "iou"             # iou | dice | cosine

    # Method 3: segmentation-guided alignment
    use_alignment: bool = False
    align_metric: str = "iou"                   # iou | dice | kl
    align_intervention: str = "threshold"       # threshold | scale | topk
    align_threshold: float = 0.1
    align_topk: int = 8

    # shared
    normalize_weights: str = "linear"           # linear | softmax
    weight_temperature: float = 1.0
    binarize_mask: bool = False

    # visualisation (a SegVizRecorder or None)
    viz: object = None

    def any_enabled(self):
        return self.use_repair or self.use_weighted_average or self.use_alignment


def apply_seg_correction(attn_weights, img_start, img_end, seg_mask, cfg, alpha, layer_idx=None):
    """In-place segmentation-guided correction of the last-token visual logits."""
    v = attn_weights[:, :, -1, img_start:img_end]              # (B, H, N) view
    B, H, N = v.shape
    m = seg_mask.to(device=v.device, dtype=v.dtype)
    if cfg.binarize_mask:
        m = (m > 0).to(v.dtype)
    scale = v.abs().mean(dim=-1, keepdim=True)                 # per-head magnitude
    p = attention_distribution(v)

    # ---- Method 1: repair scattered heads -----------------------------------
    v_rep = v
    if cfg.use_repair:
        v_rep = repair_attention(v, p, m, scale, cfg)
        p = attention_distribution(v_rep)

    # ---- Method 2 + 3: head weights ------------------------------------------
    w = torch.ones(B, H, 1, device=v.device, dtype=v.dtype)
    grounding = alignment = None
    if cfg.use_weighted_average:
        grounding = head_grounding_scores(p, m, cfg.head_weight_metric)
        w = w * grounding.unsqueeze(-1)
    if cfg.use_alignment:
        alignment = head_alignment_scores(p, m, cfg.align_metric)
        w = w * alignment_modifier(alignment, cfg).unsqueeze(-1)
    w = normalize_weights(w, cfg.normalize_weights, cfg.weight_temperature)

    # ---- head averaging + existing correction --------------------------------
    aggregate = weighted_head_average(v_rep.abs(), w)          # (B, 1, N)
    attn_weights[:, :, -1, img_start:img_end] = v_rep + alpha * aggregate

    if cfg.viz is not None and layer_idx is not None:
        cfg.viz.maybe_record(
            layer_idx, attention_distribution(v), p, m, aggregate, grounding, alignment
        )
