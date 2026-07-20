"""Method 2: mask-guided head weighting.

Replaces the equal-weight head average with a weighted average, where each head
is weighted by how well its visual attention aligns with the segmentation mask.
Returns raw per-head grounding scores; final normalisation (softmax/linear) is
applied by the orchestrator so Methods 2 and 3 compose.
"""

from seg_attention_utils import (
    compute_attention_cosine,
    compute_attention_dice,
    compute_attention_iou,
    normalize_attention,
)


def head_grounding_scores(p, m, metric):
    """Per-head grounding score, higher = better aligned, all >= 0. (B, H)."""
    if metric == "iou":
        return compute_attention_iou(normalize_attention(p, "max"), m)
    if metric == "dice":
        return compute_attention_dice(normalize_attention(p, "max"), m)
    if metric == "cosine":
        return compute_attention_cosine(p, m).clamp(min=0.0)
    raise ValueError(f"unknown head_weight_metric: {metric}")
