"""Reusable helpers for the segmentation-guided attention-correction methods.

All functions operate on the last query token's visual attention over the 576
image tokens. Conventions:

  * ``v``  : (B, H, N) pre-softmax attention *logits* over image tokens.
  * ``p``  : (B, H, N) per-head spatial attention *distribution* (softmax of v).
  * ``a``  : (B, H, N) attention map normalised to peak 1 (for IoU/Dice).
  * ``m``  : (N,) segmentation mask in [0, 1] (broadcast over B, H).

These are cheap tensor ops so they run every decode step without noticeable
overhead (N=576, H<=32).
"""

import torch

EPS = 1e-8


def attention_distribution(v):
    """Softmax over image tokens -> per-head spatial distribution (B, H, N)."""
    return torch.softmax(v, dim=-1)


def normalize_attention(x, mode="max"):
    """Normalise a map. ``max`` -> peak 1; ``sum`` -> sums to 1 (distribution)."""
    if mode == "sum":
        return x / (x.sum(dim=-1, keepdim=True) + EPS)
    return x / (x.amax(dim=-1, keepdim=True) + EPS)


def compute_attention_entropy(p):
    """Normalised Shannon entropy in [0, 1]; higher = more scattered. (B, H)."""
    n = p.shape[-1]
    ent = -(p * (p + EPS).log()).sum(dim=-1)
    return ent / torch.log(torch.tensor(float(n), device=p.device))


def compute_attention_iou(a, m):
    """Soft IoU between peak-normalised attention ``a`` and mask ``m``. (B, H)."""
    m = m.view(1, 1, -1)
    inter = torch.minimum(a, m).sum(dim=-1)
    union = torch.maximum(a, m).sum(dim=-1)
    return inter / (union + EPS)


def compute_attention_dice(a, m):
    """Soft Dice between peak-normalised attention ``a`` and mask ``m``. (B, H)."""
    m = m.view(1, 1, -1)
    inter = (a * m).sum(dim=-1)
    return (2 * inter) / (a.sum(dim=-1) + m.sum(dim=-1) + EPS)


def compute_attention_cosine(p, m):
    """Cosine similarity between attention ``p`` and mask ``m``. (B, H)."""
    m = m.view(1, 1, -1)
    dot = (p * m).sum(dim=-1)
    return dot / (p.norm(dim=-1) * m.norm(dim=-1) + EPS)


def compute_attention_kl(p, m):
    """KL(p || mask_distribution); lower = better aligned. (B, H)."""
    m = m.view(1, 1, -1)
    m_dist = m / (m.sum(dim=-1, keepdim=True) + EPS)
    return (p * ((p + EPS).log() - (m_dist + EPS).log())).sum(dim=-1)


def mass_outside_mask(p, m):
    """Fraction of attention mass falling outside the mask (leakage). (B, H)."""
    m = m.view(1, 1, -1)
    return (p * (1.0 - m)).sum(dim=-1)


def connected_components(a, m=None, grid=24, thr=0.5):
    """Number of connected components of the thresholded attention. (B, H).

    Uses scipy if available; otherwise returns NaNs so callers can fall back to
    a cheaper scatter metric.
    """
    try:
        from scipy.ndimage import label
    except ImportError:
        return torch.full(a.shape[:2], float("nan"), device=a.device)
    B, H, _ = a.shape
    binary = (a > thr).view(B, H, grid, grid).cpu().numpy()
    out = torch.zeros(B, H, device=a.device)
    for b in range(B):
        for h in range(H):
            out[b, h] = label(binary[b, h])[1]
    return out


def normalize_weights(w, scheme="linear", temperature=1.0):
    """Turn per-head raw weights (B, H, 1) into normalised weights over heads.

    ``linear`` -> proportional (sums to 1). ``softmax`` -> softmax(temp * w) with
    exact-zero weights excluded (so gating/top-k stays zeroed). Uniform input
    yields uniform 1/H under both schemes, preserving baseline behaviour.
    """
    if scheme == "softmax":
        logits = (temperature * w).clone()
        logits[w <= 0] = float("-inf")
        out = torch.softmax(logits, dim=1)
        return torch.nan_to_num(out, nan=0.0)
    s = w.sum(dim=1, keepdim=True)
    uniform = torch.full_like(w, 1.0 / w.shape[1])
    return torch.where(s > 0, w / (s + EPS), uniform)


def weighted_head_average(v_abs, w):
    """Aggregate |attention| across heads with weights ``w`` (B, H, 1) -> (B,1,N)."""
    return (w * v_abs).sum(dim=1, keepdim=True)
