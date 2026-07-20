"""Sanity check: seg-correction with all methods OFF == head-guide correction.

Run on the server (needs torch). Verifies the baseline-preservation requirement
and exercises each method for shape/finiteness.
"""

import torch

from seg_attention import SegAttentionConfig, apply_seg_correction

B, H, Q, KV = 1, 32, 40, 620
S, E = 5, 5 + 576  # image token span
torch.manual_seed(0)


def head_guide_reference(attn, alpha):
    v = attn[:, :, -1, S:E]
    attn[:, :, -1, S:E] = v + alpha * v.abs().mean(dim=1, keepdim=True)
    return attn


def run(cfg, alpha=0.5):
    attn = torch.randn(B, H, Q, KV)
    mask = (torch.rand(576) > 0.6).float()
    apply_seg_correction(attn.clone() if False else attn, S, E, mask, cfg, alpha)
    return attn


base_attn = torch.randn(B, H, Q, KV)
mask = (torch.rand(576) > 0.6).float()

ref = head_guide_reference(base_attn.clone(), 0.5)
got = base_attn.clone()
apply_seg_correction(got, S, E, mask, SegAttentionConfig(), 0.5)  # all methods off
assert torch.allclose(ref, got, atol=1e-5), "all-off != head-guide!"
print("PASS: all-methods-off reproduces head-guide exactly.")

# exercise each method (shape / finiteness only)
for name, cfg in [
    ("M1 repair", SegAttentionConfig(use_repair=True, repair_scatter_metric="leakage")),
    ("M2 weighted", SegAttentionConfig(use_weighted_average=True, head_weight_metric="dice")),
    ("M3 align", SegAttentionConfig(use_alignment=True, align_metric="kl", align_intervention="topk")),
    ("M1+M2+M3", SegAttentionConfig(use_repair=True, use_weighted_average=True, use_alignment=True)),
]:
    out = run(cfg)
    assert torch.isfinite(out).all(), f"{name} produced non-finite values"
    print(f"PASS: {name} runs, output finite.")
