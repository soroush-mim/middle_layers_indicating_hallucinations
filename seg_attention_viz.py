"""Optional debug visualisation for the segmentation-guided attention methods.

When enabled, saves a small number of figures (one per decode step, for a single
chosen layer) showing the mask, the original vs repaired attention, the head-
weighted aggregate, and the per-head grounding/alignment scores.
"""

import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class SegVizRecorder:
    def __init__(self, out_dir, layer=14, max_steps=8, tag="seg"):
        self.out_dir = out_dir
        self.layer = layer
        self.max_steps = max_steps
        self.tag = tag
        self.count = 0
        os.makedirs(out_dir, exist_ok=True)

    def maybe_record(self, layer_idx, p_orig, p_rep, m, aggregate, grounding, alignment):
        if layer_idx != self.layer or self.count >= self.max_steps:
            return
        step = self.count
        self.count += 1
        g = int(round(math.sqrt(p_orig.shape[-1])))

        def grid(x):
            return x[0].mean(dim=0).view(g, g).float().cpu().numpy()  # mean over heads

        panels = [
            ("segmentation mask", m.view(g, g).float().cpu().numpy()),
            ("orig attn (head-mean)", grid(p_orig)),
            ("repaired attn (head-mean)", grid(p_rep)),
            ("weighted aggregate", aggregate[0, 0].view(g, g).float().cpu().numpy()),
        ]
        bars = [("grounding score / head", grounding), ("alignment score / head", alignment)]
        bars = [(t, s) for t, s in bars if s is not None]

        ncol = len(panels) + len(bars)
        fig, axes = plt.subplots(1, ncol, figsize=(3 * ncol, 3))
        if ncol == 1:
            axes = [axes]
        for ax, (title, img) in zip(axes, panels):
            ax.imshow(img, cmap="hot")
            ax.set_title(title, fontsize=8)
            ax.axis("off")
        for ax, (title, scores) in zip(axes[len(panels):], bars):
            ax.bar(range(scores.shape[1]), scores[0].float().cpu().numpy())
            ax.set_title(title, fontsize=8)
            ax.set_xlabel("head")
        fig.suptitle(f"{self.tag} | layer {layer_idx} | step {step}", fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(self.out_dir, f"{self.tag}_layer{layer_idx}_step{step}.png"), dpi=110)
        plt.close(fig)
