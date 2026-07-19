"""Version A: teacher-forced oracle for per-noun attention targeting.

For each image we let LLaVA generate a caption greedily, then for every COCO
object word in that caption we re-run one forward pass at the step that predicts
the object token and read the probability the model assigns to that exact token,
twice:

  * normal          - no attention intervention
  * correct-object  - boost the "correct" patches at that step:
        real object      -> its own category mask
        hallucinated obj -> the union of all real GT objects
                            (i.e. "look at the real things, not the phantom")

If pointing attention at the correct object RAISES the probability of real
object words and LOWERS the probability of hallucinated ones, per-noun targeting
has headroom. This is a diagnostic (no new captions are produced), so run it
before building/evaluating Version B.

Example:
    python run_version_a.py --model llava-1.5 \
        --data-path "$IMAGES" --instances-path "$INSTANCES" \
        --coco_path "$ANNODIR" --num-images 100 --alpha 0.5
"""

import argparse
import csv
import math
import os

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from eval_data_loader import COCODataSet
from foreground_masks import COCOForegroundMasker
from model_manager import ModelManager
from modify_attention import DynamicMaskHolder, llama_dynamic_guide
from noun_resolver import NounResolver, load_evaluator
from utils import disable_torch_init, setup_seeds


def object_token_positions(gen_ids, resolver, tokenizer):
    """(token_index, node_word) for each object word as it completes.

    token_index is the step whose predicted token completed the object word;
    for the single-token COCO words that dominate captions this is exactly the
    object token, so p(token) measures p(object).
    """
    positions = []
    prev_count = 0
    for t in range(len(gen_ids)):
        text = tokenizer.decode(gen_ids[: t + 1], skip_special_tokens=True)
        node_words = resolver.detect(text)
        if len(node_words) > prev_count:
            positions.append((t, node_words[-1]))
            prev_count = len(node_words)
    return positions


def token_prob(model, full_ids, images, target_id):
    with torch.inference_mode():
        out = model(input_ids=full_ids, images=images, use_cache=False)
    logits = out.logits[0, -1].float()
    return torch.softmax(logits, dim=-1)[target_id].item()


def make_figures(records, out_dir):
    if not records:
        print("no object records collected; skipping figures")
        return
    real = [r for r in records if r["real"]]
    hall = [r for r in records if not r["real"]]
    eps = 1e-12
    for r in records:
        r["dlogp"] = math.log(r["p_boost"] + eps) - math.log(r["p_noboost"] + eps)

    # 1) scatter p_boost vs p_noboost -------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 6))
    for grp, color, lab in [(real, "tab:green", "real"), (hall, "tab:red", "hallucinated")]:
        if grp:
            ax.scatter([r["p_noboost"] for r in grp], [r["p_boost"] for r in grp],
                       s=18, alpha=0.6, color=color, label=lab)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("p(object token) - no boost")
    ax.set_ylabel("p(object token) - correct-object boost")
    ax.set_title("Above y=x: boost helped. Want real above, hallucinated below.")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "a_scatter_pboost_vs_pnoboost.png"), dpi=130)
    plt.close(fig)

    # 2) distribution of delta log-prob -----------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    bins = np.linspace(
        min(r["dlogp"] for r in records), max(r["dlogp"] for r in records), 40
    )
    for grp, color, lab in [(real, "tab:green", "real"), (hall, "tab:red", "hallucinated")]:
        if grp:
            ax.hist([r["dlogp"] for r in grp], bins=bins, alpha=0.5, color=color,
                    label=lab, density=True)
    ax.axvline(0, color="k", ls="--", lw=0.8)
    ax.set_xlabel(r"$\Delta \log p$ = logp(boost) - logp(no boost)")
    ax.set_ylabel("density")
    ax.set_title("Correct-object boost should push real right (+), hallucinated left (-)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "a_dlogp_hist.png"), dpi=130)
    plt.close(fig)

    # 3) mean delta log-prob bar with 95% CI ------------------------------------
    fig, ax = plt.subplots(figsize=(5, 5))
    labels, means, errs, colors = [], [], [], []
    for grp, color, lab in [(real, "tab:green", "real"), (hall, "tab:red", "hallucinated")]:
        if grp:
            vals = np.array([r["dlogp"] for r in grp])
            labels.append(f"{lab}\n(n={len(vals)})")
            means.append(vals.mean())
            errs.append(1.96 * vals.std(ddof=1) / math.sqrt(len(vals)) if len(vals) > 1 else 0)
            colors.append(color)
    ax.bar(labels, means, yerr=errs, color=colors, alpha=0.8, capsize=6)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel(r"mean $\Delta \log p$")
    ax.set_title("Headroom check: bars should point away from 0 in opposite directions")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "a_mean_dlogp_bar.png"), dpi=130)
    plt.close(fig)

    # console summary -----------------------------------------------------------
    def summ(grp, name):
        if not grp:
            print(f"  {name}: none")
            return
        d = np.array([r["dlogp"] for r in grp])
        print(f"  {name:13} n={len(grp):4d}  mean dlogp={d.mean():+.3f}  "
              f"frac p_boost>p_noboost={np.mean([r['p_boost'] > r['p_noboost'] for r in grp]):.2f}")

    print("\nVersion A summary:")
    summ(real, "real")
    summ(hall, "hallucinated")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="llava-1.5")
    ap.add_argument("--data-path", required=True)
    ap.add_argument("--instances-path", required=True)
    ap.add_argument("--cache", default="chair.pkl")
    ap.add_argument("--coco_path", default=None, help="only needed if no --cache exists")
    ap.add_argument("--guide-range", default="5,18")
    ap.add_argument("--alpha", type=float, default=0.5, help="correct-object boost strength")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--num-images", type=int, default=100)
    ap.add_argument("--out-dir", default="./version_a")
    args = ap.parse_args()

    setup_seeds()
    disable_torch_init()
    os.makedirs(args.out_dir, exist_ok=True)

    layer_range = [int(x) for x in args.guide_range.split(",")]
    layer_range[1] += 1  # [start, end]

    model_manager = ModelManager(args.model)
    masker = COCOForegroundMasker(args.instances_path, model_manager.image_processor)
    resolver = NounResolver(load_evaluator(args.cache, args.coco_path))
    name_map = resolver.name_map()

    dataset = COCODataSet(data_path=args.data_path, trans=model_manager.image_processor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)

    model = model_manager.llm_model
    tokenizer = model_manager.tokenizer

    records = []
    for batch_id, data in tqdm(enumerate(loader), total=args.num_images):
        if batch_id == args.num_images:
            break
        img_id = int(data["img_id"][0])
        image = data["image"]
        try:
            cat_masks = masker.category_token_masks(img_id, name_map)
            union = masker.token_mask(img_id)
        except KeyError:
            continue  # image not in instances json
        gt = resolver.gt_objects(img_id)

        query = ["Please help me describe the image in detail."]
        _, input_ids, kwargs = model_manager.prepare_inputs_for_model(query, image, use_dataloader=True)
        images = kwargs["images"]

        # install the mutable holder once; None => no boost for the greedy pass.
        holder = DynamicMaskHolder(mask=None)
        llama_dynamic_guide(model, layer_range, holder, model_manager.img_start_idx, model_manager.img_end_idx)

        holder.mask = None
        with torch.inference_mode():
            outputs = model.generate(
                input_ids, do_sample=False, num_beams=1, max_new_tokens=args.max_tokens,
                use_cache=True, **kwargs,
            )
        gen_ids = outputs[0, input_ids.shape[1]:]

        for t, node_word in object_token_positions(gen_ids, resolver, tokenizer):
            real = node_word in gt
            mask = cat_masks[node_word] if (real and node_word in cat_masks) else union
            prefix = gen_ids[:t].unsqueeze(0)
            full_ids = torch.cat([input_ids, prefix], dim=1)
            target_id = int(gen_ids[t])

            holder.mask = None
            p_nb = token_prob(model, full_ids, images, target_id)
            holder.set(mask, 0.0, args.alpha)
            p_b = token_prob(model, full_ids, images, target_id)

            records.append({
                "image_id": img_id, "node_word": node_word, "real": real,
                "token_pos": t, "p_noboost": p_nb, "p_boost": p_b,
            })

    csv_path = os.path.join(args.out_dir, "version_a_records.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()) if records else
                           ["image_id", "node_word", "real", "token_pos", "p_noboost", "p_boost"])
        w.writeheader()
        w.writerows(records)
    print(f"wrote {csv_path}  ({len(records)} object records)")

    make_figures(records, args.out_dir)
    print(f"figures in {args.out_dir}/")


if __name__ == "__main__":
    main()
