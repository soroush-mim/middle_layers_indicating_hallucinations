"""Version B: causal per-noun attention lock during free generation.

State machine (one active mask fed to the guided layers each decode step):

  * idle  -> paper head-guide: boost ALL image tokens (alpha=idle_alpha).
             This is the default whenever no object is being elaborated.
  * lock  -> when a REAL COCO object noun is emitted, boost ONLY that object's
             patches (base=0, alpha=lock_alpha) for a fixed window of W tokens,
             then fall back to idle.
  * hallucinated noun -> the object has no GT mask; behaviour is a CLI choice:
       --halluc-policy keep  : re-lock onto the LAST real object for W tokens.
       --halluc-policy paper : fall back to idle (paper head-guide).

Captions are written to ./log/<model>/...jsonl so chair.py / plot_chair_curves.py
score them exactly like the other runs. For the first --trace-samples images a
per-token lock timeline PNG is saved to visualise the state machine.

Example:
    python run_version_b.py --model llava-1.5 \
        --data-path "$IMAGES" --instances-path "$INSTANCES" \
        --coco_path "$ANNODIR" --num-images 100 \
        --lock-window 4 --halluc-policy keep
"""

import argparse
import json
import os

import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList

from eval_data_loader import COCODataSet
from foreground_masks import COCOForegroundMasker
from model_manager import ModelManager
from modify_attention import DynamicMaskHolder, llama_dynamic_guide
from noun_resolver import NounResolver, load_evaluator
from utils import disable_torch_init, setup_seeds


class PerNounLock(LogitsProcessor):
    """Mutates the shared holder each step; never changes the logits."""

    def __init__(self, holder, resolver, tokenizer, ones_mask,
                 window, lock_alpha, idle_alpha, halluc_policy):
        self.holder = holder
        self.resolver = resolver
        self.tokenizer = tokenizer
        self.ones = ones_mask
        self.window = window
        self.lock_alpha = lock_alpha
        self.idle_alpha = idle_alpha
        self.policy = halluc_policy
        self.cat_masks, self.gt, self.trace, self.prompt_len = {}, set(), None, 0

    def configure(self, cat_masks, gt, prompt_len, trace_list=None):
        self.cat_masks, self.gt, self.prompt_len, self.trace = cat_masks, gt, prompt_len, trace_list
        self.n_seen, self.lock_remaining, self.last_real_mask = 0, 0, None
        self._set_idle()

    def _set_idle(self):
        self.holder.set(self.ones, 0.0, self.idle_alpha)
        self.lock_remaining = 0

    def _lock(self, mask):
        self.holder.set(mask, 0.0, self.lock_alpha)
        self.lock_remaining = self.window

    def __call__(self, input_ids, scores):
        gen_ids = input_ids[0, self.prompt_len:]
        cat, obj = "idle", None
        node_words = self.resolver.detect(self.tokenizer.decode(gen_ids, skip_special_tokens=True))

        if len(node_words) > self.n_seen:
            obj = node_words[-1]
            self.n_seen = len(node_words)
            mask = self.cat_masks.get(obj)
            if obj in self.gt and mask is not None:          # real object w/ mask -> lock
                self._lock(mask)
                self.last_real_mask = mask
                cat = "lock"
            elif obj in self.gt:                             # real but no seg mask
                self._set_idle()
                cat = "idle"
            elif self.policy == "keep" and self.last_real_mask is not None:  # hallucination
                self._lock(self.last_real_mask)
                cat = "keep"
            else:
                self._set_idle()
                cat = "idle"
        elif self.lock_remaining > 0:                        # continue an active lock
            self.lock_remaining -= 1
            cat = "lock" if self.lock_remaining > 0 else "idle"
            if self.lock_remaining == 0:
                self._set_idle()

        if self.trace is not None:
            tok = self.tokenizer.decode(gen_ids[-1:], skip_special_tokens=True) if len(gen_ids) else ""
            self.trace.append({"token": tok, "cat": cat, "obj": obj})
        return scores


def render_trace(trace, caption, out_path, title):
    colors = {"idle": "#bbbbbb", "lock": "#1f77b4", "keep": "#ff7f0e"}
    fig, ax = plt.subplots(figsize=(max(6, len(trace) * 0.13), 2.4))
    for i, s in enumerate(trace):
        ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=colors.get(s["cat"], "#bbbbbb")))
        if s["obj"]:
            ax.annotate(s["obj"], (i + 0.5, 1.05), rotation=90, fontsize=6, ha="center", va="bottom")
    ax.set_xlim(0, max(1, len(trace)))
    ax.set_ylim(0, 2)
    ax.set_yticks([])
    ax.set_xlabel("generated token index")
    ax.set_title(title, fontsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()]
    ax.legend(handles, colors.keys(), ncol=3, fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="llava-1.5")
    ap.add_argument("--data-path", required=True)
    ap.add_argument("--instances-path", required=True)
    ap.add_argument("--cache", default="chair.pkl")
    ap.add_argument("--coco_path", default=None, help="only needed if no --cache exists")
    ap.add_argument("--guide-range", default="5,18")
    ap.add_argument("--lock-window", type=int, default=4, help="W: tokens to hold a lock")
    ap.add_argument("--lock-alpha", type=float, default=0.5, help="object-only boost during lock")
    ap.add_argument("--idle-alpha", type=float, default=0.5, help="paper head-guide boost when idle")
    ap.add_argument("--halluc-policy", choices=["keep", "paper"], default="keep")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--num-images", type=int, default=100)
    ap.add_argument("--trace-samples", type=int, default=6, help="images to render lock timelines for")
    args = ap.parse_args()

    setup_seeds()
    disable_torch_init()

    layer_range = [int(x) for x in args.guide_range.split(",")]
    layer_range[1] += 1  # [start, end]

    model_manager = ModelManager(args.model)
    masker = COCOForegroundMasker(args.instances_path, model_manager.image_processor)
    resolver = NounResolver(load_evaluator(args.cache, args.coco_path))
    name_map = resolver.name_map()

    dataset = COCODataSet(data_path=args.data_path, trans=model_manager.image_processor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)

    model = model_manager.llm_model
    ones = torch.ones(masker.grid_size ** 2)
    holder = DynamicMaskHolder(mask=None)
    processor = PerNounLock(holder, resolver, model_manager.tokenizer, ones,
                            args.lock_window, args.lock_alpha, args.idle_alpha, args.halluc_policy)

    base_dir = os.path.join("./log", args.model)
    os.makedirs(base_dir, exist_ok=True)
    trace_dir = os.path.join(base_dir, "lock_traces")
    os.makedirs(trace_dir, exist_ok=True)
    file_name = (
        f"chair_eval_{args.num_images}images_locknoun_win{args.lock_window}_{args.halluc_policy}"
        f"_lockA{args.lock_alpha}_idleA{args.idle_alpha}"
        f"_layers_{layer_range[0]}-{layer_range[1]}_tokens_{args.max_tokens}"
    )
    out_path = os.path.join(base_dir, file_name + ".jsonl")
    open(out_path, "w").close()  # truncate any prior run

    for batch_id, data in tqdm(enumerate(loader), total=args.num_images):
        if batch_id == args.num_images:
            break
        img_id = int(data["img_id"][0])
        image = data["image"]
        try:
            cat_masks = masker.category_token_masks(img_id, name_map)
        except KeyError:
            continue
        gt = resolver.gt_objects(img_id)

        query = ["Please help me describe the image in detail."]
        _, input_ids, kwargs = model_manager.prepare_inputs_for_model(query, image, use_dataloader=True)
        llama_dynamic_guide(model, layer_range, holder, model_manager.img_start_idx, model_manager.img_end_idx)

        trace = [] if batch_id < args.trace_samples else None
        processor.configure(cat_masks, gt, input_ids.shape[1], trace)

        with torch.inference_mode():
            outputs = model.generate(
                input_ids, do_sample=False, num_beams=1, max_new_tokens=args.max_tokens,
                use_cache=True, logits_processor=LogitsProcessorList([processor]), **kwargs,
            )
        caption = model_manager.decode(outputs)[0]

        with open(out_path, "a") as f:
            json.dump({"image_id": img_id, "caption": caption}, f)
            f.write("\n")

        if trace is not None:
            render_trace(trace, caption, os.path.join(trace_dir, f"{file_name}_img{img_id}.png"),
                         title=f"img {img_id} | win={args.lock_window} policy={args.halluc_policy}\n{caption[:110]}")

    print(f"\nwrote {out_path}")
    print(f"lock timelines in {trace_dir}/")
    print(f"score with: python chair.py --cap_file {out_path} --coco_path $ANNODIR")


if __name__ == "__main__":
    main()
