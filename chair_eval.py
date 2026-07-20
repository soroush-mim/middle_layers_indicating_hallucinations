'''
Modified from: https://github.com/LALBJ/PAI/blob/master/chair_eval.py
'''

import argparse
import json
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from constants import INSTRUCTION_TEMPLATE, SYSTEM_MESSAGE
from eval_data_loader import COCODataSet
from model_manager import ModelManager
from tqdm import tqdm
from transformers.generation.logits_process import LogitsProcessorList

# modify attention
from modify_attention import (
    llama_foreground_guide,
    llama_head_guide,
    llama_segmentation_guide,
)
from foreground_masks import COCOForegroundMasker
from seg_attention import SegAttentionConfig

from utils import setup_seeds, disable_torch_init


parser = argparse.ArgumentParser(description="CHAIR evaluation on LVLMs.")
parser.add_argument("--model", type=str, default='llava-1.5', help="model")
parser.add_argument(
    "--options",
    nargs="+",
    help="override some settings in the used config, the key-value pair "
    "in xxx=yyy format will be merged into config file (deprecate), "
    "change to --cfg-options instead.",
)
# TODO
parser.add_argument(
    "--data-path",
    type=str,
    default="/path/to/COCO/val2014", # path
    help="data path",
)
parser.add_argument("--batch-size", type=int, default=1)
parser.add_argument("--beam", type=int, default=1) # Greedy decoding
parser.add_argument("--sample", action="store_true")
parser.add_argument("--alpha", type=float, default=0.5)
parser.add_argument("--use-head-guide", action="store_true")
parser.add_argument(
    "--use-foreground-guide",
    action="store_true",
    help="Oracle foreground attention guidance from COCO instance segmentations.",
)
parser.add_argument("--aggregation", type=str, default="mean")
parser.add_argument("--guide-range", type=str, default="5,18")
parser.add_argument(
    "--instances-path",
    type=str,
    default=None,
    help="Path to instances_val2014.json; required by --use-foreground-guide.",
)
parser.add_argument(
    "--foreground-alpha",
    type=float,
    default=None,
    help="Extra boost on foreground patches (defaults to --alpha).",
)
parser.add_argument(
    "--foreground-base",
    type=float,
    default=0.0,
    help="Baseline boost applied to ALL visual tokens (head-guide enrichment). "
    "Per-token boost is base + foreground_alpha * mask. 0.0 (default) = "
    "foreground-only; set e.g. 0.5 to keep the context enrichment and add "
    "foreground on top.",
)
parser.add_argument(
    "--foreground-binary",
    action="store_true",
    help="Treat a visual token as fully foreground (1.0) if >=1 of its pixels "
    "is foreground; otherwise use fractional per-patch occupancy.",
)
parser.add_argument(
    "--foreground-normalize",
    action="store_true",
    help="Scale foreground_alpha by 1/coverage per image, where coverage is the "
    "fraction of visual tokens with non-zero foreground mask (e.g. 40%% "
    "coverage -> alpha x2.5). Keeps the total foreground boost independent "
    "of object size. Applies only to the alpha (mask) term, not --foreground-base.",
)
# --- Segmentation-guided attention correction (Methods 1/2/3) ---------------
parser.add_argument(
    "--use-seg-correct",
    action="store_true",
    help="Enable the segmentation-guided correction pipeline (needs --instances-path). "
    "With all three sub-methods off it reproduces head-guide exactly.",
)
# Method 1: scattered-attention repair
parser.add_argument("--use-attention-repair", action="store_true")
parser.add_argument("--repair-scatter-metric", default="entropy",
                    choices=["entropy", "variance", "leakage", "components"])
parser.add_argument("--repair-threshold", type=float, default=0.5)
parser.add_argument("--repair-strategy", default="suppress",
                    choices=["suppress", "redistribute", "blend"])
parser.add_argument("--repair-gamma", type=float, default=1.0)
parser.add_argument("--repair-blend-alpha", type=float, default=0.5)
# Method 2: mask-guided head weighting
parser.add_argument("--use-mask-weighted-average", action="store_true")
parser.add_argument("--head-weight-metric", default="iou", choices=["iou", "dice", "cosine"])
# Method 3: segmentation-guided alignment
parser.add_argument("--use-attention-alignment", action="store_true")
parser.add_argument("--align-metric", default="iou", choices=["iou", "dice", "kl"])
parser.add_argument("--align-intervention", default="threshold",
                    choices=["threshold", "scale", "topk"])
parser.add_argument("--align-threshold", type=float, default=0.1)
parser.add_argument("--align-topk", type=int, default=8)
# shared
parser.add_argument("--normalize-weights", default="linear", choices=["linear", "softmax"])
parser.add_argument("--weight-temperature", type=float, default=1.0)
parser.add_argument("--seg-binary", action="store_true", help="binarize the seg mask")
parser.add_argument("--seg-alpha", type=float, default=None,
                    help="correction strength for seg pipeline (defaults to --alpha)")
# visualization
parser.add_argument("--seg-viz-dir", default=None, help="if set, dump debug figures here")
parser.add_argument("--seg-viz-layer", type=int, default=14)
parser.add_argument("--seg-viz-steps", type=int, default=8)
parser.add_argument("--max-tokens", type=int, default=512)
parser.add_argument("--num-images", type=int, default=500)
args = parser.parse_known_args()[0]

setup_seeds()
disable_torch_init() # accelerate the training process

# Due to the ‘prepare_xxx_inputs’ function in model_manager.py, the batch size must be 1 :)
assert(args.batch_size == 1)

print(f'Evaluated model: {args.model}')
_interventions = [args.use_head_guide, args.use_foreground_guide, args.use_seg_correct]
if sum(bool(x) for x in _interventions) > 1:
    raise ValueError("Choose one of --use-head-guide / --use-foreground-guide / --use-seg-correct")
if (args.use_foreground_guide or args.use_seg_correct) and not args.instances_path:
    raise ValueError("--instances-path is required by --use-foreground-guide / --use-seg-correct")
model_manager = ModelManager(args.model)
foreground_masker = (
    COCOForegroundMasker(
        args.instances_path,
        model_manager.image_processor,
        binarize=args.foreground_binary,
    )
    if args.use_foreground_guide
    else None
)

# Segmentation-guided correction: one masker + one config, reused per image.
if args.use_seg_correct:
    seg_masker = COCOForegroundMasker(
        args.instances_path, model_manager.image_processor, binarize=args.seg_binary
    )
    seg_viz = None
    if args.seg_viz_dir:
        from seg_attention_viz import SegVizRecorder
        seg_viz = SegVizRecorder(args.seg_viz_dir, layer=args.seg_viz_layer,
                                 max_steps=args.seg_viz_steps)
    seg_cfg = SegAttentionConfig(
        use_repair=args.use_attention_repair,
        repair_scatter_metric=args.repair_scatter_metric,
        repair_threshold=args.repair_threshold,
        repair_strategy=args.repair_strategy,
        repair_gamma=args.repair_gamma,
        repair_blend_alpha=args.repair_blend_alpha,
        use_weighted_average=args.use_mask_weighted_average,
        head_weight_metric=args.head_weight_metric,
        use_alignment=args.use_attention_alignment,
        align_metric=args.align_metric,
        align_intervention=args.align_intervention,
        align_threshold=args.align_threshold,
        align_topk=args.align_topk,
        normalize_weights=args.normalize_weights,
        weight_temperature=args.weight_temperature,
        binarize_mask=args.seg_binary,
        viz=seg_viz,
    )
    seg_alpha = args.seg_alpha if args.seg_alpha is not None else args.alpha
else:
    seg_masker = None

base_dir = "./log/" + args.model
if not os.path.exists(base_dir):
    os.makedirs(base_dir)


# Load COCO2014 val dataset
coco_dataset = COCODataSet(data_path=args.data_path, trans=model_manager.image_processor)
coco_loader = torch.utils.data.DataLoader(
    coco_dataset, batch_size=args.batch_size, shuffle=False, num_workers=20
)

### set some parameters
guided_layer_range = [int(x) for x in args.guide_range.split(",")] # [start, end)
guided_layer_range[1] += 1 # [start, end]

# Build a compact tag describing the enabled seg-correction methods.
seg_tag = ""
if args.use_seg_correct:
    parts = [f"_seg_alpha{seg_alpha}"]
    if args.use_attention_repair:
        parts.append(f"_m1repair-{args.repair_scatter_metric}-{args.repair_strategy}")
    if args.use_mask_weighted_average:
        parts.append(f"_m2wavg-{args.head_weight_metric}")
    if args.use_attention_alignment:
        parts.append(f"_m3align-{args.align_metric}-{args.align_intervention}")
    if len(parts) == 1:
        parts.append("_none")  # all methods off == head-guide baseline
    seg_tag = "".join(parts)

# Construct the output file name
file_parts = [
    f"chair_eval_{args.num_images}images",
    f"_{args.aggregation}" if args.use_head_guide else "",
    f"_head_guided_alpha{args.alpha}" if args.use_head_guide else "",
    f"_foreground_guided_alpha{args.foreground_alpha if args.foreground_alpha is not None else args.alpha}"
    if args.use_foreground_guide else "",
    f"_base{args.foreground_base}"
    if (args.use_foreground_guide and args.foreground_base != 0.0) else "",
    "_binary" if (args.use_foreground_guide and args.foreground_binary) else "",
    "_covnorm" if (args.use_foreground_guide and args.foreground_normalize) else "",
    seg_tag,
    f"_layers_{guided_layer_range[0]}-{guided_layer_range[1]}"
    if (args.use_head_guide or args.use_foreground_guide or args.use_seg_correct) else "",
    f"_tokens_{args.max_tokens}",
    "_sample" if args.sample else "",
    f"_beams_{args.beam}" if args.beam != 1 else "",
]

file_name = "".join(file_parts)

# Generate captions for each image
for batch_id, data in tqdm(enumerate(coco_loader), total=min(args.num_images, len(coco_loader))):
    if batch_id == args.num_images: # Randomly select images for CHAIR evaluation
        break
    img_id = data["img_id"]
    image = data["image"]

    batch_size = img_id.shape[0]
    query = ["Please help me describe the image in detail."] * batch_size
    questions, input_ids, kwargs = model_manager.prepare_inputs_for_model(query, image, use_dataloader=True)

    if args.use_head_guide:
        llama_head_guide(
            model_manager.llm_model,
            guided_layer_range=guided_layer_range,
            aggregation=args.aggregation,
            alpha=args.alpha,
            img_start_idx=model_manager.img_start_idx,
            img_end_idx=model_manager.img_end_idx
        )
    elif args.use_foreground_guide:
        foreground_mask = foreground_masker.token_mask(int(img_id[0]))
        foreground_alpha = (
            args.foreground_alpha if args.foreground_alpha is not None else args.alpha
        )
        if args.foreground_normalize:
            # coverage = fraction of visual tokens with any foreground.
            coverage = (foreground_mask > 0).float().mean().item()
            foreground_alpha = foreground_alpha / coverage if coverage > 0 else 0.0
        llama_foreground_guide(
            model=model_manager.llm_model,
            guided_layer_range=guided_layer_range,
            foreground_mask=foreground_mask,
            alpha=foreground_alpha,
            base=args.foreground_base,
            img_start_idx=model_manager.img_start_idx,
            img_end_idx=model_manager.img_end_idx,
        )
    elif args.use_seg_correct:
        llama_segmentation_guide(
            model=model_manager.llm_model,
            guided_layer_range=guided_layer_range,
            seg_mask=seg_masker.token_mask(int(img_id[0])),
            cfg=seg_cfg,
            alpha=seg_alpha,
            img_start_idx=model_manager.img_start_idx,
            img_end_idx=model_manager.img_end_idx,
        )

    with torch.inference_mode():
        outputs = model_manager.llm_model.generate(
            input_ids,
            do_sample=args.sample,
            max_new_tokens=args.max_tokens,
            use_cache=True,
            num_beams=args.beam,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
            **kwargs,
        )

    output_text = model_manager.decode(outputs)

    # Save the output to json file
    for i in range(len(output_text)):
        with open(os.path.join(base_dir, file_name + ".jsonl"), "a") as f:
            json.dump({"image_id": int(img_id[i]), "caption": output_text[i]}, f)
            f.write("\n")
