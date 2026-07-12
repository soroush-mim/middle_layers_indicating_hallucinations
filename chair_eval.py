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
from modify_attention import llama_foreground_guide, llama_head_guide
from foreground_masks import COCOForegroundMasker

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
parser.add_argument("--max-tokens", type=int, default=512)
parser.add_argument("--num-images", type=int, default=500)
args = parser.parse_known_args()[0]

setup_seeds()
disable_torch_init() # accelerate the training process

# Due to the ‘prepare_xxx_inputs’ function in model_manager.py, the batch size must be 1 :)
assert(args.batch_size == 1)

print(f'Evaluated model: {args.model}')
if args.use_head_guide and args.use_foreground_guide:
    raise ValueError("Choose one intervention: --use-head-guide or --use-foreground-guide")
if args.use_foreground_guide and not args.instances_path:
    raise ValueError("--instances-path is required by --use-foreground-guide")
model_manager = ModelManager(args.model)
foreground_masker = (
    COCOForegroundMasker(args.instances_path, model_manager.image_processor)
    if args.use_foreground_guide
    else None
)

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

# Construct the output file name
file_parts = [
    f"chair_eval_{args.num_images}images",
    f"_{args.aggregation}" if args.use_head_guide else "",
    f"_head_guided_alpha{args.alpha}" if args.use_head_guide else "",
    f"_foreground_guided_alpha{args.foreground_alpha if args.foreground_alpha is not None else args.alpha}"
    if args.use_foreground_guide else "",
    f"_base{args.foreground_base}"
    if (args.use_foreground_guide and args.foreground_base != 0.0) else "",
    f"_layers_{guided_layer_range[0]}-{guided_layer_range[1]}"
    if (args.use_head_guide or args.use_foreground_guide) else "",
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
        llama_foreground_guide(
            model=model_manager.llm_model,
            guided_layer_range=guided_layer_range,
            foreground_mask=foreground_masker.token_mask(int(img_id[0])),
            alpha=args.foreground_alpha if args.foreground_alpha is not None else args.alpha,
            base=args.foreground_base,
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
