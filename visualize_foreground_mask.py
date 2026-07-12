"""Overlay COCO foreground token masks on the exact image LLaVA-1.5 sees.

For each sample this renders three panels:
  1. the center-cropped 336x336 model input (pixel_values, denormalized)
  2. the 24x24 token mask as a heatmap
  3. the mask overlaid on the model input

Purpose: eyeball that foreground tokens land on the objects. This is the
cheapest way to catch a vertical flip / transpose in the CLIP-patch ordering
before running the full CHAIR sweep.

Only the CLIP image processor is loaded (not the 7B LLM), so it is fast.

Example:
    python visualize_foreground_mask.py \
        --instances-path /path/to/annotations/instances_val2014.json \
        --data-path /path/to/COCO/val2014 \
        --num-samples 6
"""

import argparse
import os

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from transformers import CLIPImageProcessor

from foreground_masks import COCOForegroundMasker


def coco_filename(image_id):
    return f"COCO_val2014_{image_id:012d}.jpg"


def denormalize(pixel_values, processor):
    """(3, H, W) normalized tensor -> (H, W, 3) displayable array in [0, 1]."""
    mean = torch.tensor(processor.image_mean).view(3, 1, 1)
    std = torch.tensor(processor.image_std).view(3, 1, 1)
    img = pixel_values * std + mean
    return img.clamp(0, 1).permute(1, 2, 0).numpy()


def select_image_ids(masker, data_path, num_samples, explicit_ids):
    if explicit_ids:
        return explicit_ids
    ids = []
    # masker.annotations keys are exactly the image ids that have >=1 instance.
    for iid in masker.annotations:
        if os.path.exists(os.path.join(data_path, coco_filename(iid))):
            ids.append(iid)
        if len(ids) >= num_samples:
            break
    return ids


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--instances-path", required=True)
    ap.add_argument("--data-path", required=True, help="folder with COCO val2014 jpgs")
    ap.add_argument(
        "--image-processor",
        default="openai/clip-vit-large-patch14-336",
        help="hub id or local dir of the CLIP processor LLaVA-1.5 uses",
    )
    ap.add_argument("--image-ids", type=int, nargs="*", default=None)
    ap.add_argument("--num-samples", type=int, default=6)
    ap.add_argument("--out-dir", default="./foreground_overlays")
    ap.add_argument("--grid-size", type=int, default=24)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    processor = CLIPImageProcessor.from_pretrained(args.image_processor)
    masker = COCOForegroundMasker(args.instances_path, processor, grid_size=args.grid_size)

    image_ids = select_image_ids(masker, args.data_path, args.num_samples, args.image_ids)
    if not image_ids:
        print("No matching images found. Check --data-path / --instances-path.")
        return

    for iid in image_ids:
        fpath = os.path.join(args.data_path, coco_filename(iid))
        if not os.path.exists(fpath):
            print(f"skip {iid}: file not found at {fpath}")
            continue

        image = Image.open(fpath).convert("RGB")
        pixel_values = processor(image, return_tensors="pt")["pixel_values"][0]
        crop = denormalize(pixel_values, processor)

        mask = masker.token_mask(iid).view(args.grid_size, args.grid_size).numpy()
        # Nearest-neighbour upsample so patch boundaries stay visible in the overlay.
        factor = crop.shape[0] // args.grid_size
        upsampled = np.kron(mask, np.ones((factor, factor)))

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(crop)
        axes[0].set_title(f"model input crop\nid={iid}")
        axes[1].imshow(mask, cmap="hot", vmin=0, vmax=1)
        axes[1].set_title("token mask 24x24")
        axes[2].imshow(crop)
        axes[2].imshow(upsampled, cmap="hot", alpha=0.45, vmin=0, vmax=1)
        axes[2].set_title("overlay")
        for ax in axes:
            ax.axis("off")

        out = os.path.join(args.out_dir, f"overlay_{iid}.png")
        fig.tight_layout()
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"saved {out}  foreground_fraction={float(mask.mean()):.3f}")


if __name__ == "__main__":
    main()
