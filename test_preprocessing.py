"""Quantitatively test the COCOForegroundMasker preprocessing geometry.

The masker reimplements CLIP preprocessing (resize shortest edge -> center crop
-> area-pool to 24x24 -> row-major flatten) to map a full-resolution COCO mask
onto visual tokens. This script checks that reimplementation against the REAL
image processor, using the processor itself as an oracle:

    1. Paint the GT foreground onto the raw image with a solid colour.
    2. Push BOTH the painted and clean images through the real image_processor.
    3. Where the processor placed the paint (per-pixel diff) reveals where the
       foreground actually lands in the 336x336 model input; pool that to 24x24.
    4. Compare this "empirical" mask with masker.token_mask().

High identity IoU / correlation => geometry is correct. The orientation table
is the key diagnostic: if a flipped/transposed variant scores higher than
"identity", the token ordering in foreground_masks.py is wrong.

Only the CLIP image processor is loaded (not the 7B LLM).

Example:
    python test_preprocessing.py \
        --instances-path /path/to/annotations/instances_val2014.json \
        --data-path /path/to/COCO/val2014 \
        --num-samples 8
"""

import argparse
import os

import numpy as np
import torch
from PIL import Image
from transformers import CLIPImageProcessor

from foreground_masks import COCOForegroundMasker

PAINT_COLOR = np.array([255, 0, 255], dtype=np.uint8)  # magenta, rare in photos


def coco_filename(image_id):
    return f"COCO_val2014_{image_id:012d}.jpg"


def pool_to_grid(arr, grid_size):
    """Mean-pool a (S, S) array to (grid, grid); S must be divisible by grid."""
    s = arr.shape[0]
    assert s % grid_size == 0, f"{s} not divisible by {grid_size}"
    factor = s // grid_size
    return arr.reshape(grid_size, factor, grid_size, factor).mean(axis=(1, 3))


def iou(a_bin, b_bin):
    inter = np.logical_and(a_bin, b_bin).sum()
    union = np.logical_or(a_bin, b_bin).sum()
    return float(inter / union) if union else 1.0


def pearson(a, b):
    a, b = a.ravel(), b.ravel()
    if a.std() < 1e-8 or b.std() < 1e-8:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def empirical_token_mask(image, gt, processor, grid_size):
    """Where does the processor place the painted foreground, at the token grid?"""
    painted = np.array(image).copy()
    painted[gt] = PAINT_COLOR
    painted_img = Image.fromarray(painted)

    pv_clean = processor(image, return_tensors="pt")["pixel_values"][0]
    pv_paint = processor(painted_img, return_tensors="pt")["pixel_values"][0]

    diff = (pv_paint - pv_clean).abs().sum(dim=0).numpy()  # (336, 336)
    pooled = pool_to_grid(diff, grid_size)
    peak = pooled.max()
    return pooled / peak if peak > 0 else pooled


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--instances-path", required=True)
    ap.add_argument("--data-path", required=True, help="folder with COCO val2014 jpgs")
    ap.add_argument("--image-processor", default="openai/clip-vit-large-patch14-336")
    ap.add_argument("--image-ids", type=int, nargs="*", default=None)
    ap.add_argument("--num-samples", type=int, default=8)
    ap.add_argument("--grid-size", type=int, default=24)
    ap.add_argument("--threshold", type=float, default=0.5, help="binarization for IoU")
    args = ap.parse_args()

    processor = CLIPImageProcessor.from_pretrained(args.image_processor)
    masker = COCOForegroundMasker(args.instances_path, processor, grid_size=args.grid_size)

    if args.image_ids:
        image_ids = args.image_ids
    else:
        image_ids = []
        for iid in masker.annotations:
            if os.path.exists(os.path.join(args.data_path, coco_filename(iid))):
                image_ids.append(iid)
            if len(image_ids) >= args.num_samples:
                break

    orientations = {
        "identity": lambda m: m,
        "flip_vertical": np.flipud,
        "flip_horizontal": np.fliplr,
        "transpose": lambda m: m.T,
        "rot180": lambda m: np.flip(m),
    }

    corr_sum = 0.0
    identity_iou_sum = 0.0
    n = 0
    orientation_wins = {k: 0 for k in orientations}

    header = f"{'image_id':>10} {'corr':>7} {'IoU':>7} | best orientation (IoU)"
    print(header)
    print("-" * len(header))

    for iid in image_ids:
        fpath = os.path.join(args.data_path, coco_filename(iid))
        if not os.path.exists(fpath):
            print(f"{iid:>10}   skip: file not found")
            continue

        image = Image.open(fpath).convert("RGB")
        gt = masker._decode_union(iid).bool().numpy()  # (H, W)

        if gt.shape != (image.height, image.width):
            print(f"{iid:>10}   skip: json size {gt.shape} != image {(image.height, image.width)}")
            continue
        if gt.sum() == 0:
            print(f"{iid:>10}   skip: empty GT mask")
            continue

        token = masker.token_mask(iid).view(args.grid_size, args.grid_size).numpy()
        emp = empirical_token_mask(image, gt, processor, args.grid_size)

        token_bin = token >= args.threshold
        # Score token (fixed) against each orientation of the empirical oracle.
        ious = {name: iou(token_bin, fn(emp) >= args.threshold) for name, fn in orientations.items()}
        best = max(ious, key=ious.get)
        orientation_wins[best] += 1

        corr = pearson(token, emp)
        corr_sum += 0.0 if np.isnan(corr) else corr
        identity_iou_sum += ious["identity"]
        n += 1

        flag = "" if best == "identity" else "  <-- ORIENTATION MISMATCH"
        print(f"{iid:>10} {corr:>7.3f} {ious['identity']:>7.3f} | {best} ({ious[best]:.3f}){flag}")

    if n == 0:
        print("\nNo usable samples.")
        return

    print("-" * len(header))
    print(f"mean correlation (token vs oracle): {corr_sum / n:.3f}")
    print(f"mean identity IoU                 : {identity_iou_sum / n:.3f}")
    print(f"best-orientation counts over {n} samples: {orientation_wins}")
    if orientation_wins["identity"] == n:
        print("PASS: 'identity' wins on every sample -> token ordering is correct.")
    else:
        print("CHECK: a non-identity orientation won on some samples -> inspect flatten order.")


if __name__ == "__main__":
    main()
