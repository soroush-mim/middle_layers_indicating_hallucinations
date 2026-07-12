"""COCO-instance masks aligned to LLaVA-1.5's 24x24 CLIP token grid."""

import json
from collections import defaultdict

import torch
import torch.nn.functional as F
class COCOForegroundMasker:
    """Build union-of-instance foreground masks from COCO instance annotations.

    The mask follows the normal CLIP preprocessing used by this repository:
    resize shortest edge, centre crop, then area-pool to the visual-token grid.
    ``pycocotools`` is used so both polygons and RLE/crowd annotations work.
    """

    def __init__(self, instances_path, image_processor, grid_size=24):
        try:
            from pycocotools import mask as coco_mask
        except ImportError as exc:
            raise ImportError(
                "COCO foreground guidance requires pycocotools. Install it with "
                "`pip install pycocotools`."
            ) from exc

        with open(instances_path) as f:
            instances = json.load(f)
        self.coco_mask = coco_mask
        self.annotations = defaultdict(list)
        for annotation in instances["annotations"]:
            self.annotations[annotation["image_id"]].append(annotation)
        self.image_sizes = {
            image["id"]: (image["height"], image["width"])
            for image in instances["images"]
        }
        self.grid_size = grid_size
        size = image_processor.size
        crop_size = image_processor.crop_size
        self.shortest_edge = size.get("shortest_edge", size.get("height"))
        self.crop_height = crop_size.get("height", crop_size.get("shortest_edge"))
        self.crop_width = crop_size.get("width", crop_size.get("shortest_edge"))
        if self.crop_height != self.crop_width:
            raise ValueError("This implementation expects a square CLIP crop.")

    def _decode_union(self, image_id):
        height, width = self.image_sizes[image_id]
        union = torch.zeros((height, width), dtype=torch.bool)
        for annotation in self.annotations[image_id]:
            segmentation = annotation["segmentation"]
            if isinstance(segmentation, dict):
                decoded = self.coco_mask.decode(segmentation)
            else:
                rles = self.coco_mask.frPyObjects(segmentation, height, width)
                decoded = self.coco_mask.decode(rles)
            if decoded.ndim == 3:
                decoded = decoded.any(axis=2)
            union |= torch.from_numpy(decoded.astype("bool"))
        return union.float()

    def token_mask(self, image_id):
        """Return foreground occupancy for the row-major 24x24 visual tokens."""
        mask = self._decode_union(image_id).unsqueeze(0).unsqueeze(0)
        height, width = self.image_sizes[image_id]
        scale = self.shortest_edge / min(height, width)
        resized_h, resized_w = round(height * scale), round(width * scale)
        mask = F.interpolate(mask, size=(resized_h, resized_w), mode="nearest")
        top = (resized_h - self.crop_height) // 2
        left = (resized_w - self.crop_width) // 2
        mask = mask[:, :, top : top + self.crop_height, left : left + self.crop_width]
        mask = F.interpolate(mask, size=(self.grid_size, self.grid_size), mode="area")
        return mask.flatten().clamp_(0, 1)
