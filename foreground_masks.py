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

    def __init__(self, instances_path, image_processor, grid_size=24, binarize=False):
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
        # category_id -> COCO category name (e.g. 18 -> "dog"); used for
        # per-object masks in the per-noun targeting experiments.
        self.category_names = {
            cat["id"]: cat["name"] for cat in instances.get("categories", [])
        }
        self.grid_size = grid_size
        # If True, a visual token is fully foreground (1.0) when >=1 of its
        # pixels is foreground, and 0.0 only when every pixel is background.
        self.binarize = binarize
        size = image_processor.size
        crop_size = image_processor.crop_size
        self.shortest_edge = size.get("shortest_edge", size.get("height"))
        self.crop_height = crop_size.get("height", crop_size.get("shortest_edge"))
        self.crop_width = crop_size.get("width", crop_size.get("shortest_edge"))
        if self.crop_height != self.crop_width:
            raise ValueError("This implementation expects a square CLIP crop.")

    def _decode_annotation(self, annotation, height, width):
        """Decode one COCO annotation to a full-resolution (H, W) bool array."""
        segmentation = annotation["segmentation"]
        if isinstance(segmentation, dict):
            # Uncompressed RLE (counts is a list) vs already-compressed RLE.
            if isinstance(segmentation["counts"], list):
                rle = self.coco_mask.frPyObjects(segmentation, height, width)
            else:
                rle = segmentation
        else:
            # Polygon(s)
            rle = self.coco_mask.frPyObjects(segmentation, height, width)
        decoded = self.coco_mask.decode(rle)
        if decoded.ndim == 3:
            decoded = decoded.any(axis=2)
        return decoded.astype(bool)

    def _decode_union(self, image_id):
        height, width = self.image_sizes[image_id]
        union = torch.zeros((height, width), dtype=torch.bool)
        for annotation in self.annotations[image_id]:
            union |= torch.from_numpy(self._decode_annotation(annotation, height, width))
        return union.float()

    def _grid_from_fullres(self, fullres_mask, height, width):
        """Project a full-resolution (H, W) float mask onto the token grid.

        Follows CLIP preprocessing (resize shortest edge -> centre crop ->
        area-pool to grid_size) so token i lines up with visual token i.
        """
        mask = fullres_mask.unsqueeze(0).unsqueeze(0)
        scale = self.shortest_edge / min(height, width)
        resized_h, resized_w = round(height * scale), round(width * scale)
        mask = F.interpolate(mask, size=(resized_h, resized_w), mode="nearest")
        top = (resized_h - self.crop_height) // 2
        left = (resized_w - self.crop_width) // 2
        mask = mask[:, :, top : top + self.crop_height, left : left + self.crop_width]
        mask = F.interpolate(mask, size=(self.grid_size, self.grid_size), mode="area")
        mask = mask.flatten().clamp_(0, 1)
        if self.binarize:
            # Area pooling of a binary mask is exactly the per-patch foreground
            # fraction, so ``> 0`` means "at least one foreground pixel".
            mask = (mask > 0).float()
        return mask

    def token_mask(self, image_id):
        """Return foreground occupancy for the row-major 24x24 visual tokens."""
        height, width = self.image_sizes[image_id]
        return self._grid_from_fullres(self._decode_union(image_id), height, width)

    def category_token_masks(self, image_id, name_map=None):
        """Return one token mask per object category present in the image.

        ``name_map`` optionally maps a COCO category name to a different key
        (e.g. CHAIR's canonical ``node_word``); categories mapping to the same
        key are unioned at the token level. Returns ``{key: length-576 mask}``.
        """
        height, width = self.image_sizes[image_id]
        fullres = {}
        for annotation in self.annotations[image_id]:
            name = self.category_names.get(annotation["category_id"])
            if name is None:
                continue
            key = name_map.get(name, name) if name_map else name
            decoded = torch.from_numpy(
                self._decode_annotation(annotation, height, width)
            )
            if key in fullres:
                fullres[key] |= decoded
            else:
                fullres[key] = decoded.clone()
        return {
            key: self._grid_from_fullres(mask.float(), height, width)
            for key, mask in fullres.items()
        }
