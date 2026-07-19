"""Bridge CHAIR's object vocabulary to the per-noun targeting experiments.

Wraps a CHAIR evaluator so both Version A (teacher-forced oracle) and Version B
(causal lock generator) can, from generated text, recover the canonical COCO
object names, decide whether each is a real GT object or a hallucination, and
line them up with the per-category segmentation masks.
"""

import os
import pickle

from chair import CHAIR


def load_evaluator(cache="chair.pkl", coco_path=None):
    """Load the cached CHAIR evaluator, or build (and cache) one from coco_path."""
    if cache and os.path.exists(cache):
        print(f"loaded CHAIR evaluator from cache: {cache}")
        return pickle.load(open(cache, "rb"))
    if not coco_path:
        raise SystemExit(f"No CHAIR cache at {cache!r}; pass --coco_path to build one.")
    print("building CHAIR evaluator from scratch...")
    evaluator = CHAIR(coco_path)
    if cache:
        pickle.dump(evaluator, open(cache, "wb"))
    return evaluator


class NounResolver:
    """Detect COCO objects in generated text and label them real/hallucinated."""

    def __init__(self, evaluator):
        self.ev = evaluator

    def gt_objects(self, image_id):
        """Set of canonical node_words that are ground-truth for the image."""
        return set(self.ev.imid_to_objects.get(image_id, set()))

    def name_map(self):
        """COCO category name -> CHAIR canonical node_word (for mask keying).

        ``inverse_synonym_dict`` already maps every surface/category name to its
        canonical node_word, so it is exactly the mapping needed to key the
        per-category masks the same way generated objects resolve.
        """
        return self.ev.inverse_synonym_dict

    def detect(self, text):
        """Ordered list of canonical node_words mentioned in ``text``."""
        _, node_words, _, _ = self.ev.caption_to_words(text)
        return node_words
