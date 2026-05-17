from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO

from magformer.data.coco_loader_cache import CocoLoaderCache


def verify_coco_loader_cache(
    source_json: str | Path,
    cache_path: str | Path,
    sample_count: int = 10,
) -> dict[str, Any]:
    coco = COCO(str(source_json))
    cache = CocoLoaderCache(cache_path)

    json_image_ids = sorted(coco.imgs.keys())
    cache_image_ids = cache.getImgIds()
    if json_image_ids != cache_image_ids:
        raise ValueError("image id parity failed between JSON COCO and cache")

    json_cat_ids = coco.getCatIds()
    cache_cat_ids = cache.getCatIds()
    if json_cat_ids != cache_cat_ids:
        raise ValueError("category id parity failed between JSON COCO and cache")

    json_ann_ids = coco.getAnnIds()
    cache_ann_ids = cache.getAnnIds()
    if json_ann_ids != cache_ann_ids:
        raise ValueError("annotation id parity failed between JSON COCO and cache")

    sample_ids = _sample_image_ids(json_image_ids, sample_count)
    for image_id in sample_ids:
        if coco.loadImgs(image_id) != cache.loadImgs(image_id):
            raise ValueError(f"image metadata parity failed for image_id={image_id}")
        ann_ids = coco.getAnnIds(imgIds=image_id)
        if ann_ids != cache.getAnnIds(imgIds=image_id):
            raise ValueError(f"annotation id parity failed for image_id={image_id}")
        json_anns = coco.loadAnns(ann_ids)
        cache_anns = cache.loadAnns(ann_ids)
        if json_anns != cache_anns:
            raise ValueError(f"annotation JSON parity failed for image_id={image_id}")
        image = coco.loadImgs(image_id)[0]
        height = int(image["height"])
        width = int(image["width"])
        for ann in json_anns:
            cached_ann = cache.loadAnns(ann["id"])[0]
            left = _decode_mask(ann["segmentation"], height, width)
            right = _decode_mask(cached_ann["segmentation"], height, width)
            if not np.array_equal(left, right):
                raise ValueError(f"mask decode parity failed for annotation_id={ann['id']}")

    result = {
        "image_count": len(json_image_ids),
        "annotation_count": len(json_ann_ids),
        "category_count": len(json_cat_ids),
        "sample_image_ids": sample_ids,
    }
    return result


def _sample_image_ids(image_ids: list[int], sample_count: int) -> list[int]:
    if sample_count <= 0 or len(image_ids) <= sample_count:
        return image_ids
    if sample_count == 1:
        return [image_ids[0]]
    positions = np.linspace(0, len(image_ids) - 1, num=sample_count, dtype=int)
    return [image_ids[int(position)] for position in positions]


def _decode_mask(segmentation: Any, height: int, width: int) -> np.ndarray:
    if isinstance(segmentation, list):
        rles = coco_mask.frPyObjects(segmentation, height, width)
        mask = coco_mask.decode(rles)
        if mask.ndim == 3:
            mask = mask.any(axis=2)
        return mask.astype(bool)
    mask = coco_mask.decode(segmentation)
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask.astype(bool)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify first-phase COCO loader cache parity")
    parser.add_argument("source_json", type=Path)
    parser.add_argument("cache_path", type=Path)
    parser.add_argument("--sample-count", type=int, default=10)
    args = parser.parse_args()
    result = verify_coco_loader_cache(args.source_json, args.cache_path, args.sample_count)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
