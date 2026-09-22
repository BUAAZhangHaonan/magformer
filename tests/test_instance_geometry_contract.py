from __future__ import annotations

import json

import cv2
import numpy as np
import pytest
from pycocotools import mask as coco_mask

from magformer.data.dataset import CocoRgbdDataset
from magformer.data.transforms import (
    FixedSizeCrop,
    ResizeScale,
    synchronize_instances_after_geometry,
)


def _sample(height: int, width: int, masks: np.ndarray, **metadata):
    result = {
        "image": np.zeros((height, width, 3), dtype=np.uint8),
        "depth": np.zeros((height, width), dtype=np.float32),
        "masks": masks,
        "boxes": np.full((masks.shape[2], 4), 999.0, dtype=np.float32),
        "labels": np.arange(masks.shape[2], dtype=np.int64),
    }
    result.update(metadata)
    return result


def test_lsj_1024_to_427_two_pixel_mask_erasure_produces_valid_empty_target() -> None:
    """Replay the reproduced geometry failure without changing its mask geometry."""
    masks = np.zeros((1024, 1024, 1), dtype=bool)
    masks[393, 789:791, 0] = True
    result = _sample(1024, 1024, masks)

    transformed = ResizeScale(
        min_scale=427.0 / 1024.0,
        max_scale=427.0 / 1024.0,
        target_size=1024,
    )(result)

    assert transformed["masks"].shape == (427, 427, 0)
    assert transformed["labels"].shape == (0,)
    assert transformed["boxes"].shape == (0, 4)


def test_crop_erased_instance_is_removed_from_every_instance_field() -> None:
    masks = np.zeros((6, 6, 2), dtype=bool)
    masks[2:4, 2:5, 0] = True
    masks[5, 5, 1] = True
    result = _sample(
        6,
        6,
        masks,
        areas=np.array([123, 456], dtype=np.int64),
        iscrowd=np.array([0, 1], dtype=np.int64),
        annotation_ids=[10, 20],
    )
    # The second source bbox remains positive after the center crop even
    # though its true one-pixel mask falls just outside the crop window.
    result["boxes"] = np.array(
        [[2.0, 2.0, 5.0, 4.0], [3.0, 3.0, 4.0, 4.0]], dtype=np.float32
    )

    transformed = FixedSizeCrop((4, 4), random_crop=False)(result)

    assert transformed["masks"].shape == (4, 4, 1)
    np.testing.assert_array_equal(transformed["labels"], [0])
    np.testing.assert_array_equal(transformed["iscrowd"], [0])
    assert transformed["annotation_ids"] == [10]
    np.testing.assert_array_equal(transformed["areas"], [6])
    np.testing.assert_array_equal(transformed["boxes"], [[1.0, 1.0, 4.0, 3.0]])


def test_boxes_are_recomputed_from_retained_transformed_masks() -> None:
    masks = np.zeros((5, 7, 1), dtype=bool)
    masks[1:4, 2:6, 0] = True
    result = _sample(5, 7, masks)

    synchronized = synchronize_instances_after_geometry(result)

    np.testing.assert_array_equal(synchronized["boxes"], [[2.0, 1.0, 6.0, 4.0]])


def test_all_instances_empty_is_a_valid_zero_instance_target() -> None:
    masks = np.zeros((4, 5, 2), dtype=bool)
    result = _sample(
        4,
        5,
        masks,
        areas=np.array([1, 2], dtype=np.int64),
        iscrowd=(0, 0),
        annotation_ids=np.array([7, 8], dtype=np.int64),
    )

    synchronized = synchronize_instances_after_geometry(result)

    assert synchronized["masks"].shape == (4, 5, 0)
    assert synchronized["labels"].shape == (0,)
    assert synchronized["boxes"].shape == (0, 4)
    assert synchronized["areas"].shape == (0,)
    assert synchronized["iscrowd"] == ()
    assert synchronized["annotation_ids"].shape == (0,)


def test_empty_decoded_annotation_fails_loudly_with_sample_context() -> None:
    dataset = CocoRgbdDataset.__new__(CocoRgbdDataset)
    dataset.category_ids = [1]
    dataset.category_id_to_label = {1: 0}
    empty = coco_mask.encode(np.asfortranarray(np.zeros((4, 5), dtype=np.uint8)))

    with pytest.raises(ValueError) as exc_info:
        dataset._build_instances(
            [{"id": 889214, "category_id": 1, "segmentation": empty}],
            (4, 5),
            sample_id=16271,
            image_filename="687106149022_100_scene_000006_016271_v13.png",
        )

    message = str(exc_info.value)
    assert "Decoded instance mask is empty" in message
    assert "sample_id=16271" in message
    assert "889214" in message
    assert "687106149022_100_scene_000006_016271_v13.png" in message


def test_evaluation_dataset_rejects_empty_decoded_annotation_before_inference(tmp_path) -> None:
    root = tmp_path / "dataset"
    (root / "images" / "val").mkdir(parents=True)
    (root / "depth" / "val").mkdir(parents=True)
    (root / "annotations").mkdir(parents=True)
    filename = "empty.png"
    assert cv2.imwrite(
        str(root / "images" / "val" / filename), np.zeros((4, 5, 3), dtype=np.uint8)
    )
    np.save(root / "depth" / "val" / "empty.npy", np.ones((4, 5), dtype=np.float32))
    segmentation = coco_mask.encode(np.asfortranarray(np.zeros((4, 5), dtype=np.uint8)))
    segmentation["counts"] = segmentation["counts"].decode("ascii")
    (root / "annotations" / "instances_val.json").write_text(
        json.dumps(
            {
                "images": [{"id": 16271, "file_name": filename, "width": 5, "height": 4}],
                "annotations": [
                    {
                        "id": 889214,
                        "image_id": 16271,
                        "category_id": 1,
                        "bbox": [0, 0, 1, 1],
                        "segmentation": segmentation,
                        "iscrowd": 0,
                    }
                ],
                "categories": [{"id": 1, "name": "component"}],
            }
        ),
        encoding="utf-8",
    )

    dataset = CocoRgbdDataset(
        dataset_root=root,
        ann_file="annotations/instances_val.json",
        split="val",
        is_train=False,
    )
    with pytest.raises(ValueError, match="sample_id=16271.*annotation_id=889214"):
        dataset[0]
