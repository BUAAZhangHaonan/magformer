from __future__ import annotations


def _add_uoais_to_syspath() -> None:
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    uoais_root = repo_root / "baselines" / "icra_2026" / "uoais"

    import sys

    sys.path.insert(0, str(uoais_root))


def test_uoais_annotations_to_instances_modal_fallback_polygon_masks() -> None:
    """
    UOAIS original code expects amodal/visible/occluded masks.
    For ECC baselines we need it to accept standard COCO modal masks:
      - annotations only have `segmentation`
      - `visible_mask`/`occluded_mask` may be missing
      - `occluded_rate` may be missing
    """
    _add_uoais_to_syspath()

    import torch
    from detectron2.structures import BitMasks, BoxMode

    from adet.data.detection_utils import annotations_to_instances

    annos = [
        {
            "bbox": [1.0, 1.0, 8.0, 8.0],
            "bbox_mode": BoxMode.XYXY_ABS,
            "category_id": 0,
            # One rectangle polygon.
            "segmentation": [[1.0, 1.0, 8.0, 1.0, 8.0, 8.0, 1.0, 8.0]],
        }
    ]
    inst = annotations_to_instances(annos, image_size=(10, 10), mask_format="polygon", amodal=False)

    assert inst.has("gt_boxes")
    assert inst.has("gt_classes")
    assert inst.has("gt_occludeds")
    assert inst.gt_occludeds.dtype == torch.int64
    assert int(inst.gt_occludeds[0].item()) == 0

    assert inst.has("gt_masks")
    assert isinstance(inst.gt_masks, BitMasks)
    assert tuple(inst.gt_masks.tensor.shape) == (1, 10, 10)

