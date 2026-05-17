from __future__ import annotations

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _write_coco(root: Path, ann_name: str, image_dir: str, file_name: str, annotations: list | None = None) -> Path:
    ann_dir = root / "annotations"
    img_dir = root / image_dir
    ann_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)
    (img_dir / file_name).write_bytes(b"")
    payload = {
        "images": [{"id": 1, "file_name": file_name, "width": 8, "height": 8}],
        "annotations": annotations or [],
        "categories": [{"id": 1, "name": "component"}],
    }
    path = ann_dir / ann_name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_normalized_coco_converts_rle_masks_to_polygons() -> None:
    import numpy as np
    from pycocotools import mask as mask_utils

    from baselines.run_official_mask2former_ecc import _normalize_coco_metadata

    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        root = tmp_path / "pseudo_real_512"
        binary_mask = np.zeros((8, 8), dtype=np.uint8)
        binary_mask[2:6, 2:6] = 1
        rle = mask_utils.encode(np.asfortranarray(binary_mask))
        rle["counts"] = rle["counts"].decode("ascii")
        ann = {
            "id": 1,
            "image_id": 1,
            "category_id": 1,
            "bbox": [2, 2, 4, 4],
            "area": 16,
            "iscrowd": 0,
            "segmentation": rle,
        }
        ann_path = _write_coco(root, "instances_target_labeled.json", "images/train", "target_labeled.png", [ann])

        normalized = _normalize_coco_metadata(ann_path, str(tmp_path / "diagnostics"))
        payload = json.loads(normalized.read_text(encoding="utf-8"))
        segmentation = payload["annotations"][0]["segmentation"]

        assert isinstance(segmentation, list)
        assert segmentation
        assert all(isinstance(poly, list) for poly in segmentation)
        assert all(len(poly) >= 6 and len(poly) % 2 == 0 for poly in segmentation)


def test_rle_to_polygons_keeps_tiny_nonempty_masks() -> None:
    import numpy as np
    from pycocotools import mask as mask_utils

    from baselines.run_official_mask2former_ecc import _rle_to_polygons

    binary_mask = np.zeros((8, 8), dtype=np.uint8)
    binary_mask[3, 4] = 1
    rle = mask_utils.encode(np.asfortranarray(binary_mask))
    rle["counts"] = rle["counts"].decode("ascii")

    polygons = _rle_to_polygons(rle)

    assert polygons
    assert all(len(poly) >= 6 and len(poly) % 2 == 0 for poly in polygons)


def test_explicit_pseudo_real_val_registration() -> None:
    from detectron2.data import DatasetCatalog, MetadataCatalog

    from baselines.run_official_mask2former_ecc import register_official_mask2former_datasets

    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        root = tmp_path / "pseudo_real_512"
        val_ann = _write_coco(root, "instances_target_unlabeled.json", "images/train", "target_unlabeled.png")
        _write_coco(root, "instances_target_labeled.json", "images/train", "target_labeled.png")

        train_name, val_name = register_official_mask2former_datasets(
            register="pseudo_real_512",
            dataset_root=str(root),
            train_ann="annotations/instances_target_labeled.json",
            val_ann="annotations/instances_target_unlabeled.json",
            train_image_dir="images/train",
            val_image_dir="images/train",
            train_split="target_labeled",
            val_split="target_unlabeled",
            normalized_ann_dir=str(tmp_path / "diagnostics"),
        )

        assert train_name == "eccpseudo_real_512_target_labeled"
        assert val_name == "eccpseudo_real_512_target_unlabeled"
        assert Path(DatasetCatalog.get(val_name)[0]["file_name"]).name == "target_unlabeled.png"

        metadata = MetadataCatalog.get(val_name)
        normalized_json = Path(metadata.json_file)
        assert normalized_json != val_ann
        normalized_payload = json.loads(normalized_json.read_text(encoding="utf-8"))
        assert normalized_payload["info"]["source_annotation"] == str(val_ann.resolve())
        assert normalized_payload["licenses"] == []
        assert "info" not in json.loads(val_ann.read_text(encoding="utf-8"))


def test_explicit_val_registration_fails_for_missing_ann() -> None:
    from baselines.run_official_mask2former_ecc import register_official_mask2former_datasets

    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        root = tmp_path / "pseudo_real_missing"
        _write_coco(root, "instances_target_labeled.json", "images/train", "target_labeled.png")

        try:
            register_official_mask2former_datasets(
                register="pseudo_real_missing",
                dataset_root=str(root),
                train_ann="annotations/instances_target_labeled.json",
                val_ann="annotations/instances_target_unlabeled.json",
                train_image_dir="images/train",
                val_image_dir="images/train",
                train_split="target_labeled",
                val_split="target_unlabeled",
                normalized_ann_dir=str(tmp_path / "diagnostics"),
            )
        except FileNotFoundError as exc:
            assert "COCO annotation file not found" in str(exc)
        else:
            raise AssertionError("missing val annotation did not raise FileNotFoundError")


if __name__ == "__main__":
    test_normalized_coco_converts_rle_masks_to_polygons()
    test_rle_to_polygons_keeps_tiny_nonempty_masks()
    test_explicit_pseudo_real_val_registration()
    test_explicit_val_registration_fails_for_missing_ann()
