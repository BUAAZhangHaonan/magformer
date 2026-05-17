from __future__ import annotations

import json
import sqlite3
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


def _write_coco_sqlite_cache(path: Path, image_file_name: str = "cached.png") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image_payload = {
        "id": 7,
        "file_name": image_file_name,
        "width": 8,
        "height": 6,
    }
    ann_payload = {
        "id": 11,
        "image_id": 7,
        "category_id": 1,
        "bbox": [1, 2, 3, 4],
        "area": 12,
        "iscrowd": 0,
        "segmentation": [[1, 2, 4, 2, 4, 6, 1, 6]],
    }
    category_payload = {"id": 1, "name": "component", "supercategory": "component"}
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE images (
                id INTEGER PRIMARY KEY,
                file_name TEXT NOT NULL,
                json TEXT NOT NULL
            );
            CREATE TABLE annotations (
                id INTEGER PRIMARY KEY,
                image_id INTEGER NOT NULL,
                category_id INTEGER NOT NULL,
                iscrowd INTEGER NOT NULL,
                area REAL NOT NULL,
                ordinal INTEGER NOT NULL,
                json TEXT NOT NULL
            );
            CREATE TABLE categories (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                supercategory TEXT,
                json TEXT NOT NULL
            );
            CREATE TABLE manifest (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )
        conn.execute("INSERT INTO images VALUES (?, ?, ?)", (7, image_file_name, json.dumps(image_payload)))
        conn.execute("INSERT INTO annotations VALUES (?, ?, ?, ?, ?, ?, ?)", (11, 7, 1, 0, 12.0, 0, json.dumps(ann_payload)))
        conn.execute("INSERT INTO categories VALUES (?, ?, ?, ?)", (1, "component", "component", json.dumps(category_payload)))


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


def test_train_sqlite_cache_registration_returns_detectron2_dicts() -> None:
    from detectron2.data import DatasetCatalog, MetadataCatalog
    from detectron2.structures import BoxMode

    from baselines.run_official_mask2former_ecc import register_official_mask2former_datasets

    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        root = tmp_path / "cache_dataset"
        train_image_root = root / "images" / "train"
        train_image_root.mkdir(parents=True)
        (train_image_root / "cached.png").write_bytes(b"")
        cache_path = root / "cache" / "instances_train.sqlite"
        _write_coco_sqlite_cache(cache_path)
        val_ann = _write_coco(root, "instances_val.json", "images/val", "val.png")

        train_name, val_name = register_official_mask2former_datasets(
            register="cache_dataset",
            dataset_root=str(root),
            train_cache="cache/instances_train.sqlite",
            val_ann="annotations/instances_val.json",
            train_image_dir="images/train",
            val_image_dir="images/val",
            train_split="train",
            val_split="val",
            normalized_ann_dir=str(tmp_path / "diagnostics"),
        )

        records = DatasetCatalog.get(train_name)

        assert len(records) == 1
        assert records[0]["file_name"] == str((train_image_root / "cached.png").resolve())
        assert records[0]["height"] == 6
        assert records[0]["width"] == 8
        assert records[0]["image_id"] == 7
        assert records[0]["annotations"] == [
            {
                "bbox": [1, 2, 3, 4],
                "bbox_mode": BoxMode.XYWH_ABS,
                "segmentation": [[1, 2, 4, 2, 4, 6, 1, 6]],
                "category_id": 0,
                "iscrowd": 0,
            }
        ]

        train_metadata = MetadataCatalog.get(train_name)
        assert train_metadata.thing_classes == ["component"]
        assert train_metadata.thing_dataset_id_to_contiguous_id == {1: 0}
        assert train_metadata.evaluator_type == "coco"
        assert not hasattr(train_metadata, "json_file")

        val_metadata = MetadataCatalog.get(val_name)
        assert Path(val_metadata.json_file) != val_ann


def test_train_sqlite_cache_registration_fails_for_missing_cache() -> None:
    from baselines.run_official_mask2former_ecc import register_official_mask2former_datasets

    with TemporaryDirectory() as tmp:
        root = Path(tmp) / "cache_dataset_missing_cache"
        (root / "images" / "train").mkdir(parents=True)
        _write_coco(root, "instances_val.json", "images/val", "val.png")

        try:
            register_official_mask2former_datasets(
                register="cache_dataset_missing_cache",
                dataset_root=str(root),
                train_cache="cache/missing.sqlite",
                val_ann="annotations/instances_val.json",
                train_image_dir="images/train",
                val_image_dir="images/val",
                normalized_ann_dir=str(Path(tmp) / "diagnostics"),
            )
        except FileNotFoundError as exc:
            assert "SQLite cache not found" in str(exc)
        else:
            raise AssertionError("missing train cache did not raise FileNotFoundError")


def test_train_sqlite_cache_registration_fails_for_missing_image_root() -> None:
    from baselines.run_official_mask2former_ecc import register_official_mask2former_datasets

    with TemporaryDirectory() as tmp:
        root = Path(tmp) / "cache_dataset_missing_images"
        cache_path = root / "cache" / "instances_train.sqlite"
        _write_coco_sqlite_cache(cache_path)
        _write_coco(root, "instances_val.json", "images/val", "val.png")

        try:
            register_official_mask2former_datasets(
                register="cache_dataset_missing_images",
                dataset_root=str(root),
                train_cache="cache/instances_train.sqlite",
                val_ann="annotations/instances_val.json",
                train_image_dir="images/train",
                val_image_dir="images/val",
                normalized_ann_dir=str(Path(tmp) / "diagnostics"),
            )
        except FileNotFoundError as exc:
            assert "COCO image directory not found" in str(exc)
        else:
            raise AssertionError("missing train image root did not raise FileNotFoundError")


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
    test_train_sqlite_cache_registration_returns_detectron2_dicts()
    test_train_sqlite_cache_registration_fails_for_missing_cache()
    test_train_sqlite_cache_registration_fails_for_missing_image_root()
    test_explicit_val_registration_fails_for_missing_ann()
