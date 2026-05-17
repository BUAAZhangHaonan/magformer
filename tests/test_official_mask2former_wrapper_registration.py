from __future__ import annotations

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _write_coco(root: Path, ann_name: str, image_dir: str, file_name: str) -> Path:
    ann_dir = root / "annotations"
    img_dir = root / image_dir
    ann_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)
    (img_dir / file_name).write_bytes(b"")
    payload = {
        "images": [{"id": 1, "file_name": file_name, "width": 8, "height": 8}],
        "annotations": [],
        "categories": [{"id": 1, "name": "component"}],
    }
    path = ann_dir / ann_name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


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
    test_explicit_pseudo_real_val_registration()
    test_explicit_val_registration_fails_for_missing_ann()
