from __future__ import annotations

import sys
from pathlib import Path

import pytest


def test_explicit_pseudo_real_target150_eval_split_registration(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from baselines import run_detectron2_ecc as runner

    root = tmp_path / "pseudo_real_512"
    calls: list[dict] = []

    def fake_register_coco_split(**kwargs) -> None:
        calls.append(kwargs)

    monkeypatch.setattr(runner, "_register_coco_split", fake_register_coco_split, raising=False)

    for val_split in ("val28", "remaining75"):
        calls.clear()

        train_name, val_name = runner.register_detectron2_datasets(
            register="pseudo_real_512",
            dataset_root=str(root),
            train_ann="annotations/instances_target150.json",
            val_ann=f"annotations/instances_{val_split}.json",
            train_image_dir="images/target150",
            val_image_dir=f"images/{val_split}",
            train_split="target150",
            val_split=val_split,
            normalized_ann_dir=str(tmp_path / "normalized"),
        )

        assert train_name == "eccpseudo_real_512_target150"
        assert val_name == f"eccpseudo_real_512_{val_split}"
        assert calls == [
            {
                "name": "eccpseudo_real_512_target150",
                "ann_file": (root / "annotations" / "instances_target150.json").resolve(),
                "image_root": (root / "images" / "target150").resolve(),
                "normalized_ann_dir": str(tmp_path / "normalized"),
            },
            {
                "name": f"eccpseudo_real_512_{val_split}",
                "ann_file": (root / "annotations" / f"instances_{val_split}.json").resolve(),
                "image_root": (root / "images" / val_split).resolve(),
                "normalized_ann_dir": str(tmp_path / "normalized"),
            },
        ]


def test_explicit_split_registration_requires_all_ann_and_image_args(tmp_path: Path) -> None:
    from baselines import run_detectron2_ecc as runner

    with pytest.raises(ValueError, match="requires --train-ann, --val-ann, --train-image-dir, and --val-image-dir"):
        runner.register_detectron2_datasets(
            register="pseudo_real_512",
            dataset_root=str(tmp_path / "pseudo_real_512"),
            train_ann="annotations/instances_target150.json",
            val_ann=None,
            train_image_dir="images/target150",
            val_image_dir="images/val28",
            train_split="target150",
            val_split="val28",
        )


def test_default_registration_keeps_legacy_ecc_register(monkeypatch: pytest.MonkeyPatch) -> None:
    from baselines import run_detectron2_ecc as runner

    calls = []

    def fake_register_ecc_coco(register: str, dataset_root: str | None):
        calls.append((register, dataset_root))
        return "ecc0831_1k_train", "ecc0831_1k_val"

    monkeypatch.setattr(runner, "register_ecc_coco", fake_register_ecc_coco)
    monkeypatch.setattr(
        runner,
        "_register_coco_split",
        lambda **kwargs: pytest.fail("default registration should not use explicit split registration"),
        raising=False,
    )

    assert runner.register_detectron2_datasets(register="0831", dataset_root=None) == (
        "ecc0831_1k_train",
        "ecc0831_1k_val",
    )
    assert calls == [("0831", None)]


def test_main_accepts_explicit_split_args(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from baselines import run_detectron2_ecc as runner

    detectron2_root = tmp_path / "detectron2"
    train_py = detectron2_root / "tools" / "train_net.py"
    train_py.parent.mkdir(parents=True)
    train_py.write_text("", encoding="utf-8")

    captured = {}

    def fake_register_detectron2_datasets(**kwargs):
        captured["register_kwargs"] = kwargs
        return "eccpseudo_real_512_target150", "eccpseudo_real_512_val28"

    def fake_run_path(path: str, run_name: str):
        captured["run_path"] = path
        captured["run_name"] = run_name
        captured["sys_argv"] = list(sys.argv)

    monkeypatch.setattr(runner, "register_detectron2_datasets", fake_register_detectron2_datasets, raising=False)
    monkeypatch.setattr(runner.runpy, "run_path", fake_run_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_detectron2_ecc.py",
            "--register",
            "pseudo_real_512",
            "--dataset-root",
            str(tmp_path / "pseudo_real_512"),
            "--detectron2-root",
            str(detectron2_root),
            "--train-ann",
            "annotations/instances_target150.json",
            "--val-ann",
            "annotations/instances_val28.json",
            "--train-image-dir",
            "images/target150",
            "--val-image-dir",
            "images/val28",
            "--train-split",
            "target150",
            "--val-split",
            "val28",
            "--normalized-ann-dir",
            str(tmp_path / "normalized"),
            "--",
            "--config-file",
            "configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
        ],
    )

    runner.main()

    assert captured["register_kwargs"] == {
        "register": "pseudo_real_512",
        "dataset_root": str(tmp_path / "pseudo_real_512"),
        "train_ann": "annotations/instances_target150.json",
        "val_ann": "annotations/instances_val28.json",
        "train_image_dir": "images/target150",
        "val_image_dir": "images/val28",
        "train_split": "target150",
        "val_split": "val28",
        "normalized_ann_dir": str(tmp_path / "normalized"),
    }
    assert captured["run_path"] == str(train_py.resolve())
    assert captured["run_name"] == "__main__"
    assert captured["sys_argv"] == [
        str(train_py),
        "--config-file",
        "configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
    ]
