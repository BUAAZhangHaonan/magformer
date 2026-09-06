from pathlib import Path


def test_train_script_imports_coco_rgbd_dataset() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    train_py = (repo_root / "tools" / "train.py").read_text(encoding="utf-8")
    assert "from magformer.data import CocoRgbdDataset" in train_py
