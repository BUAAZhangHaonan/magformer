from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def test_stardist_runner_help_works_as_script() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "baselines" / "run_stardist_instance_ecc.py"
    res = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--image-size" in res.stdout
    assert "--prob-thresh" in res.stdout
    assert "--nms-thresh" in res.stdout
    assert "--ram-limit-pct" in res.stdout
    assert "--allow-cpu" in res.stdout


def test_stardist_shell_wrapper_uses_cuda_env_and_ram_guard(tmp_path: Path) -> None:
    import json

    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_0831_1k_20ep_1024_revisit_stardist_inst.sh"
    dataset_root = tmp_path / "ecc"
    (dataset_root / "annotations").mkdir(parents=True, exist_ok=True)
    payload = {"images": [{"id": 1, "file_name": "sample.png", "width": 8, "height": 8}], "annotations": [], "categories": []}
    (dataset_root / "annotations" / "instances_train.json").write_text(json.dumps(payload), encoding="utf-8")

    res = subprocess.run(
        [
            "bash",
            str(script),
            "--dataset-root",
            str(dataset_root),
            "--output-root",
            str(tmp_path / "out"),
            "--image-size",
            "512",
            "--epochs",
            "100",
            "--batch",
            "4",
            "--num-workers",
            "0",
            "--ram-limit-pct",
            "50",
            "--dry-run",
        ],
        cwd=str(repo_root),
        check=True,
        capture_output=True,
        text=True,
    )

    assert "TF_FORCE_GPU_ALLOW_GROWTH=true" in res.stdout
    assert "conda run -n 'stardist' python baselines/run_stardist_instance_ecc.py" in res.stdout
    assert "--ram-limit-pct 50" in res.stdout
    assert "--num-workers 0" in res.stdout


def test_stardist_tensorflow_guard_rejects_cpu_only(monkeypatch: pytest.MonkeyPatch) -> None:
    from baselines import run_stardist_instance_ecc as runner

    class FakeConfig:
        @staticmethod
        def list_physical_devices(kind: str):
            return []

        class experimental:
            @staticmethod
            def set_memory_growth(device, enabled):
                raise AssertionError("no GPUs should be configured")

            @staticmethod
            def get_memory_growth(device):
                return False

    class FakeTest:
        @staticmethod
        def is_built_with_cuda() -> bool:
            return False

    class FakeTF:
        __version__ = "0.fake"
        config = FakeConfig()
        test = FakeTest()

    monkeypatch.setitem(sys.modules, "tensorflow", FakeTF)

    with pytest.raises(RuntimeError, match="TensorFlow does not see a GPU"):
        runner.configure_tensorflow_runtime(require_gpu=True, allow_cpu=False)


def test_stardist_tensorflow_guard_allows_cpu_debug_override(monkeypatch: pytest.MonkeyPatch) -> None:
    from baselines import run_stardist_instance_ecc as runner

    class FakeConfig:
        @staticmethod
        def list_physical_devices(kind: str):
            return []

        class experimental:
            @staticmethod
            def set_memory_growth(device, enabled):
                raise AssertionError("no GPUs should be configured")

            @staticmethod
            def get_memory_growth(device):
                return False

    class FakeTest:
        @staticmethod
        def is_built_with_cuda() -> bool:
            return False

    class FakeTF:
        __version__ = "0.fake"
        config = FakeConfig()
        test = FakeTest()

    monkeypatch.setitem(sys.modules, "tensorflow", FakeTF)

    info = runner.configure_tensorflow_runtime(require_gpu=True, allow_cpu=True)

    assert info["gpu_visible"] is False
    assert info["physical_gpu_count"] == 0
    assert info["memory_growth"] == []


def test_stardist_ram_guard_raises_when_limit_exceeded(monkeypatch: pytest.MonkeyPatch) -> None:
    from baselines import run_stardist_instance_ecc as runner

    monkeypatch.setattr(runner, "_current_ram_used_pct", lambda: 51.0)

    with pytest.raises(RuntimeError, match="RAM usage 51.0% exceeds limit 50.0%"):
        runner.enforce_ram_limit(50.0, label="before-load")


def test_stardist_ram_guard_returns_snapshot_when_below_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    from baselines import run_stardist_instance_ecc as runner

    monkeypatch.setattr(runner, "_current_ram_used_pct", lambda: 12.5)

    assert runner.enforce_ram_limit(50.0, label="before-load") == 12.5


def test_stardist_prediction_details_to_coco_rows_uses_probabilities_and_binary_masks() -> None:
    from baselines.stardist_instance_utils import stardist_prediction_to_coco_rows

    labels = np.zeros((6, 6), dtype=np.int32)
    labels[1:3, 1:4] = 1
    labels[3:5, 3:5] = 2
    details = {"prob": np.asarray([0.91, 0.37], dtype=np.float32)}

    rows = stardist_prediction_to_coco_rows(image_id=7, labels=labels, details=details, score_threshold=0.05)

    assert len(rows) == 2
    assert rows[0]["image_id"] == 7
    assert rows[0]["category_id"] == 1
    assert rows[0]["score"] == pytest.approx(0.91, rel=1e-6)
    assert rows[0]["mask"].dtype == np.uint8
    assert rows[0]["mask"].sum() == 6
    assert rows[1]["score"] == pytest.approx(0.37, rel=1e-6)
    assert rows[1]["bbox"] == [3.0, 3.0, 2.0, 2.0]


def test_stardist_prediction_to_coco_rows_resizes_to_original_output_size() -> None:
    from baselines.stardist_instance_utils import stardist_prediction_to_coco_rows

    labels = np.zeros((4, 4), dtype=np.int32)
    labels[1:3, 1:3] = 1
    rows = stardist_prediction_to_coco_rows(
        image_id=7,
        labels=labels,
        details={"prob": np.asarray([0.91], dtype=np.float32)},
        output_size=(8, 8),
    )

    assert len(rows) == 1
    assert rows[0]["bbox"] == [2.0, 2.0, 4.0, 4.0]


def test_stardist_ecc_split_normalizes_images_to_float(tmp_path: Path) -> None:
    import json

    from baselines.stardist_instance_utils import load_stardist_ecc_split

    dataset_root = tmp_path / "ecc"
    (dataset_root / "annotations").mkdir(parents=True, exist_ok=True)
    (dataset_root / "images" / "train").mkdir(parents=True, exist_ok=True)
    image_name = "sample.png"
    Image.new("RGB", (8, 8), color=(128, 64, 32)).save(dataset_root / "images" / "train" / image_name)
    payload = {
        "images": [{"id": 1, "file_name": image_name, "width": 8, "height": 8}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "segmentation": [[1, 1, 4, 1, 4, 4, 1, 4]],
                "area": 9,
                "bbox": [1, 1, 3, 3],
                "iscrowd": 0,
            }
        ],
        "categories": [{"id": 1, "name": "component"}],
    }
    (dataset_root / "annotations" / "instances_train.json").write_text(json.dumps(payload), encoding="utf-8")

    images, labels, records = load_stardist_ecc_split(dataset_root, "train", image_size=8)

    assert records[0]["image_id"] == 1
    assert labels[0].shape == (8, 8)
    assert images[0].dtype == np.float32
    assert 0.0 <= float(images[0].min()) <= float(images[0].max()) <= 1.0


def test_stardist_runner_smoke_with_fake_backend_writes_standard_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json

    from baselines import run_stardist_instance_ecc as runner

    dataset_root = tmp_path / "ecc"
    (dataset_root / "annotations").mkdir(parents=True, exist_ok=True)
    (dataset_root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (dataset_root / "images" / "val").mkdir(parents=True, exist_ok=True)

    image_name = "sample.png"
    Image.new("RGB", (8, 8), color=(10, 20, 30)).save(dataset_root / "images" / "train" / image_name)
    Image.new("RGB", (8, 8), color=(10, 20, 30)).save(dataset_root / "images" / "val" / image_name)
    payload = {
        "images": [{"id": 1, "file_name": image_name, "width": 8, "height": 8}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "segmentation": [[1, 1, 4, 1, 4, 4, 1, 4]],
                "area": 9,
                "bbox": [1, 1, 3, 3],
                "iscrowd": 0,
            }
        ],
        "categories": [{"id": 1, "name": "component"}],
    }
    (dataset_root / "annotations" / "instances_train.json").write_text(json.dumps(payload), encoding="utf-8")
    (dataset_root / "annotations" / "instances_val.json").write_text(json.dumps(payload), encoding="utf-8")

    class FakeKerasModel:
        trainable_weights: list[np.ndarray] = [np.ones((2, 3), dtype=np.float32)]

        def save_weights(self, path: str) -> None:
            Path(path).write_text("fake-weights", encoding="utf-8")

        def count_params(self) -> int:
            return 6

    class FakeModel:
        def __init__(self) -> None:
            self.keras_model = FakeKerasModel()
            self.train_calls: list[dict[str, object]] = []

        def train(
            self,
            X,
            Y,
            *,
            validation_data=None,
            classes=None,
            augmenter=None,
            seed=None,
            epochs=None,
            steps_per_epoch=None,
            workers=None,
        ):
            self.train_calls.append(
                {
                    "X_len": len(X),
                    "Y_len": len(Y),
                    "validation_len": len(validation_data[0]) if validation_data is not None else None,
                    "epochs": epochs,
                    "steps_per_epoch": steps_per_epoch,
                    "workers": workers,
                }
            )
            return self

        def predict_instances(self, image, prob_thresh=0.5, nms_thresh=0.3):
            labels = np.zeros((image.shape[0], image.shape[1]), dtype=np.int32)
            scale_y = max(1, image.shape[0] // 8)
            scale_x = max(1, image.shape[1] // 8)
            labels[1 * scale_y : 4 * scale_y, 1 * scale_x : 4 * scale_x] = 1
            return labels, {"prob": np.asarray([0.99], dtype=np.float32)}

    class FakeBackend:
        def __init__(self) -> None:
            self.Config2D = lambda **kwargs: kwargs
            self.StarDist2D = lambda config, name, basedir: FakeModel()

    monkeypatch.setattr(runner, "_load_stardist_backend", lambda: FakeBackend())
    monkeypatch.setattr(
        runner,
        "configure_tensorflow_runtime",
        lambda require_gpu=True, allow_cpu=False: {
            "version": "fake-tf",
            "cuda_built": False,
            "gpu_visible": False,
            "physical_gpu_count": 0,
            "physical_gpus": [],
            "memory_growth": [],
        },
    )
    monkeypatch.setattr(runner, "enforce_ram_limit", lambda limit_pct, label: 10.0)

    out_dir = tmp_path / "out"
    args = runner.build_argparser().parse_args(
        [
            "--dataset-root",
            str(dataset_root),
            "--output-dir",
            str(out_dir),
            "--image-size",
            "512",
            "--epochs",
            "1",
            "--batch",
            "1",
            "--num-workers",
            "0",
            "--max-train-images",
            "1",
            "--max-val-images",
            "1",
            "--prob-thresh",
            "0.5",
            "--nms-thresh",
            "0.3",
            "--allow-cpu",
        ]
    )

    result = runner.train_and_eval(args)

    assert (out_dir / "coco_instances_results.json").exists()
    assert (out_dir / "metrics.cocoeval.json").exists()
    assert (out_dir / "metadata.json").exists()
    assert (out_dir / "last_checkpoint").exists()
    assert (out_dir / "wall_time_sec.txt").exists()
    assert (out_dir / "params_trainable.txt").exists()
    assert "segm/AP" in result["metrics"]
    assert result["metrics"]["bbox/AP"] > 99.0
    metadata = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["model_id"] == "stardist"
    assert metadata["tensorflow"]["version"] == "fake-tf"
    assert metadata["ram_limit_pct"] == 0.0


def test_stardist_runner_passes_auto_classes_to_train(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json

    from baselines import run_stardist_instance_ecc as runner

    created_models: list[object] = []

    dataset_root = tmp_path / "ecc"
    (dataset_root / "annotations").mkdir(parents=True, exist_ok=True)
    (dataset_root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (dataset_root / "images" / "val").mkdir(parents=True, exist_ok=True)

    image_name = "sample.png"
    Image.new("RGB", (8, 8), color=(10, 20, 30)).save(dataset_root / "images" / "train" / image_name)
    Image.new("RGB", (8, 8), color=(10, 20, 30)).save(dataset_root / "images" / "val" / image_name)
    payload = {
        "images": [{"id": 1, "file_name": image_name, "width": 8, "height": 8}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "segmentation": [[1, 1, 4, 1, 4, 4, 1, 4]],
                "area": 9,
                "bbox": [1, 1, 3, 3],
                "iscrowd": 0,
            }
        ],
        "categories": [{"id": 1, "name": "component"}],
    }
    (dataset_root / "annotations" / "instances_train.json").write_text(json.dumps(payload), encoding="utf-8")
    (dataset_root / "annotations" / "instances_val.json").write_text(json.dumps(payload), encoding="utf-8")

    class FakeKerasModel:
        trainable_weights: list[np.ndarray] = [np.ones((2, 3), dtype=np.float32)]

        def save_weights(self, path: str) -> None:
            Path(path).write_text("fake-weights", encoding="utf-8")

        def count_params(self) -> int:
            return 6

    class FakeModel:
        def __init__(self) -> None:
            self.keras_model = FakeKerasModel()
            self.train_calls: list[dict[str, object]] = []

        def train(
            self,
            X,
            Y,
            *,
            validation_data=None,
            classes=None,
            augmenter=None,
            seed=None,
            epochs=None,
            steps_per_epoch=None,
            workers=None,
        ):
            self.train_calls.append(
                {
                    "classes": classes,
                    "X_len": len(X),
                    "Y_len": len(Y),
                    "validation_len": len(validation_data[0]) if validation_data is not None else None,
                    "epochs": epochs,
                    "steps_per_epoch": steps_per_epoch,
                    "workers": workers,
                }
            )
            return self

        def predict_instances(self, image, prob_thresh=0.5, nms_thresh=0.3):
            labels = np.zeros((image.shape[0], image.shape[1]), dtype=np.int32)
            scale_y = max(1, image.shape[0] // 8)
            scale_x = max(1, image.shape[1] // 8)
            labels[1 * scale_y : 4 * scale_y, 1 * scale_x : 4 * scale_x] = 1
            return labels, {"prob": np.asarray([0.99], dtype=np.float32)}

    class FakeBackend:
        def __init__(self) -> None:
            self.Config2D = lambda **kwargs: kwargs
            self.StarDist2D = self._make_model

        def _make_model(self, config, name, basedir):
            model = FakeModel()
            created_models.append(model)
            return model

    monkeypatch.setattr(runner, "_load_stardist_backend", lambda: FakeBackend())
    monkeypatch.setattr(
        runner,
        "configure_tensorflow_runtime",
        lambda require_gpu=True, allow_cpu=False: {
            "version": "fake-tf",
            "cuda_built": False,
            "gpu_visible": False,
            "physical_gpu_count": 0,
            "physical_gpus": [],
            "memory_growth": [],
        },
    )
    monkeypatch.setattr(runner, "enforce_ram_limit", lambda limit_pct, label: 10.0)

    out_dir = tmp_path / "out"
    args = runner.build_argparser().parse_args(
        [
            "--dataset-root",
            str(dataset_root),
            "--output-dir",
            str(out_dir),
            "--image-size",
            "512",
            "--epochs",
            "1",
            "--batch",
            "1",
            "--num-workers",
            "0",
            "--max-train-images",
            "1",
            "--max-val-images",
            "1",
            "--prob-thresh",
            "0.5",
            "--nms-thresh",
            "0.3",
            "--allow-cpu",
        ]
    )

    result = runner.train_and_eval(args)

    assert result["artifacts"]["metadata"].exists()
    metadata = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["model_id"] == "stardist"
    assert created_models
    assert created_models[0].train_calls[0]["classes"] == "auto"
