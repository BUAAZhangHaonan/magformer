import json
from pathlib import Path

import pytest
from pycocotools import mask as mask_utils


def _rle(area_size, *, height=32, width=32):
    mask = [[0 for _ in range(width)] for _ in range(height)]
    filled = 0
    for y in range(height):
        for x in range(width):
            if filled < area_size:
                mask[y][x] = 1
                filled += 1
    import numpy as np

    encoded = mask_utils.encode(np.asfortranarray(np.array(mask, dtype="uint8")))
    encoded["counts"] = encoded["counts"].decode("ascii")
    return encoded


def _write_json(path: Path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


def test_stats_builder_rejects_forbidden_prediction_fields(tmp_path):
    from tools.build_target_unlabeled_sampling_stats import build_sampling_stats

    target_ann = tmp_path / "target.json"
    r46_pred = tmp_path / "r46.json"
    r12_pred = tmp_path / "r12.json"
    out = tmp_path / "stats.json"

    _write_json(target_ann, {"images": [{"id": 1, "file_name": "a.png"}], "annotations": [{"id": 1}]})
    _write_json(r46_pred, [{"image_id": 1, "score": 0.9, "segmentation": _rle(4), "gt_count": 1}])
    _write_json(r12_pred, [{"image_id": 1, "score": 0.8, "segmentation": _rle(4)}])

    with pytest.raises(ValueError, match="gt_count"):
        build_sampling_stats(target_ann, r46_pred, r12_pred, out)


def test_stats_builder_emits_prediction_only_repeats_without_gt_fields(tmp_path):
    from tools.build_target_unlabeled_sampling_stats import build_sampling_stats

    target_ann = tmp_path / "target.json"
    r46_pred = tmp_path / "r46.json"
    r12_pred = tmp_path / "r12.json"
    out = tmp_path / "stats.json"

    _write_json(
        target_ann,
        {
            "images": [
                {"id": 1, "file_name": "normal.png"},
                {"id": 2, "file_name": "dense.png"},
                {"id": 3, "file_name": "dense_tiny.png"},
            ],
            "annotations": [{"id": 99, "image_id": 3}],
        },
    )

    def preds(image_id, count, small_count):
        values = []
        for idx in range(count):
            area = 100 if idx < small_count else 400
            values.append({"image_id": image_id, "score": 0.5 + idx / 1000, "segmentation": _rle(area)})
        return values

    _write_json(r46_pred, preds(1, 4, 0) + preds(2, 90, 0) + preds(3, 90, 30))
    _write_json(r12_pred, preds(1, 5, 0) + preds(2, 92, 0) + preds(3, 91, 32))

    summary = build_sampling_stats(target_ann, r46_pred, r12_pred, out)

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert summary["repeat_counts"] == {"normal": 1, "dense": 1, "dense_tiny": 1}
    assert summary["sequence_length"] == 6
    assert [item["repeat"] for item in payload["images"]] == [1, 2, 3]
    encoded = json.dumps(payload)
    assert "annotations" not in encoded
    assert "gt_count" not in encoded
    assert "gt_density_bucket" not in encoded
