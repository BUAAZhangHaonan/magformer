import json

from pycocotools import mask as mask_utils

from tools.diagnose_prediction_union_oracle import run_diagnostic


def _rle(x0, y0, x1, y1, height=10, width=10):
    mask = [[0 for _ in range(width)] for _ in range(height)]
    for y in range(y0, y1):
        for x in range(x0, x1):
            mask[y][x] = 1
    import numpy as np

    arr = np.asarray(mask, dtype="uint8", order="F")
    rle = mask_utils.encode(arr)
    return {"size": list(rle["size"]), "counts": rle["counts"].decode("ascii")}


def _ann(idx, image_id, rle):
    bbox = mask_utils.toBbox({**rle, "counts": rle["counts"].encode("ascii")}).tolist()
    area = float(mask_utils.area({**rle, "counts": rle["counts"].encode("ascii")}))
    return {
        "id": idx,
        "image_id": image_id,
        "category_id": 1,
        "segmentation": rle,
        "bbox": bbox,
        "area": area,
        "iscrowd": 0,
    }


def _pred(image_id, rle, score=0.9):
    bbox = mask_utils.toBbox({**rle, "counts": rle["counts"].encode("ascii")}).tolist()
    return {
        "image_id": image_id,
        "category_id": 1,
        "segmentation": rle,
        "bbox": bbox,
        "score": score,
    }


def test_run_diagnostic_reports_union_and_run_only_coverage(tmp_path):
    gt1 = _rle(0, 0, 4, 4)
    gt2 = _rle(5, 5, 9, 9)
    gt3 = _rle(1, 6, 3, 8)
    ann_path = tmp_path / "ann.json"
    ann_path.write_text(
        json.dumps(
            {
                "images": [
                    {"id": 1, "file_name": "a.png", "height": 10, "width": 10},
                    {"id": 2, "file_name": "b.png", "height": 10, "width": 10},
                ],
                "annotations": [
                    _ann(1, 1, gt1),
                    _ann(2, 1, gt2),
                    _ann(3, 2, gt3),
                ],
                "categories": [{"id": 1, "name": "object"}],
            }
        )
    )

    r1_path = tmp_path / "r1.json"
    r1_path.write_text(json.dumps([_pred(1, gt1)]))
    r2_path = tmp_path / "r2.json"
    r2_path.write_text(json.dumps([_pred(1, gt2), _pred(2, gt3)]))

    summary = run_diagnostic(
        ann_path=ann_path,
        predictions=[("r1", r1_path), ("r2", r2_path)],
        out_dir=tmp_path / "out",
        skip_coco_eval=True,
    )

    assert summary["union"]["oracle_recall"]["@75"] == 1.0
    assert summary["runs"]["r1"]["oracle_recall"]["@75"] == 1 / 3
    assert summary["runs"]["r2"]["oracle_recall"]["@75"] == 2 / 3
    assert summary["coverage_overlap"]["shared_misses@75"] == 0
    assert summary["coverage_overlap"]["run_only_covered@75"] == {"r1": 1, "r2": 2}
    assert summary["union"]["by_density_bucket"][">90"]["gt"] == 0
    assert summary["union"]["by_area_bucket"]["<=256"]["oracle_recall"]["@75"] == 1.0
    assert (tmp_path / "out" / "r53_prediction_union_oracle_summary.json").exists()
