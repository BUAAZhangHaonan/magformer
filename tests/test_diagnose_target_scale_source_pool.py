from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.diagnose_target_scale_source_pool import DiagnosisError, diagnose_pool


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _image(image_id: int, file_name: str | None = None) -> dict[str, object]:
    return {
        "id": image_id,
        "file_name": file_name or f"image_{image_id}.png",
        "width": 1024,
        "height": 1024,
    }


def _ann(ann_id: int, image_id: int, area: float) -> dict[str, object]:
    side = area**0.5
    return {
        "id": ann_id,
        "image_id": image_id,
        "category_id": 1,
        "bbox": [0, 0, side, side],
        "area": area,
        "segmentation": [[0, 0, side, 0, side, side, 0, side]],
        "iscrowd": 0,
    }


def _coco(images: list[dict[str, object]], annotations: list[dict[str, object]]) -> dict[str, object]:
    return {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "component"}],
    }


def test_diagnose_pool_projects_source_and_mixed_pool_scale(tmp_path: Path) -> None:
    source_path = _write_json(
        tmp_path / "source.json",
        _coco(
            [_image(1, "source_a.png"), _image(2, "source_b.png")],
            [_ann(1, 1, 1000), _ann(2, 1, 4000), _ann(3, 2, 9000)],
        ),
    )
    target_path = _write_json(
        tmp_path / "target.json",
        _coco(
            [_image(10, "target.png")],
            [_ann(10, 10, 250), _ann(11, 10, 500), _ann(12, 10, 750)],
        ),
    )
    existing_path = _write_json(
        tmp_path / "existing.json",
        _coco([_image(20, "existing.png")], [_ann(20, 20, 484), _ann(21, 20, 900)]),
    )
    forbidden_path = _write_json(
        tmp_path / "forbidden.json",
        _coco([_image(30, "target.png")], [_ann(30, 30, 100)]),
    )

    summary = diagnose_pool(
        source_ann=source_path,
        target_ann=target_path,
        scale_factors=[0.25, 0.5],
        summary_json=tmp_path / "summary.json",
        existing_pool_ann=existing_path,
        forbidden_anns={"target": forbidden_path},
        min_images=2,
        p50_lower=100.0,
        p50_upper=600.0,
        small_area_abs_tol=0.20,
    )

    candidate_025 = summary["candidates"][0]
    assert candidate_025["scale_factor"] == 0.25
    assert candidate_025["projected_source"]["images"] == 2
    assert candidate_025["projected_source"]["instances"] == 3
    assert candidate_025["projected_source"]["mask_area"]["p50"] == pytest.approx(250.0)
    assert candidate_025["projected_source"]["area_le_256_ratio"] == pytest.approx(2 / 3)
    assert candidate_025["projected_source"]["instances_per_image"]["p50"] == pytest.approx(1.5)
    assert candidate_025["projected_source"]["instances_per_image"]["p90"] == pytest.approx(1.9)
    assert candidate_025["projected_source"]["gate"]["pass"] is False
    assert candidate_025["projected_source"]["gate"]["basename_overlap_pass"] is True

    candidate_05 = summary["candidates"][1]
    assert candidate_05["projected_source"]["mask_area"]["p50"] == pytest.approx(1000.0)
    assert candidate_05["projected_source"]["gate"]["mask_area_p50_pass"] is False
    assert candidate_05["mixed_pool"]["images"] == 3
    assert candidate_05["mixed_pool"]["instances"] == 5
    assert candidate_05["mixed_pool"]["gate"]["min_images_pass"] is True

    assert summary["target"]["basename_overlap"]["target"]["count"] == 1
    assert summary == json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))


def test_diagnose_pool_fails_on_missing_area(tmp_path: Path) -> None:
    bad_ann = _ann(1, 1, 100)
    del bad_ann["area"]
    source_path = _write_json(tmp_path / "source.json", _coco([_image(1)], [bad_ann]))
    target_path = _write_json(tmp_path / "target.json", _coco([_image(2)], [_ann(2, 2, 100)]))

    with pytest.raises(DiagnosisError, match="missing required field: area"):
        diagnose_pool(
            source_ann=source_path,
            target_ann=target_path,
            scale_factors=[0.25],
            summary_json=tmp_path / "summary.json",
        )
