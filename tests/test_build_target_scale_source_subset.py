from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools.build_target_scale_source_subset import SubsetBuildError, build_subset


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _image(image_id: int) -> dict[str, object]:
    return {"id": image_id, "file_name": f"image_{image_id}.png", "width": 64, "height": 64}


def _ann(ann_id: int, image_id: int, area: float, category_id: int = 1) -> dict[str, object]:
    side = max(1.0, area**0.5)
    return {
        "id": ann_id,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": [0, 0, side, side],
        "area": area,
        "segmentation": [[0, 0, side, 0, side, side, 0, side]],
        "iscrowd": 0,
    }


def _coco(images: list[dict[str, object]], annotations: list[dict[str, object]]) -> dict[str, object]:
    return {
        "info": {"description": "toy"},
        "licenses": [{"id": 1, "name": "toy"}],
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "component"}],
    }


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _small_ratio(values: list[float]) -> float:
    return sum(1 for value in values if value <= 256.0) / len(values)


def test_build_subset_keeps_original_ids_and_whole_images(tmp_path: Path) -> None:
    source_path = _write_json(
        tmp_path / "source.json",
        _coco(
            [_image(10), _image(20), _image(30)],
            [
                _ann(100, 10, 120),
                _ann(101, 10, 200),
                _ann(200, 20, 8000),
                _ann(300, 30, 450),
                _ann(301, 30, 520),
            ],
        ),
    )
    target_path = _write_json(
        tmp_path / "target.json",
        _coco(
            [_image(1), _image(2)],
            [_ann(1, 1, 430), _ann(2, 1, 470), _ann(3, 2, 220), _ann(4, 2, 260)],
        ),
    )
    output_path = tmp_path / "subset.json"
    summary_path = tmp_path / "summary.json"

    summary = build_subset(
        source_ann=source_path,
        target_ann=target_path,
        output_ann=output_path,
        summary_json=summary_path,
        target_images=1,
        seed=7,
        area_key="area",
    )

    subset = json.loads(output_path.read_text(encoding="utf-8"))
    selected_image_ids = {image["id"] for image in subset["images"]}
    assert selected_image_ids == {30}
    assert [ann["id"] for ann in subset["annotations"]] == [300, 301]
    assert {ann["image_id"] for ann in subset["annotations"]} == selected_image_ids
    assert subset["categories"] == [{"id": 1, "name": "component"}]
    assert subset["info"] == {"description": "toy"}
    assert subset["licenses"] == [{"id": 1, "name": "toy"}]
    assert summary == json.loads(summary_path.read_text(encoding="utf-8"))


def test_build_subset_fails_on_bad_annotation_reference(tmp_path: Path) -> None:
    source_path = _write_json(
        tmp_path / "bad_source.json",
        _coco([_image(1)], [_ann(1, 999, 100)]),
    )
    target_path = _write_json(tmp_path / "target.json", _coco([_image(2)], [_ann(2, 2, 100)]))

    with pytest.raises(SubsetBuildError, match="references missing image_id"):
        build_subset(
            source_ann=source_path,
            target_ann=target_path,
            output_ann=tmp_path / "subset.json",
            summary_json=tmp_path / "summary.json",
            target_images=1,
            seed=0,
            area_key="area",
        )


def test_build_subset_fails_on_invalid_bbox_area_or_category(tmp_path: Path) -> None:
    target_path = _write_json(tmp_path / "target.json", _coco([_image(2)], [_ann(2, 2, 100)]))
    invalid_bbox = _ann(1, 1, 100)
    invalid_bbox["bbox"] = [0, 0, 0, 10]
    source_path = _write_json(tmp_path / "bad_bbox.json", _coco([_image(1)], [invalid_bbox]))

    with pytest.raises(SubsetBuildError, match="bbox width/height must be positive"):
        build_subset(
            source_ann=source_path,
            target_ann=target_path,
            output_ann=tmp_path / "subset.json",
            summary_json=tmp_path / "summary.json",
            target_images=1,
            seed=0,
            area_key="area",
        )

    invalid_category = _ann(3, 3, 100, category_id=7)
    source_path = _write_json(tmp_path / "bad_category.json", _coco([_image(3)], [invalid_category]))
    with pytest.raises(SubsetBuildError, match="invalid category_id"):
        build_subset(
            source_ann=source_path,
            target_ann=target_path,
            output_ann=tmp_path / "subset.json",
            summary_json=tmp_path / "summary.json",
            target_images=1,
            seed=0,
            area_key="area",
        )


def test_subset_distribution_is_closer_to_target_than_source(tmp_path: Path) -> None:
    source_areas_by_image = {
        10: [90, 110, 120],
        20: [400, 450, 520],
        30: [410, 470, 530],
        40: [7600, 8100, 8400],
        50: [9000, 9200, 9500],
    }
    source_annotations = []
    ann_id = 1
    for image_id, areas in source_areas_by_image.items():
        for area in areas:
            source_annotations.append(_ann(ann_id, image_id, area))
            ann_id += 1
    target_areas = [220, 240, 410, 455, 500, 540]
    source_path = _write_json(tmp_path / "source.json", _coco([_image(i) for i in source_areas_by_image], source_annotations))
    target_path = _write_json(
        tmp_path / "target.json",
        _coco([_image(1), _image(2)], [_ann(i + 1, 1 + (i // 3), area) for i, area in enumerate(target_areas)]),
    )

    summary = build_subset(
        source_ann=source_path,
        target_ann=target_path,
        output_ann=tmp_path / "subset.json",
        summary_json=tmp_path / "summary.json",
        target_images=2,
        seed=11,
        area_key="area",
    )

    source_all = [area for areas in source_areas_by_image.values() for area in areas]
    subset_areas = [ann["area"] for ann in json.loads((tmp_path / "subset.json").read_text(encoding="utf-8"))["annotations"]]
    target_median = _median(target_areas)
