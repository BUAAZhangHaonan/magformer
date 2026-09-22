from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from pycocotools import mask as mask_utils

from tools import canonicalize_coco_annotations as tool


def _events(value, prefix=""):
    if isinstance(value, dict):
        yield prefix, "start_map", None
        for key, child in value.items():
            yield prefix, "map_key", key
            child_prefix = f"{prefix}.{key}" if prefix else key
            yield from _events(child, child_prefix)
        yield prefix, "end_map", None
    elif isinstance(value, list):
        yield prefix, "start_array", None
        for child in value:
            child_prefix = f"{prefix}.item" if prefix else "item"
            yield from _events(child, child_prefix)
        yield prefix, "end_array", None
    elif value is None:
        yield prefix, "null", None
    elif isinstance(value, bool):
        yield prefix, "boolean", value
    elif isinstance(value, int):
        yield prefix, "integer", value
    elif isinstance(value, float):
        yield prefix, "double", value
    else:
        yield prefix, "string", value


@pytest.fixture(autouse=True)
def fake_ijson(monkeypatch):
    def parse(handle):
        yield from _events(json.load(handle))

    def items(handle, prefix):
        value = json.load(handle)
        for part in prefix.split("."):
            if part == "item":
                for child in value:
                    yield child
                return
            value = value[part]
        yield value

    monkeypatch.setitem(sys.modules, "ijson", types.SimpleNamespace(parse=parse, items=items))


def _rle(height: int, width: int, filled: bool):
    mask = np.ones((height, width), dtype=np.uint8) if filled else np.zeros((height, width), dtype=np.uint8)
    encoded = mask_utils.encode(np.asfortranarray(mask))
    return {"size": encoded["size"], "counts": encoded["counts"].decode("ascii")}


def _source(annotations, *, images=None, categories=None):
    return {
        "info": {"description": "mini"},
        "licenses": [{"id": 1, "name": "test"}],
        "images": images or [{"id": 7, "width": 10, "height": 10, "file_name": "a.png"}],
        "categories": categories or [{"id": 1, "name": "component"}],
        "annotations": annotations,
    }


def _write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")


def _run(tmp_path: Path, data, stem="out"):
    source = tmp_path / "source.json"
    output = tmp_path / f"{stem}.json"
    manifest = tmp_path / f"{stem}.manifest.json"
    _write(source, data)
    tool.canonicalize_file(source, output, manifest)
    return source, json.loads(output.read_text()), json.loads(manifest.read_text()), output, manifest


def test_valid_polygon_gets_canonical_bbox_and_area(tmp_path):
    _, output, manifest, output_path, _ = _run(
        tmp_path,
        _source([{"id": 11, "image_id": 7, "category_id": 1, "iscrowd": 0, "segmentation": [[1, 2, 4, 2, 4, 6, 1, 6]], "bbox": [99, 99, 1, 1], "area": 1}]),
    )
    annotation = output["annotations"][0]
    assert annotation["id"] == 11
    assert annotation["segmentation"] == [[1, 2, 4, 2, 4, 6, 1, 6]]
    assert annotation["bbox"] == [1.0, 2.0, 3.0, 4.0]
    assert annotation["area"] == 12.0
    assert output["info"] == {"description": "mini"}
    assert output["licenses"] == [{"id": 1, "name": "test"}]
    assert output["images"][0]["id"] == 7
    assert output["categories"][0]["id"] == 1
    assert manifest["canonical_corrections"] == {"bbox": 1, "area": 1}
    assert manifest["category_counts"] == {"1": {"original": 1, "retained": 1, "removed": 0}}
    tool.canonicalize_file(output_path, output_path, output_path, validate_only=True)


def test_zero_raster_polygon_and_malformed_segmentation_are_removed(tmp_path):
    _, output, manifest, _, _ = _run(
        tmp_path,
        _source([
            {"id": 1, "image_id": 7, "category_id": 1, "segmentation": [[0.1, 0.1, 0.2, 0.1, 0.2, 0.2]], "bbox": [0, 0, 1, 1], "area": 1},
            {"id": 2, "image_id": 7, "category_id": 1, "segmentation": [], "bbox": [0, 0, 1, 1], "area": 1},
        ]),
    )
    assert output["annotations"] == []
    assert manifest["removed_ids_by_reason"] == {"malformed_segmentation": [2], "polygon_zero_area": [1]}
    assert manifest["images_that_become_zero_instance"] == [7]


def test_valid_and_zero_rle_are_handled_without_bitmap_decoding(tmp_path):
    _, output, manifest, _, _ = _run(
        tmp_path,
        _source([
            {"id": 3, "image_id": 7, "category_id": 1, "iscrowd": 1, "segmentation": _rle(10, 10, True), "bbox": [0, 0, 1, 1], "area": 1},
            {"id": 4, "image_id": 7, "category_id": 1, "segmentation": _rle(10, 10, False), "bbox": [0, 0, 1, 1], "area": 1},
        ]),
    )
    assert output["annotations"][0]["id"] == 3
    assert output["annotations"][0]["iscrowd"] == 1
    assert output["annotations"][0]["bbox"] == [0.0, 0.0, 10.0, 10.0]
    assert output["annotations"][0]["area"] == 100.0
    assert manifest["removed_ids_by_reason"] == {"rle_zero_area": [4]}


@pytest.mark.parametrize(
    "data, message",
    [
        (_source([], images=[{"id": 7, "width": 10, "height": 10}, {"id": 7, "width": 10, "height": 10}]), "duplicate image id"),
        (_source([], categories=[{"id": 1}, {"id": 1}]), "duplicate category id"),
        (_source([{"id": 1, "image_id": 99, "category_id": 1, "segmentation": _rle(10, 10, True)}]), "dangling image_id"),
        (_source([{"id": 1, "image_id": 7, "category_id": 9, "segmentation": _rle(10, 10, True)}]), "dangling category_id"),
        (_source([{"id": 1, "image_id": 7, "category_id": 1, "segmentation": _rle(10, 10, True)}, {"id": 1, "image_id": 7, "category_id": 1, "segmentation": _rle(10, 10, True)}]), "duplicate annotation id"),
    ],
)
def test_duplicate_and_dangling_references_fail_loudly(tmp_path, data, message):
    source = tmp_path / "source.json"
    _write(source, data)
    with pytest.raises(tool.CocoValidationError, match=message):
        tool.canonicalize_file(source, tmp_path / "out.json", tmp_path / "manifest.json")


def test_existing_outputs_are_refused_atomically(tmp_path):
    source = tmp_path / "source.json"
    output = tmp_path / "out.json"
    manifest = tmp_path / "manifest.json"
    _write(source, _source([]))
    output.write_text("sentinel", encoding="utf-8")
    with pytest.raises(FileExistsError, match="--overwrite"):
        tool.canonicalize_file(source, output, manifest)
    assert output.read_text(encoding="utf-8") == "sentinel"
    output.unlink()
    manifest.write_text("sentinel", encoding="utf-8")
    with pytest.raises(FileExistsError, match="--overwrite"):
        tool.canonicalize_file(source, output, manifest)
    assert manifest.read_text(encoding="utf-8") == "sentinel"


def test_output_and_manifest_are_deterministic_without_timestamps(tmp_path):
    data = _source([
        {"id": 2, "image_id": 7, "category_id": 1, "segmentation": _rle(10, 10, False)},
        {"id": 1, "image_id": 7, "category_id": 1, "segmentation": _rle(10, 10, True)},
    ])
    source_a, _, _, output_a, manifest_a = _run(tmp_path / "a", data, "canonical")
    source_b = tmp_path / "b" / "source.json"
    source_b.parent.mkdir(parents=True)
    source_b.write_bytes(source_a.read_bytes())
    output_b = tmp_path / "b" / "canonical.json"
    manifest_b = tmp_path / "b" / "canonical.manifest.json"
    tool.canonicalize_file(source_b, output_b, manifest_b)
    assert output_a.read_bytes() == output_b.read_bytes()
    assert manifest_a.read_bytes() == manifest_b.read_bytes()
