from __future__ import annotations

import argparse
import json
import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Any

from pycocotools import mask as coco_mask

from magformer.data.coco_loader_cache import SCHEMA_VERSION, digest_json, sha256_file


REQUIRED_IMAGE_FIELDS = ("id", "file_name", "height", "width")
REQUIRED_CATEGORY_FIELDS = ("id", "name")
REQUIRED_ANNOTATION_FIELDS = (
    "id",
    "image_id",
    "category_id",
    "bbox",
    "area",
    "segmentation",
    "iscrowd",
)


def build_coco_loader_cache(source_json: str | Path, cache_path: str | Path) -> dict[str, Any]:
    """Build a first-phase SQLite cache from a small COCO JSON file.

    This implementation intentionally uses json.load. It is suitable for small fixtures and
    first-phase parity checks, not for the 9.9G full COCO cache build.
    """

    source = Path(source_json).resolve()
    target = Path(cache_path).resolve()
    if not source.exists():
        raise FileNotFoundError(f"COCO source JSON not found: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)

    with source.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    images = _required_list(data, "images")
    annotations = _required_list(data, "annotations")
    categories = _required_list(data, "categories")
    info = data.get("info", {})
    licenses = data.get("licenses", [])
    if not isinstance(info, dict):
        raise ValueError("COCO info must be an object")
    if not isinstance(licenses, list):
        raise ValueError("COCO licenses must be a list")

    image_by_id = _validate_images(images)
    category_by_id = _validate_categories(categories)
    _validate_annotations(annotations, image_by_id, category_by_id)

    image_ids = sorted(image_by_id)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "source_path": str(source),
        "source_sha256": sha256_file(source),
        "image_count": len(images),
        "image_ids": image_ids,
        "image_ids_digest": digest_json(image_ids),
        "annotation_count": len(annotations),
        "category_count": len(categories),
        "categories": categories,
        "info": info,
        "licenses": licenses,
        "builder": "tools/build_coco_loader_cache.py:first-phase-json-load",
        "streaming_supported": False,
    }

    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        _write_sqlite_cache(tmp_path, manifest, images, annotations, categories)
        os.replace(tmp_path, target)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return manifest


def _write_sqlite_cache(
    path: Path,
    manifest: dict[str, Any],
    images: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
    categories: list[dict[str, Any]],
) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.execute("PRAGMA journal_mode=OFF")
        conn.execute("PRAGMA synchronous=OFF")
        conn.executescript(
            """
            CREATE TABLE manifest (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
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
            CREATE INDEX idx_annotations_image_ordinal ON annotations(image_id, ordinal);
            CREATE INDEX idx_annotations_category ON annotations(category_id);
            CREATE TABLE categories (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                supercategory TEXT,
                json TEXT NOT NULL
            );
            """
        )
        conn.executemany(
            "INSERT INTO manifest(key, value) VALUES (?, ?)",
            [(_key, _json_dumps(value)) for _key, value in sorted(manifest.items())],
        )
        conn.executemany(
            "INSERT INTO images(id, file_name, json) VALUES (?, ?, ?)",
            [(int(image["id"]), str(image["file_name"]), _json_dumps(image)) for image in images],
        )
        conn.executemany(
            """
            INSERT INTO annotations(id, image_id, category_id, iscrowd, area, ordinal, json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    int(ann["id"]),
                    int(ann["image_id"]),
                    int(ann["category_id"]),
                    int(ann["iscrowd"]),
                    float(ann["area"]),
                    ordinal,
                    _json_dumps(ann),
                )
                for ordinal, ann in enumerate(annotations)
            ],
        )
        conn.executemany(
            "INSERT INTO categories(id, name, supercategory, json) VALUES (?, ?, ?, ?)",
            [
                (
                    int(category["id"]),
                    str(category["name"]),
                    category.get("supercategory"),
                    _json_dumps(category),
                )
                for category in categories
            ],
        )
        conn.commit()
    finally:
        conn.close()


def _validate_images(images: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    by_id: dict[int, dict[str, Any]] = {}
    for index, image in enumerate(images):
        _require_fields(image, REQUIRED_IMAGE_FIELDS, f"images[{index}]")
        image_id = _strict_int(image["id"], f"images[{index}].id")
        if image_id in by_id:
            raise ValueError(f"duplicate image id in COCO source: {image_id}")
        if not str(image["file_name"]):
            raise ValueError(f"images[{index}].file_name must be non-empty")
        _strict_positive_int(image["height"], f"images[{index}].height")
        _strict_positive_int(image["width"], f"images[{index}].width")
        by_id[image_id] = image
    return by_id


def _validate_categories(categories: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    by_id: dict[int, dict[str, Any]] = {}
    for index, category in enumerate(categories):
        _require_fields(category, REQUIRED_CATEGORY_FIELDS, f"categories[{index}]")
        category_id = _strict_int(category["id"], f"categories[{index}].id")
        if category_id in by_id:
            raise ValueError(f"duplicate category id in COCO source: {category_id}")
        if not str(category["name"]):
            raise ValueError(f"categories[{index}].name must be non-empty")
        by_id[category_id] = category
    return by_id


def _validate_annotations(
    annotations: list[dict[str, Any]],
    image_by_id: dict[int, dict[str, Any]],
    category_by_id: dict[int, dict[str, Any]],
) -> None:
    seen_ids: set[int] = set()
    for index, ann in enumerate(annotations):
        label = f"annotations[{index}]"
        _require_fields(ann, REQUIRED_ANNOTATION_FIELDS, label)
        ann_id = _strict_int(ann["id"], f"{label}.id")
        if ann_id in seen_ids:
            raise ValueError(f"duplicate annotation id in COCO source: {ann_id}")
        seen_ids.add(ann_id)
        image_id = _strict_int(ann["image_id"], f"{label}.image_id")
        category_id = _strict_int(ann["category_id"], f"{label}.category_id")
        if image_id not in image_by_id:
            raise ValueError(f"{label}.image_id {image_id} does not exist in images")
        if category_id not in category_by_id:
            raise ValueError(f"{label}.category_id {category_id} does not exist in categories")
        iscrowd = _strict_int(ann["iscrowd"], f"{label}.iscrowd")
        if iscrowd not in (0, 1):
            raise ValueError(f"{label}.iscrowd must be 0 or 1, got {iscrowd}")
        _validate_bbox(ann["bbox"], label)
        float(ann["area"])
        image = image_by_id[image_id]
        _validate_segmentation(ann["segmentation"], int(image["height"]), int(image["width"]), label)


def _validate_bbox(bbox: Any, label: str) -> None:
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError(f"{label}.bbox must be [x, y, width, height]")
    for value in bbox:
        float(value)
    if float(bbox[2]) < 0 or float(bbox[3]) < 0:
        raise ValueError(f"{label}.bbox width and height must be non-negative")


def _validate_segmentation(segmentation: Any, height: int, width: int, label: str) -> None:
    try:
        if isinstance(segmentation, list):
            if not segmentation:
                raise ValueError("polygon list is empty")
            coco_mask.frPyObjects(segmentation, height, width)
        elif isinstance(segmentation, dict):
            if "counts" not in segmentation or "size" not in segmentation:
                raise ValueError("RLE requires counts and size")
            if list(segmentation["size"]) != [height, width]:
                raise ValueError(
                    f"RLE size {segmentation['size']} does not match image size {[height, width]}"
                )
            coco_mask.decode(segmentation)
        else:
            raise ValueError("segmentation must be polygon list or RLE object")
    except Exception as exc:
        raise ValueError(f"invalid {label}.segmentation: {exc}") from exc


def _required_list(data: dict[str, Any], key: str) -> list[dict[str, Any]]:
    value = data.get(key)
    if not isinstance(value, list):
        raise ValueError(f"COCO {key} must be a list")
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"COCO {key}[{index}] must be an object")
    return value


def _require_fields(item: dict[str, Any], fields: tuple[str, ...], label: str) -> None:
    missing = [field for field in fields if field not in item]
    if missing:
        raise ValueError(f"{label} missing required field(s): {missing}")


def _strict_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer, got bool")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer, got {value!r}") from exc
    if result != value:
        raise ValueError(f"{label} must be an integer, got {value!r}")
    return result


def _strict_positive_int(value: Any, label: str) -> int:
    result = _strict_int(value, label)
    if result <= 0:
        raise ValueError(f"{label} must be positive, got {result}")
    return result


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a first-phase lazy COCO loader SQLite cache")
    parser.add_argument("source_json", type=Path)
    parser.add_argument("cache_path", type=Path)
    args = parser.parse_args()
    manifest = build_coco_loader_cache(args.source_json, args.cache_path)
    print(
        json.dumps(
            {
                "cache_path": str(args.cache_path.resolve()),
                "image_count": manifest["image_count"],
                "annotation_count": manifest["annotation_count"],
                "streaming_supported": manifest["streaming_supported"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
