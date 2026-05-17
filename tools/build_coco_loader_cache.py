from __future__ import annotations

import argparse
import json
import os
import sqlite3
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any, TextIO

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
STREAMING_THRESHOLD_BYTES = 64 * 1024 * 1024
TOP_LEVEL_STREAM_ARRAYS = {"images", "annotations", "categories"}


def build_coco_loader_cache(
    source_json: str | Path,
    cache_path: str | Path,
    mode: str = "auto",
    streaming_threshold_bytes: int = STREAMING_THRESHOLD_BYTES,
) -> dict[str, Any]:
    """Build a lazy COCO loader SQLite cache.

    ``mode="auto"`` keeps small fixture behavior on the json.load path and switches large
    source files to the streaming builder.
    """

    source = Path(source_json).resolve()
    target = Path(cache_path).resolve()
    if not source.exists():
        raise FileNotFoundError(f"COCO source JSON not found: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)

    build_mode = _resolve_build_mode(source, mode, streaming_threshold_bytes)
    if build_mode == "json-load":
        return _build_coco_loader_cache_json_load(source, target)
    if build_mode == "streaming":
        return _build_coco_loader_cache_streaming(source, target)
    raise AssertionError(f"unhandled COCO loader cache build mode: {build_mode}")


def _build_coco_loader_cache_json_load(source: Path, target: Path) -> dict[str, Any]:
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
    manifest = _build_manifest(
        source=source,
        source_sha256=sha256_file(source),
        image_ids=image_ids,
        image_count=len(images),
        annotation_count=len(annotations),
        categories=categories,
        metadata=_top_level_metadata(data),
        build_mode="json-load",
    )

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


def _build_coco_loader_cache_streaming(source: Path, target: Path) -> dict[str, Any]:
    source_sha256 = sha256_file(source)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        with source.open("r", encoding="utf-8") as handle:
            parser = _JsonObjectStream(handle)
            with sqlite3.connect(tmp_path) as conn:
                conn.row_factory = sqlite3.Row
                _configure_write_connection(conn)
                _create_sqlite_schema(conn)
                counts, categories, metadata = _stream_source_into_sqlite(conn, parser)
                _validate_streamed_cache(conn, counts)
                image_ids = _read_sqlite_image_ids(conn)
                manifest = _build_manifest(
                    source=source,
                    source_sha256=source_sha256,
                    image_ids=image_ids,
                    image_count=counts["images"],
                    annotation_count=counts["annotations"],
                    categories=categories,
                    metadata=metadata,
                    build_mode="streaming",
                )
                _insert_manifest(conn, manifest)
                conn.commit()
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
        _configure_write_connection(conn)
        _create_sqlite_schema(conn)
        _insert_manifest(conn, manifest)
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


def _configure_write_connection(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=OFF")
    conn.execute("PRAGMA synchronous=OFF")


def _create_sqlite_schema(conn: sqlite3.Connection) -> None:
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


def _insert_manifest(conn: sqlite3.Connection, manifest: dict[str, Any]) -> None:
    conn.executemany(
        "INSERT INTO manifest(key, value) VALUES (?, ?)",
        [(_key, _json_dumps(value)) for _key, value in sorted(manifest.items())],
    )


def _stream_source_into_sqlite(
    conn: sqlite3.Connection,
    parser: "_JsonObjectStream",
) -> tuple[dict[str, int], list[dict[str, Any]], dict[str, Any]]:
    counts = {"images": 0, "annotations": 0, "categories": 0}
    categories: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {}
    seen_top_level_keys: set[str] = set()

    for key, value in parser.iter_top_level_values(TOP_LEVEL_STREAM_ARRAYS):
        if key in seen_top_level_keys:
            raise ValueError(f"duplicate top-level COCO field: {key}")
        seen_top_level_keys.add(key)

        if key == "images":
            for index, image in enumerate(value):
                _insert_streamed_image(conn, image, index)
                counts["images"] += 1
        elif key == "annotations":
            for index, ann in enumerate(value):
                _insert_streamed_annotation(conn, ann, index)
                counts["annotations"] += 1
        elif key == "categories":
            for index, category in enumerate(value):
                _insert_streamed_category(conn, category, index)
                categories.append(category)
                counts["categories"] += 1
        else:
            metadata[key] = value

    missing = [key for key in TOP_LEVEL_STREAM_ARRAYS if key not in seen_top_level_keys]
    if missing:
        raise ValueError(f"COCO source missing required top-level field(s): {missing}")
    info = metadata.get("info", {})
    licenses = metadata.get("licenses", [])
    if not isinstance(info, dict):
        raise ValueError("COCO info must be an object")
    if not isinstance(licenses, list):
        raise ValueError("COCO licenses must be a list")
    return counts, categories, metadata


def _insert_streamed_image(conn: sqlite3.Connection, image: dict[str, Any], index: int) -> None:
    _require_fields(image, REQUIRED_IMAGE_FIELDS, f"images[{index}]")
    image_id = _strict_int(image["id"], f"images[{index}].id")
    if not str(image["file_name"]):
        raise ValueError(f"images[{index}].file_name must be non-empty")
    _strict_positive_int(image["height"], f"images[{index}].height")
    _strict_positive_int(image["width"], f"images[{index}].width")
    try:
        conn.execute(
            "INSERT INTO images(id, file_name, json) VALUES (?, ?, ?)",
            (image_id, str(image["file_name"]), _json_dumps(image)),
        )
    except sqlite3.IntegrityError as exc:
        raise ValueError(f"duplicate image id in COCO source: {image_id}") from exc


def _insert_streamed_annotation(conn: sqlite3.Connection, ann: dict[str, Any], index: int) -> None:
    label = f"annotations[{index}]"
    _require_fields(ann, REQUIRED_ANNOTATION_FIELDS, label)
    ann_id = _strict_int(ann["id"], f"{label}.id")
    image_id = _strict_int(ann["image_id"], f"{label}.image_id")
    category_id = _strict_int(ann["category_id"], f"{label}.category_id")
    iscrowd = _strict_int(ann["iscrowd"], f"{label}.iscrowd")
    if iscrowd not in (0, 1):
        raise ValueError(f"{label}.iscrowd must be 0 or 1, got {iscrowd}")
    _validate_bbox(ann["bbox"], label)
    try:
        area = float(ann["area"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}.area must be numeric, got {ann['area']!r}") from exc
    try:
        conn.execute(
            """
            INSERT INTO annotations(id, image_id, category_id, iscrowd, area, ordinal, json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (ann_id, image_id, category_id, iscrowd, area, index, _json_dumps(ann)),
        )
    except sqlite3.IntegrityError as exc:
        raise ValueError(f"duplicate annotation id in COCO source: {ann_id}") from exc


def _insert_streamed_category(conn: sqlite3.Connection, category: dict[str, Any], index: int) -> None:
    _require_fields(category, REQUIRED_CATEGORY_FIELDS, f"categories[{index}]")
    category_id = _strict_int(category["id"], f"categories[{index}].id")
    if not str(category["name"]):
        raise ValueError(f"categories[{index}].name must be non-empty")
    try:
        conn.execute(
            "INSERT INTO categories(id, name, supercategory, json) VALUES (?, ?, ?, ?)",
            (
                category_id,
                str(category["name"]),
                category.get("supercategory"),
                _json_dumps(category),
            ),
        )
    except sqlite3.IntegrityError as exc:
        raise ValueError(f"duplicate category id in COCO source: {category_id}") from exc


def _validate_streamed_cache(conn: sqlite3.Connection, counts: dict[str, int]) -> None:
    for table, expected in counts.items():
        actual = int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
        if actual != expected:
            raise ValueError(f"streamed COCO {table} count mismatch: expected {expected}, got {actual}")

    rows = conn.execute(
        """
        SELECT
            annotations.ordinal AS ordinal,
            annotations.id AS annotation_id,
            annotations.image_id AS image_id,
            annotations.category_id AS category_id,
            annotations.json AS annotation_json,
            images.json AS image_json,
            categories.id AS existing_category_id
        FROM annotations
        LEFT JOIN images ON images.id = annotations.image_id
        LEFT JOIN categories ON categories.id = annotations.category_id
        ORDER BY annotations.ordinal
        """
    )
    for row in rows:
        ordinal = int(row["ordinal"])
        label = f"annotations[{ordinal}]"
        if row["image_json"] is None:
            raise ValueError(f"{label}.image_id {int(row['image_id'])} does not exist in images")
        if row["existing_category_id"] is None:
            raise ValueError(
                f"{label}.category_id {int(row['category_id'])} does not exist in categories"
            )
        ann = json.loads(str(row["annotation_json"]))
        image = json.loads(str(row["image_json"]))
        _validate_segmentation(ann["segmentation"], int(image["height"]), int(image["width"]), label)


def _read_sqlite_image_ids(conn: sqlite3.Connection) -> list[int]:
    rows = conn.execute("SELECT id FROM images ORDER BY id").fetchall()
    return [int(row[0]) for row in rows]


def _build_manifest(
    *,
    source: Path,
    source_sha256: str,
    image_ids: list[int],
    image_count: int,
    annotation_count: int,
    categories: list[dict[str, Any]],
    metadata: dict[str, Any],
    build_mode: str,
) -> dict[str, Any]:
    info = metadata.get("info", {})
    licenses = metadata.get("licenses", [])
    if not isinstance(info, dict):
        raise ValueError("COCO info must be an object")
    if not isinstance(licenses, list):
        raise ValueError("COCO licenses must be a list")
    return {
        "schema_version": SCHEMA_VERSION,
        "source_path": str(source),
        "source_sha256": source_sha256,
        "image_count": image_count,
        "image_ids": image_ids,
        "image_ids_digest": digest_json(image_ids),
        "annotation_count": annotation_count,
        "category_count": len(categories),
        "categories": categories,
        "info": info,
        "licenses": licenses,
        "metadata": metadata,
        "builder": f"tools/build_coco_loader_cache.py:{build_mode}",
        "build_mode": build_mode,
        "streaming_supported": True,
    }


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


def _resolve_build_mode(source: Path, mode: str, streaming_threshold_bytes: int) -> str:
    if mode not in {"auto", "json-load", "streaming"}:
        raise ValueError(f"mode must be one of auto, json-load, streaming; got {mode!r}")
    if mode != "auto":
        return mode
    if source.stat().st_size >= streaming_threshold_bytes:
        return "streaming"
    return "json-load"


def _top_level_metadata(data: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in data.items() if key not in TOP_LEVEL_STREAM_ARRAYS}


class _JsonObjectStream:
    def __init__(self, handle: TextIO, chunk_size: int = 1024 * 1024):
        self._handle = handle
        self._chunk_size = chunk_size
        self._decoder = json.JSONDecoder()
        self._buffer = ""
        self._position = 0
        self._eof = False

    def iter_top_level_values(
        self,
        stream_array_keys: set[str],
    ) -> Iterator[tuple[str, Any | Iterator[dict[str, Any]]]]:
        self._consume_char("{")
        if self._peek_non_ws() == "}":
            self._consume_char("}")
            self._ensure_trailing_whitespace()
            return

        while True:
            key = self._read_value()
            if not isinstance(key, str):
                raise ValueError(f"COCO top-level object key must be a string, got {key!r}")
            self._consume_char(":")
            if key in stream_array_keys:
                yield key, self._iter_array_items(key)
            else:
                yield key, self._read_value()

            next_char = self._peek_non_ws()
            if next_char == ",":
                self._consume_char(",")
                continue
            if next_char == "}":
                self._consume_char("}")
                self._ensure_trailing_whitespace()
                return
            raise ValueError(f"expected ',' or '}}' after top-level COCO field {key!r}")

    def _iter_array_items(self, key: str) -> Iterator[dict[str, Any]]:
        self._consume_char("[")
        if self._peek_non_ws() == "]":
            self._consume_char("]")
            return

        index = 0
        while True:
            item = self._read_value()
            if not isinstance(item, dict):
                raise ValueError(f"COCO {key}[{index}] must be an object")
            yield item
            index += 1

            next_char = self._peek_non_ws()
            if next_char == ",":
                self._consume_char(",")
                continue
            if next_char == "]":
                self._consume_char("]")
                return
            raise ValueError(f"expected ',' or ']' after COCO {key}[{index - 1}]")

    def _read_value(self) -> Any:
        self._skip_ws()
        start = self._position
        while True:
            try:
                value, end = self._decoder.raw_decode(self._buffer, start)
            except json.JSONDecodeError as exc:
                if self._eof:
                    raise ValueError(f"invalid COCO JSON near byte offset {self._position}") from exc
                self._fill()
                continue

            first_char = self._buffer[start]
            if end == len(self._buffer) and not self._eof and first_char in "-0123456789tfn":
                self._fill()
                continue

            self._position = end
            self._compact()
            return value

    def _consume_char(self, expected: str) -> None:
        actual = self._peek_non_ws()
        if actual != expected:
            raise ValueError(f"expected {expected!r} in COCO JSON, got {actual!r}")
        self._position += 1
        self._compact()

    def _peek_non_ws(self) -> str:
        self._skip_ws()
        if self._position >= len(self._buffer):
            raise ValueError("unexpected end of COCO JSON")
        return self._buffer[self._position]

    def _skip_ws(self) -> None:
        while True:
            while self._position < len(self._buffer) and self._buffer[self._position].isspace():
                self._position += 1
            if self._position < len(self._buffer) or self._eof:
                self._compact()
                return
            self._fill()

    def _ensure_trailing_whitespace(self) -> None:
        while True:
            self._skip_ws()
            if self._position < len(self._buffer):
                raise ValueError("unexpected trailing data after COCO JSON object")
            if self._eof:
                return
            self._fill()

    def _fill(self) -> None:
        chunk = self._handle.read(self._chunk_size)
        if chunk == "":
            self._eof = True
            return
        self._buffer += chunk

    def _compact(self) -> None:
        if self._position < self._chunk_size:
            return
        self._buffer = self._buffer[self._position :]
        self._position = 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a first-phase lazy COCO loader SQLite cache")
    parser.add_argument("source_json", type=Path)
    parser.add_argument("cache_path", type=Path)
    parser.add_argument(
        "--mode",
        choices=("auto", "json-load", "streaming"),
        default="auto",
        help="Cache build mode. auto uses streaming for large source JSON files.",
    )
    args = parser.parse_args()
    manifest = build_coco_loader_cache(args.source_json, args.cache_path, mode=args.mode)
    print(
        json.dumps(
            {
                "cache_path": str(args.cache_path.resolve()),
                "image_count": manifest["image_count"],
                "annotation_count": manifest["annotation_count"],
                "build_mode": manifest["build_mode"],
                "streaming_supported": manifest["streaming_supported"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
