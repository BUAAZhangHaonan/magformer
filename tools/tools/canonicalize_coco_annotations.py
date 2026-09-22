#!/usr/bin/env python3
"""Build a canonical, trainable COCO annotation file without decoding masks.

The input is parsed with :mod:`ijson` and annotation identity/reference state is
kept in a temporary SQLite database.  Memory therefore scales with one COCO
record, rather than with the annotation JSON size.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import sys
import tempfile
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from pycocotools import mask as mask_utils


TOOL_VERSION = "1.0.0"
PROGRESS_EVERY = 100_000
REMOVAL_REASONS = (
    "missing_segmentation",
    "malformed_segmentation",
    "unconvertible_segmentation",
    "polygon_zero_area",
    "rle_zero_area",
)


class CocoValidationError(ValueError):
    """The input violates a COCO contract required for reproducible training."""


class MissingStreamingDependencyError(RuntimeError):
    """The environment lacks the deliberately required streaming JSON parser."""


class EventValueBuilder:
    """Build one JSON value from ijson events without retaining sibling values."""

    def __init__(self) -> None:
        self._stack: list[Any] = []
        self._keys: list[str | None] = []
        self.value: Any = _UNSET

    def event(self, event: str, value: Any) -> None:
        if event == "map_key":
            if not self._stack or not isinstance(self._stack[-1], dict):
                raise CocoValidationError("invalid JSON map_key event")
            self._keys[-1] = value
            return
        if event == "start_map":
            self._stack.append(self._attach({}))
            self._keys.append(None)
            return
        if event == "start_array":
            self._stack.append(self._attach([]))
            self._keys.append(None)
            return
        if event in {"string", "number", "integer", "double", "boolean", "null"}:
            self._attach(value)
            return
        if event in {"end_map", "end_array"}:
            if not self._stack:
                raise CocoValidationError("invalid JSON container end event")
            self._stack.pop()
            self._keys.pop()
            return
        raise CocoValidationError(f"unsupported streaming JSON event: {event}")

    def _attach(self, child: Any) -> Any:
        if not self._stack:
            if self.value is not _UNSET:
                raise CocoValidationError("stream attempted to build multiple root values")
            self.value = child
            return child
        parent = self._stack[-1]
        if isinstance(parent, list):
            parent.append(child)
            return child
        key = self._keys[-1]
        if key is None:
            raise CocoValidationError("JSON object value appeared without a key")
        parent[key] = child
        self._keys[-1] = None
        return child


_UNSET = object()


def _require_ijson() -> Any:
    try:
        import ijson  # type: ignore
    except ImportError as exc:
        raise MissingStreamingDependencyError(
            "ijson is required for bounded-memory COCO canonicalization. "
            "Install it in the magformer environment with: "
            "python -m pip install 'ijson==3.4.0.post0'"
        ) from exc
    return ijson


def _id_key(value: Any, *, field: str, context: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise CocoValidationError(f"{context}: {field} must be an int or string, got {value!r}")
    return json.dumps([type(value).__name__, value], ensure_ascii=False, separators=(",", ":"))


def _positive_dimension(value: Any, *, field: str, image_id: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CocoValidationError(f"image_id={image_id!r}: {field} must be a positive integer")
    return value


def _json_equal(left: Any, right: Any) -> bool:
    if isinstance(left, Sequence) and not isinstance(left, (str, bytes)):
        return isinstance(right, Sequence) and not isinstance(right, (str, bytes)) and list(left) == list(right)
    return left == right


def _canonical_rle(segmentation: Any, height: int, width: int) -> tuple[dict[str, Any], str]:
    if segmentation is None:
        raise _Removal("missing_segmentation")
    try:
        if isinstance(segmentation, list):
            if not segmentation or any(
                not isinstance(poly, list)
                or len(poly) < 6
                or len(poly) % 2
                or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in poly)
                for poly in segmentation
            ):
                raise _Removal("malformed_segmentation")
            rles = mask_utils.frPyObjects(segmentation, height, width)
            rle = mask_utils.merge(rles) if isinstance(rles, list) else rles
            return rle, "polygon"
        if isinstance(segmentation, Mapping):
            size = segmentation.get("size")
            counts = segmentation.get("counts")
            if (
                not isinstance(size, list)
                or len(size) != 2
                or size != [height, width]
                or counts is None
            ):
                raise _Removal("malformed_segmentation")
            if isinstance(counts, list):
                rle = mask_utils.frPyObjects(dict(segmentation), height, width)
            elif isinstance(counts, str):
                rle = dict(segmentation)
                rle["counts"] = counts.encode("ascii")
            else:
                raise _Removal("malformed_segmentation")
            return rle, "rle"
        raise _Removal("malformed_segmentation")
    except _Removal:
        raise
    except Exception as exc:
        raise _Removal("unconvertible_segmentation") from exc


class _Removal(Exception):
    def __init__(self, reason: str) -> None:
        self.reason = reason


def canonicalize_annotation(annotation: Mapping[str, Any], height: int, width: int) -> tuple[dict[str, Any] | None, str | None, bool, bool]:
    """Return canonical annotation, removal reason, and changed bbox/area flags."""
    try:
        rle, representation = _canonical_rle(annotation.get("segmentation"), height, width)
        area = float(mask_utils.area(rle))
        bbox = [float(x) for x in mask_utils.toBbox(rle).tolist()]
    except _Removal as removal:
        return None, removal.reason, False, False
    except Exception:
        return None, "unconvertible_segmentation", False, False
    if not area > 0.0:
        return None, f"{representation}_zero_area", False, False
    result = dict(annotation)
    bbox_changed = not _json_equal(annotation.get("bbox"), bbox)
    area_changed = not _json_equal(annotation.get("area"), area)
    result["bbox"] = bbox
    result["area"] = area
    return result, None, bbox_changed, area_changed


@contextmanager
def _database(parent_directory: Path) -> Iterator[sqlite3.Connection]:
    """Keep the potentially large ID index on the destination filesystem, not tmpfs."""
    with tempfile.TemporaryDirectory(prefix=".magformer_coco_canonicalize_", dir=parent_directory) as directory:
        conn = sqlite3.connect(Path(directory) / "index.sqlite3")
        try:
            conn.executescript(
                """
                PRAGMA journal_mode=OFF;
                PRAGMA synchronous=OFF;
                CREATE TABLE images (
                    id_key TEXT PRIMARY KEY, id_json TEXT NOT NULL, height INTEGER NOT NULL,
                    width INTEGER NOT NULL, retained_count INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE categories (id_key TEXT PRIMARY KEY, id_json TEXT NOT NULL);
                CREATE TABLE annotation_ids (
                    id_key TEXT PRIMARY KEY, id_json TEXT NOT NULL, image_key TEXT NOT NULL,
                    category_key TEXT NOT NULL, retained INTEGER NOT NULL
                );
                CREATE TABLE removed (
                    sequence INTEGER PRIMARY KEY, id_json TEXT NOT NULL, reason TEXT NOT NULL
                );
                """
            )
            yield conn
            conn.commit()
        finally:
            conn.close()


def _first_pass(input_path: Path, conn: sqlite3.Connection) -> tuple[Any, list[Any], list[dict[str, Any]], list[dict[str, Any]]]:
    ijson = _require_ijson()
    info: Any = None
    licenses: list[Any] = []
    images: list[dict[str, Any]] = []
    categories: list[dict[str, Any]] = []
    builders: dict[str, EventValueBuilder | None] = {"info": None, "licenses": None, "images": None, "categories": None}
    seen_arrays = set()

    def finalize(kind: str, value: Any) -> None:
        nonlocal info
        if kind == "info":
            if info is not None:
                raise CocoValidationError("top-level info appears more than once")
            info = value
            return
        if kind == "licenses":
            licenses.append(value)
            return
        if not isinstance(value, dict):
            raise CocoValidationError(f"{kind} entries must be JSON objects")
        if kind == "images":
            image_id = value.get("id")
            key = _id_key(image_id, field="id", context="image")
            height = _positive_dimension(value.get("height"), field="height", image_id=image_id)
            width = _positive_dimension(value.get("width"), field="width", image_id=image_id)
            try:
                conn.execute("INSERT INTO images VALUES (?, ?, ?, ?, 0)", (key, _compact(image_id), height, width))
            except sqlite3.IntegrityError as exc:
                raise CocoValidationError(f"duplicate image id: {image_id!r}") from exc
            images.append(value)
            return
        category_id = value.get("id")
        key = _id_key(category_id, field="id", context="category")
        try:
            conn.execute("INSERT INTO categories VALUES (?, ?)", (key, _compact(category_id)))
        except sqlite3.IntegrityError as exc:
            raise CocoValidationError(f"duplicate category id: {category_id!r}") from exc
        categories.append(value)

    with input_path.open("rb") as handle:
        for prefix, event, value in ijson.parse(handle):
            if prefix in {"images", "categories", "annotations", "licenses"} and event == "start_array":
                seen_arrays.add(prefix)
            if prefix == "info" and event == "start_map":
                builders["info"] = EventValueBuilder()
            if prefix in {"licenses.item", "images.item", "categories.item"} and event in {"start_map", "start_array"}:
                kind = prefix.split(".", 1)[0]
                if builders[kind] is not None:
                    raise CocoValidationError(f"nested concurrent {kind} stream item")
                builders[kind] = EventValueBuilder()
            for kind, builder in tuple(builders.items()):
                if builder is None:
                    continue
                root_prefix = kind if kind == "info" else f"{kind}.item"
                if prefix == root_prefix or prefix.startswith(root_prefix + "."):
                    builder.event(event, value)
                    is_end = event in {"end_map", "end_array"} and prefix == root_prefix
                    if is_end:
                        if builder.value is _UNSET:
                            raise CocoValidationError(f"failed to parse {kind}")
                        finalize(kind, builder.value)
                        builders[kind] = None
    if not {"images", "categories", "annotations"}.issubset(seen_arrays):
        missing = sorted({"images", "categories", "annotations"} - seen_arrays)
        raise CocoValidationError(f"missing required top-level arrays: {missing}")
    if not images:
        raise CocoValidationError("COCO images array is empty")
    return info, licenses, images, categories


def _compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)



def _ensure_output_paths(input_path: Path, output_path: Path, manifest_path: Path, overwrite: bool) -> None:
    for path in (output_path, manifest_path):
        if path.resolve() == input_path.resolve():
            raise CocoValidationError("input, output, and manifest paths must be distinct")
        if path.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite existing path without --overwrite: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)


def _write_header(handle: Any, info: Any, licenses: list[Any], images: list[dict[str, Any]], categories: list[dict[str, Any]]) -> None:
    handle.write('{"info":')
    handle.write(_compact(info if info is not None else {}))
    handle.write(',"licenses":')
    handle.write(_compact(licenses))
    handle.write(',"images":')
    handle.write(_compact(sorted(images, key=lambda item: _id_key(item["id"], field="id", context="image"))))
    handle.write(',"categories":')
    handle.write(_compact(sorted(categories, key=lambda item: _id_key(item["id"], field="id", context="category"))))
    handle.write(',"annotations":[')


def _annotation_pass(
    input_path: Path,
    conn: sqlite3.Connection,
    output_handle: Any | None,
    *,
    validate_only: bool,
) -> tuple[Counter[str], Counter[str], int, int, int, int, Counter[str]]:
    ijson = _require_ijson()
    source_count = retained_count = corrected_bbox = corrected_area = 0
    removed = Counter()
    categories_before = Counter()
    categories_retained = Counter()
    wrote_annotation = False
    with input_path.open("rb") as handle:
        for annotation in ijson.items(handle, "annotations.item"):
            source_count += 1
            if source_count % PROGRESS_EVERY == 0:
                print(f"processed annotations={source_count} retained={retained_count} removed={sum(removed.values())}", file=sys.stderr, flush=True)
                conn.commit()
            if not isinstance(annotation, dict):
                raise CocoValidationError(f"annotation #{source_count} is not a JSON object")
            annotation_id = annotation.get("id")
            annotation_key = _id_key(annotation_id, field="id", context=f"annotation #{source_count}")
            image_id = annotation.get("image_id")
            category_id = annotation.get("category_id")
            image_key = _id_key(image_id, field="image_id", context=f"annotation_id={annotation_id!r}")
            category_key = _id_key(category_id, field="category_id", context=f"annotation_id={annotation_id!r}")
            image = conn.execute("SELECT height, width FROM images WHERE id_key=?", (image_key,)).fetchone()
            if image is None:
                raise CocoValidationError(f"annotation_id={annotation_id!r}: dangling image_id={image_id!r}")
            if conn.execute("SELECT 1 FROM categories WHERE id_key=?", (category_key,)).fetchone() is None:
                raise CocoValidationError(f"annotation_id={annotation_id!r}: dangling category_id={category_id!r}")
            try:
                conn.execute(
                    "INSERT INTO annotation_ids VALUES (?, ?, ?, ?, 0)",
                    (annotation_key, _compact(annotation_id), image_key, category_key),
                )
            except sqlite3.IntegrityError as exc:
                raise CocoValidationError(f"duplicate annotation id: {annotation_id!r}") from exc
            categories_before[category_key] += 1
            canonical, reason, bbox_changed, area_changed = canonicalize_annotation(annotation, image[0], image[1])
            if reason is not None:
                if validate_only:
                    raise CocoValidationError(f"annotation_id={annotation_id!r} is invalid: {reason}")
                removed[reason] += 1
                conn.execute("INSERT INTO removed VALUES (?, ?, ?)", (source_count, _compact(annotation_id), reason))
                continue
            assert canonical is not None
            if validate_only and (bbox_changed or area_changed):
                raise CocoValidationError(f"annotation_id={annotation_id!r} has non-canonical bbox or area")
            retained_count += 1
            categories_retained[category_key] += 1
            corrected_bbox += int(bbox_changed)
            corrected_area += int(area_changed)
            conn.execute("UPDATE annotation_ids SET retained=1 WHERE id_key=?", (annotation_key,))
            conn.execute("UPDATE images SET retained_count=retained_count+1 WHERE id_key=?", (image_key,))
            if output_handle is not None:
                if wrote_annotation:
                    output_handle.write(",")
                output_handle.write(_compact(canonical))
                wrote_annotation = True
    return categories_before, categories_retained, source_count, retained_count, corrected_bbox, corrected_area, removed


def _stream_json_values(handle: Any, rows: Iterator[tuple[str]]) -> None:
    first = True
    for (value,) in rows:
        if not first:
            handle.write(",")
        handle.write(value)
        first = False


def _write_manifest(
    manifest_path: Path,
    *,
    source_count: int,
    retained_count: int,
    corrected_bbox: int,
    corrected_area: int,
    conn: sqlite3.Connection,
) -> None:
    temporary = tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=manifest_path.parent, prefix=f".{manifest_path.name}.", suffix=".tmp", delete=False)
    temporary_path = Path(temporary.name)
    try:
        with temporary as handle:
            handle.write('{"tool":{"name":"canonicalize_coco_annotations","version":')
            handle.write(_compact(TOOL_VERSION))
            handle.write('},"config":{"mask_decoder":"pycocotools_rle","invalid_annotation_policy":"remove_only","output_order":"fixed_top_level_and_source_annotation_order"}')
            handle.write(',"annotation_counts":')
            handle.write(_compact({"original": source_count, "retained": retained_count, "removed": source_count - retained_count}))
            handle.write(',"canonical_corrections":')
            handle.write(_compact({"bbox": corrected_bbox, "area": corrected_area}))
            handle.write(',"removed_ids_by_reason":{')
            first_reason = True
            for reason in REMOVAL_REASONS:
                count = conn.execute("SELECT COUNT(*) FROM removed WHERE reason=?", (reason,)).fetchone()[0]
                if not count:
                    continue
                if not first_reason:
                    handle.write(",")
                handle.write(_compact(reason))
                handle.write(":[")
                _stream_json_values(handle, conn.execute("SELECT id_json FROM removed WHERE reason=? ORDER BY sequence", (reason,)))
                handle.write("]")
                first_reason = False
            handle.write("}")
            handle.write(',"images_that_become_zero_instance":[')
            _stream_json_values(handle, conn.execute("SELECT id_json FROM images WHERE retained_count=0 ORDER BY id_json"))
            handle.write("]")
            handle.write(',"category_counts":{')
            first_category = True
            for category_id, original, retained in conn.execute(
                "SELECT c.id_json, COUNT(a.id_key), COALESCE(SUM(a.retained), 0) "
                "FROM categories c LEFT JOIN annotation_ids a ON a.category_key=c.id_key "
                "GROUP BY c.id_key ORDER BY c.id_json"
            ):
                if not first_category:
                    handle.write(",")
                handle.write(_compact(str(json.loads(category_id))))
                handle.write(":")
                handle.write(_compact({"original": original, "retained": retained, "removed": original - retained}))
                first_category = False
            handle.write("}}")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, manifest_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def canonicalize_file(input_path: Path, output_path: Path, manifest_path: Path, *, overwrite: bool = False, validate_only: bool = False) -> None:
    input_path = input_path.resolve()
    output_path = output_path.resolve()
    manifest_path = manifest_path.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"input COCO JSON does not exist: {input_path}")
    _require_ijson()
    if validate_only:
        if output_path != input_path or manifest_path != input_path:
            raise CocoValidationError("--validate-only accepts the existing JSON with --input only; omit --output and --manifest")
    else:
        _ensure_output_paths(input_path, output_path, manifest_path, overwrite)
    database_parent = input_path.parent if validate_only else output_path.parent
    with _database(database_parent) as conn:
        info, licenses, images, categories = _first_pass(input_path, conn)
        if validate_only:
            _annotation_pass(input_path, conn, None, validate_only=True)
            print(f"validation passed: {input_path}", file=sys.stderr)
            return
        temporary = tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=output_path.parent, prefix=f".{output_path.name}.", suffix=".tmp", delete=False)
        temporary_path = Path(temporary.name)
        try:
            with temporary as handle:
                _write_header(handle, info, licenses, images, categories)
                _, _, source_count, retained_count, corrected_bbox, corrected_area, _ = _annotation_pass(input_path, conn, handle, validate_only=False)
                handle.write("]}")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, output_path)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise
        _write_manifest(
            manifest_path,
            source_count=source_count,
            retained_count=retained_count,
            corrected_bbox=corrected_bbox,
            corrected_area=corrected_area,
            conn=conn,
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="source COCO JSON, or existing canonical JSON with --validate-only")
    parser.add_argument("--output", type=Path, help="canonicalized COCO JSON")
    parser.add_argument("--manifest", type=Path, help="manifest JSON")
    parser.add_argument("--overwrite", action="store_true", help="atomically replace existing output and manifest")
    parser.add_argument("--validate-only", action="store_true", help="validate an existing canonical JSON without writing files")
    args = parser.parse_args(argv)
    if args.validate_only:
        if args.output is not None or args.manifest is not None or args.overwrite:
            parser.error("--validate-only cannot be combined with --output, --manifest, or --overwrite")
    elif args.output is None or args.manifest is None:
        parser.error("--output and --manifest are required unless --validate-only is used")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.validate_only:
            canonicalize_file(args.input, args.input, args.input, validate_only=True)
        else:
            canonicalize_file(args.input, args.output, args.manifest, overwrite=args.overwrite)
    except (CocoValidationError, MissingStreamingDependencyError, FileExistsError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
