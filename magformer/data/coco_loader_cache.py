from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is available in MAGFormer, keep helper standalone.
    NUMPY_INTEGER_TYPES: tuple[type[Any], ...] = ()
else:
    NUMPY_INTEGER_TYPES = (np.integer,)


SCHEMA_VERSION = 1


class CocoLoaderCache:
    """Read-only lazy SQLite backend for the COCO fields used by CocoRgbdDataset."""

    def __init__(self, cache_path: str | Path):
        self.cache_path = Path(cache_path).resolve()
        if not self.cache_path.exists():
            raise FileNotFoundError(f"COCO loader cache not found: {self.cache_path}")
        self._conn: sqlite3.Connection | None = None
        self._pid: int | None = None
        self._manifest: dict[str, Any] | None = None

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_conn"] = None
        state["_pid"] = None
        return state

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
        self._conn = None
        self._pid = None

    def _connect(self) -> sqlite3.Connection:
        pid = os.getpid()
        if self._conn is not None and self._pid == pid:
            return self._conn
        if self._conn is not None:
            self._conn.close()

        uri = f"file:{self.cache_path.as_posix()}?mode=ro&immutable=1"
        conn = sqlite3.connect(uri, uri=True)
        conn.row_factory = sqlite3.Row
        self._conn = conn
        self._pid = pid
        self._load_manifest()
        return conn

    @property
    def manifest(self) -> dict[str, Any]:
        self._connect()
        assert self._manifest is not None
        return self._manifest

    @property
    def dataset(self) -> dict[str, Any]:
        manifest = self.manifest
        return {
            "info": manifest.get("info", {}),
            "licenses": manifest.get("licenses", []),
            "categories": self.loadCats(self.getCatIds()),
        }

    @property
    def imgs(self) -> dict[int, dict[str, Any]]:
        return {image_id: self.loadImgs(image_id)[0] for image_id in self.getImgIds()}

    def _load_manifest(self) -> None:
        assert self._conn is not None
        rows = self._conn.execute("SELECT key, value FROM manifest").fetchall()
        manifest = {str(row["key"]): json.loads(row["value"]) for row in rows}
        schema_version = int(manifest.get("schema_version", -1))
        if schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported COCO loader cache schema version {schema_version}; "
                f"expected {SCHEMA_VERSION}: {self.cache_path}"
            )

        source_path = manifest.get("source_path")
        source_sha256 = manifest.get("source_sha256")
        if source_path and source_sha256:
            path = Path(str(source_path))
            if not path.exists():
                raise FileNotFoundError(
                    f"COCO loader cache source file missing: {path}; cache={self.cache_path}"
                )
            actual = sha256_file(path)
            if actual != source_sha256:
                raise ValueError(
                    "source hash mismatch for COCO loader cache: "
                    f"source={path} expected={source_sha256} actual={actual} cache={self.cache_path}"
                )

        image_ids = self._read_image_ids()
        if image_ids != manifest.get("image_ids", []):
            raise ValueError(f"image id manifest mismatch in COCO loader cache: {self.cache_path}")
        if digest_json(image_ids) != manifest.get("image_ids_digest"):
            raise ValueError(f"image id digest mismatch in COCO loader cache: {self.cache_path}")
        ann_count = int(
            self._conn.execute("SELECT COUNT(*) AS count FROM annotations").fetchone()["count"]
        )
        if ann_count != int(manifest.get("annotation_count", -1)):
            raise ValueError(f"annotation count mismatch in COCO loader cache: {self.cache_path}")

        self._manifest = manifest

    def _read_image_ids(self) -> list[int]:
        assert self._conn is not None
        rows = self._conn.execute("SELECT id FROM images ORDER BY id").fetchall()
        return [int(row["id"]) for row in rows]

    def getImgIds(self, imgIds: Any = [], catIds: Any = []) -> list[int]:
        conn = self._connect()
        image_ids = _normalize_ids(imgIds)
        category_ids = _normalize_ids(catIds)
        if not image_ids and not category_ids:
            return list(self.manifest["image_ids"])
        clauses: list[str] = []
        params: list[int] = []
        if image_ids:
            clauses.append(f"images.id IN ({_placeholders(image_ids)})")
            params.extend(image_ids)
        if category_ids:
            clauses.append(f"annotations.category_id IN ({_placeholders(category_ids)})")
            params.extend(category_ids)
            query = (
                "SELECT DISTINCT images.id FROM images "
                "JOIN annotations ON annotations.image_id = images.id "
                f"WHERE {' AND '.join(clauses)} ORDER BY images.id"
            )
        else:
            query = f"SELECT id FROM images WHERE {' AND '.join(clauses)} ORDER BY id"
        rows = conn.execute(query, params).fetchall()
        return [int(row[0]) for row in rows]

    def loadImgs(self, ids: Any = []) -> list[dict[str, Any]]:
        image_ids = _normalize_ids(ids)
        if not image_ids:
            return []
        conn = self._connect()
        rows = conn.execute(
            f"SELECT id, json FROM images WHERE id IN ({_placeholders(image_ids)})", image_ids
        ).fetchall()
        by_id = {int(row["id"]): json.loads(row["json"]) for row in rows}
        missing = [image_id for image_id in image_ids if image_id not in by_id]
        if missing:
            raise KeyError(f"missing image ids in COCO loader cache {self.cache_path}: {missing}")
        return [by_id[image_id] for image_id in image_ids]

    def getAnnIds(
        self,
        imgIds: Any = [],
        catIds: Any = [],
        areaRng: Any = [],
        iscrowd: Any = None,
    ) -> list[int]:
        conn = self._connect()
        clauses: list[str] = []
        params: list[Any] = []
        image_ids = _normalize_ids(imgIds)
        category_ids = _normalize_ids(catIds)
        if image_ids:
            clauses.append(f"image_id IN ({_placeholders(image_ids)})")
            params.extend(image_ids)
        if category_ids:
            clauses.append(f"category_id IN ({_placeholders(category_ids)})")
            params.extend(category_ids)
        if areaRng:
            if len(areaRng) != 2:
                raise ValueError(f"areaRng must contain [min, max], got {areaRng}")
            clauses.append("area BETWEEN ? AND ?")
            params.extend([float(areaRng[0]), float(areaRng[1])])
        if iscrowd is not None:
            clauses.append("iscrowd = ?")
            params.append(int(iscrowd))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = conn.execute(f"SELECT id FROM annotations {where} ORDER BY ordinal", params).fetchall()
        return [int(row["id"]) for row in rows]

    def loadAnns(self, ids: Any = []) -> list[dict[str, Any]]:
        ann_ids = _normalize_ids(ids)
        if not ann_ids:
            return []
        conn = self._connect()
        rows = conn.execute(
            f"SELECT id, json FROM annotations WHERE id IN ({_placeholders(ann_ids)})", ann_ids
        ).fetchall()
        by_id = {int(row["id"]): json.loads(row["json"]) for row in rows}
        missing = [ann_id for ann_id in ann_ids if ann_id not in by_id]
        if missing:
            raise KeyError(f"missing annotation ids in COCO loader cache {self.cache_path}: {missing}")
        return [by_id[ann_id] for ann_id in ann_ids]

    def getCatIds(self, catIds: Any = [], catNms: Any = [], supNms: Any = []) -> list[int]:
        category_ids = _normalize_ids(catIds)
        conn = self._connect()
        clauses: list[str] = []
        params: list[Any] = []
        if category_ids:
            clauses.append(f"id IN ({_placeholders(category_ids)})")
            params.extend(category_ids)
        if catNms:
            names = _normalize_strings(catNms)
            clauses.append(f"name IN ({_placeholders(names)})")
            params.extend(names)
        if supNms:
            names = _normalize_strings(supNms)
            clauses.append(f"supercategory IN ({_placeholders(names)})")
            params.extend(names)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = conn.execute(f"SELECT id FROM categories {where} ORDER BY id", params).fetchall()
        return [int(row["id"]) for row in rows]

    def loadCats(self, ids: Any = []) -> list[dict[str, Any]]:
        category_ids = _normalize_ids(ids)
        if not category_ids:
            return []
        conn = self._connect()
        rows = conn.execute(
            f"SELECT id, json FROM categories WHERE id IN ({_placeholders(category_ids)})",
            category_ids,
        ).fetchall()
        by_id = {int(row["id"]): json.loads(row["json"]) for row in rows}
        missing = [category_id for category_id in category_ids if category_id not in by_id]
        if missing:
            raise KeyError(f"missing category ids in COCO loader cache {self.cache_path}: {missing}")
        return [by_id[category_id] for category_id in category_ids]

    def get_annotation_category_ids(self, include_crowd: bool = False) -> list[int]:
        conn = self._connect()
        where = "" if include_crowd else "WHERE iscrowd = 0"
        rows = conn.execute(
            f"SELECT DISTINCT category_id FROM annotations {where} ORDER BY category_id"
        ).fetchall()
        return [int(row["category_id"]) for row in rows]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _normalize_ids(ids: Any) -> list[int]:
    if ids is None or ids == []:
        return []
    if isinstance(ids, (int,) + NUMPY_INTEGER_TYPES):
        return [int(ids)]
    return [int(value) for value in ids]


def _normalize_strings(values: Any) -> list[str]:
    if isinstance(values, str):
        return [values]
    return [str(value) for value in values]


def _placeholders(values: list[Any]) -> str:
    if not values:
        raise ValueError("cannot create SQL placeholders for an empty list")
    return ",".join("?" for _ in values)
