# -*- coding: utf-8 -*-
"""Threaded producer/consumer overlap for the eval export path.

Topology (``procs`` backend, the default choice on this workload):

    main thread          transfer thread              N worker processes
    ----------           ---------------              ------------------
    GPU forward          slot.event.synchronize()     unpack bits ->
    GPU postprocess      copy payload out of pinned   np.asfortranarray (free,
    async DtoH ring  ->  ring slot (~13 MB/img)   ->  F-contiguous views)
    (no host sync)       release ring slot            pycocotools RLE encode
    next forward         pool.apply_async(task)       COCO row building

The heavy per-image CPU export work (``np.unpackbits`` over ~12.8 MB of
bit-packed masks plus ~100 ``coco_mask.encode`` calls) is moved off the
inference thread entirely. ``pycocotools.mask.encode`` does NOT release the
GIL (measured: 2 encode threads are ~2x SLOWER than serial), so real
parallelism requires worker *processes*; the ``threads`` backend is kept for
measurement and runs the same code in consumer threads.

Ordering / identity: the transfer thread is the single submitter, tasks carry
monotonic sequence numbers, and ``drain()`` collects results strictly in
sequence order. ``magformer.engine.coco_export.process_export_task``
replicates the serial ``predictions_to_coco_instances`` fast path exactly, so
the aggregated COCO rows are byte-identical to the synchronous path.

Backpressure: in-flight images are bounded by the pinned ring depth (slots
are only released after their payload has been copied out) and by the
bounded hand-off queue between the main thread and the transfer thread.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..models.magformer.gpu_postprocess import DeferredExportRing


class ExportPipeline:
    """Overlapped GPU inference -> host export pipeline.

    Args:
        workers: number of consumer processes/threads.
        depth: pinned ring depth (bounds in-flight images).
        backend: "procs" (multiprocessing pool) or "threads".
        pool: pre-created ``multiprocessing.Pool`` (fork context, created
            before CUDA initialization). Required for "procs".
        category_id_list / category_offset / thresholds: forwarded verbatim
            to every worker task (same values as the serial export path).
    """

    def __init__(
        self,
        workers: int = 4,
        depth: int = 8,
        backend: str = "procs",
        category_id_list: Optional[List[int]] = None,
        category_offset: int = 1,
        score_threshold: float = 0.0,
        mask_threshold: float = 0.5,
        allow_empty_fallback: bool = False,
        empty_fallback_ratio: float = 0.01,
        pool: Optional[Any] = None,
    ):
        if backend not in ("procs", "threads"):
            raise ValueError(f"unknown export pipeline backend: {backend!r}")
        if backend == "procs" and pool is None:
            raise ValueError("procs backend requires a pre-created pool")
        if int(workers) < 1:
            raise ValueError(f"workers must be >= 1, got {workers}")
        self.backend = backend
        self.pool = pool
        self.workers = int(workers)
        self.ring = DeferredExportRing(depth=max(2, int(depth)))
        self.category_id_list = category_id_list
        self.category_offset = int(category_offset)
        self.score_threshold = float(score_threshold)
        self.mask_threshold = float(mask_threshold)
        self.allow_empty_fallback = bool(allow_empty_fallback)
        self.empty_fallback_ratio = float(empty_fallback_ratio)

        self._q_in: "queue.Queue" = queue.Queue(maxsize=max(2, int(depth)))
        self._q_tasks: "queue.Queue" = queue.Queue()
        self._results: Dict[int, Tuple[int, List[Dict[str, Any]]]] = {}
        self._results_exc: Optional[BaseException] = None
        self._results_cond = threading.Condition()
        self._expected = 0
        self._pending: List[Tuple[int, int, Any]] = []
        self._pending_lock = threading.Lock()
        self._transfer_thread: Optional[threading.Thread] = None
        self._consumer_threads: List[threading.Thread] = []
        # lightweight instrumentation (no behavioral effect)
        self.stats: Dict[str, float] = {
            "transfer_wait_s": 0.0,
            "copy_out_s": 0.0,
            "submit_s": 0.0,
        }

    # -- producer API (main thread) ---------------------------------------

    def start(self) -> None:
        self._transfer_thread = threading.Thread(
            target=self._transfer_loop, name="export-transfer", daemon=True
        )
        self._transfer_thread.start()
        if self.backend == "threads":
            for w in range(self.workers):
                t = threading.Thread(
                    target=self._consumer_loop,
                    name=f"export-worker-{w}",
                    daemon=True,
                )
                t.start()
                self._consumer_threads.append(t)

    def submit(self, seq: int, image_ids: List[int], slot: Any) -> None:
        """Hand one exported batch (pinned ring slot) to the transfer thread."""
        self._expected = int(seq) + len(image_ids)
        self._q_in.put((int(seq), [int(i) for i in image_ids], slot))

    # -- transfer thread ---------------------------------------------------

    def _transfer_loop(self) -> None:
        from .coco_export import process_export_task  # light: numpy + pycocotools

        while True:
            item = self._q_in.get()
            if item is None:
                break
            seq, image_ids, slot = item
            try:
                t0 = time.perf_counter()
                slot.event.synchronize()
                t1 = time.perf_counter()
                self.stats["transfer_wait_s"] += t1 - t0

                scores_np = slot.scores.numpy()
                cats_np = slot.cats.numpy()
                bbox_np = slot.bbox.numpy()
                valid_np = slot.valid.numpy().astype(bool)
                packed_np = slot.packed.numpy()
                B, K, W, packed_last = packed_np.shape
                H = int(slot.key[2])

                tasks = []
                for i, image_id in enumerate(image_ids):
                    tasks.append(
                        (
                            seq + i,
                            image_id,
                            {
                                "seq": seq + i,
                                "image_id": image_id,
                                "scores": np.array(scores_np[i], dtype=np.float32),
                                "cats": np.array(cats_np[i], dtype=np.int64),
                                "bbox": np.array(bbox_np[i], dtype=np.float32),
                                "valid": np.array(valid_np[i], dtype=bool),
                                "packed": np.array(packed_np[i], dtype=np.uint8),
                                "H": H,
                                "W": W,
                                "category_id_list": self.category_id_list,
                                "category_offset": self.category_offset,
                                "score_threshold": self.score_threshold,
                                "mask_threshold": self.mask_threshold,
                                "allow_empty_fallback": self.allow_empty_fallback,
                                "empty_fallback_ratio": self.empty_fallback_ratio,
                            },
                        )
                    )
                self.ring.release(slot)
                t2 = time.perf_counter()
                self.stats["copy_out_s"] += t2 - t1

                for task_seq, image_id, task in tasks:
                    if self.backend == "procs":
                        ar = self.pool.apply_async(process_export_task, (task,))
                        with self._pending_lock:
                            self._pending.append((task_seq, image_id, ar))
                    else:
                        self._q_tasks.put((task_seq, image_id, task))
                self.stats["submit_s"] += time.perf_counter() - t2
            except BaseException as exc:  # pragma: no cover - defensive
                with self._results_cond:
                    self._results_exc = exc
                    self._results_cond.notify_all()
                continue

    # -- consumer threads (threads backend only) ---------------------------

    def _consumer_loop(self) -> None:
        from .coco_export import process_export_task

        while True:
            item = self._q_tasks.get()
            if item is None:
                break
            task_seq, image_id, task = item
            try:
                out = process_export_task(task)
                with self._results_cond:
                    self._results[task_seq] = (image_id, out["rows"])
                    self._results_cond.notify_all()
            except BaseException as exc:
                with self._results_cond:
                    self._results_exc = exc
                    self._results_cond.notify_all()

    # -- drain (main thread) ------------------------------------------------

    def drain(self) -> List[Tuple[int, List[Dict[str, Any]]]]:
        """Block until every submitted image is done; return rows in order."""
        self._q_in.put(None)
        if self._transfer_thread is not None:
            self._transfer_thread.join()

        if self.backend == "procs":
            with self._pending_lock:
                pending = list(self._pending)
                self._pending = []
            ordered: List[Tuple[int, List[Dict[str, Any]]]] = []
            for task_seq, image_id, ar in pending:
                out = ar.get()  # raises if the worker raised
                ordered.append((image_id, out["rows"]))
            if self._results_exc is not None:
                raise self._results_exc
            return ordered

        # threads backend
        while True:
            with self._results_cond:
                if self._results_exc is not None:
                    raise self._results_exc
                if len(self._results) >= self._expected:
                    break
                self._results_cond.wait(timeout=1.0)
        ordered = [
            self._results[seq] for seq in range(self._expected)
        ]
        for _ in self._consumer_threads:
            self._q_tasks.put(None)
        for t in self._consumer_threads:
            t.join(timeout=5.0)
        return ordered

    def close(self) -> None:
        try:
            self._q_in.put_nowait(None)
        except queue.Full:
            pass
        if self._transfer_thread is not None:
            self._transfer_thread.join(timeout=5.0)
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
