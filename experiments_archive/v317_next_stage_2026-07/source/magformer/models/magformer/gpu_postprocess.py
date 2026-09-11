# -*- coding: utf-8 -*-
"""GPU-native inference postprocess + compact async device-to-host transfer.

This module backs the ``gpu_export=True`` path of
``MagFormerArch.forward_inference_raw``. It performs, entirely on the GPU:

  * per-image top-k selection / sigmoid / bilinear mask upsampling /
    binarization / mask-score fusion (bit-identical op sequence to
    ``_inference_raw``),
  * bbox extraction from the binary masks (exact same semantics as
    ``magformer.engine.coco_export._bbox_xyxy_from_binary_mask``:
    half-open XYXY from nonzero extents, empty -> invalid),
  * mask layout for transfer, either (default, ``pack_masks=False``) a
    column-major contiguous uint8 {0,1} copy on device transferred as
    ~105 MB for 100x1024x1024, or (``pack_masks=True``) bit-packed to
    ~13 MB with host-side ``np.unpackbits``.

The compact payload (scores float32, category_ids int64, bbox_xyxy float32,
valid flags + masks) is moved to pre-allocated pinned host buffers via a
handful of non-blocking copies on a dedicated CUDA side stream, followed by
ONE event synchronize (the reference path instead performs three
synchronous pageable DtoH copies, ~140 ms, of the raw scores/category_ids
and the full topk x H x W mask tensor).

Host side reproduces the exact numpy structures the reference path produces:

  scores       float32 (K,)
  category_ids int64   (K,)
  masks        uint8 {0,1} (K, H, W)

Masks are transferred column-major (packed along H), so each per-instance
2-D slice ``masks[k]`` is an F-contiguous view and
``coco_mask.encode(np.asfortranarray(mask))`` in the COCO export performs no
extra copy. Small arrays are copied out of the staging buffers; the mask
array is either freshly allocated by ``np.unpackbits`` (packed mode) or a
zero-copy view into a double-buffered pinned staging buffer (raw mode) --
i.e. the current result stays valid until two further calls overwrite the
ring slot.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import threading
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Cached side stream + pinned staging buffers (keyed by shape/dtype).
# ---------------------------------------------------------------------------

_SIDE_STREAM: Optional["torch.cuda.Stream"] = None
_PINNED: Dict[Tuple[Any, ...], torch.Tensor] = {}
_MASK_RING = 0  # double-buffer the (large) mask staging buffer
_MASK_RING_DEPTH = 2


def _side_stream() -> "torch.cuda.Stream":
    global _SIDE_STREAM
    if _SIDE_STREAM is None:
        _SIDE_STREAM = torch.cuda.Stream()
    return _SIDE_STREAM


def _pinned(key: str, shape: Tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    full_key = (key, tuple(shape), dtype)
    buf = _PINNED.get(full_key)
    if buf is None:
        buf = torch.empty(shape, dtype=dtype, pin_memory=True)
        _PINNED[full_key] = buf
    return buf


def release_pinned_buffers() -> None:
    """Drop cached pinned staging buffers (frees host memory)."""
    _PINNED.clear()


# ---------------------------------------------------------------------------
# GPU bbox extraction (exact replica of the numpy semantics downstream).
# ---------------------------------------------------------------------------

def compute_bbox_xyxy_gpu(binary_masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Half-open XYXY bboxes from binary mask nonzero extents, on GPU.

    Replicates ``coco_export._bbox_xyxy_from_binary_mask`` exactly:
    for ``ys, xs = np.where(mask > 0)`` -> [xs.min(), ys.min(), xs.max()+1,
    ys.max()+1]. Masks with no nonzero pixel get ``bbox_valid=False``
    (downstream export skips those instances, same as ``bbox is None``).

    Args:
        binary_masks: (B, K, H, W) bool tensor.

    Returns:
        bbox:  (B, K, 4) float32 (x1, y1, x2, y2); garbage where invalid.
        valid: (B, K) bool.
    """
    B, K, H, W = binary_masks.shape
    device = binary_masks.device
    rows_any = binary_masks.any(dim=3)  # (B, K, H)
    cols_any = binary_masks.any(dim=2)  # (B, K, W)
    valid = rows_any.any(dim=2)         # (B, K)

    row_idx = torch.arange(H, device=device, dtype=torch.int32).view(1, 1, H)
    col_idx = torch.arange(W, device=device, dtype=torch.int32).view(1, 1, W)

    rows_kept = row_idx.masked_fill(~rows_any, H)   # H where row empty
    rows_high = row_idx.masked_fill(~rows_any, -1)  # -1 where row empty
    y1 = rows_kept.amin(dim=2)
    y2 = rows_high.amax(dim=2) + 1

    cols_kept = col_idx.expand(B, K, W).masked_fill(~cols_any, W)
    cols_high = col_idx.expand(B, K, W).masked_fill(~cols_any, -1)
    x1 = cols_kept.amin(dim=2)
    x2 = cols_high.amax(dim=2) + 1

    bbox = torch.stack((x1, y1, x2, y2), dim=-1).to(torch.float32)
    return bbox, valid


# ---------------------------------------------------------------------------
# GPU bit-packing (torch equivalent of np.packbits along a chosen dim).
# ---------------------------------------------------------------------------

def pack_bits_gpu(u8: torch.Tensor) -> torch.Tensor:
    """Pack {0,1} uint8 masks to bits, big-endian, along the LAST dim.

    Args:
        u8: (..., L) uint8 tensor with values in {0, 1}.

    Returns:
        (..., ceil(L/8)) uint8 where each byte holds 8 consecutive elements,
        first element in the MSB (identical to ``np.packbits(..., axis=-1)``).
        The input last dim is zero-padded to a multiple of 8 first.
    """
    L = u8.shape[-1]
    if L % 8 != 0:
        pad = (L + 7) // 8 * 8 - L
        u8 = F.pad(u8, (0, pad))
    x = u8
    bits = 1
    # Pairwise digit combine: new = a * 2^bits + b (a = even elements, b = odd).
    # After each step values < 2^(2*bits); stops at 8 bits/byte (max 255).
    while bits < 8:
        a = x[..., 0::2]
        b = x[..., 1::2]
        x = a * (2 ** bits) + b
        bits *= 2
    return x.contiguous()


# ---------------------------------------------------------------------------
# GPU postprocess tensor math (shared by eager + CUDA-graph paths).
# ---------------------------------------------------------------------------

def postprocess_tensors(
    pred_logits: torch.Tensor,
    pred_masks: torch.Tensor,
    image_shape: Tuple[int, ...],
    inference_topk: int = 100,
) -> Dict[str, torch.Tensor]:
    """Device-side top-k / gather / upsample / sigmoid / threshold / score.

    Exact op sequence of ``MagFormerArch._inference_raw`` / ``_inference_raw_gpu``
    up to (and including) the final per-instance scores and binary masks, so
    eager and CUDA-graph replay produce bit-identical values. All shapes are
    input-shape-derived and data-independent (fixed top-k), hence capturable.

    Returns dict with:
      final_scores      (B, K) float32
      class_indices     (B, K) int64
      binary_masks_uint8 (B, K, H, W) uint8 {0,1}
      binary_masks      (B, K, H, W) bool (the > 0.5 masks)
    """
    B, Nq, _ = pred_logits.shape
    H_img, W_img = image_shape[-2:]

    class_scores = pred_logits.sigmoid()[..., :-1]
    num_classes = class_scores.shape[-1]

    topk = min(inference_topk, Nq * max(num_classes, 1))
    top_scores, top_indices = class_scores.flatten(1).topk(topk, dim=1)

    labels = torch.arange(num_classes, device=pred_logits.device).unsqueeze(
        0).repeat(Nq, 1).flatten(0, 1)

    # Vectorized batch processing
    query_indices = top_indices // max(num_classes, 1)
    class_indices = labels[top_indices] if num_classes > 0 else torch.zeros_like(
        query_indices)

    # Gather masks: (B, Nq, H, W) -> (B, topk, H, W)
    masks = pred_masks.gather(
        1,
        query_indices.unsqueeze(-1).unsqueeze(-1).expand(
            -1, -1, pred_masks.shape[-2], pred_masks.shape[-1])
    )

    # Batched interpolation
    if masks.shape[-2:] != (H_img, W_img):
        masks = F.interpolate(
            masks.reshape(B * topk, 1, masks.shape[-2], masks.shape[-1]),
            size=(H_img, W_img),
            mode="bilinear",
            align_corners=False,
        ).reshape(B, topk, H_img, W_img)

    # Batched sigmoid + threshold + scoring
    mask_probs = masks.sigmoid()
    binary_masks = mask_probs > 0.5
    mask_scores = (mask_probs.flatten(2) * binary_masks.float().flatten(2)).sum(2) / (
        binary_masks.float().flatten(2).sum(2) + 1e-6
    )
    final_scores = top_scores * mask_scores

    binary_masks_uint8 = binary_masks.to(torch.uint8)

    return {
        "final_scores": final_scores,
        "class_indices": class_indices,
        "binary_masks_uint8": binary_masks_uint8,
        "binary_masks": binary_masks,
    }


# ---------------------------------------------------------------------------
# One-shot compact async transfer + host unpack.
# ---------------------------------------------------------------------------

def export_batch_to_host(
    final_scores: torch.Tensor,
    class_indices: torch.Tensor,
    binary_masks_uint8: torch.Tensor,
    binary_masks: torch.Tensor,
    pack_masks: bool = False,
) -> List[Dict[str, Any]]:
    """GPU bbox + compact masks -> pinned async copy -> host numpy preds.

    Args:
        final_scores: (B, K) float32 CUDA (from the reference op sequence).
        class_indices: (B, K) int64 CUDA.
        binary_masks_uint8: (B, K, H, W) uint8 {0,1} CUDA.
        binary_masks: (B, K, H, W) bool CUDA (the > 0.5 masks, pre-uint8).
        pack_masks: if True, masks are transferred as packed bits
            (~13 MB for 100x1024x1024) and unpacked on host with
            np.unpackbits (~40 ms); if False (default), the uint8 {0,1}
            masks are transferred as-is in column-major layout (~105 MB,
            ~10 ms over pinned memory) and the host only takes a zero-copy
            transpose view. Both modes return identical values; the raw
            mode is faster end-to-end on PCIe Gen4.

    Returns:
        List of B dicts:
          image_id / scores / category_ids / masks  -- identical values to the
              reference CPU path (scores/category_ids dtypes and mask
              uint8 {0,1} included),
          bbox_xyxy -- (K, 4) float32 numpy, GPU-derived,
          bbox_valid -- (K,) bool numpy.
    """
    bbox, valid = compute_bbox_xyxy_gpu(binary_masks)
    return export_prepared_to_host(
        final_scores=final_scores,
        class_indices=class_indices,
        bbox=bbox,
        valid=valid,
        mask_payload=mask_transfer_payload(binary_masks_uint8, pack_masks),
        orig_hw=tuple(binary_masks_uint8.shape[-2:]),
        pack_masks=pack_masks,
    )


def mask_transfer_payload(
    binary_masks_uint8: torch.Tensor,
    pack_masks: bool = False,
) -> torch.Tensor:
    """Device-side layout transform of the binary masks for host transfer.

    Column-major (F-order per instance) raw uint8 by default, or bit-packed
    along H with :func:`pack_bits_gpu` when ``pack_masks``. Pure GPU ops with
    static shapes -- safe to run inside a CUDA graph.
    """
    if pack_masks:
        # Pack along H (column-major) so host unpack yields F-contiguous 2-D
        # per-instance views (no asfortranarray copy during RLE encoding).
        return pack_bits_gpu(binary_masks_uint8.transpose(-2, -1))
    # Column-major raw uint8: one GPU transpose-copy (fast, on-device) and a
    # single big async DtoH; the host-side result is a zero-copy transpose
    # view whose per-instance 2-D slices are F-contiguous.
    return binary_masks_uint8.transpose(-2, -1).contiguous()


def export_prepared_to_host(
    final_scores: torch.Tensor,
    class_indices: torch.Tensor,
    bbox: torch.Tensor,
    valid: torch.Tensor,
    mask_payload: torch.Tensor,
    orig_hw: Tuple[int, int],
    pack_masks: bool = False,
) -> List[Dict[str, Any]]:
    """Move an already-prepared payload to pinned host buffers (one sync).

    Companion of :func:`mask_transfer_payload` / ``compute_bbox_xyxy_gpu``:
    performs only the pinned async DtoH copies + host numpy assembly, so the
    device-side parts can execute inside a CUDA graph while this part (event
    synchronize + numpy views) stays outside.
    """
    B, K = final_scores.shape
    H, W = orig_hw
    mask_shape = tuple(mask_payload.shape)

    p_scores = _pinned("scores", (B, K), torch.float32)
    p_cats = _pinned("cats", (B, K), torch.int64)
    p_bbox = _pinned("bbox", (B, K, 4), torch.float32)
    p_valid = _pinned("valid", (B, K), torch.uint8)
    # Double-buffer the big mask staging buffer so the previous call's
    # zero-copy mask views stay valid while the next batch is in flight.
    global _MASK_RING
    ring_key = f"masks/{_MASK_RING}"
    _MASK_RING = (_MASK_RING + 1) % _MASK_RING_DEPTH
    p_masks = _pinned(ring_key, mask_shape, torch.uint8)

    device = final_scores.device
    current = torch.cuda.current_stream(device)
    side = _side_stream()
    side.wait_stream(current)
    with torch.cuda.stream(side):
        p_scores.copy_(final_scores, non_blocking=True)
        p_cats.copy_(class_indices, non_blocking=True)
        p_bbox.copy_(bbox, non_blocking=True)
        p_valid.copy_(valid.to(torch.uint8), non_blocking=True)
        p_masks.copy_(mask_payload, non_blocking=True)  # the single big async copy
    done = torch.cuda.Event()
    done.record(side)
    done.synchronize()  # one host sync for the whole payload

    scores_np = p_scores.numpy()
    cats_np = p_cats.numpy()
    bbox_np = p_bbox.numpy()
    valid_np = p_valid.numpy().astype(bool)
    if pack_masks:
        # np.unpackbits allocates fresh memory (never aliases the pinned
        # buffer) and returns values in {0, 1} -- identical to the reference.
        masks_np = np.unpackbits(p_masks.numpy(), axis=-1, count=H).transpose(0, 1, 3, 2)
    else:
        # Zero-copy view: (B,K,W,H) C-order -> (B,K,H,W) with per-instance
        # F-contiguous slices; values are the verbatim uint8 {0,1} masks.
        # Views a double-buffered pinned ring slot (see docstring).
        masks_np = p_masks.numpy().transpose(0, 1, 3, 2)

    predictions: List[Dict[str, Any]] = []
    for i in range(B):
        predictions.append(
            {
                "image_id": i,
                "scores": np.array(scores_np[i], dtype=np.float32),
                "category_ids": np.array(cats_np[i], dtype=np.int64),
                "masks": masks_np[i],
                "bbox_xyxy": np.array(bbox_np[i], dtype=np.float32),
                "bbox_valid": np.array(valid_np[i], dtype=bool),
            }
        )
    return predictions


# ---------------------------------------------------------------------------
# Deferred (no host sync) export: ring of pinned slots consumed off-thread.
# ---------------------------------------------------------------------------


class _ExportSlot:
    """One pinned staging slot: small payload arrays + packed mask payload."""

    def __init__(self, key: Tuple[Any, ...]):
        B, K, H, W, packed_last = key
        self.key = key
        self.scores = torch.empty((B, K), dtype=torch.float32, pin_memory=True)
        self.cats = torch.empty((B, K), dtype=torch.int64, pin_memory=True)
        self.bbox = torch.empty((B, K, 4), dtype=torch.float32, pin_memory=True)
        self.valid = torch.empty((B, K), dtype=torch.uint8, pin_memory=True)
        self.packed = torch.empty((B, K, W, packed_last), dtype=torch.uint8, pin_memory=True)
        self.event = torch.cuda.Event()


class DeferredExportRing:
    """Bounded ring of pinned export slots with blocking acquire.

    ``acquire`` blocks when every slot for the requested shape is still in
    flight, which provides natural backpressure: the number of in-flight
    (GPU-copied, un-consumed) images is bounded by ``depth``. Slots are
    released by the consumer side only after the payload has been copied out
    of pinned memory, so no aliasing hazard exists.
    """

    def __init__(self, depth: int = 8):
        if int(depth) < 1:
            raise ValueError(f"DeferredExportRing depth must be >= 1, got {depth}")
        self.depth = int(depth)
        self._cond = threading.Condition()
        self._free: Dict[Tuple[Any, ...], List[_ExportSlot]] = {}
        self._allocated: Dict[Tuple[Any, ...], int] = {}

    def acquire(self, key: Tuple[Any, ...]) -> _ExportSlot:
        key = tuple(key)
        with self._cond:
            while True:
                free_list = self._free.setdefault(key, [])
                if free_list:
                    return free_list.pop()
                if self._allocated.get(key, 0) < self.depth:
                    self._allocated[key] = self._allocated.get(key, 0) + 1
                    # Allocated while holding the lock; pinned allocation is
                    # done on the caller's (main) thread only.
                    slot = _ExportSlot(key)
                    return slot
                self._cond.wait()

    def release(self, slot: _ExportSlot) -> None:
        with self._cond:
            self._free[slot.key].append(slot)
            self._cond.notify()


def export_batch_deferred(
    final_scores: torch.Tensor,
    class_indices: torch.Tensor,
    binary_masks_uint8: torch.Tensor,
    binary_masks: torch.Tensor,
    ring: DeferredExportRing,
) -> _ExportSlot:
    """GPU bbox + bit-packed masks -> async copies into a pinned ring slot.

    Same GPU-side computation as ``export_batch_to_host(..., pack_masks=True)``
    (bit-identical values), but WITHOUT the trailing host event sync: the
    returned slot's CUDA event is handed to a consumer thread that performs
    ``event.synchronize()`` and copies the payload out of pinned memory while
    the main thread immediately continues with the next forward pass.

    ``record_stream`` is called on every source tensor used by the side-stream
    copies so the caching allocator cannot hand that device memory to later
    allocations before the copies complete.

    Returns:
        slot (``_ExportSlot``) with pinned buffers ``scores``/``cats``/``bbox``/
        ``valid``/``packed`` and completion ``event``. ``packed`` holds the
        (B, K, W, ceil(H/8)) big-endian bit payload consumed by
        ``magformer.engine.coco_export.unpack_packed_masks``.
    """
    B, K, H, W = binary_masks_uint8.shape
    bbox, valid = compute_bbox_xyxy_gpu(binary_masks)
    packed = pack_bits_gpu(binary_masks_uint8.transpose(-2, -1).contiguous())
    valid_u8 = valid.to(torch.uint8)

    key = (B, K, H, W, packed.shape[-1])
    slot = ring.acquire(key)

    device = final_scores.device
    current = torch.cuda.current_stream(device)
    side = _side_stream()
    side.wait_stream(current)
    with torch.cuda.stream(side):
        slot.scores.copy_(final_scores, non_blocking=True)
        slot.cats.copy_(class_indices, non_blocking=True)
        slot.bbox.copy_(bbox, non_blocking=True)
        slot.valid.copy_(valid_u8, non_blocking=True)
        slot.packed.copy_(packed, non_blocking=True)
    slot.event.record(side)

    # The tensors above are freed (refs dropped) as soon as this function
    # returns; mark them as used on the side stream so the allocator defers
    # reuse until the async copies complete.
    for t in (final_scores, class_indices, bbox, valid_u8, packed):
        t.record_stream(side)
    return slot
