# -*- coding: utf-8 -*-
"""Manual CUDA Graph capture of the fixed-shape MAGFormer inference path.

Captures, in ONE graph, the whole per-image inference region that the eval
loop runs at bs=1 / fixed 1024^2 inputs::

    forward_inference_decoder_outputs(images, depths, padding_masks, ...)
        -> dual backbones (Swin-T RGB + MBv3-L depth, parallel CUDA streams)
        -> DCCG fusion -> AGPE -> MSDeformAttn pixel decoder
        -> 8-head transformer decoder
    + GPU postprocess (gpu_postprocess.postprocess_tensors):
        sigmoid/top-k/gather/bilinear-upsample/threshold/mask-score fusion
    + export prep (compute_bbox_xyxy_gpu + column-major mask payload)

Only the pinned DtoH export (``export_prepared_to_host``: async copies +
event synchronize + host numpy assembly) stays outside the graph -- host
synchronization cannot be captured.

CUDA graphs replay the exact same kernel DAG with the exact same math on the
static input buffers, so outputs are bit-identical to the eager path. All
host-side guards that read device scalars (validation ``.any()`` checks, the
DCCG temperature ``.item()``, the MSDA shape assert) are skipped/replaced
with value-identical branch-free equivalents while
``torch.cuda.is_current_stream_capturing()`` (see the touched modules); the
warmup pass still executes the original guards on real data.

Usage (through ``MagFormerArch.forward_inference_graphed``)::

    runner = None
    for batch in loader:
        outputs, runner = run_graphed_inference(
            model, batch_images, batch_depths, padding_masks=...,
            depth_valid_masks=..., depth_noise_masks=...,
            runner=runner, gpu_export=True)
"""

from typing import Any, Dict, Optional, Tuple

import torch

from .gpu_postprocess import (
    compute_bbox_xyxy_gpu,
    export_prepared_to_host,
    mask_transfer_payload,
    postprocess_tensors,
)


class InferenceGraphRunner:
    """Owns the static input/output buffers and the captured graph.

    The first :meth:`run` call captures the graph with the caller's actual
    first-image data in the static buffers (warmup runs the original eager
    code including all validation guards); every later call just copies the
    new inputs into the static buffers, replays the graph, and exports the
    static outputs to host.
    """

    def __init__(self, model, warmup_iters: int = 3):
        self.model = model
        self.warmup_iters = warmup_iters
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self._static: Dict[str, Optional[torch.Tensor]] = {}
        self._static_outputs: Optional[Dict[str, torch.Tensor]] = None
        self._input_spec: Optional[Dict[str, Tuple[Tuple[int, ...], torch.dtype]]] = {}
        # set at capture time:
        self._pack_masks = False
        self._capture_ms = None

    # ------------------------------------------------------------------
    # captured region
    # ------------------------------------------------------------------
    def _graph_body(self) -> Dict[str, torch.Tensor]:
        static = self._static
        outputs = self.model.forward_inference_decoder_outputs(
            images=static["images"],
            depths=static["depths"],
            padding_masks=static.get("padding_masks"),
            depth_valid_masks=static.get("depth_valid_masks"),
            depth_noise_masks=static.get("depth_noise_masks"),
        )
        tensors = postprocess_tensors(
            pred_logits=outputs["pred_logits"],
            pred_masks=outputs["pred_masks"],
            image_shape=static["images"].shape,
            inference_topk=self.model.inference_topk,
        )
        bbox, valid = compute_bbox_xyxy_gpu(tensors["binary_masks"])
        mask_payload = mask_transfer_payload(
            tensors["binary_masks_uint8"], self._pack_masks
        )
        return {
            "final_scores": tensors["final_scores"],
            "class_indices": tensors["class_indices"],
            "bbox": bbox,
            "valid": valid,
            "mask_payload": mask_payload,
        }

    # ------------------------------------------------------------------
    # input handling
    # ------------------------------------------------------------------
    @staticmethod
    def _spec_of(t: Optional[torch.Tensor]):
        return None if t is None else (tuple(t.shape), t.dtype, str(t.device), t.layout)

    def _record_spec(self, **inputs) -> None:
        self._input_spec = {k: self._spec_of(v) for k, v in inputs.items()}

    def _check_spec(self, **inputs) -> None:
        for key, value in inputs.items():
            expected = self._input_spec.get(key)
            got = self._spec_of(value)
            if expected != got:
                raise ValueError(
                    f"CUDA-graph input '{key}' changed after capture: "
                    f"captured={expected}, got={got}. The graph is static-"
                    "shape; rebuild it for new shapes/dtypes/devices."
                )

    def _copy_inputs(self, **inputs) -> None:
        for key, value in inputs.items():
            if value is None:
                continue
            self._static[key].copy_(value, non_blocking=True)

    # ------------------------------------------------------------------
    # capture
    # ------------------------------------------------------------------
    def _capture(self, **inputs) -> None:
        import time

        device = inputs["images"].device
        # Static input buffers (same shapes/dtypes as the first real batch).
        self._static = {
            k: (None if v is None else torch.empty_like(v, device=device))
            for k, v in inputs.items()
        }
        self._record_spec(**inputs)
        self._copy_inputs(**inputs)

        # Warmup on a side stream (standard recipe): exercises cudnn/MSDA
        # lazy init and the allocator so capture-time allocations are stable.
        # Runs the ORIGINAL eager path (guards included) on real data.
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            for _ in range(self.warmup_iters):
                self._graph_body()
        torch.cuda.current_stream(device).wait_stream(stream)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        t0 = time.perf_counter()
        # capture_error_mode="thread_local": the eval harness runs a DataLoader
        # pin-memory thread whose cudaHostAlloc calls are unrelated to the
        # captured work but are rejected as "unsafe" by the default "global"
        # capture mode. thread_local confines capture to this thread only.
        with torch.cuda.graph(graph, capture_error_mode="thread_local"):
            self._static_outputs = self._graph_body()
        self._capture_ms = (time.perf_counter() - t0) * 1e3
        self.graph = graph
        # The capture pass only RECORDS kernels; outputs are undefined.
        # Replay once so the static outputs hold this image's results.
        graph.replay()

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def is_captured(self) -> bool:
        return self.graph is not None

    @torch.inference_mode()
    def run(
        self,
        images: torch.Tensor,
        depths: torch.Tensor,
        padding_masks: Optional[torch.Tensor] = None,
        depth_valid_masks: Optional[torch.Tensor] = None,
        depth_noise_masks: Optional[torch.Tensor] = None,
        pack_masks: bool = False,
    ) -> Dict[str, Any]:
        """Replay (capturing on first call) and export predictions to host.

        Returns ``{"predictions": [...]}`` in the exact structure of
        ``MagFormerArch._inference_raw_gpu`` (scores/category_ids/masks numpy
        + GPU-derived bbox_xyxy/bbox_valid).
        """
        inputs = dict(
            images=images,
            depths=depths,
            padding_masks=padding_masks,
            depth_valid_masks=depth_valid_masks,
            depth_noise_masks=depth_noise_masks,
        )
        if self.graph is None:
            self._pack_masks = pack_masks
            self._capture(**inputs)
        else:
            if pack_masks != self._pack_masks:
                raise ValueError(
                    "pack_masks changed after CUDA-graph capture "
                    f"({self._pack_masks} -> {pack_masks})"
                )
            self._check_spec(**inputs)
            self._copy_inputs(**inputs)
            self.graph.replay()

        out = self._static_outputs
        predictions = export_prepared_to_host(
            final_scores=out["final_scores"],
            class_indices=out["class_indices"],
            bbox=out["bbox"],
            valid=out["valid"],
            mask_payload=out["mask_payload"],
            orig_hw=tuple(self._static["images"].shape[-2:]),
            pack_masks=self._pack_masks,
        )
        return {"predictions": predictions}


def run_graphed_inference(
    model,
    images: torch.Tensor,
    depths: torch.Tensor,
    padding_masks: Optional[torch.Tensor] = None,
    depth_valid_masks: Optional[torch.Tensor] = None,
    depth_noise_masks: Optional[torch.Tensor] = None,
    runner: Optional[InferenceGraphRunner] = None,
    warmup_iters: int = 3,
    pack_masks: bool = False,
) -> Tuple[Dict[str, Any], InferenceGraphRunner]:
    """Convenience wrapper creating/reusing an :class:`InferenceGraphRunner`."""
    if runner is None:
        runner = InferenceGraphRunner(model, warmup_iters=warmup_iters)
    outputs = runner.run(
        images=images,
        depths=depths,
        padding_masks=padding_masks,
        depth_valid_masks=depth_valid_masks,
        depth_noise_masks=depth_noise_masks,
        pack_masks=pack_masks,
    )
    return outputs, runner
