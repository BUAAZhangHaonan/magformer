from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
MASK2FORMER_ROOT = REPO_ROOT / "baselines" / "Mask2Former"
if str(MASK2FORMER_ROOT) not in sys.path:
    sys.path.insert(0, str(MASK2FORMER_ROOT))


def _make_model():
    import detectron2

    if not hasattr(detectron2, "_C"):
        # The unit test only needs MaskFormer.instance_inference. The magformer
        # test env may not have Detectron2 C++ eval ops compiled.
        detectron2._C = SimpleNamespace()

    from mask2former.maskformer_model import MaskFormer

    model = MaskFormer.__new__(MaskFormer)
    nn.Module.__init__(model)
    model.sem_seg_head = SimpleNamespace(num_classes=1)
    model.num_queries = 2
    model.test_topk_per_image = 2
    model.panoptic_on = False
    model.metadata = SimpleNamespace(thing_dataset_id_to_contiguous_id={1: 0})
    model.register_buffer("pixel_mean", torch.zeros(1), persistent=False)
    return model


def test_instance_inference_derives_pred_boxes_from_binary_masks() -> None:
    model = _make_model()
    mask_cls = torch.tensor([[5.0, -5.0], [4.0, -4.0]])
    mask_pred = torch.full((2, 6, 7), -10.0)
    mask_pred[0, 1:4, 2:5] = 10.0
    mask_pred[1, 4:6, 0:2] = 10.0

    result = model.instance_inference(mask_cls, mask_pred)

    assert result.pred_boxes.tensor.tolist() == [[2.0, 1.0, 5.0, 4.0], [0.0, 4.0, 2.0, 6.0]]


def test_instance_inference_empty_binary_mask_has_zero_box() -> None:
    model = _make_model()
    mask_cls = torch.tensor([[5.0, -5.0], [4.0, -4.0]])
    mask_pred = torch.full((2, 6, 7), -10.0)
    mask_pred[0, 1:3, 2:4] = 10.0

    result = model.instance_inference(mask_cls, mask_pred)

    assert result.pred_boxes.tensor.tolist() == [[2.0, 1.0, 4.0, 3.0], [0.0, 0.0, 0.0, 0.0]]
