"""Copy-paste depth-channel tests (P2 bugfix, 2026-09-21).

The instance bank historically stored RGB crops only and the paste path
never wrote ``result["depth"]`` — enabling copy-paste on this RGB-D model
would have pasted RGB/mask evidence of objects the depth map contradicts.
These tests pin the fixed contract: depth-aware deposit/paste, invalid
(==0) source depth preserving target depth, mixed-bank consistency guard,
and bank state_dict v2 round-trip + v1 compatibility.
"""

import random

import numpy as np
import torch

from magformer.data.dataset import _InstanceBank
from magformer.data.transforms import CopyPasteTransform


def _synthetic_sample(h=96, w=96, seed=0):
    """One source image with one 8x8-ish instance whose depth has holes."""
    rng = np.random.RandomState(seed)
    image = rng.randint(0, 255, size=(h, w, 3)).astype(np.uint8)
    mask = np.zeros((h, w), dtype=bool)
    mask[20:28, 30:38] = True
    depth = rng.uniform(0.2, 0.5, size=(h, w)).astype(np.float32)
    depth[mask] = 0.5
    depth[22, 32] = 0.0  # hole: invalid source pixel inside the instance
    depth[25, 35] = 0.0  # second hole
    masks = mask[:, :, None]
    boxes = np.array([[30.0, 20.0, 38.0, 28.0]], dtype=np.float32)
    labels = np.zeros(1, dtype=np.int64)
    return image, masks, boxes, labels, depth, mask


def _empty_target(h=128, w=128, seed=1):
    rng = np.random.RandomState(seed)
    return {
        "image": rng.randint(0, 255, size=(h, w, 3)).astype(np.uint8),
        "depth": rng.uniform(0.1, 0.3, size=(h, w)).astype(np.float32),
        "masks": np.zeros((h, w, 0), dtype=bool),
        "boxes": np.zeros((0, 4), dtype=np.float32),
        "labels": np.zeros((0,), dtype=np.int64),
    }


def _deposit_bank(with_depth=True):
    bank = _InstanceBank(capacity=10, small_threshold=1024)
    image, masks, boxes, labels, depth, _ = _synthetic_sample()
    bank.deposit(
        image=image, masks=masks, boxes=boxes, labels=labels,
        depth=depth if with_depth else None,
    )
    assert len(bank) > 0
    return bank


def _transform(bank, **kw):
    params = dict(prob=1.0, max_paste_instances=1, min_instance_area=4,
                  max_instance_area_ratio=0.3, prefer_small=True,
                  scale_jitter=(1.0, 1.0), iou_threshold=0.7)
    params.update(kw)
    return CopyPasteTransform(instance_bank=bank, **params)


def test_paste_writes_depth_consistently():
    bank = _deposit_bank(with_depth=True)
    result = _empty_target()
    depth_before = result["depth"].copy()
    random.seed(0)
    out = _transform(bank)(result)
    n_new = out["masks"].shape[2]
    assert n_new > 0, "one instance should be pasted onto the empty canvas"
    pasted_union = out["masks"].any(axis=2)
    changed = out["depth"] != depth_before
    # (a) depth changed inside the pasted region only
    assert changed.sum() > 0
    assert (~changed | pasted_union).all(), (
        "depth must not change outside the pasted instance mask")
    # (b) every changed pixel got a valid source value (0.5 inside instance)
    assert np.allclose(out["depth"][changed], 0.5)
    # (c) invalid source pixels (holes) kept the target depth
    assert (pasted_union & ~changed).any(), (
        "hole pixels inside the pasted mask must preserve target depth")
    assert np.allclose(
        out["depth"][pasted_union & ~changed], depth_before[pasted_union & ~changed])


def test_depth_result_rejects_depthless_bank_entries():
    bank = _deposit_bank(with_depth=False)  # legacy RGB-only entries
    result = _empty_target()
    random.seed(0)
    out = _transform(bank)(result)
    assert out["masks"].shape[2] == 0, (
        "RGB-only bank entries must be skipped when the sample has depth "
        "(would paste RGB/mask evidence the depth map contradicts)")


def test_rgb_only_pipeline_still_pastes():
    bank = _deposit_bank(with_depth=False)
    result = _empty_target()
    del result["depth"]
    random.seed(0)
    out = _transform(bank)(result)
    assert out["masks"].shape[2] > 0, (
        "RGB-only samples must keep pasting from legacy depth-less entries")


def test_bank_state_dict_roundtrip_v2():
    bank = _deposit_bank(with_depth=True)
    state = bank.state_dict()
    assert state["version"] == 2
    restored = _InstanceBank(capacity=10, small_threshold=1024)
    restored.load_state_dict(state)
    entry = restored.sample(1, prefer_small=True)[0]
    assert entry["crop_depth"] is not None
    assert entry["crop_depth"].shape == entry["crop_mask"].shape


def test_bank_state_dict_v1_backcompat():
    state = _deposit_bank(with_depth=True).state_dict()
    for entry in state["small_bank"] + state["all_bank"]:
        entry.pop("crop_depth", None)
    state["version"] = 1
    restored = _InstanceBank(capacity=10, small_threshold=1024)
    restored.load_state_dict(state)
    entry = restored.sample(1, prefer_small=True)[0]
    assert entry["crop_depth"] is None, (
        "v1 (pre-depth) states must load with crop_depth=None")


def test_deposit_ignores_mismatched_depth():
    bank = _InstanceBank(capacity=10)
    image, masks, boxes, labels, depth, _ = _synthetic_sample()
    bank.deposit(image=image, masks=masks, boxes=boxes, labels=labels,
                 depth=depth[:4, :4])  # wrong shape: must not be stored
    entry = bank.sample(1, prefer_small=True)[0]
    assert entry["crop_depth"] is None
