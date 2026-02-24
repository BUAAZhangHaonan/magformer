from __future__ import annotations

import pickle
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch


def test_convert_mask2former_ckpt_1class_detectron2_reduces_heads(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "analysis" / "convert_mask2former_ckpt_1class_detectron2.py"
    assert script.exists()

    ckpt_pkl = tmp_path / "model_final_fake.pkl"
    out_pth = tmp_path / "model_final_1class.pth"

    # Minimal fake checkpoint dict.
    w = np.random.randn(81, 256).astype("float32")
    b = np.random.randn(81).astype("float32")
    ew = np.random.randn(81).astype("float32")
    sd = OrderedDict(
        {
            "sem_seg_head.predictor.class_embed.weight": w,
            "sem_seg_head.predictor.class_embed.bias": b,
            "criterion.empty_weight": ew,
        }
    )
    # Simulate detectron2 state_dict metadata so conversion must preserve it.
    sd._metadata = OrderedDict({"": {"version": 2}})
    fake = {
        "model": sd,
    }
    ckpt_pkl.write_bytes(pickle.dumps(fake))

    subprocess.run(
        [
            sys.executable,
            str(script),
            "--input",
            str(ckpt_pkl),
            "--output",
            str(out_pth),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = torch.load(out_pth, map_location="cpu")
    state = payload["model"]
    assert tuple(state["sem_seg_head.predictor.class_embed.weight"].shape) == (2, 256)
    assert tuple(state["sem_seg_head.predictor.class_embed.bias"].shape) == (2,)
    assert tuple(state["criterion.empty_weight"].shape) == (2,)
    assert hasattr(state, "_metadata")
    assert "" in state._metadata


def test_convert_mask2former_ckpt_1class_detectron2_can_rename_prefixes(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "analysis" / "convert_mask2former_ckpt_1class_detectron2.py"
    assert script.exists()

    ckpt_pkl = tmp_path / "model_final_fake.pkl"
    out_pth = tmp_path / "model_final_1class_mgm.pth"

    sd = OrderedDict(
        {
            "backbone.foo": np.random.randn(3, 3).astype("float32"),
            "sem_seg_head.predictor.class_embed.weight": np.random.randn(81, 256).astype("float32"),
        }
    )
    sd._metadata = OrderedDict({"": {"version": 2}})
    ckpt_pkl.write_bytes(pickle.dumps({"model": sd}))

    subprocess.run(
        [
            sys.executable,
            str(script),
            "--input",
            str(ckpt_pkl),
            "--output",
            str(out_pth),
            "--rename-prefix",
            "backbone=rgb_backbone",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = torch.load(out_pth, map_location="cpu")
    state = payload["model"]
    assert "rgb_backbone.foo" in state
    assert "backbone.foo" not in state
    assert tuple(state["sem_seg_head.predictor.class_embed.weight"].shape) == (2, 256)
    assert hasattr(state, "_metadata")
