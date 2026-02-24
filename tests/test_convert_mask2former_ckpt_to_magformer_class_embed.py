from __future__ import annotations

import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


def test_convert_mask2former_ckpt_maps_class_embed_to_1class(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "analysis" / "convert_mask2former_ckpt_to_magformer.py"
    assert script.exists()

    ckpt_pkl = tmp_path / "model_final_fake.pkl"
    out_pth = tmp_path / "magformer_ws.pth"

    # Fake COCO-style class head: 80 classes + 1 no-object = 81.
    w = np.random.randn(81, 256).astype("float32")
    b = np.random.randn(81).astype("float32")
    fake = {"model": {"sem_seg_head.predictor.class_embed.weight": w, "sem_seg_head.predictor.class_embed.bias": b}}
    ckpt_pkl.write_bytes(pickle.dumps(fake))

    magformer_cfg = repo_root / "configs" / "magformer_0909_512_20ep_trackp.yaml"
    assert magformer_cfg.exists()

    subprocess.run(
        [
            sys.executable,
            str(script),
            "--input",
            str(ckpt_pkl),
            "--output",
            str(out_pth),
            "--magformer-config",
            str(magformer_cfg),
            "--dataset-root",
            "/tmp/ecc0909",
            "--include-class-embed",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = torch.load(out_pth, map_location="cpu")
    state = payload.get("model_state_dict", {})

    assert "decoder.class_embed.weight" in state
    assert "decoder.class_embed.bias" in state
    assert tuple(state["decoder.class_embed.weight"].shape) == (2, 256)
    assert tuple(state["decoder.class_embed.bias"].shape) == (2,)
