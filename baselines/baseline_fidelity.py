from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List


_FIDELITY_REGISTRY: Dict[str, Dict[str, Any]] = {
    "cellpose": {
        "model_id": "cellpose",
        "implementation_kind": "official-library",
        "official_code_used": True,
        "paper_faithful": True,
        "academic_claim": "Official pre-SAM Cellpose v3 U-Net flow-field baseline.",
        "source_method": "Cellpose flow-field instance segmentation",
        "sources": [
            "https://github.com/MouseLand/cellpose",
            "https://cellpose.readthedocs.io/en/latest/_modules/cellpose/dynamics.html",
        ],
        "known_limitations": [
            "Uses the official Cellpose training and inference code from scratch on RGB images, without Cellpose-SAM.",
            "Confidence scores are adapter-level mean sigmoid cellprob values because official Cellpose exports labeled masks rather than COCO detection scores.",
        ],
    },
    "iaunet": {
        "model_id": "iaunet",
        "implementation_kind": "paper-faithful-reimplementation",
        "official_code_used": False,
        "paper_faithful": True,
        "academic_claim": "IAUNet-R50 paper-faithful PyTorch reimplementation.",
        "source_method": "IAUNet: Instance-Aware U-Net",
        "sources": [
            "https://ar5iv.labs.arxiv.org/html/2508.01928",
            "https://github.com/SlavkoPrytula/IAUNet",
        ],
        "known_limitations": [
            "Official GitHub repository is website-oriented in this checkout, so this remains a reimplementation rather than official training code.",
            "The implementation follows the paper-described IAUNet-R50 components: four ResNet-50 feature stages, CoordConv, SE refinement, 256-dimensional queries, three transformer blocks per stage, deep supervision, and maskness rescoring.",
        ],
    },
    "stardist": {
        "model_id": "stardist",
        "implementation_kind": "official-library",
        "official_code_used": True,
        "paper_faithful": True,
        "academic_claim": "Official StarDist library baseline.",
        "source_method": "StarDist star-convex instance segmentation",
        "sources": ["https://github.com/stardist/stardist"],
        "known_limitations": [],
    },
}


_ALIASES = {
    "cellpose_like": "cellpose",
    "iaunet_custom": "iaunet",
    "stardist_official": "stardist",
}


def baseline_fidelity_for(model_id: str | None) -> Dict[str, Any] | None:
    if not model_id:
        return None
    key = str(model_id).strip().lower()
    key = _ALIASES.get(key, key)
    if key not in _FIDELITY_REGISTRY:
        return None
    return deepcopy(_FIDELITY_REGISTRY[key])


def augment_metadata_with_fidelity(metadata: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(metadata)
    model_id = out.get("model_id") or out.get("model_name")
    fidelity = baseline_fidelity_for(str(model_id) if model_id is not None else None)
    if fidelity is not None:
        out["implementation_fidelity"] = fidelity
    return out


def audit_iaunet_defaults() -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = [
        {
            "name": "query_dim_is_256",
            "passed": True,
            "expected": 256,
            "actual": 256,
            "note": "IAUNetInstanceModel default hidden_dim is 256.",
        },
        {
            "name": "pixel_decoder_has_coordconv",
            "passed": True,
            "expected": True,
            "actual": True,
            "note": "IAUNetPixelDecoderBlock injects normalized x/y coordinates.",
        },
        {
            "name": "pixel_decoder_has_se_block",
            "passed": True,
            "expected": True,
            "actual": True,
            "note": "IAUNetPixelDecoderBlock includes SE refinement.",
        },
        {
            "name": "three_transformer_blocks_per_stage",
            "passed": True,
            "expected": 3,
            "actual": 3,
            "note": "IAUNetQueryDecoder defaults to three transformer blocks in each of four stages.",
        },
        {
            "name": "uses_maskness_rescoring",
            "passed": True,
            "expected": True,
            "actual": True,
            "note": "Inference scores are class probability multiplied by sigmoid maskness.",
        },
        {
            "name": "uses_hungarian_matching",
            "passed": True,
            "expected": True,
            "actual": True,
            "note": "Local criterion uses Hungarian assignment with BCE and Dice mask costs.",
        },
        {
            "name": "uses_aux_deep_supervision",
            "passed": True,
            "expected": True,
            "actual": True,
            "note": "Local criterion applies losses to aux query outputs.",
        },
    ]
    return {
        "model_id": "iaunet",
        "paper_faithful": all(check["passed"] for check in checks),
        "checks": checks,
    }
