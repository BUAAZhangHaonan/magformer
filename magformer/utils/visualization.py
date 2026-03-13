from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np


_PALETTE: List[Tuple[int, int, int]] = [
    (57, 197, 187),
    (253, 121, 168),
    (116, 185, 255),
    (85, 239, 196),
    (255, 234, 167),
    (162, 155, 254),
    (232, 67, 147),
    (0, 184, 148),
]


def _as_numpy_mask(mask: Any) -> np.ndarray:
    arr = np.asarray(mask)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {arr.shape}")
    if arr.dtype == np.bool_:
        return arr.astype(np.uint8)
    if np.issubdtype(arr.dtype, np.floating):
        return (arr >= 0.5).astype(np.uint8)
    return (arr > 0).astype(np.uint8)


def _color_for_index(index: int) -> Tuple[int, int, int]:
    return _PALETTE[index % len(_PALETTE)]


def draw_yolov8_mask(
    image: np.ndarray,
    mask: Any,
    *,
    color: Tuple[int, int, int],
    alpha: float = 0.3,
) -> np.ndarray:
    out = image.copy()
    mask_u8 = _as_numpy_mask(mask).astype(bool)
    if not np.any(mask_u8):
        return out
    color_arr = np.asarray(color, dtype=np.float32)
    blended = np.round((1.0 - float(alpha)) *
                       out[mask_u8] + float(alpha) * color_arr).astype(np.uint8)
    out[mask_u8] = blended
    return out


def draw_yolov8_contour(
    image: np.ndarray,
    mask: Any,
    *,
    color: Tuple[int, int, int],
    thickness: int = 1,
) -> np.ndarray:
    out = image.copy()
    mask_u8 = (_as_numpy_mask(mask) * 255).astype(np.uint8)
    contours, _ = cv2.findContours(
        mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(out, contours, -1, color, int(thickness))
    return out


def prediction_to_lists(prediction: Dict[str, Any]) -> Tuple[List[np.ndarray], List[float], List[int]]:
    masks_raw = prediction.get("masks", [])
    if isinstance(masks_raw, np.ndarray) and masks_raw.ndim == 2:
        masks = [_as_numpy_mask(masks_raw)]
    else:
        masks = [_as_numpy_mask(mask) for mask in list(masks_raw)]

    scores = [float(x) for x in list(prediction.get("scores", []))]
    labels_raw = (
        prediction.get("labels")
        or prediction.get("category_ids")
        or prediction.get("class_ids")
        or []
    )
    labels = [int(x) for x in list(labels_raw)]

    while len(scores) < len(masks):
        scores.append(0.0)
    while len(labels) < len(masks):
        labels.append(0)

    return masks, scores[: len(masks)], labels[: len(masks)]


def _draw_label(
    image: np.ndarray,
    *,
    text: str,
    mask: np.ndarray,
    color: Tuple[int, int, int],
) -> np.ndarray:
    out = image.copy()
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return out
    x, y = int(xs.min()), int(ys.min())
    cv2.putText(
        out,
        text,
        (x, max(y - 4, 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        color,
        1,
        cv2.LINE_AA,
    )
    return out


def visualize_predictions(
    *,
    image: np.ndarray,
    masks: Sequence[Any],
    scores: Sequence[float],
    labels: Sequence[int],
    class_names: Sequence[str] | None = None,
    score_threshold: float = 0.5,
    alpha: float = 0.3,
    show_labels: bool = False,
    show_contours: bool = True,
    contour_thickness: int = 1,
    show_masks: bool = True,
    output_path: str | None = None,
) -> np.ndarray:
    out = image.copy()
    class_names = list(class_names or [])
    for idx, (mask, score, label) in enumerate(zip(masks, scores, labels)):
        if float(score) < float(score_threshold):
            continue
        mask_u8 = _as_numpy_mask(mask)
        color = _color_for_index(idx)
        if show_masks:
            out = draw_yolov8_mask(out, mask_u8, color=color, alpha=alpha)
        if show_contours:
            out = draw_yolov8_contour(
                out, mask_u8, color=color, thickness=contour_thickness)
        if show_labels:
            label_name = class_names[label] if 0 <= int(
                label) < len(class_names) else str(int(label))
            out = _draw_label(
                out, text=f"{label_name}:{float(score):.2f}", mask=mask_u8, color=color)

    if output_path:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    return out


def _render_gt_panel(
    image: np.ndarray,
    masks: Iterable[Any],
    *,
    alpha: float,
) -> np.ndarray:
    out = image.copy()
    for idx, mask in enumerate(masks):
        color = _color_for_index(idx)
        out = draw_yolov8_mask(out, mask, color=color, alpha=alpha)
        out = draw_yolov8_contour(out, mask, color=color, thickness=1)
    return out


def render_triptych_comparison(
    *,
    image: np.ndarray,
    gt_masks: Sequence[Any],
    magformer_prediction: Dict[str, Any],
    mask2former_prediction: Dict[str, Any],
    score_threshold: float = 0.5,
    alpha: float = 0.3,
    show_labels: bool = False,
    add_titles: bool = False,
) -> np.ndarray:
    gt_panel = _render_gt_panel(image, gt_masks, alpha=alpha)
    mag_masks, mag_scores, mag_labels = prediction_to_lists(
        magformer_prediction)
    mag_panel = visualize_predictions(
        image=image,
        masks=mag_masks,
        scores=mag_scores,
        labels=mag_labels,
        score_threshold=score_threshold,
        alpha=alpha,
        show_labels=show_labels,
    )
    mask_masks, mask_scores, mask_labels = prediction_to_lists(
        mask2former_prediction)
    mask_panel = visualize_predictions(
        image=image,
        masks=mask_masks,
        scores=mask_scores,
        labels=mask_labels,
        score_threshold=score_threshold,
        alpha=alpha,
        show_labels=show_labels,
    )

    stacked = cv2.hconcat([gt_panel, mag_panel, mask_panel])
    if not add_titles:
        return stacked

    title_band = np.full((24, stacked.shape[1], 3), 255, dtype=np.uint8)
    titles = ["GT", "MagFormer", "Mask2Former"]
    width = image.shape[1]
    for idx, title in enumerate(titles):
        x = idx * width + 8
        cv2.putText(title_band, title, (x, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
    return cv2.vconcat([title_band, stacked])
