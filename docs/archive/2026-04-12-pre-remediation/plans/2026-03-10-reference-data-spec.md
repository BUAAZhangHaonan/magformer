# Reference Data Generation Specification

**⚠️ The features described in this document are not planned for implementation and are retained for historical reference only.**

> **For ECC-Dataset Development:** This document defines the data requirements and output contract for generating single-part reference banks used by the next-stage reference-conditioned segmentation pipeline.

**Goal:** Produce a reusable, per-part, multi-view RGB-D reference dataset for a single electronic component model, suitable for training and inference in a reference-conditioned segmentation system.

**Architecture:** The reference bank is generated as a separate pipeline inside `ecc-dataset`, reusing MuJoCo for depth and masks and Blender/3DGS for RGB where appropriate, while enforcing strict cross-modal geometric alignment.

**Tech Stack:** `ecc-dataset` pipeline framework, MuJoCo simulator, Blender rendering, optional 3DGS rendering, JSON metadata, PNG RGB/masks, NPY depth.

---

## 1. Problem Definition

The downstream model no longer assumes zero-shot multi-category instance segmentation.

Instead, each run targets a **single part ID** and receives:

- a cluttered query scene (`RGB-D`)
- a per-part reference bank (`RGB-D + optional mask`)

The purpose of the reference bank is to tell the model:
- what the part looks like
- what its geometric silhouette is likely to be
- how its visible fragments should be grouped under occlusion

This specification defines how to generate that reference bank.

---

## 2. Target Output Layout

The generated dataset must follow this directory structure:

```text
outputs/reference_data/<reference_set_name>/<part_id>/
  rgb/
    view_000.png
    view_001.png
    ...
  depth/
    view_000.npy
    view_001.npy
    ...
  mask/
    view_000.png
    view_001.png
    ...
  camera/
    view_000.json
    view_001.json
    ...
  meta/
    manifest.json
    shape_stats.json
    qa_report.json
    preview_contact_sheet.png
```

Additionally, the root of the generated reference set must include:

```text
outputs/reference_data/<reference_set_name>/manifest.json
```

This manifest describes all parts contained in the set.

---

## 3. Required Data Modalities

## 3.1 RGB

Allowed sources:
- Blender render
- 3DGS render

Requirements:
- one isolated object only
- no occlusion
- no distracting background
- stable image size across all views
- exactly aligned with corresponding depth and mask
- stored as `.png`

## 3.2 Depth

Preferred source:
- MuJoCo

Requirements:
- float32 `.npy`
- unit = meters
- same pose / same camera / same resolution as RGB
- invalid pixels encoded consistently (recommended: `0.0`)

## 3.3 Mask

Preferred source:
- MuJoCo or Blender object mask export

Requirements:
- binary mask
- same resolution as RGB and depth
- stored as `.png`
- foreground should correspond exactly to the part silhouette in the RGB image

---

## 4. Alignment Rules (Critical)

If RGB and depth come from different rendering systems (for example Blender/3DGS RGB + MuJoCo depth), the following must all hold:

- identical camera intrinsics
- identical camera extrinsics
- identical object pose
- identical resolution
- identical crop convention
- identical orientation convention

If any of these are not guaranteed, the view is invalid and must not be exported.

### Mandatory alignment QA per view
Each view must generate or verify:
- `rgb-mask overlay`
- `depth-mask overlay`
- `rgb edge vs depth edge preview`

These do not need to be kept per view long-term, but must be used to produce the final QA report and preview sheet.

---

## 5. View Sampling Requirements

### Recommended v1 sampling
- `24` views total
- `12` azimuth angles
- `2` elevation levels

Recommended default:
- azimuths: `0, 30, 60, ..., 330`
- elevations: `0°, 30°`

### Minimum acceptable version
- no fewer than `8` views

### Strong recommendation
The exported set should include broad silhouette variation, not just tiny camera perturbations.

---

## 6. Required Metadata

## 6.1 Per-view camera JSON
Each view must have `camera/view_xxx.json` with at least:

- `view_id`
- `part_id`
- `intrinsics`
- `extrinsics`
- `resolution`
- `fov`
- `distance_to_object`
- `azimuth`
- `elevation`
- `rgb_source`
- `depth_source`
- `mask_source`

## 6.2 Per-part manifest
`meta/manifest.json` must include:

- `part_id`
- `num_views`
- `image_size`
- `rgb_source`
- `depth_source`
- `mask_source`
- `view_ids`
- `generation_time`
- `pipeline_version`
- `canonicalization_info` (path or embedded summary)

## 6.3 Shape statistics
`meta/shape_stats.json` must include:

- `mean_area_ratio`
- `mean_bbox_aspect_ratio`
- `mean_mask_perimeter`
- `viewwise_area_ratio`
- `viewwise_aspect_ratio`

These are required by the downstream graph-merge stage.

---

## 7. Quality Control Requirements

Each part reference bank must pass these checks:

- RGB / depth / mask counts match
- file stems match exactly across all modalities
- masks are non-empty
- depth is not all zero
- all modalities share the same resolution
- camera metadata exists for every exported view
- view count meets the minimum threshold

### Required QA artifacts
For each part:
- `meta/qa_report.json`
- `meta/preview_contact_sheet.png`

### `qa_report.json` must record
- missing file count
- invalid alignment count
- empty mask count
- invalid depth count
- final exported view count
- pass / fail flag

---

## 8. Module Placement in `ecc-dataset`

The new reference data generator should live under:

```text
src/ecc_dataset/reference_data/
  __init__.py
  reference_builder.py
  io.py
  manifest.py
  qa.py

src/ecc_dataset/steps/
  build_reference_data.py
```

### Why here
- the task spans MuJoCo, Blender, and optional 3DGS
- it is not specific to one renderer
- it fits the current `ecc_dataset` multi-step orchestration pattern

---

## 9. CLI Integration Requirements

The new functionality should be exposed through the existing `ecc_dataset` CLI.

Recommended commands:
- `build-reference-data`
- `verify-reference-data`

### Minimum CLI arguments
- `--part-id`
- `--output-root`
- `--rgb-source {blender,3dgs}`
- `--depth-source mujoco`
- `--mask-source {mujoco,blender}`
- `--num-views`
- `--image-size`
- `--overwrite`
- `--dry-run`

---

## 10. Source Reuse Guidance

## 10.1 Blender / 3DGS RGB
Use Blender or 3DGS only as the RGB appearance source.

Best reuse candidates in the current `ecc-dataset` repo:
- Blender render scripts already producing RGB and camera poses
- 3DGS pipeline outputs when rendered under controlled views

## 10.2 MuJoCo depth / masks
Use MuJoCo as the preferred source of:
- depth
- clean foreground mask
- geometry-consistent visibility

This gives the downstream pipeline a stable geometric reference even if RGB comes from a different renderer.

---

## 11. Data Contract for Downstream Training

The downstream `magformer` / `unet_reference_inst` side should assume this reference bank can be passed through a single argument:

```text
--reference-root /path/to/outputs/reference_data/<reference_set_name>/<part_id>
```

The generator therefore must ensure that the per-part folder is self-contained and complete.

---

## 12. Non-Goals for v1

The following are explicitly out of scope for the first version:

- multi-SKU / zero-shot support
- category taxonomy or part classification
- automatic retrieval of the correct reference set
- full 3D reconstruction quality benchmarking
- mesh-level amodal supervision

The first version only serves the single-part reference-conditioned segmentation pipeline.

---

## 13. Recommended Development Sequence

1. Implement output layout and manifest writing
2. Implement RGB/depth/mask aligned export for one part
3. Implement QA report and preview generation
4. Add CLI wiring
5. Validate one part end-to-end
6. Only then scale to multiple parts

---

## 14. Acceptance Criteria

The module is accepted when:

- a single part can be exported into the target directory structure
- every exported view has aligned RGB, depth, and mask
- `manifest.json`, `shape_stats.json`, and `qa_report.json` are present
- the downstream reference runner can consume the produced directory without manual edits
- at least one manually inspected preview sheet confirms correct alignment

---

## 15. Assumptions

- each run targets one part ID at a time
- the downstream segmentation system is allowed to rely on explicit reference data
- RGB may come from Blender or 3DGS
- depth and mask should preferentially come from MuJoCo
- strict cross-modal alignment is a hard requirement, not an optional quality improvement
