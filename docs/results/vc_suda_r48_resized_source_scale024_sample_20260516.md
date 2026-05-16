# VC-SUDA R48 Resized Source Scale 0.24 Sample, 2026-05-16

R48 adds a real COCO RGB-D resize builder and validates it on a 64-image source sample. No full 25,654-image build, training, or model eval was run.

## Tool

Added:

- `tools/build_resized_coco_rgbd_dataset.py`
- `tests/test_build_resized_coco_rgbd_dataset.py`

The builder reads the existing COCO RGB-D layout:

- RGB: `images/<split>/<file_name>`
- depth: `depth/depth_npy/<split>/<stem>.npy`
- annotations: COCO JSON

Resize policy:

- RGB uses `PIL.Image.Resampling.BILINEAR`.
- depth uses `cv2.INTER_LINEAR`, keeps `float32` depth values, and does not normalize depth.
- polygon segmentation scales coordinates directly by `scale`; bbox and area are scaled numerically.
- RLE segmentation is decoded, resized with `cv2.INTER_NEAREST`, re-encoded as compressed COCO RLE, and bbox/area are recomputed from the resized mask.
- image and annotation ids are preserved. No id remap is done.
- existing outputs are rejected unless `--overwrite` is passed.

## Why Scale 0.24

R47 showed ordinary 32K source annotations become target-scale after projection at `scale=0.24`:

- projected images: `25,654`
- projected mask area p50: `436.1472`
- projected `area<=256`: `0.227160`
- basename overlap with target splits: `0`

R47 also showed annotation-only files are not enough for training. R48 therefore materializes real RGB/depth/annotation samples before any full build.

## Sample Build

Command:

```bash
/home/hdd3/zhanghaonan/anaconda3/bin/conda run -n magformer python tools/build_resized_coco_rgbd_dataset.py \
  --source-root magformer_datasets/20260318_1K_32254 \
  --source-ann annotations/instances_train.json \
  --output-root output/diagnostics/r48_resized_source_scale024_sample_20260516/dataset \
  --output-ann annotations/instances_train_scale024_sample.json \
  --scale 0.24 \
  --max-images 64 \
  --split train \
  --summary-json output/diagnostics/r48_resized_source_scale024_sample_20260516/summary.json \
  --overwrite
```

Output:

- dataset root: `output/diagnostics/r48_resized_source_scale024_sample_20260516/dataset`
- annotation: `output/diagnostics/r48_resized_source_scale024_sample_20260516/dataset/annotations/instances_train_scale024_sample.json`
- summary: `output/diagnostics/r48_resized_source_scale024_sample_20260516/summary.json`
- protocol summary: `output/diagnostics/r48_resized_source_scale024_sample_20260516/protocol_summary.json`

Summary:

| metric | value |
|---|---:|
| images | 64 |
| annotations | 2,792 |
| RGB files | 64 |
| depth `.npy` files | 64 |
| mask area p50 | 453.456 |
| area<=256 ratio | 0.113539 |

The sample wrote about `51M` under the dataset directory. It is intentionally small.

## Protocol Check

Command:

```bash
/home/hdd3/zhanghaonan/anaconda3/bin/conda run -n magformer python tools/analyze_vc_suda_dataset_protocol.py \
  --ann r48_sample=output/diagnostics/r48_resized_source_scale024_sample_20260516/dataset/annotations/instances_train_scale024_sample.json \
  --ann target_unlabeled=magformer_datasets/pseudo_real_512/annotations/instances_target_unlabeled.json \
  --output-json output/diagnostics/r48_resized_source_scale024_sample_20260516/protocol_summary.json \
  --output-md output/diagnostics/r48_resized_source_scale024_sample_20260516/protocol_summary.md
```

| name | images | instances | inst/img p50 | mask area p50 | area<=256 | density 25-30 | density 46-60 | density >90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| r48_sample | 64 | 2,792 | 50.000 | 453.456 | 0.113539 | 16 | 48 | 0 |
| target_unlabeled | 200 | 11,750 | 50.000 | 452.000 | 0.249702 | 22 | 131 | 47 |

The p50 mask scale matches target closely. The `area<=256` ratio is lower than target in this first-64 sample, so this small sample is a mapping and loader validation only. It is not a replacement for the R47 full annotation projection gate.

## Loader Check

Command:

```bash
/home/hdd3/zhanghaonan/anaconda3/bin/conda run -n magformer python -c 'from magformer.data.dataset import CocoRgbdDataset; root="output/diagnostics/r48_resized_source_scale024_sample_20260516/dataset"; ds=CocoRgbdDataset(dataset_root=root, ann_file="annotations/instances_train_scale024_sample.json", split="train", is_train=True, has_annotations=True); s=ds[0]; print({"len": len(ds), "image_shape": tuple(s["image"].shape), "depth_shape": tuple(s["depth"].shape), "masks_shape": tuple(s["masks"].shape), "boxes_shape": tuple(s["boxes"].shape), "labels_shape": tuple(s["labels"].shape), "height": s["height"], "width": s["width"]})'
```

Result:

```text
{'len': 64, 'image_shape': (246, 246, 3), 'depth_shape': (246, 246), 'masks_shape': (246, 246, 25), 'boxes_shape': (25, 4), 'labels_shape': (25,), 'height': 246, 'width': 246}
```

RGB, depth, and annotations are readable through `CocoRgbdDataset`.

## Validation

Commands run:

```bash
/home/hdd3/zhanghaonan/anaconda3/bin/conda run -n magformer python tools/build_resized_coco_rgbd_dataset.py --help
/home/hdd3/zhanghaonan/anaconda3/bin/conda run -n magformer python -m pytest tests/test_build_resized_coco_rgbd_dataset.py -q
/home/hdd3/zhanghaonan/anaconda3/bin/conda run -n magformer python -m py_compile tools/build_resized_coco_rgbd_dataset.py tests/test_build_resized_coco_rgbd_dataset.py
```

Unit test result: `4 passed`.

## Full Build Gate

Full construction is allowed only as a later data-build step, not as part of R48 and not as training input yet. Before full construction:

- keep `scale=0.24` and no id remap;
- confirm disk budget for all resized RGB/depth files and JSON output;
- run from a clean git state;
- build with no `--max-images` only after accepting the write size;
- rerun protocol analysis on the full output and compare to R47 projected metrics;
- run loader/read checks on sampled files from the full output.

Training remains blocked until the full real dataset exists and passes those data gates.
