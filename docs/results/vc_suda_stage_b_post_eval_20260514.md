# VC-SUDA Stage B Post-Eval - 2026-05-14

## Scope

- Agent: AK.
- Host: `WS-4029GP-TRT` via `ssh 4029`.
- Repository: `/home/hdd3/zhanghaonan/magformer`.
- Branch: `feature/vc-suda-sim2real`.
- HEAD before eval: `5c761af`.
- Conda env: `magformer`.
- Eval config: `configs/eval_vc_suda_stage_b_1024_teacher8499_segm.yaml`.
- Dataset root: `magformer_datasets/pseudo_real_512`.
- Validation annotation: `annotations/instances_val.json`.
- Validation split: `val`.
- Checkpoint evaluated: `output/experiments/vc_suda_stage_b_1024_teacher8499_20260514_005821/checkpoint_iter_0008999.pth`.
- Eval output dir: `output/experiments/20260514_stage_b_post_eval/checkpoint_8999_segm`.
- GPU: physical GPU 4, exposed as logical `cuda:0` by `CUDA_VISIBLE_DEVICES=4`.

## Preflight

- Target Stage B run dir existed: `output/experiments/vc_suda_stage_b_1024_teacher8499_20260514_005821`.
- Required files existed before eval:
  - `checkpoint_iter_0008999.pth`, 556 MiB.
  - `checkpoint_iter_0009000.pth`, 556 MiB.
  - `model_best.pth`, 556 MiB.
- No Stage B train or torchrun process was running. The only train-like process found was another user's `python train.py --method lora --gpu-ids 0`, started from outside this project.
- GPU preflight for physical GPU 4 and 5:
  - GPU 4: 15 MiB / 24576 MiB, 0% util.
  - GPU 5: 15 MiB / 24576 MiB, 0% util.
- The eval ran on GPU 4. No CUDA illegal access occurred, so no retry on GPU 5 was needed.

## Command

```bash
ssh 4029
cd /home/hdd3/zhanghaonan/magformer
source /home/hdd3/zhanghaonan/anaconda3/bin/activate
conda activate magformer
NVJIT_PATH=$(python -c "import nvidia.nvjitlink; print(nvidia.nvjitlink.__path__[0]+'/lib')")
CUSPARSE_PATH=$(python -c "import nvidia.cusparse; print(nvidia.cusparse.__path__[0]+'/lib')")
CUBLAS_PATH=$(python -c "import nvidia.cublas; print(nvidia.cublas.__path__[0]+'/lib')")
export LD_LIBRARY_PATH=${NVJIT_PATH}:${CUSPARSE_PATH}:${CUBLAS_PATH}:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
CUDA_VISIBLE_DEVICES=4 python tools/evaluate.py \
  --config-file configs/eval_vc_suda_stage_b_1024_teacher8499_segm.yaml \
  --weights output/experiments/vc_suda_stage_b_1024_teacher8499_20260514_005821/checkpoint_iter_0008999.pth \
  --output output/experiments/20260514_stage_b_post_eval/checkpoint_8999_segm \
  --batch-size 4 \
  --num-workers 4 \
  2>&1 | tee output/experiments/20260514_stage_b_post_eval/checkpoint_8999_segm/eval_gpu4.log
```

## Eval Contract

- `tools/evaluate.py` passes `fail_on_empty=True` into `run_inference_evaluation`.
- The eval config requested both IoU types: `bbox` and `segm`.
- The command loaded the checkpoint through `--weights`, not through resume state.
- The output JSON was written to `output/experiments/20260514_stage_b_post_eval/checkpoint_8999_segm/coco_instances_results.json`.
- The output JSON is under ignored `output/experiments/` and was not staged for commit.

## Metrics

| Type | AP | AP50 | AP75 | AP_small | AP_medium | AP_large |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bbox | 0.1637 | 0.4824 | 0.0727 | 0.2146 | -1.0000 | -1.0000 |
| segm | 0.1128 | 0.3810 | 0.0350 | 0.1550 | -1.0000 | -1.0000 |

Raw values from the printed metrics:

- `bbox_AP`: 0.16374048014700276.
- `bbox_AP50`: 0.4824453284166137.
- `bbox_AP75`: 0.07272659691894581.
- `segm_AP`: 0.11284750849899325.
- `segm_AP50`: 0.38104397305161036.
- `segm_AP75`: 0.034973678789648875.

## Prediction Diagnostics

- Validation images in annotation: 28.
- Ground-truth annotations in val: 1892.
- Total predictions: 1700.
- Predictions with bbox: 1700.
- Predictions with segmentation: 1700.
- Images with predictions: 28.
- Images without predictions: 0.
- Predictions per image: min 48, max 82, mean 60.714285714285715.
- Category counts: `{1: 1700}`.
- Score range: min 0.05115976557135582, max 0.9663563966751099, mean 0.7985988650295665.

Note: the runtime log says `Inference done: 7 images`; `magformer/engine/eval_runtime.py` increments that counter once per batch. With batch size 4 and 28 validation images, the log's 7 is the batch count. The COCO prediction JSON and val annotation both confirm 28 unique evaluated image IDs.

## Status

- Result status: post-eval passed.
- `fail_on_empty` did not trigger because 1700 COCO predictions were exported.
- No CUDA illegal issue was observed on physical GPU 4.
- Large prediction output was left ignored under `output/experiments/` and was not committed.
