# R132 Final and R134 Official Mask2Former Continuation

Date: 2026-05-19
Host: `4029` (`/home/hdd3/zhanghaonan/magformer`)
Branch: `feature/vc-suda-sim2real`

## R132 final

R132 output: `output/baseline/r132_official_m2f_32k_cache_continue_g67`

Final checkpoint: `model_final.pth`

Source validation at iter 3000:

```text
bbox AP: 45.3970
segm AP: 50.0597
```

The iter 2500 segm AP was `45.2183`, so iter 2500 to 3000 improved by `+4.8414`. Source AP was still rising, so this is not a converged source baseline.

## R134 continuation

Continuation method: new output directory seeded with a copied R132 final checkpoint as `model_0002999.pth`, plus `last_checkpoint` pointing to that copy. This preserves R132 artifacts and still lets Detectron2 `--resume` load trainer state, including optimizer, AMP scaler, and LR scheduler.

Output: `output/baseline/r134_official_m2f_32k_cache_continue_3000_5000_g67`

TMUX session: `r134_official_m2f_32k_cache_continue_3000_5000_g67`

Exact command is recorded in:

```text
output/baseline/r134_official_m2f_32k_cache_continue_3000_5000_g67/tmux_command.txt
output/baseline/r134_official_m2f_32k_cache_continue_3000_5000_g67/launch_command.sh
```

Key options:

```text
CUDA_VISIBLE_DEVICES=6,7
--resume
--num-gpus 2
OUTPUT_DIR /home/hdd3/zhanghaonan/magformer/output/baseline/r134_official_m2f_32k_cache_continue_3000_5000_g67
MODEL.WEIGHTS /home/hdd3/zhanghaonan/magformer/output/baseline/r134_official_m2f_32k_cache_continue_3000_5000_g67/model_0002999.pth
SOLVER.MAX_ITER 5000
SOLVER.CHECKPOINT_PERIOD 500
TEST.EVAL_PERIOD 500
```

Resume verification from `log.txt`:

```text
Loading from .../r134_official_m2f_32k_cache_continue_3000_5000_g67/model_0002999.pth
Loading trainer from .../model_0002999.pth
Loading scheduler from state_dict ...
Starting training from iteration 3000
```

Initial training verification:

```text
iter: 3019 lr: 0.0001 max_mem: 9306M
iter: 3039 lr: 0.0001 max_mem: 9364M
```

GPU check at launch showed the R134 worker processes on physical GPUs 6 and 7. Existing GPU 4/5 jobs were left running and were not touched.

Do not start pseudo-real finetune until this finite source-continuation stage finishes and is evaluated.
