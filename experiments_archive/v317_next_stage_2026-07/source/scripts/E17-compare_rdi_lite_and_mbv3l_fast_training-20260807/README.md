# E17 RDI Lite 与 RDI MBV3-L 快速对比

本实验只比较两个不同方法，各运行一个 seed42：

- GPU4：RDI Lite，配置 `rdi_lite.yaml`
- GPU5：RDI MBV3-L，配置 `rdi_mbv3l.yaml`

两份配置都使用单卡训练，`ims_per_batch=2`、`grad_accum_steps=2`、`num_workers=8`、`eval_period=5000`、`checkpoint_period=5000`、`max_iter=300000`，并从同一个初始化权重开始。服务器有 40 个物理核和 80 个逻辑核；两个任务合计使用 16 个数据加载进程，不占满整机 CPU。

active core 已允许配置不设置 CUDA allocator 上限；每次评估会复用项目已有 YOLO 风格掩码绘制并保存 GT 与 prediction 并排图；checkpoint 只保留 `best.pt` 和 `last.pt`。E17 配置没有显存比例字段，因此不会设置 CUDA allocator 上限。

启动命令：

```bash
cd /home/g203-4028/magformer
bash scripts/E17-compare_rdi_lite_and_mbv3l_fast_training-20260807/launch.sh
```

tmux 会话：

- `magformer-e17-rdi-lite-gpu4`
- `magformer-e17-rdi-mbv3l-gpu5`

输出目录：

- `output/experiments/E17-compare_rdi_lite_and_mbv3l_fast_training-20260807/rdi_lite`
- `output/experiments/E17-compare_rdi_lite_and_mbv3l_fast_training-20260807/rdi_mbv3l`
