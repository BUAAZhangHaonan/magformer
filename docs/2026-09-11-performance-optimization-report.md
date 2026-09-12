# 2026-09-11 · c0 (v317) 性能优化终报

目标变体：**c0_corrected_300k_seed42**（全部 magformer 变体中指标最高，fullval segm AP **0.8603**）。优化代码已按阶段落在 `experiments_archive/v317_next_stage_2026-07/source/`（该目录即本谱系的正典优化版源码；基线快照见 git 历史）。实验环境：WS-4029GP-TRT，GPU 4/5（RTX 3090）。

## 成果（同机同协议配对测量）

| 指标 | 基线 | 优化后 | 提升 |
|---|---|---|---|
| 训练 s/optimizer-step（bs1×accum4, AMP） | 7.18 | 2.48 | **2.89×** |
| dataloader 供给 s/sample（4 workers） | 1.41（RSS 98.8GB） | 0.45（PSS 28GB） | 3.1× / 内存↓72% |
| 推理单图延迟 p50 @1024²（含后处理） | 207 ms | 149 ms | 1.39×（p99 279→155ms） |
| 推理峰值显存 | 2826 MB | 509 MB | ↓5.6× |
| e2e 评估（subset-1000） | 1055s（0.95 img/s） | 210.5s（**4.8 img/s**） | **5.05×** |
| 精度 | segm AP 0.8419384637093063 | **0.8419384637093063** | **比特级一致** |

## 分阶段改动（对应 commit）

1. **perf(data)** — 数据管线：mask 轴向占据 cv2.reduce、uint8 打包最近邻 resize（与 bool 路径比特一致）、box 计算融合、光度增强移 GPU（loader 传 uint8+参数）。门禁：val split 逐位一致；train split mask/box 精确、图像 ≤1/255。
2. **perf(train)** — 训练计算：trainer 每 micro 步 `.item()` 同步消除（归并至 log_period）、criterion num_masks 免设备往返、DCCG 温度 GPU 张量化（eval 缓存）、可选 `runtime.torch_compile_modules`（建议 `[fusion, rgb_backbone]`，schema 新字段，默认关闭）。0.733→0.617 s/micro。唯一数值效应：conf 除法 fp16→fp32 提升（固定种子确定性 1.3e-3 相对漂移，方向更精确）。
3. **perf(infer)** — GPU 后处理：新增 `gpu_postprocess.py`（top-k/上采样/sigmoid/阈值/bbox 全 GPU + packed-bit 单次异步 D2H，替代 3 次 140ms 同步拷贝）；coco_export 直接消费 GPU bbox、F 序 mask 免拷贝。
4. **perf(infer) graphs+pipeline** — 新增 `cuda_graph_runner.py`（固定形状 fp32 推理区手工 CUDA Graph 捕获，回放逐位一致）与 `export_pipeline.py`（fork 进程池 RLE 导出 + 传输线程 + pinned ring）；pixel decoder/DPE/MSDA 模块宿主同步提升出捕获区。
5. **perf(wire)** — arch 导出路径 + `evaluate.py` 三个开关：`--gpu-export`、`--cuda-graph`、`--export-pipeline procs --export-workers 4`（不加任何 flag = 原始行为，完全可回退）。

## 使用

```bash
cd experiments_archive/v317_next_stage_2026-07/source
# 训练（相对原配置仅 +num_workers:4, +torch_compile_modules:[fusion,rgb_backbone]）
python tools/train.py --config <c0 yaml> --dataset-root <root>
# 评估（快速模式，逐位一致）
python tools/evaluate.py ... --export-pipeline procs --export-workers 4 --num-workers 8
```

## TensorRT（2026-09-12 补全）

安装：TRT 10.9.0.34 + torch-tensorrt 2.5.0，经 **g203 clash 跳板代理**（`http://10.134.132.166:7890`，`~/.bashrc` 已加 `proxy_on/proxy_off`）从 pypi.nvidia.com 拉取 manylinux_2_28 wheel（pypi.org 直连只有 sdist 且构建期要拉被墙 CDN）。

实测结论（GPU 7, RTX 3090, 1024²；AP门禁 = subset-100 segm AP与eager差 ≤0.3pt）：

| 配置 | 段延迟 | AP Δ | 判定 |
|---|---|---|---|
| 全图 ONNX→TRT fp32 | 70 ms（同段eager≈135ms，2×） | 预测错乱 | ❌ 融合/解码段的数据依赖分支被ONNX导出烤错侧（架构性，需无分支重构才能解锁） |
| towers(Swin+MBv3L)→TRT **fp32** | 34.3→19.6 ms（1.75×） | **0.0pt** | ✅ 可用 |
| towers→TRT **fp16** | 34.3→**6.4 ms（5.4×）** | **0.0pt** | ✅ **推荐** |
| 全模型（fp16-TRT塔+eager其余） | 156→**123 ms（1.27×）** | 0.0pt | ✅ 与CUDA-Graph栈叠加待验证 |

**结论（2026-09-12修正）**：TensorRT 对本模型**可用但只能模块化**——双塔（纯卷积）TRT化在 fp32 与 fp16 下都 AP 无损（初版报告用"比特级一致"作门禁属误判，编译引擎跨框架不可能比特一致，正确门禁是AP容差；实测 towers-TRT 的 21% mask XOR 全部是AP中性的边界抖动）。全图 TRT 仍被融合/解码段的数据依赖分支挡住（架构性，torch.where 无分支重构可解锁，未实施）；fp16 全模型失败同样源于 DCCG 门控段而非塔。torch-tensorrt 的 dynamo/TS 前端在本模型图上分别死于 dynamo 内部错误与分区器段错误，纯 ONNX→TRT python API 路线是唯一能构建成功的路径。

## 否定性结论（防止后人踩坑）

朴素 fp16/bf16 autocast、channels_last、裸 torch.compile 在本模型上**全部负优化**（DCCG 门控对精度极敏感，fp16 mask XOR 65%）；批量化需先向量化逐图后处理；训练 loss 含随机点采样——任何 loss 门禁必须每前向固定种子（否则噪声底 7%）。

## 基础设施备注

9月11日 15:01 GPU 0（PCI 1a:00）Xid 79 掉卡为**复发性硬件故障**（9月9日 17:16 已发生过一次，与本项目无关）；GPU 0 上租户进程无任何 Xid 记录；进程级 Xid 31（GPU 4/6，含一个被 SIGKILL 的进程）至多为诱因。运维建议：GPU 进程用 SIGTERM 退出；关键任务避开 GPU 0。
