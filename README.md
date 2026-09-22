# MAGFormer（封档版 · 2026-09-23）

RGB-D 融合的小目标实例分割模型。**Swin-T（RGB）+ MobileNetV3-L（深度）双编码器**
+ DCCG 跨模态融合（置信门控 + DPE 深度位置编码）+ Mask2Former 式多尺度掩码解码器。

> 本仓库已于 2026-09-23 封档：30+ 优化模块的探索战役全部判负后深度清理，
> 只保留终版代码、128K 基线权重与终局记录。入口文档：
> **[docs/2026-09-23-project-seal.md](docs/2026-09-23-project-seal.md)**

## 最终指标（F1@128K，校准 EMA，全 val 3276 图）

| segm AP | segm APs | bbox AP |
|---|---|---|
| **0.8744** | **0.2698** | 0.8596 |

结构性结论：唯一持续有效的杠杆 = 双塔编码器 + 延长训练；标准 concat 融合头
（mAP 0.91 / APs 0.354）优于本族全部融合设计。AP_s 理论上限（GT-oracle）0.768。
单前景类（COCO RGB-D 格式）；多类未实现。

## 目录

```
magformer/            模型包（探索模块全部 config-gated 默认关 = 逐位基线）
tools/                训练/评估/推理入口（train.py 兼容 torchrun）
configs/              base.yaml + full_design_256k.yaml（终版训练配置）+ templates/
tests/                核心测试套件（基线集成测试在 baselines/）
ops/                  4 卡启动 / 内存看门狗 / 停机脚本
baselines/            对比基线模型族（detectron2/Mask2Former/ultralytics 子模块等，
                      未做安全加固，勿喂不可信数据）
pretrained_weights/   编码器预训练权重（swin_tiny / mobilenetv3_depth / model_init）
output/               运行产物：aps_20260913/p5_runs/f1_seal_128k = 128K 基线权重
                      （best.pt 校准 EMA / last.pt 完整训练态）；pretrained/ = 基线权重
magformer_datasets/   数据集（20260318_1K_32254）
docs/                 终局文档（封档主文档 / 探索路线图表 / 七场斗兽场裁决书 / 证据链）
```

## 数据格式

```
dataset_root/
├── images/{train,val,test}/
├── depth/depth_npy/{train,val,test}/ + depth_noise_mask/
└── annotations/instances_{train,val,test}.json
```

## 快速开始

```bash
conda activate magformer            # torch 2.5.1+cu124（environment.magformer.yml）
# 4 卡训练（GPUs 4-7 物理索引；勿设 CUDA_VISIBLE_DEVICES）
python -m torch.distributed.run --nproc_per_node=4 --master_port=29611 \
  tools/train.py --config configs/full_design_256k.yaml
# 评估
python tools/train.py --config configs/full_design_256k.yaml --eval-only
# 从 128K 基线 warm-start：model.finetune_weights 指
#   output/aps_20260913/p5_runs/f1_seal_128k/best.pt
```

硬规则：只用 GPUs 4-7；251GB 主机同时仅一个 4-rank 束；系统内存 ≤90%
（`ops/launch_4rank.sh` 自带看门狗，88%×3 拍自动杀整棵进程树）。

## License

Apache License 2.0
