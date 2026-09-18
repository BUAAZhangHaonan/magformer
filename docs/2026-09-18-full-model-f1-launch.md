# F1 全量训练启动 (2026-09-18 14:08, 用户决策: 小模型证据充分, 转完整版)

## 小模型双臂终局 (证据已在 output/aps_20260913/EVIDENCE.md §9)
- 停止于 s1c 96.9K / s0c 104.2K 步 (38-41% of 256K); AP_s 相对优势 8 个评估窗口单调扩大
  (+19%→+30%→+35%→+36%→+42%→+47%→+52%→+54%), ΔAP_s 绝对 +0.97→+3.37pt, ΔmAP 回升至 +3.0pt;
  P3 分数成型面板: 设计臂 TP 分数中位 <64:0.51 / 64-256:0.65 / 256-1k:0.88 vs 对照 0.31/0.33/0.83。
- 用户判定: 方案有效性确立, 小模型剩余瓶颈 (100query/4层/4000图) 需完整容量+全量数据表达。

## DDP 修复 (0.70× → 可用)
- trainer.py DDPTrainer: broadcast_buffers=False (buffer 除 _dccg_step 外为常量且各 rank 同步推进),
  gradient_as_bucket_view=True (消除 grad→bucket 拷贝), static_graph 配置门控 (实测更差, 默认关);
  _preclip_gradient_norm_tensor + foreach clip: 消灭逐步 per-param python 循环与 backward 后 .item() 硬同步。
- schema.py runtime: ddp_broadcast_buffers / ddp_gradient_as_bucket_view / ddp_static_graph 三键。
- 实测 (小模型 2 卡, 有效 batch 4): 3.03 → 1.70 s/it (1.78×); static_graph 2.17 (弃用)。
- 实测 (4 卡全模型, bench6 150步): 1.44-1.60 s/step = 相对单卡全模型 ~2.6×缩放, 过 2.2×门。

## F1 run (configs/aps_20260913_full/f1_full_design_256k.yaml)
- 完整架构 (c0: dec8/FFN2048/200q/enc6) + 胜出方案包 (stride-2 画布/s4末层注意/topk门/scale_balanced/
  min_scale 0.5/hires_lr 20) + **AIM matcher** (matcher_small_gt_mode: aim, 斗兽场赢家, commit d1c482d0)。
- 初始化: Swin-T ImageNet D2 权重 + MobileNetV3-L depth 权重 + model_init.pth (M2F 转换头, c0 同款) — 非从头。
- 数据: instances_train.validated.json 全量 25654 图; eval 全val 3276, eval/ckpt 每 8K。
- 训练: 4卡DDP (GPU 4,5,6,7), ga=1 → 有效 batch 4 (= c0 配方), lr 1e-5 cosine, 256K 步;
  torchrun --nproc_per_node=4 --master_port=29501; 吞吐 ~1.5-1.6s/step → 预计 ~5.4 天 (含 32 次 eval)。
- 对照锚点: c0 fullval mAP 0.8603 / AP_s 0.264; 首 8K eval 健康门 AP>0.1。
- 监控: f1_watch.sh (eval快照+分桶TP分数面板) + f1_five_hour.sh (digest 落盘 p5_runs/digest_history.log,
  服务器端守护, 跨应用关闭存活) + in-app 4h 定时检查。
