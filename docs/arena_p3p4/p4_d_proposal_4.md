# P4-d 提案 #4：TREX——解码轨迹可靠性重打分 + 排序式去重（零训练、零额外前向）

- 场：P4-d 导出与分数融合｜提案人：#4（独立，未读他人提案）
- 基线锚（封存校准，EMA best.pt，标准协议全 val 3276）：segm mAP **0.8744 / AP_s 0.2700**（`docs/2026-09-21-f1-128k-calibrated.md` §二）
- 已关闭分支确认不再提案：topk 100→200 仅 +0.09 AP_s；二值化阈值 0.45/0.40 ≈0/负（同上 §三.3）；事后单调重标定（分位映射 −13.1pt、score=IoU −11.9pt，`reviews_20260920/r1_score/report.md` §1）。
- 一句话：**利用解码器已经在算、但目前被丢弃的 8 层中间预测（每层 cls logits / mask / box，`multiscale_decoder.py:1001-1038` 的 `aux_outputs`），为每个导出检测构造"轨迹可靠性"新特征与"喷洒重复组"结构，以桶中位锚定的乘法重排导出分数**——新信息改变排序，而非对旧分数做单调变换。

---

## 0. 机制概述

当前导出分数只用最后层的两个标量：`score = cls × maskness`（`gpu_postprocess.py:212-217`）。而解码器每层都产出完整预测（cls logits、mask（einsum 渲染在 mask 画布上）、box），8 层轨迹在推理时**已经算完并被扔掉**（`arch.py:1493-1494` 只取 `pred_logits`/`pred_masks`）。本提案从中提取五类零成本特征：

| 信号 | 定义（对 query q，层 l=0..7） | 对应 CQTR 三信号 | 假设的判别机理 |
|---|---|---|---|
| z1 语义持续 | `z1 = mean(sigmoid(p_L)) − mean(sigmoid(p_{0..1}))`；`z1b = min_{l≥4} sigmoid(p_l)`（末 4 层最深回撤） | semantic persistence | 真 keeper 的 cls 沿层单调爬升并保持；喷洒重复/迟到的 hard-FP 呈尖峰或震荡（锚：cls 停滞在 0.29/0.62 已 88K 步，r1 §2，说明是均衡态而非噪声） |
| z2 空间收敛 | `z2 = mean_{l∈5..7} IoU(bin(m_l), bin(m_{l+1}))`（逐层掩码在同一画布二值化后相邻层 IoU） | spatial convergence | 锁定真实目标的 query 掩码轨迹收敛；噪声 query 游走 |
| z3 跨尺度一致 | `z3 = IoU(bin(m_7↓), bin(m_6))`（末层 stride-2 渲染下采样到 stride-4 画布 vs 第 6 层渲染；可选再加 vs 粗层渲染均值） | cross-scale conflict（免费版） | CQTR 用"缩放图像重前向"测跨尺度冲突（超预算，不做）；我们利用解码器每层天然在不同 head level 渲染（`multiscale_decoder.py:982-991`：层 0-5 在 stride 8/16/32 轮转、末两层在 4/2）获得同构信号 |
| z4 重复组结构 | 导出 dets 间 box-IoU≥τ_g 连通成组（`compute_bbox_xyxy_gpu` 的 bbox 现成）；组内按基础分最高者为 keeper，非 keeper 乘 ε；组大小 log 为特征 | （CQTR 未覆盖，本地证据驱动） | 每个被命中的 <64 GT 平均挂 **14.05** 个 IoU≥0.5 的不同 det（r1 §3）；AP 最优排序=“每 GT 一个代表、代表按质量排、冗余沉底”（r1 §1 判读），该结构没有任何 per-det 标量变换能表达 |
| z5 核心掩码度（F2 臂，可选） | `mean(mask_prob ∈ erode(bin_mask, r))`，r=1-2px，空则取质心 3×3 峰值 | — | tiny mask 几乎全是边界，maskness 被边界带压低（理想小掩码上限 0.95-0.96 vs 大目标 0.99，EVIDENCE §5）；核区均值去边界偏置，是**特征级替换**（非 maskness 的单调函数，可改排序） |

融合（桶中位锚定乘法）：

```
final' = (cls × maskness) × g(z),   g = clamp(exp(Σ w_k·(z_k−c_k)/σ_k), g_min, g_max)
```

- 常数 {w_k(≤4), c_k(≤4), ε, τ_g} ≤ 10 个，一次性离线拟合（零训练红线内）；
- **c_k 按检测面积桶（<64 / 64-256 / 256-1k，det 面积口径同 r1 §3）校准使每桶 median(g)=1**：不净抬任何桶（正面规避 r1 (c) 系 −13.1pt 的"整桶抬分让 80.5% hard-FP 搭车"），只做**桶内重排**——而桶内 TP/FP 分离恰是活口（<64 桶分数 AUC 0.753、64-256 桶 **0.490=随机**，r1 §3：这两个桶里排序信息尚未榨干）。

---

## (a) 机制因果链：为什么新信息能改排序而单调变换不能

1. **AP 对分数的任意单调变换不变**（Rank-DETR 立论；本仓库实测：分位重标定 −13.1pt、score=IoU −11.9pt、oracle 版桶校准 −2.9pt，r1 §1）。所以唯一有资格的机制必须引入**不在 (cls, maskness) 中的信息**改变相对序。z1-z5 全部来自被丢弃的解码轨迹/检测间结构，与末层双标量统计上不共线（Gate A 直接实测增量 AUC）。
2. **因果方向**：小目标 det 池 80.5% hard-FP（<64 桶）意味着"抬分"必死；但**降 FP 的相对位**不需要知道 GT——hard-FP 的生成过程（多 query 抢占同一目标的喷洒、迟到的低置信尖峰、跨层不一致的掩码）在轨迹里留痕。keeper 是"早锁定、跨层一致、语义持续"的 query；喷洒 det 是"晚出现、掩码游走、语义回撤"的 query。z1-z3 把这三种痕做成特征，z4 把喷洒结构做成组内排序。
3. **为什么 64-256 桶是主战场**：该桶分数 TP/FP AUC=0.490（r1 §3）——现有分数在此**零信息**，任何有效新信号几乎是无本万利；且该桶 keeper 图内排名中位 85/100、36% 排 ≥90（r1 §3），改善其桶内排序直接作用在 AP_s 的 PR 前段。
4. **maskness 解耦的正确打开方式**（对杠杆 2 的回应）：COCO 面积档评估中，出档 det 被忽略（AP_s 只统计小面积档内的 det 与 GT），**跨桶排序不进入 AP_s**——所以"小掩码被结构性压分 vs 大目标"对 AP_s 只是间接问题（影响 mAP 与部署阈值），对 AP_s 有意义的只有**桶内** TP/FP 排序。这把"maskness 解耦"从已被证伪的跨桶抬分框架（r1 (c)）重定向到活的桶内框架：z5 核心掩码度与指数族 `cls × maskness^β, β∈{0,0.25,0.5,1}` 扫描（β≠1 时非单调变换，合法；封存 dets 没存 cls/maskness 分量，需 Phase 1 一次性补齐才能扫——这是本提案顺手关闭的一问）。
5. **对杠杆 3（配额）的裁决性论证**：配额只能从 200 query 里选人；topk200（全量导出后按分截断）是任何配额的**超集实验**，实测仅 +0.09 AP_s → 一切"从 101-200 名捞小 det"的配额上界 ≤+0.09pt，按证据关闭，仅留一个消融臂出数（§e）。

## (b) 与 set-prediction / 导出契约的兼容性

- **不改集合成员**：topk 切割仍在 cls 上、仍在融合前（`gpu_postprocess.py:184-185` 原位）；导出的实例集合（哪些 mask、多少个、什么类）与基线**逐位相同**，只改 `score` 字段。o2o 匹配语义、eos、NMS-free 契约零触碰。
- **排序式去重而非删除式 NMS**：冗余 det 不删（乘 ε 沉底），maxDets=100 / score_threshold=0 的评估契约不变；召回无下降风险（oracle 版沉底 +0.26pt，r1 §1 `dedup_fixed`，证明"只沉不删"有正收益且无召回代价）。
- **分数语义不漂移**：桶中位锚定保证每桶分数分布中位不变，部署阈值 τ=0.5 的 operating point 漂移有界（g∈[g_min,g_max]，默认 [0.5,1.25]）；验证中强制报告 F1@0.5 前后差。
- **CUDA-graph 兼容**：特征形状全静态（8×200×{1,1,4}），常数同 `mask_threshold` 一样烘焙进图；eager 路径（`_inference_raw`）与 GPU 路径（`_inference_raw_gpu`→`postprocess_tensors`）各加一个纯读 side-output 分支，rescore 关闭时输出与现状 bit-identical（Gate 0 强制）。

## (c) 理论收益上界与依据

全部锚定在封存 det 集上的反事实上界（r1 §1，标准 COCOeval，112K EMA dets；Phase 0 将在 128K 封存 dets 上重锚）：

| 锚 | AP_s 增量 | 出处 |
|---|---|---|
| 完美打分+oracle 去重 `a_plus_dedup_ranked`（排序侧总上界） | **+1.33pt**（0.2925→0.3058） | r1 §1 表 |
| 纯 TP/FP 保序分离 `a_separate` | +0.52pt | 同上 |
| oracle 去重（沉底） | +0.26pt | 同上 |
| AR_s（重打分召回锚） | +1.61pt；base 已兑现 95.5% | r1 §1 结论 1 |
| CQTR 锚（跨模型先例，非本仓库） | APs +1.95~3.81、TinyPerson +3.85，9 冻结检测器零训练 | arXiv 2609.06581（`research_20260921/query_existence.md` 候选 5） |

- **为什么 CQTR 量级不能全额迁移**：CQTR 收益来自分数-定位错位大的 det 集；我们的 det 集已兑现自身排序上限的 95.5%（r1 §1），且 CQTR 中最贵的"反事实尺度干预"腿（多次前向）被 +10ms 预算排除，只保留免费的轨迹腿+本地重复结构腿。诚实折算：捕获上界的 20-40% → **条件收益 +0.3~0.8pt AP_s**；z4 单独 ≤0.26pt；排序臂 ≤0.52pt。
- **存在性轴无收益**（明示）：38.1% 小 GT 在全部 200 query 中无 IoU≥0.5 掩码（调研 B §1），排序机制不造 det；topk200 +0.09 已封顶该轴。
- 置信度：机制可实现性 0.95（信号已在显存里）；信号有效性（Gate A 通过）≈0.45——这是实验问题，本提案用门控把它变成可裁决的二元判据。期望值 ≈ +0.15~0.35pt AP_s，近零开销（判分公式"收益×置信度÷开销"下分母→0）。

## (d) 开销预算（推理 ms 增量 + 零训练声明）

- **额外前向 FLOPs = 0**：z1-z3 的原料（`predictions_class/mask/boxes` 逐层 append，`multiscale_decoder.py:1001-1003`；`out["aux_outputs"/"query_embeddings"/"reference_points"]`，1031-1038）已无条件计算，现状被 `_inference_raw` 丢弃。
- 后处理增量（每图，GPU，全部静态形状）：
  - z1：8×200 sigmoid+归约 ≈ 3.2×10³ 元素 → <0.05ms（核启动开销为主）；
  - z2/z3：逐层掩码在 256² 画布二值化 + 相邻层 IoU。掩码已在显存；二值化 8×200×65,536 ≈ 1.0×10⁸ 元素访存 ≈ 400MB → ~0.3ms（可只对导出前 100 算，再减半）；
  - z4：box-IoU 全对 C(100,2)=4,950 次 4 标量运算 ≈ 2×10⁴ flops → <0.1ms（若需低分辨掩码 IoU 兜底：4,950×128² ≈ 8×10⁷ → ~0.1ms）；
  - z5（可选）：3×3 腐蚀/峰值 2-3 遍 100×1024² uint8 ≈ 内存带宽 2.4GB / ~2TB/s ≈ **≤1.5ms**；
  - 融合乘法：100×5 → ≈0。
- **合计：核心（z1-z4）≤ +0.5ms；含 z5 ≤ +2ms**（红线 +10ms，留 5 倍余量；当前基线 ~150ms 量级，目标 ~100ms 不受威胁）。显存增量：8×200×6 float ≈ 38KB + 256² 中间 uint8 ≈ 13MB（可复用）。
- **零训练声明**：封存 EMA 权重（`p5_runs/f1_seal_128k/best.pt`，SHA256 校验前后各一次）只读前向；拟合对象仅 ≤10 个标量常数（w/c/ε/τ_g/β），一次性离线，不回传任何梯度。

## (e) 正确性验证方案（全离线，封存权重/dets）

**Phase 0 重锚（CPU，0.5 天）**：在 128K 封存 dets（`p5_runs/f1_seal_128k/coco_instances_results.json`，3276 图）复算 r1 全部反事实与桶统计（脚本复用 `reviews_20260920/r1_score/analyze*.py`），消灭 112K↔128K 协议漂移（0.2925 vs 0.2700）。产出：本封存集的 base AP_s/AP_m/AP_l、桶 AP、AUC(0.753/0.490/0.654 的 128K 版)、14.05 重复度、keeper 排名分布——此后所有 Δ 以此为基。

**Phase 1 特征转储（GPU，~2h）**：`forward_inference_raw` 加 `dump_trajectory=True` 开关，对全 val 3276 用 best.pt 前向一次，额外落盘：全部 200 query 的 8 层 cls sigmoid、逐层掩码统计（质心/面积/相邻 IoU/z3）、逐层 box、导出 100 的 `top_scores`（cls）与 `mask_scores`（maskness）分量、z5 特征、bbox。≈200×(8×6)+100×8 float/图 ≈ 12KB/图，~40MB 总量。
- **Gate 0（正确性）**：rescore 关闭时，3276/3276 图 mask RLE 逐位相等、score float32 全等——不满足即实现有 bug，停止。

**Phase 2 离线融合与判读（CPU，1 天）**：常数在 300 图子集上 5 折拟合，其余 2976 图为测试集；最终裁决数字在**全 val 3276** 上用校准链协议（evaluate.py 标准协议、maxDets=100、EMA）产出。
- **Gate A（特征有效性，先验判死门）**：融合排序器对 TP-vs-hardFP 的 AUC，<64 桶 ≥ Phase0 基线+0.05（≈0.75→0.80）且 64-256 桶 ≥ +0.08（≈0.49→0.57）。不过 → **null 报告收场**（"导出端重打分在 128K 封存上无活口"，用数字关闭该分支，同样是本场交付物）。
- **Gate B（组纯度）**：box-IoU 重复组中 ≥98% 只含单一 GT（防相邻零件误并）；不达标升 τ_g 或砍 z4。
- **Gate C（采纳门，主判读数）**：ΔAP_s ≥ **+0.30pt** 且 ΔAP_m、ΔAP_l ≥ −0.10pt 且 ΔmAP ≥ −0.10pt、ΔF1@0.5 ≥ −0.5pt。同时报告 <64/64-256/256-1k 桶 AP、AR_s、<64 TP 分中位（0.29→?）与 hard-FP 分中位（0.063→?，**必须基本不动**——这是我们与被证伪抬分路线的分界线）。
- **Gate D（延迟）**：≥200 图实测新增后处理 p50 ≤1ms（核心）/≤3ms（含 z5）。
- 消融矩阵（一次跑齐）：z1-z3 排序臂单独 / z4 去重臂单独 / 合体 / z5+β 扫描（β∈{0,0.25,0.5}）/ 配额守卫臂（预留 k∈{10,20} 小 det 槽位，预期 ≤+0.09，出数封分支）。
- 总成本：≤2 人日 + ~2-4 GPU 时（3276×~150ms 前向 ≈8min 纯算，余为转储与重跑裕量）。

## (f) 与其它场的叠加兼容性

- **P3-a/P3-b（掩码质量/表示）**：正交。它们改 mask 本体，我们排序既有 det；掩码变好 → maskness 上限上移 → 桶中位锚定常数重拟合即可（仍零训练）。P3-b 若改画布/上采样，轨迹特征接口（逐层 box/cls/画布 IoU）不变。
- **P4-c（MAL 训练期分数成型）**：互补且部分重叠——P4-c 从根上修 cls 形成，本提案在其落地前榨干导出端存量；P4-c 生效后 Gate A 重跑（预期头寸缩小），由数字决定去留，不互斥。
- **P4-a（o2m 供给）/P4-b（种子 query/覆盖）**：训练期或 query 侧改动；本方案特征 per-query、在切割前计算，对任意 query 数/出生方式成立；query 变密时 z4 重复组结构更有用（喷洒更多）。
- **P3-c（EMA/退火收尾）**：权重替换后 Phase 1 重跑一次（~2h）即可迁移，常数重校准。

## 风险与自检

1. 轨迹信号可能无增量 AUC（<64 桶已 0.753）→ Gate A 判死，null 报告（成本低、关闭有价值）。
2. CQTR 是 2026-09 未评审新作（单源）→ 只作先例锚，全部上界以 r1 本地反事实为准。
3. 拟合过拟合 → ≤10 常数 + 5 折 + 留出测试，裁决只认全 val 数字。
4. 组误并伤害近邻双零件 → Gate B 纯度门 + τ_g 扫描。
5. 分数语义漂移 → 桶中位锚定 + F1@0.5 强制报告。
6. 与 (c) 系证伪家族的形式区分：g 依赖轨迹特征而非分数本身，桶内非单调——若 Phase 2 发现 g 退化为分数的单调函数（与 maskness 相关系数 >0.9），按证伪家族处理弃用。

## 实现挂点（供裁决后实验室执行）

- `experiments_archive/v317_next_stage_2026-07/source/magformer/models/magformer/gpu_postprocess.py`：`postprocess_tensors`（L156-226）后接 `trajectory_features(aux, ...)` + `apply_rescore(...)`；常数走 config（同 `inference_mask_threshold` 先例）。
- `.../magformer/arch.py`：`_inference_raw`（L1485-1593）与 `_inference_raw_gpu`（L1597-1695）把 `outputs["aux_outputs"]`（`multiscale_decoder.py:1031-1038` 已返回）传入上述函数；eager/GPU/CUDA-graph 三路径共享同一纯函数保证 bit-identical。
- 评估：evaluate.py 标准协议 + `--truncate-to-max-dets`（commit 4ebc733a）；GT `magformer_datasets/20260318_1K_32254/annotations/instances_val.validated.json`。
