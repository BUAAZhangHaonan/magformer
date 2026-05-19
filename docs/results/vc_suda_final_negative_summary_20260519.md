# VC-SUDA 最终负结果总结 - 2026-05-19

## 结论

复杂 VC-SUDA 流程在当前 `pseudo_real_512` 模拟真实任务里没有产生有效收益。这个结论不应该再被包装成“还差一点”的正结果。

1. MagFormer 的 32K 仿真 source 能力很强。R138 full 32K source val 达到 bbox AP `0.7424`、segm AP `0.7839`，而且这次是完整 source val，不是 R98 之前那种 subset 漏洞。
2. 目标域 target150 上，labeled-only/full RGB-D R114 已经给出很强锚点：val28 segm AP `0.321831`，remaining75 segm AP `0.392173`。
3. R118/R122 这类 VC-SUDA depth-boundary 和 pseudo-bank 复杂流程没有超过 R114。R122 final remaining75 segm AP `0.392074`、val28 segm AP `0.321671`，几乎等于 R114。

所以后续论文主张必须收缩。可以说 MagFormer 在这个任务上是强的。不能说当前 VC-SUDA 复杂训练流程、pseudo bank、boundary/prototype/modality 多 loss 给 target adaptation 带来了可见收益。

## 实验矩阵

| run | 目的 | 关键设置 | val28 segm AP | remaining75 segm AP | 结论 |
| --- | --- | --- | ---: | ---: | --- |
| R138 | full 32K source val | R98 source ckpt，全量 3276 source val，无 subset | n/a | n/a | source 能力强：bbox AP `0.7424`，segm AP `0.7839` |
| R114 | labeled-only/full RGB-D target150 formal | R113 warm-start，target150，RGB-D，2000 iter | `0.321831` | `0.392173` | 当前最重要 target150 锚点 |
| R118 | VC-SUDA pseudo-bank small-step | R114 warm，R117 pseudo bank，unsup weight `0.05`，100 iter gate | `0.321894` | `0.392568` | gate 失败；未达到设定成功线 |
| R122 final | VC-SUDA depth-boundary | R114 warm，depth-boundary `w=0.01`，300 iter final | `0.321671` | `0.392074` | 基本等于 R114，没有有效增益 |
| R139 | RGB-only target150 | R98 warm-start，关闭 depth backbone 和 fusion，2000 iter | `0.322550` | `0.388223` | 与 RGB-D 接近；val28 还略高 |
| R141 | RGB-D target150 matched control | 与 R139 同 R98 warm-start，打开 depth backbone 和 fusion | `0.319539` | `0.388543` | 相对 R139 没有 mask AP 增益 |
| R137 | RGB Mask2Former baseline | official Mask2Former RGB target150，fixed1024 replay | `0.253215` | `0.324278` | MagFormer target150 仍高约 6 AP，但优势不像来自 depth/fusion |

R139/R141 是更干净的 depth/fusion 对照。它们同用 R98 warm-start、同用 R114 target150 split、同跑 2000 iter。差异就是 R139 关闭 depth/fusion，R141 打开 depth/fusion。结果是 R141 val28 segm AP 比 R139 低 `0.003011`，remaining75 segm AP 只高 `0.000320`，这不支持“depth/fusion 带来有效 mask AP 增益”。

## 原始计划逐项复盘

### 1. source teacher / MagFormer source 能力

这部分成立。

- R138 full 32K source val：bbox AP `0.742428`，segm AP `0.783884`。
- 输出路径：`output/diagnostics/r138_magformer_32k_fullval_20260519/metrics.cocoeval.json`。
- R138 是完整 source val 任务，记录要求不允许 `--max-images`、topk subset、first300 或类似限制。R98 之前的 subset 漏洞已经补掉，不能再用旧 subset 数字代表 source 能力。

这说明 MagFormer 在仿真 source 上确实强。这个事实可以保留。

### 2. labeled-only target150 anchor

这部分也成立，而且它反过来压缩了 VC-SUDA 的空间。

- R114 target150 formal：val28 segm AP `0.321831`。
- R114 target150 formal：remaining75 segm AP `0.392173`。
- 输出文档：`docs/results/baseline_r114_magformer_r113warm_target150_balanced_20260518.md`。
- 指标日志：`output/diagnostics/r114_magformer_r113warm_target150_iter2000_val28_1024_backmap_topk200_20260518.eval.log` 和 `output/diagnostics/r114_magformer_r113warm_target150_iter2000_remaining75_1024_backmap_topk200_20260518.eval.log`。

这说明 target150 labeled-only 已经是强 baseline。任何 VC-SUDA claim 都必须超过它，而不是只超过较弱 RGB baseline。

### 3. pseudo bank / EMA teacher-student

目前没有证据支持。

R118 使用 R114 formal teacher 和 R117 passing pseudo bank，伪标签质量门槛看起来不差，但 100 iter gate 失败：

- val28 segm AP `0.321894`，低于成功线 `0.322`。
- remaining75 segm AP `0.392568`，低于成功线 `0.397`。
- full200 reference segm AP `0.520202`，低于成功线 `0.523`。
- 文档：`docs/results/vc_suda_r118_stage_c_smallstep_20260518.md`。

这不是“强正向信号”。它最多说明 pseudo bank 没有立刻把模型打崩。它没有证明 teacher-student 或 pseudo bank 给 target adaptation 带来了可靠收益。

### 4. boundary / prototype / modality 多 loss

目前没有证据支持，继续堆模块风险大。

R122 depth-boundary final 结果：

- remaining75 segm AP `0.392074561`。
- val28 segm AP `0.321670983`。
- R114 对应是 remaining75 `0.392173`、val28 `0.321831`。

差异只有 `-0.000098` 和 `-0.000160`。这在当前实验尺度下不是收益。R122 go/no-go 的 bucket 比较还给出 FAIL，tiny area 和 high100 的 boundary/best-IoU/R75 gate 多项失败。

原始计划里的 EMA teacher-student、pseudo bank、boundary/prototype/modality 复杂多 loss，目前都没有形成可写进论文的正证据。继续叠更多 loss 或 watcher 只会扩大调参空间，不会自动变成一个可信算法。

### 5. depth/fusion 分支

当前 target150 recipe 下没有可见贡献。

R139 RGB-only 和 R141 RGB-D 是严格同源对照：

- R139：R98 warm-start，target150，关闭 `model.magformer.depth_backbone.enabled` 和 `model.magformer.modality_fusion.enabled`。
- R141：同一个 R98 warm-start，同一个 target150 recipe，打开这两个开关。
- R139 val28/remaining75 segm AP：`0.322550` / `0.388223`。
- R141 val28/remaining75 segm AP：`0.319539` / `0.388543`。

代码审计结论也支持这个对照的解释：`magformer/models/magformer/arch.py` 会构建 depth backbone 和 fusion 模块，所以参数结构仍保留；但 forward 里用 `modality_fusion_enabled and depth_backbone_enabled` 决定是否调用 depth backbone 和 fusion。开关关闭时，`fused_features = rgb_features`，depth/fusion 前向被跳过。因此 R139 不是换了一个完全不同模型，而是在同一 MagFormer 结构内跳过 depth/fusion 路径。

## 为什么判定 VC-SUDA 无效

结论：在当前 `pseudo_real_512` 任务和 target150 recipe 下，VC-SUDA 无效。

1. 最强复杂流程没有超过简单 anchor。R122 final 和 R114 几乎完全相同，且略低。
2. 伪标签流程没有通过自设成功门。R118 在 val28、remaining75、full200 三个成功线下都失败。
3. depth/fusion 的 matched ablation 不支持 target mask AP 增益。R141 相对 R139 的 remaining75 只高 `0.000320`，val28 还低 `0.003011`。

这三个点合在一起，说明问题不只是某一次训练没调好。更可能的解释是：当前模拟真实任务中，target150 labeled-only 已经吸收了主要有效信号；复杂 VC-SUDA 组件没有提供额外可分辨的 target-domain 信息。

## 仍然成立的事实

- MagFormer 在 32K 仿真 source 上很强。R138 full 32K source val 的 bbox AP `0.7424`、segm AP `0.7839` 可以作为 source 能力证据。
- MagFormer target150 仍明显强于 official RGB Mask2Former target150 baseline。R137 remaining75 segm AP 约 `0.324278`，R114 是 `0.392173`，差距约 `0.067895`，也就是约 `6.8 AP`。
- 这个优势更像来自 MagFormer 主干、训练 recipe、query/decoder/分割头或整体实现差异，而不是来自当前能证明的 depth/fusion 分支。
- R114 target150 formal 是后续比较的主要 anchor。它比早期 25/50/100 target 设置更适合作为论文里 target labeled budget 的基线。

## 不能再声称的内容

- 不能再声称当前 VC-SUDA 的 EMA teacher-student 机制已经带来有效 target adaptation 收益。
- 不能再声称 pseudo bank 是有效增益来源。现在只有“没有明显崩坏”和“gate 未通过”，没有正收益。
- 不能再声称 boundary/prototype/modality 多 loss 提升了 target mask AP。R122 final 不支持这个说法。
- 不能再声称 depth/fusion 是 MagFormer target150 优势的主要来源。R139/R141 matched ablation 不支持。
- 不能把 R98 之前的 source subset 数字继续当作 full 32K source 能力证据。source 能力应引用 R138 fullval。
- 不能把“MagFormer 比 RGB Mask2Former 高约 6 AP”直接解释成“RGB-D/fusion 有效”。这个解释已经被 R139/R141 削弱。

## 后续建议

Recommend 收缩论文主张，alternative 继续做 VC-SUDA 但先重写问题定义。

推荐方向：

- 把论文主张收缩为“MagFormer 在 32K 仿真 source 和 pseudo-real target150 supervised adaptation 上很强”。
- 把 VC-SUDA 复杂流程写成负结果或探索性附录，不要放在主贡献里。
- 若保留 RGB-D 叙事，只能说“当前实现保留 RGB-D 能力，但在 target150 matched ablation 中没有证明 depth/fusion 的 mask AP 增益”。

替代方向：

- 如果还要救 VC-SUDA，先提出一个能超过 R114 的最小假设，再设计单因子实验。不要继续叠 EMA、pseudo bank、boundary、prototype、modality loss。
- 先证明一个组件能稳定超过 R114 至少 `+0.005` segm AP，再谈组合。否则组合实验只是在制造不可解释的自由度。

当前阶段最诚实的写法是：MagFormer strong，VC-SUDA negative，depth/fusion target gain not supported，paper claim should shrink.
