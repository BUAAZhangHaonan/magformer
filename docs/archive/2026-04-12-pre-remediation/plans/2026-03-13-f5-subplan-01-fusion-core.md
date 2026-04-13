# F5 Fusion Core Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 为 `ModalityFusionModule` 新增独立的 `prior_guided_cross_attn` 模式，并用确定性测试锁死其 prior-conditioned K/V 语义。

**Architecture:** 保留现有 `cross_attn` 行为不变，在融合模块中增加一条独立分支，使用 `guide = 1 + tanh(gate(prior_stack))` 对深度 K/V 做显式调制，再附加 `prior_proj(prior_stack)`，最后复用现有单层 MHA、残差和 LayerNorm。任何 prior 通道缺失都直接报错，避免 silent fallback。

**Tech Stack:** PyTorch, pytest.

---

## 涉及文件

- Modify: `magformer/models/magformer/fusion.py`
- Modify: `tests/test_light_fusion_modes.py`
- Create: `tests/test_prior_guided_cross_attn_fusion_mode.py`
- Optional modify: `magformer/models/magformer/arch.py`

## 步骤

1. 在 `tests/test_prior_guided_cross_attn_fusion_mode.py` 写失败测试，覆盖：
   - 只更新 `fuse_scales` 指定尺度
   - prior 通道缺失时报错
   - prior 改变时 attention 接收到的 K/V token 改变
   - 零初始化下输出与普通 `cross_attn` 数值接近
   - `valid-hole` 在 guidance 前生效
2. 运行：
   - `pytest -q tests/test_prior_guided_cross_attn_fusion_mode.py`
   - 预期失败，且失败点对应新增模式不存在或行为不符
3. 在 `magformer/models/magformer/fusion.py` 增加：
   - `prior_guided_cross_attn` 白名单
   - 独立的 module dict
   - `_apply_prior_guided_cross_attn`
   - prior 缺失时报错
4. 如确有必要，再最小化修改 `magformer/models/magformer/arch.py`，仅确保新 mode 可直通，不新增 public config 字段
5. 运行：
   - `pytest -q tests/test_prior_guided_cross_attn_fusion_mode.py tests/test_cross_attn_fusion_mode.py tests/test_light_fusion_modes.py`
6. 阅读本子计划，确认没有遗漏后提交

## 完成判据

- `cross_attn` 旧测试仍通过
- 新 mode 有独立测试文件并全部通过
- `prior_guided_cross_attn` 初始行为接近普通 `cross_attn`
- prior 缺失不会 silent fallback

## 提交信息

- `feat: add prior-guided cross-attention fusion mode`
