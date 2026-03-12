# 2026-03-12 RGB-D Fusion Method Survey

## Goal

Identify at least 5 RGB-D fusion methods from literature and open-source code that are more defensible than the current `direct_add / gated_add / film / cross_attn(res3)` batch, while still fitting the lightweight design direction.

## Candidate Methods

### 1. ESANet-style asymmetric RGB-D fusion

- Paper: `ESANet: Efficient RGB-D Semantic Segmentation for Indoor Scene Analysis`
- Code: `https://github.com/TUI-NICR/ESANet`
- Core idea:
  - keep an asymmetric encoder budget between RGB and depth
  - use lightweight fusion blocks at multiple scales
  - optimize explicitly for strong speed / accuracy trade-off
- Why it matters here:
  - this is the closest published reference to the current lightweight objective
  - it is specifically built around efficient RGB-D segmentation
- Likely adaptation:
  - convert the current single-scale fusion block into an ESANet-style lightweight context fusion block at `res3`

### 2. CMX-style cross-modal token mixing

- Paper: `CMX: Cross-Modal Fusion for RGB-X Semantic Segmentation with Transformers`
- Paper URL: `https://arxiv.org/abs/2203.04838`
- Code: `https://github.com/huaaaliu/RGBX_Semantic_Segmentation`
- Core idea:
  - use cross-modal token interaction without keeping two fully independent semantic trunks to the end
  - mix RGB and geometric tokens through structured cross-modal blocks
- Why it matters here:
  - this is a stronger reference than the current ad hoc `cross_attn(res3)` implementation
  - it provides a published transformer fusion design rather than generic attention dropped into a CNN pipeline
- Likely adaptation:
  - keep fusion at `res3`, but replace the current plain attention block with a CMX-style compact cross-modal mixer

### 3. SA-Gate-style separation-and-aggregation gate

- Paper family:
  - `Bi-directional Cross-Modality Feature Propagation with Separation-and-Aggregation Gate for RGB-D Semantic Segmentation`
- Code collection:
  - `https://github.com/charlesCXK/RGBD_Semantic_Segmentation_PyTorch`
- Core idea:
  - explicitly separate shared and modality-specific components
  - aggregate them through a learned gate instead of naive add or concat
- Why it matters here:
  - the current direct/gated/film variants do not explicitly model shared-vs-specific decomposition
  - this is a stronger inductive bias for RGB texture vs depth geometry
- Likely adaptation:
  - replace scalar mask gating with dual branch `shared / specific` decomposition at `res3`

### 4. ACNet-style complementary channel attention fusion

- Paper: `ACNet: Attention Based Network to Exploit Complementary Features for RGBD Semantic Segmentation`
- Paper URL: `https://arxiv.org/abs/1905.10089`
- Code URL referenced in paper: `https://github.com/anheidelonghu/ACNet`
- Core idea:
  - learn complementary channel weighting between RGB and depth features
  - suppress redundant channels and enhance cross-modal complementarity
- Why it matters here:
  - current fusion variants barely exploit channel-level modality complementarity
  - this is cheaper than transformer attention and more targeted than direct add
- Likely adaptation:
  - add a low-cost channel attention fusion head at `res3` using pooled RGB/depth statistics

### 5. RDFNet / residual fusion block family

- Paper family:
  - `RDFNet` / `RFBNet` style residual RGB-D fusion blocks
- Reference paper URL:
  - `https://arxiv.org/abs/1907.00135`
- Code collection containing RGB-D baselines:
  - `https://github.com/charlesCXK/RGBD_Semantic_Segmentation_PyTorch`
- Core idea:
  - use residual multimodal fusion blocks rather than plain summation
  - preserve backbone features while injecting depth features through structured residual transforms
- Why it matters here:
  - the current best result being direct add suggests the model benefits from low-disruption fusion
  - a residual fusion block is a more principled next step than full cross-attention
- Likely adaptation:
  - add a lightweight residual fusion bottleneck with depth-conditioned modulation

### 6. DFormer-style depth-aware transformer fusion

- Paper family:
  - `DFormer` / depth-aware transformer fusion for RGB-D segmentation
- Code: `https://github.com/VCIP-RGBD/DFormer`
- Core idea:
  - encode depth as a stronger geometric signal and fuse it through dedicated depth-aware attention blocks
  - avoid treating depth as a plain residual side channel
- Why it matters here:
  - this is a stronger modern reference for structured RGB-D interaction than raw `direct_add`
  - it is useful as an upper-complexity reference when choosing what not to implement in the lightweight line
- Likely adaptation:
  - distill the depth-aware attention idea into a low-rank or local attention gate at `res3`

## Shortlist For Next Implementation

### Priority 1

- `ACNet-style complementary channel attention`
- `SA-Gate-style shared/specific gated fusion`
- `CMX-style compact cross-modal mixer`

Reason:
- all 3 are lighter than transformer-style cross-attention
- all 3 are more structured than raw `direct_add`
- all 3 fit the `res3`-only lightweight branch constraint

### Priority 2

- `ESANet-style asymmetric context fusion`
- `RDFNet-style residual fusion block`
- `DFormer-style depth-aware attention`

Reason:
- both are strong references
- both are better used once the next low-cost gating family is in place

## Current Interpretation

- The current `cross_attn(res3)` underperformed because the available depth signal is likely too weak/noisy for generic token attention to help under the current training budget.
- The current `direct_add` result suggests the model mainly benefits from:
  - a better geometric prior path
  - low-disruption feature injection
- The next fusion family should therefore bias toward:
  - lightweight attention
  - channel/spatial complementarity
  - residual or gated injection
  - minimal semantic competition with RGB
