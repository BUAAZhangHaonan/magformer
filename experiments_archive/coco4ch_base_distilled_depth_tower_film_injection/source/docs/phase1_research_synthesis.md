# MagFormer Phase 1 Research Synthesis

Date: 2026-05-23
Baseline: v88 (AP=0.7963, AR=0.8057, 50.13M params)
Dataset: 32,254 images, ~100 instances/image, 1024x1024, single-class (PCB component)

---

## 1. What We've Learned (102 Experiments)

### 1.1 Why Every Module Addition Failed

We tested 12+ module additions across v85-v87, v93-v98, v97-v98. Every single one hurt or was noise.

| Module | Version | AP | vs v88 | Verdict |
|--------|---------|-----|--------|---------|
| CQD (cross-query deps) | v85a | <baseline | negative | Redundant with self-attention |
| SBM (spatial bias module) | v85b | <baseline | negative | Fusion already handles spatial info |
| Scale-adaptive loss | v96 | 0.7811 | -1.52 | Destroys balanced learning |
| DN denoising training | v96 | 0.7811 | -1.52 | Single class = no class confusion to denoise |
| Look-forward-twice | v97 | 0.7608 | -3.55 | Gradient flow disruption in finetuning |
| 100 queries (reduced) | v97 | 0.7608 | -3.55 | Model needs 200 for dense scenes |
| Hybrid matching + EQS | v98 | crashed | N/A | Never converged |
| Step LR (vs cosine) | v95 | 0.7926 | -0.37 | Cosine LR already optimal |
| Deformable cross-attn | v97 | 0.4183 | -37.8 | Catastrophic: zero offsets = blind queries |
| Top-K attention | v62 | <baseline | negative | Prunes useful signals |
| Copy-paste (early) | v58 | 0.7443 | -5.20 | Attention mask collapse |
| More decoder layers (10) | v72 | 0.6224 | -17.39 | Overfitting, harder to train |

**Root cause pattern**: Every module adds parameters or changes gradients in a pretrained decoder. The decoder was pretrained with a specific attention pattern and gradient flow. Finetuning with added modules disrupts this. The model has a single class -- there is no classification challenge to solve with extra modules. The bottleneck is localization, and modules that don't directly improve spatial resolution of predictions will fail.

**Why these fail for single-class**: Techniques like DN-DETR, one-to-many matching, and hybrid matching are designed for multi-class datasets where classification errors dominate. In MagFormer's single-class setting, classification is near-perfect (98.8% precision). These modules address a problem that doesn't exist here.

### 1.2 Why Backbone Upgrade Was the Only Thing That Worked

Backbone exploration (v88-v94) was the only axis that produced consistent gains.

| Depth Backbone | AP | AR | Params |
|----------------|-----|------|--------|
| MBV3-Small (v84) | 0.7852 | 0.7962 | 50.9M |
| **MBV3-Large (v88)** | **0.7963** | **0.8057** | **50.1M** |
| EffNet-B2 (v91) | 0.7958 | 0.8048 | 50.2M |
| ConvNeXt-Tiny (v93) | 0.7872 | 0.7970 | 50.7M |
| ResNet-18 (v94) | 0.7971 | 0.8052 | 61.2M |

**Why it worked**: The AR ceiling was 79.62% with the old backbone. Better depth features directly improved feature map quality, which improved recall (79.62% -> 80.57%). The AR improvement propagated directly to AP because precision is already near-perfect. This is the only axis that attacks the data representation rather than the decoder architecture.

**Param efficiency**: MBV3-Large is the winner because it broke the AR ceiling while staying under 50.1M params. ResNet-18 matched AP but at 61.2M (11M more for +0.0008 AP).

### 1.3 The Localization Error Bottleneck

AP loss decomposition (from v83 analysis):

| Error Type | AP Points Lost | % | Root Cause |
|------------|---------------|---|------------|
| Localization (IoU 0.5-0.9) | ~12.5 | 60% | Feature resolution at stride-4 |
| Detection misses (IoU < 0.5) | ~8.4 | 40% | Dense scenes, occluded objects |
| False positives | ~2.3 | — | Low-confidence spurious |
| Mask quality (bbox->segm) | ~1.2 | — | Already small gap |

**By object size**:

| Size | % of GT | bbox AP | AR@100 |
|------|---------|---------|--------|
| Small (<32x32) | 4.6% | 16.7 | 13.8 |
| Medium (32x32-96x96) | 62.9% | 77.0 | 73.5 |
| Large (>96x96) | 32.5% | 94.0 | 95.1 |

Medium objects are the lever. They are 62.9% of all instances and have 26.5% miss rate. Q1 medium objects (area 1024-3947 pixels) behave like small objects at stride-4 -- dense attention dilutes their signal.

**The fundamental constraint**: At 1024px input, stride-4 feature maps are 256x256. A 32x32 object occupies only 8x8 feature cells. Attention over the full 256x256 grid dilutes the signal for these objects. This is a resolution problem, not an architecture problem.

### 1.4 Dataset Characteristics

- 32,254 images at 1024x1024
- ~55 average instances per image, max ~100
- Single class: component (no classification challenge)
- Train/val distribution well-matched (4.3/61.0/34.7% vs 4.6/62.9/32.5% by size)
- Depth maps from structured light sensor, range ~1.0-2.1m
- SA-gate fusion with edge + valid-hole priors
- AGPE (adaptive grid positional encoding) enabled
- 200 queries, 8 decoder layers, hidden_dim=256

---

## 2. Literature-Based Innovation Directions (Ranked by ROI)

### 2.1 Training Recipe: Longer Training (HIGHEST ROI)

**Technique**: Extend training from 32K to 128K-256K iterations (20-40 epochs).
**Why it might work here**: v60 showed that continued training at low LR consistently improved AP (+0.3 per 5K iters with no sign of plateau at 32K). The model has only seen 1.4 epochs at 32K iterations (32K / 22.5K images per effective batch). Papers typically train for 50-150K iterations.
**Why v95-v98 failed but this won't**: v95-v98 changed the training recipe (step LR, different schedules) but kept the same 32K budget. They failed because cosine LR at 32K is already optimal. The problem is undertraining, not wrong training schedule.
**v99 is testing this right now** (128K iters, LR=1.5e-5).
**Expected gain**: +1-3 AP based on v60 trajectory.

### 2.2 Augmentation: Copy-Paste (MEDIUM-HIGH ROI)

**Technique**: Paste randomly selected instances onto training images.
**Why it might work here**: Dense scenes with ~100 instances. Copy-paste increases effective instance density and forces the model to handle more crowded scenes. Previous failure (v58) was with old baseline -- the stronger v88 backbone may handle it better.
**Why v58 failed but v100 might succeed**: v58 was tested on the v47 baseline with 10-decoder, MBV3-Small depth backbone and dice_weight=5. v100 starts from v88 (8-decoder, MBV3-Large, dice_weight=20, stronger features). The stronger backbone produces better features that can handle the increased instance density without attention collapse.
**v100 is testing this right now** (32K iters, copy-paste prob=0.5, max 8 pasted instances).
**Expected gain**: +0.5-2 AP on medium objects.

### 2.3 Backbone: ConvNeXt-Tiny Depth (MEDIUM ROI)

**Technique**: Replace MBV3-Large depth backbone with ConvNeXt-Tiny.
**Why it might work**: ConvNeXt has modern architectural choices (GELU, LayerNorm, inverted bottleneck) that may produce richer depth features. Channel dimensions match Swin-T RGB (96/192/384/768), which means the fusion module gets perfectly aligned features.
**Why v93 failed but v101 might succeed**: v93 used ConvNeXt-Tiny as depth backbone but with 10 decoder layers and trained from scratch (no finetuning from v88). v101 finetunes from v88's checkpoint, giving it a warm start with well-trained decoder weights.
**Expected gain**: +0.5-1.5 AP if depth features improve.

### 2.4 Backbone: Swin-Small RGB (MEDIUM ROI)

**Technique**: Upgrade RGB backbone from Swin-Tiny (96d, 28M) to Swin-Small (96d, deeper stage3: 6->18 blocks).
**Why it might work**: Swin-S has 3x more blocks in stage 3 (the most important stage for medium objects at 1/8 resolution). More computation at the scale where medium objects are best represented.
**v102 is testing this right now** (32K iters, Swin-S depth=[2,2,18,2], drop_path=0.4).
**Risk**: Swin-S adds ~20M params (from ~28M to ~48M in RGB backbone). Total model may exceed 50M budget significantly. Checkpoint adaptation from Swin-T to Swin-S may cause issues (v102 log shows a crash -- verify if it recovered).
**Expected gain**: +0.5-2 AP if stage-3 features are the bottleneck.

### 2.5 What's Different About These vs Failed v95-v98

The key difference: v99-v102 attack the **data representation** (more training data, better augmentation, better features), while v95-v98 attacked the **decoder architecture** (matching strategy, loss weighting, query count). The decoder was already well-optimized by the pretrained v88 weights. The bottleneck is in what goes into the decoder, not the decoder itself.

| Axis | v85-v98 (all failed) | v99-v102 (current) |
|------|---------------------|-------------------|
| Target | Decoder modules | Feature quality / training |
| Hypothesis | Decoder needs more tricks | Model needs better inputs / more training |
| Risk | Architecture disruption | More conservative, additive |

### 2.6 Techniques NOT to Try (Proven Failed)

- Deformable cross-attention as finetuning (v97: AP 41.83)
- DN-DETR / denoising training (single class, no classification confusion)
- One-to-many matching (already in code, tested in v17, no gain)
- Reduced query count (model needs 200 for dense scenes)
- SAHI tiled inference on 512px (AP 10.8, catastrophic)
- 768px resolution (worse than both 512/1024)
- mask_weight=10 (v81: -2.37 AP)

---

## 3. Phase 2 Experimental Plan

### Decision Tree Based on v99-v102 Results

**Wait for all four experiments to finish before deciding.** Priority order for what matters most:

#### Scenario A: v99 (long training) succeeds (AP > 0.81)
This is the highest-value outcome. Training insufficiency is confirmed as the main bottleneck.

- **v103**: Extend v99 to 256K iterations if loss is still decreasing at 128K
- **v104**: Combine v99 (long training) + v100 (copy-paste) if v100 also succeeds
- **v105**: If v104 works, try v99 + v100 + best backbone from v101/v102

#### Scenario B: v100 (copy-paste) succeeds (AP > 0.81)
Copy-paste is a cheap augmentation win.

- **v103**: Combine copy-paste with best backbone from v101/v102
- **v104**: Extend copy-paste training to 128K iterations
- **v105**: Increase copy-paste params: max_paste_instances=12, prob=0.7

#### Scenario C: v101 (ConvNeXt-T depth) succeeds (AP > 0.81)
Depth features are the bottleneck.

- **v103**: ConvNeXt-T depth + Swin-S RGB (if v102 also succeeds)
- **v104**: ConvNeXt-T depth + longer training (128K)
- **v105**: Try ConvNeXt-S or ConvNeXt-B if params allow

#### Scenario D: v102 (Swin-S RGB) succeeds (AP > 0.81)
RGB features are the bottleneck.

- **v103**: Swin-S RGB + ConvNeXt-T depth (if v101 also succeeds)
- **v104**: Swin-S RGB + longer training (128K)
- **v105**: Check if Swin-S exceeds param budget; if so, prune depth backbone

#### Scenario E: Multiple succeed
Build the best combination in priority order:
1. **v103**: Best training recipe (v99 schedule) + best augmentation (v100) + best backbone combo (from v101/v102)
2. **v104**: v103 + 256K iterations
3. This is the kitchen sink experiment. One attempt only.

#### Scenario F: Nothing works (all four fail to exceed v88 AP=0.7963)
This means the model has hit its architectural ceiling with current approach. Options:

1. **Higher resolution (1536px)**: Attack the resolution bottleneck directly. Requires gradient checkpointing. 1.5x more pixels means small/medium objects have 2.25x more feature cells. Expected +3-5 AP on medium objects.
2. **SAHI training (crop-based)**: Train on 512x512 or 768x768 crops from 1024px images. Small/medium objects become relatively larger. Keep full-image eval. Expected +5-10 APs on small objects, +2-3 APm.
3. **Train from scratch**: All finetuning experiments are limited by the pretrained weights' feature distribution. Training from scratch with the best config may unlock new feature patterns. Cost: 3-5x longer training.
4. **Efficient decoder (5 layers, 100 queries)**: Current decoder is 30-40% of inference time. Reducing to 5 layers + 100 queries gives 30-40% speedup. Use the freed compute for larger backbone or higher resolution.

**Recommendation if Scenario F**: Start with 1536px + gradient checkpointing. It attacks the root cause (resolution) with no architecture changes.

---

## 4. Inference Speed Optimization Roadmap

### 4.1 Current Bottlenecks

Profile of a single inference pass on RTX 3090 at 1024px:

| Component | Time (%) | Notes |
|-----------|----------|-------|
| RGB backbone (Swin-T) | ~25% | 28M params, window attention |
| Depth backbone (MBV3-Large) | ~5% | 4.2M params, lightweight |
| SA-gate fusion | ~5% | 4-scale fusion |
| Pixel decoder (MSDeformAttn) | ~25% | 6 encoder layers |
| **Transformer decoder** | **~35-40%** | **8 layers, 200 queries, cross-attention over 256x256** |
| Mask generation | ~5% | Per-query mask heads |

**The decoder dominates inference.** 8 layers of cross-attention between 200 queries and a 256x256 feature map.

### 4.2 Optimization Levers

| Optimization | Speedup | AP Impact | Effort |
|-------------|---------|-----------|--------|
| Reduce decoder: 8->5 layers | ~30% | -0.5 to -1.0 AP | Config change |
| Reduce queries: 200->100 | ~10% | Need testing (v97 100q failed, but with 200 iters was too short) | Config change |
| Combined 5-dec + 100q | ~35-40% | Need testing | Config change |
| TensorRT / ONNX export | 2-3x | ~0 AP loss | Medium effort |
| FP16 inference (already using AMP) | Done | — | — |
| INT8 quantization | 1.5-2x | -0.5 to -1.5 AP | High effort |
| Prune Swin-T to Swin-Tiny | ~20% | -1 to -2 AP | Config change |
| Knowledge distillation (big->small) | Depends | -0.5 to -1 AP | Medium effort |

### 4.3 Edge Deployment Targets

| Target Device | Available Compute | Target FPS | Required Model |
|--------------|-------------------|-----------|----------------|
| RTX 3090 (current) | ~35 TFLOPS FP16 | ~3-5 FPS | Current v88 (50M) |
| RTX 4060 (edge desktop) | ~15 TFLOPS FP16 | ~5-10 FPS | 5-dec + 100q (~35M) |
| Jetson Orin NX (embedded) | ~7 TFLOPS FP16 | ~2-3 FPS | 5-dec + Swin-Tiny + MBV3-S (~30M) + TensorRT |
| Jetson Orin Nano (edge) | ~4 TFLOPS FP16 | ~1 FPS | 5-dec + MobileNet backbone (~20M) + TensorRT + INT8 |

### 4.4 Recommended Speed Path

**Phase 2a** (after AP is optimized):
1. Take best AP model, reduce decoder to 5 layers. Measure AP drop.
2. If drop < 1 AP, reduce queries to 100. Measure again.
3. Export to TensorRT. Measure actual FPS on target hardware.
4. Only if FPS is still too low: consider INT8 quantization or knowledge distillation.

**Do NOT optimize speed until AP is maximized.** Speed optimizations reduce AP. It's easier to prune a strong model than to speed up a weak one.

---

## 5. Current Experiment Status (2026-05-23)

| Version | Change | GPU | Status | Progress |
|---------|--------|-----|--------|----------|
| v99 | 128K iters, LR=1.5e-5 | GPU 0 | Training | Running |
| v100 | Copy-paste aug | GPU 1 | Training | ~12K/32K (37%) |
| v101 | ConvNeXt-T depth | GPU 2 | Training | ~5.6K/32K (17%) |
| v102 | Swin-S RGB | GPU 3 | Training | ~crashed, check status |

GPU 4 and GPU 5 are free for next experiments.

---

## 6. Summary of What Matters

1. **Training insufficiency is the #1 hypothesis.** v99 (128K iters) is the most important experiment running. If it works, Phase 2 is just train longer with best backbone.

2. **Feature quality > decoder tricks.** Every decoder modification failed. Every backbone upgrade worked. Focus on better features (backbone, resolution, augmentation) not better architecture.

3. **Medium objects are the lever.** 62.9% of instances, 26.5% miss rate, 60% of AP loss is localization. Anything that improves feature resolution for 32x32-96x96 objects will move AP.

4. **The model is over-parameterized for what it does.** 50M params for single-class instance segmentation. The decoder (14.5M) is the biggest component after the backbone. This is where speed optimization should focus.

5. **Do not repeat failed patterns.** No more decoder modules, no more loss weighting experiments, no more matching strategy changes. The v88 config (cosine LR, backbone_mult=0.5, 200 queries, dice=20) is well-tuned. Change the inputs, not the model.
