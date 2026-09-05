# DPTD - depth trunk + distilled RGB side-path (old code name: "DPTD", E24 fusion arms)

Mirror of the winning depth_only recipe with roles swapped: the Swin trunk is
fed the z-scored depth channel replicated 3x; a distilled RGB tower
(mask-proxy pretrained) injects RGB texture per stage via zero-init FiLM.
Registered as D2SwinTransformerDPTD in dptd_backbone.py.

Provenance: 6401:/home/hdd1/wanghaoran/magformer/d2m2f_src (unversioned
working tree, 2026-08). Full results in docs/BASELINE_RESULTS.md; weights in
archive_20260906/staging_6401/.../E24-fusion-6401-20260821/.

Old-name mapping: DPTD=depth_trunk_distilled_rgb_sidepath,
hybrid_ptd=rgb_trunk_distilled_depth_sidepath, concat4ch=naive 4-channel
concat, E23=m2f_47m_rgb_concat_finetune (fullval 0.9069).
