from argparse import Namespace

from magformer.config import load_config
from magformer.config.loader import load_yaml_file, save_yaml_file
import tools.evaluate_1024_backmap as eval_1024


def test_eval_runtime_config_loads_target_unlabeled_ann_as_eval_data(tmp_path):
    args = Namespace(
        dataset_root="magformer_datasets/pseudo_real_512",
        ann="annotations/instances_target_unlabeled_r114_balanced_minus125.json",
        split="train",
        image_size=1024,
        weights="output/vc_suda/r122_depth_boundary_w001_r114warm_pseudo300/checkpoint_iter_0000099.pth",
        output_dir=str(tmp_path),
        num_workers=0,
        iou_types=["bbox", "segm"],
    )
    base_config = load_yaml_file("configs/baseline_vc_suda_r122_depth_boundary_w001_pseudo300.yaml")

    assert hasattr(eval_1024, "build_eval_runtime_config")
    cfg_dict = eval_1024.build_eval_runtime_config(base_config, args, output_dir=tmp_path)
    cfg_path = tmp_path / "eval_1024_runtime.yaml"
    save_yaml_file(cfg_dict, str(cfg_path))
    cfg = load_config(str(cfg_path))

    assert cfg.data.val_ann == args.ann
    assert cfg.data.val_split == "train"
    assert cfg.vc_suda.enabled is False
    assert cfg.vc_suda.stage == "A"
    assert cfg.vc_suda.target_unlabeled_ann is None
