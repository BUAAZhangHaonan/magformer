import pytest
import torch

from tools.diagnose_vc_suda_signals import _aggregate, build_arg_parser, summarize_loss_dict


def test_summarize_loss_dict_keeps_only_scalar_tensors():
    losses = {
        "total_loss": torch.tensor(3.5),
        "loss_vector": torch.tensor([1.0, 2.0]),
        "pseudo_diagnostics": {"matched_query_count": 1},
        "plain_float": 7.0,
    }

    summary = summarize_loss_dict(losses)

    assert summary == {"total_loss": pytest.approx(3.5)}


def test_signal_diagnostic_cli_requires_output_json_and_defaults_to_two_batches():
    parser = build_arg_parser()

    args = parser.parse_args(
        [
            "--config",
            "configs/vc_suda_stage_c_r12_32k_source_r8b_ckpt999_continue_1024_teacher8499.yaml",
            "--output-json",
            "output/diagnostics/example.json",
        ]
    )

    assert args.max_batches == 2
    assert args.high_score_thresholds == [0.7, 0.9]
    assert args.exterior_ring_radius == 2
    assert args.output_json == "output/diagnostics/example.json"


def test_signal_diagnostic_aggregate_includes_exterior_ring_fields():
    records = [
        {
            "source": {"total_loss": 1.0},
            "target_labeled": {"total_loss_raw": 2.0, "total_loss_weighted": 2.0},
            "pseudo": {
                "total_loss_raw": 3.0,
                "total_loss_weighted": 0.0,
                "scored_count": 1,
                "diagnostics": {
                    "image_count": 1,
                    "kept_pseudo_count": 1,
                    "matched_query_count": 1,
                    "unmatched_query_count": 0,
                    "unmatched_high_score_counts": {"0.700": 0, "0.900": 0},
                    "unmatched_high_score_scores": {"0.700": [], "0.900": []},
                    "unmatched_high_score_max_ious": {"0.700": [], "0.900": []},
                    "matched_exterior_ring": {
                        "radius": 2,
                        "matched_count": 1,
                        "gt_density_bucket_supported": False,
                        "exterior_ring_prob_values": [0.8],
                        "interior_prob_values": [0.9],
                        "background_far_prob_values": [0.1],
                        "pred_target_area_ratio_values": [1.5],
                        "ring_pixel_count_values": [12.0],
                    },
                },
            },
        }
    ]

    aggregate = _aggregate(records, (0.7, 0.9))

    ring = aggregate["matched_exterior_ring"]
    assert ring["radius"] == 2
    assert ring["matched_count"] == 1
    assert ring["gt_density_bucket_supported"] is False
    assert ring["exterior_ring_prob_distribution"]["mean"] == pytest.approx(0.8)
    assert ring["interior_prob_distribution"]["median"] == pytest.approx(0.9)
    assert ring["background_far_prob_distribution"]["mean"] == pytest.approx(0.1)
    assert ring["pred_target_area_ratio_distribution"]["p90"] == pytest.approx(1.5)
    assert ring["ring_pixel_count_distribution"]["count"] == 1
