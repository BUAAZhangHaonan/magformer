import pytest
import torch

from tools.diagnose_vc_suda_signals import build_arg_parser, summarize_loss_dict


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
    assert args.output_json == "output/diagnostics/example.json"
