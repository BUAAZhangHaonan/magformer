from __future__ import annotations

import importlib
import os
import sys
import tempfile
from contextlib import contextmanager, nullcontext
from datetime import timedelta
from pathlib import Path
from typing import Dict

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
repo_root_str = str(REPO_ROOT)
while repo_root_str in sys.path:
    sys.path.remove(repo_root_str)
sys.path.insert(0, repo_root_str)

stateful_module = importlib.import_module("magformer.data.stateful")
arch_module = importlib.import_module("magformer.models.magformer.arch")

for module in (stateful_module, arch_module):
    module_path = Path(module.__file__).resolve()
    assert module_path.is_relative_to(
        REPO_ROOT
    ), f"spawned test imported {module.__name__} outside repository: {module_path}"

EpochStatefulDistributedSampler = stateful_module.EpochStatefulDistributedSampler
MagFormerArch = arch_module.MagFormerArch


def _new_extractor(
    rgb_backbone: torch.nn.Module,
    depth_backbone: torch.nn.Module,
    fusion: torch.nn.Module,
) -> MagFormerArch:
    extractor = MagFormerArch.__new__(MagFormerArch)
    torch.nn.Module.__init__(extractor)
    extractor.rgb_backbone = rgb_backbone
    extractor.depth_backbone = depth_backbone
    extractor.fusion = fusion
    extractor.depth_only = False
    extractor.modality_fusion_enabled = True
    extractor.depth_backbone_enabled = True
    extractor.register_buffer(
        "pixel_mean",
        torch.tensor([0.25, 0.5, 0.75], dtype=torch.float32).view(3, 1, 1),
        persistent=False,
    )
    extractor.register_buffer(
        "pixel_std",
        torch.tensor([0.5, 0.75, 1.0], dtype=torch.float32).view(3, 1, 1),
        persistent=False,
    )
    return extractor


class _FakeTensor:
    is_cuda = True

    def __init__(self, name: str, events: list[str], device: torch.device) -> None:
        self.name = name
        self.events = events
        self.device = device

    def __sub__(self, other) -> "_FakeTensor":
        del other
        return _FakeTensor("images_norm", self.events, self.device)

    def __truediv__(self, other) -> "_FakeTensor":
        del other
        return self

    def record_stream(self, stream) -> None:
        self.events.append(f"{self.name}.record({stream.name})")


class _FakeStream:
    def __init__(self, name: str, events: list[str]) -> None:
        self.name = name
        self.events = events

    def wait_stream(self, stream) -> None:
        self.events.append(f"{self.name}.wait({stream.name})")


class _EventBackbone(torch.nn.Module):
    def __init__(self, name: str, events: list[str], active_stream: Dict[str, _FakeStream]) -> None:
        super().__init__()
        self.name = name
        self.events = events
        self.active_stream = active_stream

    def forward(self, value: _FakeTensor):
        self.events.append(f"{self.name}.work({self.active_stream['value'].name})")
        return {"res2": value}


class _EventFusion(torch.nn.Module):
    def __init__(self, events: list[str], active_stream: Dict[str, _FakeStream]) -> None:
        super().__init__()
        self.events = events
        self.active_stream = active_stream

    def forward(self, image_features, depth_features, **kwargs):
        del depth_features, kwargs
        self.events.append(f"fusion.work({self.active_stream['value'].name})")
        return image_features, {}, {}


def test_dual_backbone_cuda_stream_order_and_allocator_lifetime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    producer = _FakeStream("producer", events)
    rgb_stream = _FakeStream("rgb", events)
    depth_stream = _FakeStream("depth", events)
    active_stream = {"value": producer}
    created_streams = iter((rgb_stream, depth_stream))
    requested_devices: list[torch.device] = []
    fake_device = torch.device("cuda:7")

    def current_stream(*, device):
        requested_devices.append(device)
        return producer

    def new_stream(*, device):
        requested_devices.append(device)
        return next(created_streams)

    monkeypatch.setattr(torch.cuda, "current_stream", current_stream)
    monkeypatch.setattr(torch.cuda, "Stream", new_stream)

    @contextmanager
    def use_stream(stream):
        previous = active_stream["value"]
        active_stream["value"] = stream
        try:
            yield
        finally:
            active_stream["value"] = previous

    monkeypatch.setattr(torch.cuda, "stream", use_stream)
    extractor = _new_extractor(
        _EventBackbone("rgb", events, active_stream),
        _EventBackbone("depth", events, active_stream),
        _EventFusion(events, active_stream),
    )

    fused, confidence, losses = extractor._extract_dual_features(
        _FakeTensor("images", events, fake_device),
        _FakeTensor("depths", events, fake_device),
    )

    assert list(fused) == ["res2"]
    assert confidence == {}
    assert losses == {}
    assert requested_devices == [fake_device, fake_device, fake_device]
    assert events == [
        "rgb.wait(producer)",
        "depth.wait(producer)",
        "rgb.work(rgb)",
        "depth.work(depth)",
        "producer.wait(rgb)",
        "producer.wait(depth)",
        "images_norm.record(producer)",
        "depths.record(producer)",
        "fusion.work(producer)",
    ]


class _ConvBackbone(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 8) -> None:
        super().__init__()
        self.projection = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, value: torch.Tensor):
        return {"res2": self.projection(value)}


class _AddFusion(torch.nn.Module):
    def forward(self, image_features, depth_features, **kwargs):
        del kwargs
        return {"res2": image_features["res2"] + depth_features["res2"]}, {}, {}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.skipif(not hasattr(torch.cuda, "_sleep"), reason="CUDA sleep is required")
def test_dual_backbone_cuda_stream_stress_matches_ordered_reference() -> None:
    device = torch.device("cuda:0")
    extractor = _new_extractor(_ConvBackbone(3), _ConvBackbone(1), _AddFusion()).to(device)
    extractor.train()

    with torch.no_grad():
        for step in range(16):
            images = torch.empty((2, 3, 48, 48), device=device)
            depths = torch.empty((2, 1, 48, 48), device=device)
            torch.cuda._sleep(2_000_000)
            images.fill_(0.03125 * (step + 1))
            depths.fill_(0.015625 * (step + 1))

            with torch.amp.autocast("cuda", dtype=torch.float16):
                fused, _, _ = extractor._extract_dual_features(images, depths)
                expected_rgb = extractor.rgb_backbone(
                    (images - extractor.pixel_mean) / extractor.pixel_std
                )["res2"]
                expected_depth = extractor.depth_backbone(depths)["res2"]
                expected = expected_rgb + expected_depth

            torch.testing.assert_close(fused["res2"], expected, rtol=1.0e-3, atol=1.0e-3)
            assert torch.isfinite(fused["res2"]).all()


class _ShortDualLossModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.extractor = _new_extractor(_ConvBackbone(3), _ConvBackbone(1), _AddFusion())
        self.classifier = torch.nn.Linear(8, 2)

    def forward(self, images: torch.Tensor, depths: torch.Tensor) -> torch.Tensor:
        fused, _, _ = self.extractor._extract_dual_features(images, depths)
        pooled = fused["res2"].mean(dim=(2, 3))
        return self.classifier(pooled)


def _two_rank_finite_worker(rank: int, world_size: int, init_method: str) -> None:
    torch.cuda.set_device(rank)
    torch.cuda.set_per_process_memory_fraction(0.20, rank)
    device = torch.device(f"cuda:{rank}")
    dist.init_process_group(
        backend="nccl",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60),
    )
    try:
        torch.manual_seed(42)
        model = _ShortDualLossModel().to(device)
        ddp = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[rank],
            broadcast_buffers=True,
        )
        optimizer = torch.optim.SGD(ddp.parameters(), lr=1.0e-3)
        scaler = torch.amp.GradScaler("cuda", init_scale=64.0)
        optimizer.zero_grad(set_to_none=True)

        sampler = EpochStatefulDistributedSampler(
            range(25_654),
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=42,
        )
        sampler.set_epoch(0)
        iterator = iter(sampler)
        sample_indices = [next(iterator), next(iterator)]
        expected = [3_472, 1_518] if rank == 0 else [16_335, 1_603]
        assert sample_indices == expected
        print(f"DDP_RANK={rank} SAMPLE_INDICES={sample_indices}", flush=True)

        all_finite = True
        loss_values = []
        for micro_step, sample_index in enumerate(sample_indices):
            images = torch.empty((1, 3, 64, 64), device=device)
            depths = torch.empty((1, 1, 64, 64), device=device)
            torch.cuda._sleep(1_000_000)
            images.fill_((sample_index % 1_024) / 1_024.0)
            depths.fill_((sample_index % 257) / 257.0)
            target = torch.tensor([sample_index % 2], device=device)

            sync_context = ddp.no_sync() if micro_step == 0 else nullcontext()
            with sync_context:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    logits = ddp(images, depths)
                    loss = F.cross_entropy(logits.float(), target)
                all_finite = all_finite and bool(torch.isfinite(logits).all())
                all_finite = all_finite and bool(torch.isfinite(loss))
                loss_values.append(float(loss.detach().item()))
                scaler.scale(loss / 2.0).backward()

        scaler.unscale_(optimizer)
        all_finite = all_finite and all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in ddp.parameters()
        )
        finite_flag = torch.tensor(int(all_finite), device=device)
        dist.all_reduce(finite_flag, op=dist.ReduceOp.MIN)
        assert int(finite_flag.item()) == 1
        scaler.step(optimizer)
        scaler.update()
        print(
            f"DDP_RANK={rank} LOSSES={loss_values} " f"BACKWARD_FINITE={all_finite} EXIT=0",
            flush=True,
        )
        dist.barrier()
    finally:
        dist.destroy_process_group()


_RUN_DDP_STRESS = os.environ.get("MAGFORMER_RUN_CUDA_DDP_STRESS") == "1"


@pytest.mark.skipif(not _RUN_DDP_STRESS, reason="set MAGFORMER_RUN_CUDA_DDP_STRESS=1")
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="two CUDA devices are required")
@pytest.mark.skipif(not dist.is_nccl_available(), reason="NCCL is required")
def test_two_rank_ddp_rank1_failing_sequence_has_finite_loss_and_gradients() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        init_file = Path(temporary_directory) / "distributed_init"
        torch.multiprocessing.spawn(
            _two_rank_finite_worker,
            args=(2, f"file://{init_file}"),
            nprocs=2,
            join=True,
        )
