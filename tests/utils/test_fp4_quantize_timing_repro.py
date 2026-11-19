"""Timing of nvfp4_quantize (bf16 -> NVFP4, 128x4 scale layout, PDL on) for the #9445 before/after.

Reports median microseconds per shape through a deliberate pytest failure so the
numbers land in the JUnit report of a controller run.
"""

import pytest
import torch

from flashinfer import SfLayout, nvfp4_quantize

SHAPES = [
    (8, 7168),
    (64, 7168),
    (1024, 7168),
    (4096, 7168),
    (16384, 7168),
    (4096, 16384),
]


def test_nvfp4_quantize_timing():
    device = torch.device("cuda")
    lines = []
    for m, k in SHAPES:
        torch.manual_seed(0)
        x = torch.randn(m, k, device=device, dtype=torch.bfloat16)
        gs = (448 * 6) / x.float().abs().max()
        for _ in range(20):
            nvfp4_quantize(x, gs, sfLayout=SfLayout.layout_128x4, enable_pdl=True)
        torch.cuda.synchronize()
        # 50 launches per graph replay: GPU time per call, not Python launch overhead.
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(50):
                nvfp4_quantize(x, gs, sfLayout=SfLayout.layout_128x4, enable_pdl=True)
        g.replay()
        torch.cuda.synchronize()
        times = []
        for _ in range(20):
            start, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            start.record()
            g.replay()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end) * 1000.0 / 50)
        times.sort()
        lines.append(
            f"{m}x{k}: median {times[len(times) // 2]:.1f} us, min {times[0]:.1f} us"
        )
    pytest.fail(
        "nvfp4_quantize timing ("
        + torch.cuda.get_device_name(device)
        + "): "
        + "; ".join(lines)
    )
