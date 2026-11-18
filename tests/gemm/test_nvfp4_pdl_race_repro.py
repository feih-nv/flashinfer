"""Reproducer for the SM103 PDL race between nvfp4_quantize and the CUTLASS FP4 GEMM.

The quantization kernel triggers ``griddepcontrol.launch_dependents`` before
its stores are flushed; the SM103 CUTLASS block-scaled GEMM is launched with
PDL (``fp4_gemm_template_sm103.h``, ``enablePDL=true``) and, before CUTLASS
v4.8, its ``main_sf_load`` warp reads the scale factors without
``griddepcontrol.wait``. TRT-LLM saw it as NaNs / a GSM8K drop on GB300
(nvbug 5970614): DeepSeek-R1 decode, the vectorized (non-TMA) quantize kernel,
K=7168, N=4096, a few tokens, GEMM starting 128 ns after the quantize.

Design (round 4):
- decode shapes, so the quantize kernel takes the vectorized path (m < 1024)
  and leaves most SMs free for the GEMM to start early;
- two inputs alternate, each with its own reference, so the previous
  iteration's quantize output never equals the current one (no allocator
  tricks needed, works under CUDA graph replay too);
- the reference runs quantize and GEMM with a ``torch.cuda.synchronize()``
  between them (the no-PDL "reference" of earlier rounds raced as well: the
  producer always triggers and the GEMM is always a PDL launch);
- eager mode enqueues a ~100 us filler kernel before each pair so both
  launches are queued before the quantize runs; graph mode replays one graph
  whose static input is refilled between replays.

Any element differing from the reference, or any NaN, is a hit.
``NVFP4_PDL_REPRO_ITERS`` sets the iterations per case (default 100000);
``NVFP4_PDL_REPRO_ANY_SM=1`` runs on other architectures (API check only).
"""

import os

import pytest
import torch

from flashinfer import SfLayout, mm_fp4, nvfp4_quantize
from flashinfer.utils import get_compute_capability

ITERS = int(os.environ.get("NVFP4_PDL_REPRO_ITERS", "100000"))
# (m tokens, n out, k in): DeepSeek-R1 decode at K=7168 / N=4096, plus the
# non-multiple-of-128 hidden sizes TRT-LLM validated (1568, 2880).
SHAPES = [(m, 4096, 7168) for m in (1, 2, 4, 8, 16, 32, 64, 100)] + [
    (8, 4096, 1568),
    (32, 4096, 1568),
    (8, 4096, 2880),
    (32, 4096, 2880),
]


def _ids(shape):
    return "x".join(str(v) for v in shape)


@pytest.mark.parametrize("graph", [False, True], ids=["eager", "graph"])
@pytest.mark.parametrize("shape", SHAPES, ids=_ids)
def test_quantize_then_cutlass_gemm_with_pdl(shape, graph):
    device = torch.device("cuda")
    if get_compute_capability(device) != (10, 3) and not os.environ.get(
        "NVFP4_PDL_REPRO_ANY_SM"
    ):
        pytest.skip("The unguarded main_sf_load is the SM103 CUTLASS kernel")
    m, n, k = shape
    torch.manual_seed(0)
    inputs = [torch.randn(m, k, device=device, dtype=torch.bfloat16) for _ in range(2)]
    w = torch.randn(n, k, device=device, dtype=torch.bfloat16)
    gs_x = torch.tensor(
        448 * 6 / 6.0, device=device
    )  # one global scale for both inputs
    gs_w = (448 * 6) / w.float().abs().max()
    w_fp4, w_sf = nvfp4_quantize(w, gs_w, sfLayout=SfLayout.layout_128x4)
    alpha = 1.0 / (gs_x * gs_w)
    filler_a = torch.randn(4096, 4096, device=device, dtype=torch.bfloat16)
    filler_out = torch.empty(4096, 4096, device=device, dtype=torch.bfloat16)

    def gemm(a_fp4, a_sf, out):
        mm_fp4(
            a_fp4,
            w_fp4.T,
            a_sf,
            w_sf.T,
            alpha,
            torch.bfloat16,
            out,
            block_size=16,
            use_8x4_sf_layout=False,
            backend="cutlass",
            use_nvfp4=True,
        )

    def pair(x, out):
        a_fp4, a_sf = nvfp4_quantize(
            x, gs_x, sfLayout=SfLayout.layout_128x4, enable_pdl=True
        )
        gemm(a_fp4, a_sf, out)

    # References: quantize, synchronize, GEMM. No overlap is possible.
    refs = []
    for x in inputs:
        a_fp4, a_sf = nvfp4_quantize(
            x, gs_x, sfLayout=SfLayout.layout_128x4, enable_pdl=False
        )
        torch.cuda.synchronize()
        ref = torch.empty(m, n, device=device, dtype=torch.bfloat16)
        gemm(a_fp4, a_sf, ref)
        torch.cuda.synchronize()
        assert not torch.isnan(ref).any(), (
            "reference pair produced NaN; the harness is wrong"
        )
        refs.append(ref)
    assert not torch.equal(refs[0], refs[1]), (
        "the two inputs must give different outputs"
    )

    out = torch.empty(m, n, device=device, dtype=torch.bfloat16)
    x_static = inputs[0].clone()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        if graph:
            pair(x_static, out)  # warm up on the side stream before capture
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=stream):
                pair(x_static, out)
        mismatches = torch.zeros((), device=device, dtype=torch.int64)
        nans = torch.zeros((), device=device, dtype=torch.int64)
        for i in range(ITERS):
            cur = i & 1
            if graph:
                x_static.copy_(inputs[cur])
                g.replay()
            else:
                torch.mm(
                    filler_a, filler_a, out=filler_out
                )  # ~100 us: both launches below queue up behind it
                pair(inputs[cur], out)
            mismatches += (out != refs[cur]).any()
            nans += torch.isnan(out).any()
    torch.cuda.synchronize()
    bad, nan = int(mismatches), int(nans)
    print(
        f"\n{_ids(shape)} {'graph' if graph else 'eager'}: {bad}/{ITERS} iterations differ from the "
        f"synchronized reference, {nan} with NaN"
    )
    assert bad == 0, f"PDL race: {bad}/{ITERS} iterations differ ({nan} with NaN)"
