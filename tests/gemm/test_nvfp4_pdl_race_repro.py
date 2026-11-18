"""Reproducer for the SM103 PDL race between nvfp4_quantize and the CUTLASS FP4 GEMM.

The quantization kernel triggers ``griddepcontrol.launch_dependents`` before
its stores are flushed; the SM103 CUTLASS block-scaled GEMM is launched with
PDL (``fp4_gemm_template_sm103.h``, ``enablePDL=true``) and, before CUTLASS
v4.8, its ``main_sf_load`` warp reads the scale factors without
``griddepcontrol.wait``. TRT-LLM saw it as NaNs / a GSM8K drop on GB300
(nvbug 5970614); FlashInfer's fix on the producer side is #9445.

Both kernels are deterministic, so the pair run with PDL must produce
bitwise the output of the pair run without PDL on the producer. The buffers
the quantization will write are poisoned with 0xFF (an E4M3 NaN scale) just
before each call so a scale factor read before its store lands turns into a
NaN in the output. Any mismatch or NaN over the iterations is a hit.

``NVFP4_PDL_REPRO_ITERS`` sets the iterations per case (default 2000);
``NVFP4_PDL_REPRO_ANY_SM=1`` runs on other architectures (API check only).
"""

import os

import pytest
import torch

from flashinfer import SfLayout, mm_fp4, nvfp4_quantize
from flashinfer.utils import get_compute_capability

ITERS = int(os.environ.get("NVFP4_PDL_REPRO_ITERS", "2000"))
# DeepSeek-R1 NVFP4 shapes (m tokens, n out, k in) plus a tall-skinny one.
SHAPES = [
    (8192, 7168, 2048),
    (4096, 2048, 7168),
    (16384, 4096, 4096),
    (512, 7168, 7168),
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
    x = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    w = torch.randn(n, k, device=device, dtype=torch.bfloat16)
    gs_x = (448 * 6) / x.float().abs().max()
    gs_w = (448 * 6) / w.float().abs().max()
    w_fp4, w_sf = nvfp4_quantize(w, gs_w, sfLayout=SfLayout.layout_128x4)
    alpha = 1.0 / (gs_x * gs_w)

    def pair(enable_pdl, out):
        a_fp4, a_sf = nvfp4_quantize(
            x, gs_x, sfLayout=SfLayout.layout_128x4, enable_pdl=enable_pdl
        )
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
        return a_fp4, a_sf

    ref = torch.empty(m, n, device=device, dtype=torch.bfloat16)
    a_fp4, a_sf = pair(False, ref)
    torch.cuda.synchronize()
    assert not torch.isnan(ref).any(), (
        "reference pair produced NaN; the harness is wrong"
    )
    shapes = [(a_fp4.shape, a_fp4.dtype), (a_sf.shape, a_sf.dtype)]
    del a_fp4, a_sf

    def poison():
        # Same sizes as the quantization outputs: the caching allocator hands these blocks back
        # to the next torch.empty of the same size, so a stale read sees 0xFF (NaN as E4M3).
        blocks = [
            torch.full(s, 255, device=device, dtype=torch.uint8).view(d)
            for s, d in shapes
        ]
        del blocks

    out = torch.empty_like(ref)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        if graph:
            poison()
            pair(True, out)  # warm up inside the side stream before capture
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=stream):
                pair(True, out)
        mismatches = torch.zeros((), device=device, dtype=torch.int64)
        nans = torch.zeros((), device=device, dtype=torch.int64)
        for _ in range(ITERS):
            poison()
            if graph:
                g.replay()
            else:
                pair(True, out)
            mismatches += (out != ref).any()
            nans += torch.isnan(out).any()
    torch.cuda.synchronize()
    bad, nan = int(mismatches), int(nans)
    print(
        f"\n{_ids(shape)} {'graph' if graph else 'eager'}: {bad}/{ITERS} iterations differ from the "
        f"no-PDL result, {nan} with NaN"
    )
    assert bad == 0, f"PDL race: {bad}/{ITERS} iterations differ ({nan} with NaN)"
