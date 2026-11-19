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
- eager mode enqueues a few milliseconds of filler GEMMs before each pair so
  both launches are queued before the quantize runs (the host side of the two
  FlashInfer calls is a few hundred microseconds); the loop time and the final
  synchronize time are printed so a run without backlog is visible; graph mode replays one graph
  whose static input is refilled between replays.

Any element differing from the reference, or any NaN, is a hit.
``NVFP4_PDL_REPRO_ITERS`` sets the iterations per graph case (default 20000;
eager cases run a fifth of it because of the filler);
``NVFP4_PDL_REPRO_ANY_SM=1`` runs on other architectures (API check only).
"""

import os
import time

import pytest
import torch

from flashinfer import SfLayout, mm_fp4, nvfp4_quantize
from flashinfer.utils import get_compute_capability

ITERS = int(os.environ.get("NVFP4_PDL_REPRO_ITERS", "20000"))
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


def _skip_unless_sm103(device):
    if get_compute_capability(device) != (10, 3) and not os.environ.get(
        "NVFP4_PDL_REPRO_ANY_SM"
    ):
        pytest.skip("The unguarded main_sf_load is the SM103 CUTLASS kernel")


def _race_loop(shape, graph, label):
    """One quantize->GEMM pair loop; returns (mismatches, nans, iterations)."""
    device = torch.device("cuda")
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
    filler_a = torch.randn(8192, 8192, device=device, dtype=torch.bfloat16)
    filler_out = torch.empty(8192, 8192, device=device, dtype=torch.bfloat16)

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
    assert out.data_ptr() % 32 == 0, (
        "K768 tactics need a 32-byte aligned output or they fall back"
    )
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
        torch.cuda.synchronize()
        t_loop = time.perf_counter()
        iters = (
            ITERS if graph else ITERS // 5
        )  # eager pays ~2 ms of filler per iteration
        for i in range(iters):
            cur = i & 1
            if graph:
                x_static.copy_(inputs[cur])
                g.replay()
            else:
                for _ in range(
                    4
                ):  # ~2 ms of GEMM: both launches below queue up behind it
                    torch.mm(filler_a, filler_a, out=filler_out)
                pair(inputs[cur], out)
            mismatches += (out != refs[cur]).any()
            nans += torch.isnan(out).any()
    t_sync = time.perf_counter()
    torch.cuda.synchronize()
    t_end = time.perf_counter()
    bad, nan = int(mismatches), int(nans)
    backlog = (
        ""
        if t_end - t_sync >= 0.05 * (t_sync - t_loop)
        else " (no GPU backlog: launches were not queued ahead)"
    )
    print(
        f"\n{label} {'graph' if graph else 'eager'}: {bad}/{iters} iterations differ from the "
        f"synchronized reference, {nan} with NaN; loop {t_sync - t_loop:.1f}s, final sync {t_end - t_sync:.1f}s{backlog}"
    )
    return bad, nan, iters


@pytest.mark.parametrize("graph", [False, True], ids=["eager", "graph"])
@pytest.mark.parametrize("shape", SHAPES, ids=_ids)
def test_quantize_then_cutlass_gemm_with_pdl(shape, graph):
    _skip_unless_sm103(torch.device("cuda"))
    bad, nan, iters = _race_loop(shape, graph, _ids(shape))
    assert bad == 0, f"PDL race: {bad}/{iters} iterations differ ({nan} with NaN)"


# mm_fp4(backend="cutlass") on SM103 dispatches by tile: candidates 0-35 (K128 / K256 tiles) run the
# SM100 kernel, whose load warp waits on the dependent grid; only 36-62 (K768 tiles) run the SM103
# block-scaled kernel whose main_sf_load warp does not. The default tactics {22, 20, 29, 4, 18} never
# reach it, so the untuned test above exercises the safe kernel. This sweep forces the tactic.
TACTICS = [22, 36, 40, 44, 48, 53, 58, 62]


@pytest.mark.parametrize("graph", [False, True], ids=["eager", "graph"])
@pytest.mark.parametrize("tactic", TACTICS, ids=lambda t: f"t{t}")
def test_quantize_then_cutlass_gemm_with_pdl_tactic(tactic, graph, monkeypatch):
    _skip_unless_sm103(torch.device("cuda"))
    from flashinfer.autotuner import AutoTuner

    monkeypatch.setattr(
        AutoTuner,
        "choose_one",
        lambda self, name, runners, cfg, inputs, **kw: (runners[0], tactic),
    )
    bad, nan, iters = _race_loop((8, 4096, 7168), graph, f"tactic {tactic} 8x4096x7168")
    assert bad == 0, (
        f"PDL race with tactic {tactic}: {bad}/{iters} iterations differ ({nan} with NaN)"
    )


def test_pdl_overlap_trace():
    """Measure, with the CUDA profiler, whether the GEMM starts before the quantize kernel ends.

    Reports the median (GEMM start - quantize end) over 30 queued pairs through
    ``pytest.fail`` so the numbers land in the JUnit report; negative means the
    PDL launch overlapped. Run on the positive-control branch, overlap should be
    close to -200 us.
    """
    device = torch.device("cuda")
    if get_compute_capability(device) != (10, 3) and not os.environ.get(
        "NVFP4_PDL_REPRO_ANY_SM"
    ):
        pytest.skip("The unguarded main_sf_load is the SM103 CUTLASS kernel")
    m, n, k = 8, 4096, 7168
    torch.manual_seed(0)
    x = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    w = torch.randn(n, k, device=device, dtype=torch.bfloat16)
    gs_x = torch.tensor(448 * 6 / 6.0, device=device)
    gs_w = (448 * 6) / w.float().abs().max()
    w_fp4, w_sf = nvfp4_quantize(w, gs_w, sfLayout=SfLayout.layout_128x4)
    alpha = 1.0 / (gs_x * gs_w)
    out = torch.empty(m, n, device=device, dtype=torch.bfloat16)
    filler_a = torch.randn(8192, 8192, device=device, dtype=torch.bfloat16)
    filler_out = torch.empty_like(filler_a)

    def pair():
        a_fp4, a_sf = nvfp4_quantize(
            x, gs_x, sfLayout=SfLayout.layout_128x4, enable_pdl=True
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

    for _ in range(3):
        pair()
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        for _ in range(30):
            for _ in range(4):
                torch.mm(filler_a, filler_a, out=filler_out)
            pair()
        torch.cuda.synchronize()
    events = sorted(
        (
            (e.start_ns(), e.start_ns() + e.duration_ns(), e.name())
            for e in prof.profiler.kineto_results.events()
            if e.device_type() == torch.autograd.DeviceType.CUDA and e.duration_ns() > 0
        ),
        key=lambda t: t[0],
    )
    gaps = []
    for (qs, qe, qn), (gs, ge, gn) in zip(events, events[1:], strict=False):
        if "quantize_with_block_size" in qn and "device_kernel" in gn:
            gaps.append(
                (gs - qe) / 1000.0,
            )
            gaps[-1] = ((gs - qe) / 1000.0, (qe - qs) / 1000.0, (ge - gs) / 1000.0)
    if not gaps:
        pytest.fail(
            "no quantize->GEMM pairs found in the trace: "
            + ", ".join(sorted({e[2][:40] for e in events}))
        )
    gaps_sorted = sorted(g[0] for g in gaps)
    med = gaps_sorted[len(gaps_sorted) // 2]
    pytest.fail(
        f"PDL overlap trace over {len(gaps)} pairs: median GEMM start - quantize end = {med:.1f} us "
        f"(min {gaps_sorted[0]:.1f}, max {gaps_sorted[-1]:.1f}); quantize {gaps[-1][1]:.1f} us, "
        f"GEMM {gaps[-1][2]:.1f} us. Negative = the GEMM started before the quantize ended."
    )


# Long run on the exposed kernel only: every K768 tactic, three decode row counts, graph replay
# (about 50 us per pair, so 500k pairs per case in under a minute).
LONG_ITERS = int(os.environ.get("NVFP4_PDL_REPRO_LONG_ITERS", "500000"))
K768_TACTICS = list(range(36, 63))


@pytest.mark.parametrize("m", [1, 8, 64], ids=lambda m: f"m{m}")
@pytest.mark.parametrize("tactic", K768_TACTICS, ids=lambda t: f"t{t}")
def test_quantize_then_cutlass_gemm_with_pdl_tactic_long(tactic, m, monkeypatch):
    _skip_unless_sm103(torch.device("cuda"))
    from flashinfer.autotuner import AutoTuner

    monkeypatch.setattr(
        AutoTuner,
        "choose_one",
        lambda self, name, runners, cfg, inputs, **kw: (runners[0], tactic),
    )
    monkeypatch.setitem(globals(), "ITERS", LONG_ITERS)
    bad, nan, iters = _race_loop(
        (m, 4096, 7168), True, f"tactic {tactic} {m}x4096x7168"
    )
    assert bad == 0, (
        f"PDL race with tactic {tactic}, m={m}: {bad}/{iters} iterations differ ({nan} with NaN)"
    )
