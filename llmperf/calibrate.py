"""Stage 2: cheap hardware microbenchmarks that calibrate the performance model.

The claim this script has to support is that a few minutes of measurement per
machine is enough to predict hours of LLM benchmarks. So it measures the two
quantities the roofline needs -- achievable memory bandwidth and achievable
compute throughput -- plus one LLM-side reference point.

Three calibration sources are recorded, deliberately, because they disagree and
the paper needs to report which one actually predicts best:

  cpu_triad    STREAM-style bandwidth from the CPU cores. On Apple Silicon this
               UNDER-reports what the GPU can reach: unified memory is shared,
               but CPU cores cannot saturate the fabric that the GPU can. Do not
               use this as "the" bandwidth on Metal machines.
  torch_gpu    Real device bandwidth and FLOPS, only available where torch sees
               a CUDA device. This is the honest number on the discrete GPU.
  llm_ref      Effective decode bandwidth back-solved from ONE llama-bench run
               on the smallest model. One-point calibration; portable to every
               backend because it goes through the same code path being
               predicted.

Reporting all three, and being explicit that cpu_triad is not valid on Metal,
is the difference between a measurement paper and a misleading one.

    python -m llmperf.calibrate
    python -m llmperf.calibrate --skip-llm-ref
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

from .common import (MODELS_DIR, RESULTS_DIR, discover_models, find_llama_bench,
                     git_commit, host_id, platform_tag, read_gguf_meta, write_json)


def bench_memory_bandwidth(mib: int = 512, reps: int = 7) -> dict:
    """STREAM triad: a = b + s*c. Reports the best run, in GB/s.

    Best-of-N rather than mean: we want achievable peak, and a slow run only
    ever means interference from something else on the machine.
    """
    import numpy as np

    n = (mib * 1024 * 1024) // 4  # float32
    b = np.ones(n, dtype=np.float32)
    c = np.full(n, 2.0, dtype=np.float32)
    a = np.empty(n, dtype=np.float32)
    s = np.float32(3.0)

    np.add(b, s * c, out=a)  # warmup / page-in

    bytes_moved = 3 * n * 4  # read b, read c, write a
    best = 0.0
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        np.add(b, s * c, out=a)
        dt_s = time.perf_counter() - t0
        times.append(dt_s)
        best = max(best, bytes_moved / dt_s / 1e9)

    assert abs(float(a[0]) - 7.0) < 1e-6, "triad produced wrong values"
    return {"gb_s": best, "array_mib": mib, "reps": reps,
            "median_s": float(sorted(times)[len(times) // 2])}


def bench_matmul_flops(n: int = 4096, reps: int = 5) -> dict:
    """Dense fp32 matmul throughput in GFLOP/s via the platform BLAS.

    Accelerate on macOS, MKL/OpenBLAS on Windows. This is the prefill-side
    compute number.
    """
    import numpy as np

    rng = np.random.default_rng(0)
    a = rng.standard_normal((n, n), dtype=np.float32)
    b = rng.standard_normal((n, n), dtype=np.float32)
    a @ b  # warmup

    flops = 2.0 * n ** 3
    best = 0.0
    for _ in range(reps):
        t0 = time.perf_counter()
        a @ b
        best = max(best, flops / (time.perf_counter() - t0) / 1e9)
    return {"gflops": best, "n": n, "reps": reps, "dtype": "float32"}


def bench_torch_gpu() -> dict | None:
    """Device-side bandwidth and FLOPS on whichever accelerator torch can see.

    Covers both consumer paths: CUDA on the discrete-GPU machine and Metal/MPS
    on Apple Silicon. Without this the compute term falls back to a CPU fp32
    matmul, which is not the GPU's throughput, so eta_p ends up a fitted
    constant absorbing a scale error rather than a real efficiency.
    """
    try:
        import torch
    except ImportError:
        return {"available": False, "reason": "torch not installed"}

    if torch.cuda.is_available():
        backend, dev = "cuda", torch.device("cuda")
        props = torch.cuda.get_device_properties(0)
        name, mem_gb = props.name, props.total_memory / 1e9
        cap = f"{props.major}.{props.minor}"
        sync = torch.cuda.synchronize
        dtype = torch.float16
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        backend, dev = "mps", torch.device("mps")
        name, cap = f"{platform.processor() or platform.machine()} (Metal)", "mps"
        # Unified memory: there is no separate VRAM figure, report system RAM.
        mem_gb = (os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
                  if hasattr(os, "sysconf") else 0.0)
        sync = torch.mps.synchronize
        dtype = torch.float16
    else:
        return {"available": False,
                "reason": "no CUDA or MPS device visible to torch"}

    # Bandwidth: a device-to-device copy moves 2 bytes of traffic per byte.
    n = 256 * 1024 * 1024 // 4
    x = torch.ones(n, dtype=torch.float32, device=dev)
    y = torch.empty_like(x)
    sync()
    best_bw = 0.0
    for _ in range(7):
        t0 = time.perf_counter()
        y.copy_(x)
        sync()
        best_bw = max(best_bw, 2 * n * 4 / (time.perf_counter() - t0) / 1e9)

    m = 8192
    a = torch.randn(m, m, device=dev, dtype=dtype)
    b = torch.randn(m, m, device=dev, dtype=dtype)
    a @ b
    sync()
    best_fl = 0.0
    for _ in range(5):
        t0 = time.perf_counter()
        a @ b
        sync()
        best_fl = max(best_fl, 2.0 * m ** 3 / (time.perf_counter() - t0) / 1e12)

    del x, y, a, b
    return {"available": True, "backend": backend, "name": name,
            "vram_gb": mem_gb, "copy_gb_s": best_bw, "fp16_tflops": best_fl,
            "capability": cap, "torch": torch.__version__,
            "cuda": getattr(torch.version, "cuda", None)}


def _probe_models(models_dir: Path, n: int = 3) -> list[Path]:
    """Choose calibration probes: TRAIN-split models, largest and highest-bit.

    Two rules, both learned the hard way.

    Restrict to the train split: the anchor normalises every prediction, so
    measuring it on a test model leaks test information into the fit.

    Prefer large, high-precision models. The first version of this anchored on
    the *smallest* model, which is precisely the worst reference — a small
    low-bit model pays maximum dequantisation overhead and never saturates the
    memory fabric, so it reports a floor and every efficiency computed against
    it came out above 1. On this machine that single choice inflated eta by
    1.49x across the board.
    """
    models = discover_models(models_dir)
    if not models:
        return []

    manifest = models_dir / "manifest.json"
    train = None
    if manifest.exists():
        try:
            m = json.loads(manifest.read_text(encoding="utf-8"))
            train = {k for k, v in m.items() if v.get("split") == "train"}
        except Exception:
            train = None
    if train:
        eligible = [p for p in models if p.name in train] or models
    else:
        eligible = models

    # Higher bit-width first, then larger file: closest to a pure stream.
    bits = {"F32": 32, "F16": 16, "BF16": 16, "Q8_0": 8, "Q6_K": 6, "Q5_K": 5,
            "Q4_K_M": 4, "Q4_K": 4, "Q4_0": 4, "MXFP4": 4, "Q3_K": 3, "Q2_K": 2}
    from .common import _quant_from_name
    eligible.sort(key=lambda p: (bits.get(_quant_from_name(p.name), 4),
                                 p.stat().st_size), reverse=True)
    return eligible[:n]


def bench_llm_reference(models_dir: Path, reps: int = 5, n_probes: int = 3) -> dict | None:
    """Back-solve effective decode bandwidth, taking the CEILING over probes.

    decode tok/s ~= BW_eff / bytes_read_per_token, so a measured tok/s on a
    known-size model gives BW_eff directly. Any single model reports what *it*
    achieved, which is a lower bound on the machine. Probing several and taking
    the maximum estimates the machine's demonstrated ceiling, which is what an
    efficiency should be measured against.
    """
    probes = _probe_models(models_dir, n_probes)
    if not probes:
        return {"available": False, "reason": f"no GGUF in {models_dir}"}
    try:
        binary = find_llama_bench()
    except Exception as e:
        return {"available": False, "reason": f"{type(e).__name__}: {e}"}

    results = []
    for path in probes:
        try:
            meta = read_gguf_meta(path)
        except Exception:
            continue
        cmd = [str(binary), "-m", str(path), "-p", "0", "-n", "128", "-d", "0",
               "-ngl", "99", "-r", str(reps), "-fa", "on",
               "-ctk", "f16", "-ctv", "f16", "-o", "json"]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            rows = json.loads(proc.stdout)
        except Exception:
            continue
        tg = [r for r in rows if (r.get("n_gen") or 0) > 0]
        if not tg:
            continue
        tok_s = float(tg[0]["avg_ts"])
        bpt = meta.working_bytes(context=0)
        results.append({"model": path.name, "file_gb": meta.file_bytes / 1e9,
                        "quant": meta.quant, "decode_tok_s": tok_s,
                        "stddev_tok_s": float(tg[0].get("stddev_ts") or 0),
                        "bytes_per_token": bpt,
                        "eff_bw_gb_s": tok_s * bpt / 1e9})

    if not results:
        return {"available": False, "reason": "no probe produced a decode row"}

    best = max(results, key=lambda r: r["eff_bw_gb_s"])
    return {"available": True, "eff_bw_gb_s": best["eff_bw_gb_s"],
            "anchor_model": best["model"], "anchor_quant": best["quant"],
            "model": best["model"],          # kept: analysis excludes the probes
            "probe_models": [r["model"] for r in results],
            "probes": results,
            "decode_tok_s": best["decode_tok_s"],
            "stddev_tok_s": best["stddev_tok_s"],
            "bytes_per_token": best["bytes_per_token"],
            "file_gb": best["file_gb"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    ap.add_argument("--mib", type=int, default=512)
    ap.add_argument("--matmul-n", type=int, default=4096)
    ap.add_argument("--skip-llm-ref", action="store_true")
    args = ap.parse_args(argv)

    print(f"Calibrating {host_id()} ({platform_tag()})\n")

    print("  memory bandwidth (CPU triad) ... ", end="", flush=True)
    mem = bench_memory_bandwidth(args.mib)
    print(f"{mem['gb_s']:.1f} GB/s")

    print("  compute (fp32 matmul)        ... ", end="", flush=True)
    flops = bench_matmul_flops(args.matmul_n)
    print(f"{flops['gflops']:.0f} GFLOP/s")

    print("  gpu (torch/cuda)             ... ", end="", flush=True)
    gpu = bench_torch_gpu()
    print(f"{gpu['copy_gb_s']:.0f} GB/s, {gpu['fp16_tflops']:.1f} TFLOPS ({gpu['name']})"
          if gpu.get("available") else f"n/a ({gpu.get('reason')})")

    llm = {"available": False, "reason": "skipped"}
    if not args.skip_llm_ref:
        print("  llm decode reference         ... ", end="", flush=True)
        llm = bench_llm_reference(args.models_dir)
        print(f"{llm['eff_bw_gb_s']:.1f} GB/s effective "
              f"({llm['decode_tok_s']:.1f} tok/s on {llm['model']})"
              if llm.get("available") else f"n/a ({llm.get('reason')})")

    is_metal = platform.system() == "Darwin" and platform.machine() == "arm64"
    result = {
        "host": host_id(), "platform": platform_tag(),
        "timestamp": dt.datetime.now().astimezone().isoformat(),
        "git_commit": git_commit(),
        "cpu": platform.processor() or platform.machine(),
        "cpu_count": __import__("os").cpu_count(),
        "python": sys.version.split()[0],
        "cpu_triad": mem, "cpu_matmul": flops, "torch_gpu": gpu, "llm_ref": llm,
        "caveats": {
            "cpu_triad_valid_for_gpu": not is_metal,
            "note": ("On Apple Silicon the CPU triad under-reports the bandwidth "
                     "the GPU can reach through unified memory; use llm_ref as "
                     "the bandwidth term on this machine.")
            if is_metal else
            ("On a discrete GPU the CPU triad measures host RAM, which is the "
             "relevant term only for offloaded layers; use torch_gpu for "
             "resident layers."),
        },
    }

    out = RESULTS_DIR / f"calibration_{host_id()}.json"
    write_json(out, result)
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
