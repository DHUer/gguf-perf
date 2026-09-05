"""Environment check: run this FIRST on any new machine, before measuring.

Verifies every dependency the pipeline needs, runs all module self-checks, and
reports what the harness will actually do on this host (which backend, which
calibration source, how many threads). Prints a PASS/WARN/FAIL summary and
exits non-zero if anything is broken.

Catching a missing sm_120 build or an absent llama-bench here costs a minute;
catching it three hours into a sweep costs the sweep.

    python -m llmperf.doctor
    python -m llmperf.doctor --quick    # skip the live llama-bench run
"""
from __future__ import annotations

import argparse
import importlib
import math
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from .common import MODELS_DIR, RESULTS_DIR, ROOT, discover_models, host_id, platform_tag

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name: str, status: str, detail: str = "") -> None:
    results.append((name, status, detail))
    mark = {PASS: "  ok  ", WARN: " warn ", FAIL: " FAIL "}[status]
    print(f"[{mark}] {name}" + (f" — {detail}" if detail else ""), flush=True)


def check_python() -> None:
    v = sys.version_info
    if v < (3, 9):
        check("python", FAIL, f"{platform.python_version()}; need 3.9+")
    else:
        check("python", PASS, f"{platform.python_version()} at {sys.executable}")


def check_deps() -> None:
    required = [
        "numpy", "pandas", "matplotlib", "scipy", "gguf", "psutil",
        "huggingface_hub",
    ]
    missing = []
    versions = []
    for mod in required:
        try:
            m = importlib.import_module(mod)
            versions.append(f"{mod} {getattr(m, '__version__', '?')}")
        except ImportError:
            missing.append(mod)
    if missing:
        check("python deps", FAIL,
              f"missing {', '.join(missing)} — run: pip install -r requirements.txt")
    else:
        check("python deps", PASS, ", ".join(versions))

    try:
        import torch
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            cap = f"{p.major}.{p.minor}"
            detail = (f"{p.name}, {p.total_memory / 1e9:.0f} GB, sm_{p.major}{p.minor}, "
                      f"torch {torch.__version__} / cuda {torch.version.cuda}")
            # Blackwell consumer parts are sm_120 and need cu128 wheels; older
            # builds silently have no kernels for them.
            if p.major >= 12 and (torch.version.cuda or "") < "12.8":
                check("torch cuda", WARN,
                      detail + " — sm_120 needs CUDA 12.8+; reinstall with "
                      "--index-url https://download.pytorch.org/whl/cu128")
            else:
                check("torch cuda", PASS, detail)
        else:
            check("torch cuda", WARN,
                  "torch present but no CUDA device; GPU calibration unavailable")
    except ImportError:
        if platform.system() == "Windows":
            check("torch cuda", WARN,
                  "torch not installed — FLOPS calibration falls back to a CPU "
                  "fp32 matmul, which is not a device measurement. On the "
                  "discrete-GPU machine install: pip install torch "
                  "--index-url https://download.pytorch.org/whl/cu128")
        else:
            check("torch cuda", PASS, "not installed (not needed on Metal)")


def check_llama_bench(quick: bool) -> Path | None:
    from .common import find_llama_bench
    try:
        binary = find_llama_bench()
    except FileNotFoundError as e:
        check("llama-bench", FAIL, str(e).split(". ")[0] +
              " — macOS: brew install llama.cpp; Windows: download a CUDA 12.8+ "
              "release from github.com/ggml-org/llama.cpp/releases and add to PATH")
        return None
    check("llama-bench", PASS, str(binary))

    try:
        out = subprocess.run([str(binary), "--list-devices"],
                             capture_output=True, text=True, timeout=180)
        blob = (out.stdout + out.stderr)
        backends = []
        for token, label in (("CUDA", "CUDA"), ("Metal", "Metal"),
                             ("Vulkan", "Vulkan"), ("BLAS", "BLAS")):
            if token in blob:
                backends.append(label)
        gpu = [l.strip() for l in blob.splitlines()
               if "GPU name" in l or l.strip().startswith("Device ")]
        if not backends:
            check("llama-bench backend", WARN,
                  "no GPU backend detected — CPU-only inference will be very slow")
        elif backends == ["BLAS"]:
            check("llama-bench backend", FAIL,
                  "BLAS only, no GPU backend. On Windows make sure you took a "
                  "CUDA build, not the plain cpu zip.")
        else:
            check("llama-bench backend", PASS,
                  ", ".join(backends) + (f" | {gpu[0]}" if gpu else ""))
    except subprocess.TimeoutExpired:
        check("llama-bench backend", WARN, "--list-devices timed out")
    except Exception as e:
        check("llama-bench backend", WARN, f"{type(e).__name__}: {e}")
    return binary


def check_selfchecks() -> None:
    for mod in ("llmperf.common", "llmperf.sweep", "llmperf.refine"):
        argv = [sys.executable, "-m", mod]
        if mod in ("llmperf.sweep", "llmperf.refine"):
            argv.append("--selfcheck")
        try:
            out = subprocess.run(argv, cwd=ROOT, capture_output=True,
                                 text=True, timeout=300)
            if out.returncode == 0 and "selfcheck ok" in out.stdout:
                check(f"selfcheck {mod}", PASS)
            else:
                tail = (out.stderr or out.stdout).strip().splitlines()
                check(f"selfcheck {mod}", FAIL,
                      tail[-1][:200] if tail else f"exit {out.returncode}")
        except Exception as e:
            check(f"selfcheck {mod}", FAIL, f"{type(e).__name__}: {e}")


def check_threads() -> None:
    from .sweep import default_threads
    try:
        n = default_threads()
        check("thread count", PASS,
              f"{n} threads (logical CPUs: {os.cpu_count()})")
    except Exception as e:
        check("thread count", WARN, f"{type(e).__name__}: {e}")


def check_load_average() -> None:
    from .sweep import load_average
    try:
        load = load_average()
        if math.isfinite(load) and load >= 0:
            detail = (f"{load:.1f} busy logical CPUs (1 s sample)"
                      if platform.system() == "Windows"
                      else f"{load:.1f} (1 min average)")
            check("system load", PASS, detail)
        else:
            check("system load", FAIL,
                  "load metric is unavailable — verify psutil is installed")
    except Exception as e:
        check("system load", FAIL, f"{type(e).__name__}: {e}")


def check_disk() -> None:
    try:
        usage = shutil.disk_usage(ROOT)
        free_gb = usage.free / 1e9
        have = sum(p.stat().st_size for p in MODELS_DIR.glob("*.gguf")) / 1e9 \
            if MODELS_DIR.is_dir() else 0
        detail = f"{free_gb:.0f} GB free, {have:.0f} GB of models present"
        # The full model set is ~324 GB.
        if free_gb < 50:
            check("disk", FAIL, detail + " — the full model set needs ~324 GB")
        elif free_gb < 330 - have:
            check("disk", WARN, detail +
                  " — not enough for the full set; use --max-file-gb to subset")
        else:
            check("disk", PASS, detail)
    except Exception as e:
        check("disk", WARN, f"{type(e).__name__}: {e}")


def check_network() -> None:
    import urllib.request
    req = urllib.request.Request(
        "https://huggingface.co/api/models/ggml-org/SmolLM3-3B-GGUF",
        headers={"User-Agent": "llmperf/1.0"})
    if os.environ.get("HF_TOKEN"):
        req.add_header("Authorization", f"Bearer {os.environ['HF_TOKEN']}")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            ok = r.status == 200
        token = "HF_TOKEN set" if os.environ.get("HF_TOKEN") else \
            "no HF_TOKEN — downloads will be rate-limited; set one to go faster"
        check("huggingface reachable", PASS if ok else WARN, token)
    except Exception as e:
        check("huggingface reachable", WARN,
              f"{type(e).__name__}: {e} — needed only to fetch models")


def check_models() -> None:
    models = discover_models()
    if not models:
        check("models", WARN,
              f"no GGUF in {MODELS_DIR} — run: python -m llmperf.fetch --set pilot")
        return
    from .common import read_gguf_meta
    try:
        m = read_gguf_meta(min(models, key=lambda p: p.stat().st_size))
        check("models", PASS,
              f"{len(models)} on disk; metadata parses "
              f"({m.name}: {m.n_params / 1e9:.1f}B, {m.n_layers} layers)")
    except Exception as e:
        check("models", FAIL, f"GGUF metadata unreadable: {type(e).__name__}: {e}")


def check_live_run(binary: Path | None) -> None:
    """One real llama-bench invocation — the only end-to-end proof."""
    if binary is None:
        check("live benchmark", FAIL, "skipped, no llama-bench")
        return
    models = discover_models()
    if not models:
        check("live benchmark", WARN, "skipped, no models on disk")
        return
    smallest = min(models, key=lambda p: p.stat().st_size)
    cmd = [str(binary), "-m", str(smallest), "-p", "64", "-n", "16", "-d", "0",
           "-ngl", "99", "-r", "1", "-fa", "on", "-ctk", "f16", "-ctv", "f16",
           "-o", "json"]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        if out.returncode != 0:
            tail = (out.stderr or "").strip().splitlines()
            check("live benchmark", FAIL,
                  " | ".join(tail[-2:])[:300] if tail else f"exit {out.returncode}")
            return
        import json as _json
        rows = _json.loads(out.stdout)
        tg = [r for r in rows if (r.get("n_gen") or 0) > 0]
        check("live benchmark", PASS,
              f"{smallest.name}: {tg[0]['avg_ts']:.1f} tok/s decode" if tg
              else "ran, but produced no decode row")
    except subprocess.TimeoutExpired:
        check("live benchmark", FAIL, "timed out after 15 min on the smallest model")
    except Exception as e:
        check("live benchmark", FAIL, f"{type(e).__name__}: {e}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true",
                    help="skip the live llama-bench run")
    args = ap.parse_args(argv)

    print(f"llmperf doctor — host={host_id()} platform={platform_tag()}")
    print(f"repo={ROOT}\n")

    check_python()
    check_deps()
    binary = check_llama_bench(args.quick)
    check_selfchecks()
    check_threads()
    check_load_average()
    check_disk()
    check_network()
    check_models()
    if not args.quick:
        check_live_run(binary)

    n_fail = sum(1 for _, s, _ in results if s == FAIL)
    n_warn = sum(1 for _, s, _ in results if s == WARN)
    print(f"\n{len(results) - n_fail - n_warn} pass, {n_warn} warn, {n_fail} fail")

    if n_fail:
        print("\nFix the FAIL items before measuring. Numbers produced by a "
              "broken environment are worse than no numbers.")
        return 1
    if n_warn:
        print("\nUsable. Review the warnings — some only degrade calibration "
              "quality, others (no HF_TOKEN) only cost time.")
    else:
        print("\nReady. Next: python -m llmperf.campaign")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
