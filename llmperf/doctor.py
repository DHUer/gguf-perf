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
import json
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

    apple_silicon = platform.system() == "Darwin" and platform.machine() == "arm64"
    try:
        import torch
        if apple_silicon:
            mps = getattr(getattr(torch, "backends", None), "mps", None)
            if mps and mps.is_available():
                check("torch mps", PASS,
                      f"Metal/MPS available, torch {torch.__version__}")
            else:
                check("torch mps", FAIL,
                      "torch present but no MPS device; the declared Mac "
                      "calibration requires accelerator copy bandwidth and "
                      "FP16 throughput")
        elif torch.cuda.is_available():
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
        elif apple_silicon:
            check("torch mps", FAIL,
                  "torch not installed — the declared Mac calibration requires "
                  "MPS device-copy bandwidth and FP16 throughput")
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
        if out.returncode != 0:
            tail = blob.strip().splitlines()
            check("llama-bench backend", FAIL,
                  f"--list-devices exited {out.returncode}" +
                  ((": " + " | ".join(tail[-2:])[:260]) if tail else ""))
            return binary
        # Only trust devices in llama.cpp's inventory. Initialisation logs can
        # mention CUDA/Metal while reporting that the backend failed, which is
        # not evidence that an accelerator is usable.
        listed = []
        in_inventory = False
        for line in blob.splitlines():
            stripped = line.strip()
            if stripped.casefold() == "available devices:":
                in_inventory = True
                continue
            if not in_inventory or ":" not in stripped:
                continue
            device_id = stripped.split(":", 1)[0].upper()
            if device_id.startswith("CUDA"):
                listed.append("CUDA")
            elif device_id.startswith(("MTL", "METAL")):
                listed.append("Metal")
            elif device_id.startswith("VULKAN"):
                listed.append("Vulkan")
            elif device_id == "BLAS":
                listed.append("BLAS")
        backends = [name for name in ("CUDA", "Metal", "Vulkan", "BLAS")
                    if name in listed]
        gpu = [l.strip() for l in blob.splitlines()
               if "GPU name" in l or l.strip().startswith("Device ")]
        expected = None
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            expected = "Metal"
        elif platform.system() == "Windows":
            expected = "CUDA"

        if expected and expected not in backends:
            found = ", ".join(backends) if backends else "none"
            check("llama-bench backend", FAIL,
                  f"expected {expected} on {platform.system()} "
                  f"{platform.machine()}, found {found}")
        elif not backends:
            check("llama-bench backend", FAIL,
                  "no accelerator backend detected")
        elif backends == ["BLAS"]:
            check("llama-bench backend", FAIL,
                  "BLAS only, no GPU backend. On Windows make sure you took a "
                  "CUDA build, not the plain cpu zip.")
        else:
            check("llama-bench backend", PASS,
                  ", ".join(backends) + (f" | {gpu[0]}" if gpu else ""))
    except subprocess.TimeoutExpired:
        check("llama-bench backend", FAIL, "--list-devices timed out")
    except Exception as e:
        check("llama-bench backend", FAIL, f"{type(e).__name__}: {e}")
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


def _declared_model_sizes() -> dict[str, int]:
    """Return the frozen campaign filenames and exact byte sizes."""
    path = RESULTS_DIR / "model_manifest.json"
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise ValueError(f"cannot read frozen model manifest {path}: {e}") from e
    if not isinstance(document, dict) or not document:
        raise ValueError(f"frozen model manifest {path} must be a non-empty object")

    sizes = {}
    for name, entry in document.items():
        if (not isinstance(name, str) or not name or Path(name).name != name
                or not isinstance(entry, dict)):
            raise ValueError(f"invalid model entry {name!r} in {path}")
        size = entry.get("size_bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError(f"invalid size_bytes for {name!r} in {path}: {size!r}")
        sizes[name] = size
    return sizes


def _model_download_progress(sizes: dict[str, int]) -> tuple[int, int]:
    """Return (verified final bytes, resumable partial bytes)."""
    complete = partial = 0
    for name, expected in sizes.items():
        final = MODELS_DIR / name
        if final.is_file() and final.stat().st_size == expected:
            complete += expected
            continue
        part = final.with_suffix(final.suffix + ".part")
        if part.is_file():
            have = part.stat().st_size
            # fetch.py resumes a short/equal partial, but discards an oversized
            # one. Only bytes that can actually contribute count as progress.
            if 0 <= have <= expected:
                partial += have
    return complete, partial


def check_disk() -> None:
    try:
        sizes = _declared_model_sizes()
        # Models may be redirected to another volume. Check the filesystem that
        # will actually hold them, falling back to its existing parent before
        # the directory has been created.
        disk_path = MODELS_DIR
        while not disk_path.exists() and disk_path != disk_path.parent:
            disk_path = disk_path.parent
        usage = shutil.disk_usage(disk_path)
        complete, partial = _model_download_progress(sizes)
        total = sum(sizes.values())
        remaining = total - complete - partial
        detail = (f"{usage.free / 1e9:.1f} GB free; declared cohort "
                  f"{total / 1e9:.3f} GB, {complete / 1e9:.3f} GB complete, "
                  f"{partial / 1e9:.3f} GB resumable, "
                  f"{remaining / 1e9:.3f} GB remaining")
        if usage.free < remaining:
            check("disk", FAIL, detail + " — insufficient free space")
        else:
            check("disk", PASS, detail)
    except Exception as e:
        check("disk", FAIL, f"{type(e).__name__}: {e}")


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
    try:
        sizes = _declared_model_sizes()
        missing = []
        wrong = []
        complete = []
        for name, expected in sizes.items():
            path = MODELS_DIR / name
            if not path.is_file():
                missing.append(name)
                continue
            actual = path.stat().st_size
            if actual != expected:
                wrong.append(f"{name} ({actual} != {expected} bytes)")
            else:
                complete.append(path)

        partials = sorted(
            p.relative_to(MODELS_DIR).as_posix()
            for p in MODELS_DIR.rglob("*.gguf.part")
        ) if MODELS_DIR.is_dir() else []
        unexpected = sorted(
            p.relative_to(MODELS_DIR).as_posix()
            for p in discover_models(MODELS_DIR)
            if p.parent != MODELS_DIR or p.name not in sizes
        )
        if missing or wrong or partials or unexpected:
            details = [f"{len(complete)}/{len(sizes)} declared GGUF files complete"]
            if missing:
                details.append("missing: " + ", ".join(sorted(missing)))
            if wrong:
                details.append("wrong size: " + ", ".join(sorted(wrong)))
            if partials:
                details.append("partial downloads: " + ", ".join(partials))
            if unexpected:
                details.append("undeclared GGUF files: " + ", ".join(unexpected))
            check("models", FAIL, "; ".join(details))
            return

        from .common import read_gguf_meta
        smallest = min(complete, key=lambda p: p.stat().st_size)
        m = read_gguf_meta(smallest)
        check("models", PASS,
              f"all {len(complete)} declared files have exact sizes; metadata "
              f"parses ({m.name}: {m.n_params / 1e9:.1f}B, {m.n_layers} layers)")
    except Exception as e:
        check("models", FAIL, f"{type(e).__name__}: {e}")


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
