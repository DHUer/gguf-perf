"""Variance before effect: how repeatable is a single measurement cell?

Run this BEFORE trusting any difference in the main sweep. If run-to-run spread
on one fixed configuration is larger than the effect you want to report, the
grid is not worth running yet -- the settling protocol has to be fixed first.

This matters most on a thermally constrained laptop, where back-to-back cells
are measured at progressively higher die temperatures and the numbers drift
downward through the sweep. It is the single most common way a consumer-hardware
benchmark quietly produces wrong conclusions.

    python -m llmperf.repeatability --runs 5 --settle 60
    python -m llmperf.repeatability --runs 6 --settle 0 --settle-ramp
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import statistics
import subprocess
import time
from pathlib import Path

from .common import (MODELS_DIR, RESULTS_DIR, discover_models, find_llama_bench,
                     git_commit, host_id, platform_tag, read_gguf_meta)
from .sweep import default_threads


def one_run(binary: Path, model: Path, threads: int, reps: int,
            prompt: int, gen: int, timeout: int) -> dict:
    cmd = [str(binary), "-m", str(model), "-p", str(prompt), "-n", str(gen),
           "-d", "0", "-ngl", "99", "-t", str(threads), "-r", str(reps),
           "-fa", "on", "-ctk", "f16", "-ctv", "f16", "-o", "json"]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or "")[-300:])
    rows = json.loads(proc.stdout)
    pp = next((r for r in rows if (r.get("n_prompt") or 0) and not (r.get("n_gen") or 0)), None)
    tg = next((r for r in rows if (r.get("n_gen") or 0) > 0), None)
    return {"prefill_ts": float(pp["avg_ts"]) if pp else float("nan"),
            "decode_ts": float(tg["avg_ts"]) if tg else float("nan"),
            "decode_sd_within": float(tg.get("stddev_ts") or 0) if tg else float("nan")}


def cv(xs) -> float:
    """Coefficient of variation in percent."""
    xs = [x for x in xs if x == x]
    if len(xs) < 2:
        return float("nan")
    return statistics.stdev(xs) / statistics.mean(xs) * 100.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    ap.add_argument("--model", type=str, default=None,
                    help="filename substring; default is the smallest model")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--settle", type=float, default=60.0)
    ap.add_argument("--settle-ramp", action="store_true",
                    help="sweep the settle delay 0,15,30,60,120s instead of "
                         "holding it fixed, to find how long the machine "
                         "actually needs to return to a steady thermal state")
    ap.add_argument("--repetitions", type=int, default=5)
    ap.add_argument("--prompt", type=int, default=512)
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--timeout", type=int, default=1800)
    args = ap.parse_args(argv)

    models = discover_models(args.models_dir)
    if not models:
        raise SystemExit(f"No GGUF in {args.models_dir}")
    if args.model:
        models = [m for m in models if args.model.lower() in m.name.lower()] or models
    model = min(models, key=lambda p: p.stat().st_size)

    binary = find_llama_bench()
    threads = default_threads()
    meta = read_gguf_meta(model)

    settles = ([0, 15, 30, 60, 120] * ((args.runs // 5) + 1))[:args.runs] \
        if args.settle_ramp else [args.settle] * args.runs

    print(f"host={host_id()} ({platform_tag()}) threads={threads}")
    print(f"model={meta.name} {meta.file_bytes/1e9:.2f} GB {meta.quant}")
    print(f"{args.runs} runs, settle={'ramp ' + str(settles) if args.settle_ramp else str(args.settle) + 's'}, "
          f"-r {args.repetitions} within each run\n")

    out = RESULTS_DIR / f"repeatability_{host_id()}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    new = not out.exists()
    fh = out.open("a", newline="", encoding="utf-8")
    w = csv.DictWriter(fh, fieldnames=[
        "timestamp", "host", "platform", "git_commit", "model", "quant",
        "run_index", "settle_s", "prefill_ts", "decode_ts", "decode_sd_within",
        "elapsed_s"])
    if new:
        w.writeheader()

    rows = []
    for i, settle in enumerate(settles, 1):
        if settle:
            print(f"  settling {settle:.0f}s ... ", end="", flush=True)
            time.sleep(settle)
        print(f"run {i}/{args.runs} ... ", end="", flush=True)
        t0 = time.time()
        try:
            r = one_run(binary, model, threads, args.repetitions,
                        args.prompt, args.gen, args.timeout)
        except Exception as e:
            print(f"FAIL: {e}")
            continue
        el = time.time() - t0
        rows.append({**r, "settle_s": settle, "run_index": i})
        w.writerow({"timestamp": dt.datetime.now().astimezone().isoformat(),
                    "host": host_id(), "platform": platform_tag(),
                    "git_commit": git_commit(), "model": meta.name,
                    "quant": meta.quant, "run_index": i, "settle_s": settle,
                    "elapsed_s": round(el, 1), **r})
        fh.flush()
        print(f"prefill={r['prefill_ts']:7.1f}  decode={r['decode_ts']:6.2f} "
              f"(within-run sd {r['decode_sd_within']:.2f})")
    fh.close()

    if len(rows) < 2:
        print("\nnot enough successful runs to estimate variance")
        return 1

    dec = [r["decode_ts"] for r in rows]
    pre = [r["prefill_ts"] for r in rows]
    within = statistics.mean(r["decode_sd_within"] / r["decode_ts"] * 100
                             for r in rows if r["decode_ts"])

    print(f"\n=== repeatability over {len(rows)} independent runs ===")
    print(f"  decode  mean {statistics.mean(dec):7.2f} tok/s   "
          f"min {min(dec):7.2f}  max {max(dec):7.2f}  "
          f"spread {(max(dec)-min(dec))/statistics.mean(dec)*100:5.1f}%")
    print(f"  prefill mean {statistics.mean(pre):7.1f} tok/s   "
          f"min {min(pre):7.1f}  max {max(pre):7.1f}  "
          f"spread {(max(pre)-min(pre))/statistics.mean(pre)*100:5.1f}%")
    print(f"\n  between-run CV (decode) : {cv(dec):5.2f}%   <-- the real noise floor")
    print(f"  within-run  CV (decode) : {within:5.2f}%   <-- what llama-bench -r reports")
    print(f"  between-run CV (prefill): {cv(pre):5.2f}%")

    ratio = cv(dec) / within if within else float("inf")
    print(f"\n  between/within ratio: {ratio:.1f}x")
    if ratio > 2:
        print("  WARNING: llama-bench's own stddev badly understates the true "
              "noise floor. Reporting -r stddev as the error bar would be "
              "misleading; error bars must come from independent runs.")
    print(f"\n  Any effect smaller than ~{2*cv(dec):.1f}% (2 sigma) is not "
          f"measurable on this host under this protocol.")

    if args.settle_ramp:
        print("\n=== decode vs settle delay ===")
        for r in sorted(rows, key=lambda r: r["settle_s"]):
            print(f"   settle {r['settle_s']:5.0f}s -> {r['decode_ts']:7.2f} tok/s")

    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
