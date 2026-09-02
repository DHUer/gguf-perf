"""Stage 3: run the measurement sweep by driving llama-bench.

We do not write our own timing harness. llama-bench already does warmup runs,
repeats each test, and reports mean and standard deviation for prefill and
decode separately. Reimplementing that would add bugs, not rigour.

What this script adds on top is the experimental discipline the paper needs:

  * Every confounding llama-bench default is PINNED, not left on "auto".
    In particular -fa defaults to auto, which silently selects a different
    attention kernel depending on model and backend -- that alone would make
    cross-model comparisons invalid.
  * One process per (model, n_gpu_layers) cell, so a crash or OOM on a large
    model cannot corrupt the rest of the sweep, and memory fragmentation does
    not accumulate across a multi-hour queue.
  * A settling delay before each cell so a thermally-throttled laptop is not
    measured while still hot from the previous cell.
  * Append-only, resumable CSV keyed on the full cell identity, with the git
    commit recorded on every row.

    python -m llmperf.sweep --dry-run
    python -m llmperf.sweep --repetitions 5 --settle 30
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

from .common import (FIGURES_DIR, MODELS_DIR, RESULTS_DIR, discover_models,
                     find_llama_bench, git_commit, host_id, platform_tag,
                     read_gguf_meta, write_json)

# Prefill is compute-bound and decode is memory-bound, so both are measured.
DEFAULT_PROMPT = 512
DEFAULT_GEN = 128
# KV-cache depths. The decode roofline predicts throughput decays as the KV
# cache grows; without a depth sweep that term is untestable.
DEFAULT_DEPTHS = [0, 4096, 16384]

CSV_FIELDS = [
    "timestamp", "host", "platform", "git_commit", "llama_bench_version",
    "model_file", "model_name", "quant", "arch", "split",
    "file_bytes", "n_params", "n_active_params", "n_layers", "n_kv_heads",
    "head_dim", "n_expert", "n_expert_used",
    "n_gpu_layers", "n_prompt", "n_gen", "n_depth", "n_threads",
    "flash_attn", "type_k", "type_v", "n_batch", "n_ubatch",
    "test", "avg_ts", "stddev_ts", "avg_ns", "stddev_ns", "samples_ts",
    "settle_s", "repetitions", "backend", "load_before", "load_after",
    "attempts", "kept_cv_pct", "max_cv_pct", "error",
]


def cell_key(row: dict) -> tuple:
    """Identity of one measured row, used for resume.

    Keyed on PHASE rather than on the raw (n_prompt, n_gen) pair. llama-bench
    emits prefill and decode as separate rows -- prefill carries n_gen=0 and
    decode carries n_prompt=0 -- so a key built from the requested (prompt,
    gen) values matches neither, resume silently never fires, and every run
    re-measures the whole grid. That failure is invisible in the logs except
    as an implausible "N cells already done ... 0 skipped".
    """
    try:
        gen = int(float(row.get("n_gen") or 0))
    except (TypeError, ValueError):
        gen = 0
    return (
        str(row.get("host")), str(row.get("model_file")), str(row.get("n_gpu_layers")),
        "decode" if gen > 0 else "prefill", str(row.get("n_depth")),
    )


def load_done(csv_path: Path) -> set:
    if not csv_path.exists():
        return set()
    done = set()
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row.get("error"):
                done.add(cell_key(row))
    return done


def migrate_csv_schema(csv_path: Path) -> None:
    """Rewrite an existing results CSV when CSV_FIELDS has gained columns.

    Appending with a DictWriter built from the current CSV_FIELDS writes one
    value per current column, but the file on disk still carries the header it
    was created with. Every new row is then wider than its own header and the
    file no longer parses — silently, until something tries to read it.

    Adding a column is routine as the protocol improves (load average, retry
    bookkeeping), so the append path has to reconcile the header rather than
    assume it. Old rows get empty strings for the new columns; a backup of the
    original is kept beside it.
    """
    if not csv_path.exists():
        return
    with csv_path.open(newline="", encoding="utf-8") as f:
        header = next(csv.reader(f), None)
    if header is None or header == CSV_FIELDS:
        return

    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    added = [c for c in CSV_FIELDS if c not in header]
    dropped = [c for c in header if c not in CSV_FIELDS]

    backup = csv_path.with_suffix(csv_path.suffix + ".pre-migration.bak")
    if not backup.exists():
        csv_path.replace(backup)
    else:
        backup = None

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({c: row.get(c, "") for c in CSV_FIELDS})
    print(f"migrated {csv_path.name} schema: +{len(added)} column(s) "
          f"{added}" + (f", dropped {dropped}" if dropped else "")
          + (f"; backup at {backup.name}" if backup else ""))


def llama_bench_version(binary: Path) -> str:
    try:
        out = subprocess.run([str(binary), "--version"], capture_output=True,
                             text=True, timeout=60)
        return (out.stdout + out.stderr).strip().splitlines()[0][:120]
    except Exception:
        return "unknown"


def load_average() -> float:
    """1-minute load average, or NaN where unavailable.

    Recorded around every cell because "the machine was idle" is an assumption,
    not a fact. On a corporate-managed host, endpoint-monitoring and
    log-shipping agents can hold the load average near 100 with no user
    processes running at all, and a throughput number taken then is not
    comparable to one taken in a quiet window. Storing it makes contaminated
    cells filterable after the fact instead of silently poisoning the dataset.
    """
    # os.getloadavg() does not exist on Windows. psutil emulates it there, and
    # without this the gate degrades to a no-op on exactly the platform the
    # discrete-GPU measurements come from -- silently, because NaN compares
    # false against any threshold.
    try:
        import psutil
        return float(psutil.getloadavg()[0])
    except Exception:
        pass
    try:
        return float(os.getloadavg()[0])
    except (OSError, AttributeError):
        return float("nan")


def wait_for_quiet(max_load: float, timeout_s: float, poll_s: float = 30.0) -> float:
    """Block until the load average drops below max_load, or give up.

    Retrying on a high within-run CV (see run_cell_quality_gated) reacts to
    contention after paying for it. This refuses to start in the first place.

    It matters because the operator cannot reliably pick a quiet moment: on the
    host used here the one-minute load average swung between 12 and 296 within a
    single afternoon, entirely from managed background agents, with no user
    processes involved. A sweep launched at the wrong moment produces numbers
    that look fine and are wrong by a factor of two.

    Returns the load average it settled at. Never raises: on a host that is
    permanently busy, measuring with a recorded and flagged load beats not
    measuring at all, so this warns and proceeds.
    """
    start = time.time()
    load = load_average()
    if not (load > max_load):          # also covers NaN
        return load
    print(f"    load {load:.0f} > {max_load:.0f}, waiting for the machine to "
          f"settle (up to {timeout_s / 60:.0f} min)", flush=True)
    while time.time() - start < timeout_s:
        time.sleep(poll_s)
        load = load_average()
        if load <= max_load:
            print(f"    load {load:.0f}, proceeding "
                  f"(waited {(time.time() - start) / 60:.1f} min)", flush=True)
            return load
    print(f"    WARNING: load still {load:.0f} after {timeout_s / 60:.0f} min; "
          f"measuring anyway. This cell is flagged and should be treated as "
          f"provisional.", flush=True)
    return load


def default_threads() -> int:
    """Physical performance cores where we can tell, else a sane fallback.

    Oversubscribing threads is a classic silent confound in CPU-side numbers,
    and the naive os.cpu_count() // 2 fallback is badly wrong on a
    many-core workstation: a 64-thread Threadripper would get 32 llama.cpp
    threads, well past the point where memory-bound decode stops scaling and
    starts thrashing.
    """
    system = platform.system()
    try:
        if system == "Darwin":
            # Performance cores only; the efficiency cores hurt more than help.
            out = subprocess.run(["sysctl", "-n", "hw.perflevel0.physicalcpu"],
                                 capture_output=True, text=True, timeout=5)
            n = int(out.stdout.strip())
            if n > 0:
                return n
        elif system == "Windows":
            # NUMBER_OF_PROCESSORS counts logical CPUs; ask for physical cores.
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_Processor | "
                 "Measure-Object -Property NumberOfCores -Sum).Sum"],
                capture_output=True, text=True, timeout=20)
            n = int(out.stdout.strip())
            if n > 0:
                # Beyond ~16 threads batch-1 decode is memory-bound and extra
                # threads only add contention.
                return min(n, 16)
        elif system == "Linux":
            out = subprocess.run(["nproc", "--all"], capture_output=True,
                                 text=True, timeout=5)
            n = int(out.stdout.strip())
            if n > 0:
                return min(max(1, n // 2), 16)
    except Exception:
        pass
    return max(1, min((os.cpu_count() or 8) // 2, 16))


def decode_cv(rows: list[dict]) -> float:
    """Worst within-run coefficient of variation across the decode rows.

    llama-bench already reports a standard deviation over its repetitions. That
    number is the instrument telling us how much it trusts itself, and it is
    available before any analysis — which makes it a legitimate basis for a
    retry rule, unlike filtering on the residual after the fact.
    """
    worst = 0.0
    for r in rows:
        if (r.get("n_gen") or 0) > 0 and r.get("avg_ts"):
            worst = max(worst, abs(float(r.get("stddev_ts") or 0))
                        / float(r["avg_ts"]) * 100.0)
    return worst


def run_cell_quality_gated(binary: Path, model: Path, ngl: int, prompt: int,
                           gen: int, depths: list[int], threads: int, reps: int,
                           timeout: int, max_cv: float, retries: int,
                           settle: float) -> tuple[list[dict], str, float, int]:
    """Measure a cell, retrying while the instrument reports instability.

    The rule is declared before measuring, not chosen after seeing which cells
    predicted badly: if a cell's own within-run CV exceeds max_cv, the machine
    was disturbed during it, so settle longer and take it again. The lowest-CV
    attempt is kept and the attempt count is recorded.

    This is what separates a quality protocol from cherry-picking — the
    criterion is a property of the measurement, is fixed in advance, and is
    reported with the data.
    """
    best_rows, best_cv, best_err = [], float("inf"), ""
    attempts = 0
    for attempt in range(1, retries + 2):
        attempts = attempt
        rows, err = run_cell(binary, model, ngl, prompt, gen, depths,
                             threads, reps, timeout)
        if err:
            if not best_rows:
                best_err = err
            break
        cv = decode_cv(rows)
        if cv < best_cv:
            best_rows, best_cv, best_err = rows, cv, ""
        if cv <= max_cv:
            break
        if attempt <= retries:
            # Back off further each time: a disturbed machine usually needs
            # longer than the nominal settle to return to a steady state.
            time.sleep(settle * (attempt + 1))
    return best_rows, best_err, best_cv, attempts


def run_cell(binary: Path, model: Path, ngl: int, prompt: int, gen: int,
             depths: list[int], threads: int, reps: int, timeout: int) -> tuple[list[dict], str]:
    """One llama-bench invocation covering all depths for this (model, ngl).

    Batching depths into a single call reuses one model load, which for a 20 GB
    model saves more wall-clock than everything else in this script combined.
    """
    cmd = [
        str(binary),
        "-m", str(model),
        "-p", str(prompt),
        "-n", str(gen),
        "-d", ",".join(str(d) for d in depths),
        "-ngl", str(ngl),
        "-t", str(threads),
        "-r", str(reps),
        # --- pinned to remove confounds; see module docstring ---
        "-fa", "on",
        "-ctk", "f16",
        "-ctv", "f16",
        "-o", "json",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return [], f"timeout after {timeout}s"

    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()
        return [], f"exit {proc.returncode}: {' | '.join(tail[-3:])[:400]}"

    try:
        return json.loads(proc.stdout), ""
    except json.JSONDecodeError as e:
        return [], f"unparseable json: {e}; stdout head={proc.stdout[:200]!r}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--prompt", type=int, default=DEFAULT_PROMPT)
    ap.add_argument("--gen", type=int, default=DEFAULT_GEN)
    ap.add_argument("--depths", type=int, nargs="+", default=DEFAULT_DEPTHS)
    ap.add_argument("--ngl", type=int, nargs="+", default=[99],
                    help="n_gpu_layers levels. Use e.g. --ngl 0 8 16 24 32 99 on a "
                         "discrete GPU to trace the CPU-offload cliff.")
    ap.add_argument("--repetitions", type=int, default=5)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--settle", type=float, default=20.0,
                    help="seconds of idle before each cell, so a throttled "
                         "laptop is not measured while hot from the last cell")
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--max-cv", type=float, default=3.0,
                    help="retry a cell whose own within-run CV exceeds this "
                         "percentage. A pre-declared measurement-quality rule, "
                         "not a post-hoc filter on the residual.")
    ap.add_argument("--cv-retries", type=int, default=2,
                    help="extra attempts allowed per cell before keeping the best")
    ap.add_argument("--max-load", type=float, default=None,
                    help="wait for the 1-minute load average to fall below this "
                         "before each cell. Defaults to 1.5x the thread count. "
                         "Set 0 to disable.")
    ap.add_argument("--load-wait", type=float, default=900,
                    help="seconds to wait for a quiet window before measuring anyway")
    ap.add_argument("--only", type=str, default=None,
                    help="restrict to models whose filename contains this "
                         "substring; use with --force to re-measure specific cells")
    ap.add_argument("--force", action="store_true",
                    help="re-measure cells already present in the CSV")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    models = discover_models(args.models_dir)
    if args.only:
        models = [m for m in models if args.only.lower() in m.name.lower()]
    if not models:
        print(f"No .gguf files in {args.models_dir}. Run "
              f"`python -m llmperf.fetch --set pilot` first.", file=sys.stderr)
        return 2

    binary = find_llama_bench()
    threads = args.threads or default_threads()
    out = args.out or (RESULTS_DIR / f"measurements_{host_id()}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)

    # Model metadata is read once per file; it is static and GGUF parsing of a
    # 60 GB file is not free.
    metas = {}
    for m in models:
        try:
            metas[m] = read_gguf_meta(m)
        except Exception as e:
            print(f"[SKIP] {m.name}: cannot read GGUF metadata: {e}", file=sys.stderr)

    cells = [(m, ngl) for m in models if m in metas for ngl in args.ngl]
    done = load_done(out)

    print(f"host={host_id()} platform={platform_tag()} threads={threads}")
    print(f"llama-bench: {binary}")
    print(f"{len(metas)} models x {len(args.ngl)} ngl levels = {len(cells)} cells")
    print(f"depths={args.depths} prompt={args.prompt} gen={args.gen} "
          f"reps={args.repetitions} settle={args.settle}s "
          f"max_cv={args.max_cv}% max_load="
          f"{args.max_load if args.max_load is not None else 1.5 * threads:.0f}")
    print(f"output: {out}  ({len(done)} cells already done)\n")

    for m in models:
        if m in metas:
            mm = metas[m]
            print(f"  {mm.name:52s} {mm.quant:8s} {mm.file_bytes/1e9:6.2f} GB "
                  f"{mm.n_params/1e9:6.2f}B total / {mm.n_active_params/1e9:5.2f}B active"
                  f"{'  [MoE]' if mm.n_expert else ''}")

    if args.dry_run:
        return 0

    max_load = (args.max_load if args.max_load is not None else 1.5 * threads)
    version = llama_bench_version(binary)
    write_json(RESULTS_DIR / f"env_{host_id()}.json", {
        "host": host_id(), "platform": platform_tag(),
        "python": sys.version, "llama_bench": str(binary),
        "llama_bench_version": version, "threads": threads,
        "git_commit": git_commit(),
        "cpu": platform.processor() or platform.machine(),
        "timestamp": dt.datetime.now().astimezone().isoformat(),
        "protocol": {
            "flash_attn": "on (pinned)", "cache_type_k": "f16", "cache_type_v": "f16",
            "repetitions": args.repetitions, "settle_seconds": args.settle,
            "warmup": "llama-bench default (enabled)",
        },
    })

    migrate_csv_schema(out)
    new_file = not out.exists()
    fh = out.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
    if new_file:
        writer.writeheader()

    commit = git_commit()
    n_run = n_skip = n_fail = 0
    try:
        for i, (model, ngl) in enumerate(cells, 1):
            mm = metas[model]
            # A cell counts as done only if the decode row for EVERY requested
            # depth is present, plus the prefill row. Checking one depth would
            # leave holes when the sweep is rerun with more depths.
            base = {"host": host_id(), "model_file": model.name,
                    "n_gpu_layers": ngl}
            wanted = [{**base, "n_gen": args.gen, "n_depth": d} for d in args.depths]
            wanted.append({**base, "n_gen": 0, "n_depth": args.depths[0]})
            if not args.force and all(cell_key(w) in done for w in wanted):
                n_skip += 1
                continue

            print(f"[{i}/{len(cells)}] {mm.name} ngl={ngl} ... ", end="", flush=True)
            if args.settle:
                time.sleep(args.settle)

            if max_load > 0:
                wait_for_quiet(max_load, args.load_wait)
            load_before = load_average()
            t0 = time.time()
            rows, err, kept_cv, attempts = run_cell_quality_gated(
                binary, model, ngl, args.prompt, args.gen, args.depths, threads,
                args.repetitions, args.timeout, args.max_cv, args.cv_retries,
                args.settle)
            dur = time.time() - t0
            load_after = load_average()

            base = {
                "timestamp": dt.datetime.now().astimezone().isoformat(),
                "host": host_id(), "platform": platform_tag(), "git_commit": commit,
                "llama_bench_version": version,
                "model_file": model.name, "model_name": mm.name, "quant": mm.quant,
                "arch": mm.arch, "split": "",
                "file_bytes": mm.file_bytes, "n_params": mm.n_params,
                "n_active_params": mm.n_active_params, "n_layers": mm.n_layers,
                "n_kv_heads": mm.n_kv_heads, "head_dim": mm.head_dim,
                "n_expert": mm.n_expert, "n_expert_used": mm.n_expert_used,
                "n_gpu_layers": ngl, "n_threads": threads,
                "settle_s": args.settle, "repetitions": args.repetitions,
                "load_before": round(load_before, 2), "load_after": round(load_after, 2),
                "attempts": attempts, "max_cv_pct": args.max_cv,
                "kept_cv_pct": round(kept_cv, 3) if kept_cv == kept_cv and
                               kept_cv != float("inf") else "",
            }

            if err:
                writer.writerow({**base, "n_prompt": args.prompt, "n_gen": 0,
                                 "n_depth": args.depths[0], "error": err})
                fh.flush()
                n_fail += 1
                print(f"FAIL ({dur:.0f}s): {err}")
                continue

            for r in rows:
                writer.writerow({**base,
                    "n_prompt": r.get("n_prompt"), "n_gen": r.get("n_gen"),
                    "n_depth": r.get("n_depth", 0), "test": r.get("test"),
                    "avg_ts": r.get("avg_ts"), "stddev_ts": r.get("stddev_ts"),
                    "avg_ns": r.get("avg_ns"), "stddev_ns": r.get("stddev_ns"),
                    "samples_ts": json.dumps(r.get("samples_ts", [])),
                    "flash_attn": r.get("flash_attn"), "type_k": r.get("type_k"),
                    "type_v": r.get("type_v"), "n_batch": r.get("n_batch"),
                    "n_ubatch": r.get("n_ubatch"),
                    "backend": r.get("backends") or r.get("gpu_info"),
                    "error": ""})
            fh.flush()
            n_run += 1
            tg = [r for r in rows if (r.get("n_gen") or 0) > 0]
            pp = [r for r in rows if (r.get("n_prompt") or 0) > 0 and not (r.get("n_gen") or 0)]
            summary = []
            if pp:
                summary.append(f"pp={pp[0].get('avg_ts', 0):.0f}")
            for r in tg:
                summary.append(f"tg@{r.get('n_depth', 0)}={r.get('avg_ts', 0):.1f}")
            warn = "  [LOAD]" if load_before > 4 * threads else ""
            if kept_cv > args.max_cv:
                warn += f"  [CV {kept_cv:.1f}% after {attempts} attempts]"
            print(f"ok ({dur:.0f}s) {' '.join(summary)} "
                  f"load={load_before:.0f}->{load_after:.0f} "
                  f"cv={kept_cv:.1f}%{warn}")
    finally:
        fh.close()

    print(f"\n{n_run} cells measured, {n_skip} skipped (already done), {n_fail} failed")
    print(f"-> {out}")
    return 0


def _selfcheck() -> None:
    """Regression guard for the resume key.

    The original key was built from the requested (n_prompt, n_gen) pair and
    could never match a written row, so resume silently never fired and every
    campaign iteration re-measured the entire grid. The re-measurements landed
    in a differently-loaded session and disagreed with the originals by up to
    3x, which is larger than any effect the study is trying to detect. Nothing
    about that was visible without reading the CSV, hence this check.
    """
    # Rows exactly as llama-bench emits them: prefill has n_gen=0, decode has
    # n_prompt=0. This asymmetry is the whole bug.
    written = [
        {"host": "h", "model_file": "m.gguf", "n_gpu_layers": 99,
         "n_prompt": 512, "n_gen": 0, "n_depth": 0},
        {"host": "h", "model_file": "m.gguf", "n_gpu_layers": 99,
         "n_prompt": 0, "n_gen": 128, "n_depth": 0},
        {"host": "h", "model_file": "m.gguf", "n_gpu_layers": 99,
         "n_prompt": 0, "n_gen": 128, "n_depth": 4096},
    ]
    done = {cell_key(r) for r in written}
    base = {"host": "h", "model_file": "m.gguf", "n_gpu_layers": 99}

    # What the sweep asks for must match what was written.
    assert cell_key({**base, "n_gen": 128, "n_depth": 0}) in done
    assert cell_key({**base, "n_gen": 128, "n_depth": 4096}) in done
    assert cell_key({**base, "n_gen": 0, "n_depth": 0}) in done, "prefill must resume"

    # A depth never measured must NOT count as done, or the grid gets holes.
    assert cell_key({**base, "n_gen": 128, "n_depth": 16384}) not in done

    # Prefill and decode at the same depth are distinct cells.
    assert cell_key({**base, "n_gen": 0, "n_depth": 0}) != \
           cell_key({**base, "n_gen": 128, "n_depth": 0})

    # Different model / host / offload level are distinct cells.
    assert cell_key({**base, "n_gen": 128, "n_depth": 0}) != \
           cell_key({**base, "model_file": "other.gguf", "n_gen": 128, "n_depth": 0})
    assert cell_key({**base, "n_gen": 128, "n_depth": 0}) != \
           cell_key({**base, "n_gpu_layers": 0, "n_gen": 128, "n_depth": 0})

    # CSV round-trips everything as strings; the key must survive that.
    as_str = {k: str(v) for k, v in written[1].items()}
    assert cell_key(as_str) == cell_key(written[1]), "key must be str/int agnostic"

    print("sweep.py selfcheck ok")


if __name__ == "__main__":
    import sys as _sys
    if "--selfcheck" in _sys.argv:
        _selfcheck()
    else:
        raise SystemExit(main())
