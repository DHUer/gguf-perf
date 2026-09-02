"""Unattended campaign driver: fetch one model, sweep it idle, analyze, repeat.

Replaces the original shell script so there is ONE driver rather than a bash
version and a PowerShell version drifting apart. Runs identically on macOS and
Windows.

Never downloads while measuring: concurrent I/O costs ~20% throughput and would
poison every timing number (measured here: 1.6% CV idle vs 15-26% under load).

Resumable and idempotent. Fetch skips files already on disk, sweep skips
completed cells, so killing and restarting loses nothing.

    python -m llmperf.campaign                 # run to completion
    python -m llmperf.campaign --max-iters 3
    python -m llmperf.campaign --max-iters 0   # skip fetching, analysis only
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from .common import MODELS_DIR, RESULTS_DIR, ROOT, host_id, platform_tag

LOCK = RESULTS_DIR / "campaign.lock.json"
LOG = RESULTS_DIR / "campaign.log"
STATUS = RESULTS_DIR / "STATUS.md"

# A background thread refreshes the heartbeat continuously, so the staleness
# window only has to cover a missed tick or two rather than the longest step.
# Pinning it to the longest step instead would be wrong in both directions: a
# 63 GB fetch runs over four hours, while a killed campaign would block a
# restart for that whole window despite nothing running.
HEARTBEAT_EVERY_S = 30
STALE_AFTER_S = 5 * 60

_stop_beating = threading.Event()


def log(msg: str) -> None:
    line = f"[{dt.datetime.now():%F %T}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def acquire_lock() -> bool:
    """Refuse to start if another campaign holds the lock.

    Liveness is a heartbeat timestamp, NOT os.kill(pid, 0). On Windows
    os.kill with a non-CTRL signal calls TerminateProcess, so the usual Unix
    liveness idiom would kill the very process it is checking for.

    Two concurrent campaigns would run two sweeps at once, and concurrent load
    is exactly the contamination this project already lost hours to.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if LOCK.exists():
        try:
            held = json.loads(LOCK.read_text(encoding="utf-8"))
            age = time.time() - float(held.get("heartbeat", 0))
        except Exception:
            held, age = {}, STALE_AFTER_S + 1
        if age < STALE_AFTER_S:
            log(f"another campaign holds the lock (pid {held.get('pid')}, "
                f"heartbeat {age:.0f}s ago); refusing to start a second")
            return False
        log(f"clearing stale lock (last heartbeat {age / 3600:.1f}h ago)")

    beat()
    return True


def beat() -> None:
    LOCK.write_text(json.dumps({
        "pid": os.getpid(), "host": host_id(), "heartbeat": time.time(),
        "started": dt.datetime.now().astimezone().isoformat(),
    }), encoding="utf-8")


def release_lock() -> None:
    _stop_beating.set()
    try:
        held = json.loads(LOCK.read_text(encoding="utf-8"))
        if held.get("pid") == os.getpid():
            LOCK.unlink()
    except Exception:
        pass


def start_heartbeat() -> None:
    """Refresh the lock on a timer for as long as this process lives.

    Beating only between stages would leave the lock looking dead during a
    multi-hour download, so the window would have to be hours wide, which in
    turn means a killed campaign blocks a restart for hours. A ticking
    heartbeat decouples the two.
    """
    def loop():
        while not _stop_beating.wait(HEARTBEAT_EVERY_S):
            try:
                beat()
            except Exception:
                pass
    threading.Thread(target=loop, daemon=True, name="heartbeat").start()


def install_signal_handlers() -> None:
    """Release the lock on SIGTERM/SIGINT.

    Python does not run `finally` blocks on an un-handled SIGTERM, so without
    this a terminated campaign leaves its lock behind and the next run has to
    wait out the staleness window for no reason.
    """
    def handler(signum, _frame):
        log(f"received signal {signum}; releasing lock and exiting")
        release_lock()
        raise SystemExit(128 + signum)

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, handler)
        except (ValueError, OSError):
            pass  # not on the main thread, or unsupported on this platform


def run(args: list[str], capture: Path | None = None) -> int:
    """Run a pipeline stage in a child process, appending output to the log."""
    cmd = [sys.executable, "-m"] + args
    beat()
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    out = (proc.stdout or "") + (proc.stderr or "")
    if capture:
        capture.write_text(out, encoding="utf-8")
    with LOG.open("a", encoding="utf-8") as f:
        f.write(out)
    beat()
    return proc.returncode


def n_models() -> int:
    return len(list(MODELS_DIR.glob("*.gguf"))) if MODELS_DIR.is_dir() else 0


def n_missing() -> int:
    """Models in the manifest that are not yet on disk."""
    cmd = [sys.executable, "-m", "llmperf.fetch", "--set", "all", "--dry-run"]
    try:
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                              timeout=600)
    except subprocess.TimeoutExpired:
        return -1
    return sum(1 for line in proc.stdout.splitlines() if line.startswith("  [    ]"))


def missing_list() -> list[str]:
    cmd = [sys.executable, "-m", "llmperf.fetch", "--set", "all", "--dry-run"]
    try:
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                              timeout=600)
    except subprocess.TimeoutExpired:
        return []
    return [l for l in proc.stdout.splitlines() if l.startswith("  [    ]")]


def section(text: str, start: str) -> str:
    """Pull one '=== heading ===' block out of a captured stage's output."""
    lines, out, on = text.splitlines(), [], False
    for line in lines:
        if line.startswith(start):
            on = True
        elif on and not line.strip():
            break
        if on:
            out.append(line)
    return "\n".join(out) or "(not produced)"


def write_status(analyze_out: str, refine_out: str) -> None:
    hosts = sorted(RESULTS_DIR.glob("measurements_*.csv"))
    rows = 0
    mine = RESULTS_DIR / f"measurements_{host_id()}.csv"
    if mine.exists():
        rows = max(0, sum(1 for _ in mine.open(encoding="utf-8")) - 1)

    miss = missing_list()
    parts = [
        "# Campaign status", "",
        f"Generated {dt.datetime.now():%F %T} on `{host_id()}` "
        f"({platform_tag()}) by `llmperf.campaign`.",
        "Regenerated on every campaign completion — do not hand-edit.", "",
        "## Coverage", "",
        f"- models on disk: {n_models()}",
        f"- models still to fetch: {len(miss)}",
        f"- measurement rows (this host): {rows}",
        f"- hosts with results: {len(hosts)} "
        f"({', '.join(h.stem.replace('measurements_', '') for h in hosts) or 'none'})",
        "",
    ]
    if miss:
        parts += ["### Not yet fetched", "", "```", *miss, "```", ""]
    parts += [
        "## Prediction error", "", "```",
        section(analyze_out, "=== prediction error ==="), "```", "",
        "## Cross-model eta spread (go/no-go, paper section 5.7)", "", "```",
        section(refine_out, "=== eta spread"), "```", "",
        "## Next", "",
    ]
    if len(hosts) < 2:
        parts += [
            "**Only one host has results.** Leave-one-machine-out is undefined "
            "and no cross-architecture claim is admissible. Run the same "
            "commands on the other machines — see HANDOFF.md, \"Next, in "
            "priority order\".",
        ]
    else:
        parts += ["Multiple hosts present. Check the leave-one-machine-out "
                  "section of `results/_refine.out`."]
    STATUS.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-iters", type=int, default=999)
    ap.add_argument("--settle", type=float,
                    default=float(os.environ.get("LLMPERF_SETTLE", 45)))
    ap.add_argument("--repetitions", type=int, default=5)
    ap.add_argument("--depths", type=int, nargs="+", default=[0, 4096, 16384])
    ap.add_argument("--ngl", type=int, nargs="+", default=None,
                    help="offload levels; on a discrete GPU use "
                         "--ngl 0 8 16 24 32 48 99 to trace the cliff")
    ap.add_argument("--max-file-gb", type=float, default=None,
                    help="skip models larger than this; a model that does not "
                         "fit this machine cannot be benchmarked here")
    ap.add_argument("--give-up-after", type=int, default=5,
                    help="consecutive failed fetches before abandoning the fetch loop")
    args = ap.parse_args(argv)

    if not acquire_lock():
        return 1
    install_signal_handlers()
    start_heartbeat()

    try:
        log(f"campaign start on {host_id()} ({platform_tag()}) pid {os.getpid()}: "
            f"max_iters={args.max_iters} settle={args.settle}s")

        if run(["llmperf.calibrate"]) == 0:
            log("calibration ok")
        else:
            log("calibration FAILED — analysis will not be able to fit")

        sweep_args = ["llmperf.sweep", "--settle", str(args.settle),
                      "--repetitions", str(args.repetitions),
                      "--depths", *[str(d) for d in args.depths]]
        if args.ngl:
            sweep_args += ["--ngl", *[str(n) for n in args.ngl]]

        fetch_args = ["llmperf.fetch", "--set", "all", "--limit", "1"]
        if args.max_file_gb is not None:
            fetch_args += ["--max-file-gb", str(args.max_file_gb)]

        stalled = 0
        for i in range(1, args.max_iters + 1):
            before = n_models()
            log(f"iter {i}: fetching one model")
            run(fetch_args)
            after = n_models()
            log(f"iter {i}: {after} models on disk (was {before})")

            log(f"iter {i}: sweeping")
            if run(sweep_args) != 0:
                log(f"iter {i}: sweep returned nonzero")
            run(["llmperf.analyze"])

            left = n_missing()
            if left == 0:
                log("all models fetched and swept")
                break

            # A failed download also leaves the count unchanged, so "no new
            # model" must not be read as "done" — that bug silently abandoned
            # an earlier campaign after a single connection reset.
            if after == before:
                stalled += 1
                log(f"iter {i}: fetch made no progress ({stalled} in a row), "
                    f"{left} still missing")
                if stalled >= args.give_up_after:
                    log(f"GIVING UP on fetching after {stalled} consecutive "
                        f"failures; {left} models unfetched. See the [FAIL] "
                        f"lines above. Analysis still runs on what is on disk.")
                    break
                time.sleep(min(300, 30 * stalled))
            else:
                stalled = 0
                log(f"iter {i}: {left} models still to fetch")

        log("running analysis suite")
        a_out = RESULTS_DIR / "_analyze.out"
        r_out = RESULTS_DIR / "_refine.out"
        if run(["llmperf.analyze"], capture=a_out) != 0:
            log("analyze FAILED")
        if run(["llmperf.refine"], capture=r_out) != 0:
            log("refine FAILED")

        write_status(
            a_out.read_text(encoding="utf-8") if a_out.exists() else "",
            r_out.read_text(encoding="utf-8") if r_out.exists() else "",
        )
        log(f"wrote {STATUS}")
        log("campaign end")
        if a_out.exists():
            print(section(a_out.read_text(encoding="utf-8"),
                          "=== prediction error ==="))
        return 0
    finally:
        release_lock()


if __name__ == "__main__":
    raise SystemExit(main())
