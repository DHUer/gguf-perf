"""Stage 1: download GGUF models from HuggingFace into models/.

Model selection rules (these are methodological, not cosmetic):

1. Current generation only. Everything here is from the 2026 model cohort;
   benchmarking 2024 models would date the paper before it is reviewed.
2. Official or well-known faithful quantisations only (ggml-org, google,
   unsloth, lmstudio-community, bartowski). Community finetunes with modified
   weights ("uncensored", "abliterated", "heretic", merges) are excluded --
   they change the weights, which would confound a controlled study.
3. Dense and MoE in both TRAIN and TEST. MoE is the dominant 2026 architecture
   and it decouples capacity (total params) from bandwidth (active params),
   which is the effect this paper is built around.
4. TEST models are architecturally distinct from TRAIN so the reported error is
   genuine out-of-sample generalisation, not interpolation.

Filenames differ between GGUF repos, so we list what the repo actually has and
match by regex instead of hard-coding paths that rot.

    python -m llmperf.fetch --set pilot
    python -m llmperf.fetch --set train --dry-run     # show sizes, download nothing
    python -m llmperf.fetch --set all --max-gb 400
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

from .common import MODELS_DIR

# (repo_id, quant regex, split) where split is "train" or "test".
# Sizes span ~0.5 GB to ~65 GB so the roofline fit has real leverage instead of
# extrapolating from a narrow band.

# --- TRAIN: eta is fitted on these -----------------------------------------
# One quantisation per model, spanning the size axis. Dense first, then MoE.
TRAIN = [
    # Dense, small -> large
    ("ggml-org/SmolLM3-3B-GGUF",              r"Q4_K_M", "train"),
    ("ggml-org/SmolLM3-3B-GGUF",              r"Q8_0",   "train"),
    ("unsloth/Qwen3.5-4B-GGUF",               r"Q4_K_M", "train"),
    ("unsloth/Qwen3.5-4B-GGUF",               r"Q8_0",   "train"),
    ("unsloth/Qwen3.5-9B-GGUF",               r"Q4_K_M", "train"),
    ("unsloth/Qwen3.5-9B-GGUF",               r"Q8_0",   "train"),
    ("google/gemma-4-12B-it-qat-q4_0-gguf",   r"q4_0",   "train"),  # official QAT
    ("unsloth/Qwen3.8-27B-GGUF",              r"UD-Q4_K_XL", "train"),
    # MoE: capacity and bandwidth decouple here
    ("ggml-org/gpt-oss-20b-GGUF",             r"\.gguf$", "train"),  # native MXFP4
    ("unsloth/gemma-4-26B-A4B-it-GGUF",       r"Q4_K_M", "train"),
    ("unsloth/Qwen3.6-35B-A3B-GGUF",          r"Q4_K_M", "train"),
]

# --- LADDER: one model, many quantisations ---------------------------------
# Holding the model fixed and sweeping the quantisation format is the clean way
# to measure eta(format): any variation left is the format's dequantisation
# cost, not a difference between models. Comparing one quant per model would
# confound the two. Qwen3.8-27B is used because it ships a full ladder from
# IQ1_S to Q8_0 from a single publisher.
LADDER = [
    ("unsloth/Qwen3.8-27B-GGUF", r"UD-IQ2_XXS", "train"),
    ("unsloth/Qwen3.8-27B-GGUF", r"UD-Q2_K_XL", "train"),
    ("unsloth/Qwen3.8-27B-GGUF", r"UD-IQ3_XXS", "train"),
    ("unsloth/Qwen3.8-27B-GGUF", r"UD-Q3_K_XL", "train"),
    ("unsloth/Qwen3.8-27B-GGUF", r"UD-Q5_K_XL", "train"),
    ("unsloth/Qwen3.8-27B-GGUF", r"UD-Q6_K_XL", "train"),
    ("unsloth/Qwen3.8-27B-GGUF", r"Q8_0",       "train"),
]

# --- TEST: never used for fitting; this is the headline error number --------
TEST = [
    ("lmstudio-community/NVIDIA-Nemotron-3-Nano-4B-GGUF",      r"Q4_K_M", "test"),
    # Brand-new Meta family, absent from TRAIN entirely.
    ("unsloth/Muse-Glimmer-30B-GGUF",                          r"UD-Q4_K_XL", "test"),
    ("unsloth/Nemotron-3-Nano-30B-A3B-GGUF",                   r"Q4_K_M", "test"),
    ("bartowski/inclusionAI_Ling-mini-2.0-GGUF",               r"Q4_K_M", "test"),
    # Extreme extrapolation: ~117B total / ~5B active. Fits 128 GB unified
    # memory, cannot fit 16 GB VRAM. This single point is the paper's
    # capacity-vs-bandwidth argument made concrete.
    ("ggml-org/gpt-oss-120b-GGUF",                             r"\.gguf$", "test"),
]

# Small and fast: enough to validate the pipeline end to end in under an hour.
PILOT = [t for t in TRAIN if any(k in t[0] for k in ("SmolLM3", "Qwen3.5-4B"))]

SETS = {"pilot": PILOT, "train": TRAIN, "ladder": LADDER, "test": TEST,
        "all": TRAIN + LADDER + TEST}


def pick_file(repo_id: str, pattern: str):
    """Return (filename, size_bytes) for the single-file GGUF matching pattern."""
    from huggingface_hub import HfApi

    info = HfApi().model_info(repo_id, files_metadata=True)
    rx = re.compile(pattern, re.IGNORECASE)

    cands = []
    for s in info.siblings:
        name = s.rfilename
        base = Path(name).name
        if not name.endswith(".gguf"):
            continue
        if not rx.search(base):
            continue
        # Skip sharded weights: llama-bench wants one file, and shards break the
        # file-size term the whole performance model is built on.
        if re.search(r"-\d{5}-of-\d{5}\.gguf$", name):
            continue
        # Skip auxiliary companion weights that live in the same repo. These are
        # not language models, and picking one silently would benchmark the
        # wrong thing: mmproj is a vision projector, mtp is a speculative
        # multi-token-prediction head.
        low = base.lower()
        if low.startswith("mmproj") or low.startswith("mtp-") or "/" in name.strip("/"):
            continue
        cands.append((name, s.size or 0))

    if not cands:
        return None, 0
    # Shortest name is the plain variant (avoids -imat, -UD, -MTP suffixes).
    return min(cands, key=lambda c: len(c[0]))


def http_download(repo_id: str, filename: str, dest: Path, expect: int = 0,
                  attempts: int = 5) -> None:
    """Fetch with resume, retrying transient network failures.

    Multi-gigabyte transfers get reset by the CDN often enough that a
    single-shot download loses whole files. Each retry resumes from the .part
    file, so a reset costs seconds rather than restarting a 20 GB transfer.
    """
    last = None
    for attempt in range(1, attempts + 1):
        try:
            _http_download_once(repo_id, filename, dest, expect)
            return
        except (OSError, IOError) as e:      # includes ConnectionResetError
            last = e
            if attempt == attempts:
                break
            wait = min(60, 5 * attempt)
            print(f"\n    retry {attempt}/{attempts - 1} after "
                  f"{type(e).__name__}: {e}; waiting {wait}s", flush=True)
            time.sleep(wait)
    raise last


def _http_download_once(repo_id: str, filename: str, dest: Path, expect: int = 0) -> None:
    """Stream a file straight off the HF CDN, resuming a partial file.

    We deliberately do NOT use hf_hub_download here. Its Xet transfer path
    stalls at zero bytes on unauthenticated connections, while a plain ranged
    GET to the same CDN sustains full line rate. Pure stdlib, so this behaves
    the same on macOS and Windows without depending on curl being present.
    """
    import urllib.error
    import urllib.request

    url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
    part = dest.with_suffix(dest.suffix + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)

    have = part.stat().st_size if part.exists() else 0

    # A partial file at or beyond the expected size cannot be a valid resume
    # point. It means two writers appended to the same .part -- which happened
    # here when two campaign drivers with different locks ran at once. Resuming
    # from it would append forever. Start over instead.
    if expect and have >= expect:
        if have > expect:
            print(f"    discarding corrupt partial ({have} bytes > expected "
                  f"{expect}); restarting download")
            part.unlink()
            have = 0
        else:
            part.replace(dest)
            return

    req = urllib.request.Request(url, headers={"User-Agent": "llmperf/1.0"})
    token = os.environ.get("HF_TOKEN")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    if have:
        req.add_header("Range", f"bytes={have}-")

    try:
        resp = urllib.request.urlopen(req, timeout=60)
    except urllib.error.HTTPError as e:
        if e.code == 416 and expect and have >= expect:
            part.replace(dest)          # already complete
            return
        raise

    # A server that ignores our Range header would otherwise silently append a
    # second copy of the file onto the partial one.
    if have and resp.status != 206:
        have = 0
        part.unlink(missing_ok=True)

    total = have + int(resp.headers.get("Content-Length") or 0)
    mode = "ab" if have else "wb"
    last = time.time()
    with resp, part.open(mode) as f:
        while True:
            chunk = resp.read(4 * 1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            have += len(chunk)
            if time.time() - last > 5:
                pct = f"{have / total * 100:5.1f}%" if total else "  ?  "
                print(f"\r    {pct}  {have/1e9:6.2f} GB", end="", flush=True)
                last = time.time()
    print(f"\r    100.0%  {have/1e9:6.2f} GB")

    if expect and have != expect:
        raise IOError(f"size mismatch: got {have}, expected {expect}")
    part.replace(dest)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", dest="which", choices=sorted(SETS), default="pilot")
    ap.add_argument("--out", type=Path, default=MODELS_DIR)
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve filenames and report total size, download nothing")
    ap.add_argument("--max-gb", type=float, default=None,
                    help="abort before exceeding this total download size")
    ap.add_argument("--limit", type=int, default=None,
                    help="download at most N missing files then exit. Lets a "
                         "driver loop alternate fetching and measuring, so the "
                         "sweep never runs while a download is competing for "
                         "I/O -- concurrent load costs ~20%% throughput.")
    ap.add_argument("--max-file-gb", type=float, default=None,
                    help="skip individual files larger than this. A model that "
                         "does not fit a machine's memory cannot be benchmarked "
                         "there, so there is no point downloading it.")
    args = ap.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)

    plan, unresolved = [], []
    for repo_id, pattern, split in SETS[args.which]:
        try:
            fname, size = pick_file(repo_id, pattern)
        except Exception as e:
            print(f"[SKIP] {repo_id}: {type(e).__name__}: {e}", file=sys.stderr)
            unresolved.append(repo_id)
            continue
        if not fname:
            print(f"[SKIP] {repo_id}: no file matching /{pattern}/", file=sys.stderr)
            unresolved.append(repo_id)
            continue
        if args.max_file_gb is not None and size / 1e9 > args.max_file_gb:
            print(f"[skip>{args.max_file_gb}GB] {repo_id}/{Path(fname).name} "
                  f"({size/1e9:.1f} GB)")
            continue
        plan.append((repo_id, fname, size, split))

    todo_gb = sum(s for _, f, s, _ in plan
                  if not (args.out / Path(f).name).exists()) / 1e9
    print(f"\n{len(plan)} files resolved, {todo_gb:.1f} GB to download "
          f"into {args.out}\n")
    for repo_id, fname, size, split in plan:
        have = "have" if (args.out / Path(fname).name).exists() else "    "
        print(f"  [{have}] {split:5s} {size/1e9:7.2f} GB  {repo_id}/{Path(fname).name}")

    if args.max_gb is not None and todo_gb > args.max_gb:
        print(f"\nABORT: {todo_gb:.1f} GB exceeds --max-gb {args.max_gb}", file=sys.stderr)
        return 2
    if args.dry_run:
        return 0

    ok = failed = fetched = 0
    for repo_id, fname, size, split in plan:
        dest = args.out / Path(fname).name
        if dest.exists():
            ok += 1
            continue
        if args.limit is not None and fetched >= args.limit:
            break
        fetched += 1
        print(f"\n[fetch {size/1e9:.1f} GB] {repo_id}/{Path(fname).name}")
        try:
            http_download(repo_id, fname, dest, expect=size)
            ok += 1
        except Exception as e:
            print(f"[FAIL] {repo_id}/{fname}: {type(e).__name__}: {e}", file=sys.stderr)
            failed += 1

    # Record the train/test split at download time, not at analysis time. If the
    # split were decided later it could be nudged after seeing the errors, which
    # would quietly turn out-of-sample validation into cherry-picking.
    manifest_path = args.out / "manifest.json"
    manifest = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}
    for repo_id, fname, size, split in plan:
        manifest[Path(fname).name] = {"repo_id": repo_id, "split": split,
                                      "size_bytes": size}
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True),
                             encoding="utf-8")

    have = list(args.out.glob("*.gguf"))
    print(f"\n{ok} ok, {failed} failed, {len(unresolved)} unresolved. "
          f"{len(have)} GGUF files, {sum(p.stat().st_size for p in have)/1e9:.1f} GB.")
    print(f"split manifest -> {manifest_path}")
    return 1 if failed and not ok else 0


if __name__ == "__main__":
    raise SystemExit(main())
