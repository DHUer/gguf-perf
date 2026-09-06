"""Stage 4: fit the performance model, validate out-of-sample, emit figures.

THE MODEL
---------
Decode (memory-bound at batch 1). Every generated token requires reading the
active weights plus the KV cache:

    bytes_per_token = file_bytes * (n_active / n_total) + kv_bytes(depth)
    tok/s           = eta_d * BW / bytes_per_token

The active/total ratio is 1 for dense models. For a 2026-era MoE it can be 1/12,
which is the whole point: an MoE's CAPACITY cost scales with total parameters
but its BANDWIDTH cost scales with active parameters. Those two decouple, and
that decoupling is what makes a large-unified-memory machine behave completely
differently from a small-VRAM discrete GPU on the same model.

Prefill (compute-bound):

    pp_tok/s = eta_p * FLOPS / (2 * n_active_params)

eta_d and eta_p are the efficiency factors the naive roofline omits. They are
fitted on TRAIN models only and applied unchanged to TEST models, so the headline
error is genuine out-of-sample generalisation.

BASELINES (a reviewer will demand these)
    B0  unfitted roofline, eta = 1, total params
    B1  fitted, but total params -- isolates how much the MoE sparsity term
        is actually worth
    B2  this paper: fitted + active params + KV term

    python -m llmperf.analyze
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: must render identically on macOS and Windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import (FIGURES_DIR, MODELS_DIR, RESULTS_DIR, load_model_metadata,
                     load_model_splits)


# The predictor's declared baseline is llama.cpp's "offload as many layers as
# possible" setting. Lower values are a separate intervention used to draw the
# offload cliff; mixing those rows into eta fitting would make the headline
# error depend on how many offload points happened to be collected.
PRIMARY_N_GPU_LAYERS = 99
PREFILL_FIT_DEPTH = 0
DECLARED_REPETITIONS = 5
DECLARED_SETTLE_S = 45.0
DECLARED_MAX_CV_PCT = 3.0
DECLARED_DEPTHS = frozenset({0, 4096, 16384})
DECLARED_N_BATCH = 2048
DECLARED_N_UBATCH = 512
_QUALITY_BOOKKEEPING_FIELDS = (
    "load_before", "load_after", "attempts", "kept_cv_pct", "max_cv_pct",
)
_PROTOCOL_DECLARATION_FIELDS = (
    "repetitions", "settle_s", "max_cv_pct", "n_depth", "n_prompt", "n_gen",
    "flash_attn", "type_k", "type_v", "n_batch", "n_ubatch",
)

# Diagnostic plots are not publication figures, but host identity still needs
# a stable visual encoding.  In particular, inserting the Studio between the
# alphabetically sorted MacBook and RTX IDs must not silently change the RTX
# colour.  Unknown future hosts get a deterministic fallback rather than a
# position-dependent colour.
DIAGNOSTIC_HOST_STYLES = {
    "lun-mac": {"color": "#2a78d6", "marker": "o", "linestyle": "-"},
    "rtx5080": {"color": "#eb6834", "marker": "s", "linestyle": "--"},
    "mac-studio-m4-max": {
        "color": "#1baf7a", "marker": "D", "linestyle": "-."
    },
}
_FALLBACK_HOST_COLOURS = ("#eda100", "#e87ba4", "#008300")
_FALLBACK_HOST_MARKERS = ("^", "v", "P", "X", "*")
_FALLBACK_HOST_LINES = (":", (0, (3, 1, 1, 1)), (0, (5, 2)))
_HOST_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

plt.rcParams.update({
    "figure.dpi": 140, "savefig.dpi": 200, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False,
    "axes.spines.right": False, "figure.autolayout": True,
})


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def diagnostic_host_style(host: str) -> dict:
    """Return a stable style for a known or future host ID."""
    host = str(host)
    if host in DIAGNOSTIC_HOST_STYLES:
        return DIAGNOSTIC_HOST_STYLES[host]
    digest = hashlib.sha256(host.encode("utf-8")).digest()
    return {
        "color": _FALLBACK_HOST_COLOURS[digest[0] % len(_FALLBACK_HOST_COLOURS)],
        "marker": _FALLBACK_HOST_MARKERS[digest[1] % len(_FALLBACK_HOST_MARKERS)],
        "linestyle": _FALLBACK_HOST_LINES[digest[2] % len(_FALLBACK_HOST_LINES)],
    }


def _validate_host_id(value, source: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"missing or malformed host ID in {source}: {value!r}")
    if not _HOST_ID_RE.fullmatch(value):
        raise ValueError(
            f"malformed host ID in {source}: {value!r}; use only letters, "
            "digits, dot, underscore, and hyphen")
    return value


def _filename_host_id(path: Path, prefix: str, suffix: str) -> str:
    name = path.name
    if not name.startswith(prefix) or not name.endswith(suffix):
        raise ValueError(f"unexpected result filename: {path}")
    return _validate_host_id(name[len(prefix):-len(suffix)], str(path))


def _numeric_column(df: pd.DataFrame, name: str) -> pd.Series:
    if name not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[name], errors="coerce")


def declared_protocol_mask(df: pd.DataFrame) -> pd.Series:
    """Rows produced with every setting in the declared measurement protocol.

    Merely having retry/load bookkeeping is not sufficient.  This predicate is
    deliberately exact so a quick exploratory run cannot enter a paper cohort
    just because it has a low CV or a newer timestamp.
    """
    numeric = {name: _numeric_column(df, name) for name in (
        "repetitions", "settle_s", "max_cv_pct", "n_gpu_layers", "n_depth",
        "n_prompt", "n_gen", "n_batch", "n_ubatch", "load_before",
        "load_after", "attempts", "kept_cv_pct",
    )}
    text = {
        name: (df[name].astype("string").str.strip().str.lower()
               if name in df.columns else
               pd.Series(pd.NA, index=df.index, dtype="string"))
        for name in ("flash_attn", "type_k", "type_v")
    }

    bookkeeping = pd.Series(True, index=df.index, dtype=bool)
    for name in _QUALITY_BOOKKEEPING_FIELDS:
        bookkeeping &= np.isfinite(numeric[name])
    bookkeeping &= numeric["attempts"].ge(1)
    bookkeeping &= numeric["attempts"].eq(numeric["attempts"].round())
    bookkeeping &= numeric["kept_cv_pct"].ge(0)

    prefill = numeric["n_prompt"].eq(512) & numeric["n_gen"].eq(0)
    decode = numeric["n_prompt"].eq(0) & numeric["n_gen"].eq(128)
    flash_on = text["flash_attn"].isin(
        {"1", "1.0", "true", "on", "yes"})

    return (bookkeeping
            & numeric["repetitions"].eq(DECLARED_REPETITIONS)
            & numeric["settle_s"].eq(DECLARED_SETTLE_S)
            & numeric["max_cv_pct"].eq(DECLARED_MAX_CV_PCT)
            & numeric["n_gpu_layers"].eq(PRIMARY_N_GPU_LAYERS)
            & numeric["n_depth"].isin(DECLARED_DEPTHS)
            & flash_on
            & text["type_k"].eq("f16")
            & text["type_v"].eq("f16")
            & numeric["n_batch"].eq(DECLARED_N_BATCH)
            & numeric["n_ubatch"].eq(DECLARED_N_UBATCH)
            & (prefill | decode)).fillna(False).astype(bool)


def legacy_protocol_mask(df: pd.DataFrame) -> pd.Series:
    """Rows predating quality bookkeeping, eligible only as a last fallback."""
    present = [name for name in _QUALITY_BOOKKEEPING_FIELDS
               if name in df.columns]
    if not present:
        return pd.Series(True, index=df.index, dtype=bool)
    # Any populated quality-control field means this was a protocol-aware run;
    # if its settings are wrong it must not masquerade as legacy data.
    return df[present].isna().all(axis=1)


def validate_host_coverage(df: pd.DataFrame, cal: dict,
                           require_exact: bool = True) -> None:
    """Fail closed unless measurement and calibration host IDs agree.

    ``require_exact=False`` supports callers intentionally operating on a
    measurement subset with a superset of calibrations.  Even then, every
    measured host must have exactly identified calibration data.
    """
    if "host" not in df.columns:
        raise ValueError("measurements have no host column")
    if df.empty:
        raise ValueError("measurements contain no rows; host coverage is unknown")

    measured = set()
    for value in df["host"].drop_duplicates().tolist():
        measured.add(_validate_host_id(value, "measurement rows"))

    calibrated = set()
    for key, value in cal.items():
        host = _validate_host_id(key, "calibration mapping key")
        if not isinstance(value, dict):
            raise ValueError(f"calibration for {host!r} is not a JSON object")
        embedded = _validate_host_id(value.get("host"),
                                     f"calibration payload for {host!r}")
        if embedded != host:
            raise ValueError(
                f"calibration host mismatch: mapping key {host!r}, "
                f"payload host {embedded!r}")
        calibrated.add(host)

    missing_cal = sorted(measured - calibrated)
    missing_measurements = sorted(calibrated - measured) if require_exact else []
    if missing_cal or missing_measurements:
        details = []
        if missing_cal:
            details.append("missing calibration for " + ", ".join(missing_cal))
        if missing_measurements:
            details.append("missing measurements for " +
                           ", ".join(missing_measurements))
        raise ValueError("measurement/calibration host ID mismatch: " +
                         "; ".join(details))


def load_measurements(results_dir: Path, *, deduplicate: bool = True) -> pd.DataFrame:
    """Load successful measurements, optionally retaining repeated cells.

    Analysis uses the default quality-aware deduplication.  The opt-out exists
    for diagnostics such as ``refine --keep-all`` that explicitly need every
    recorded observation; it does not change publication defaults.
    """
    files = sorted(results_dir.glob("measurements_*.csv"))
    if not files:
        raise SystemExit(f"No measurements_*.csv in {results_dir}. "
                         f"Run `python -m llmperf.sweep` first.")
    frames = []
    origins: dict[str, Path] = {}
    problems = []
    for f in files:
        try:
            filename_host = _filename_host_id(f, "measurements_", ".csv")
        except ValueError as e:
            problems.append(str(e))
            continue
        frame = pd.read_csv(f)
        frames.append(frame)
        if "host" not in frame.columns:
            problems.append(f"{f}: missing host column")
            continue
        if frame.empty:
            problems.append(f"{f}: no rows, so its host ID cannot be verified")
            continue

        row_hosts = set()
        for value in frame["host"].drop_duplicates().tolist():
            try:
                row_hosts.add(_validate_host_id(value, f"{f} host column"))
            except ValueError as e:
                problems.append(str(e))
        if row_hosts != {filename_host}:
            problems.append(
                f"measurement host mismatch in {f}: filename declares "
                f"{filename_host!r}, rows declare {sorted(row_hosts)!r}")
        for host in row_hosts:
            if host in origins and origins[host] != f:
                problems.append(
                    f"duplicate measurement host ID {host!r} in "
                    f"{origins[host]} and {f}")
            else:
                origins[host] = f

    if problems:
        raise ValueError("invalid measurement host provenance:\n- " +
                         "\n- ".join(problems))

    df = pd.concat(frames, ignore_index=True)
    df = df[df["error"].isna() | (df["error"].astype(str).str.len() == 0)]
    for c in ("avg_ts", "stddev_ts", "file_bytes", "n_params", "n_active_params",
              "n_layers", "n_kv_heads", "head_dim", "n_depth", "n_prompt",
              "n_gen", "n_gpu_layers", "n_expert", "n_batch", "n_ubatch",
              "settle_s", "repetitions", "load_before", "load_after",
              "attempts", "kept_cv_pct", "max_cv_pct"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["avg_ts"])

    # This column is recomputed from the raw protocol fields on every load.  A
    # stale or hand-edited quality_protocol column in a CSV is never trusted.
    df["quality_protocol"] = declared_protocol_mask(df)

    # A cell may be measured more than once. Prefer rows produced by the full
    # quality-gated protocol, then keep the best attempt inside that protocol.
    # Legacy rows predate load/CV bookkeeping and are not interchangeable with
    # the declared experiment merely because their within-run CV is smaller.
    #
    # Ranking is on protocol eligibility and measurement quality only -- never
    # on a prediction residual. `drop_duplicates` is intentional: pandas
    # GroupBy.first() chooses the first *non-null value in each column*, which
    # can splice timestamp/throughput from a legacy row together with load and
    # retry metadata from a later row that was never actually selected.
    if deduplicate and "timestamp" in df.columns:
        before = len(df)
        df = df.copy()
        df["_legacy_protocol"] = legacy_protocol_mask(df)
        kept_cv = (df["kept_cv_pct"] if "kept_cv_pct" in df.columns
                   else pd.Series(np.nan, index=df.index))
        df["_cv"] = pd.to_numeric(kept_cv, errors="coerce")
        # Fall back to the per-row stddev where the retry bookkeeping predates
        # the column, so old rows are ranked on the same basis.
        fallback = (pd.to_numeric(df["stddev_ts"], errors="coerce")
                    / pd.to_numeric(df["avg_ts"], errors="coerce") * 100)
        df["_cv"] = df["_cv"].fillna(fallback).fillna(np.inf)
        load_before = (df["load_before"] if "load_before" in df.columns
                       else pd.Series(np.nan, index=df.index))
        df["_load"] = pd.to_numeric(load_before, errors="coerce").fillna(np.inf)
        keys = ["host", "model_file", "n_gpu_layers", "n_depth",
                "n_prompt", "n_gen"]
        df = (df.sort_values(
                    ["quality_protocol", "_legacy_protocol", "_cv", "_load",
                     "timestamp"],
                    ascending=[False, False, True, True, False])
                .drop_duplicates(keys, keep="first")
                .drop(columns=["_legacy_protocol", "_cv", "_load"],
                      errors="ignore"))
        if len(df) < before:
            n_protocol = int(df["quality_protocol"].sum())
            print(f"deduplicated {before - len(df)} re-measured row(s); "
                  f"kept protocol-complete rows first, then lowest CV "
                  f"({n_protocol}/{len(df)} selected rows protocol-complete)")
    return df


def load_calibration(results_dir: Path) -> dict:
    out = {}
    origins: dict[str, Path] = {}
    problems = []
    for f in sorted(results_dir.glob("calibration_*.json")):
        try:
            filename_host = _filename_host_id(f, "calibration_", ".json")
        except ValueError as e:
            problems.append(str(e))
            continue
        c = json.loads(f.read_text(encoding="utf-8"))
        if not isinstance(c, dict):
            problems.append(f"{f}: calibration must be a JSON object")
            continue
        try:
            host = _validate_host_id(c.get("host"), f"{f} host field")
        except ValueError as e:
            problems.append(str(e))
            continue
        if host != filename_host:
            problems.append(
                f"calibration host mismatch in {f}: filename declares "
                f"{filename_host!r}, payload declares {host!r}")
        if host in origins:
            problems.append(
                f"duplicate calibration host ID {host!r} in "
                f"{origins[host]} and {f}")
        else:
            origins[host] = f
            out[host] = c
    if problems:
        raise ValueError("invalid calibration host provenance:\n- " +
                         "\n- ".join(problems))
    return out


def load_splits(models_dir: Path, results_dir: Path = RESULTS_DIR) -> dict:
    """Load live and frozen split provenance; reject any disagreement."""
    return load_model_splits(models_dir, results_dir)


def resolve_splits(df: pd.DataFrame, splits: dict[str, str]) -> pd.DataFrame:
    """Resolve each row's fixed split without silently training on unknowns.

    New measurement rows carry their split directly. Legacy rows have a blank
    column and are recovered from the tracked result-side manifest. Either is
    sufficient, but if both exist they must agree.
    """
    out = df.copy()
    if "split" in out.columns:
        recorded = out["split"].fillna("").astype(str).str.strip().str.lower()
    else:
        recorded = pd.Series("", index=out.index, dtype="object")
    mapped = out["model_file"].map(splits)
    valid_recorded = recorded.isin({"train", "test"})
    invalid_recorded = recorded.ne("") & ~valid_recorded
    if invalid_recorded.any():
        details = sorted({
            f"{mf}: {split!r}" for mf, split in
            zip(out.loc[invalid_recorded, "model_file"], recorded[invalid_recorded])
        })
        raise ValueError("invalid train/test split in measurement CSV: " +
                         "; ".join(details))

    conflict = valid_recorded & mapped.notna() & recorded.ne(mapped)
    if conflict.any():
        details = sorted({
            f"{mf}: csv={csv_split}, manifest={manifest_split}"
            for mf, csv_split, manifest_split in zip(
                out.loc[conflict, "model_file"], recorded[conflict], mapped[conflict])
        })
        raise ValueError("conflicting train/test split provenance: " +
                         "; ".join(details))

    resolved = recorded.where(valid_recorded, mapped)
    missing = sorted(out.loc[resolved.isna() | ~resolved.isin({"train", "test"}),
                             "model_file"].astype(str).unique())
    if missing:
        raise ValueError(
            "missing fixed train/test split provenance for: " + ", ".join(missing) +
            ". Restore results/model_manifest.json or provide split values in "
            "the measurement CSV; unknown models are never assumed to be train.")
    out["split"] = resolved
    return out


def select_primary_measurements(df: pd.DataFrame) -> pd.DataFrame:
    """Rows eligible for predictor fitting and headline error metrics."""
    if "n_gpu_layers" not in df.columns:
        raise ValueError("measurements have no n_gpu_layers column")
    ngl = pd.to_numeric(df["n_gpu_layers"], errors="coerce")
    eligible = ngl == PRIMARY_N_GPU_LAYERS
    # Once a dataset contains rows from the exact declared protocol, neither
    # legacy nor explicitly wrong-protocol rows may enter headline fits.  A
    # wholly old dataset remains analysable, but only genuinely legacy rows
    # (with no quality-control bookkeeping) receive that fallback.
    has_raw_protocol = any(name in df.columns
                           for name in _PROTOCOL_DECLARATION_FIELDS)
    if has_raw_protocol:
        # Do not trust a caller-supplied flag when the source fields are here:
        # recompute so hand-edited/stale derived columns cannot bypass checks.
        strict = declared_protocol_mask(df)
    elif "quality_protocol" in df.columns:
        # Compatibility for small in-memory callers that carry only the
        # already-derived flag rather than raw measurement columns.
        strict = df["quality_protocol"].fillna(False).astype(bool)
    else:
        strict = pd.Series(False, index=df.index, dtype=bool)
    if strict.any():
        eligible &= strict
    elif has_raw_protocol or "quality_protocol" in df.columns:
        eligible &= legacy_protocol_mask(df)
    return df[eligible].copy()


def bandwidth_for(cal: dict) -> tuple[float, str]:
    """Pick the calibration bandwidth term, honouring the platform caveat.

    On Apple Silicon the CPU triad cannot saturate the fabric the GPU reaches,
    so using it would systematically under-predict. Prefer the accelerator
    device-copy result on both MPS and CUDA; use the LLM reference as a
    diagnostic fallback when no such microbenchmark is available.
    """
    gpu = cal.get("torch_gpu") or {}
    if gpu.get("available") and gpu.get("copy_gb_s"):
        return float(gpu["copy_gb_s"]) * 1e9, "torch_gpu.copy"
    ref = cal.get("llm_ref") or {}
    if ref.get("available") and ref.get("eff_bw_gb_s"):
        return float(ref["eff_bw_gb_s"]) * 1e9, "llm_ref.effective"
    return float(cal["cpu_triad"]["gb_s"]) * 1e9, "cpu_triad"


def compute_for(cal: dict) -> tuple[float, str]:
    gpu = cal.get("torch_gpu") or {}
    if gpu.get("available") and gpu.get("fp16_tflops"):
        return float(gpu["fp16_tflops"]) * 1e12, "torch_gpu.fp16"
    return float(cal["cpu_matmul"]["gflops"]) * 1e9, "cpu_matmul.fp32"


# --------------------------------------------------------------------------
# feature construction
# --------------------------------------------------------------------------

def calibration_probe_models(cal: dict) -> dict:
    """{host: model_file} used as the llm_ref calibration probe on that host.

    That model's eta is 1.0 by construction, so scoring it would flatter the
    result. It has to be excluded from the error metrics.
    """
    out = {}
    for host, c in cal.items():
        ref = c.get("llm_ref") or {}
        if not ref.get("available"):
            continue
        # Every probe is excluded, not just the winning one: each contributed a
        # measurement to selecting the anchor, so scoring any of them would be
        # scoring the calibration against itself.
        names = ref.get("probe_models") or ([ref["model"]] if ref.get("model") else [])
        if names:
            out[host] = set(names)
    return out


def refresh_metadata(df: pd.DataFrame, models_dir: Path = MODELS_DIR,
                     metadata: dict[str, dict] | None = None,
                     prefer_live: bool = True) -> pd.DataFrame:
    """Refresh metadata from GGUFs, falling back to the audited snapshot.

    sweep.py denormalises architecture fields into every measurement row, which
    freezes whatever the parser believed at measurement time. When the parser is
    later corrected — as it was for MoE architectures that fuse gate and up into
    one expert tensor — the fix would otherwise never reach data already on disk,
    and re-measuring 300 GB of models to pick up a metadata change is absurd.

    Live GGUF headers take precedence by default. The tracked snapshot makes
    the same corrected metadata available on a clean clone; retaining whatever
    happened to be denormalised into an old CSV would resurrect fixed parser
    bugs. Set ``prefer_live=False`` for frozen-artifact generation whose inputs
    must not depend on which model files happen to be installed locally.
    """
    from .common import read_gguf_meta

    cols = ["file_bytes", "n_params", "n_active_params", "n_layers",
            "n_kv_heads", "head_dim", "n_expert", "n_expert_used", "quant",
            "arch"]
    metadata = metadata or {}
    fresh, changed, missing = {}, [], []
    from_snapshot = live = 0
    for name in df["model_file"].dropna().unique():
        name = str(name)
        snapshot = metadata.get(name)
        vals = ({c: snapshot[c] for c in cols} if snapshot is not None else None)
        source = "snapshot" if vals is not None else ""

        path = models_dir / name
        if prefer_live and path.is_file():
            try:
                m = read_gguf_meta(path)
            except Exception as e:
                if vals is None:
                    raise ValueError(
                        f"cannot read GGUF metadata for {name}, and no audited "
                        f"snapshot is available: {e}") from e
                print(f"WARNING: cannot read {path}; using audited metadata "
                      f"snapshot ({type(e).__name__}: {e})", file=sys.stderr)
            else:
                live_vals = {c: getattr(m, c) for c in cols}
                if vals is not None:
                    disagreements = [c for c in cols if vals[c] != live_vals[c]]
                    if disagreements:
                        print(f"WARNING: live GGUF metadata for {name} differs "
                              f"from the frozen snapshot in "
                              f"{', '.join(disagreements)}; using live GGUF",
                              file=sys.stderr)
                vals, source = live_vals, "live"

        if vals is None:
            missing.append(name)
            continue
        if source == "snapshot" and "file_bytes" in df.columns:
            recorded_sizes = pd.to_numeric(
                df.loc[df["model_file"] == name, "file_bytes"], errors="coerce"
            ).dropna().unique()
            if any(int(size) != int(vals["file_bytes"]) for size in recorded_sizes):
                raise ValueError(
                    f"audited metadata for {name} describes a different file "
                    f"size ({vals['file_bytes']}) than the measurement CSV "
                    f"({', '.join(str(int(x)) for x in recorded_sizes)}). Make "
                    f"the measured GGUF available or update the snapshot from "
                    f"that exact file.")
        fresh[name] = vals
        if source == "live":
            live += 1
        else:
            from_snapshot += 1

    if missing:
        raise ValueError(
            "no trustworthy model metadata for: " + ", ".join(sorted(missing)) +
            ". Restore results/model_metadata.json or make the GGUF files "
            "available; stale CSV metadata is not used as a silent fallback.")

    df = df.copy()
    for name, vals in fresh.items():
        rows = df["model_file"] == name
        for c, v in vals.items():
            old = df.loc[rows, c]
            if len(old) and pd.notna(old.iloc[0]):
                if c in {"quant", "arch"}:
                    differs = str(old.iloc[0]) != str(v)
                    detail = f"{old.iloc[0]} -> {v}"
                else:
                    differs = float(old.iloc[0]) != float(v)
                    detail = f"{old.iloc[0]:g} -> {v:g}"
                if differs:
                    changed.append(f"{name}.{c}: {detail}")
            df.loc[rows, c] = v
    if changed:
        print(f"refreshed metadata for {len(fresh)} models "
              f"({live} live GGUF, {from_snapshot} frozen); "
              f"{len(changed)} field(s) corrected:")
        for c in changed[:10]:
            print(f"    {c}")
    elif from_snapshot:
        print(f"metadata: {live} live GGUF, {from_snapshot} audited snapshot")
    return df


def add_features(df: pd.DataFrame, cal: dict, splits: dict,
                 metadata: dict[str, dict] | None = None,
                 models_dir: Path = MODELS_DIR,
                 prefer_live: bool = True) -> pd.DataFrame:
    # Library callers may deliberately pass a row subset, so extra calibration
    # records are allowed here.  Missing calibration for any represented host
    # is never allowed: otherwise those rows quietly receive NaN ceilings and
    # disappear from downstream scoring.
    validate_host_coverage(df, cal, require_exact=False)
    metadata = metadata or {}
    df = refresh_metadata(df, models_dir, metadata, prefer_live=prefer_live)
    df = resolve_splits(df, splits)

    probes = calibration_probe_models(cal)
    df["is_calibration_probe"] = [
        mf in probes.get(h, set()) for h, mf in zip(df["host"], df["model_file"])
    ]

    df["active_frac"] = np.where(df["n_params"] > 0,
                                 df["n_active_params"] / df["n_params"], 1.0)
    df["is_moe"] = df["n_expert"].fillna(0) > 0

    # Prefer the live ModelMeta calculation. Without a local GGUF, use the exact
    # per-depth values frozen when the committed results were generated. Never
    # reconstruct KV traffic from scalar columns: that silently revives the
    # uniform-global-attention bug for sliding-window and recurrent models.
    from .common import read_gguf_meta
    kv = []
    unreadable: set[str] = set()
    for name, depth in zip(df["model_file"], df["n_depth"].fillna(0)):
        name, depth = str(name), int(depth)
        p = models_dir / name
        if prefer_live and p.is_file():
            try:
                kv.append(read_gguf_meta(p).kv_bytes(depth))
                continue
            except Exception as e:
                if name not in unreadable:
                    print(f"WARNING: cannot compute KV bytes from {p}; using "
                          f"audited snapshot ({type(e).__name__}: {e})",
                          file=sys.stderr)
                    unreadable.add(name)
        entry = metadata.get(name) or {}
        by_depth = entry.get("kv_bytes_by_depth") or {}
        if str(depth) not in by_depth:
            raise ValueError(
                f"no audited KV metadata for {name} at depth {depth}. Make the "
                f"GGUF available or extend results/model_metadata.json; a "
                f"uniform-attention estimate would be unsafe.")
        kv.append(int(by_depth[str(depth)]))
    df["kv_bytes"] = kv
    df["bytes_active"] = df["file_bytes"] * df["active_frac"] + df["kv_bytes"]
    df["bytes_total"] = df["file_bytes"] + df["kv_bytes"]

    df["bw"] = df["host"].map(lambda h: bandwidth_for(cal[h])[0] if h in cal else np.nan)
    df["flops"] = df["host"].map(lambda h: compute_for(cal[h])[0] if h in cal else np.nan)

    df["phase"] = np.where(df["n_gen"] > 0, "decode", "prefill")
    return df


# --------------------------------------------------------------------------
# fitting
# --------------------------------------------------------------------------

def fit_eta(train: pd.DataFrame, bytes_col: str) -> pd.Series:
    """Efficiency factor per (host, quant), fitted on TRAIN rows only.

    eta = measured tok/s * bytes_per_token / BW. Fitting in this ratio space
    rather than by least squares on tok/s avoids letting the largest-throughput
    (smallest) models dominate the fit.
    """
    eta = train["avg_ts"] * train[bytes_col] / train["bw"]
    return eta.groupby([train["host"], train["quant"]]).median()


def validate_host_training_coverage(
        df: pd.DataFrame, expected_hosts=None, *, context: str = "fitting cohort") -> None:
    """Require at least one TRAIN row for every host in a host-adapted fit.

    This is deliberately separate from ``fit_eta`` and ``apply_eta``. Library
    callers use those helpers for transfer experiments and intentional row
    subsets, where the unseen-host/quant fallback is meaningful. Publication
    entry points call this guard after excluding calibration probes so a partial
    host cannot be presented as locally fitted when it actually used a global
    fallback.
    """
    missing_columns = sorted({"host", "split"} - set(df.columns))
    if missing_columns:
        raise ValueError(
            f"{context} lacks required column(s): {', '.join(missing_columns)}")
    represented = ({str(host) for host in expected_hosts}
                   if expected_hosts is not None else
                   set(df["host"].dropna().astype(str)))
    trained = set(df.loc[df["split"].eq("train"), "host"].dropna().astype(str))
    missing = sorted(represented - trained)
    if missing:
        raise ValueError(
            f"{context} has no TRAIN rows after calibration-probe/protocol "
            f"exclusion for host(s): {', '.join(missing)}")


def apply_eta(df: pd.DataFrame, eta: pd.Series, bytes_col: str) -> np.ndarray:
    """Predict decode tok/s. Unseen (host, quant) falls back to the host median."""
    host_med = eta.groupby(level=0).median()
    overall = float(eta.median())

    def lookup(h, q):
        if (h, q) in eta.index:
            return float(eta.loc[(h, q)])
        if h in host_med.index:
            return float(host_med.loc[h])
        return overall

    e = np.array([lookup(h, q) for h, q in zip(df["host"], df["quant"])])
    return e * df["bw"].to_numpy() / df[bytes_col].to_numpy()


def ape(pred: np.ndarray, actual: np.ndarray) -> np.ndarray:
    return np.abs(pred - actual) / np.abs(actual) * 100.0


def evaluate_prefill(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit the compute-bound regime at an empty cache and score every depth.

    The paper states a two-regime model; testing only decode leaves half of it
    unevaluated. Prefill processes the whole prompt in parallel, so it is
    compute- rather than bandwidth-bound:

        pp_tok/s = eta_p * FLOPS / (2 * n_active_params)

    Caveat that must travel with these numbers: without a device-side FLOPS
    measurement, `compute_for` falls back to a CPU fp32 matmul, so FLOPS is not
    this machine's GPU throughput and eta_p is a fitted constant absorbing the
    scale error rather than a literal efficiency. The *shape* of the model — that
    prefill rate goes as 1 / active parameters — is still testable, and that is
    what the error columns below measure.

    The equation has no context-depth term, so its primary fit and headline
    score are restricted to n_depth=0. Predictions are still emitted for every
    measured depth; their degradation is a scope diagnostic rather than being
    pooled into a deceptively benign headline average.

    Two variants are fitted: one eta_p per host (quantisation independent) and
    one per host x quantisation. Returned rows carry both predictions and APEs
    so every prefill observation is available to the paper supplement.
    """
    pf = df[(df["phase"] == "prefill") & df["flops"].notna()
            & (df["n_active_params"] > 0)].copy()
    if pf.empty:
        return pf, pd.DataFrame()

    pf["flops_per_token"] = 2.0 * pf["n_active_params"]
    train = pf[(pf["split"] == "train") &
               (pf["n_depth"] == PREFILL_FIT_DEPTH)]
    if train.empty:
        raise ValueError(
            f"no training prefill rows at n_depth={PREFILL_FIT_DEPTH}")

    ratio = train["avg_ts"] * train["flops_per_token"] / train["flops"]
    by_host = ratio.groupby(train["host"]).median()
    by_hq = ratio.groupby([train["host"], train["quant"]]).median()

    def pred_host(d):
        e = d["host"].map(by_host).fillna(float(by_host.median()))
        return e.to_numpy() * d["flops"].to_numpy() / d["flops_per_token"].to_numpy()

    def pred_hq(d):
        hm = by_hq.groupby(level=0).median()
        overall = float(by_hq.median())
        e = np.array([float(by_hq.loc[(h, q)]) if (h, q) in by_hq.index
                      else float(hm.loc[h]) if h in hm.index else overall
                      for h, q in zip(d["host"], d["quant"])])
        return e * d["flops"].to_numpy() / d["flops_per_token"].to_numpy()

    rows = []
    for short, name, fn in (
            ("P1", "P1 eta_p per host", pred_host),
            ("P2", "P2 eta_p per host x quant", pred_hq)):
        p = fn(pf)
        pf[f"pred_{short}"] = p
        pf[f"ape_{short}"] = ape(p, pf["avg_ts"].to_numpy())
        scored = pf[pf["n_depth"] == PREFILL_FIT_DEPTH]
        for split in ("train", "test"):
            m = (scored["split"] == split).to_numpy()
            if not m.any():
                continue
            e = scored.loc[scored["split"] == split, f"ape_{short}"].to_numpy()
            rows.append({"model": name, "split": split, "n": int(m.sum()),
                         "n_depth": PREFILL_FIT_DEPTH,
                         "MAPE_%": e.mean(), "median_APE_%": np.median(e),
                         "p90_APE_%": np.percentile(e, 90), "max_APE_%": e.max()})
    return pf, pd.DataFrame(rows)


def prefill_error_table_by_host(pf: pd.DataFrame) -> pd.DataFrame:
    """Summarize depth-zero prefill error separately for every host."""
    scored = pf[pf["n_depth"] == PREFILL_FIT_DEPTH]
    rows = []
    for host in sorted(scored["host"].dropna().unique()):
        host_rows = scored[scored["host"] == host]
        for short, name in (("P1", "P1 eta_p per host"),
                            ("P2", "P2 eta_p per host x quant")):
            for split in ("train", "test"):
                values = host_rows.loc[
                    host_rows["split"] == split, f"ape_{short}"
                ].to_numpy()
                if not len(values):
                    continue
                rows.append({
                    "host": host, "model": name, "split": split,
                    "n": len(values), "n_depth": PREFILL_FIT_DEPTH,
                    "MAPE_%": values.mean(),
                    "median_APE_%": np.median(values),
                    "p90_APE_%": np.percentile(values, 90),
                    "max_APE_%": values.max(),
                })
    return pd.DataFrame(rows)


def error_table(df: pd.DataFrame, preds: dict) -> pd.DataFrame:
    rows = []
    for name, p in preds.items():
        for split in ("train", "test"):
            m = df["split"] == split
            if not m.any():
                continue
            e = ape(p[m.to_numpy()], df.loc[m, "avg_ts"].to_numpy())
            rows.append({"model": name, "split": split, "n": int(m.sum()),
                         "MAPE_%": e.mean(), "median_APE_%": np.median(e),
                         "p90_APE_%": np.percentile(e, 90), "max_APE_%": e.max()})
    return pd.DataFrame(rows)


def error_table_by_host(df: pd.DataFrame, preds: dict) -> pd.DataFrame:
    """Return the same error summary without pooling unlike hardware.

    A pooled table is useful as an inventory summary, but it is not a valid
    substitute for reporting each host when cohort sizes differ.  Keep this
    export beside the pooled table so downstream papers cannot accidentally
    hide a weak host behind the larger one.
    """
    rows = []
    for host in sorted(df["host"].dropna().unique()):
        mask = (df["host"] == host).to_numpy()
        host_df = df.loc[mask].reset_index(drop=True)
        host_preds = {name: np.asarray(values)[mask]
                      for name, values in preds.items()}
        table = error_table(host_df, host_preds)
        table.insert(0, "host", host)
        rows.append(table)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------

def fig_pred_vs_actual(df, pred, out: Path):
    fig, ax = plt.subplots(figsize=(5.0, 4.6))
    for split, colour, marker in (("train", "#4C72B0", "o"), ("test", "#C44E52", "^")):
        m = (df["split"] == split).to_numpy()
        if not m.any():
            continue
        dense = m & ~df["is_moe"].to_numpy()
        moe = m & df["is_moe"].to_numpy()
        if dense.any():
            ax.scatter(df["avg_ts"][dense], pred[dense], s=26, c=colour,
                       marker=marker, alpha=.75, label=f"{split} dense")
        if moe.any():
            ax.scatter(df["avg_ts"][moe], pred[moe], s=44, facecolors="none",
                       edgecolors=colour, marker=marker, linewidths=1.4,
                       label=f"{split} MoE")
    lo = min(df["avg_ts"].min(), np.nanmin(pred)) * 0.7
    hi = max(df["avg_ts"].max(), np.nanmax(pred)) * 1.4
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="perfect")
    for f, ls in ((1.25, ":"), (0.8, ":")):
        ax.plot([lo, hi], [lo * f, hi * f], "k", ls=ls, lw=.7, alpha=.5)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("measured decode throughput (tok/s)")
    ax.set_ylabel("predicted decode throughput (tok/s)")
    ax.set_title("Calibrated roofline, out-of-sample\n(dotted lines: ±25%)")
    ax.legend(fontsize=7, loc="upper left")
    fig.savefig(out); plt.close(fig)


def fig_eta(df, bytes_col, out: Path):
    eta = (df["avg_ts"] * df[bytes_col] / df["bw"]).rename("eta")
    d = pd.concat([df[["host", "quant"]], eta], axis=1)
    order = sorted(d["quant"].unique())
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    hosts = sorted(d["host"].unique())
    width = 0.8 / max(1, len(hosts))
    for i, h in enumerate(hosts):
        vals = [d[(d.host == h) & (d["quant"] == q)]["eta"].median() for q in order]
        style = diagnostic_host_style(h)
        ax.bar(np.arange(len(order)) + i * width, vals, width, label=h,
               color=style["color"])
    ax.set_xticks(np.arange(len(order)) + width * (len(hosts) - 1) / 2)
    ax.set_xticklabels(order, rotation=30, ha="right")
    ax.set_ylabel(r"efficiency $\eta$  (fraction of roofline)")
    ax.set_title("Achieved fraction of the memory roofline, by quantisation")
    ax.legend(fontsize=7)
    fig.savefig(out); plt.close(fig)


def fig_context(df, pred, out: Path):
    d = df.copy(); d["pred"] = pred
    d = d[d["n_depth"] >= 0]
    if d["n_depth"].nunique() < 2:
        return False
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    for (host, name), g in d.groupby(["host", "model_name"]):
        g = g.sort_values("n_depth")
        if g["n_depth"].nunique() < 2:
            continue
        style = diagnostic_host_style(host)
        ax.plot(g["n_depth"], g["avg_ts"], marker=style["marker"],
                ls=style["linestyle"], ms=4, lw=1.4,
                color=style["color"], label=f"{host}: {name[:22]}")
        ax.plot(g["n_depth"], g["pred"], ":", lw=1, alpha=.7,
                color=style["color"])
    ax.set_xlabel("KV-cache depth (tokens)")
    ax.set_ylabel("decode throughput (tok/s)")
    ax.set_yscale("log")
    ax.set_title("Throughput decay with context\n(host-styled: measured, dotted: predicted)")
    ax.legend(fontsize=6, ncol=2)
    fig.savefig(out); plt.close(fig)
    return True


def fig_moe(df, preds, out: Path):
    """The paper's central comparison: does modelling sparsity actually pay?"""
    if not df["is_moe"].any():
        return False
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    groups = []
    for host in sorted(df["host"].unique()):
        for is_moe, label in ((False, "dense"), (True, "MoE")):
            mask = ((df["host"] == host) &
                    (df["is_moe"] == is_moe)).to_numpy()
            if mask.any():
                groups.append((f"{host}\n{label}", mask))
    names = list(preds)
    width = 0.8 / len(names)
    for i, name in enumerate(names):
        vals = [ape(preds[name][m], df["avg_ts"].to_numpy()[m]).mean()
                for _, m in groups if m.any()]
        ax.bar(np.arange(len(vals)) + i * width, vals, width, label=name)
    ax.set_xticks(np.arange(len(groups)) + width * (len(names) - 1) / 2)
    ax.set_xticklabels([g for g, m in groups if m.any()])
    ax.set_ylabel("MAPE (%)")
    ax.set_title("Sparsity-aware bytes-per-token is what MoE needs")
    ax.legend(fontsize=7)
    fig.savefig(out); plt.close(fig)
    return True


def fig_offload(df, out: Path):
    # One point per offload level. The depth sweep is a separate experiment;
    # connecting all depths would draw a meaningless zig-zag at every level.
    d = df[df["n_gpu_layers"].notna() & (df["n_depth"] == 0)]
    if d["n_gpu_layers"].nunique() < 3:
        return False
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    for (host, name), g in d.groupby(["host", "model_name"]):
        if g["n_gpu_layers"].nunique() < 3:
            continue
        g = g.sort_values("n_gpu_layers")
        frac = np.minimum(g["n_gpu_layers"] / g["n_layers"], 1.0)
        style = diagnostic_host_style(host)
        ax.plot(frac * 100, g["avg_ts"], marker=style["marker"],
                ls=style["linestyle"], ms=4, lw=1.4,
                color=style["color"], label=f"{host}: {name[:22]}")
    ax.set_xlabel("% of layers resident on GPU")
    ax.set_ylabel("decode throughput (tok/s)")
    ax.set_yscale("log")
    ax.set_title("The CPU-offload cliff")
    ax.legend(fontsize=6)
    fig.savefig(out); plt.close(fig)
    return True


# --------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=RESULTS_DIR)
    ap.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    ap.add_argument("--figures", type=Path, default=FIGURES_DIR)
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args(argv)

    if args.selfcheck:
        _selfcheck()
        return 0

    df = load_measurements(args.results)
    cal = load_calibration(args.results)
    if not cal:
        raise SystemExit(f"No calibration_*.json in {args.results}. "
                         f"Run `python -m llmperf.calibrate` first.")
    validate_host_coverage(df, cal)
    splits = load_splits(args.models_dir, args.results)
    metadata = load_model_metadata(args.results)
    # These CSVs are publication artifacts. Their values must be determined by
    # the audited snapshot, not by whichever same-named GGUFs happen to be
    # installed on the machine doing the regeneration.
    df = add_features(df, cal, splits, metadata, args.models_dir,
                      prefer_live=False)
    args.figures.mkdir(parents=True, exist_ok=True)
    expected_hosts = set(cal)

    for host, c in cal.items():
        bw, bw_src = bandwidth_for(c)
        fl, fl_src = compute_for(c)
        print(f"calibration[{host}]: BW={bw/1e9:.1f} GB/s ({bw_src}), "
              f"compute={fl/1e12:.2f} TFLOP/s ({fl_src})")

    all_dec = df[(df["phase"] == "decode") & df["bw"].notna()].reset_index(drop=True)
    dec = select_primary_measurements(all_dec).reset_index(drop=True)
    n_offload = len(all_dec) - len(dec)
    if n_offload:
        print(f"\nexcluding {n_offload} partial-offload decode row(s) from "
              f"predictor fitting/scoring; retained for the offload-cliff figure")
    if dec.empty:
        raise SystemExit(
            f"No decode rows with calibration at n_gpu_layers="
            f"{PRIMARY_N_GPU_LAYERS}. Nothing to fit.")

    # The model used as the llm_ref calibration probe has eta == 1 by
    # construction. Scoring it would inflate accuracy, so drop it.
    n_probe = int(dec["is_calibration_probe"].sum())
    if n_probe:
        probes = sorted(dec.loc[dec["is_calibration_probe"], "model_file"].unique())
        print(f"\nexcluding {n_probe} rows from the historical calibration-probe "
              f"cohort ({', '.join(probes)}); this exclusion is preserved "
              "independently of the selected bandwidth source")
        dec = dec[~dec["is_calibration_probe"]].reset_index(drop=True)
    if dec.empty:
        raise SystemExit(
            "Every decode row is the calibration probe. Download more models "
            "(`python -m llmperf.fetch --set all`) before analysing.")

    validate_host_training_coverage(
        dec, expected_hosts, context="decode host-adapted fit")
    train = dec[dec["split"] == "train"]

    eta_active = fit_eta(train, "bytes_active")
    eta_total = fit_eta(train, "bytes_total")

    preds = {
        "B0 unfitted (eta=1, total)": dec["bw"].to_numpy() / dec["bytes_total"].to_numpy(),
        "B1 fitted, total params": apply_eta(dec, eta_total, "bytes_total"),
        "B2 fitted, active params (ours)": apply_eta(dec, eta_active, "bytes_active"),
    }

    print("\n=== fitted eta (median per host x quant, TRAIN only) ===")
    print(eta_active.round(3).to_string())

    tbl = error_table(dec, preds)
    print("\n=== prediction error ===")
    print(tbl.round(1).to_string(index=False))
    host_tbl = error_table_by_host(dec, preds)
    if dec["host"].nunique() > 1:
        print("\n=== prediction error by host (do not replace with pooled values) ===")
        print(host_tbl.round(1).to_string(index=False))

    primary = select_primary_measurements(df)
    prefill_input = primary[~primary["is_calibration_probe"]].copy()
    prefill_fit_rows = prefill_input[
        prefill_input["phase"].eq("prefill")
        & prefill_input["flops"].notna()
        & prefill_input["n_active_params"].gt(0)
        & prefill_input["n_depth"].eq(PREFILL_FIT_DEPTH)
    ]
    validate_host_training_coverage(
        prefill_fit_rows, expected_hosts, context="prefill host-adapted fit")
    pf, pf_err = evaluate_prefill(prefill_input)
    if not pf_err.empty:
        print(f"\n=== prefill at empty cache (n_depth={PREFILL_FIT_DEPTH}) ===")
        print(pf_err.round(1).to_string(index=False))
        print("\n=== prefill scope diagnostic by existing-prefix depth ===")
        depth_err = (pf.groupby(["split", "n_depth"])[["ape_P1", "ape_P2"]]
                     .agg(["count", "mean", "median", "max"]))
        print(depth_err.round(1).to_string())
        spread = (pf.groupby("model_name")["avg_ts"].median()
                  / pf["avg_ts"].median()).agg(["min", "max"])
        print(f"{len(pf)} prefill measurements, "
              f"{pf['model_name'].nunique()} models; rate spans "
              f"{spread['min']:.2f}-{spread['max']:.2f}x the median")
        pf_err.round(2).to_csv(args.results / "error_table_prefill.csv", index=False)
        pf_host_err = prefill_error_table_by_host(pf)
        pf_host_err.round(2).to_csv(
            args.results / "error_table_prefill_by_host.csv", index=False)
        pf.to_csv(args.results / "predictions_prefill.csv", index=False)

    n_models = dec["model_name"].nunique()
    n_moe = dec[dec["is_moe"]]["model_name"].nunique()
    print(f"\n{len(dec)} decode measurements, {n_models} models "
          f"({n_moe} MoE), {dec['host'].nunique()} host(s)")

    ours = preds["B2 fitted, active params (ours)"]
    made = []
    fig_pred_vs_actual(dec, ours, args.figures / "fig1_pred_vs_actual.png"); made.append("fig1_pred_vs_actual.png")
    fig_eta(dec, "bytes_active", args.figures / "fig2_eta_by_quant.png"); made.append("fig2_eta_by_quant.png")
    if fig_context(dec, ours, args.figures / "fig3_context_decay.png"): made.append("fig3_context_decay.png")
    if fig_moe(dec, preds, args.figures / "fig4_moe_sparsity.png"): made.append("fig4_moe_sparsity.png")
    if fig_offload(all_dec, args.figures / "fig5_offload_cliff.png"): made.append("fig5_offload_cliff.png")

    tbl.round(2).to_csv(args.results / "error_table.csv", index=False)
    host_tbl.round(2).to_csv(args.results / "error_table_by_host.csv", index=False)
    exported = dec.copy()
    for short, name in (("B0", "B0 unfitted (eta=1, total)"),
                        ("B1", "B1 fitted, total params"),
                        ("B2", "B2 fitted, active params (ours)")):
        exported[f"pred_{short}"] = preds[name]
        exported[f"ape_{short}"] = ape(
            preds[name], dec["avg_ts"].to_numpy())
    # Backward-compatible aliases used by existing figures and supplements.
    exported["pred_ours"] = exported["pred_B2"]
    exported["ape_ours"] = exported["ape_B2"]
    exported.to_csv(args.results / "predictions.csv", index=False)

    print(f"\nfigures -> {args.figures}: {', '.join(made)}")
    print(f"tables  -> {args.results}/error_table.csv, error_table_by_host.csv, predictions.csv, "
          f"error_table_prefill.csv, error_table_prefill_by_host.csv, "
          f"predictions_prefill.csv")
    return 0


def _selfcheck() -> None:
    """Regression guards for split provenance and offload isolation."""
    frozen = load_splits(MODELS_DIR, RESULTS_DIR)
    assert frozen.get("SmolLM3-Q4_K_M.gguf") == "train"
    assert frozen.get("NVIDIA-Nemotron-3-Nano-4B-Q4_K_M.gguf") == "test"
    metadata = load_model_metadata(RESULTS_DIR)
    assert metadata["gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"][
        "n_active_params"] == 3_822_527_246
    assert metadata["NVIDIA-Nemotron-3-Nano-4B-Q4_K_M.gguf"][
        "kv_bytes_by_depth"]["0"] == 616_366_080

    legacy = pd.DataFrame({
        "model_file": ["train.gguf", "test.gguf"],
        "split": ["", np.nan],
        "n_gpu_layers": [99, 16],
    })
    resolved = resolve_splits(
        legacy, {"train.gguf": "train", "test.gguf": "test"})
    assert resolved["split"].tolist() == ["train", "test"]
    assert select_primary_measurements(resolved)["model_file"].tolist() == ["train.gguf"]

    embedded = pd.DataFrame({"model_file": ["future.gguf"], "split": ["test"]})
    assert resolve_splits(embedded, {})["split"].tolist() == ["test"]

    try:
        resolve_splits(pd.DataFrame({"model_file": ["unknown.gguf"]}), {})
    except ValueError as e:
        assert "never assumed to be train" in str(e)
    else:
        raise AssertionError("unknown split must fail closed")

    try:
        resolve_splits(embedded, {"future.gguf": "train"})
    except ValueError as e:
        assert "conflicting" in str(e)
    else:
        raise AssertionError("conflicting split provenance must fail")
    print("analyze.py selfcheck ok")


if __name__ == "__main__":
    raise SystemExit(main())
