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
import json
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

plt.rcParams.update({
    "figure.dpi": 140, "savefig.dpi": 200, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False,
    "axes.spines.right": False, "figure.autolayout": True,
})


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def load_measurements(results_dir: Path) -> pd.DataFrame:
    files = sorted(results_dir.glob("measurements_*.csv"))
    if not files:
        raise SystemExit(f"No measurements_*.csv in {results_dir}. "
                         f"Run `python -m llmperf.sweep` first.")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df[df["error"].isna() | (df["error"].astype(str).str.len() == 0)]
    for c in ("avg_ts", "stddev_ts", "file_bytes", "n_params", "n_active_params",
              "n_layers", "n_kv_heads", "head_dim", "n_depth", "n_prompt",
              "n_gen", "n_gpu_layers", "n_expert"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["avg_ts"])

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
    if "timestamp" in df.columns:
        before = len(df)
        df = df.copy()
        protocol_fields = [
            "load_before", "load_after", "attempts", "kept_cv_pct",
            "max_cv_pct",
        ]
        if all(c in df.columns for c in protocol_fields):
            df["quality_protocol"] = df[protocol_fields].notna().all(axis=1)
        else:
            df["quality_protocol"] = False
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
                    ["quality_protocol", "_cv", "_load", "timestamp"],
                    ascending=[False, True, True, False])
                .drop_duplicates(keys, keep="first")
                .drop(columns=["_cv", "_load"], errors="ignore"))
        if len(df) < before:
            n_protocol = int(df["quality_protocol"].sum())
            print(f"deduplicated {before - len(df)} re-measured row(s); "
                  f"kept protocol-complete rows first, then lowest CV "
                  f"({n_protocol}/{len(df)} selected rows protocol-complete)")
    return df


def load_calibration(results_dir: Path) -> dict:
    out = {}
    for f in results_dir.glob("calibration_*.json"):
        c = json.loads(f.read_text(encoding="utf-8"))
        out[c["host"]] = c
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
    # Once a dataset contains rows from the declared load/CV-gated protocol,
    # legacy rows are excluded from headline fits rather than mixed into the
    # cohort. A fully legacy dataset remains analysable, but must not be
    # described as quality-gated.
    if "quality_protocol" in df.columns and df["quality_protocol"].any():
        eligible &= df["quality_protocol"].fillna(False).astype(bool)
    return df[eligible].copy()


def bandwidth_for(cal: dict) -> tuple[float, str]:
    """Pick the calibration bandwidth term, honouring the platform caveat.

    On Apple Silicon the CPU triad cannot saturate the fabric the GPU reaches,
    so using it would systematically under-predict. The LLM reference probe
    goes through the same code path being predicted and is the honest choice
    there. On a discrete GPU the device copy bandwidth is the right term.
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
        ax.bar(np.arange(len(order)) + i * width, vals, width, label=h)
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
    for name, g in d.groupby("model_name"):
        g = g.sort_values("n_depth")
        if len(g) < 2:
            continue
        line, = ax.plot(g["n_depth"], g["avg_ts"], "o-", ms=4, lw=1.4,
                        label=name[:26])
        ax.plot(g["n_depth"], g["pred"], "--", lw=1, alpha=.7,
                color=line.get_color())
    ax.set_xlabel("KV-cache depth (tokens)")
    ax.set_ylabel("decode throughput (tok/s)")
    ax.set_yscale("log")
    ax.set_title("Throughput decay with context\n(solid: measured, dashed: predicted)")
    ax.legend(fontsize=6, ncol=2)
    fig.savefig(out); plt.close(fig)
    return True


def fig_moe(df, preds, out: Path):
    """The paper's central comparison: does modelling sparsity actually pay?"""
    if not df["is_moe"].any():
        return False
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    groups = [("dense", ~df["is_moe"].to_numpy()), ("MoE", df["is_moe"].to_numpy())]
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
    for name, g in d.groupby("model_name"):
        if g["n_gpu_layers"].nunique() < 3:
            continue
        g = g.sort_values("n_gpu_layers")
        frac = np.minimum(g["n_gpu_layers"] / g["n_layers"], 1.0)
        ax.plot(frac * 100, g["avg_ts"], "o-", ms=4, lw=1.4, label=name[:26])
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
    splits = load_splits(args.models_dir, args.results)
    metadata = load_model_metadata(args.results)
    df = add_features(df, cal, splits, metadata, args.models_dir)
    args.figures.mkdir(parents=True, exist_ok=True)

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

    train = dec[dec["split"] == "train"]
    if train.empty:
        print("WARNING: no TRAIN rows; fitting on everything. Reported test "
              "error would not be out-of-sample.", file=sys.stderr)
        train = dec

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
    pf, pf_err = evaluate_prefill(primary[~primary["is_calibration_probe"]])
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
