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
    B0  uncalibrated roofline, eta = 1, total params
    B1  calibrated, but total params -- isolates how much the MoE sparsity term
        is actually worth
    B2  this paper: calibrated + active params + KV term

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

from .common import FIGURES_DIR, MODELS_DIR, RESULTS_DIR, KV_BYTES_PER_ELEM

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

    # A cell may be measured more than once. Keep the BEST attempt, not the
    # most recent: the sweep proceeds after its load-wait times out, so a later
    # row can have been taken on a busier machine than an earlier one, and
    # most-recent-wins would let a load-264 measurement overwrite a load-16 one.
    #
    # Ranking is on measurement quality only -- llama-bench's own within-run CV,
    # then load average, then recency. None of these looks at the residual, so
    # this is a quality rule rather than selection on the outcome.
    if "timestamp" in df.columns:
        before = len(df)
        df = df.copy()
        df["_cv"] = pd.to_numeric(df.get("kept_cv_pct"), errors="coerce")
        # Fall back to the per-row stddev where the retry bookkeeping predates
        # the column, so old rows are ranked on the same basis.
        fallback = (pd.to_numeric(df["stddev_ts"], errors="coerce")
                    / pd.to_numeric(df["avg_ts"], errors="coerce") * 100)
        df["_cv"] = df["_cv"].fillna(fallback).fillna(np.inf)
        df["_load"] = pd.to_numeric(df.get("load_before"), errors="coerce").fillna(np.inf)
        df = (df.sort_values(["_cv", "_load", "timestamp"],
                             ascending=[True, True, False])
                .groupby(["host", "model_file", "n_gpu_layers", "n_depth",
                          df["n_gen"].gt(0)], dropna=False, as_index=False)
                .first()
                .drop(columns=["_cv", "_load"], errors="ignore"))
        if len(df) < before:
            print(f"deduplicated {before - len(df)} re-measured row(s); "
                  f"kept the lowest-CV attempt per cell")
    return df


def load_calibration(results_dir: Path) -> dict:
    out = {}
    for f in results_dir.glob("calibration_*.json"):
        c = json.loads(f.read_text(encoding="utf-8"))
        out[c["host"]] = c
    return out


def load_splits(models_dir: Path) -> dict:
    p = models_dir / "manifest.json"
    if not p.exists():
        return {}
    return {k: v.get("split", "") for k, v in
            json.loads(p.read_text(encoding="utf-8")).items()}


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


def refresh_metadata(df: pd.DataFrame, models_dir: Path = MODELS_DIR) -> pd.DataFrame:
    """Re-derive model metadata from the GGUF files where they are present.

    sweep.py denormalises architecture fields into every measurement row, which
    freezes whatever the parser believed at measurement time. When the parser is
    later corrected — as it was for MoE architectures that fuse gate and up into
    one expert tensor — the fix would otherwise never reach data already on disk,
    and re-measuring 300 GB of models to pick up a metadata change is absurd.

    Rows whose GGUF is not on this machine keep their recorded values, so another
    host's CSV still analyses correctly.
    """
    from .common import read_gguf_meta

    cols = ["file_bytes", "n_params", "n_active_params", "n_layers",
            "n_kv_heads", "head_dim", "n_expert", "n_expert_used", "quant"]
    fresh, changed = {}, []
    for name in df["model_file"].dropna().unique():
        path = models_dir / str(name)
        if not path.is_file():
            continue
        try:
            m = read_gguf_meta(path)
        except Exception:
            continue
        fresh[name] = {c: getattr(m, c) for c in cols}

    df = df.copy()
    for name, vals in fresh.items():
        rows = df["model_file"] == name
        for c, v in vals.items():
            old = df.loc[rows, c]
            if c != "quant" and len(old) and pd.notna(old.iloc[0]) \
                    and float(old.iloc[0]) != float(v):
                changed.append(f"{name}.{c}: {old.iloc[0]:g} -> {v:g}")
            df.loc[rows, c] = v
    if changed:
        print(f"refreshed metadata from GGUF for {len(fresh)} models; "
              f"{len(changed)} field(s) corrected:")
        for c in changed[:10]:
            print(f"    {c}")
    return df


def add_features(df: pd.DataFrame, cal: dict, splits: dict) -> pd.DataFrame:
    df = refresh_metadata(df)
    df["split"] = df["model_file"].map(splits).fillna("train")

    probes = calibration_probe_models(cal)
    df["is_calibration_probe"] = [
        mf in probes.get(h, set()) for h, mf in zip(df["host"], df["model_file"])
    ]

    df["active_frac"] = np.where(df["n_params"] > 0,
                                 df["n_active_params"] / df["n_params"], 1.0)
    df["is_moe"] = df["n_expert"].fillna(0) > 0

    # Delegate to ModelMeta.kv_bytes rather than recomputing inline. The inline
    # version silently kept the uniform-global-attention assumption after the
    # per-layer model was written, so sliding-window and recurrent layers were
    # still being charged a full growing cache here even though common.py knew
    # better. One formula, one place.
    from .common import read_gguf_meta
    kv = []
    for name, depth in zip(df["model_file"], df["n_depth"].fillna(0)):
        p = MODELS_DIR / str(name)
        if p.is_file():
            try:
                kv.append(read_gguf_meta(p).kv_bytes(int(depth)))
                continue
            except Exception:
                pass
        # Model not on this machine: fall back to the uniform estimate from the
        # recorded columns, which is all another host's CSV carries.
        row = df[df["model_file"] == name].iloc[0]
        kv.append(2 * row["n_layers"] * row["n_kv_heads"] * row["head_dim"]
                  * depth * KV_BYTES_PER_ELEM)
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
    """Fit and score the compute-bound regime.

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

    Two variants are fitted so the compute-bound assumption is itself checked:
    one eta_p per host (what the model predicts, if prefill is quantisation
    independent) and one per host x quantisation (what is needed if it is not).
    """
    pf = df[(df["phase"] == "prefill") & df["flops"].notna()
            & (df["n_active_params"] > 0)].copy()
    if pf.empty:
        return pf, pd.DataFrame()

    pf["flops_per_token"] = 2.0 * pf["n_active_params"]
    train = pf[pf["split"] == "train"]
    if train.empty:
        train = pf

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
    for name, fn in (("P1 eta_p per host", pred_host),
                     ("P2 eta_p per host x quant", pred_hq)):
        p = fn(pf)
        for split in ("train", "test"):
            m = (pf["split"] == split).to_numpy()
            if not m.any():
                continue
            e = ape(p[m], pf["avg_ts"].to_numpy()[m])
            rows.append({"model": name, "split": split, "n": int(m.sum()),
                         "MAPE_%": e.mean(), "median_APE_%": np.median(e),
                         "p90_APE_%": np.percentile(e, 90), "max_APE_%": e.max()})
    return pf, pd.DataFrame(rows)


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
    d = df[df["n_gpu_layers"].notna()]
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
    args = ap.parse_args(argv)

    df = load_measurements(args.results)
    cal = load_calibration(args.results)
    if not cal:
        raise SystemExit(f"No calibration_*.json in {args.results}. "
                         f"Run `python -m llmperf.calibrate` first.")
    splits = load_splits(args.models_dir)
    df = add_features(df, cal, splits)
    args.figures.mkdir(parents=True, exist_ok=True)

    for host, c in cal.items():
        bw, bw_src = bandwidth_for(c)
        fl, fl_src = compute_for(c)
        print(f"calibration[{host}]: BW={bw/1e9:.1f} GB/s ({bw_src}), "
              f"compute={fl/1e12:.2f} TFLOP/s ({fl_src})")

    dec = df[(df["phase"] == "decode") & df["bw"].notna()].reset_index(drop=True)
    if dec.empty:
        raise SystemExit("No decode rows with calibration. Nothing to fit.")

    # The model used as the llm_ref calibration probe has eta == 1 by
    # construction. Scoring it would inflate accuracy, so drop it.
    n_probe = int(dec["is_calibration_probe"].sum())
    if n_probe:
        probes = sorted(dec.loc[dec["is_calibration_probe"], "model_file"].unique())
        print(f"\nexcluding {n_probe} rows from the calibration probe "
              f"({', '.join(probes)}): eta is 1.0 there by construction")
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
        "B0 uncalibrated (eta=1, total)": dec["bw"].to_numpy() / dec["bytes_total"].to_numpy(),
        "B1 calibrated, total params": apply_eta(dec, eta_total, "bytes_total"),
        "B2 calibrated, active params (ours)": apply_eta(dec, eta_active, "bytes_active"),
    }

    print("\n=== fitted eta (median per host x quant, TRAIN only) ===")
    print(eta_active.round(3).to_string())

    tbl = error_table(dec, preds)
    print("\n=== prediction error ===")
    print(tbl.round(1).to_string(index=False))

    pf, pf_err = evaluate_prefill(df[~df["is_calibration_probe"]])
    if not pf_err.empty:
        print("\n=== prefill (compute-bound regime) ===")
        print(pf_err.round(1).to_string(index=False))
        spread = (pf.groupby("model_name")["avg_ts"].median()
                  / pf["avg_ts"].median()).agg(["min", "max"])
        print(f"{len(pf)} prefill measurements, "
              f"{pf['model_name'].nunique()} models; rate spans "
              f"{spread['min']:.2f}-{spread['max']:.2f}x the median")
        pf_err.round(2).to_csv(RESULTS_DIR / "error_table_prefill.csv", index=False)

    n_models = dec["model_name"].nunique()
    n_moe = dec[dec["is_moe"]]["model_name"].nunique()
    print(f"\n{len(dec)} decode measurements, {n_models} models "
          f"({n_moe} MoE), {dec['host'].nunique()} host(s)")

    ours = preds["B2 calibrated, active params (ours)"]
    made = []
    fig_pred_vs_actual(dec, ours, args.figures / "fig1_pred_vs_actual.png"); made.append("fig1_pred_vs_actual.png")
    fig_eta(dec, "bytes_active", args.figures / "fig2_eta_by_quant.png"); made.append("fig2_eta_by_quant.png")
    if fig_context(dec, ours, args.figures / "fig3_context_decay.png"): made.append("fig3_context_decay.png")
    if fig_moe(dec, preds, args.figures / "fig4_moe_sparsity.png"): made.append("fig4_moe_sparsity.png")
    if fig_offload(dec, args.figures / "fig5_offload_cliff.png"): made.append("fig5_offload_cliff.png")

    tbl.round(2).to_csv(args.results / "error_table.csv", index=False)
    dec.assign(pred_ours=ours,
               ape_ours=ape(ours, dec["avg_ts"].to_numpy())
               ).to_csv(args.results / "predictions.csv", index=False)

    print(f"\nfigures -> {args.figures}: {', '.join(made)}")
    print(f"tables  -> {args.results}/error_table.csv, predictions.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
