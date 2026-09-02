"""Publication figures. Separate from analyze.py's diagnostics on purpose.

analyze.py draws fast throwaway plots every campaign iteration. These are the
ones that go in the manuscript: column-width, vector output, print-safe.

Three constraints that do not apply to on-screen charts and drive most of the
choices here:

* **The paper gets printed in black and white.** Colour may not survive, so
  every series carries a second encoding — marker shape and line style — and no
  claim is ever readable by hue alone.
* **Figures are read at column width**, ~3.4 in single / ~7 in double. Type set
  at 8 pt on a 5-inch canvas becomes 5 pt on the page. Sizes here are the real
  printed sizes.
* **Vector for the typesetter, raster for previewing.** Each figure is written
  as both PDF and PNG.

Palette is the validated categorical order (worst adjacent CVD dE 9.1, above the
8.0 target). Hues are assigned in fixed order and never cycled.

    python -m llmperf.figures
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from .analyze import (add_features, apply_eta, ape, fit_eta, load_calibration,
                      load_measurements, load_splits)
from .common import FIGURES_DIR, MODELS_DIR, RESULTS_DIR

# Validated categorical order. Do not reorder or cycle: slot N is always the
# same hue, so a series keeps its colour when another is added or dropped.
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, INK2, MUTED = "#1a1a19", "#4a4a46", "#8a8a82"
SURFACE = "#fcfcfb"

# Single- and double-column widths in inches.
COL, WIDE = 3.4, 7.0

plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    # Recessive chrome: the data should be the darkest thing on the page.
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
    "axes.grid": True, "grid.color": "#e2e2dc", "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": INK2, "ytick.color": INK2,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.labelcolor": INK, "text.color": INK,
    "legend.frameon": False, "legend.handlelength": 1.6,
    "lines.linewidth": 1.4, "lines.markersize": 4.5,
})


def save(fig, out: Path, name: str) -> str:
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}")
    plt.close(fig)
    return name


def _load():
    df = load_measurements(RESULTS_DIR)
    cal = load_calibration(RESULTS_DIR)
    if not cal:
        raise SystemExit("no calibration_*.json; run `python -m llmperf.calibrate`")
    df = add_features(df, cal, load_splits(MODELS_DIR))
    dec = df[(df["phase"] == "decode") & df["bw"].notna()].reset_index(drop=True)
    dec = dec[~dec["is_calibration_probe"]].reset_index(drop=True)
    return df, dec


# --------------------------------------------------------------------------

def fig_accuracy(dec, pred, out: Path) -> str:
    """Predicted vs measured. Job: does the model work at all?

    Log-log because throughput spans an order of magnitude; a linear axis would
    compress every small model into the corner. Marker shape carries dense-vs-MoE
    and fill carries train-vs-test, so the figure survives greyscale printing.
    """
    fig, ax = plt.subplots(figsize=(COL, COL * 0.95))

    groups = [
        ("train, dense", ~dec["is_moe"] & (dec["split"] == "train"), "o", C[0], True),
        ("train, MoE",   dec["is_moe"] & (dec["split"] == "train"),  "s", C[0], False),
        ("test, dense",  ~dec["is_moe"] & (dec["split"] == "test"),  "^", C[1], True),
        ("test, MoE",    dec["is_moe"] & (dec["split"] == "test"),   "D", C[1], False),
    ]
    for label, mask, marker, colour, filled in groups:
        m = mask.to_numpy()
        if not m.any():
            continue
        ax.scatter(dec["avg_ts"][m], pred[m], marker=marker, s=22,
                   facecolors=colour if filled else "none",
                   edgecolors=colour, linewidths=1.0, alpha=0.9,
                   label=f"{label} (n={m.sum()})", zorder=3)

    lo = float(min(dec["avg_ts"].min(), np.nanmin(pred))) * 0.65
    hi = float(max(dec["avg_ts"].max(), np.nanmax(pred))) * 1.5
    ax.plot([lo, hi], [lo, hi], "-", color=INK2, lw=0.9, zorder=2, label="exact")
    for f in (1.25, 0.8):
        ax.plot([lo, hi], [lo * f, hi * f], "--", color=MUTED, lw=0.6, zorder=1)
    ax.text(hi * 0.97, hi * 0.8 * 0.97, "±25%", color=MUTED, fontsize=6.5,
            ha="right", va="top")

    ax.set(xscale="log", yscale="log", xlim=(lo, hi), ylim=(lo, hi))
    ax.set_xlabel("measured decode throughput (tok/s)")
    ax.set_ylabel("predicted (tok/s)")
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=6.2, borderpad=0.2, labelspacing=0.3)
    return save(fig, out, "fig1_accuracy")


def fig_ablation(dec, preds, out: Path) -> str:
    """The paper's central claim. Job: compare magnitudes across few categories.

    Grouped bars, not a table, because the whole point is that one bar is
    dramatically shorter out of sample. Values are printed on the bars: the
    palette validator flags these hues for relief on a light surface, and direct
    labels are how that obligation is met.
    """
    cells, labels = [], []
    for split in ("train", "test"):
        for is_moe, tag in ((False, "dense"), (True, "MoE")):
            m = ((dec["split"] == split) & (dec["is_moe"] == is_moe)).to_numpy()
            if m.any():
                cells.append(m)
                labels.append(f"{split}\n{tag}\n(n={m.sum()})")
    if not cells:
        return ""

    names = list(preds)
    fig, ax = plt.subplots(figsize=(WIDE * 0.62, COL * 0.82))
    width = 0.8 / len(names)
    x = np.arange(len(cells))
    hatches = ["", "//", ".."]

    for i, name in enumerate(names):
        vals = [ape(preds[name][m], dec["avg_ts"].to_numpy()[m]).mean() for m in cells]
        bars = ax.bar(x + i * width, vals, width * 0.92, label=name,
                      color=C[i], edgecolor=SURFACE, linewidth=1.2,
                      hatch=hatches[i % len(hatches)], zorder=3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}",
                    ha="center", va="bottom", fontsize=6.5, color=INK)

    ax.set_xticks(x + width * (len(names) - 1) / 2)
    ax.set_xticklabels(labels)
    ax.set_ylabel("mean absolute percentage error")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.set_axisbelow(True)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", ncol=1)
    return save(fig, out, "fig2_ablation")


def fig_noise_floor(out: Path) -> str:
    """Repeatability. Job: show a distribution and a threshold, not a mean.

    Every point is drawn. With six runs a box plot would hide the data behind a
    summary of six numbers, which is exactly the sort of thing this figure
    exists to argue against.
    """
    path = RESULTS_DIR / "repeatability_lun-mac.csv"
    cands = sorted(RESULTS_DIR.glob("repeatability_*.csv"))
    if not path.exists():
        if not cands:
            return ""
        path = cands[0]
    r = pd.read_csv(path)
    if len(r) < 3:
        return ""

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(WIDE * 0.78, COL * 0.8),
        gridspec_kw={"width_ratios": [1.5, 1]})

    dec = r["decode_ts"].to_numpy()
    runs = r["run_index"].to_numpy()
    mean = dec.mean()
    cv_ctl = dec.std(ddof=1) / mean * 100

    ax1.axhspan(mean * (1 - cv_ctl / 100), mean * (1 + cv_ctl / 100),
                color=C[0], alpha=0.10, zorder=1)
    ax1.axhline(mean, color=C[0], lw=1.0, zorder=2)
    ax1.plot(runs, dec, "o-", color=C[0], mfc=SURFACE, mew=1.2, lw=1.0, zorder=4)

    # The residual is not stationary: throughput climbs monotonically over the
    # later runs. Drawing the trend keeps the figure honest — a CV alone would
    # present a systematic drift as if it were random scatter.
    slope, intercept = np.polyfit(runs, dec, 1)
    if abs(slope) * len(runs) > 0.3 * dec.std(ddof=1):
        ax1.plot(runs, intercept + slope * runs, ls="--", lw=0.9, color=C[1],
                 zorder=3)
        ax1.text(runs[-1], intercept + slope * runs[-1],
                 f"  drift {slope / mean * 100:+.1f}%/run", va="center",
                 ha="left", fontsize=6.2, color=C[1])
    ax1.text(runs[0], mean, f"mean ±{cv_ctl:.1f}%  ", va="bottom", ha="left",
             fontsize=6.2, color=INK2)
    ax1.set_xlabel("independent run")
    ax1.set_ylabel("decode throughput (tok/s)")
    ax1.set_xticks(runs)
    ax1.margins(x=0.16)
    ax1.set_title("controlled: idle, 45 s settling", loc="left", color=INK)

    # The uncontrolled figure is the observed spread when a download was
    # competing for I/O; it is a range, not a CV.
    bars = [("controlled\n(idle, 45 s)", cv_ctl, C[2]),
            ("uncontrolled\n(background I/O)", 26.0, C[1])]
    xs = np.arange(len(bars))
    for i, (lab, v, col) in enumerate(bars):
        ax2.bar(i, v, 0.55, color=col, edgecolor=SURFACE, linewidth=1.2,
                hatch="" if i == 0 else "//", zorder=3)
        ax2.text(i, v + 0.7, f"{v:.1f}%", ha="center", va="bottom",
                 fontsize=7, color=INK)
    ax2.axhline(3.2, color=INK2, ls=":", lw=0.9, zorder=4)
    # Anchored between the two bars so it clears both the short bar and its
    # value label.
    ax2.annotate("3.2% detection threshold (2σ)", xy=(0.5, 3.2),
                 xytext=(0.5, 7.5), ha="center", fontsize=6.2, color=INK2,
                 arrowprops=dict(arrowstyle="-", lw=0.6, color=INK2))
    ax2.set_ylim(0, max(v for _, v, _ in bars) * 1.22)
    ax2.set_xticks(xs)
    ax2.set_xticklabels([b[0] for b in bars])
    ax2.set_ylabel("run-to-run variation")
    ax2.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax2.grid(axis="x", visible=False)
    ax2.set_axisbelow(True)
    ax2.set_title("noise floor vs effect size", loc="left", color=INK)
    return save(fig, out, "fig3_noise_floor")


def fig_eta(dec, out: Path) -> str:
    """Efficiency by quantisation. Job: magnitude across categories, with a
    hard reference line at eta = 1.

    Points, not bars, because each model is an observation and the spread within
    a format is the finding. eta > 1 is unphysical and the line makes that
    legible without a caption.
    """
    d = dec.copy()
    d["eta"] = d["avg_ts"] * d["bytes_active"] / d["bw"]
    d = d[d["n_depth"] == 0]
    if d.empty or d["quant"].nunique() < 2:
        return ""

    order = d.groupby("quant")["eta"].median().sort_values().index.tolist()
    fig, ax = plt.subplots(figsize=(WIDE * 0.66, COL * 0.78))

    ax.axhspan(1.0, max(1.05, d["eta"].max() * 1.08), color=C[1], alpha=0.07, zorder=1)
    ax.axhline(1.0, color=C[1], lw=1.0, ls="--", zorder=2)
    ax.text(len(order) - 0.5, 1.02, "unphysical: above the roofline",
            ha="right", va="bottom", fontsize=6.2, color=C[1])

    rng = np.random.default_rng(0)
    for i, q in enumerate(order):
        g = d[d["quant"] == q]
        jitter = rng.uniform(-0.12, 0.12, len(g))
        moe = g["is_moe"].to_numpy()
        for mask, marker in ((~moe, "o"), (moe, "s")):
            if mask.any():
                ax.scatter(i + jitter[mask], g["eta"].to_numpy()[mask],
                           marker=marker, s=24, facecolors="none",
                           edgecolors=C[0], linewidths=1.1, zorder=4)
        ax.plot([i - 0.26, i + 0.26], [g["eta"].median()] * 2,
                color=INK, lw=1.4, zorder=5)

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=30, ha="right")
    ax.set_ylabel(r"$\eta$  (fraction of memory roofline)")
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", mfc="none", mec=C[0], label="dense"),
        Line2D([], [], marker="s", ls="", mfc="none", mec=C[0], label="MoE"),
        Line2D([], [], color=INK, lw=1.4, label="median"),
    ], loc="upper left", ncol=3)
    return save(fig, out, "fig4_eta_by_quant")


def fig_context(dec, out: Path) -> str:
    """Throughput vs KV depth. Job: change along an ordered axis.

    Normalised to each model's own depth-0 value so a dozen models spanning an
    order of magnitude in absolute speed can share one axis — the question is
    the shape of the decay, not who is fastest.
    """
    d = dec[dec["n_depth"] >= 0]
    if d["n_depth"].nunique() < 2 or d["model_name"].nunique() < 2:
        return ""

    fig, ax = plt.subplots(figsize=(COL, COL * 0.8))
    for name, g in d.groupby("model_name"):
        g = g.sort_values("n_depth")
        base = g[g["n_depth"] == 0]["avg_ts"]
        if g["n_depth"].nunique() < 2 or base.empty:
            continue
        moe = bool(g["is_moe"].iloc[0])
        ax.plot(g["n_depth"], g["avg_ts"] / float(base.iloc[0]),
                marker="s" if moe else "o", ms=3.5,
                ls="--" if moe else "-", lw=1.0,
                color=C[1] if moe else C[0], alpha=0.75, zorder=3)

    ax.set_xlabel("KV-cache depth (tokens)")
    ax.set_ylabel("throughput relative to empty cache")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.legend(handles=[
        Line2D([], [], color=C[0], marker="o", ms=3.5, label="dense"),
        Line2D([], [], color=C[1], marker="s", ms=3.5, ls="--", label="MoE"),
    ], loc="lower left")
    return save(fig, out, "fig5_context_decay")


def fig_quant_ladder(dec, out: Path) -> str:
    """One model, every quantisation. The paper's cleanest single-factor result.

    Distinct from fig4, which shows eta across the whole model set and therefore
    mixes model differences into the format effect. Here the model is held fixed,
    so anything left is the format's own dequantisation cost.

    Connected points rather than bars: the x axis is ordered by bit width and the
    monotonicity IS the finding. Bars would imply independent categories.
    """
    d = dec[dec["n_depth"] == 0].copy()
    if d.empty:
        return ""
    # The ladder model is whichever has the most quantisation levels measured.
    fam = d["model_name"].str.replace(r"[-_](UD[-_])?(I?Q\d.*|MXFP4|BF16|F16)$", "",
                                      regex=True, case=False)
    d["family"] = fam
    counts = d.groupby("family")["quant"].nunique().sort_values(ascending=False)
    if counts.empty or counts.iloc[0] < 4:
        return ""
    target = counts.index[0]
    g = d[d["family"] == target].copy()

    bits = {"IQ2_XXS": 2.1, "Q2_K": 2.6, "IQ3_XXS": 3.1, "Q3_K": 3.4, "Q4_0": 4.0,
            "Q4_K": 4.5, "Q4_K_M": 4.5, "Q5_K": 5.5, "Q6_K": 6.6, "Q8_0": 8.5,
            "MXFP4": 4.3, "F16": 16.0, "BF16": 16.0}
    g["bits"] = g["quant"].map(bits)
    g = g.dropna(subset=["bits"]).sort_values("bits")
    if len(g) < 4:
        return ""
    g["eta"] = g["avg_ts"] * g["bytes_active"] / g["bw"]
    g["is_iquant"] = g["quant"].str.upper().str.startswith("IQ")

    fig, ax = plt.subplots(figsize=(COL, COL * 0.82))
    ax.plot(g["bits"], g["eta"], "-", color=C[0], lw=1.2, zorder=2, alpha=.8)
    for iq, marker, colour, lab in ((False, "o", C[0], "K-quant"),
                                    (True, "D", C[1], "I-quant")):
        m = (g["is_iquant"] == iq).to_numpy()
        if m.any():
            ax.scatter(g["bits"][m], g["eta"][m], marker=marker, s=34, zorder=4,
                       facecolors=colour, edgecolors=SURFACE, linewidths=0.8,
                       label=lab)
    for _, r in g.iterrows():
        ax.annotate(r["quant"], (r["bits"], r["eta"]), textcoords="offset points",
                    xytext=(0, -11), ha="center", fontsize=5.6, color=INK2)

    ax.set_xlabel("effective bits per weight")
    ax.set_ylabel(r"$\eta$  (fraction of bandwidth ceiling)")
    ax.set_title(f"{target}: efficiency rises with bit width", loc="left",
                 color=INK, fontsize=8)
    ax.legend(loc="lower right")
    ax.margins(y=0.16)
    return save(fig, out, "fig6_quant_ladder")


def fig_kv_correction(dec, out: Path) -> str:
    """What the uniform-attention assumption costs, per model.

    A slope chart, because the claim is about the SIZE of a per-model error and
    which models carry it — two paired values each, not a distribution. Log axis
    because the errors span an order of magnitude.
    """
    from .common import read_gguf_meta

    d = dec[dec["n_depth"] == dec["n_depth"].max()].copy()
    if d.empty or d["n_depth"].iloc[0] == 0:
        return ""
    depth = int(d["n_depth"].iloc[0])

    rows = []
    for name, g in d.groupby("model_file"):
        p = MODELS_DIR / str(name)
        if not p.is_file():
            continue
        try:
            m = read_gguf_meta(p)
        except Exception:
            continue
        uniform = (2 * m.n_layers * m.n_kv_heads * m.head_dim * depth
                   * 2) / 1e9
        actual = m.kv_bytes(depth) / 1e9
        if uniform <= 0 or actual <= 0:
            continue
        rows.append((g["model_name"].iloc[0], uniform, actual, uniform / actual))
    rows = [r for r in rows if r[3] > 1.05]
    if len(rows) < 2:
        return ""
    rows.sort(key=lambda r: -r[3])

    fig, ax = plt.subplots(figsize=(WIDE * 0.56, COL * 0.9))
    for i, (name, uni, act, ratio) in enumerate(rows):
        ax.plot([0, 1], [uni, act], "-", color=C[1], lw=1.1, alpha=.75, zorder=2)
        ax.scatter([0], [uni], s=26, color=C[1], zorder=3)
        ax.scatter([1], [act], s=26, color=C[2], zorder=3)
        ax.annotate(f"{name[:26]}  {ratio:.1f}×", (0, uni),
                    textcoords="offset points", xytext=(-6, 0), ha="right",
                    va="center", fontsize=6, color=INK)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["uniform global\nattention (assumed)",
                        "per-layer\n(measured config)"])
    ax.set_xlim(-0.95, 1.25)
    ax.set_yscale("log")
    ax.set_ylabel(f"KV cache read per token at {depth:,} ctx (GB)")
    ax.grid(axis="x", visible=False)
    ax.set_title("Charging every layer a full growing cache", loc="left",
                 color=INK, fontsize=8)
    return save(fig, out, "fig7_kv_correction")


def fig_quality_vs_error(dec, pred, out: Path) -> str:
    """Measurement quality against prediction error.

    Supports the claim that the model's worst failures are unstable measurements
    rather than unmodelled architecture. llama-bench's own within-run CV is on
    the x axis and is known before any prediction is made, so this is not
    circular.
    """
    d = dec.copy()
    d["cv"] = d["stddev_ts"] / d["avg_ts"] * 100
    d["ape"] = ape(pred, d["avg_ts"].to_numpy())
    d = d[d["cv"].notna() & (d["cv"] > 0)]
    if len(d) < 10:
        return ""

    fig, ax = plt.subplots(figsize=(COL, COL * 0.82))
    for split, colour, marker in (("train", C[0], "o"), ("test", C[1], "^")):
        m = (d["split"] == split).to_numpy()
        if m.any():
            ax.scatter(d["cv"][m], d["ape"][m], s=20, marker=marker,
                       facecolors="none", edgecolors=colour, linewidths=1.0,
                       label=split, zorder=3)
    ax.axvline(3.0, color=INK2, ls=":", lw=0.9, zorder=2)
    ax.text(3.15, ax.get_ylim()[1] * 0.94, "quality gate", fontsize=6.2,
            color=INK2, va="top")

    r = d[["cv", "ape"]].corr(method="spearman").iloc[0, 1]
    ax.set_xlabel("within-run CV of the measurement (%)")
    ax.set_ylabel("prediction error (APE %)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(f"Bad predictions are bad measurements\nSpearman ρ = {r:.2f}, "
                 f"n = {len(d)}", loc="left", color=INK, fontsize=8)
    ax.legend(loc="upper left")
    return save(fig, out, "fig8_quality_vs_error")


def fig_offload(dec, out: Path) -> str:
    """The offload cliff. Only produced on a discrete-GPU host with an ngl sweep."""
    d = dec[dec["n_gpu_layers"].notna() & (dec["n_depth"] == 0)]
    if d["n_gpu_layers"].nunique() < 3:
        return ""
    fig, ax = plt.subplots(figsize=(COL, COL * 0.8))
    for i, (name, g) in enumerate(d.groupby("model_name")):
        if g["n_gpu_layers"].nunique() < 3:
            continue
        g = g.sort_values("n_gpu_layers")
        frac = np.minimum(g["n_gpu_layers"] / g["n_layers"], 1.0) * 100
        ax.plot(frac, g["avg_ts"], marker="o", ms=3.5, color=C[i % len(C)],
                label=name[:22], zorder=3)
    ax.set_xlabel("% of layers resident on GPU")
    ax.set_ylabel("decode throughput (tok/s)")
    ax.set_yscale("log")
    ax.legend(loc="upper left")
    return save(fig, out, "fig9_offload_cliff")


# --------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=FIGURES_DIR / "paper")
    args = ap.parse_args(argv)

    _, dec = _load()
    train = dec[dec["split"] == "train"]
    if train.empty:
        train = dec
    eta_a = fit_eta(train, "bytes_active")
    eta_t = fit_eta(train, "bytes_total")
    preds = {
        "B0 uncalibrated": dec["bw"].to_numpy() / dec["bytes_total"].to_numpy(),
        "B1 total params": apply_eta(dec, eta_t, "bytes_total"),
        "B2 active params": apply_eta(dec, eta_a, "bytes_active"),
    }
    ours = preds["B2 active params"]

    made = [
        fig_accuracy(dec, ours, args.out),
        fig_ablation(dec, preds, args.out),
        fig_noise_floor(args.out),
        fig_eta(dec, args.out),
        fig_context(dec, args.out),
        fig_quant_ladder(dec, args.out),
        fig_kv_correction(dec, args.out),
        fig_quality_vs_error(dec, ours, args.out),
        fig_offload(dec, args.out),
    ]
    made = [m for m in made if m]

    skipped = {
        "fig6_quant_ladder": "needs >=4 quantisations of one model",
        "fig7_kv_correction": "needs models with non-uniform attention",
        "fig8_quality_vs_error": "needs stddev recorded on >=10 rows",
        "fig9_offload_cliff": "needs an n_gpu_layers sweep on a discrete-GPU host",
    }
    for name, why in skipped.items():
        if name not in made:
            print(f"skipped {name}: {why}")

    print(f"\n{len(made)} figures (pdf + png) -> {args.out}")
    for m in made:
        print(f"  {m}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
