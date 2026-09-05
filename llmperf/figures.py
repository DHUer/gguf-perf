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
                       load_measurements, load_splits,
                       select_primary_measurements)
from .common import (FIGURES_DIR, MODELS_DIR, RESULTS_DIR,
                     load_model_metadata)

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
    # ICASSP requires every label, including text inside figures, to be at
    # least 9 pt at its final printed size.
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "pdf.fonttype": 42, "ps.fonttype": 42,
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
    # Publication figures must be reproducible from the archived result bundle;
    # an unrelated GGUF currently installed in models/ must not change them.
    df = add_features(df, cal, load_splits(MODELS_DIR),
                      load_model_metadata(RESULTS_DIR), MODELS_DIR,
                      prefer_live=False)
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
    lo = float(min(dec["avg_ts"].min(), np.nanmin(pred))) * 0.65
    hi = float(max(dec["avg_ts"].max(), np.nanmax(pred))) * 1.5
    hosts = sorted(dec["host"].unique())
    fig, axes = plt.subplots(1, len(hosts), figsize=(WIDE, 3.35),
                             sharex=True, sharey=True, squeeze=False)
    axes = axes.ravel()
    host_names = {"lun-mac": "Apple M4 Max", "rtx5080": "RTX 5080"}
    groups = [
        ("train, dense", False, "train", "o", C[0], True),
        ("train, MoE", True, "train", "s", C[0], False),
        ("test, dense", False, "test", "^", C[1], True),
        ("test, MoE", True, "test", "D", C[1], False),
    ]
    handles, labels = [], []
    for ax, host in zip(axes, hosts):
        hm = dec["host"].eq(host).to_numpy()
        for label, is_moe, split, marker, colour, filled in groups:
            m = (hm & dec["is_moe"].eq(is_moe).to_numpy()
                 & dec["split"].eq(split).to_numpy())
            if not m.any():
                continue
            artist = ax.scatter(
                dec["avg_ts"][m], np.asarray(pred)[m], marker=marker, s=22,
                facecolors=colour if filled else "none", edgecolors=colour,
                linewidths=1.0, alpha=0.9, zorder=3,
            )
            if label not in labels:
                handles.append(artist)
                labels.append(label)
        ax.plot([lo, hi], [lo, hi], "-", color=INK2, lw=0.9, zorder=2)
        for factor in (1.25, 0.75):
            ax.plot([lo, hi], [lo * factor, hi * factor], "--",
                    color=MUTED, lw=0.6, zorder=1)
        ax.set(xscale="log", yscale="log", xlim=(lo, hi), ylim=(lo, hi))
        ax.set_aspect("equal")
        ax.set_title(host_names.get(host, host), fontsize=9, fontweight="bold")
        ax.set_xlabel("measured decode (tok/s)")
    axes[0].set_ylabel("predicted decode (tok/s)")
    axes[-1].text(hi * 0.97, hi * 0.8 * 0.97, "±25%", color=MUTED,
                  fontsize=9, ha="right", va="top")
    axes[0].legend(handles, labels, loc="upper left", ncol=2,
                   borderpad=0.2, columnspacing=0.7, handletextpad=0.3,
                   labelspacing=0.3)
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.18, top=0.90,
                        wspace=0.16)
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
                    ha="center", va="bottom", fontsize=9, color=INK)

    ax.set_xticks(x + width * (len(names) - 1) / 2)
    ax.set_xticklabels(labels)
    ax.set_ylabel("mean absolute percentage error")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.set_axisbelow(True)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3,
              columnspacing=0.8, handletextpad=0.4)
    return save(fig, out, "fig2_ablation")


def fig_mechanisms(dec, preds, out: Path) -> str:
    """Compact main-paper evidence for the two structural corrections.

    Panel (a) deliberately separates held-out dense and MoE configurations.
    Bars summarize all context-depth observations, while the open points are
    per-configuration means over the three depths.  The points make the small
    held-out sample (two configurations per architecture) explicit.

    Panel (b) derives both sides of the KV ratio from the audited metadata
    snapshot.  It must not depend on the large GGUF files being present in a
    checkout, and the result is labelled metadata-aware rather than measured.
    """
    d = dec.reset_index(drop=True).copy()
    pred_arrays = {name: np.asarray(values) for name, values in preds.items()}

    # select_primary_measurements() normally enforces this before the function
    # is called. Keep the guard here so the publication figure cannot silently
    # mix legacy and quality-gated rows when called on its own.
    if "quality_protocol" in d.columns:
        gated = d["quality_protocol"].fillna(False).astype(bool).to_numpy()
        if gated.any():
            d = d.loc[gated].reset_index(drop=True)
            pred_arrays = {name: values[gated] for name, values in pred_arrays.items()}

    held = d["split"].eq("test").to_numpy()
    if not held.any():
        return ""

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(WIDE, 2.95),
        gridspec_kw={"width_ratios": [1.08, 0.92]},
    )

    # (a) Held-out decode ablation, reported separately by host. Pooling would
    # give the larger Mac cohort twice the weight of RTX and hide the latter's
    # substantially higher error.
    hosts = sorted(d.loc[held, "host"].unique())
    host_names = {"lun-mac": "Apple M4 Max", "rtx5080": "RTX 5080"}
    names = list(pred_arrays)
    pretty = {
        "B0 uncalibrated": "B0",
        "B1 total params": "B1",
        "B2 active params": "B2",
    }
    hatches = ["", "//", ".."]
    x = np.arange(len(hosts), dtype=float)
    width = 0.23
    offsets = (np.arange(len(names)) - (len(names) - 1) / 2) * width
    legend_handles = []
    legend_labels = []
    model_counts = []

    measured = d["avg_ts"].to_numpy()
    for host in hosts:
        model_counts.append(d.loc[held & d["host"].eq(host).to_numpy(),
                                  "model_file"].nunique())

    for i, name in enumerate(names):
        values = []
        per_config = []
        for host in hosts:
            mask = held & d["host"].eq(host).to_numpy()
            row_ape = ape(pred_arrays[name][mask], measured[mask])
            values.append(float(np.mean(row_ape)))
            cfg = pd.DataFrame({
                "model_file": d.loc[mask, "model_file"].to_numpy(),
                "is_moe": d.loc[mask, "is_moe"].to_numpy(),
                "ape": row_ape,
            }).groupby(["model_file", "is_moe"], sort=True)["ape"].mean().reset_index()
            per_config.append(cfg)

        bars = ax1.bar(
            x + offsets[i], values, width * 0.88,
            color=C[i], edgecolor=INK, linewidth=0.55,
            hatch=hatches[i], zorder=2,
        )
        legend_handles.append(bars[0])
        legend_labels.append(pretty.get(name, name))

        for bar_x, value, cfg in zip(x + offsets[i], values, per_config):
            jitter = (np.linspace(-0.035, 0.035, len(cfg))
                      if len(cfg) > 1 else np.array([0.0]))
            for is_moe, marker in ((False, "o"), (True, "s")):
                cm = cfg["is_moe"].eq(is_moe).to_numpy()
                if cm.any():
                    ax1.scatter(
                        bar_x + np.asarray(jitter)[cm], cfg.loc[cm, "ape"],
                        marker=marker, s=20, facecolors=SURFACE,
                        edgecolors=INK, linewidths=0.8, zorder=4,
                    )
            ax1.text(bar_x, value + 2.2, f"{value:.1f}", ha="center",
                     va="bottom", fontsize=9, color=INK)

    legend_handles.append(Line2D([], [], marker="o", ls="", ms=4.5,
                                 mfc=SURFACE, mec=INK, mew=0.8))
    legend_labels.append("dense config.")
    legend_handles.append(Line2D([], [], marker="s", ls="", ms=4.5,
                                 mfc=SURFACE, mec=INK, mew=0.8))
    legend_labels.append("MoE config.")
    ax1.set_xticks(x)
    ax1.set_xticklabels([
        f"{host_names.get(host, host)}\n({n} configs. × 3 depths)"
        for host, n in zip(hosts, model_counts)
    ])
    ax1.set_ylabel("held-out decode MAPE (%)")
    ax1.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax1.set_ylim(0, max(80, ax1.get_ylim()[1]))
    ax1.grid(axis="x", visible=False)
    ax1.set_axisbelow(True)
    ax1.set_title("(a) Active-parameter ablation", loc="left",
                  fontweight="bold", fontsize=9)
    bars_legend = ax1.legend(
        legend_handles[:3], legend_labels[:3], loc="upper left", ncol=3,
        columnspacing=0.55, handletextpad=0.3, borderpad=0.2)
    ax1.add_artist(bars_legend)
    ax1.legend(legend_handles[3:], ["dense", "MoE"], loc="upper right",
               ncol=1, handletextpad=0.3, borderpad=0.2, labelspacing=0.25)

    # (b) Metadata-derived KV accounting ratios at the maximum tested depth.
    depth = 16_384
    metadata = load_model_metadata(RESULTS_DIR)
    targets = [
        ("Gemma-4 12B", "gemma-4-12b-it-qat-q4_0.gguf"),
        ("Gemma-4 26B-A4B", "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"),
        ("Nemotron 4B", "NVIDIA-Nemotron-3-Nano-4B-Q4_K_M.gguf"),
        ("Nemotron 30B-A3B", "Nemotron-3-Nano-30B-A3B-Q4_K_M.gguf"),
    ]
    kv_rows = []
    for label, filename in targets:
        if filename not in metadata:
            raise KeyError(f"metadata snapshot lacks {filename}")
        m = metadata[filename]
        per_layer = float(m["kv_bytes_by_depth"][str(depth)])
        uniform = (2 * float(m["n_layers"]) * float(m["n_kv_heads"])
                   * float(m["head_dim"]) * depth * 2)
        if per_layer <= 0:
            raise ValueError(f"non-positive per-layer KV bytes for {filename}")
        kv_rows.append((label, uniform / per_layer))

    labels = [row[0] for row in kv_rows]
    ratios = np.asarray([row[1] for row in kv_rows])
    y = np.arange(len(kv_rows))
    ax2.axvline(1.0, color=MUTED, lw=0.8, ls="--", zorder=1)
    ax2.hlines(y, 1.0, ratios, color=C[2], lw=3.0, zorder=2)
    ax2.scatter(ratios, y, marker="D", s=34, color=C[2], edgecolors=INK,
                linewidths=0.55, zorder=3)
    for yi, ratio in zip(y, ratios):
        ax2.annotate(f"{ratio:.1f}×", (ratio, yi), xytext=(6, 0),
                     textcoords="offset points", ha="left", va="center",
                     fontsize=9, color=INK)
    ax2.set_xscale("log", base=2)
    ax2.set_xlim(0.9, 21)
    ax2.set_xticks([1, 2, 4, 8, 16])
    ax2.set_xticklabels(["1×", "2×", "4×", "8×", "16×"])
    ax2.set_yticks(y)
    ax2.set_yticklabels(labels)
    ax2.invert_yaxis()
    ax2.set_xlabel("uniform-global / per-layer KV bytes at 16,384")
    ax2.grid(axis="y", visible=False)
    ax2.set_axisbelow(True)
    ax2.set_title("(b) Metadata-aware KV correction", loc="left",
                  fontweight="bold", fontsize=9)

    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.23, top=0.88,
                        wspace=0.50)
    return save(fig, out, "fig2_mechanisms")


def fig_noise_floor(out: Path) -> str:
    """Repeatability and archived interference, presented descriptively.

    Every point is drawn. With six runs a box plot would hide the data behind a
    summary of six numbers, which is exactly the sort of thing this figure
    exists to argue against. The archived 15--26% range is not the same
    statistic as the controlled CV and is intentionally drawn as an interval,
    not as a bar or an inferential threshold.
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
        1, 2, figsize=(WIDE, 2.65),
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
                 ha="left", fontsize=9, color=C[1])
    ax1.text(runs[0], mean, f"mean ±{cv_ctl:.1f}%  ", va="bottom", ha="left",
             fontsize=9, color=INK2)
    ax1.set_xlabel("independent run")
    ax1.set_ylabel("decode throughput (tok/s)")
    ax1.set_xticks(runs)
    ax1.margins(x=0.16)
    ax1.set_title("(a) Six controlled repetitions", loc="left", color=INK,
                  fontweight="bold", fontsize=9)

    # These archived interference sessions yielded a 15--26% descriptive
    # range. It is not a CV estimate and must not be presented as one.
    interference_low, interference_high = 15.0, 26.0
    ax2.scatter([0], [cv_ctl], marker="o", s=38, color=C[2], edgecolors=INK,
                linewidths=0.6, zorder=4)
    ax2.text(0, cv_ctl + 1.2, f"CV {cv_ctl:.1f}%", ha="center", va="bottom",
             fontsize=9, color=INK)
    ax2.vlines(1, interference_low, interference_high, color=C[1], lw=5.0,
               zorder=3)
    ax2.hlines([interference_low, interference_high], 0.90, 1.10,
               color=INK, lw=0.9, zorder=4)
    ax2.text(1, interference_high + 1.2, "15–26% range", ha="center",
             va="bottom", fontsize=9, color=INK)
    ax2.set_ylim(0, 31)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["controlled\n(6 runs)",
                         "archived\ninterference sessions"])
    ax2.set_ylabel("descriptive variation (%)")
    ax2.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax2.grid(axis="x", visible=False)
    ax2.set_axisbelow(True)
    ax2.set_title("(b) Controlled vs interference", loc="left", color=INK,
                  fontweight="bold", fontsize=9)
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.25, top=0.89,
                        wspace=0.38)
    return save(fig, out, "fig3_noise_floor")


def fig_eta(dec, out: Path) -> str:
    """Efficiency by quantisation. Job: magnitude across categories, with a
    hard reference line at eta = 1.

    Points, not bars, because each model is an observation and the spread within
    a format is the finding. Values above one mean that file size and the copy
    microbenchmark are imperfect traffic/ceiling proxies, not impossible data.
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
    ax.text(len(order) - 0.5, 1.02, "above nominal copy ceiling",
            ha="right", va="bottom", fontsize=9, color=C[1])

    hosts = sorted(d["host"].unique())
    host_colours = {host: C[i % len(C)] for i, host in enumerate(hosts)}
    rng = np.random.default_rng(0)
    for i, q in enumerate(order):
        for hi, host in enumerate(hosts):
            g = d[(d["quant"] == q) & (d["host"] == host)]
            if g.empty:
                continue
            centre = i + (hi - (len(hosts) - 1) / 2) * 0.18
            jitter = rng.uniform(-0.04, 0.04, len(g))
            moe = g["is_moe"].to_numpy()
            for mask, marker in ((~moe, "o"), (moe, "s")):
                if mask.any():
                    ax.scatter(centre + jitter[mask], g["eta"].to_numpy()[mask],
                               marker=marker, s=24, facecolors="none",
                               edgecolors=host_colours[host], linewidths=1.1,
                               zorder=4)
            ax.plot([centre - 0.07, centre + 0.07], [g["eta"].median()] * 2,
                    color=host_colours[host], lw=1.4, zorder=5)

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=30, ha="right")
    ax.set_ylabel(r"$\eta$  (fraction of memory roofline)")
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    host_names = {"lun-mac": "Apple M4 Max", "rtx5080": "RTX 5080"}
    handles = [Line2D([], [], color=host_colours[h], lw=1.4,
                      label=host_names.get(h, h)) for h in hosts]
    handles += [
        Line2D([], [], marker="o", ls="", mfc="none", mec=INK, label="dense"),
        Line2D([], [], marker="s", ls="", mfc="none", mec=INK, label="MoE"),
    ]
    ax.legend(handles=handles, loc="upper left", ncol=2)
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
    hosts = sorted(d["host"].unique())
    host_colours = {host: C[i % len(C)] for i, host in enumerate(hosts)}
    for (host, name), g in d.groupby(["host", "model_name"]):
        g = g.sort_values("n_depth")
        base = g[g["n_depth"] == 0]["avg_ts"]
        if g["n_depth"].nunique() < 2 or base.empty:
            continue
        moe = bool(g["is_moe"].iloc[0])
        ax.plot(g["n_depth"], g["avg_ts"] / float(base.iloc[0]),
                marker="s" if moe else "o", ms=3.5,
                ls="--" if moe else "-", lw=1.0,
                color=host_colours[host], alpha=0.72, zorder=3)

    ax.set_xlabel("KV-cache depth (tokens)")
    ax.set_ylabel("throughput relative to empty cache")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    host_names = {"lun-mac": "Apple M4 Max", "rtx5080": "RTX 5080"}
    handles = [Line2D([], [], color=host_colours[h], lw=1.4,
                      label=host_names.get(h, h)) for h in hosts]
    handles += [
        Line2D([], [], color=INK, marker="o", ms=3.5, label="dense"),
        Line2D([], [], color=INK, marker="s", ms=3.5, ls="--", label="MoE"),
    ]
    ax.legend(handles=handles, loc="lower left", ncol=2)
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
    g = g.dropna(subset=["bits"]).sort_values(["host", "bits"])
    if len(g) < 4:
        return ""
    g["eta"] = g["avg_ts"] * g["bytes_active"] / g["bw"]
    g["is_iquant"] = g["quant"].str.upper().str.startswith("IQ")

    fig, ax = plt.subplots(figsize=(COL, COL * 0.82))
    hosts = sorted(g["host"].unique())
    host_names = {"lun-mac": "Apple M4 Max", "rtx5080": "RTX 5080"}
    for i, host in enumerate(hosts):
        hg = g[g["host"] == host].sort_values("bits")
        ax.plot(hg["bits"], hg["eta"], "-o", color=C[i % len(C)],
                lw=1.2, ms=4.5, markeredgecolor=SURFACE,
                markeredgewidth=0.6, zorder=3,
                label=host_names.get(host, host))
    ticks = (g.sort_values("bits").drop_duplicates("quant")
             [["bits", "quant"]])
    ax.set_xticks(ticks["bits"])
    ax.set_xticklabels(ticks["quant"], rotation=35, ha="right")

    ax.set_xlabel("effective bits per weight")
    ax.set_ylabel(r"$\eta$  (fraction of bandwidth ceiling)")
    ax.set_title(f"{target}: efficiency rises with bit width", loc="left",
                 color=INK, fontsize=9)
    ax.legend(loc="lower right")
    ax.margins(y=0.16)
    return save(fig, out, "fig6_quant_ladder")


def fig_scope(all_rows, out: Path) -> str:
    """Cross-host quantization behavior and prefill scope failure.

    Panel (a) holds the model fixed and separates hosts, avoiding the invalid
    implication that one backend's efficiency curve transfers to the other.
    Panel (b) reads the frozen row-level predictions and likewise reports each
    host instead of a cohort-size-weighted pooled mean.
    """
    decode = all_rows[(all_rows["phase"] == "decode")
                      & all_rows["bw"].notna()].copy()
    decode = select_primary_measurements(decode).reset_index(drop=True)
    ladder = decode[
        decode["model_name"].str.contains("Qwen3.8-27B", case=False, na=False)
        & decode["n_depth"].eq(0)
    ].copy()
    if ladder.empty:
        return ""
    ladder["file_bpp"] = 8 * ladder["file_bytes"] / ladder["n_params"]
    ladder["eta"] = ladder["avg_ts"] * ladder["bytes_active"] / ladder["bw"]
    quant_order = (ladder.groupby("quant")["file_bpp"].median()
                   .sort_values().index.tolist())

    prefill_path = RESULTS_DIR / "predictions_prefill.csv"
    if not prefill_path.exists():
        return ""
    prefill = pd.read_csv(prefill_path)
    prefill = prefill[prefill["split"].eq("test")].copy()
    if "quality_protocol" in prefill.columns:
        gated = prefill["quality_protocol"].astype(str).str.lower().eq("true")
        if gated.any():
            prefill = prefill[gated]
    for col in ("n_depth", "ape_P1", "ape_P2"):
        prefill[col] = pd.to_numeric(prefill[col], errors="coerce")
    summary = (prefill.groupby(["host", "n_depth"])["ape_P2"]
               .mean().unstack("n_depth").reindex(columns=[0, 4096, 16384]))
    summary = summary.dropna()
    if summary.empty:
        return ""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(WIDE, 2.72),
                                   gridspec_kw={"width_ratios": [1.08, 0.92]})

    # (a) Single-family quantization ladder at empty context.
    xpos = np.arange(len(quant_order), dtype=float)
    ax1.axhline(1.0, color=MUTED, lw=0.8, ls="--", zorder=1)
    ax1.text(-0.35, 1.015, r"$T_dD/B_h=1$", ha="left", va="bottom",
             fontsize=9, color=INK2)
    host_names = {"lun-mac": "Apple M4 Max", "rtx5080": "RTX 5080"}
    for i, host in enumerate(sorted(ladder["host"].unique())):
        hg = (ladder[ladder["host"] == host].set_index("quant")
              .reindex(quant_order))
        present = hg["eta"].notna().to_numpy()
        marker = "o" if host == "lun-mac" else "s"
        linestyle = "-" if host == "lun-mac" else "--"
        ax1.plot(xpos[present], hg.loc[present, "eta"], marker=marker,
                 linestyle=linestyle,
                 color=C[i % len(C)], markerfacecolor=C[i % len(C)],
                 markeredgecolor=INK, markeredgewidth=0.5,
                 label=host_names.get(host, host), zorder=3)
        probes = hg["is_calibration_probe"].fillna(False).astype(bool).to_numpy()
        probe_points = present & probes
        if probe_points.any():
            ax1.scatter(xpos[probe_points], hg.loc[probe_points, "eta"],
                        marker="X", s=52, facecolors=SURFACE,
                        edgecolors=INK, linewidths=1.0,
                        label="M4 reference probe", zorder=5)
    ax1.set_xticks(xpos)
    ax1.set_xticklabels(quant_order, rotation=38, ha="right",
                        rotation_mode="anchor")
    ax1.set_ylabel(r"back-solved ratio $T_dD/B_h$")
    ax1.set_xlabel("quantization (increasing file bits/parameter)")
    ax1.set_ylim(0.34, max(1.24, float(ladder["eta"].max()) + 0.11))
    ax1.grid(axis="x", visible=False)
    ax1.set_axisbelow(True)
    ax1.set_title("(a) Qwen3.8-27B across systems", loc="left",
                  fontweight="bold", fontsize=9)
    ax1.legend(loc="lower right", labelspacing=0.3, handletextpad=0.4)

    # (b) Held-out prefill error at the primary and stress-only prefix depths.
    px = np.arange(3, dtype=float)
    ax2.axvspan(-0.25, 0.25, color=C[2], alpha=0.09, zorder=0)
    for i, (host, row) in enumerate(summary.iterrows()):
        values = row.to_numpy(float)
        colour = C[i % len(C)]
        yoff = 7 if i == 0 else -14
        ax2.plot(px, values, color=colour, marker="o" if i == 0 else "s",
                 ls="-" if i == 0 else "--",
                 markerfacecolor=SURFACE, markeredgecolor=colour,
                 markeredgewidth=1.2, label=host_names.get(host, host), zorder=3)
        for xi, value in zip(px, values):
            at_right = xi == px[-1]
            ax2.annotate(f"{value:.1f}%", (xi, value),
                         xytext=(-5 if at_right else 0, yoff),
                         textcoords="offset points",
                         ha="right" if at_right else "center",
                         va="bottom" if yoff > 0 else "top", fontsize=9,
                         color=colour)
    ax2.set_xticks(px)
    ax2.set_xticklabels(["0\n(primary)", "4,096", "16,384"])
    ax2.set_xlabel("prefix depth (tokens)")
    ax2.set_ylabel("held-out prefill MAPE (%)")
    ax2.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax2.set_ylim(0, max(150, float(summary.to_numpy().max()) + 18))
    ax2.set_xlim(-0.35, 2.22)
    ax2.grid(axis="x", visible=False)
    ax2.set_axisbelow(True)
    ax2.set_title("(b) P2 prefill error by host", loc="left",
                  fontweight="bold", fontsize=9)
    ax2.legend(loc="upper left", labelspacing=0.3, handletextpad=0.4)

    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.29, top=0.89,
                        wspace=0.44)
    return save(fig, out, "fig3_scope")


def fig_kv_correction(dec, out: Path) -> str:
    """Snapshot-backed detail view of uniform versus per-layer KV bytes."""
    del dec  # The audited snapshot, not live GGUF availability, is authoritative.
    depth = 16_384
    metadata = load_model_metadata(RESULTS_DIR)
    targets = [
        ("Gemma-4 12B", "gemma-4-12b-it-qat-q4_0.gguf"),
        ("Gemma-4 26B-A4B", "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"),
        ("Nemotron 4B", "NVIDIA-Nemotron-3-Nano-4B-Q4_K_M.gguf"),
        ("Nemotron 30B-A3B", "Nemotron-3-Nano-30B-A3B-Q4_K_M.gguf"),
    ]
    rows = []
    for label, filename in targets:
        if filename not in metadata:
            continue
        m = metadata[filename]
        uniform = (2 * float(m["n_layers"]) * float(m["n_kv_heads"])
                   * float(m["head_dim"]) * depth * 2) / 1e9
        per_layer = float(m["kv_bytes_by_depth"][str(depth)]) / 1e9
        if uniform > 0 and per_layer > 0:
            rows.append((label, uniform, per_layer, uniform / per_layer))
    if len(rows) < 2:
        return ""

    fig, ax = plt.subplots(figsize=(WIDE, 2.65))
    for name, uniform, per_layer, ratio in rows:
        ax.plot([0, 1], [uniform, per_layer], "-", color=C[1], lw=1.4,
                alpha=.8, zorder=2)
        ax.scatter([0], [uniform], marker="o", s=34, color=C[1],
                   edgecolors=INK, linewidths=0.55, zorder=3)
        ax.scatter([1], [per_layer], marker="D", s=34, color=C[2],
                   edgecolors=INK, linewidths=0.55, zorder=3)
        ax.annotate(f"{name}  {ratio:.1f}×", (0, uniform),
                    textcoords="offset points", xytext=(-7, 0), ha="right",
                    va="center", fontsize=9, color=INK)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["uniform global\nassumption",
                        "metadata-aware\nper-layer accounting"])
    ax.set_xlim(-0.72, 1.18)
    ax.set_yscale("log")
    ax.set_ylabel(f"KV bytes per token at {depth:,} prefix (GB)")
    ax.grid(axis="x", visible=False)
    ax.set_title("Uniform attention overstates hybrid-model KV traffic",
                 loc="left", color=INK, fontsize=9, fontweight="bold")
    fig.subplots_adjust(left=0.31, right=0.98, bottom=0.27, top=0.87)
    return save(fig, out, "fig7_kv_correction")


def fig_quality_vs_error(dec, pred, all_rows, out: Path) -> str:
    """Decode residual check plus the phase-specific measurement-quality gap."""
    d = dec.copy()
    d["cv"] = d["stddev_ts"] / d["avg_ts"] * 100
    d["ape"] = ape(pred, d["avg_ts"].to_numpy())
    d = d[d["cv"].notna() & (d["cv"] > 0)]
    if len(d) < 10:
        return ""

    quality = select_primary_measurements(all_rows).copy()
    quality["cv"] = quality["stddev_ts"] / quality["avg_ts"] * 100
    quality = quality[quality["cv"].notna() & (quality["cv"] > 0)]

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(WIDE, 2.72), gridspec_kw={"width_ratios": [1.08, 0.92]})
    host_names = {"lun-mac": "Apple M4 Max", "rtx5080": "RTX 5080"}
    host_colours = {h: C[i % len(C)] for i, h in enumerate(sorted(d["host"].unique()))}
    for host in sorted(d["host"].unique()):
        for split, marker in (("train", "o"), ("test", "^")):
            m = ((d["host"] == host) & (d["split"] == split)).to_numpy()
            if m.any():
                ax.scatter(d["cv"][m], d["ape"][m], s=18, marker=marker,
                           facecolors=("none" if host == "lun-mac"
                                       else host_colours[host]),
                           edgecolors=host_colours[host],
                           linewidths=0.9,
                           label=f"{'M4' if host == 'lun-mac' else 'RTX'} {split}",
                           zorder=3)
    ax.axvline(3.0, color=INK2, ls=":", lw=0.9, zorder=2)
    r = d[["cv", "ape"]].corr(method="spearman").iloc[0, 1]
    ax.set_xlabel("decode within-run CV (%)")
    ax.set_ylabel("B2 prediction error (APE %)")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_title(f"(a) No obvious pooled CV--residual association  ($ρ={r:.2f}$, n={len(d)})",
                 loc="left", color=INK, fontsize=9, fontweight="bold")
    ax.legend(loc="upper left", ncol=2, fontsize=9,
              columnspacing=0.6, handletextpad=0.25)

    groups = []
    labels = []
    colours = []
    for host in sorted(quality["host"].unique()):
        for phase, colour in (("decode", C[0]), ("prefill", C[1])):
            values = quality.loc[
                (quality["host"] == host) & (quality["phase"] == phase), "cv"
            ].to_numpy(float)
            if len(values):
                groups.append(values)
                short_host = "M4" if host == "lun-mac" else "RTX"
                short_phase = "D" if phase == "decode" else "P"
                labels.append(f"{short_host} {short_phase}")
                colours.append(colour)
    boxes = ax2.boxplot(groups, patch_artist=True, widths=0.52,
                        showfliers=False, medianprops={"color": INK, "lw": 1.0},
                        whiskerprops={"color": INK2, "lw": 0.7},
                        capprops={"color": INK2, "lw": 0.7})
    for patch, colour in zip(boxes["boxes"], colours):
        patch.set_facecolor(colour)
        patch.set_alpha(0.48)
        patch.set_edgecolor(INK)
        patch.set_linewidth(0.6)
    for xpos, values, colour in zip(range(1, len(groups) + 1), groups, colours):
        jitter = np.linspace(-0.16, 0.16, len(values))
        ax2.scatter(xpos + jitter, values, s=5, color=colour, alpha=0.48,
                    edgecolors="none", zorder=3)
        above = int((values > 3.0).sum())
        ax2.text(xpos, 44, f"{above}/{len(values)} >3%", ha="center",
                 va="top", fontsize=9, color=INK2)
    ax2.axhline(3.0, color=INK2, ls=":", lw=0.9, zorder=2)
    ax2.set_yscale("log")
    ax2.set_ylim(0.05, 55)
    ax2.set_xticks(range(1, len(labels) + 1))
    ax2.set_xticklabels(labels)
    ax2.set_xlabel("D = decode; P = prefill")
    ax2.set_ylabel("within-run CV (%)")
    ax2.set_title("(b) The gate covered decode, not prefill", loc="left",
                  color=INK, fontsize=9, fontweight="bold")
    ax2.grid(axis="x", visible=False)
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.23, top=0.88,
                        wspace=0.38)
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

    all_rows, all_dec = _load()
    dec = select_primary_measurements(all_dec).reset_index(drop=True)
    if dec.empty:
        raise SystemExit("no n_gpu_layers=99 decode rows for predictor figures")
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
        fig_mechanisms(dec, preds, args.out),
        fig_noise_floor(args.out),
        fig_eta(dec, args.out),
        fig_context(dec, args.out),
        fig_quant_ladder(dec, args.out),
        fig_scope(all_rows, args.out),
        fig_kv_correction(dec, args.out),
        fig_quality_vs_error(dec, ours, all_rows, args.out),
        fig_offload(all_dec, args.out),
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
