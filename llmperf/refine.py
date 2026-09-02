"""Stage 4b: the section 5.7 go/no-go -- is eta predictable, or a lookup table?

WHY THIS EXISTS
---------------
analyze.py fits one efficiency factor eta_d per (host, quantisation) and treats
it as a constant. Section 5.2 shows it is not: on one machine at Q8_0,
SmolLM3-3B back-solves to 385 GB/s effective while Qwen3.5-4B back-solves to
287 GB/s. If eta cannot be predicted from GGUF metadata, the paper's model is a
per-model lookup table wearing a roofline costume.

The leading hypothesis (paper 5.7) is the OUTPUT PROJECTION: the logit GEMV is a
vocab_size x d_model product run once per token, depth-invariant, so it should
penalise wide-vocabulary shallow models and shrink in relative terms with depth.
That gives a second, non-bandwidth term in the per-token time:

    t_token = bytes_per_token / (eta_d * BW)  +  (2 * vocab * d_model) / (eta_o * FLOPS)

This module fits that two-term model against the existing one-term model on the
same rows and asks two questions: does it shrink the CROSS-MODEL spread of eta
inside a (host, quant) group, and does it shrink MAPE out of sample.

Caveat that must travel with any answer: the output head's weights are already
inside file_bytes, so the second term partly re-charges bytes the first term
already charged. It is an empirical "extra per-token cost proportional to
output-head size", not a clean compute roofline. eta_o is a fitted scale, and on
a host with no torch_gpu calibration the FLOPS term is a CPU fp32 number, so
eta_o there absorbs the wrong compute scale and is not interpretable on its own.

Nothing in llmperf/ is modified: a measurement campaign imports those modules
while this runs.

    python -m llmperf.refine
    python -m llmperf.refine --selfcheck
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

# Importing analyze pins the Agg backend and the shared rcParams, so figures
# from this module are styled identically to the rest of the paper's without
# copying the block.
from .analyze import (ape, add_features, load_calibration, load_measurements,
                      load_splits)
import matplotlib.pyplot as plt

from .common import FIGURES_DIR, MODELS_DIR, RESULTS_DIR, read_gguf_meta


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def model_shapes(model_files: list[str], models_dir: Path) -> pd.DataFrame:
    """vocab_size and d_model per GGUF file.

    Neither is in the measurement CSV -- sweep.py's CSV_FIELDS predates this
    question and we must not edit it while a campaign is writing to it -- so
    both are re-read from the GGUF headers. d_model and the embedding-table
    size come from common.read_gguf_meta; vocab_size is the leading dimension
    of token_embd.weight, which is present even in files whose architecture
    omits a vocab_size metadata key (every Qwen3.x file here does).

    A missing or unreadable file yields no row rather than an exception: the
    measurement CSV outlives the weights on a disk-constrained machine.
    """
    from gguf import GGUFReader  # lazy, matching common.read_gguf_meta

    rows = []
    for name in sorted(set(model_files)):
        path = models_dir / name
        if not path.is_file():
            continue
        try:
            meta = read_gguf_meta(path)
            reader = GGUFReader(str(path))
            vocab = 0
            for t in reader.tensors:
                if t.name == "token_embd.weight":
                    # GGUF stores ne as [n_embd, n_vocab].
                    vocab = int(t.shape[-1])
                    break
            if not vocab:
                tok = reader.fields.get("tokenizer.ggml.tokens")
                vocab = int(len(tok.data)) if tok is not None else 0
        except Exception as e:  # a truncated download must not kill the run
            print(f"[skip shapes] {name}: {e}")
            continue
        rows.append({
            "model_file": name,
            "vocab_size": vocab,
            "d_model": meta.n_embd,
            "embd_bytes": meta.embd_bytes,
            "tied_embeddings": meta.tied_embeddings,
        })
    return pd.DataFrame(rows)


def collapse_repeats(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep the fastest observation of each measurement cell.

    sweep.py's resume key is built with both n_prompt and n_gen set, but
    llama-bench writes prefill and decode as separate rows (one with n_gen=0,
    one with n_prompt=0), so the key never matches anything on disk and a
    re-run re-measures every cell and appends a second observation. Where that
    has happened the two passes can disagree by 3x -- two orders of magnitude
    above the 1.6% noise floor of paper 5.1 -- because the later pass ran under
    contention. Contention only ever removes throughput, so the maximum over
    repeats is the least-contaminated estimate of each cell.

    Returns (collapsed, disagreement) where disagreement lists every cell with
    more than one observation and the ratio between them.
    """
    key = ["host", "model_file", "n_gpu_layers", "n_depth"]
    g = df.groupby(key)["avg_ts"]
    dis = pd.DataFrame({"n_obs": g.size(), "slowest": g.min(), "fastest": g.max()})
    dis = dis[dis["n_obs"] > 1].reset_index()
    dis["ratio"] = dis["fastest"] / dis["slowest"]
    keep = df.loc[df.groupby(key)["avg_ts"].idxmax()].reset_index(drop=True)
    return keep, dis.sort_values("ratio", ascending=False)


def load_rows(results_dir: Path = RESULTS_DIR, models_dir: Path = MODELS_DIR,
              collapse: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Decode rows with per-row effective bandwidth and back-solved eta.

    Effective bandwidth is what the machine actually delivered on this cell:
    bytes_per_token * tok/s. eta is that as a fraction of the calibrated peak.
    Both are per row, so the cross-model variation of 5.7 can be read straight
    off the frame without going through any fit.
    """
    df = load_measurements(results_dir)
    cal = load_calibration(results_dir)
    if not cal:
        raise SystemExit(f"No calibration_*.json in {results_dir}. "
                         f"Run `python -m llmperf.calibrate` first.")
    df = add_features(df, cal, load_splits(models_dir))

    dec = df[(df["phase"] == "decode") & df["bw"].notna()].reset_index(drop=True)
    # eta is 1.0 by construction on the llm_ref probe model; keeping it would
    # flatter every number below. analyze.py drops it for the same reason.
    dec = dec[~dec["is_calibration_probe"]].reset_index(drop=True)

    dis = pd.DataFrame()
    if collapse:
        dec, dis = collapse_repeats(dec)

    shapes = model_shapes(dec["model_file"].tolist(), models_dir)
    dec = dec.merge(shapes, on="model_file", how="left")

    dec["t_meas"] = 1.0 / dec["avg_ts"]
    dec["eff_bw_gb_s"] = dec["bytes_active"] * dec["avg_ts"] / 1e9
    dec["eta"] = dec["bytes_active"] * dec["avg_ts"] / dec["bw"]
    # Same quantity with the untied-embedding correction of ModelMeta applied.
    # analyze.py charges the whole file per token; for an untied model the
    # embedding table is gathered, not streamed, so it should not be charged.
    streamed = np.where(dec["tied_embeddings"].fillna(True),
                        dec["file_bytes"],
                        dec["file_bytes"] - dec["embd_bytes"].fillna(0))
    dec["bytes_streamed"] = streamed * dec["active_frac"] + dec["kv_bytes"]
    dec["eta_streamed"] = dec["bytes_streamed"] * dec["avg_ts"] / dec["bw"]
    # The work the output-projection hypothesis charges per token, and that
    # work per streamed byte -- the quantity the hypothesis says eta falls with.
    dec["out_flops"] = 2.0 * dec["vocab_size"] * dec["d_model"]
    dec["out_load"] = dec["out_flops"] / dec["bytes_active"]
    return dec, dis


# --------------------------------------------------------------------------
# fitting
# --------------------------------------------------------------------------

@dataclass
class Fit:
    """eta_d per (host, quant), plus a shared eta_o for the two-term model."""

    eta_d: dict[tuple[str, str], float] = field(default_factory=dict)
    eta_o: float | None = None

    def eta_for(self, host: str, quant: str) -> float:
        """Fallback chain for a group not seen in training.

        Same shape as analyze.apply_eta: exact group, then the same quant on
        other hosts (this is what makes leave-one-machine-out possible at all),
        then the global median.
        """
        if (host, quant) in self.eta_d:
            return self.eta_d[(host, quant)]
        same_quant = [v for (h, q), v in self.eta_d.items() if q == quant]
        if same_quant:
            return float(np.median(same_quant))
        return float(np.median(list(self.eta_d.values())))

    def predict_time(self, df: pd.DataFrame) -> np.ndarray:
        e = np.array([self.eta_for(h, q) for h, q in zip(df["host"], df["quant"])])
        t = df["bytes_active"].to_numpy(float) / (e * df["bw"].to_numpy(float))
        if self.eta_o is not None:
            t = t + df["out_flops"].to_numpy(float) / (self.eta_o * df["flops"].to_numpy(float))
        return t

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return 1.0 / self.predict_time(df)


def fit_model(train: pd.DataFrame, use_output_term: bool,
              relative: bool = False) -> Fit:
    """Least squares on per-token TIME, TRAIN rows only.

    Time space rather than tok/s space: a 3B model at 160 tok/s and a 27B model
    at 10 tok/s differ by 16x in throughput, so a least-squares fit on tok/s is
    decided almost entirely by the smallest model in the set.

    Parameters are optimised in log space, which keeps eta positive without
    bounds and puts the two very differently-scaled etas on one footing.
    """
    groups = sorted({(h, q) for h, q in zip(train["host"], train["quant"])})
    idx = np.array([groups.index((h, q))
                    for h, q in zip(train["host"], train["quant"])])

    b = train["bytes_active"].to_numpy(float)
    bw = train["bw"].to_numpy(float)
    fl = train["flops"].to_numpy(float)
    out = train["out_flops"].to_numpy(float)
    t_meas = 1.0 / train["avg_ts"].to_numpy(float)

    # Start from the closed-form one-term estimate, which is what analyze.py
    # reports; the optimiser then only has to move as far as the data asks.
    eta0 = np.array([np.median(b[idx == i] * train["avg_ts"].to_numpy(float)[idx == i]
                               / bw[idx == i]) for i in range(len(groups))])
    x0 = np.log(np.clip(eta0, 1e-6, None))
    if use_output_term:
        x0 = np.append(x0, 0.0)

    def residual(x):
        eta_d = np.exp(x[:len(groups)])
        t = b / (eta_d[idx] * bw)
        if use_output_term:
            t = t + out / (np.exp(x[-1]) * fl)
        r = t - t_meas
        return r / t_meas if relative else r

    sol = least_squares(residual, x0, method="lm" if len(x0) < len(b) else "trf")
    eta_d = {g: float(np.exp(v)) for g, v in zip(groups, sol.x[:len(groups)])}
    return Fit(eta_d=eta_d,
               eta_o=float(np.exp(sol.x[-1])) if use_output_term else None)


def back_solved_eta(df: pd.DataFrame, eta_o: float | None) -> np.ndarray:
    """eta_d implied by each row once the output-projection time is removed.

    This is the quantity 5.7 is about. If the output projection is the whole
    story, this is constant across models inside a (host, quant) group.
    """
    t = 1.0 / df["avg_ts"].to_numpy(float)
    if eta_o is not None:
        t = t - df["out_flops"].to_numpy(float) / (eta_o * df["flops"].to_numpy(float))
    with np.errstate(divide="ignore", invalid="ignore"):
        eta = df["bytes_active"].to_numpy(float) / (df["bw"].to_numpy(float) * t)
    return np.where(t > 0, eta, np.nan)


def eta_spread(df: pd.DataFrame, eta_o: float | None = None,
               col: str | None = None) -> pd.DataFrame:
    """Cross-MODEL dispersion of eta inside each (host, quant) group.

    One row per model first, so a model measured at three KV depths does not
    get three votes; then max/min across models. Groups holding a single model
    are reported with ratio 1.0 and are not evidence either way.
    """
    d = df[["host", "quant", "model_name"]].copy()
    d["eta"] = df[col].to_numpy(float) if col else back_solved_eta(df, eta_o)
    per_model = d.groupby(["host", "quant", "model_name"])["eta"].median().reset_index()
    rows = []
    for (h, q), g in per_model.groupby(["host", "quant"]):
        v = g["eta"].dropna().to_numpy()
        if not len(v):
            continue
        rows.append({"host": h, "quant": q, "n_models": len(v),
                     "eta_min": v.min(), "eta_max": v.max(),
                     "max_over_min": v.max() / v.min() if v.min() > 0 else np.nan,
                     "cv_%": (v.std(ddof=0) / v.mean() * 100) if v.mean() else np.nan})
    return pd.DataFrame(rows).sort_values(["host", "quant"]).reset_index(drop=True)


def output_load_check(df: pd.DataFrame) -> pd.DataFrame:
    """Sign test of the hypothesis, independent of any fit.

    If the logit GEMV is what eta is missing, then inside a (host, quant) group
    eta must FALL as the output-projection work per streamed byte rises. That
    is a rank correlation, needs no fitted eta_o, and so cannot be rescued by
    the extra free parameter the two-term fit brings with it. The hypothesis
    predicts rho < 0.
    """
    from scipy.stats import spearmanr

    per_model = (df.groupby(["host", "quant", "model_name"])[["eta", "out_load"]]
                 .median().reset_index())
    rows = []
    for (h, q), g in per_model.groupby(["host", "quant"]):
        if len(g) < 3:  # rank correlation on two points is meaningless
            continue
        rho, p = spearmanr(g["out_load"], g["eta"])
        rows.append({"host": h, "quant": q, "n_models": len(g),
                     "spearman_rho": float(rho), "p": float(p),
                     "sign_as_predicted": bool(rho < 0)})
    return pd.DataFrame(rows)


def error_rows(name: str, fit: Fit, df: pd.DataFrame) -> list[dict]:
    pred = fit.predict(df)
    out = []
    for split in ("train", "test"):
        m = (df["split"] == split).to_numpy()
        if not m.any():
            continue
        e = ape(pred[m], df["avg_ts"].to_numpy()[m])
        out.append({"model": name, "split": split, "n": int(m.sum()),
                    "MAPE_%": e.mean(), "median_APE_%": float(np.median(e)),
                    "p90_APE_%": float(np.percentile(e, 90)), "max_APE_%": e.max()})
    return out


# --------------------------------------------------------------------------
# leave one machine out
# --------------------------------------------------------------------------

def leave_one_machine_out(df: pd.DataFrame, use_output_term: bool,
                          relative: bool = False) -> pd.DataFrame:
    """Fit on every host but one, predict the held-out host.

    The only honest cross-hardware validation available: within-host error can
    always be driven down by fitting more per-host constants. With a single
    host in the data this is undefined, and says so instead of returning a
    number that looks like a result.
    """
    hosts = sorted(df["host"].unique())
    if len(hosts) < 2:
        print(f"leave-one-machine-out: only {len(hosts)} host "
              f"({', '.join(hosts) or 'none'}) in the data -- undefined. "
              f"Needs at least 2 hosts.")
        return pd.DataFrame()

    rows = []
    for h in hosts:
        train = df[(df["host"] != h) & (df["split"] == "train")]
        held = df[df["host"] == h]
        if train.empty or held.empty:
            rows.append({"held_out_host": h, "n_train": len(train),
                         "n_held": len(held), "MAPE_%": np.nan})
            continue
        fit = fit_model(train, use_output_term, relative)
        e = ape(fit.predict(held), held["avg_ts"].to_numpy())
        rows.append({"held_out_host": h, "n_train": len(train), "n_held": len(held),
                     "MAPE_%": e.mean(), "median_APE_%": float(np.median(e)),
                     "max_APE_%": e.max()})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# figure
# --------------------------------------------------------------------------

def fig_eta_variation(df: pd.DataFrame, out: Path) -> bool:
    """eta per model, grouped by quantisation -- 5.7 at a glance.

    If eta were a property of the format, every bar inside a group would be the
    same height. The figure exists because they are not.
    """
    d = df[["host", "quant", "model_name"]].copy()
    d["eta"] = df["eta"].to_numpy(float)
    per_model = (d.groupby(["quant", "model_name"])["eta"].median()
                 .reset_index().dropna(subset=["eta"]))
    if per_model.empty:
        return False

    quants = sorted(per_model["quant"].unique())
    models = sorted(per_model["model_name"].unique())
    colours = plt.get_cmap("tab20")(np.linspace(0, 1, max(len(models), 2)))
    cmap = dict(zip(models, colours))

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    widest = max(len(per_model[per_model["quant"] == q]) for q in quants)
    width = 0.82 / widest
    seen = set()
    for i, q in enumerate(quants):
        g = per_model[per_model["quant"] == q].sort_values("model_name")
        for j, (_, r) in enumerate(g.iterrows()):
            x = i + width * (j - (len(g) - 1) / 2)  # centre the group on its tick
            ax.bar(x, r["eta"], width * 0.92, color=cmap[r["model_name"]],
                   label=r["model_name"] if r["model_name"] not in seen else None)
            seen.add(r["model_name"])
        if len(g) > 1:
            lo, hi = g["eta"].min(), g["eta"].max()
            ax.annotate(f"{hi / lo:.2f}x", (i, hi), textcoords="offset points",
                        xytext=(0, 4), ha="center", fontsize=7.5)

    ax.set_xticks(range(len(quants)))
    ax.set_xticklabels(quants, rotation=30, ha="right")
    ax.set_ylim(0, per_model["eta"].max() * 1.45)  # headroom for the legend
    ax.set_ylabel(r"back-solved $\eta_d$  (fraction of calibrated roofline)")
    ax.set_title(r"$\eta$ is not a property of the quantisation format alone"
                 "\n(one bar per model; label = max/min inside the group)")
    ax.axhline(1.0, color="k", lw=.8, ls="--", alpha=.6)
    ax.legend(fontsize=6, ncol=3, loc="upper left", framealpha=.9)
    fig.savefig(out)
    plt.close(fig)
    return True


# --------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=RESULTS_DIR)
    ap.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    ap.add_argument("--figures", type=Path, default=FIGURES_DIR)
    ap.add_argument("--keep-all", action="store_true",
                    help="do not collapse repeated measurements of a cell to "
                         "the fastest observation")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args(argv)

    if args.selfcheck:
        _selfcheck()
        return 0

    dec, dis = load_rows(args.results, args.models_dir, collapse=not args.keep_all)
    if dec.empty:
        raise SystemExit("No decode rows with calibration. Nothing to fit.")

    if not dis.empty:
        print(f"!! {len(dis)} measurement cells have more than one observation; "
              f"worst disagreement {dis['ratio'].max():.2f}x, "
              f"median {dis['ratio'].median():.2f}x. Keeping the fastest of each.")
        print(dis.head(5)[["model_file", "n_depth", "slowest", "fastest", "ratio"]]
              .round(2).to_string(index=False))
        print("   (sweep.py's resume key cannot match a written row, so a "
              "re-run re-measures everything; the repeats above are not noise, "
              "they are two different machine states.)\n")

    missing = sorted(dec.loc[dec["vocab_size"].isna(), "model_file"].unique())
    if missing:
        print(f"no GGUF on disk for {len(missing)} measured model(s): "
              f"{', '.join(missing)} -- excluded from the two-term comparison\n")
    dec = dec.dropna(subset=["vocab_size", "flops"]).reset_index(drop=True)
    if dec.empty:
        raise SystemExit("No rows with both a GGUF and a compute calibration.")

    # ---- 1. per-row effective bandwidth ---------------------------------
    print("=== per-model effective bandwidth and eta (median over KV depths) ===")
    at0 = dec[dec["n_depth"] == 0].set_index(["host", "quant", "model_name"])["avg_ts"]
    per_model = (dec.groupby(["host", "quant", "model_name", "split"])
                 .agg(n=("avg_ts", "size"), vocab=("vocab_size", "first"),
                      d_model=("d_model", "first"), n_layers=("n_layers", "first"),
                      eff_bw_gb_s=("eff_bw_gb_s", "median"),
                      eta=("eta", "median"), eta_streamed=("eta_streamed", "median"))
                 .reset_index())
    per_model["tok_s_d0"] = [at0.get((h, q, m), np.nan) for h, q, m in
                             zip(per_model["host"], per_model["quant"],
                                 per_model["model_name"])]
    print(per_model.sort_values(["quant", "eta"]).round(3).to_string(index=False))

    print("\n=== does eta fall with output-projection load, as 5.7 predicts? ===")
    load = (dec.groupby(["host", "quant", "model_name"])[["out_load", "eta"]]
            .median().reset_index().sort_values(["quant", "out_load"]))
    print(load.round(4).to_string(index=False))
    ranks = output_load_check(dec)
    if ranks.empty:
        print("no (host, quant) group has 3+ models -- the rank test needs 3")
    else:
        print(ranks.round(3).to_string(index=False))

    print("\n=== cross-model spread of eta, one-term model ===")
    spread1 = eta_spread(dec, None, col="eta")
    print(spread1.round(3).to_string(index=False))
    print("\n=== same, with the untied-embedding correction applied ===")
    print(eta_spread(dec, None, col="eta_streamed").round(3).to_string(index=False))

    # ---- 2. one-term vs two-term ----------------------------------------
    train = dec[dec["split"] == "train"]
    if train.empty:
        print("\nWARNING: no TRAIN rows; fitting on everything.")
        train = dec

    f1 = fit_model(train, use_output_term=False)
    f2 = fit_model(train, use_output_term=True)

    print(f"\n=== fitted parameters (TRAIN only, n={len(train)}, "
          f"absolute time residuals) ===")
    both = pd.DataFrame([
        {"host": h, "quant": q, "eta_d_1term": v, "eta_d_2term": f2.eta_d[(h, q)]}
        for (h, q), v in sorted(f1.eta_d.items())])
    print(both.round(3).to_string(index=False))
    print(f"eta_o = {f2.eta_o:.4g}  "
          f"(with FLOPS = {dec['flops'].iloc[0] / 1e12:.2f} TFLOP/s; on a host "
          f"without torch_gpu that is a CPU fp32 number, so eta_o absorbs the "
          f"wrong compute scale and is not an efficiency)")
    out_share = (dec["out_flops"] / (f2.eta_o * dec["flops"]) / dec["t_meas"])
    print(f"output-projection term = {out_share.min() * 100:.1f}-"
          f"{out_share.max() * 100:.1f}% of per-token time "
          f"(median {out_share.median() * 100:.1f}%)")
    if out_share.max() > 1.0:
        print("   UNPHYSICAL: on some rows the fitted output-projection term "
              "alone exceeds the whole measured per-token time, which leaves "
              "the bandwidth term negative. The fit is absorbing something "
              "other than the mechanism it is named after.")

    print("\n=== cross-model spread of eta after the output-projection term ===")
    spread2 = eta_spread(dec, f2.eta_o)
    print(spread2.round(3).to_string(index=False))
    cmp = spread1.merge(spread2, on=["host", "quant"], suffixes=("_1term", "_2term"))
    multi = cmp[cmp["n_models_1term"] > 1]
    if multi.empty:
        print("no (host, quant) group holds more than one model -- the spread "
              "question cannot be answered on this data at all")
    else:
        print("\ngroups with >1 model (the only ones that carry information; "
              "n_models_2term < n_models_1term means the correction drove a "
              "model's bandwidth term negative):")
        print(multi[["host", "quant", "n_models_1term", "n_models_2term",
                     "max_over_min_1term", "max_over_min_2term"]]
              .round(3).to_string(index=False))

    # Both loss choices are reported because the brief's absolute-time residual
    # is dominated by the slowest cells, which is the opposite of where the
    # output projection is supposed to bite. A conclusion that survives only
    # one of the two is not a conclusion.
    print("\n=== prediction error ===")
    rows = []
    for loss, rel in (("absolute time", False), ("relative time", True)):
        a = fit_model(train, use_output_term=False, relative=rel)
        b = fit_model(train, use_output_term=True, relative=rel)
        for r in (error_rows("one-term (bandwidth only)", a, dec)
                  + error_rows("two-term (+ output projection)", b, dec)):
            rows.append({"loss": loss, **r})
    print(pd.DataFrame(rows).round(1).to_string(index=False))

    # ---- 3. leave one machine out ---------------------------------------
    print("\n=== leave-one-machine-out ===")
    lomo = leave_one_machine_out(dec, use_output_term=True)
    if not lomo.empty:
        print(lomo.round(1).to_string(index=False))

    # ---- 4. is any of this above the measurement floor? ------------------
    if not dis.empty and not multi.empty:
        # eta is linear in tok/s at fixed bytes, so a cell's tok/s disagreement
        # between sessions IS its eta disagreement. If that rivals the spread
        # between models, no architectural term can be identified from the data.
        print(f"\n=== is the cross-model spread even measurable? ===")
        print(f"same model, same cell, different session: eta varies by up to "
              f"{dis['ratio'].max():.2f}x ({len(dis)} cells re-measured)")
        print(f"different models, same quantisation:      eta varies by up to "
              f"{multi['max_over_min_1term'].max():.2f}x")
        if dis["ratio"].max() >= multi["max_over_min_1term"].max():
            print("The within-cell variation is as large as the between-model "
                  "variation. On this data 5.7 is not answerable: any eta term "
                  "fitted here is fitting measurement state, not architecture.")

    # ---- 4. figure -------------------------------------------------------
    args.figures.mkdir(parents=True, exist_ok=True)
    fig_path = args.figures / "fig6_eta_variation.png"
    if fig_eta_variation(dec, fig_path):
        print(f"\nfigure -> {fig_path}")

    n_models = dec["model_name"].nunique()
    print(f"\n{len(dec)} decode rows, {n_models} models, "
          f"{dec['host'].nunique()} host(s), "
          f"{dec[dec['is_moe']]['model_name'].nunique()} MoE")
    return 0


# --------------------------------------------------------------------------

def _synthetic(eta_d: dict, eta_o: float | None, bw=1e11, flops=1e12) -> pd.DataFrame:
    """Rows generated from the model itself, so the right answer is known.

    The constants are chosen so the output-projection term is worth a few
    percent of per-token time on the narrow-vocabulary model and ~17% on the
    wide one -- the regime 5.7 is arguing about. Make it negligible and the
    check passes vacuously.
    """
    specs = [  # (model, quant, bytes, vocab, d_model, split)
        ("small-narrow", "Q4", 2e9, 32_000, 2048, "train"),
        ("small-wide", "Q4", 2e9, 256_000, 2048, "train"),
        ("big-narrow", "Q4", 2e10, 32_000, 5120, "train"),
        ("small-narrow", "Q8", 4e9, 32_000, 2048, "train"),
        ("small-wide", "Q8", 4e9, 256_000, 2048, "test"),
    ]
    rows = []
    for name, quant, b, vocab, d, split in specs:
        for depth, kv in ((0, 0.0), (4096, 1e8)):
            t = (b + kv) / (eta_d[quant] * bw)
            if eta_o is not None:
                t += 2.0 * vocab * d / (eta_o * flops)
            rows.append({
                "host": "H1", "quant": quant, "model_name": name,
                "model_file": f"{name}-{quant}.gguf", "split": split,
                "n_depth": depth, "bytes_active": b + kv, "bw": bw, "flops": flops,
                "out_flops": 2.0 * vocab * d, "avg_ts": 1.0 / t,
            })
    df = pd.DataFrame(rows)
    df["t_meas"] = 1.0 / df["avg_ts"]
    return df


def _selfcheck() -> None:
    """Assert-based check of the fitting maths on data with a known answer."""
    true_d = {"Q4": 0.80, "Q8": 0.55}

    # 1. one-term data, one-term fit: exact recovery.
    d1 = _synthetic(true_d, eta_o=None)
    f1 = fit_model(d1[d1.split == "train"], use_output_term=False)
    for q, v in true_d.items():
        got = f1.eta_d[("H1", q)]
        assert abs(got - v) < 1e-6, (q, got, v)
    assert max(ape(f1.predict(d1), d1["avg_ts"].to_numpy())) < 1e-6

    # 2. two-term data, two-term fit: both parameters recovered, under either
    #    residual weighting (main reports both).
    true_o = 0.25
    d2 = _synthetic(true_d, eta_o=true_o)
    for rel in (False, True):
        f2 = fit_model(d2[d2.split == "train"], use_output_term=True, relative=rel)
        assert abs(f2.eta_o - true_o) / true_o < 1e-4, (rel, f2.eta_o)
        for q, v in true_d.items():
            assert abs(f2.eta_d[("H1", q)] - v) / v < 1e-4, (rel, q, f2.eta_d[("H1", q)])
        assert max(ape(f2.predict(d2), d2["avg_ts"].to_numpy())) < 1e-3

    # 3. the whole premise: on two-term data the naive back-solve makes eta
    #    look model-dependent, and removing the output term makes it constant.
    naive = eta_spread(d2, eta_o=None)
    corrected = eta_spread(d2, eta_o=true_o)
    q4_naive = naive[naive["quant"] == "Q4"]["max_over_min"].iloc[0]
    q4_corr = corrected[corrected["quant"] == "Q4"]["max_over_min"].iloc[0]
    assert q4_naive > 1.05, q4_naive
    assert abs(q4_corr - 1.0) < 1e-6, q4_corr

    # 4. ...and the converse, so a positive result cannot be an artefact of the
    #    extra parameter: on one-term data the correction has nothing to remove.
    f2_on_d1 = fit_model(d1[d1.split == "train"], use_output_term=True)
    assert eta_spread(d1, eta_o=None)["max_over_min"].max() < 1 + 1e-6
    assert max(ape(f2_on_d1.predict(d1), d1["avg_ts"].to_numpy())) < 1e-3

    # 5. unseen (host, quant) falls back rather than raising.
    assert abs(f1.eta_for("H2", "Q4") - true_d["Q4"]) < 1e-6
    assert f1.eta_for("H2", "NOPE") > 0

    # 6. one host must degrade gracefully, not crash.
    assert leave_one_machine_out(d2, use_output_term=True).empty
    two = pd.concat([d2, d2.assign(host="H2", avg_ts=d2["avg_ts"] * 0.5)])
    lomo = leave_one_machine_out(two, use_output_term=True)
    assert set(lomo["held_out_host"]) == {"H1", "H2"}, lomo
    assert lomo["MAPE_%"].notna().all(), lomo

    # 7. repeat collapse keeps the fastest observation of a cell.
    dup = pd.DataFrame({
        "host": ["H1"] * 3, "model_file": ["m.gguf"] * 3, "n_gpu_layers": [99] * 3,
        "n_depth": [0, 0, 4096], "avg_ts": [30.0, 90.0, 10.0]})
    kept, disagree = collapse_repeats(dup)
    assert sorted(kept["avg_ts"]) == [10.0, 90.0], kept["avg_ts"].tolist()
    assert len(disagree) == 1 and abs(disagree["ratio"].iloc[0] - 3.0) < 1e-9

    print("refine.py selfcheck ok")


if __name__ == "__main__":
    raise SystemExit(main())
