"""Generate the ICASSP companion results appendix from frozen result files.

This script is deliberately separate from the analysis pipeline. It treats
the regenerated CSV exports as the row-level source of truth and fails loudly
if host/model/depth grids, protocol fields, or aggregate summaries disagree.

Run from anywhere in the repository with::

    python paper/icassp2027/generate_supplement.py
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = ROOT / "results"

HOST_LABELS = {
    "lun-mac": "Apple M4 Max",
    "rtx5080": "RTX 5080",
}


def host_label(value: object) -> str:
    """Human-readable label while preserving unknown host identifiers."""
    value = str(value)
    return HOST_LABELS.get(value, value)


def tex(value: object) -> str:
    """Escape ordinary text for LaTeX table cells."""
    s = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in s)


def path_cell(value: object) -> str:
    """Render a filename or repository ID with safe punctuation and breaks."""
    return r"\nolinkurl{" + str(value).replace("%", r"\%") + "}"


def hash_cell(value: object) -> str:
    """Render a long hexadecimal identifier with explicit line-break points."""
    value = str(value)
    chunks = [value[i : i + 8] for i in range(0, len(value), 8)]
    return r"\texttt{" + r"\allowbreak{}".join(chunks) + "}"


def predictor_label(value: object) -> str:
    """Use fit terminology: calibration measurements cancel in B1/B2/P1/P2."""
    value = str(value)
    value = value.replace("uncalibrated", "unfitted")
    value = value.replace("calibrated", "fitted")
    return tex(value)


def nfmt(value: object) -> str:
    return f"{int(value):,}"


def ffmt(value: object, digits: int = 2) -> str:
    x = float(value)
    if not math.isfinite(x):
        return "--"
    return f"{x:.{digits}f}"


def bool_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unavailable"


def add_longtable(
    out: list[str],
    caption: str,
    label: str,
    colspec: str,
    headers: list[str],
    rows: list[list[str]],
    landscape: bool = True,
    lead: list[str] | None = None,
) -> None:
    if landscape:
        out.append(r"\begin{landscape}")
    if lead:
        out.extend(lead)
    out.extend(
        [
            r"\begingroup",
            r"\small",
            r"\setlength{\tabcolsep}{3.5pt}",
            r"\renewcommand{\arraystretch}{1.08}",
            rf"\begin{{longtable}}{{{colspec}}}",
            rf"\caption{{{caption}}}\label{{{label}}}\\",
            r"\toprule",
            " & ".join(headers) + r" \\",
            r"\midrule",
            r"\endfirsthead",
            rf"\caption[]{{{caption} (continued)}}\\",
            r"\toprule",
            " & ".join(headers) + r" \\",
            r"\midrule",
            r"\endhead",
            r"\midrule",
            rf"\multicolumn{{{len(headers)}}}{{r}}{{Continued on next page}}\\",
            r"\endfoot",
            r"\bottomrule",
            r"\endlastfoot",
        ]
    )
    out.extend(" & ".join(row) + r" \\" for row in rows)
    out.extend([r"\end{longtable}", r"\endgroup"])
    if landscape:
        out.append(r"\end{landscape}")


def group_predictions(
    frame: pd.DataFrame, bytes_col: str, prediction_bytes_col: str | None = None
) -> np.ndarray:
    train = frame[frame["split"] == "train"]
    eta = (
        train["avg_ts"] * train[bytes_col] / train["bw"]
    ).groupby([train["host"], train["quant"]]).median()
    host_median = eta.groupby(level=0).median()
    overall = float(eta.median())

    def lookup(host: str, quant: str) -> float:
        if (host, quant) in eta.index:
            return float(eta.loc[(host, quant)])
        if host in host_median.index:
            return float(host_median.loc[host])
        return overall

    denominator = frame[prediction_bytes_col or bytes_col].to_numpy(float)
    fitted = np.array([lookup(h, q) for h, q in zip(frame.host, frame["quant"])])
    return fitted * frame["bw"].to_numpy(float) / denominator


def ape(pred: np.ndarray, measured: np.ndarray) -> np.ndarray:
    return np.abs(pred - measured) / np.abs(measured) * 100.0


def make_summary_figure(dec: pd.DataFrame, prefill: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.65), constrained_layout=True)

    ladder = dec[
        dec["model_file"].str.startswith("Qwen3.8-27B-")
        & (dec["n_depth"] == 0)
    ].copy()
    order = {
        "IQ2_XXS": 0,
        "Q2_K": 1,
        "IQ3_XXS": 2,
        "Q3_K": 3,
        "Q4_K": 4,
        "Q5_K": 5,
        "Q6_K": 6,
        "Q8_0": 7,
    }
    ladder["order"] = ladder["quant"].map(order)
    ladder = ladder.dropna(subset=["order"]).sort_values(["host", "order"])
    ladder["effective_gb_s"] = (
        ladder["avg_ts"] * ladder["bytes_active"] / 1e9
    )
    quants = [
        q for q, _ in sorted(order.items(), key=lambda item: item[1])
        if q in set(ladder["quant"])
    ]
    xpos = {q: i for i, q in enumerate(quants)}
    colours = {"lun-mac": "#377eb8", "rtx5080": "#e6550d"}
    markers = {"lun-mac": "o", "rtx5080": "s"}
    for host, group in ladder.groupby("host", sort=True):
        group = group.sort_values("order")
        x = group["quant"].map(xpos).to_numpy(float)
        axes[0].plot(
            x,
            group["effective_gb_s"],
            color=colours.get(host, "#555555"),
            marker=markers.get(host, "D"),
            lw=1.3,
            ms=4.5,
            label=host_label(host),
        )
    axes[0].set_xticks(np.arange(len(quants)), quants, rotation=35, ha="right")
    axes[0].set_ylabel("effective streamed GB/s")
    axes[0].set_title("(a) Cross-system Qwen3.8-27B ladder", loc="left")
    axes[0].legend(loc="best", frameon=False)

    test = prefill[prefill["split"] == "test"]
    depths = sorted(int(x) for x in test["n_depth"].unique())
    bx = np.arange(len(depths))
    hosts = sorted(test["host"].unique())
    width = 0.72 / max(len(hosts), 1)
    for index, host in enumerate(hosts):
        group = test[test["host"] == host]
        values = [group.loc[group.n_depth == d, "ape_P2"].mean() for d in depths]
        offset = (index - (len(hosts) - 1) / 2) * width
        bars = axes[1].bar(
            bx + offset,
            values,
            width,
            color=colours.get(host, "#555555"),
            edgecolor="#222222",
            linewidth=0.6,
            hatch="" if host == "lun-mac" else "///",
            label=host_label(host),
        )
        # Put values inside the bars so the tallest depth-zero RTX label does
        # not collide with the legend in the compact two-panel rendering.
        axes[1].bar_label(
            bars, fmt="%.1f", padding=-12, fontsize=8, color="white"
        )
    axes[1].set_xticks(bx, [f"{d:,}" for d in depths])
    axes[1].set_xlabel("existing-prefix depth (tokens)")
    axes[1].set_ylabel("P2 held-out MAPE (%)")
    axes[1].set_title("(b) Prefill error is host dependent", loc="left")
    axes[1].legend(loc="upper left", frameon=False)

    fig.savefig(HERE / "supplement_summary.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    paths = {
        "measurements (lun-mac)": RESULTS / "measurements_lun-mac.csv",
        "measurements (rtx5080)": RESULTS / "measurements_rtx5080.csv",
        "decode predictions": RESULTS / "predictions.csv",
        "prefill predictions": RESULTS / "predictions_prefill.csv",
        "decode aggregate error table": RESULTS / "error_table.csv",
        "decode error table by host": RESULTS / "error_table_by_host.csv",
        "prefill aggregate error table": RESULTS / "error_table_prefill.csv",
        "prefill error table by host": RESULTS / "error_table_prefill_by_host.csv",
        "calibration (lun-mac)": RESULTS / "calibration_lun-mac.json",
        "calibration (rtx5080)": RESULTS / "calibration_rtx5080.json",
        "environment (lun-mac)": RESULTS / "env_lun-mac.json",
        "environment (rtx5080)": RESULTS / "env_rtx5080.json",
        "manifest": RESULTS / "model_manifest.json",
        "metadata": RESULTS / "model_metadata.json",
        "repeatability (lun-mac)": RESULTS / "repeatability_lun-mac.csv",
        "archived uncontrolled comparison": (
            RESULTS / "contaminated" / "measurements_UNCONTROLLED_lun-mac.csv.bak"
        ),
        "analysis implementation": ROOT / "llmperf" / "analyze.py",
        "refinement implementation": ROOT / "llmperf" / "refine.py",
        "metadata implementation": ROOT / "llmperf" / "common.py",
        "appendix generator": HERE / "generate_supplement.py",
    }
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        raise SystemExit("missing required input(s): " + ", ".join(missing))

    dec = pd.read_csv(paths["decode predictions"])
    pf = pd.read_csv(paths["prefill predictions"])
    err = pd.read_csv(paths["decode error table by host"])
    pf_err = pd.read_csv(paths["prefill error table by host"])
    measurements_by_host = {
        "lun-mac": pd.read_csv(paths["measurements (lun-mac)"]),
        "rtx5080": pd.read_csv(paths["measurements (rtx5080)"]),
    }
    repeat = pd.read_csv(paths["repeatability (lun-mac)"])
    calibrations = {
        "lun-mac": json.loads(
            paths["calibration (lun-mac)"].read_text(encoding="utf-8")
        ),
        "rtx5080": json.loads(
            paths["calibration (rtx5080)"].read_text(encoding="utf-8")
        ),
    }
    environments = {
        "lun-mac": json.loads(
            paths["environment (lun-mac)"].read_text(encoding="utf-8")
        ),
        "rtx5080": json.loads(
            paths["environment (rtx5080)"].read_text(encoding="utf-8")
        ),
    }
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    metadata_doc = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    metadata = metadata_doc["models"]

    dec = dec.copy()
    pf = pf.copy()
    for frame in (dec, pf):
        frame["n_depth"] = pd.to_numeric(frame["n_depth"])
        frame["avg_ts"] = pd.to_numeric(frame["avg_ts"])
        frame["stddev_ts"] = pd.to_numeric(frame["stddev_ts"], errors="coerce")
        frame["row_cv_pct"] = frame["stddev_ts"] / frame["avg_ts"] * 100.0

    hosts = sorted(dec["host"].unique())
    if set(hosts) != set(pf["host"].unique()):
        raise SystemExit("decode and prefill exports describe different hosts")
    if set(hosts) != set(calibrations) or set(hosts) != set(environments):
        raise SystemExit("measurement, calibration, and environment hosts disagree")

    key_columns = ["host", "model_file", "n_depth"]
    if dec.duplicated(key_columns).any() or pf.duplicated(key_columns).any():
        raise SystemExit("row exports contain duplicate host/model/depth keys")
    decode_keys = set(dec[key_columns].itertuples(index=False, name=None))
    prefill_keys = set(pf[key_columns].itertuples(index=False, name=None))
    if decode_keys != prefill_keys:
        raise SystemExit("decode and prefill exports describe different host/model grids")
    depth_sets = {
        (host, model): tuple(sorted(group["n_depth"].astype(int).unique()))
        for (host, model), group in dec.groupby(["host", "model_file"])
    }
    if not depth_sets or len(set(depth_sets.values())) != 1:
        raise SystemExit(f"non-rectangular depth grids: {depth_sets}")
    depths = next(iter(depth_sets.values()))

    if not dec["quality_protocol"].map(bool_value).all():
        raise SystemExit("decode export contains a protocol-incomplete row")
    if not pf["quality_protocol"].map(bool_value).all():
        raise SystemExit("prefill export contains a protocol-incomplete row")

    scored = sorted(dec["model_file"].unique(), key=str.casefold)
    scored_pairs = set(
        dec[["host", "model_file"]].drop_duplicates().itertuples(index=False, name=None)
    )
    for name in scored:
        if name not in metadata or name not in manifest:
            raise SystemExit(f"missing manifest/metadata entry for {name}")
    for host, name in scored_pairs:
        row_splits = set(dec.loc[(dec.host == host) & (dec.model_file == name), "split"])
        if row_splits != {manifest[name]["split"]}:
            raise SystemExit(f"split mismatch for {host}/{name}: {row_splits}")

    loaded_by_host: dict[str, set[str]] = {}
    failed_by_host: dict[str, list[str]] = {}
    probes_by_host: dict[str, list[str]] = {}
    for host in hosts:
        raw = measurements_by_host[host]
        ok = raw["error"].fillna("").astype(str).str.strip().eq("")
        attempted = set(raw["model_file"].dropna().astype(str))
        loaded_by_host[host] = set(raw.loc[ok, "model_file"].astype(str))
        failed_by_host[host] = sorted(
            attempted - loaded_by_host[host], key=str.casefold
        )
        probes_by_host[host] = list(
            (calibrations[host].get("llm_ref") or {}).get("probe_models") or []
        )
        scored_here = {name for h, name in scored_pairs if h == host}
        if not scored_here.issubset(loaded_by_host[host]):
            missing_scored = sorted(scored_here - loaded_by_host[host])
            raise SystemExit(f"scored rows absent from raw {host} data: {missing_scored}")

    identified = sorted(
        set(scored).union(
            *(set(names) for names in probes_by_host.values())
        ),
        key=str.casefold,
    )
    ids = {name: f"M{i:02d}" for i, name in enumerate(identified, 1)}
    dec["id"] = dec["model_file"].map(ids)
    pf["id"] = pf["model_file"].map(ids)

    # Reconstruct B0/B1 solely for disaggregated audit tables; B2 is read from
    # the authoritative row export.
    b0 = dec["bw"].to_numpy(float) / dec["bytes_total"].to_numpy(float)
    b1 = group_predictions(dec, "bytes_total")
    b2 = dec["pred_ours"].to_numpy(float)
    measured = dec["avg_ts"].to_numpy(float)
    dec["pred_B0"] = b0
    dec["pred_B1"] = b1
    dec["ape_B0"] = ape(b0, measured)
    dec["ape_B1"] = ape(b1, measured)

    # Uniform-KV counterfactual, with its own training-only coefficient fit.
    dec["kv_uniform"] = (
        2
        * dec["n_layers"].to_numpy(float)
        * dec["n_kv_heads"].to_numpy(float)
        * dec["head_dim"].to_numpy(float)
        * dec["n_depth"].to_numpy(float)
        * 2
    )
    dec["bytes_uniform"] = (
        dec["file_bytes"].to_numpy(float) * dec["active_frac"].to_numpy(float)
        + dec["kv_uniform"].to_numpy(float)
    )
    pred_uniform = group_predictions(dec, "bytes_uniform")
    dec["ape_uniform"] = ape(pred_uniform, measured)

    # Negative output-projection experiment, recomputed from the authoritative
    # decode export and frozen vocabulary/model widths.
    sys.path.insert(0, str(ROOT))
    from llmperf.analyze import (
        add_features,
        apply_eta,
        calibration_probe_models,
        fit_eta,
        load_measurements,
        select_primary_measurements,
    )
    from llmperf.refine import (error_rows, fit_model,
                                leave_one_machine_out,
                                leave_one_machine_out_b2)

    refine = dec.copy()
    refine["vocab_size"] = refine["model_file"].map(
        lambda name: metadata[name]["vocab_size"]
    )
    refine["d_model"] = refine["model_file"].map(
        lambda name: metadata[name]["d_model"]
    )
    refine["out_flops"] = 2.0 * refine["vocab_size"] * refine["d_model"]
    refine["t_meas"] = 1.0 / refine["avg_ts"]
    refine_train = refine[refine["split"] == "train"]
    transfer_b2 = leave_one_machine_out_b2(refine)
    transfer_b2_test = leave_one_machine_out_b2(
        refine, target_split="test")
    transfer_two = leave_one_machine_out(
        refine, use_output_term=True, relative=False)
    target_fitted_all = {
        host: float(group["ape_B2"].mean())
        for host, group in dec.groupby("host", sort=False)
    }
    projection_rows: list[dict] = []
    for loss, relative in (("absolute time", False), ("relative time", True)):
        one = fit_model(refine_train, use_output_term=False, relative=relative)
        two = fit_model(refine_train, use_output_term=True, relative=relative)
        for host in [*hosts, "all hosts"]:
            evaluated = refine if host == "all hosts" else refine[refine.host == host]
            for row in error_rows("one term", one, evaluated) + error_rows(
                "two terms", two, evaluated
            ):
                projection_rows.append({"host": host, "loss": loss, **row})
    projection = pd.DataFrame(projection_rows)

    # Retain a diagnostic comparison with the archived, pre-protocol pass. It
    # is not mixed into the scored cohort. The raw append-only CSV contains one
    # complete legacy decode grid, so this comparison can be reconstructed
    # without selecting on prediction residuals.
    protocol_fields = [
        "load_before", "load_after", "attempts", "kept_cv_pct", "max_cv_pct"
    ]
    archived = measurements_by_host["lun-mac"].copy()
    archived["quality_protocol"] = archived[protocol_fields].notna().all(axis=1)
    numeric = [
        "avg_ts", "stddev_ts", "file_bytes", "n_params", "n_active_params",
        "n_layers", "n_kv_heads", "head_dim", "n_depth", "n_prompt",
        "n_gen", "n_gpu_layers", "n_expert",
    ]
    for column in numeric:
        archived[column] = pd.to_numeric(archived[column], errors="coerce")
    archived = archived[
        archived["error"].fillna("").astype(str).str.strip().eq("")
        & ~archived["quality_protocol"]
        & archived["n_gpu_layers"].eq(99)
    ].copy()
    calibration = calibrations["lun-mac"]
    calibration_by_host = {"lun-mac": calibration}
    split_map = {name: entry["split"] for name, entry in manifest.items()}
    archived = add_features(
        archived, calibration_by_host, split_map, metadata, ROOT / "models",
        prefer_live=False,
    )
    probe_by_host = calibration_probe_models(calibration_by_host)
    archived = archived[
        np.array(
            [
            name not in probe_by_host.get(host, set())
            for host, name in zip(archived["host"], archived["model_file"])
            ],
            dtype=bool,
        )
        & archived["phase"].eq("decode")
    ].reset_index(drop=True)
    mac_decode = dec[dec.host == "lun-mac"]
    archived_keys = set(
        archived[["host", "model_file", "n_depth"]]
        .itertuples(index=False, name=None)
    )
    mac_keys = set(
        mac_decode[["host", "model_file", "n_depth"]]
        .itertuples(index=False, name=None)
    )
    if archived_keys != mac_keys:
        raise SystemExit(
            "unexpected archived decode cohort: "
            f"archived keys={len(archived_keys)}, current Mac keys={len(mac_keys)}"
        )
    archived_train = archived[archived["split"] == "train"]
    archived_prediction = apply_eta(
        archived,
        fit_eta(archived_train, "bytes_active"),
        "bytes_active",
    )
    archived["ape_B2"] = ape(
        archived_prediction, archived["avg_ts"].to_numpy(float)
    )
    archived["row_cv_pct"] = (
        archived["stddev_ts"] / archived["avg_ts"] * 100.0
    )

    # Account for the complete final protocol cohort, including the reference
    # probes intentionally omitted from fitting and scoring.  The scored CSVs
    # contain 198 rows; the paper's 216-row acquisition count also includes
    # these 18 selected probe observations.
    selected = select_primary_measurements(load_measurements(RESULTS))
    selected["phase"] = np.where(selected["n_gen"] > 0, "decode", "prefill")
    selected["row_cv_pct"] = (
        pd.to_numeric(selected["stddev_ts"], errors="coerce")
        / pd.to_numeric(selected["avg_ts"], errors="coerce")
        * 100.0
    )
    selected_probe_mask = np.array(
        [
            name in set(probes_by_host.get(host, []))
            for host, name in zip(selected["host"], selected["model_file"])
        ],
        dtype=bool,
    )
    probe_observations = selected[selected_probe_mask].copy()
    expected_selected = len(dec) + len(pf) + len(probe_observations)
    if len(selected) != expected_selected:
        raise SystemExit(
            "selected-cohort accounting mismatch: "
            f"selected={len(selected)}, scored={len(dec) + len(pf)}, "
            f"probes={len(probe_observations)}"
        )

    # Conservative sensitivity for the historical Mac probe exclusion.  The
    # final exports use device-copy bandwidth, so these Q8 rows no longer set
    # the B0 ceiling; keeping the pre-specified exclusion avoids changing the
    # cohort after seeing results.  Quantify what inclusion would have done.
    selected_featured = add_features(
        selected.copy(), calibrations, split_map, metadata, ROOT / "models",
        prefer_live=False,
    )
    mac_probe_augmented = selected_featured[
        (selected_featured["host"] == "lun-mac")
        & (selected_featured["phase"] == "decode")
        & selected_featured["bw"].notna()
    ].copy()
    mac_probe_prediction = apply_eta(
        mac_probe_augmented,
        fit_eta(mac_probe_augmented[mac_probe_augmented["split"] == "train"],
                "bytes_active"),
        "bytes_active",
    )
    mac_probe_augmented["ape_probe_inclusion"] = ape(
        mac_probe_prediction, mac_probe_augmented["avg_ts"].to_numpy(float)
    )
    mac_probe_train = mac_probe_augmented[mac_probe_augmented["split"] == "train"]
    mac_probe_test = mac_probe_augmented[mac_probe_augmented["split"] == "test"]
    mac_original_train = mac_probe_train[~mac_probe_train["is_calibration_probe"]]

    cohort_flow_rows: list[list[str]] = []
    for host in hosts:
        raw = measurements_by_host[host]
        success = raw["error"].fillna("").astype(str).str.strip().eq("")
        protocol = success & raw[protocol_fields].notna().all(axis=1)
        selected_host = selected[selected.host == host]
        probes_host = probe_observations[probe_observations.host == host]
        cohort_flow_rows.append(
            [
                tex(host_label(host)),
                str(len(raw)),
                str(int((~success).sum())),
                str(int((success & ~protocol).sum())),
                str(int(protocol.sum())),
                str(int(protocol.sum() - len(selected_host))),
                str(len(selected_host)),
                str(len(probes_host)),
                str(len(selected_host) - len(probes_host)),
            ]
        )

    probe_observations["_phase_order"] = probe_observations["phase"].map(
        {"decode": 0, "prefill": 1}
    )
    probe_observation_rows = [
        [
            tex(host_label(row.host)),
            ids[str(row.model_file)],
            tex(row.phase),
            nfmt(row.n_depth),
            ffmt(row.avg_ts, 3),
            ffmt(row.stddev_ts, 3),
            ffmt(row.row_cv_pct, 3),
            str(int(row.attempts)),
        ]
        for row in probe_observations.sort_values(
            ["host", "model_file", "_phase_order", "n_depth"]
        ).itertuples()
    ]

    # Validate the row exports against the aggregate authoritative tables.
    for host in hosts:
        for split in ("train", "test"):
            actual = float(
                dec.loc[
                    (dec.host == host) & (dec.split == split), "ape_ours"
                ].mean()
            )
            recorded = float(
                err.loc[
                    (err.host == host)
                    & err["model"].str.startswith("B2")
                    & (err["split"] == split),
                    "MAPE_%",
                ].iloc[0]
            )
            if abs(actual - recorded) > 0.011:
                raise SystemExit(
                    f"decode summary mismatch for {host}/{split}: "
                    f"{actual} vs {recorded}"
                )
            primary = pf[
                (pf.host == host) & (pf.split == split) & (pf.n_depth == 0)
            ]
            for short in ("P1", "P2"):
                actual = float(primary[f"ape_{short}"].mean())
                recorded = float(
                    pf_err.loc[
                        (pf_err.host == host)
                        & pf_err["model"].str.startswith(short)
                        & (pf_err["split"] == split),
                        "MAPE_%",
                    ].iloc[0]
                )
                if abs(actual - recorded) > 0.011:
                    raise SystemExit(
                        f"prefill summary mismatch for {host}/{short}/{split}: "
                        f"{actual} vs {recorded}"
                    )

    make_summary_figure(dec, pf)

    configuration_rows: list[list[str]] = []
    quality_rows: list[list[str]] = []
    for host in hosts:
        dhost = dec[dec.host == host]
        phost = pf[pf.host == host]
        configurations = dhost[["model_file", "split"]].drop_duplicates()
        retry_cells = int(
            dhost.loc[pd.to_numeric(dhost.attempts) > 1, "model_file"].nunique()
        )
        configuration_rows.append(
            [
                tex(host_label(host)),
                tex(", ".join(sorted(dhost.backend.astype(str).unique()))),
                str(int((configurations.split == "train").sum())),
                str(int((configurations.split == "test").sum())),
                str(len(dhost)),
                str(len(phost)),
            ]
        )
        threshold = float(pd.to_numeric(dhost.max_cv_pct).iloc[0])
        quality_rows.append(
            [
                tex(host_label(host)),
                str(retry_cells),
                ffmt(pd.to_numeric(dhost.kept_cv_pct).median(), 2),
                ffmt(pd.to_numeric(dhost.kept_cv_pct).max(), 2),
                ffmt(phost.row_cv_pct.median(), 2),
                ffmt(phost.row_cv_pct.max(), 2),
                f"{int((phost.row_cv_pct > threshold).sum())}/{len(phost)}",
            ]
        )

    failures = [
        f"{tex(host_label(host))}: " + ", ".join(path_cell(name) for name in names)
        for host, names in failed_by_host.items()
        if names
    ]
    probe_descriptions = [
        f"{tex(host_label(host))}: "
        + (", ".join(path_cell(name) for name in names) if names else "none")
        for host, names in probes_by_host.items()
    ]
    n_configurations = len(scored_pairs)
    n_train_configurations = int(
        dec[["host", "model_file", "split"]]
        .drop_duplicates()["split"]
        .eq("train")
        .sum()
    )
    n_test_configurations = n_configurations - n_train_configurations
    noisiest_prefill = pf.loc[pf.row_cv_pct.idxmax()]

    lines: list[str] = [
        "% Generated by generate_supplement.py; do not hand-edit.",
        r"\documentclass[10pt]{article}",
        r"\usepackage[letterpaper,margin=0.65in]{geometry}",
        r"\usepackage{newtxtext,newtxmath}",
        r"\usepackage{booktabs,longtable,array,pdflscape,graphicx,url}",
        r"\usepackage[hidelinks]{hyperref}",
        r"\newcolumntype{L}[1]{>{\raggedright\arraybackslash}p{#1}}",
        r"\newcolumntype{R}[1]{>{\raggedleft\arraybackslash}p{#1}}",
        r"\setlength{\parindent}{0pt}",
        r"\setlength{\parskip}{4pt}",
        r"\begin{document}",
        r"\begin{center}",
        r"{\Large\bfseries Companion Results Appendix: GGUF Throughput Prediction\par}",
        r"\vspace{4pt}",
        r"{\normalsize Protocol-complete two-host evidence for the ICASSP manuscript\par}",
        r"\end{center}",
        r"\textbf{Status.} This is a locally generated companion appendix. It does not claim publication, archival acceptance, or independent replication.",
        r"\section{Scope and cohort}",
        f"The source manifest contains {len(manifest)} GGUF files. Across two hosts, "
        f"the scored data contain {len(scored)} unique files and {n_configurations} "
        f"host--file configurations ({n_train_configurations} training and "
        f"{n_test_configurations} held out). Each configuration has one decode and "
        f"one prefill observation at each of the {len(depths)} measured context depths, "
        f"for {len(dec)} decode and {len(pf)} prefill rows. The final protocol selector "
        f"retains {len(selected)} observations from {len(identified)} unique files in "
        f"total: {len(dec) + len(pf)} scored rows "
        f"and {len(probe_observations)} reference-probe rows excluded from prediction "
        "fits and scores. Probe exclusion is host-specific, so a file used as a probe "
        "on one host can be scored on the other.",
        "All exported rows carry the declared protocol fields. The retry decision, however, "
        r"uses the worst within-run \emph{decode} CV in an invocation. Prefill rows inherit "
        "that invocation's protocol metadata but are not gated on their own CV. This "
        f"distinction matters on {host_label(noisiest_prefill.host)}, where the noisiest "
        f"prefill row has {noisiest_prefill.row_cv_pct:.2f}\\% CV even though the "
        f"associated decode-gate CV is {float(noisiest_prefill.kept_cv_pct):.2f}\\%.",
        r"\begin{center}",
        r"\small",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Host & Backend & Train cfg. & Test cfg. & Scored decode & Scored prefill\\",
        r"\midrule",
        *(" & ".join(row) + r" \\" for row in configuration_rows),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{center}",
        r"\begin{center}",
        r"\small",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Host & Retried cells & Gate CV med. & Gate CV max & PP CV med. & PP CV max & PP $>3\%$\\",
        r"\midrule",
        *(" & ".join(row) + r" \\" for row in quality_rows),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{center}",
        "Attempted configurations with no successful row are "
        + ("; ".join(failures) if failures else "none")
        + ". On Apple M4 Max, two raw attempts of the one gpt-oss-120B "
        r"configuration report \texttt{failed to decode prompt batch, res=-3}; "
        "these are prompt-batch failures, not demonstrated model-load or "
        "out-of-memory failures. "
        "Host-specific calibration probes are "
        + "; ".join(probe_descriptions)
        + ".",
        r"\subsection{Cohort accounting}",
        "Raw records, superseded legacy rows, repeated protocol attempts, selected "
        "observations, and scoring exclusions are separated below. A protocol duplicate "
        "is a successful protocol row removed by the atomic host--model--phase--depth "
        "selector; it is not an additional scored observation.",
        r"\begin{center}",
        r"\small",
        r"\setlength{\tabcolsep}{2.8pt}",
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"Host & Raw & Failed & Legacy ok & Protocol ok & Dup. removed & Selected & Probes & Scored\\",
        r"\midrule",
        *(" & ".join(row) + r" \\" for row in cohort_flow_rows),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{center}",
    ]

    add_longtable(
        lines,
        "Selected protocol-complete reference-probe observations excluded from prediction fitting and scoring. Measured throughput and SD are in tokens/s; IDs resolve through Table~\\ref{tab:model-id}.",
        "tab:probe-observations",
        r"@{}lllrrrrr@{}",
        ["Host", "ID", "Phase", "Depth", "Measured", "SD", r"CV \%", "Attempts"],
        probe_observation_rows,
        landscape=False,
        lead=[
            r"\subsection{Excluded reference-probe observations}",
            "These rows complete the 216-observation selected cohort and make the "
            "descriptive matched-host and quantization comparisons reproducible. They "
            "are reported only as measurements, never as prediction errors.",
        ],
    )

    lines.extend(
        [
            r"\paragraph{Probe-exclusion sensitivity.}",
            "The three Mac Q8 files defined the historical LLM-reference "
            "diagnostic. Their exclusion was fixed before the final device-copy "
            "analysis and is retained to avoid a post-hoc cohort change. If they "
            "are included and B2 is refit, MAPE is "
            f"{mac_probe_train.ape_probe_inclusion.mean():.2f}\\% over "
            f"{len(mac_probe_train)} training rows and "
            f"{mac_probe_test.ape_probe_inclusion.mean():.2f}\\% over "
            f"{len(mac_probe_test)} held-out rows; the latter is unchanged because "
            "all probes are Q8 training files while the test files are Q4. On the "
            f"original {len(mac_original_train)} non-probe training rows under "
            f"this refit, MAPE is {mac_original_train.ape_probe_inclusion.mean():.2f}\\%.",
        ]
    )

    lines.extend(
        [
        r"\section{Aggregate prediction results}",
        "Results are separated by host because fitted efficiency factors are host-specific; "
        "a pooled row would weight the unequal host cohorts and is not a cross-system score.",
        r"\begin{center}",
        r"\small",
        r"\begin{tabular}{lllrrrr}",
        r"\toprule",
        r"Host & Predictor & Split & $n$ & MAPE (\%) & Median (\%) & P90 / max (\%)\\",
        r"\midrule",
        ]
    )
    for _, row in err.iterrows():
        lines.append(
            f"{tex(host_label(row['host']))} & {predictor_label(row['model'])} & "
            f"{tex(row['split'])} & {int(row['n'])} & "
            f"{ffmt(row['MAPE_%'])} & {ffmt(row['median_APE_%'])} & "
            f"{ffmt(row['p90_APE_%'])} / {ffmt(row['max_APE_%'])} \\\\"
        )
    lines.extend(
        [
            r"\midrule",
        ]
    )
    for _, row in pf_err.iterrows():
        lines.append(
            f"{tex(host_label(row['host']))} & {predictor_label(row['model'])} & "
            f"{tex(row['split'])} & {int(row['n'])} & "
            f"{ffmt(row['MAPE_%'])} & {ffmt(row['median_APE_%'])} & "
            f"{ffmt(row['p90_APE_%'])} / {ffmt(row['max_APE_%'])} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{center}",
            "Decode summaries use all three context depths. Prefill headline rows use "
            "only depth 0, because the stated prefill model has no existing-prefix "
            "term; deeper rows are scope diagnostics below.",
        ]
    )

    transfer_rows: list[list[str]] = []
    for variant, frame in (("B2: all target rows", transfer_b2),
                           ("B2: target test rows", transfer_b2_test),
                           ("rejected two-term", transfer_two)):
        for _, row in frame.iterrows():
            transfer_rows.append(
                [
                    tex(variant),
                    tex(host_label(row.held_out_host)),
                    str(int(row.n_train)),
                    str(int(row.n_held)),
                    ffmt(row["MAPE_%"]),
                    ffmt(row["median_APE_%"]),
                    ffmt(row["max_APE_%"]),
                ]
            )
    lines.extend(
        [
            r"\subsection{Leave-one-host-out transfer}",
            "B2 transfers the source host's median per-quantization coefficients "
            "and substitutes the untouched target host's bandwidth. The target "
            "set contains all eligible rows on that host; because most files also "
            "occur on the source host, this isolates host/runtime transfer rather than "
            r"a simultaneous model-and-hardware holdout. Q4\_K\_M is shared; the Mac "
            r"Q4\_K test file uses the source-wide median fallback. The "
            "second variant is the rejected absolute-time output-projection extension.",
            r"\begin{center}",
            r"\small",
            r"\begin{tabular}{llrrrrr}",
            r"\toprule",
            r"Variant & Target host & Source $n$ & Target $n$ & MAPE & Median & Max\\",
            r"\midrule",
            *(" & ".join(row) + r" \\" for row in transfer_rows),
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{center}",
            "For context, target-fitted B2 gives all-row MAPE of "
            f"{target_fitted_all['lun-mac']:.2f}\\% on Mac and "
            f"{target_fitted_all['rtx5080']:.2f}\\% on RTX. Unlike the "
            "untouched-target transfer rows, these comparators mix training "
            "rows used to fit the target coefficients with held-out rows.",
        ]
    )

    architecture_rows: list[list[str]] = []
    for host in hosts:
        for split in ("train", "test"):
            for moe, label in ((False, "dense"), (True, "MoE")):
                mask = (
                    (dec.host == host)
                    & (dec.split == split)
                    & (dec.is_moe.map(bool_value) == moe)
                )
                if not mask.any():
                    continue
                architecture_rows.append(
                    [
                        tex(host_label(host)),
                        tex(split),
                        tex(label),
                        str(int(mask.sum())),
                        ffmt(dec.loc[mask, "ape_B0"].mean()),
                        ffmt(dec.loc[mask, "ape_B1"].mean()),
                        ffmt(dec.loc[mask, "ape_ours"].mean()),
                    ]
                )
    lines.extend(
        [
            r"\begin{center}",
            r"\small",
            r"\begin{tabular}{lllrrrr}",
            r"\toprule",
            r"Host & Split & Architecture & $n$ & B0 & B1 & B2\\",
            r"\midrule",
            *(" & ".join(row) + r" \\" for row in architecture_rows),
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{center}",
        ]
    )

    uniform_rows: list[list[str]] = []
    for host in hosts:
        for split in ("train", "test"):
            m = (dec.host == host) & (dec.split == split)
            uniform_rows.append(
                [
                    tex(host_label(host)),
                    tex(split),
                    str(int(m.sum())),
                    ffmt(dec.loc[m, "ape_uniform"].mean()),
                    ffmt(dec.loc[m, "ape_ours"].mean()),
                ]
            )
    lines.extend(
        [
            r"\subsection{Per-layer KV correction}",
            "With each variant refit on training rows, replacing a uniform-global "
            "KV denominator by the frozen per-layer metadata produces:",
            r"\begin{center}",
            r"\small",
            r"\begin{tabular}{llrrr}",
            r"\toprule",
            r"Host & Split & $n$ & Uniform KV MAPE (\%) & Per-layer KV MAPE (\%)\\",
            r"\midrule",
            *(" & ".join(row) + r" \\" for row in uniform_rows),
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{center}",
        ]
    )

    depth_zero_rate = dec.loc[
        dec["n_depth"] == min(depths), ["host", "model_file", "avg_ts"]
    ].set_index(["host", "model_file"])["avg_ts"]
    dec["rate_vs_empty"] = [
        float(row.avg_ts) / float(depth_zero_rate.loc[(row.host, row.model_file)])
        for row in dec.itertuples()
    ]
    monotonic = all(
        np.all(np.diff(g.sort_values("n_depth")["avg_ts"].to_numpy(float)) <= 0)
        for _, g in dec.groupby(["host", "model_file"])
    )
    context_rows: list[list[str]] = []
    for host in hosts:
        for depth in depths:
            g = dec[(dec.host == host) & (dec["n_depth"] == depth)]
            context_rows.append(
                [
                    tex(host_label(host)),
                    nfmt(depth),
                    str(int((g["split"] == "train").sum())),
                    str(int((g["split"] == "test").sum())),
                    ffmt(g.loc[g["split"] == "train", "ape_ours"].mean()),
                    ffmt(g.loc[g["split"] == "test", "ape_ours"].mean()),
                    ffmt(g["rate_vs_empty"].median(), 3),
                ]
            )
    lines.extend(
        [
            r"\subsection{Decode context behavior}",
            (
                f"All {n_configurations} scored host--file configurations slow "
                f"monotonically from {min(depths):,} through {max(depths):,} tokens. "
                if monotonic else ""
            )
            + "The normalized rate is each model's throughput divided by its own "
            "depth-0 throughput.",
            r"\begin{center}",
            r"\small",
            r"\begin{tabular}{lrrrrrr}",
            r"\toprule",
            r"Host & Depth & Train $n$ & Test $n$ & Train B2 MAPE & Test B2 MAPE & Median normalized rate\\",
            r"\midrule",
            *(" & ".join(row) + r" \\" for row in context_rows),
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{center}",
        ]
    )

    scope_rows: list[list[str]] = []
    for host in hosts:
        for split in ("train", "test"):
            for depth in depths:
                g = pf[
                    (pf.host == host)
                    & (pf.split == split)
                    & (pf.n_depth == depth)
                ]
                scope_rows.append(
                    [
                        tex(host_label(host)),
                        tex(split),
                        nfmt(depth),
                        str(len(g)),
                        ffmt(g.ape_P1.mean()),
                        ffmt(g.ape_P2.mean()),
                        ffmt(g.ape_P2.median()),
                        ffmt(g.ape_P2.max()),
                    ]
                )
    lines.extend(
        [
            r"\subsection{Prefill scope diagnostic}",
            r"Depth 0 is the primary prefill fit and score. The same fitted coefficients are applied unchanged at larger existing-prefix depths. These errors are prediction errors, distinct from within-run prefill variability; the RTX held-out result is poor even before that distinction is considered.",
            r"\begin{center}",
            r"\small",
            r"\begin{tabular}{llrrrrrr}",
            r"\toprule",
            r"Host & Split & Depth & $n$ & P1 MAPE & P2 MAPE & P2 median & P2 max\\",
            r"\midrule",
            *(" & ".join(row) + r" \\" for row in scope_rows),
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{center}",
            r"\begin{figure}[t]",
            r"\centering",
            r"\includegraphics[width=\textwidth]{supplement_summary.pdf}",
            r"\caption{Host-separated diagnostics. Left: effective streamed bandwidth for the scored Qwen3.8-27B quantizations at zero prefix; host coverage differs, and IQ2 has 64 layers/26.90B parameters versus 65 layers/27.32B for the other shown files, so the set is a family comparison rather than a strictly identical-network quantization ladder. Right: P2 held-out error when depth-0 coefficients are applied at each existing-prefix depth. The RTX result is not pooled with the Apple result.}",
            r"\end{figure}",
        ]
    )

    identity_rows: list[list[str]] = []
    geometry_rows: list[list[str]] = []
    for name in identified:
        m = metadata[name]
        man = manifest[name]
        split = (str(dec.loc[dec.model_file == name, "split"].iloc[0])
                 if name in scored else "probe only")
        measured_hosts = ", ".join(
            "M4" if host == "lun-mac" else "RTX"
            for host in hosts
            if ((dec.host == host) & (dec.model_file == name)).any()
        ) or "--"
        identity_rows.append(
            [
                ids[name],
                tex(split),
                tex(measured_hosts),
                path_cell(name),
                path_cell(man["repo_id"]),
                tex(m["arch"]),
                tex(m["quant"]),
                nfmt(m["file_bytes"]),
            ]
        )
        active_pct = 100.0 * m["n_active_params"] / m["n_params"]
        expert = f"{m['n_expert_used']}/{m['n_expert']}" if m["n_expert"] else "--"
        kv = m["kv_bytes_by_depth"]
        geometry_rows.append(
            [
                ids[name],
                nfmt(m["n_params"]),
                nfmt(m["n_active_params"]),
                ffmt(active_pct, 1),
                nfmt(m["n_layers"]),
                nfmt(m["d_model"]),
                nfmt(m["vocab_size"]),
                nfmt(m["n_kv_heads"]),
                nfmt(m["head_dim"]),
                expert,
                nfmt(kv["0"]),
                nfmt(kv["4096"]),
                nfmt(kv["16384"]),
            ]
        )

    add_longtable(
        lines,
        "Selected model identity, source, and file size. Sizes are exact bytes; host codes are M4 (Apple M4 Max) and RTX (RTX 5080), and ``probe only'' denotes a file excluded from every prediction fit and score.",
        "tab:model-id",
        r"@{}l l L{0.75in} L{2.7in} L{2.45in} L{0.9in} l r@{}",
        [
            "ID", "Split", "Scored host(s)", "GGUF filename", "Repository",
            "Architecture", "Quant.", "Bytes",
        ],
        identity_rows,
        lead=[
            r"\section{Selected model metadata}",
            "Tables~\\ref{tab:model-id} and~\\ref{tab:model-geometry} jointly "
            f"reproduce every frozen metadata field used for the {len(identified)} "
            f"unique selected files ({len(scored)} scored and "
            f"{len(identified) - len(scored)} probe-only). Scored host coverage is "
            "explicit because the two "
            "measurement cohorts are not identical.",
        ],
    )
    add_longtable(
        lines,
        "Selected model geometry. KV columns are exact bytes at the measured depths; $k/E$ is routed/total experts.",
        "tab:model-geometry",
        r"@{}l r r r r r r r r r r r r@{}",
        [
            "ID",
            "Total P",
            "Active P",
            r"Active \%",
            "Layers",
            "Width",
            "Vocab.",
            "KV heads",
            "Head dim.",
            "Used/all",
            "KV@0",
            "KV@4096",
            "KV@16384",
        ],
        geometry_rows,
    )

    decode_rows: list[list[str]] = []
    split_order = {"train": 0, "test": 1}
    dec_sorted = dec.assign(_split=dec.split.map(split_order)).sort_values(
        ["host", "_split", "id", "n_depth"]
    )
    for _, row in dec_sorted.iterrows():
        decode_rows.append(
            [
                tex(host_label(row.host)),
                row.id,
                tex(row.split),
                nfmt(row.n_depth),
                ffmt(row.avg_ts),
                ffmt(row.pred_ours),
                ffmt(row.ape_ours),
                ffmt(row.row_cv_pct),
                ffmt(row.load_before),
                str(int(float(row.attempts))),
            ]
        )
    add_longtable(
        lines,
        "Protocol-complete decode measurements and B2 predictions. The retry gate uses the worst decode CV in each host--file invocation. Pre-load is the macOS one-minute POSIX load average or, on Windows, busy-logical-CPU equivalents; it is meaningful within a host but not comparable across hosts.",
        "tab:decode-rows",
        r"@{}l l l r r r r r r r@{}",
        ["Host", "ID", "Split", "Depth", "Measured", "B2", r"APE \%", r"Row CV \%", "Load", "Attempts"],
        decode_rows,
        lead=[
            r"\section{All scored decode observations}",
            f"These are the {len(dec)} rows exported by "
            + path_cell("results/predictions.csv")
            + ". Throughput is tokens/s; APE is absolute percentage error. "
            "Rows are ordered by host, split, model ID, and depth.",
        ],
    )

    prefill_rows: list[list[str]] = []
    pf_sorted = pf.assign(_split=pf.split.map(split_order)).sort_values(
        ["host", "_split", "id", "n_depth"]
    )
    for _, row in pf_sorted.iterrows():
        scope = r"\textbf{primary}" if int(row.n_depth) == 0 else "diagnostic"
        prefill_rows.append(
            [
                tex(host_label(row.host)),
                row.id,
                tex(row.split),
                nfmt(row.n_depth),
                scope,
                ffmt(row.avg_ts),
                ffmt(row.pred_P1),
                ffmt(row.ape_P1),
                ffmt(row.pred_P2),
                ffmt(row.ape_P2),
                ffmt(row.row_cv_pct),
            ]
        )
    add_longtable(
        lines,
        "Protocol-complete prefill measurements and predictions. The retry decision was decode-gated, not prefill-gated; depth-0 rows are the primary evaluation and larger depths are scope diagnostics.",
        "tab:prefill-rows",
        r"@{}l l l r l r r r r r r@{}",
        ["Host", "ID", "Split", "Depth", "Scope", "Measured", "P1", "P1 APE", "P2", "P2 APE", r"Row CV \%"],
        prefill_rows,
        lead=[
            r"\section{All scored prefill observations}",
            f"These are the {len(pf)} rows exported by "
            + path_cell("results/predictions_prefill.csv")
            + ". P1 has one host coefficient; P2 has one host-by-quantization "
            "coefficient. Both are fit only at depth 0. The row CV column is "
            "reported rather than used as an inclusion filter.",
        ],
    )

    repeat_mean = float(repeat.decode_ts.mean())
    repeat_cv = float(repeat.decode_ts.std(ddof=1) / repeat_mean * 100)
    repeat_prefill_mean = float(repeat.prefill_ts.mean())
    repeat_prefill_cv = float(
        repeat.prefill_ts.std(ddof=1) / repeat_prefill_mean * 100
    )
    repeat_decode_span = float(
        (repeat.decode_ts.max() - repeat.decode_ts.min()) / repeat_mean * 100
    )
    repeat_prefill_span = float(
        (repeat.prefill_ts.max() - repeat.prefill_ts.min())
        / repeat_prefill_mean
        * 100
    )
    repeat_decode_trend = float(
        np.polyfit(repeat.run_index, repeat.decode_ts, 1)[0]
        / repeat_mean
        * 100
    )
    current_mac = dec[dec.host == "lun-mac"]
    current_test = current_mac[current_mac["split"] == "test"]
    archived_test = archived[archived["split"] == "test"]
    protocol_comparison = [
        [
            "Archived pre-protocol",
            str(len(archived)),
            str(len(archived_test)),
            ffmt(archived_test["ape_B2"].mean()),
            ffmt(archived_test["ape_B2"].median()),
            ffmt(np.percentile(archived_test["ape_B2"], 90)),
            ffmt(archived_test["ape_B2"].max()),
            ffmt(archived["row_cv_pct"].median()),
            ffmt(archived["row_cv_pct"].max()),
        ],
        [
            "Final Mac protocol-complete",
            str(len(current_mac)),
            str(len(current_test)),
            ffmt(current_test["ape_ours"].mean()),
            ffmt(current_test["ape_ours"].median()),
            ffmt(np.percentile(current_test["ape_ours"], 90)),
            ffmt(current_test["ape_ours"].max()),
            ffmt(current_mac["row_cv_pct"].median()),
            ffmt(current_mac["row_cv_pct"].max()),
        ],
    ]
    repeat_rows = [
        [
            str(int(row.run_index)),
            tex(row.timestamp),
            ffmt(row.settle_s, 0),
            ffmt(row.prefill_ts, 3),
            ffmt(row.decode_ts, 3),
            ffmt(row.decode_sd_within, 3),
        ]
        for _, row in repeat.sort_values("run_index").iterrows()
    ]
    lines.extend(
        [
            r"\section{Repeatability and calibration}",
            "A separate repeatability series exists only for Apple M4 Max; RTX 5080 "
            "variability is represented by the five within-invocation repetitions in "
            "the row tables, not by an equivalent across-run series. "
            f"The six Mac runs give mean decode throughput {repeat_mean:.3f} tokens/s "
            f"and sample between-run CV {repeat_cv:.3f}\\%; prefill is "
            f"{repeat_prefill_mean:.3f} tokens/s with CV {repeat_prefill_cv:.3f}\\%. "
            f"The max-minus-min spans are {repeat_decode_span:.2f}\\% and "
            f"{repeat_prefill_span:.2f}\\% of their respective means. Decode has "
            f"a descriptive linear trend of {repeat_decode_trend:.2f}\\% per run. "
            "These are single-cell diagnostics, not general detection thresholds.",
            r"\begin{center}",
            r"\small",
            r"\begin{tabular}{rlrrrr}",
            r"\toprule",
            r"Run & Timestamp & Settle (s) & Prefill & Decode & Decode within-run SD\\",
            r"\midrule",
            *(" & ".join(row) + r" \\" for row in repeat_rows),
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{center}",
            "The append-only Mac measurement file also preserves a complete "
            "pre-protocol decode grid. It is excluded from every fit and score "
            "above. The comparison is Mac-only and is not pooled with RTX data.",
            r"\begin{center}",
            r"\small",
            r"\begin{tabular}{lrrrrrrrr}",
            r"\toprule",
            r"Pass & All $n$ & Test $n$ & Test MAPE & Median & P90 & Max & All-row CV med. & All-row CV max\\",
            r"\midrule",
            *(" & ".join(row) + r" \\" for row in protocol_comparison),
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{center}",
        ]
    )

    uncontrolled = pd.read_csv(paths["archived uncontrolled comparison"])
    u = uncontrolled[
        (uncontrolled.model_file == "Qwen3.5-4B-Q4_K_M.gguf")
        & (pd.to_numeric(uncontrolled.n_depth) == 0)
    ].copy()
    u["phase"] = np.where(pd.to_numeric(u.n_gen) > 0, "decode", "prefill")
    contamination_rows: list[list[str]] = []
    for phase in ("prefill", "decode"):
        vals = u.loc[u.phase == phase, "avg_ts"].to_numpy(float)
        high, low = float(vals.max()), float(vals.min())
        contamination_rows.append(
            [tex(phase), ffmt(high, 3), ffmt(low, 3), ffmt((high - low) / high * 100, 2)]
        )
    lines.extend(
        [
            "For comparison, the retained historical uncontrolled file contains two "
            r"depth-0 observations of the same Qwen3.5-4B Q4\_K\_M cell:",
            r"\begin{center}",
            r"\small",
            r"\begin{tabular}{lrrr}",
            r"\toprule",
            r"Phase & Higher throughput & Lower throughput & Decrease (\%)\\",
            r"\midrule",
            *(" & ".join(row) + r" \\" for row in contamination_rows),
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{center}",
        ]
    )

    runtime_rows: list[list[str]] = []
    calibration_rows: list[list[str]] = []
    probe_rows: list[list[str]] = []
    for host in hosts:
        calibration = calibrations[host]
        environment = environments[host]
        gpu = calibration["torch_gpu"]
        ref = calibration.get("llm_ref") or {}
        protocol = environment.get("protocol") or {}
        runtime_rows.append(
            [
                tex(host_label(host)),
                tex(environment["platform"]),
                tex(gpu["name"]),
                str(environment["threads"]),
                str(protocol.get("repetitions", "--")),
                ffmt(protocol.get("settle_seconds", math.nan), 0),
                tex(", ".join(str(int(x)) for x in sorted(
                    pd.to_numeric(dec.loc[dec.host == host, "n_gpu_layers"]).unique()
                ))),
            ]
        )
        calibration_rows.extend(
            [
                [
                    tex(host_label(host)),
                    "CPU STREAM-style triad",
                    f"{calibration['cpu_triad']['gb_s']:.3f} GB/s",
                    f"{calibration['cpu_triad']['reps']} repetitions",
                ],
                [
                    tex(host_label(host)),
                    "CPU FP32 matrix multiply",
                    f"{calibration['cpu_matmul']['gflops'] / 1000:.3f} TFLOP/s",
                    f"{calibration['cpu_matmul']['reps']} repetitions",
                ],
                [
                    tex(host_label(host)),
                    f"{tex(str(gpu['backend']).upper())} device copy",
                    f"{gpu['copy_gb_s']:.3f} GB/s",
                    tex(gpu["name"]),
                ],
                [
                    tex(host_label(host)),
                    f"{tex(str(gpu['backend']).upper())} FP16 matrix multiply",
                    f"{gpu['fp16_tflops']:.3f} TFLOP/s",
                    tex(gpu["name"]),
                ],
            ]
        )
        if ref.get("available"):
            calibration_rows.append(
                [
                    tex(host_label(host)),
                    "LLM reference ceiling",
                    f"{ref['eff_bw_gb_s']:.3f} GB/s",
                    path_cell(ref["anchor_model"]),
                ]
            )
            for row in ref.get("probes", []):
                probe_rows.append(
                    [
                        tex(host_label(host)),
                        path_cell(row["model"]),
                        ffmt(row["file_gb"], 2),
                        ffmt(row["decode_tok_s"], 3),
                        ffmt(row["stddev_tok_s"], 3),
                        ffmt(row["stddev_tok_s"] / row["decode_tok_s"] * 100, 2),
                        ffmt(row["eff_bw_gb_s"], 3),
                    ]
                )
    lines.extend(
        [
            r"\subsection{Host settings and calibration}",
            r"\begin{center}",
            r"\small",
            r"\begin{tabular}{lllrrrr}",
            r"\toprule",
            r"Host & Platform & Accelerator & Threads & Reps & Settle (s) & Requested ngl\\",
            r"\midrule",
            *(" & ".join(row) + r" \\" for row in runtime_rows),
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{center}",
            r"The requested \texttt{n\_gpu\_layers=99} setting asks the runtime to "
            "offload as many layers as it can; it is not direct telemetry proving "
            "that every tensor and KV allocation remained resident on the accelerator. "
            "No device-memory trace was recorded, so the appendix does not label these "
            "observations as confirmed fully resident.",
            r"\begin{center}",
            r"\small",
            r"\begin{tabular}{llrl}",
            r"\toprule",
            r"Host & Calibration & Value & Detail\\",
            r"\midrule",
            *(" & ".join(row) + r" \\" for row in calibration_rows),
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{center}",
            "For the fitted B1/B2 and P1/P2 models, the measured bandwidth or "
            "FLOP/s scale cancels algebraically after fitting on this same host. "
            "It sets the scale of B0 and supports roofline interpretation; it does "
            "not by itself produce the reported fitted-model accuracy. The exported "
            "row-level bandwidth field uses the device-copy result on both hosts. "
            "The Mac LLM reference is retained as a diagnostic and is not the B0 "
            "bandwidth value used in the current exports.",
        ]
    )
    if probe_rows:
        lines.extend(
            [
                r"\begin{center}",
                r"\small",
                r"\begin{tabular}{llrrrrr}",
                r"\toprule",
                r"Host & Probe & File GB & Decode & SD & CV (\%) & Effective GB/s\\",
                r"\midrule",
                *(" & ".join(row) + r" \\" for row in probe_rows),
                r"\bottomrule",
                r"\end{tabular}",
                r"\end{center}",
            ]
        )

    projection_table: list[list[str]] = []
    for _, row in projection.iterrows():
        projection_table.append(
            [
                tex(host_label(row.host) if row.host != "all hosts" else row.host),
                tex(row.loss),
                tex(row.model),
                tex(row.split),
                str(int(row.n)),
                ffmt(row["MAPE_%"]),
                ffmt(row["median_APE_%"]),
                ffmt(row["p90_APE_%"]),
                ffmt(row["max_APE_%"]),
            ]
        )
    lines.extend(
        [
            r"\section{Output-projection sensitivity}",
            "The additive output-projection term is reported as a "
            "sensitivity analysis. It is not selected: held-out improvement is not "
            "consistent across loss functions and hosts, and the term partly recharges "
            "weights already included in the streamed-byte term. Host rows and the "
            "unequally weighted all-host aggregate are shown separately. The one-term "
            "rows in this section are refit under the listed time-domain losses; they "
            "are not the median-ratio B2 headline and need not reproduce its MAPE.",
            r"\begin{center}",
            r"\small",
            r"\begin{tabular}{llllrrrrr}",
            r"\toprule",
            r"Host & Loss & Model & Split & $n$ & MAPE & Median & P90 & Max\\",
            r"\midrule",
            *(" & ".join(row) + r" \\" for row in projection_table),
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{center}",
        ]
    )

    provenance_rows = [
        [tex(label), path_cell(path.relative_to(ROOT).as_posix()), hash_cell(sha256(path))]
        for label, path in paths.items()
    ]
    source = metadata_doc.get("source") or {}
    rtx_rows = dec[dec.host == "rtx5080"]
    rtx_largest = rtx_rows.loc[rtx_rows.bytes_total.astype(float).idxmax()]
    rtx_vram_gb = float(calibrations["rtx5080"]["torch_gpu"]["vram_gb"])
    rtx_prefill = pf[pf.host == "rtx5080"]
    rtx_prefill_over_gate = int(
        (rtx_prefill.row_cv_pct > pd.to_numeric(rtx_prefill.max_cv_pct)).sum()
    )
    heldout_by_host = {
        host: int(
            dec.loc[(dec.host == host) & (dec.split == "test"), "model_file"].nunique()
        )
        for host in hosts
    }
    add_longtable(
        lines,
        "Exact inputs used to generate this appendix.",
        "tab:hashes",
        r"@{}L{1.35in} L{2.55in} L{5.5in}@{}",
        ["Role", "Path", "SHA-256"],
        provenance_rows,
        lead=[
            r"\section{Provenance and explicit limits}",
            "The on-disk inputs used for this appendix have the following SHA-256 "
            "digests. The repository HEAD at generation was "
            + hash_cell(git_head())
            + "; the generator does not assert a clean working tree, so the file "
            "digests, rather than this commit alone, identify the inputs.",
        ],
    )
    lines.extend(
        [
            r"\begin{itemize}",
            r"\item Apple M4 Max and RTX 5080 contributed unequal, overlapping cohorts. Headline coefficients are fit separately by host. The leave-one-host-out audit uses measured target bandwidth and only two systems, so it does not establish universal transfer to new runtime stacks or formats.",
            r"\item Every scored row records a requested \texttt{n\_gpu\_layers=99}, but no device-memory trace or runtime-reported resident-layer count was retained. This is a maximal-offload request, not proof of full accelerator residency; no partial-offload sweep or offload-cliff result exists.",
            f"\\item The largest RTX modeled file-plus-KV footprint is "
            f"{float(rtx_largest.bytes_total) / 1e9:.3f} GB for "
            + path_cell(rtx_largest.model_file)
            + f" at depth {int(rtx_largest.n_depth):,}, while calibration reports "
            f"{rtx_vram_gb:.3f} GB of device memory. The differing accounting "
            "conventions further preclude a residency claim.",
            r"\item The Mac rows use the placeholder commit value \texttt{nogit}; both runtime version queries returned usage lines rather than immutable \texttt{llama.cpp} revisions. The RTX project commit field is not itself a runtime binary revision.",
            r"\item Manifest entries identify repository, filename, and byte size, but not immutable repository revisions or full-file hashes. Exact experimental replay is therefore not guaranteed.",
            r"\item Derived parameter counts and per-depth KV bytes are consumed from the frozen metadata snapshot during appendix generation. This reproduces the analysis but is not a fresh, independent metadata extraction.",
            f"\\item Held-out coverage is small: {heldout_by_host['lun-mac']} Mac "
            f"and {heldout_by_host['rtx5080']} RTX configurations, all Q4-family. "
            "Other quantization formats occur only in training data. No learning "
            "curve establishes how many reference configurations suffice.",
            r"\item Decode MAPE treats three depths from each host--file configuration as observations; those rows are correlated. No narrow confidence claim is warranted.",
            f"\\item Prefill's primary result is restricted to depth 0. Larger-depth "
            "predictions are scope diagnostics because the equation has no existing-prefix "
            f"term. The retry rule is decode-only, and {rtx_prefill_over_gate} of "
            f"{len(rtx_prefill)} scored RTX prefill rows exceed 3\\% within-run CV; "
            "the RTX prefill result should therefore be treated as both inaccurate and noisy.",
            r"\item The three Q8 calibration probes are excluded only from the Mac fit and score; two corresponding files are scored on RTX. The 63.39-GB gpt-oss-120B Mac attempt is recorded as a prompt-batch decode failure and contributes no throughput row; it was not attempted on RTX.",
            r"\item Qwen3.8-27B IQ2 has 64 layers and 26.90B parameters, whereas the other shown Qwen3.8-27B files have 65 layers and 27.32B parameters. The quantization plot is therefore a near-family comparison, not a controlled bit-format substitution for one identical network.",
            r"\end{itemize}",
            "The metadata snapshot additionally records these parser/source Git object IDs:",
            r"\begin{center}",
            r"\small",
            r"\begin{tabular}{ll}",
            r"\toprule",
            r"Object & Identifier\\",
            r"\midrule",
            "Code commit & " + hash_cell(source.get("code_commit", "unavailable")) + r" \\",
            "Measurement blob & " + hash_cell(source.get("measurement_blob", "unavailable")) + r" \\",
            "Predictions blob & " + hash_cell(source.get("predictions_blob", "unavailable")) + r" \\",
            "Shape-log blob & " + hash_cell(source.get("shape_log_blob", "unavailable")) + r" \\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{center}",
            r"\end{document}",
            "",
        ]
    )

    output = HERE / "supplement.tex"
    output.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(f"wrote {output}")
    print(f"wrote {HERE / 'supplement_summary.pdf'}")


if __name__ == "__main__":
    main()
