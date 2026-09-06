"""Regenerate publication figures and apply three-host layout adjustments.

The measurement-source checksum intentionally freezes ``llmperf/*.py`` at the
code used for the Studio campaign.  Paper-only typography therefore lives here:
the underlying plotting functions and data remain unchanged, while labels are
shortened where adding a third host would otherwise make them collide.

Run from the repository root::

    python paper/icassp2027/generate_main_figures.py
"""
from __future__ import annotations

import re
import sys
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from llmperf import figures as base


def _with_save_adjustment(call: Callable[[], str], adjust: Callable) -> str:
    original_save = base.save

    def adjusted_save(fig, out, name):
        adjust(fig)
        return original_save(fig, out, name)

    base.save = adjusted_save
    try:
        return call()
    finally:
        base.save = original_save


def _short_host_label(text: str) -> str:
    return (text.replace("MacBook M4 Max", "MacBook")
            .replace("Mac Studio M4 Max", "Studio")
            .replace("RTX 5080", "RTX"))


def _adjust_mechanisms(fig) -> None:
    axis = fig.axes[0]
    labels = []
    for tick in axis.get_xticklabels():
        text = _short_host_label(tick.get_text())
        text = re.sub(r"\((\d+) configs\. × 3 depths\)", r"(\1 cfg. × 3)", text)
        labels.append(text)
    axis.set_xticklabels(labels)
    architecture_legend = axis.get_legend()
    if architecture_legend is not None:
        architecture_legend.set_frame_on(True)
        architecture_legend.get_frame().set_facecolor(base.SURFACE)
        architecture_legend.get_frame().set_edgecolor("none")
        architecture_legend.get_frame().set_alpha(1.0)


def _adjust_scope(fig) -> None:
    axis = fig.axes[1]
    handles, labels = axis.get_legend_handles_labels()
    legend = axis.legend(handles, [_short_host_label(label) for label in labels],
                         loc="upper left", bbox_to_anchor=(0.02, 0.98), ncol=3,
                         columnspacing=0.55, handletextpad=0.3, frameon=True)
    legend.get_frame().set_facecolor(base.SURFACE)
    legend.get_frame().set_edgecolor("none")
    legend.get_frame().set_alpha(1.0)
    for annotation in axis.texts:
        if annotation.get_text() in {"108.2%", "93.2%"}:
            annotation.set_position((5, -8))
            annotation.set_ha("left")
            annotation.set_va("top")


def _adjust_quality(fig) -> None:
    axis = fig.axes[1]
    for annotation in axis.texts:
        if ">3%" in annotation.get_text():
            annotation.set_text(annotation.get_text().split()[0])


def main() -> int:
    # Generate the complete publication set first.
    result = base.main([])

    # Recompute the three figures that need paper-only layout adjustments.
    all_rows, all_decode = base._load()
    decode = base.select_primary_measurements(all_decode).reset_index(drop=True)
    train = decode[decode["split"] == "train"]
    if train.empty:
        train = decode
    eta_active = base.fit_eta(train, "bytes_active")
    eta_total = base.fit_eta(train, "bytes_total")
    predictions = {
        "B0 uncalibrated": (
            decode["bw"].to_numpy() / decode["bytes_total"].to_numpy()
        ),
        "B1 total params": base.apply_eta(decode, eta_total, "bytes_total"),
        "B2 active params": base.apply_eta(decode, eta_active, "bytes_active"),
    }

    _with_save_adjustment(
        lambda: base.fig_mechanisms(decode, predictions, base.FIGURES_DIR / "paper"),
        _adjust_mechanisms,
    )
    _with_save_adjustment(
        lambda: base.fig_scope(all_rows, base.FIGURES_DIR / "paper"),
        _adjust_scope,
    )
    _with_save_adjustment(
        lambda: base.fig_quality_vs_error(
            decode,
            predictions["B2 active params"],
            all_rows,
            base.FIGURES_DIR / "paper",
        ),
        _adjust_quality,
    )
    print("applied three-host paper layout adjustments to figures 2, 3, and 4")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
