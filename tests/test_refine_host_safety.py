import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from llmperf.analyze import load_measurements
from llmperf.refine import (eta_variation_rows, fig_eta_variation, load_rows)


class RefinementFigureHostSafetyTests(unittest.TestCase):
    def test_eta_figure_keeps_hosts_in_separate_facets(self) -> None:
        rows = pd.DataFrame({
            "host": ["lun-mac", "lun-mac", "rtx5080", "rtx5080"],
            "quant": ["Q4_K_M"] * 4,
            "model_name": ["same-model"] * 4,
            "eta": [1.0, 1.2, 3.0, 3.2],
        })
        summary = eta_variation_rows(rows)
        self.assertEqual(len(summary), 2)
        self.assertEqual(
            summary.set_index("host")["eta"].to_dict(),
            {"lun-mac": 1.1, "rtx5080": 3.1},
        )

        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(Figure, "savefig", autospec=True), \
                patch.object(plt, "close"):
            self.assertTrue(fig_eta_variation(
                rows, Path(tmp) / "eta.png"))
            fig = plt.gcf()
        self.assertEqual(len(fig.axes), 2)
        self.assertEqual(
            [axis.get_title(loc="left") for axis in fig.axes],
            ["MacBook M4 Max", "RTX 5080"],
        )
        self.assertEqual(fig.axes[0].patches[0].get_facecolor(),
                         fig.axes[1].patches[0].get_facecolor())
        plt.close(fig)


class KeepAllTests(unittest.TestCase):
    @staticmethod
    def _row(timestamp: str, avg_ts: float) -> dict:
        return {
            "timestamp": timestamp, "host": "host-a",
            "model_file": "model.gguf", "n_gpu_layers": 99,
            "n_depth": 0, "n_prompt": 0, "n_gen": 128,
            "avg_ts": avg_ts, "stddev_ts": 0.2,
            "file_bytes": 10, "n_params": 10, "n_active_params": 10,
            "n_layers": 1, "n_kv_heads": 1, "head_dim": 1,
            "n_expert": 0, "error": "", "repetitions": 5,
            "settle_s": 45, "max_cv_pct": 3, "flash_attn": 1,
            "type_k": "f16", "type_v": "f16", "n_batch": 2048,
            "n_ubatch": 512, "load_before": 1.0, "load_after": 1.0,
            "attempts": 1, "kept_cv_pct": 1.0,
        }

    def test_loader_opt_out_retains_repeated_measurements(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pd.DataFrame([
                self._row("2026-01-01T00:00:00Z", 10.0),
                self._row("2026-01-02T00:00:00Z", 11.0),
            ]).to_csv(root / "measurements_host-a.csv", index=False)
            self.assertEqual(len(load_measurements(root)), 1)
            self.assertEqual(
                len(load_measurements(root, deduplicate=False)), 2)

    def test_keep_all_wires_loader_deduplication_off(self) -> None:
        class StopAfterLoad(Exception):
            pass

        with patch("llmperf.refine.load_measurements",
                   side_effect=StopAfterLoad) as loader:
            with self.assertRaises(StopAfterLoad):
                load_rows(Path("results"), Path("models"), collapse=False)
        loader.assert_called_once_with(Path("results"), deduplicate=False)


if __name__ == "__main__":
    unittest.main()
