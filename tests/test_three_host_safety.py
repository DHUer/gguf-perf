import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from llmperf.analyze import (fig_context as diagnostic_context,
                             declared_protocol_mask, load_calibration,
                             load_measurements, select_primary_measurements,
                             validate_host_coverage,
                             validate_host_training_coverage)
from llmperf.figures import (C, HOST_STYLES, ablation_error_rows, host_style,
                             ordered_hosts, quality_correlations_by_host)
from llmperf.refine import load_rows as load_refinement_rows


class HostInputSafetyTests(unittest.TestCase):
    def test_measurement_filename_must_match_every_row_host(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "measurements_expected.csv"
            pd.DataFrame({"host": ["expected", "other"]}).to_csv(
                path, index=False)
            with self.assertRaisesRegex(ValueError,
                                        "measurement host mismatch"):
                load_measurements(Path(tmp))

    def test_measurement_host_must_be_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "measurements_host-a.csv"
            pd.DataFrame({"host": [""]}).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError,
                                        "missing or malformed host ID"):
                load_measurements(Path(tmp))

    def test_duplicate_measurement_host_across_files_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pd.DataFrame({"host": ["host-a"]}).to_csv(
                root / "measurements_host-a.csv", index=False)
            pd.DataFrame({"host": ["host-a"]}).to_csv(
                root / "measurements_alias.csv", index=False)
            with self.assertRaisesRegex(ValueError,
                                        "duplicate measurement host ID"):
                load_measurements(root)

    def test_calibration_filename_and_payload_must_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "calibration_expected.json"
            path.write_text(json.dumps({"host": "other"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError,
                                        "calibration host mismatch"):
                load_calibration(Path(tmp))

    def test_calibration_host_must_be_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "calibration_host-a.json"
            path.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError,
                                        "missing or malformed host ID"):
                load_calibration(Path(tmp))

    def test_duplicate_calibration_host_across_files_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("host-a", "alias"):
                (root / f"calibration_{name}.json").write_text(
                    json.dumps({"host": "host-a"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError,
                                        "duplicate calibration host ID"):
                load_calibration(root)

    def test_measurements_and_calibrations_require_exact_host_coverage(self) -> None:
        rows = pd.DataFrame({"host": ["host-a", "host-b"]})
        calibrations = {"host-a": {"host": "host-a"}}
        with self.assertRaisesRegex(ValueError,
                                    "missing calibration for host-b"):
            validate_host_coverage(rows, calibrations)

        calibrations["host-b"] = {"host": "host-b"}
        calibrations["host-c"] = {"host": "host-c"}
        with self.assertRaisesRegex(ValueError,
                                    "missing measurements for host-c"):
            validate_host_coverage(rows, calibrations)
        validate_host_coverage(rows, calibrations, require_exact=False)

    def test_refinement_loader_requires_exact_host_coverage(self) -> None:
        rows = pd.DataFrame({"host": ["host-a"]})
        calibrations = {
            "host-a": {"host": "host-a"},
            "host-b": {"host": "host-b"},
        }
        with patch("llmperf.refine.load_measurements", return_value=rows), \
                patch("llmperf.refine.load_calibration",
                      return_value=calibrations), \
                self.assertRaisesRegex(ValueError,
                                       "missing measurements for host-b"):
            load_refinement_rows(Path("unused"), Path("unused"))

        # Intentional library subsets retain an explicit opt-out. Reaching the
        # feature builder proves the extra calibration was accepted.
        with patch("llmperf.refine.load_measurements", return_value=rows), \
                patch("llmperf.refine.load_calibration",
                      return_value=calibrations), \
                patch("llmperf.refine.load_model_metadata", return_value={}), \
                patch("llmperf.refine.load_splits", return_value={}), \
                patch("llmperf.refine.add_features",
                      side_effect=RuntimeError("continued")), \
                self.assertRaisesRegex(RuntimeError, "continued"):
            load_refinement_rows(
                Path("unused"), Path("unused"), require_exact_hosts=False)

    def test_refinement_loader_requests_snapshot_only_metadata(self) -> None:
        class StopAfterFeatureSelection(Exception):
            pass

        rows = pd.DataFrame({"host": ["host-a"]})
        calibrations = {"host-a": {"host": "host-a"}}
        with patch("llmperf.refine.load_measurements", return_value=rows), \
                patch("llmperf.refine.load_calibration",
                      return_value=calibrations), \
                patch("llmperf.refine.load_model_metadata", return_value={}), \
                patch("llmperf.refine.load_splits", return_value={}), \
                patch("llmperf.refine.add_features",
                      side_effect=StopAfterFeatureSelection) as add, \
                self.assertRaises(StopAfterFeatureSelection):
            load_refinement_rows(Path("unused"), Path("unused"))

        self.assertFalse(add.call_args.kwargs["prefer_live"])

    def test_every_fitted_host_requires_training_after_probe_exclusion(self) -> None:
        rows = pd.DataFrame({
            "host": ["host-a", "host-a", "host-b"],
            "split": ["train", "test", "test"],
            "quant": ["Q4", "Q8", "Q8"],
        })
        with self.assertRaisesRegex(ValueError, "host-b"):
            validate_host_training_coverage(
                rows, {"host-a", "host-b"}, context="decode test")

        # Coverage is host-level, not host-by-quant. A held-out quantization on
        # an otherwise trained host intentionally uses the host-median fallback.
        validate_host_training_coverage(
            rows[rows["host"] == "host-a"], {"host-a"},
            context="decode test")


class ProtocolEligibilityTests(unittest.TestCase):
    @staticmethod
    def _row(**overrides):
        row = {
            "timestamp": "2026-01-01T00:00:00Z",
            "host": "host-a", "model_file": "model.gguf",
            "n_gpu_layers": 99, "n_depth": 4096,
            "n_prompt": 0, "n_gen": 128,
            "avg_ts": 20.0, "stddev_ts": 0.2,
            "file_bytes": 10, "n_params": 10, "n_active_params": 10,
            "n_layers": 1, "n_kv_heads": 1, "head_dim": 1,
            "n_expert": 0, "error": "",
            "repetitions": 5, "settle_s": 45, "max_cv_pct": 3,
            "flash_attn": 1, "type_k": "f16", "type_v": "f16",
            "n_batch": 2048, "n_ubatch": 512,
            "load_before": 1.0, "load_after": 1.0, "attempts": 1,
            "kept_cv_pct": 1.0,
        }
        row.update(overrides)
        return row

    def test_exact_decode_and_prefill_shapes_are_eligible(self) -> None:
        rows = [self._row(n_depth=depth) for depth in (0, 4096, 16384)]
        rows += [self._row(n_depth=depth, n_prompt=512, n_gen=0)
                 for depth in (0, 4096, 16384)]
        self.assertTrue(declared_protocol_mask(pd.DataFrame(rows)).all())

    def test_every_declared_setting_is_required(self) -> None:
        wrong = {
            "repetitions": 4,
            "settle_s": 44,
            "max_cv_pct": 2,
            "n_gpu_layers": 98,
            "n_depth": 2048,
            "flash_attn": 0,
            "type_k": "q8_0",
            "type_v": "q8_0",
            "n_batch": 1024,
            "n_ubatch": 256,
            "n_prompt": 1,
            "n_gen": 127,
        }
        for field, value in wrong.items():
            with self.subTest(field=field):
                row = self._row(**{field: value})
                self.assertFalse(bool(
                    declared_protocol_mask(pd.DataFrame([row])).iloc[0]))

    def test_wrong_protocol_duplicate_cannot_displace_correct_row(self) -> None:
        correct = self._row()
        wrong = self._row(
            timestamp="2026-01-02T00:00:00Z", repetitions=1,
            avg_ts=999.0, stddev_ts=0.001, kept_cv_pct=0.001,
            load_before=0.01, load_after=0.01,
        )
        with tempfile.TemporaryDirectory() as tmp:
            pd.DataFrame([wrong, correct]).to_csv(
                Path(tmp) / "measurements_host-a.csv", index=False)
            got = load_measurements(Path(tmp))

        self.assertEqual(len(got), 1)
        self.assertEqual(float(got.iloc[0]["avg_ts"]), 20.0)
        self.assertTrue(bool(got.iloc[0]["quality_protocol"]))

    def test_only_genuine_legacy_rows_receive_fallback(self) -> None:
        rows = pd.DataFrame({
            "model_file": ["legacy.gguf", "wrong.gguf"],
            "n_gpu_layers": [99, 99],
            "quality_protocol": [False, False],
            "load_before": [np.nan, 1.0],
            "load_after": [np.nan, 1.0],
            "attempts": [np.nan, 1.0],
            "kept_cv_pct": [np.nan, 1.0],
            "max_cv_pct": [np.nan, 3.0],
        })
        got = select_primary_measurements(rows)
        self.assertEqual(got["model_file"].tolist(), ["legacy.gguf"])

    def test_stale_quality_flag_cannot_bypass_raw_protocol_check(self) -> None:
        wrong = self._row(repetitions=1, quality_protocol=True)
        got = select_primary_measurements(pd.DataFrame([wrong]))
        self.assertTrue(got.empty)


class ThreeHostPlotSafetyTests(unittest.TestCase):
    def test_known_host_styles_are_fixed_and_distinct(self) -> None:
        self.assertEqual(HOST_STYLES["lun-mac"]["color"], C[0])
        self.assertEqual(HOST_STYLES["rtx5080"]["color"], C[1])
        self.assertEqual(HOST_STYLES["mac-studio-m4-max"]["color"], C[2])
        styles = [HOST_STYLES[name] for name in
                  ("lun-mac", "rtx5080", "mac-studio-m4-max")]
        self.assertEqual(len({style["marker"] for style in styles}), 3)
        self.assertEqual(len({style["linestyle"] for style in styles}), 3)
        self.assertEqual(host_style("future-host"), host_style("future-host"))
        self.assertEqual(
            ordered_hosts(["rtx5080", "mac-studio-m4-max", "lun-mac"]),
            ["lun-mac", "mac-studio-m4-max", "rtx5080"],
        )

    def test_ablation_summary_never_pools_hosts(self) -> None:
        rows = pd.DataFrame({
            "host": ["host-a", "host-b"],
            "split": ["test", "test"],
            "is_moe": [True, True],
            "avg_ts": [10.0, 10.0],
        })
        summary = ablation_error_rows(rows, {"B2": np.array([10.0, 20.0])})
        got = summary.set_index("host")["MAPE_%"].to_dict()
        self.assertEqual(got, {"host-a": 0.0, "host-b": 100.0})

    def test_quality_correlation_is_computed_per_host(self) -> None:
        rows = pd.DataFrame({
            "host": ["host-a"] * 3 + ["host-b"] * 3,
            "cv": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
            "ape": [1.0, 2.0, 3.0, 3.0, 2.0, 1.0],
        })
        got = quality_correlations_by_host(rows)
        self.assertAlmostEqual(got["host-a"], 1.0)
        self.assertAlmostEqual(got["host-b"], -1.0)

    def test_diagnostic_context_never_connects_hosts(self) -> None:
        rows = pd.DataFrame({
            "host": ["host-a", "host-a", "host-b", "host-b"],
            "model_name": ["same-model"] * 4,
            "n_depth": [0, 4096, 0, 4096],
            "avg_ts": [10.0, 8.0, 20.0, 16.0],
        })
        pred = np.array([9.0, 7.0, 19.0, 15.0])
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(Axes, "plot", autospec=True) as plot, \
                patch.object(Axes, "legend", autospec=True), \
                patch.object(Figure, "savefig", autospec=True):
            self.assertTrue(
                diagnostic_context(rows, pred, Path(tmp) / "context.png"))
        # Each host gets one measured and one predicted line.  Grouping by
        # model alone would make only two four-point lines and connect hosts.
        self.assertEqual(plot.call_count, 4)
        for call in plot.call_args_list:
            self.assertEqual(len(call.args[1]), 2)


if __name__ == "__main__":
    unittest.main()
