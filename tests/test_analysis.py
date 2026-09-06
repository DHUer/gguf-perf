import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from llmperf import analyze as analyze_module
from llmperf.analyze import (add_features, apply_eta, error_table,
                             error_table_by_host, evaluate_prefill, fit_eta,
                             prefill_error_table_by_host, load_calibration,
                             load_measurements, load_splits, refresh_metadata, resolve_splits,
                             select_primary_measurements)
from llmperf.common import (RESULTS_DIR, load_model_metadata,
                            load_model_splits)
from llmperf.fetch import update_manifest
from llmperf.refine import (error_rows as refinement_error_rows,
                            fit_model as fit_refinement_model,
                            leave_one_machine_out_b2,
                            load_rows as load_refinement_rows, model_shapes)


class SplitProvenanceTests(unittest.TestCase):
    def test_frozen_manifest_covers_committed_measurements(self) -> None:
        splits = load_model_splits(Path("does-not-exist"), RESULTS_DIR)
        with (RESULTS_DIR / "measurements_lun-mac.csv").open(
                newline="", encoding="utf-8") as f:
            measured = {row["model_file"] for row in csv.DictReader(f)}

        self.assertEqual(measured - splits.keys(), set())
        self.assertEqual(splits["SmolLM3-Q4_K_M.gguf"], "train")
        self.assertEqual(
            splits["NVIDIA-Nemotron-3-Nano-4B-Q4_K_M.gguf"], "test")

    def test_legacy_blank_split_is_recovered(self) -> None:
        rows = pd.DataFrame({
            "model_file": ["train.gguf", "test.gguf"],
            "split": ["", None],
        })
        got = resolve_splits(
            rows, {"train.gguf": "train", "test.gguf": "test"})
        self.assertEqual(got["split"].tolist(), ["train", "test"])

    def test_embedded_split_is_self_contained(self) -> None:
        rows = pd.DataFrame({"model_file": ["new.gguf"], "split": ["test"]})
        self.assertEqual(resolve_splits(rows, {})["split"].tolist(), ["test"])

    def test_unknown_and_conflicting_splits_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "never assumed to be train"):
            resolve_splits(pd.DataFrame({"model_file": ["unknown.gguf"]}), {})

        rows = pd.DataFrame({"model_file": ["fixed.gguf"], "split": ["test"]})
        with self.assertRaisesRegex(ValueError, "conflicting"):
            resolve_splits(rows, {"fixed.gguf": "train"})

        malformed = pd.DataFrame({
            "model_file": ["fixed.gguf"], "split": ["testing"]})
        with self.assertRaisesRegex(ValueError, "invalid train/test split"):
            resolve_splits(malformed, {"fixed.gguf": "test"})

    def test_manifest_update_refuses_split_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            update_manifest(path, [("org/repo", "model.gguf", 10, "train")])
            with self.assertRaisesRegex(ValueError, "refusing to change"):
                update_manifest(path, [("org/repo", "model.gguf", 10, "test")])


class PrimaryCohortTests(unittest.TestCase):
    def test_repository_primary_rows_are_all_quality_protocol_complete(self) -> None:
        rows = load_measurements(RESULTS_DIR)
        self.assertTrue(rows["quality_protocol"].all())
        # Keep the frozen three-host manuscript cohort invariant.
        self.assertEqual(len(rows[rows["host"] == "lun-mac"]), 132)
        self.assertEqual(
            len(rows[rows["host"] == "mac-studio-m4-max"]), 138)
        self.assertEqual(len(rows[rows["host"] == "rtx5080"]), 84)

    def test_partial_offload_rows_are_not_primary(self) -> None:
        rows = pd.DataFrame({
            "model_file": ["m.gguf"] * 4,
            "n_gpu_layers": [0, 16, 48, 99],
            "avg_ts": [1.0, 2.0, 3.0, 4.0],
        })
        got = select_primary_measurements(rows)
        self.assertEqual(got["n_gpu_layers"].tolist(), [99])
        self.assertEqual(got["avg_ts"].tolist(), [4.0])

    def test_quality_gated_duplicate_is_preferred_without_column_splicing(self) -> None:
        base = {
            "host": "h", "model_file": "m.gguf", "n_gpu_layers": 99,
            "n_depth": 0, "n_prompt": 0, "n_gen": 128,
            "stddev_ts": 0.1, "file_bytes": 10, "n_params": 10,
            "n_active_params": 10, "n_layers": 1, "n_kv_heads": 1,
            "head_dim": 1, "n_expert": 0, "error": "",
            "repetitions": 5, "settle_s": 45, "flash_attn": 1,
            "type_k": "f16", "type_v": "f16", "n_batch": 2048,
            "n_ubatch": 512,
        }
        legacy = {
            **base, "timestamp": "2026-01-01T00:00:00Z", "avg_ts": 10.0,
            "load_before": None, "load_after": None, "attempts": None,
            "kept_cv_pct": None, "max_cv_pct": None,
        }
        gated = {
            **base, "timestamp": "2026-01-02T00:00:00Z", "avg_ts": 20.0,
            "load_before": 7.0, "load_after": 8.0, "attempts": 2,
            "kept_cv_pct": 2.0, "max_cv_pct": 3.0,
        }
        with tempfile.TemporaryDirectory() as tmp:
            pd.DataFrame([legacy, gated]).to_csv(
                Path(tmp) / "measurements_h.csv", index=False)
            got = load_measurements(Path(tmp))

        self.assertEqual(len(got), 1)
        self.assertEqual(float(got.iloc[0]["avg_ts"]), 20.0)
        self.assertEqual(got.iloc[0]["timestamp"], gated["timestamp"])
        self.assertEqual(float(got.iloc[0]["load_before"]), 7.0)
        self.assertTrue(bool(got.iloc[0]["quality_protocol"]))

    def test_primary_cohort_excludes_legacy_rows_when_gated_rows_exist(self) -> None:
        rows = pd.DataFrame({
            "model_file": ["legacy.gguf", "gated.gguf"],
            "n_gpu_layers": [99, 99],
            "quality_protocol": [False, True],
            "avg_ts": [1.0, 2.0],
        })
        got = select_primary_measurements(rows)
        self.assertEqual(got["model_file"].tolist(), ["gated.gguf"])


class HostTransferTests(unittest.TestCase):
    def test_unseen_quant_uses_trained_host_median(self) -> None:
        eta = pd.Series(
            [0.25, 0.75],
            index=pd.MultiIndex.from_tuples(
                [("host-a", "Q4"), ("host-a", "Q6")],
                names=["host", "quant"],
            ),
        )
        target = pd.DataFrame({
            "host": ["host-a"], "quant": ["Q8"],
            "bw": [100.0], "bytes_active": [10.0],
        })

        prediction = apply_eta(target, eta, "bytes_active")

        self.assertAlmostEqual(float(prediction[0]), 5.0)

    def test_b2_transfer_weights_source_hosts_equally(self) -> None:
        # source-a contributes three rows while source-b contributes one.  A
        # row-pooled median would therefore transfer eta=1; fitting each source
        # independently and then taking their median transfers eta=2.
        rows = pd.DataFrame([
            {"host": "source-a", "quant": "Q4", "split": "train",
             "avg_ts": 1.0, "bytes_active": 1.0, "bw": 1.0},
            {"host": "source-a", "quant": "Q4", "split": "train",
             "avg_ts": 1.0, "bytes_active": 1.0, "bw": 1.0},
            {"host": "source-a", "quant": "Q4", "split": "train",
             "avg_ts": 1.0, "bytes_active": 1.0, "bw": 1.0},
            {"host": "source-b", "quant": "Q4", "split": "train",
             "avg_ts": 3.0, "bytes_active": 1.0, "bw": 1.0},
            {"host": "target", "quant": "Q4", "split": "test",
             "avg_ts": 2.0, "bytes_active": 1.0, "bw": 1.0},
        ])

        transfer = leave_one_machine_out_b2(rows).set_index("held_out_host")
        target = transfer.loc["target"]

        self.assertEqual(target["source_hosts"], '["source-a","source-b"]')
        self.assertEqual(int(target["n_source_hosts"]), 2)
        self.assertEqual(int(target["n_train"]), 4)
        self.assertAlmostEqual(float(target["MAPE_%"]), 0.0)


class MetadataSnapshotTests(unittest.TestCase):
    def test_snapshot_matches_every_generated_prediction(self) -> None:
        metadata = load_model_metadata(RESULTS_DIR)
        predictions = pd.read_csv(RESULTS_DIR / "predictions.csv")
        fields = [
            "file_bytes", "n_params", "n_active_params", "n_layers",
            "n_kv_heads", "head_dim", "n_expert", "n_expert_used",
        ]
        for row in predictions.to_dict("records"):
            frozen = metadata[row["model_file"]]
            for field in fields:
                self.assertEqual(int(row[field]), int(frozen[field]),
                                 f"{row['model_file']} {field}")
            depth = str(int(row["n_depth"]))
            self.assertEqual(int(row["kv_bytes"]),
                             int(frozen["kv_bytes_by_depth"][depth]),
                             f"{row['model_file']} kv_bytes@{depth}")

    def test_snapshot_and_split_manifest_cover_same_cohort(self) -> None:
        self.assertEqual(set(load_model_metadata(RESULTS_DIR)),
                         set(load_model_splits(Path("missing"), RESULTS_DIR)))

    def test_missing_metadata_fails_instead_of_using_stale_csv_fields(self) -> None:
        rows = pd.DataFrame({"model_file": ["unknown.gguf"]})
        with tempfile.TemporaryDirectory() as tmp, \
                self.assertRaisesRegex(ValueError, "no trustworthy model metadata"):
            refresh_metadata(rows, Path(tmp), {})

    def test_live_gguf_metadata_takes_precedence(self) -> None:
        fields = {
            "file_bytes": 10, "n_params": 20, "n_active_params": 20,
            "n_layers": 2, "n_kv_heads": 1, "head_dim": 4,
            "n_expert": 0, "n_expert_used": 0, "quant": "Q4_K_M",
            "arch": "live",
        }
        rows = pd.DataFrame({"model_file": ["model.gguf"], **{
            key: [value] for key, value in fields.items()}})
        frozen = {"model.gguf": {**fields, "arch": "stale"}}
        live = SimpleNamespace(**fields)
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "model.gguf").touch()
            with patch("llmperf.common.read_gguf_meta", return_value=live):
                got = refresh_metadata(rows, Path(tmp), frozen)
        self.assertEqual(got.iloc[0]["arch"], "live")

    def test_snapshot_only_mode_ignores_an_installed_gguf(self) -> None:
        fields = {
            "file_bytes": 10, "n_params": 20, "n_active_params": 20,
            "n_layers": 2, "n_kv_heads": 1, "head_dim": 4,
            "n_expert": 0, "n_expert_used": 0, "quant": "Q4_K_M",
            "arch": "recorded",
        }
        rows = pd.DataFrame({"model_file": ["model.gguf"], **{
            key: [value] for key, value in fields.items()}})
        frozen = {"model.gguf": {**fields, "arch": "frozen"}}
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "model.gguf").touch()
            with patch("llmperf.common.read_gguf_meta",
                       side_effect=AssertionError("live read attempted")):
                got = refresh_metadata(
                    rows, Path(tmp), frozen, prefer_live=False)
        self.assertEqual(got.iloc[0]["arch"], "frozen")

    def test_analysis_cli_requests_snapshot_only_metadata(self) -> None:
        class StopAfterFeatureSelection(Exception):
            pass

        rows = pd.DataFrame({"host": ["host-a"]})
        calibration = {"host-a": {"host": "host-a"}}
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(analyze_module, "load_measurements",
                             return_value=rows), \
                patch.object(analyze_module, "load_calibration",
                             return_value=calibration), \
                patch.object(analyze_module, "load_splits", return_value={}), \
                patch.object(analyze_module, "load_model_metadata",
                             return_value={}), \
                patch.object(analyze_module, "add_features",
                             side_effect=StopAfterFeatureSelection) as add:
            root = Path(tmp)
            with self.assertRaises(StopAfterFeatureSelection):
                analyze_module.main([
                    "--results", str(root),
                    "--models-dir", str(root / "models"),
                    "--figures", str(root / "figures"),
                ])

        self.assertFalse(add.call_args.kwargs["prefer_live"])

    def test_refinement_snapshot_only_shapes_ignore_installed_gguf(self) -> None:
        snapshot = {
            "model.gguf": {
                "vocab_size": 123, "d_model": 456,
                "embd_bytes": 789, "tied_embeddings": False,
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "model.gguf").touch()
            with patch("llmperf.refine.read_gguf_meta",
                       side_effect=AssertionError("live read attempted")):
                got = model_shapes(
                    ["model.gguf"], Path(tmp), snapshot, prefer_live=False)

        self.assertEqual(int(got.iloc[0]["vocab_size"]), 123)
        self.assertEqual(int(got.iloc[0]["d_model"]), 456)

    def test_snapshot_for_different_file_size_fails(self) -> None:
        fields = {
            "file_bytes": 10, "n_params": 20, "n_active_params": 20,
            "n_layers": 2, "n_kv_heads": 1, "head_dim": 4,
            "n_expert": 0, "n_expert_used": 0, "quant": "Q4_K_M",
            "arch": "test",
        }
        rows = pd.DataFrame({"model_file": ["model.gguf"], **{
            key: [value] for key, value in fields.items()}})
        frozen = {"model.gguf": {**fields, "file_bytes": 11}}
        with tempfile.TemporaryDirectory() as tmp, \
                self.assertRaisesRegex(ValueError, "different file size"):
            refresh_metadata(rows, Path(tmp), frozen)

    def test_refine_shapes_fall_back_to_snapshot(self) -> None:
        metadata = load_model_metadata(RESULTS_DIR)
        with tempfile.TemporaryDirectory() as tmp:
            got = model_shapes(
                ["Muse-Glimmer-30B-UD-Q4_K_XL.gguf"], Path(tmp), metadata)
        self.assertEqual(int(got.iloc[0]["vocab_size"]), 202048)
        self.assertEqual(int(got.iloc[0]["d_model"]), 6656)

    def test_unfrozen_depth_fails_instead_of_using_uniform_kv(self) -> None:
        measurements = load_measurements(RESULTS_DIR).head(1).copy()
        measurements["n_depth"] = 12345
        with tempfile.TemporaryDirectory() as tmp, \
                self.assertRaisesRegex(ValueError, "no audited KV metadata"):
            add_features(
                measurements, load_calibration(RESULTS_DIR),
                load_splits(Path(tmp), RESULTS_DIR),
                load_model_metadata(RESULTS_DIR), Path(tmp))

    def test_clean_clone_reproduces_frozen_headline_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing_models = Path(tmp)
            measurements = load_measurements(RESULTS_DIR)
            featured = add_features(
                measurements, load_calibration(RESULTS_DIR),
                load_splits(missing_models, RESULTS_DIR),
                load_model_metadata(RESULTS_DIR), missing_models)

        dec = select_primary_measurements(
            featured[(featured["phase"] == "decode") & featured["bw"].notna()])
        dec = dec[(dec["host"] == "lun-mac") &
                  ~dec["is_calibration_probe"]].reset_index(drop=True)
        train = dec[dec["split"] == "train"]
        eta_active = fit_eta(train, "bytes_active")
        eta_total = fit_eta(train, "bytes_total")
        predictions = {
            "B0": dec["bw"].to_numpy() / dec["bytes_total"].to_numpy(),
            "B1": apply_eta(dec, eta_total, "bytes_total"),
            "B2": apply_eta(dec, eta_active, "bytes_active"),
        }
        got = error_table(dec, predictions).set_index(["model", "split"])
        expected = {
            ("B0", "train"): 33.57, ("B0", "test"): 46.89,
            ("B1", "train"): 15.80, ("B1", "test"): 49.36,
            ("B2", "train"): 10.39, ("B2", "test"): 13.11,
        }
        for key, want in expected.items():
            self.assertAlmostEqual(float(got.loc[key, "MAPE_%"]), want, places=2)

    def test_clean_clone_runs_refinement_with_frozen_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            all_dec, _ = load_refinement_rows(RESULTS_DIR, Path(tmp))
        dec = all_dec[all_dec["host"] == "lun-mac"].reset_index(drop=True)
        train = dec[dec["split"] == "train"]
        fit = fit_refinement_model(train, use_output_term=False)
        got = {row["split"]: row["MAPE_%"]
               for row in refinement_error_rows("one-term", fit, dec)}
        self.assertAlmostEqual(got["train"], 12.4, places=1)
        self.assertAlmostEqual(got["test"], 19.3, places=1)

        transfer = leave_one_machine_out_b2(all_dec).set_index("held_out_host")
        transfer_test = leave_one_machine_out_b2(
            all_dec, target_split="test").set_index("held_out_host")

        expected_all = {
            "lun-mac": {
                "source_hosts": '["mac-studio-m4-max","rtx5080"]',
                "n_source_hosts": 2, "n_train": 81, "n_held": 57,
                "MAPE_%": 14.47, "median_APE_%": 8.02,
                "max_APE_%": 73.61,
            },
            "mac-studio-m4-max": {
                "source_hosts": '["lun-mac","rtx5080"]',
                "n_source_hosts": 2, "n_train": 81, "n_held": 60,
                "MAPE_%": 15.42, "median_APE_%": 10.78,
                "max_APE_%": 84.30,
            },
            "rtx5080": {
                "source_hosts": '["lun-mac","mac-studio-m4-max"]',
                "n_source_hosts": 2, "n_train": 90, "n_held": 42,
                "MAPE_%": 20.80, "median_APE_%": 18.18,
                "max_APE_%": 68.13,
            },
        }
        expected_test = {
            "lun-mac": {
                "source_hosts": '["mac-studio-m4-max","rtx5080"]',
                "n_source_hosts": 2, "n_train": 81, "n_held": 12,
                "MAPE_%": 11.59, "median_APE_%": 6.00,
                "max_APE_%": 45.82,
            },
            "mac-studio-m4-max": {
                "source_hosts": '["lun-mac","rtx5080"]',
                "n_source_hosts": 2, "n_train": 81, "n_held": 15,
                "MAPE_%": 16.76, "median_APE_%": 14.36,
                "max_APE_%": 57.74,
            },
            "rtx5080": {
                "source_hosts": '["lun-mac","mac-studio-m4-max"]',
                "n_source_hosts": 2, "n_train": 90, "n_held": 6,
                "MAPE_%": 35.97, "median_APE_%": 25.66,
                "max_APE_%": 68.13,
            },
        }
        for actual, expected in (
                (transfer, expected_all), (transfer_test, expected_test)):
            self.assertEqual(set(actual.index), set(expected))
            for host, values in expected.items():
                self.assertEqual(
                    actual.loc[host, "source_hosts"], values["source_hosts"])
                for column in ("n_source_hosts", "n_train", "n_held"):
                    self.assertEqual(
                        int(actual.loc[host, column]), values[column])
                for column in ("MAPE_%", "median_APE_%", "max_APE_%"):
                    self.assertAlmostEqual(
                        float(actual.loc[host, column]), values[column], places=2)

    def test_prefill_headline_is_fit_and_scored_at_empty_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing_models = Path(tmp)
            measurements = load_measurements(RESULTS_DIR)
            featured = add_features(
                measurements, load_calibration(RESULTS_DIR),
                load_splits(missing_models, RESULTS_DIR),
                load_model_metadata(RESULTS_DIR), missing_models)
        primary = select_primary_measurements(featured)
        primary = primary[primary["host"] == "lun-mac"].reset_index(drop=True)
        pf, errors = evaluate_prefill(
            primary[~primary["is_calibration_probe"]])
        got = errors.set_index(["model", "split"])
        self.assertEqual(set(errors["n_depth"]), {0})
        self.assertAlmostEqual(
            float(got.loc[("P2 eta_p per host x quant", "train"), "MAPE_%"]),
            4.13, places=2)
        self.assertAlmostEqual(
            float(got.loc[("P2 eta_p per host x quant", "test"), "MAPE_%"]),
            18.68, places=2)
        self.assertEqual(set(pf["n_depth"]), {0, 4096, 16384})
        self.assertTrue({"pred_P1", "ape_P1", "pred_P2", "ape_P2"}
                        .issubset(pf.columns))

    def test_host_tables_preserve_the_three_host_headlines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing_models = Path(tmp)
            measurements = load_measurements(RESULTS_DIR)
            featured = add_features(
                measurements, load_calibration(RESULTS_DIR),
                load_splits(missing_models, RESULTS_DIR),
                load_model_metadata(RESULTS_DIR), missing_models)

        primary = select_primary_measurements(featured)
        dec = primary[(primary["phase"] == "decode")
                      & ~primary["is_calibration_probe"]].reset_index(drop=True)
        train = dec[dec["split"] == "train"]
        eta = fit_eta(train, "bytes_active")
        pred = apply_eta(dec, eta, "bytes_active")
        decode = error_table_by_host(dec, {"B2": pred}).set_index(
            ["host", "model", "split"])
        self.assertAlmostEqual(
            float(decode.loc[("lun-mac", "B2", "test"), "MAPE_%"]),
            13.11, places=2)
        self.assertAlmostEqual(
            float(decode.loc[
                ("mac-studio-m4-max", "B2", "test"), "MAPE_%"]),
            14.37, places=2)
        self.assertAlmostEqual(
            float(decode.loc[("rtx5080", "B2", "test"), "MAPE_%"]),
            36.15, places=2)

        pf, _ = evaluate_prefill(primary[~primary["is_calibration_probe"]])
        prefill = prefill_error_table_by_host(pf).set_index(
            ["host", "model", "split"])
        key = "P2 eta_p per host x quant"
        self.assertAlmostEqual(
            float(prefill.loc[("lun-mac", key, "test"), "MAPE_%"]),
            18.68, places=2)
        self.assertAlmostEqual(
            float(prefill.loc[
                ("mac-studio-m4-max", key, "test"), "MAPE_%"]),
            22.23, places=2)
        self.assertAlmostEqual(
            float(prefill.loc[("rtx5080", key, "test"), "MAPE_%"]),
            108.18, places=2)


if __name__ == "__main__":
    unittest.main()
