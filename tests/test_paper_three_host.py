"""Focused checks for the staged three-host paper handoff."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import pandas as pd

from paper.icassp2027 import audit_submission
from paper.icassp2027.audit_submission import (
    STUDIO_PROVENANCE_INPUTS,
    three_host_content_errors,
)
from paper.icassp2027.generate_supplement import (
    DECLARED_DEPTHS,
    EXPECTED_HOSTS,
    binary_provenance_errors,
    campaign_validation_errors,
    measurement_source_provenance_errors,
    model_integrity_provenance_errors,
    model_source_provenance_errors,
)


HOST = "mac-studio-m4-max"
ENVIRONMENT = {
    "host": HOST,
    "platform": "Darwin-arm64",
    "git_commit": "abc1234",
    "llama_bench_version": "version b10794",
    "threads": 12,
    "protocol": {
        "flash_attn": "on (pinned)",
        "cache_type_k": "f16",
        "cache_type_v": "f16",
        "repetitions": 5,
        "settle_seconds": 45.0,
        "warmup": "llama-bench default (enabled)",
    },
}


def protocol_row(model: str, phase: str, depth: int) -> dict:
    return {
        "host": HOST,
        "platform": ENVIRONMENT["platform"],
        "git_commit": ENVIRONMENT["git_commit"],
        "llama_bench_version": ENVIRONMENT["llama_bench_version"],
        "model_file": model,
        "n_gpu_layers": 99,
        "n_prompt": 0 if phase == "decode" else 512,
        "n_gen": 128 if phase == "decode" else 0,
        "n_depth": depth,
        "n_threads": ENVIRONMENT["threads"],
        "flash_attn": 1,
        "type_k": "f16",
        "type_v": "f16",
        "n_batch": 2048,
        "n_ubatch": 512,
        "samples_ts": json.dumps([1, 2, 3, 4, 5]),
        "settle_s": 45.0,
        "repetitions": 5,
        "load_before": 0.1,
        "load_after": 0.2,
        "attempts": 1,
        "kept_cv_pct": 0.5,
        "max_cv_pct": 3.0,
        "error": "",
    }


def complete_grid(model: str) -> list[dict]:
    return [
        protocol_row(model, phase, depth)
        for phase in ("decode", "prefill")
        for depth in DECLARED_DEPTHS
    ]


def error_row(model: str) -> dict:
    row = protocol_row(model, "prefill", 0)
    row.update(
        {
            "flash_attn": None,
            "type_k": None,
            "type_v": None,
            "n_batch": None,
            "n_ubatch": None,
            "samples_ts": None,
            "load_before": 0.1,
            "load_after": 0.2,
            "attempts": 1,
            "kept_cv_pct": None,
            "error": "runner failed explicitly",
        }
    )
    return row


class CampaignCompletenessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = {
            "a.gguf": {"repo_id": "owner/a"},
            "b.gguf": {"repo_id": "owner/b"},
        }

    def validate(self, selected: list[dict], raw: list[dict], *, full: bool = True):
        return campaign_validation_errors(
            pd.DataFrame(selected),
            pd.DataFrame(raw),
            self.manifest,
            ENVIRONMENT,
            HOST,
            require_manifest_coverage=full,
        )

    def test_full_grid_or_explicit_error_completes_manifest_campaign(self) -> None:
        selected = complete_grid("a.gguf")
        self.assertEqual(self.validate(selected, selected + [error_row("b.gguf")]), [])

    def test_partial_grid_fails_even_when_an_error_row_also_exists(self) -> None:
        selected = complete_grid("a.gguf")[:-1]
        errors = self.validate(selected, selected + [error_row("a.gguf"), error_row("b.gguf")])
        self.assertTrue(any("a.gguf has 5/6" in error for error in errors), errors)

    def test_intentional_subset_does_not_require_every_manifest_model(self) -> None:
        selected = complete_grid("a.gguf")
        self.assertEqual(self.validate(selected, selected, full=False), [])

    def test_selected_rows_must_match_environment_and_protocol(self) -> None:
        selected = complete_grid("a.gguf")
        selected[0]["n_threads"] = 8
        errors = self.validate(selected, selected + [error_row("b.gguf")])
        self.assertTrue(any("environment identity" in error for error in errors), errors)

    def test_environment_declaration_must_match_exact_protocol(self) -> None:
        selected = complete_grid("a.gguf")
        environment = {
            **ENVIRONMENT,
            "protocol": {**ENVIRONMENT["protocol"], "repetitions": 3},
        }
        errors = campaign_validation_errors(
            pd.DataFrame(selected),
            pd.DataFrame(selected + [error_row("b.gguf")]),
            self.manifest,
            environment,
            HOST,
            require_manifest_coverage=True,
        )
        self.assertTrue(
            any("environment protocol repetitions" in error for error in errors),
            errors,
        )


class ProvenanceAndFreshnessTests(unittest.TestCase):
    def test_model_source_manifest_and_hash_coverage(self) -> None:
        manifest = {
            "a.gguf": {"repo_id": "owner/repo"},
            "b.gguf": {"repo_id": "owner/repo"},
        }
        document = {
            "schema_version": 1,
            "frozen_manifest_sha256": "a" * 64,
            "repository_revisions": {"owner/repo": "b" * 40},
            "file_lfs_sha256": {"a.gguf": "c" * 64, "b.gguf": "d" * 64},
        }
        self.assertEqual(
            model_source_provenance_errors(document, manifest, "a" * 64), []
        )
        self.assertTrue(
            model_source_provenance_errors(document, manifest, "e" * 64)
        )


    def test_binary_checksum_must_match_package_metadata(self) -> None:
        digest = "a" * 64
        self.assertEqual(
            binary_provenance_errors(
                f"runner_sha256: {digest}\n", f"{digest}  /tmp/runner\n",
                "runner_sha256", "runner",
            ),
            [],
        )
        self.assertTrue(
            binary_provenance_errors(
                f"runner_sha256: {digest}\n", f"{'b' * 64}  /tmp/runner\n",
                "runner_sha256", "runner",
            )
        )

    def test_packaged_pdf_text_is_part_of_freshness_check(self) -> None:
        documents = {
            "main source": "TITLE ACROSS THREE SYSTEMS Mac Studio M4 Max",
            "supplement source": "Mac Studio M4 Max",
            "main PDF": "TITLE ACROSS TWO SYSTEMS Mac Studio M4 Max",
            "supplement PDF": "Mac Studio M4 Max",
        }
        errors = three_host_content_errors(documents)
        self.assertTrue(any("main PDF" in error for error in errors), errors)


class ModelIntegrityProvenanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest_digest = "f" * 64
        self.manifest = {
            "a.gguf": {"size_bytes": 10},
            "b.gguf": {"size_bytes": 20},
        }
        self.model_sources = {
            "file_lfs_sha256": {"a.gguf": "a" * 64, "b.gguf": "b" * 64}
        }
        self.report = {
            "schema_version": 1,
            "host": HOST,
            "status": "pass",
            "errors": [],
            "manifest_sha256": self.manifest_digest,
            "expected_count": 2,
            "verified_count": 2,
            "expected_bytes": 30,
            "verified_bytes": 30,
            "files": {
                name: {
                    "size_bytes": entry["size_bytes"],
                    "expected_sha256": self.model_sources[
                        "file_lfs_sha256"
                    ][name],
                    "actual_sha256": self.model_sources[
                        "file_lfs_sha256"
                    ][name],
                    "sha256_ok": True,
                    "header_ok": True,
                    "metadata_ok": True,
                    "metadata": {"arch": "test"},
                    "metadata_mismatches": {},
                }
                for name, entry in self.manifest.items()
            },
        }

    def validate(self, report: dict) -> list[str]:
        return model_integrity_provenance_errors(
            report,
            self.manifest,
            self.model_sources,
            self.manifest_digest,
            HOST,
        )

    def test_exact_integrity_report_passes(self) -> None:
        self.assertEqual(self.validate(self.report), [])

    def test_summary_fields_are_fail_closed(self) -> None:
        bad_values = {
            "schema_version": 2,
            "host": "other-host",
            "status": "fail",
            "errors": ["hash failed"],
            "manifest_sha256": "0" * 64,
            "expected_count": 1,
            "verified_count": 1,
            "expected_bytes": 29,
            "verified_bytes": 29,
        }
        for field, value in bad_values.items():
            with self.subTest(field=field):
                report = copy.deepcopy(self.report)
                report[field] = value
                self.assertTrue(self.validate(report))

    def test_file_coverage_and_attestations_are_fail_closed(self) -> None:
        report = copy.deepcopy(self.report)
        report["files"].pop("b.gguf")
        report["files"]["extra.gguf"] = copy.deepcopy(
            report["files"]["a.gguf"]
        )
        errors = self.validate(report)
        self.assertTrue(any("omits file(s): b.gguf" in error for error in errors))
        self.assertTrue(
            any("unexpected file(s): extra.gguf" in error for error in errors)
        )

        corruptions = {
            "size_bytes": 9,
            "expected_sha256": "0" * 64,
            "actual_sha256": "0" * 64,
            "sha256_ok": "true",
            "header_ok": False,
            "metadata_ok": False,
            "metadata": None,
            "metadata_mismatches": {"arch": {"expected": "x", "actual": "y"}},
        }
        for field, value in corruptions.items():
            with self.subTest(field=field):
                report = copy.deepcopy(self.report)
                report["files"]["a.gguf"][field] = value
                self.assertTrue(self.validate(report))

    def test_repository_integrity_report_passes(self) -> None:
        root = Path(__file__).resolve().parents[1]
        results = root / "results"
        manifest_path = results / "model_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        model_sources = json.loads(
            (results / "model_sources_mac-studio-m4-max.json").read_text(
                encoding="utf-8"
            )
        )
        report = json.loads(
            (results / "model_integrity_mac-studio-m4-max.json").read_text(
                encoding="utf-8"
            )
        )
        digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        self.assertEqual(
            model_integrity_provenance_errors(
                report, manifest, model_sources, digest, HOST
            ),
            [],
        )


class MeasurementSourceProvenanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        (self.root / "llmperf").mkdir()
        (self.root / "llmperf" / "__init__.py").write_text(
            "", encoding="utf-8"
        )
        (self.root / "llmperf" / "sweep.py").write_text(
            "VALUE = 1\n", encoding="utf-8"
        )
        (self.root / "requirements.txt").write_text(
            "pandas\n", encoding="utf-8"
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def inventory(self, names: list[str] | None = None) -> str:
        names = names or [
            "llmperf/__init__.py",
            "llmperf/sweep.py",
            "requirements.txt",
        ]
        return "\n".join(
            hashlib.sha256((self.root / name).read_bytes()).hexdigest()
            + "  "
            + name
            for name in names
        )

    def test_exact_measurement_source_inventory_passes(self) -> None:
        self.assertEqual(
            measurement_source_provenance_errors(
                self.inventory(), self.root
            ),
            [],
        )

    def test_new_or_omitted_module_cannot_escape_inventory(self) -> None:
        inventory = self.inventory()
        (self.root / "llmperf" / "new_module.py").write_text(
            "VALUE = 2\n", encoding="utf-8"
        )
        errors = measurement_source_provenance_errors(inventory, self.root)
        self.assertEqual(
            errors,
            [
                "measurement-source inventory omits required file(s): "
                "llmperf/new_module.py"
            ],
        )

    def test_malformed_checksum_line_fails(self) -> None:
        errors = measurement_source_provenance_errors(
            self.inventory() + "\nnot-a-sha  README.md", self.root
        )
        self.assertEqual(
            errors,
            ["measurement-source checksum line 4 is malformed"],
        )

    def test_checksum_mismatch_fails(self) -> None:
        inventory = self.inventory().replace(
            hashlib.sha256(
                (self.root / "llmperf" / "sweep.py").read_bytes()
            ).hexdigest(),
            "0" * 64,
        )
        errors = measurement_source_provenance_errors(inventory, self.root)
        self.assertEqual(len(errors), 1, errors)
        self.assertIn(
            "measurement-source checksum mismatch for llmperf/sweep.py",
            errors[0],
        )

    def test_listed_required_file_missing_fails(self) -> None:
        inventory = self.inventory()
        (self.root / "requirements.txt").unlink()
        errors = measurement_source_provenance_errors(inventory, self.root)
        self.assertEqual(
            errors,
            [
                "listed measurement source is missing: requirements.txt",
                "required measurement source is missing: requirements.txt",
            ],
        )

    def test_existing_extra_file_is_rejected_without_hashing_it(self) -> None:
        readme = self.root / "README.md"
        readme.write_text("not measurement source\n", encoding="utf-8")
        extra = hashlib.sha256(readme.read_bytes()).hexdigest() + "  README.md"
        errors = measurement_source_provenance_errors(
            self.inventory() + "\n" + extra, self.root
        )
        self.assertEqual(
            errors,
            [
                "measurement-source inventory contains unexpected file(s): "
                "README.md"
            ],
        )

    def test_self_reference_is_rejected_explicitly(self) -> None:
        self_reference = (
            "0" * 64
            + "  results/measurement_source_tree_mac-studio-m4-max.sha256"
        )
        errors = measurement_source_provenance_errors(
            self.inventory() + "\n" + self_reference, self.root
        )
        self.assertEqual(
            errors,
            [
                "measurement-source inventory must not checksum itself: "
                "results/measurement_source_tree_mac-studio-m4-max.sha256"
            ],
        )

    def test_unsafe_path_spellings_fail_without_raising(self) -> None:
        unsafe_names = (
            "../outside.py",
            "/tmp/outside.py",
            r"..\outside.py",
            "C:/outside.py",
            "llmperf/bad\x00name.py",
        )
        for name in unsafe_names:
            with self.subTest(name=repr(name)):
                errors = measurement_source_provenance_errors(
                    self.inventory() + "\n" + "0" * 64 + "  " + name,
                    self.root,
                )
                self.assertEqual(len(errors), 1, errors)
                self.assertIn("path is unsafe", errors[0])

    def test_duplicate_entry_is_rejected(self) -> None:
        duplicate = self.inventory(["llmperf/sweep.py"])
        errors = measurement_source_provenance_errors(
            self.inventory() + "\n" + duplicate, self.root
        )
        self.assertEqual(
            errors,
            ["duplicate measurement-source checksum entry: llmperf/sweep.py"],
        )

    @unittest.skipUnless(hasattr(os, "symlink"), "symlinks unavailable")
    def test_symlink_outside_repository_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as outside_directory:
            outside = Path(outside_directory) / "outside.py"
            outside.write_text("VALUE = 3\n", encoding="utf-8")
            link = self.root / "llmperf" / "outside.py"
            try:
                link.symlink_to(outside)
            except OSError as exc:
                self.skipTest(f"cannot create symlink: {exc}")
            inventory = self.inventory() + "\n" + (
                hashlib.sha256(outside.read_bytes()).hexdigest()
                + "  llmperf/outside.py"
            )
            errors = measurement_source_provenance_errors(
                inventory, self.root
            )
            self.assertTrue(
                any("outside the repository" in error for error in errors), errors
            )

    def test_submission_audit_requires_measurement_source_inventory(self) -> None:
        self.assertEqual(
            STUDIO_PROVENANCE_INPUTS["measurement-source SHA-256"].name,
            "measurement_source_tree_mac-studio-m4-max.sha256",
        )
        self.assertEqual(
            STUDIO_PROVENANCE_INPUTS["model integrity"].name,
            "model_integrity_mac-studio-m4-max.json",
        )
        self.assertEqual(
            EXPECTED_HOSTS[HOST]["provenance"]["model integrity"],
            "model_integrity_mac-studio-m4-max.json",
        )

    def test_submission_audit_propagates_inventory_validation_failure(self) -> None:
        results = self.root / "results"
        results.mkdir()
        measurement = results / "measurements_mac-studio-m4-max.csv"
        calibration = results / "calibration_mac-studio-m4-max.json"
        environment = results / "env_mac-studio-m4-max.json"
        pd.DataFrame({"host": [HOST]}).to_csv(measurement, index=False)
        identity = {
            "host": HOST,
            "platform": "Darwin-arm64",
            "git_commit": "nogit",
            "cpu": "arm",
        }
        calibration.write_text(json.dumps(identity), encoding="utf-8")
        environment.write_text(json.dumps(identity), encoding="utf-8")
        (results / "model_manifest.json").write_text("{}", encoding="utf-8")

        provenance: dict[str, Path] = {}
        for index, label in enumerate(STUDIO_PROVENANCE_INPUTS):
            path = results / f"provenance-{index}.txt"
            path.write_text(
                "{}" if label in {"model sources", "model integrity"} else "x",
                            encoding="utf-8")
            provenance[label] = path

        with (
            mock.patch.object(
                audit_submission,
                "STUDIO_INPUTS",
                (measurement, calibration, environment),
            ),
            mock.patch.object(
                audit_submission, "STUDIO_PROVENANCE_INPUTS", provenance
            ),
            mock.patch.object(audit_submission, "ROOT", self.root),
            mock.patch(
                "llmperf.analyze.load_measurements",
                return_value=pd.DataFrame(),
            ),
            mock.patch(
                "llmperf.analyze.select_primary_measurements",
                return_value=pd.DataFrame(),
            ),
            mock.patch(
                "paper.icassp2027.generate_supplement.campaign_validation_errors",
                return_value=[],
            ),
            mock.patch(
                "paper.icassp2027.generate_supplement.model_source_provenance_errors",
                return_value=[],
            ),
            mock.patch(
                "paper.icassp2027.generate_supplement.model_integrity_provenance_errors",
                return_value=[],
            ) as integrity_validator,
            mock.patch(
                "paper.icassp2027.generate_supplement.measurement_source_provenance_errors",
                return_value=["sentinel source inventory failure"],
            ) as inventory_validator,
            mock.patch(
                "paper.icassp2027.generate_supplement.binary_provenance_errors",
                return_value=[],
            ),
            mock.patch.object(audit_submission, "report") as report,
        ):
            audit_submission.audit_three_host_freshness("main source")

        inventory_validator.assert_called_once()
        integrity_validator.assert_called_once()
        report.assert_called_once_with(
            "Studio campaign completeness",
            False,
            "sentinel source inventory failure",
        )


if __name__ == "__main__":
    unittest.main()
