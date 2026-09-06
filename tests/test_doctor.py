import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from llmperf import doctor


class DoctorRegressionTests(unittest.TestCase):
    def setUp(self) -> None:
        doctor.results.clear()

    def tearDown(self) -> None:
        doctor.results.clear()

    def test_psutil_is_a_required_dependency(self) -> None:
        imported = []

        def import_module(name):
            imported.append(name)
            if name == "psutil":
                raise ImportError("missing psutil")
            return SimpleNamespace(__version__="test")

        with patch.object(doctor.importlib, "import_module",
                          side_effect=import_module), \
                patch.dict(sys.modules, {"torch": None}), \
                redirect_stdout(io.StringIO()):
            doctor.check_deps()

        self.assertIn("psutil", imported)
        self.assertEqual(doctor.results[0][0:2], ("python deps", doctor.FAIL))
        self.assertIn("psutil", doctor.results[0][2])

    def test_mps_torch_is_reported_available_on_apple_silicon(self) -> None:
        torch = SimpleNamespace(
            __version__="2.14.0",
            cuda=SimpleNamespace(is_available=lambda: False),
            backends=SimpleNamespace(
                mps=SimpleNamespace(is_available=lambda: True),
            ),
        )

        with patch.object(doctor.importlib, "import_module",
                          return_value=SimpleNamespace(__version__="test")), \
                patch.dict(sys.modules, {"torch": torch}), \
                patch.object(doctor.platform, "system", return_value="Darwin"), \
                patch.object(doctor.platform, "machine", return_value="arm64"), \
                redirect_stdout(io.StringIO()):
            doctor.check_deps()

        self.assertEqual(doctor.results[-1], (
            "torch mps", doctor.PASS, "Metal/MPS available, torch 2.14.0",
        ))

    def test_missing_torch_fails_on_apple_silicon(self) -> None:
        with patch.object(doctor.importlib, "import_module",
                          return_value=SimpleNamespace(__version__="test")), \
                patch.dict(sys.modules, {"torch": None}), \
                patch.object(doctor.platform, "system", return_value="Darwin"), \
                patch.object(doctor.platform, "machine", return_value="arm64"), \
                redirect_stdout(io.StringIO()):
            doctor.check_deps()

        name, status, detail = doctor.results[-1]
        self.assertEqual((name, status), ("torch mps", doctor.FAIL))
        self.assertIn("torch not installed", detail)
        self.assertIn("device-copy bandwidth", detail)

    def test_unavailable_mps_fails_on_apple_silicon(self) -> None:
        torch = SimpleNamespace(
            __version__="2.14.0",
            cuda=SimpleNamespace(is_available=lambda: False),
            backends=SimpleNamespace(
                mps=SimpleNamespace(is_available=lambda: False),
            ),
        )

        with patch.object(doctor.importlib, "import_module",
                          return_value=SimpleNamespace(__version__="test")), \
                patch.dict(sys.modules, {"torch": torch}), \
                patch.object(doctor.platform, "system", return_value="Darwin"), \
                patch.object(doctor.platform, "machine", return_value="arm64"), \
                redirect_stdout(io.StringIO()):
            doctor.check_deps()

        self.assertEqual(doctor.results[-1][0:2], ("torch mps", doctor.FAIL))

    def test_llama_bench_requires_successful_device_listing(self) -> None:
        completed = SimpleNamespace(
            returncode=1, stdout="", stderr="Metal backend initialization failed")
        with patch("llmperf.common.find_llama_bench",
                   return_value=Path("/tmp/llama-bench")), \
                patch.object(doctor.subprocess, "run", return_value=completed), \
                patch.object(doctor.platform, "system", return_value="Darwin"), \
                patch.object(doctor.platform, "machine", return_value="arm64"), \
                redirect_stdout(io.StringIO()):
            doctor.check_llama_bench(quick=True)

        self.assertEqual(doctor.results[-1][0:2],
                         ("llama-bench backend", doctor.FAIL))

    def test_llama_bench_requires_platform_backend(self) -> None:
        completed = SimpleNamespace(
            returncode=0,
            stdout="Available devices:\n  CUDA0: NVIDIA test GPU\n",
            stderr="")
        with patch("llmperf.common.find_llama_bench",
                   return_value=Path("/tmp/llama-bench")), \
                patch.object(doctor.subprocess, "run", return_value=completed), \
                patch.object(doctor.platform, "system", return_value="Darwin"), \
                patch.object(doctor.platform, "machine", return_value="arm64"), \
                redirect_stdout(io.StringIO()):
            doctor.check_llama_bench(quick=True)

        name, status, detail = doctor.results[-1]
        self.assertEqual((name, status), ("llama-bench backend", doctor.FAIL))
        self.assertIn("expected Metal", detail)
        self.assertIn("found CUDA", detail)

    def test_llama_bench_accepts_expected_platform_backend(self) -> None:
        completed = SimpleNamespace(
            returncode=0, stdout="Available devices:\n  MTL0: Apple M4 Max\n",
            stderr="ggml_metal_device_init: GPU name: Apple M4 Max\n")
        with patch("llmperf.common.find_llama_bench",
                   return_value=Path("/tmp/llama-bench")), \
                patch.object(doctor.subprocess, "run", return_value=completed), \
                patch.object(doctor.platform, "system", return_value="Darwin"), \
                patch.object(doctor.platform, "machine", return_value="arm64"), \
                redirect_stdout(io.StringIO()):
            doctor.check_llama_bench(quick=True)

        self.assertEqual(doctor.results[-1][0:2],
                         ("llama-bench backend", doctor.PASS))

    def test_llama_bench_requires_cuda_on_windows(self) -> None:
        completed = SimpleNamespace(
            returncode=0,
            stdout="Available devices:\n  MTL0: unexpected device\n  BLAS: CPU\n",
            stderr="")
        with patch("llmperf.common.find_llama_bench",
                   return_value=Path("C:/llama-bench.exe")), \
                patch.object(doctor.subprocess, "run", return_value=completed), \
                patch.object(doctor.platform, "system", return_value="Windows"), \
                patch.object(doctor.platform, "machine", return_value="AMD64"), \
                redirect_stdout(io.StringIO()):
            doctor.check_llama_bench(quick=True)

        name, status, detail = doctor.results[-1]
        self.assertEqual((name, status), ("llama-bench backend", doctor.FAIL))
        self.assertIn("expected CUDA", detail)

    def test_load_average_must_be_finite(self) -> None:
        with patch("llmperf.sweep.load_average", return_value=float("nan")), \
                redirect_stdout(io.StringIO()):
            doctor.check_load_average()

        self.assertEqual(doctor.results[-1][0:2],
                         ("system load", doctor.FAIL))

    def test_load_average_reports_finite_value(self) -> None:
        with patch("llmperf.sweep.load_average", return_value=12.25), \
                patch.object(doctor.platform, "system", return_value="Windows"), \
                redirect_stdout(io.StringIO()):
            doctor.check_load_average()

        self.assertEqual(doctor.results[-1], (
            "system load", doctor.PASS,
            "12.2 busy logical CPUs (1 s sample)",
        ))

    def test_selfchecks_include_refine_with_flag(self) -> None:
        calls = []

        def run(argv, **kwargs):
            calls.append(argv)
            return SimpleNamespace(returncode=0, stdout="selfcheck ok\n", stderr="")

        with patch.object(doctor.subprocess, "run", side_effect=run), \
                redirect_stdout(io.StringIO()):
            doctor.check_selfchecks()

        self.assertEqual(calls, [
            [sys.executable, "-m", "llmperf.common"],
            [sys.executable, "-m", "llmperf.sweep", "--selfcheck"],
            [sys.executable, "-m", "llmperf.refine", "--selfcheck"],
        ])

    def test_model_check_requires_complete_exact_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models = root / "models"
            results = root / "results"
            models.mkdir()
            results.mkdir()
            manifest = {
                "complete.gguf": {"size_bytes": 3, "split": "train"},
                "wrong.gguf": {"size_bytes": 4, "split": "train"},
                "missing.gguf": {"size_bytes": 5, "split": "test"},
            }
            (results / "model_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8")
            (models / "complete.gguf").write_bytes(b"abc")
            (models / "wrong.gguf").write_bytes(b"xx")
            (models / "missing.gguf.part").write_bytes(b"yy")

            with patch.object(doctor, "MODELS_DIR", models), \
                    patch.object(doctor, "RESULTS_DIR", results), \
                    redirect_stdout(io.StringIO()):
                doctor.check_models()

        name, status, detail = doctor.results[-1]
        self.assertEqual((name, status), ("models", doctor.FAIL))
        self.assertIn("1/3 declared GGUF files complete", detail)
        self.assertIn("missing.gguf", detail)
        self.assertIn("wrong.gguf (2 != 4 bytes)", detail)
        self.assertIn("missing.gguf.part", detail)

    def test_model_check_parses_after_all_sizes_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models = root / "models"
            results = root / "results"
            models.mkdir()
            results.mkdir()
            manifest = {
                "small.gguf": {"size_bytes": 3, "split": "train"},
                "large.gguf": {"size_bytes": 5, "split": "test"},
            }
            (results / "model_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8")
            (models / "small.gguf").write_bytes(b"abc")
            (models / "large.gguf").write_bytes(b"abcde")
            metadata = SimpleNamespace(name="small", n_params=3_000_000_000,
                                       n_layers=30)

            with patch.object(doctor, "MODELS_DIR", models), \
                    patch.object(doctor, "RESULTS_DIR", results), \
                    patch("llmperf.common.read_gguf_meta", return_value=metadata), \
                    redirect_stdout(io.StringIO()):
                doctor.check_models()

        self.assertEqual(doctor.results[-1][0:2], ("models", doctor.PASS))

    def test_model_check_rejects_stale_partial_and_undeclared_gguf(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models = root / "models"
            results = root / "results"
            models.mkdir()
            results.mkdir()
            (results / "model_manifest.json").write_text(json.dumps({
                "declared.gguf": {"size_bytes": 3, "split": "train"},
            }), encoding="utf-8")
            (models / "declared.gguf").write_bytes(b"abc")
            (models / "declared.gguf.part").write_bytes(b"x")
            (models / "rogue.gguf").write_bytes(b"x")

            with patch.object(doctor, "MODELS_DIR", models), \
                    patch.object(doctor, "RESULTS_DIR", results), \
                    redirect_stdout(io.StringIO()):
                doctor.check_models()

        name, status, detail = doctor.results[-1]
        self.assertEqual((name, status), ("models", doctor.FAIL))
        self.assertIn("declared.gguf.part", detail)
        self.assertIn("rogue.gguf", detail)

    def test_disk_requirement_uses_manifest_and_resumable_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models = root / "models"
            results = root / "results"
            models.mkdir()
            results.mkdir()
            manifest = {
                "complete.gguf": {"size_bytes": 100, "split": "train"},
                "pending.gguf": {"size_bytes": 200, "split": "test"},
            }
            (results / "model_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8")
            (models / "complete.gguf").write_bytes(b"x" * 100)
            (models / "pending.gguf.part").write_bytes(b"x" * 50)

            with patch.object(doctor, "MODELS_DIR", models), \
                    patch.object(doctor, "RESULTS_DIR", results), \
                    patch.object(doctor.shutil, "disk_usage",
                                 return_value=SimpleNamespace(free=149)), \
                    redirect_stdout(io.StringIO()):
                doctor.check_disk()
            self.assertEqual(doctor.results[-1][0:2], ("disk", doctor.FAIL))

            doctor.results.clear()
            with patch.object(doctor, "MODELS_DIR", models), \
                    patch.object(doctor, "RESULTS_DIR", results), \
                    patch.object(doctor.shutil, "disk_usage",
                                 return_value=SimpleNamespace(free=150)), \
                    redirect_stdout(io.StringIO()):
                doctor.check_disk()
            self.assertEqual(doctor.results[-1][0:2], ("disk", doctor.PASS))

    def test_disk_does_not_count_wrong_final_or_oversized_partial(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models = root / "models"
            results = root / "results"
            models.mkdir()
            results.mkdir()
            (results / "model_manifest.json").write_text(json.dumps({
                "model.gguf": {"size_bytes": 100, "split": "train"},
            }), encoding="utf-8")
            (models / "model.gguf").write_bytes(b"x" * 99)
            (models / "model.gguf.part").write_bytes(b"x" * 101)

            with patch.object(doctor, "MODELS_DIR", models), \
                    patch.object(doctor, "RESULTS_DIR", results), \
                    patch.object(doctor.shutil, "disk_usage",
                                 return_value=SimpleNamespace(free=99)), \
                    redirect_stdout(io.StringIO()):
                doctor.check_disk()

        self.assertEqual(doctor.results[-1][0:2], ("disk", doctor.FAIL))

    def test_repository_manifest_has_expected_exact_total(self) -> None:
        sizes = doctor._declared_model_sizes()
        self.assertEqual(len(sizes), 23)
        self.assertEqual(sum(sizes.values()), 336_367_242_336)

    def test_quick_mode_never_runs_live_benchmark(self) -> None:
        checks = (
            "check_python", "check_deps", "check_selfchecks", "check_threads",
            "check_load_average", "check_disk", "check_network", "check_models",
        )
        patches = [patch.object(doctor, name) for name in checks]
        started = [item.start() for item in patches]
        self.addCleanup(lambda: [item.stop() for item in reversed(patches)])
        with patch.object(doctor, "check_llama_bench", return_value=Path("bench")), \
                patch.object(doctor, "check_live_run") as live, \
                redirect_stdout(io.StringIO()):
            status = doctor.main(["--quick"])

        self.assertEqual(status, 0)
        live.assert_not_called()
        self.assertEqual(len(started), len(checks))


if __name__ == "__main__":
    unittest.main()
