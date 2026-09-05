import io
import sys
import unittest
from contextlib import redirect_stdout
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


if __name__ == "__main__":
    unittest.main()
