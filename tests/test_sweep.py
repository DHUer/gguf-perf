import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from llmperf import sweep


class LoadAverageRegressionTests(unittest.TestCase):
    def test_windows_uses_sampled_busy_cpu_count(self) -> None:
        cpu_percent = Mock(return_value=37.5)
        cpu_count = Mock(return_value=8)
        getloadavg = Mock(return_value=(0.0, 0.0, 0.0))
        psutil = SimpleNamespace(
            cpu_percent=cpu_percent,
            cpu_count=cpu_count,
            getloadavg=getloadavg,
        )

        with patch.dict(sys.modules, {"psutil": psutil}), \
                patch.object(sweep.platform, "system", return_value="Windows"), \
                patch.object(sweep.os, "cpu_count", return_value=16):
            load = sweep.load_average()

        self.assertEqual(load, 3.0)
        cpu_percent.assert_called_once_with(interval=sweep._WINDOWS_LOAD_SAMPLE_S)
        cpu_count.assert_called_once_with(logical=True)
        getloadavg.assert_not_called()

    def test_posix_uses_one_minute_load_average(self) -> None:
        cpu_percent = Mock(return_value=50.0)
        getloadavg = Mock(return_value=(7.5, 6.0, 5.0))
        psutil = SimpleNamespace(
            cpu_percent=cpu_percent,
            cpu_count=Mock(return_value=8),
            getloadavg=getloadavg,
        )

        with patch.dict(sys.modules, {"psutil": psutil}), \
                patch.object(sweep.platform, "system", return_value="Linux"):
            load = sweep.load_average()

        self.assertEqual(load, 7.5)
        getloadavg.assert_called_once_with()
        cpu_percent.assert_not_called()


if __name__ == "__main__":
    unittest.main()
