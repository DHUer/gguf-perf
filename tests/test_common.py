import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from llmperf import common


class RunnerDiscoveryTests(unittest.TestCase):
    def test_windows_managed_runner_survives_a_fresh_shell(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runner = (Path(tmp) / "gguf-perf" / "llama-b10794" /
                      "llama-bench.exe")
            runner.parent.mkdir(parents=True)
            runner.touch()

            with patch.dict(common.os.environ,
                            {"LOCALAPPDATA": tmp}, clear=False), \
                    patch.object(common.platform, "system",
                                 return_value="Windows"), \
                    patch.object(common.shutil, "which", return_value=None):
                self.assertEqual(common.find_llama_bench(), runner)


if __name__ == "__main__":
    unittest.main()
