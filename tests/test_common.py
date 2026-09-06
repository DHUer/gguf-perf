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


class GitProvenanceTests(unittest.TestCase):
    def test_source_archive_does_not_invoke_git(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(common, "ROOT", Path(tmp)), \
                patch.object(common.subprocess, "run") as run:
            self.assertEqual(common.git_commit(), "nogit")
            run.assert_not_called()

    def test_worktree_git_file_still_uses_git(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".git").write_text(
                "gitdir: /tmp/example-worktree-metadata\n", encoding="utf-8")
            completed = common.subprocess.CompletedProcess(
                args=[], returncode=0, stdout="abc1234\n")
            with patch.object(common, "ROOT", root), \
                    patch.object(common.subprocess, "run",
                                 return_value=completed) as run:
                self.assertEqual(common.git_commit(), "abc1234")
                run.assert_called_once()


if __name__ == "__main__":
    unittest.main()
