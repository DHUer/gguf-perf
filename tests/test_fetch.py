import unittest

from llmperf.fetch import PILOT


class PilotCohortTests(unittest.TestCase):
    def test_pilot_contains_training_and_held_out_models(self):
        splits = [split for _repo, _pattern, split in PILOT]
        self.assertEqual(splits.count("train"), 4)
        self.assertEqual(splits.count("test"), 2)

    def test_pilot_held_out_models_use_the_fitted_q4_format(self):
        test_patterns = [pattern for _repo, pattern, split in PILOT
                         if split == "test"]
        self.assertTrue(test_patterns)
        self.assertTrue(all("Q4_K_M" in pattern for pattern in test_patterns))


if __name__ == "__main__":
    unittest.main()
