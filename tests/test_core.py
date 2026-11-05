"""Unit tests (stdlib unittest - no extra dependencies).

    python -m unittest discover -s tests -v
"""
import datetime as dt
import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dga_detect.data import build_dataset, dga_for_day  # noqa: E402
from dga_detect.features import (FEATURE_NAMES, DomainFeatureExtractor,  # noqa: E402
                                 shannon_entropy, to_sld)
from dga_detect.wordlists import DGA_WORDS, LEGIT_WORDS  # noqa: E402


class TestFeatures(unittest.TestCase):
    def test_entropy(self):
        self.assertEqual(shannon_entropy("aaaa"), 0.0)
        self.assertAlmostEqual(shannon_entropy("abcd"), 2.0)
        self.assertAlmostEqual(shannon_entropy("aabb"), 1.0)
        self.assertEqual(shannon_entropy(""), 0.0)

    def test_sld(self):
        self.assertEqual(to_sld("Google.COM"), "google")
        self.assertEqual(to_sld("my-site.co.uk"), "my-site")

    def test_structural_values(self):
        ext = DomainFeatureExtractor().fit(["google.com", "facebook.com"], [0, 0])
        row = dict(zip(FEATURE_NAMES, ext.transform(["strngth42.com"])[0]))
        self.assertEqual(row["length"], 9)
        self.assertEqual(row["max_consonant_run"], 7)   # 'strngth'
        self.assertEqual(row["max_digit_run"], 2)
        self.assertAlmostEqual(row["digit_ratio"], 2 / 9)
        self.assertEqual(row["vowel_ratio"], 0.0)

    def test_ngram_model_prefers_benign_looking(self):
        benign = ["cloudshop", "bestbank", "shopcloud", "greenbank", "cloudbank"] * 5
        ext = DomainFeatureExtractor(features=["trigram_logprob"]).fit(benign, [0] * 25)
        lp_benign, lp_random = ext.transform(["bankshop.com", "xqzvkw.com"])[:, 0]
        self.assertGreater(lp_benign, lp_random)

    def test_fit_ignores_dga_rows(self):
        ext = DomainFeatureExtractor().fit(["abc", "xyz"], [0, 1])
        self.assertNotIn("xy", ext.bigrams_)
        self.assertIn("ab", ext.bigrams_)

    def test_feature_subset(self):
        ext = DomainFeatureExtractor(features=["entropy", "length"]).fit(["a"])
        self.assertEqual(ext.transform(["ab.com"]).shape, (1, 2))


class TestData(unittest.TestCase):
    def test_dga_deterministic(self):
        day = dt.date(2026, 3, 1)
        a = dga_for_day("conficker_like", day, 10)
        self.assertEqual(a, dga_for_day("conficker_like", day, 10))
        self.assertNotEqual(a, dga_for_day("conficker_like", day + dt.timedelta(1), 10))
        self.assertTrue(all(8 <= len(x) <= 11 for x in a))

    def test_vocabularies_disjoint(self):
        self.assertFalse(set(LEGIT_WORDS) & set(DGA_WORDS))

    def test_dataset(self):
        main, held = build_dataset(n_per_class=500, n_heldout_per_family=50, seed=1)
        self.assertEqual((main.label == 0).sum(), 500)
        self.assertGreater((main.label == 1).sum(), 490)
        self.assertFalse(main["sld"].duplicated().any())
        self.assertFalse(set(held["sld"]) & set(main["sld"]))
        self.assertTrue((held.label == 1).all())
        main2, _ = build_dataset(n_per_class=500, n_heldout_per_family=50, seed=1)
        self.assertTrue(main.equals(main2))


if __name__ == "__main__":
    unittest.main()
