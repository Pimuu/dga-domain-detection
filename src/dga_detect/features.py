"""Lexical feature extraction for domain names.

All features are computed from the **second-level label only** (``google`` in
``google.com``) - no DNS, WHOIS, traffic or TLD information.  This is the
cheapest possible signal: a resolver or proxy can score a name before any
packet reaches the C2 server.

The n-gram "rarity" features need a reference model of what benign names look
like.  ``DomainFeatureExtractor.fit`` builds character bigram/trigram
frequency tables from the *benign rows of the training data only* (it is a
supervised transformer, so it receives ``y`` inside a scikit-learn
``Pipeline``).  Fitting inside the pipeline guarantees the test split and
cross-validation folds never contribute to the reference model.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

VOWELS = set("aeiou")
_CONSONANT_RUN = re.compile(r"[bcdfghjklmnpqrstvwxyz]+")
_DIGIT_RUN = re.compile(r"[0-9]+")
ALPHABET_SIZE = 26 + 10 + 2 + 1  # letters, digits, '-', '_' (rare) + boundary

FEATURE_NAMES = [
    "length",               # number of characters in the label
    "entropy",              # Shannon entropy of the character distribution (bits)
    "vowel_ratio",          # vowels / letters
    "vowel_consonant_ratio",  # vowels / (consonants + 1)
    "digit_ratio",          # digits / length
    "unique_char_ratio",    # distinct characters / length
    "max_consonant_run",    # longest run of consecutive consonants
    "consonant_run_frac",   # longest consonant run / length
    "max_digit_run",        # longest run of consecutive digits
    "hyphen_count",         # number of '-' characters
    "bigram_logprob",       # mean log P(c_i | c_{i-1}) under the benign model
    "trigram_logprob",      # mean log P(c_i | c_{i-2} c_{i-1}) under the benign model
    "rare_bigram_frac",     # fraction of bigrams seen < 3 times in benign data
]

# Features that depend on the fitted benign character n-gram model.
NGRAM_FEATURES = ["bigram_logprob", "trigram_logprob", "rare_bigram_frac"]


def to_sld(domain: str) -> str:
    """Return the lowercase second-level label (text before the first dot)."""
    return str(domain).strip().lower().split(".")[0]


def shannon_entropy(s: str) -> float:
    """Shannon entropy H = -sum p(c) log2 p(c) over characters of ``s``."""
    if not s:
        return 0.0
    n = len(s)
    return -sum((k / n) * math.log2(k / n) for k in Counter(s).values())


def _ngrams(s: str, n: int) -> list[str]:
    padded = "^" * (n - 1) + s + "$"
    return [padded[i:i + n] for i in range(len(padded) - n + 1)]


class DomainFeatureExtractor(BaseEstimator, TransformerMixin):
    """scikit-learn transformer: iterable of domains -> numeric feature matrix.

    Parameters
    ----------
    features : list[str] | None
        Subset of ``FEATURE_NAMES`` to output (default: all).  Used to build
        single-feature baselines such as "entropy only".
    rare_threshold : int
        A bigram observed fewer than this many times in benign training data
        counts as "rare".
    """

    def __init__(self, features: list[str] | None = None, rare_threshold: int = 3):
        self.features = features
        self.rare_threshold = rare_threshold

    # ---- fitting: benign character n-gram reference model ----------------- #
    def fit(self, X, y=None):
        slds = [to_sld(d) for d in X]
        if y is not None:  # use benign rows only (label 0)
            slds = [s for s, lab in zip(slds, np.asarray(y)) if lab == 0]
        self.bigrams_ = Counter(g for s in slds for g in _ngrams(s, 2))
        self.trigrams_ = Counter(g for s in slds for g in _ngrams(s, 3))
        # context counts for conditional probabilities
        self.unigram_ctx_ = Counter(g for s in slds for g in _ngrams(s, 1))
        self.unigram_ctx_["^"] = len(slds)
        self.bigram_ctx_ = Counter()
        for g, c in self.bigrams_.items():
            self.bigram_ctx_[g] += c
        self.bigram_ctx_["^^"] = len(slds)
        self.feature_names_out_ = list(self.features or FEATURE_NAMES)
        return self

    def _cond_logprob(self, s: str) -> tuple[float, float, float]:
        """Mean add-one-smoothed log-probabilities and rare-bigram fraction."""
        V = ALPHABET_SIZE
        bgs = _ngrams(s, 2)
        lp2 = [math.log((self.bigrams_.get(g, 0) + 1) /
                        (self.unigram_ctx_.get(g[0], 0) + V)) for g in bgs]
        tgs = _ngrams(s, 3)
        lp3 = [math.log((self.trigrams_.get(g, 0) + 1) /
                        (self.bigram_ctx_.get(g[:2], 0) + V)) for g in tgs]
        rare = sum(self.bigrams_.get(g, 0) < self.rare_threshold for g in bgs)
        return float(np.mean(lp2)), float(np.mean(lp3)), rare / len(bgs)

    # ---- per-domain feature vector ---------------------------------------- #
    def _row(self, domain: str) -> dict[str, float]:
        s = to_sld(domain)
        n = max(len(s), 1)
        letters = [c for c in s if c.isalpha()]
        n_vow = sum(c in VOWELS for c in letters)
        n_con = len(letters) - n_vow
        con_runs = [len(m) for m in _CONSONANT_RUN.findall(s)] or [0]
        dig_runs = [len(m) for m in _DIGIT_RUN.findall(s)] or [0]
        lp2, lp3, rare = self._cond_logprob(s)
        return {
            "length": len(s),
            "entropy": shannon_entropy(s),
            "vowel_ratio": n_vow / max(len(letters), 1),
            "vowel_consonant_ratio": n_vow / (n_con + 1),
            "digit_ratio": sum(c.isdigit() for c in s) / n,
            "unique_char_ratio": len(set(s)) / n,
            "max_consonant_run": max(con_runs),
            "consonant_run_frac": max(con_runs) / n,
            "max_digit_run": max(dig_runs),
            "hyphen_count": s.count("-"),
            "bigram_logprob": lp2,
            "trigram_logprob": lp3,
            "rare_bigram_frac": rare,
        }

    def transform(self, X) -> np.ndarray:
        names = self.feature_names_out_
        return np.array([[r[f] for f in names] for r in map(self._row, X)],
                        dtype=float)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_out_)
