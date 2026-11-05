"""Synthetic dataset generation for DGA detection.

Why synthetic?
    Real DGA feeds (e.g. DGArchive, Bambenek) and popularity lists (Tranco,
    Alexa) require downloads and sometimes registration.  This project is
    designed to be fully offline and reproducible in seconds, so we generate
    domains whose *lexical statistics* mimic the two populations.  The paper
    discusses the consequences of this choice in its Limitations section.

Benign ("legit", label 0) generators
    Human-chosen names: dictionary words, word compounds, brand-like affixes,
    hyphenated phrases, pronounceable invented brand names, short acronyms and
    word+number combinations.  Acronyms and vowel-dropped brands are deliberate
    *hard negatives*: they look random to a lexical detector.

Malicious ("dga", label 1) generators - TRAINING families
    Each family is a deterministic, date-seeded generator, mirroring how real
    malware and its operator compute the same rendezvous list for a given day:
    ``domains = f(seed, date)``.  Families are *inspired by* (not exact
    re-implementations of) well-known malware DGAs:

    ===================  =======================================  ==========
    family               construction                              length
    ===================  =======================================  ==========
    short_random         LCG over a-z                              5-8
    conficker_like       LCG over a-z                              8-11
    cryptolocker_like    LCG over a-z                              12-18
    uniform_long         LCG over a-z                              19-28
    alnum_random         LCG over a-z0-9                           10-20
    hex_hash             MD5(family|date|index) hex prefix         16-32
    ===================  =======================================  ==========

Malicious - HELD-OUT families (never seen in training; used to probe evasion)
    ``wordlist_dga``       2-3 words from a vocabulary disjoint from the benign
                           one (Suppobox / Matsnu style).
    ``pronounceable_dga``  alternating consonant/vowel strings (Symmi-style).

The top-level domain is generated for realism but is **not** used as a
feature (only the second-level label is), so the synthetic TLD mix cannot
leak the label.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import string

import numpy as np
import pandas as pd

from .wordlists import DGA_WORDS, LEGIT_PREFIXES, LEGIT_SUFFIXES, LEGIT_WORDS

VOWELS = "aeiou"
CONSONANTS = "".join(c for c in string.ascii_lowercase if c not in VOWELS)

LEGIT_TLDS = ["com"] * 10 + ["net", "org", "io", "co", "de", "uk", "fr", "app", "dev"]
DGA_TLDS = ["com"] * 4 + ["net", "org", "info", "biz", "ru", "top", "xyz", "cc"]

TRAIN_DGA_FAMILIES = ["short_random", "conficker_like", "cryptolocker_like",
                      "uniform_long", "alnum_random", "hex_hash"]
HELDOUT_DGA_FAMILIES = ["wordlist_dga", "pronounceable_dga"]


# --------------------------------------------------------------------------- #
# Benign generators
# --------------------------------------------------------------------------- #
def _pronounceable_brand(rng: np.random.Generator) -> str:
    """Invented brand name from 2-4 natural syllables, e.g. 'zolami', 'kevora'."""
    onsets = ["b", "c", "d", "f", "g", "k", "l", "m", "n", "p", "r", "s", "t",
              "v", "z", "br", "cl", "st", "tr", "pl", "gr", "sh", "ch"]
    codas = ["", "", "", "n", "r", "s", "x", "l"]
    n_syll = rng.integers(2, 5)
    return "".join(rng.choice(onsets) + rng.choice(list(VOWELS)) + rng.choice(codas)
                   for _ in range(n_syll))


def _drop_vowels(word: str) -> str:
    """Brand-style vowel dropping ('flicker' -> 'flickr'): remove the last vowel."""
    for i in range(len(word) - 1, 0, -1):
        if word[i] in VOWELS:
            return word[:i] + word[i + 1:]
    return word


def generate_legit(n: int, rng: np.random.Generator) -> list[str]:
    """Generate ``n`` unique benign-looking second-level labels."""
    words = LEGIT_WORDS
    out: set[str] = set()
    styles = ["word", "compound", "affix", "hyphen", "brand", "acronym",
              "wordnum", "vowelless"]
    probs = [0.12, 0.30, 0.15, 0.08, 0.13, 0.08, 0.08, 0.06]
    while len(out) < n:
        style = rng.choice(styles, p=probs)
        w1, w2 = rng.choice(words), rng.choice(words)
        if style == "word":
            name = w1
        elif style == "compound":
            name = w1 + w2 + (rng.choice(words) if rng.random() < 0.15 else "")
        elif style == "affix":
            name = (rng.choice(LEGIT_PREFIXES) + w1) if rng.random() < 0.5 \
                else (w1 + rng.choice(LEGIT_SUFFIXES))
        elif style == "hyphen":
            name = f"{w1}-{w2}"
        elif style == "brand":
            name = _pronounceable_brand(rng)
        elif style == "acronym":           # hard negative: e.g. 'kfc', 'ibmx'
            name = "".join(rng.choice(list(string.ascii_lowercase),
                                      size=rng.integers(2, 6)))
        elif style == "wordnum":           # e.g. 'web365', '24news', 'mail2'
            num = str(rng.choice([1, 2, 3, 4, 7, 24, 360, 365, 99, 100, 247,
                                  2000, 2024, rng.integers(0, 1000)]))
            name = (w1 + num) if rng.random() < 0.6 else (num + w1)
        else:                              # 'vowelless' brand
            name = _drop_vowels(w1 + w2) if rng.random() < 0.5 else _drop_vowels(w1)
        out.add(name)
    return sorted(out)


# --------------------------------------------------------------------------- #
# DGA generators
# --------------------------------------------------------------------------- #
class _LCG:
    """Minimal 32-bit linear congruential generator (Numerical Recipes constants).

    Many real DGAs use exactly this kind of tiny PRNG seeded with the date so
    that bot and botmaster derive the same domain list independently.
    """

    def __init__(self, seed: int) -> None:
        self.state = seed & 0xFFFFFFFF

    def next(self) -> int:
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return self.state >> 8  # drop weak low bits


def _date_seed(family: str, day: _dt.date, salt: int) -> int:
    digest = hashlib.sha256(f"{family}|{day.isoformat()}|{salt}".encode()).digest()
    return int.from_bytes(digest[:4], "little")


def dga_for_day(family: str, day: _dt.date, count: int, salt: int = 0) -> list[str]:
    """Deterministically compute ``count`` domains of ``family`` for ``day``."""
    alpha = string.ascii_lowercase
    alnum = string.ascii_lowercase + string.digits
    lengths = {"short_random": (5, 8), "conficker_like": (8, 11),
               "cryptolocker_like": (12, 18), "uniform_long": (19, 28),
               "alnum_random": (10, 20), "hex_hash": (16, 32),
               "wordlist_dga": (2, 3), "pronounceable_dga": (8, 14)}
    lo, hi = lengths[family]
    prng = _LCG(_date_seed(family, day, salt))
    out = []
    for i in range(count):
        length = lo + prng.next() % (hi - lo + 1)
        if family == "hex_hash":
            h = hashlib.md5(f"{family}|{day}|{salt}|{i}".encode()).hexdigest()
            name = h[:length]
        elif family == "alnum_random":
            name = "".join(alnum[prng.next() % len(alnum)] for _ in range(length))
            if name[0].isdigit():                    # labels rarely start with digits
                name = alpha[prng.next() % 26] + name[1:]
        elif family == "wordlist_dga":
            name = "".join(DGA_WORDS[prng.next() % len(DGA_WORDS)] for _ in range(length))
        elif family == "pronounceable_dga":
            name = "".join((CONSONANTS if j % 2 == 0 else VOWELS)[
                prng.next() % (21 if j % 2 == 0 else 5)] for j in range(length))
        else:
            name = "".join(alpha[prng.next() % 26] for _ in range(length))
        out.append(name)
    return out


def generate_dga(families: list[str], n: int, start: _dt.date,
                 n_days: int = 365) -> pd.DataFrame:
    """Generate ``n`` unique DGA labels split evenly across ``families`` and days."""
    per_family = int(np.ceil(n / len(families)))
    per_day = int(np.ceil(per_family / n_days))
    rows = []
    for fam in families:
        names: dict[str, str] = {}
        salt = 0
        while len(names) < per_family:
            for d in range(n_days):
                day = start + _dt.timedelta(days=d)
                for name in dga_for_day(fam, day, per_day, salt):
                    names.setdefault(name, day.isoformat())
                if len(names) >= per_family:
                    break
            salt += 1
        for name, day in list(names.items())[:per_family]:
            rows.append((name, fam, day))
    df = pd.DataFrame(rows, columns=["sld", "family", "date"])
    return df.drop_duplicates("sld").head(n).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def build_dataset(n_per_class: int = 20000, n_heldout_per_family: int = 2000,
                  seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(main_df, heldout_df)``.

    ``main_df``    balanced benign vs. training-family DGA domains (label 0/1).
    ``heldout_df`` DGA domains from families never used for training (label 1).

    Columns: ``domain`` (with TLD), ``sld`` (feature input), ``label``,
    ``family`` ("legit" for benign rows).
    """
    rng = np.random.default_rng(seed)
    start = _dt.date(2026, 1, 1)

    legit = pd.DataFrame({"sld": generate_legit(n_per_class, rng)})
    legit["family"], legit["label"] = "legit", 0

    dga = generate_dga(TRAIN_DGA_FAMILIES, n_per_class, start)
    dga["label"] = 1
    # A name that is both benign and DGA (e.g. a short random string colliding
    # with an acronym) is ambiguous; drop it from the DGA side.
    dga = dga[~dga["sld"].isin(set(legit["sld"]))]

    main = pd.concat([legit, dga.drop(columns="date")], ignore_index=True)
    main["domain"] = main["sld"] + "." + np.where(
        main["label"] == 0, rng.choice(LEGIT_TLDS, len(main)),
        rng.choice(DGA_TLDS, len(main)))

    held = generate_dga(HELDOUT_DGA_FAMILIES, n_heldout_per_family * 2, start)
    held = held[~held["sld"].isin(set(main["sld"]))].drop(columns="date")
    held["label"] = 1
    held["domain"] = held["sld"] + "." + rng.choice(DGA_TLDS, len(held))

    cols = ["domain", "sld", "label", "family"]
    main = main[cols].sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return main, held[cols].reset_index(drop=True)
