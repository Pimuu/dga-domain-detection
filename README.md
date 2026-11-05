# DGA Domain Detection with Lexical Features

Classifying domain names as **algorithmically generated (DGA, malware C2)** or
**legitimate** from the characters of the name alone, using classical
scikit-learn models. Everything (the dataset, features, models, metrics,
figures and paper) is reproduced by one command in under a minute on a laptop
CPU. No deep learning and no downloads.

| Model (test split, n = 8,000) | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Entropy baseline | 0.7291 | 0.7251 | 0.7380 | 0.7315 | 0.8008 |
| Logistic Regression | 0.9735 | 0.9660 | 0.9815 | 0.9737 | 0.9970 |
| Random Forest | **0.9822** | **0.9784** | 0.9862 | **0.9823** | 0.9988 |
| Gradient Boosting (selected by CV) | 0.9820 | 0.9756 | **0.9888** | 0.9821 | **0.9990** |

The main caveat: dictionary-word DGAs (a family held out from training) are
detected only **33%** of the time, which is the evasion problem the paper
discusses. Full write-up: [`paper/paper.md`](paper/paper.md).

---

## Quick start

```bash
pip install -r requirements.txt
```

```bash
python run_experiment.py
```

```bash
python -m unittest discover -s tests -v
```

`run_experiment.py` regenerates `data/`, every CSV in `results/` and every PNG
in `results/figures/`. Options: `--n` (domains per class, default 20000),
`--seed` (default 42), `--cv` (folds, default 5), `--test-size` (default 0.2),
`--out`, `--data-dir`.

**Reproducibility check.** A full run took 46–49 s (4-core cloud container,
Python 3.13, scikit-learn 1.9.1). Two consecutive runs gave byte-identical
dataset, CV, confusion-matrix, feature-importance, family-detection and
ablation CSVs, and identical metrics apart from the wall-clock timing columns
(`fit_seconds`, `us_per_domain`).

## Repository layout

```
run_experiment.py          entry point (adds src/ to the path, calls pipeline.main)
src/dga_detect/
  wordlists.py             embedded vocabularies (benign words; disjoint DGA words)
  data.py                  synthetic benign generator + date-seeded DGA families
  features.py              DomainFeatureExtractor (sklearn transformer, 13 features)
  models.py                Entropy baseline, Logistic Regression, Random Forest, Gradient Boosting
  evaluate.py              metrics, threshold at fixed FPR, per-family detection
  plots.py                 ROC, confusion matrix, feature importance, entropy, family charts
  pipeline.py              orchestration: data -> split -> CV -> test -> analysis -> figures
tests/test_core.py         unit tests (features, determinism, dataset invariants)
data/                      generated dataset (domains.csv, heldout_dga.csv)
results/                   metrics CSVs, summary.json, figures/
paper/paper.md             the research paper
```

## Outputs (`results/`)

| File | Contents |
|---|---|
| `metrics.csv` | Test-split accuracy, precision, recall, F1, ROC-AUC, FPR, TN/FP/FN/TP, TPR at 1% and 0.1% FPR, recall/FPR at a CV-calibrated 1%-FPR threshold, fit time, µs per domain |
| `cv_summary.csv`, `cv_folds.csv` | 5-fold stratified CV on the training split (mean ± std, and per fold) |
| `confusion_matrix.csv` | Confusion matrix of the selected model |
| `feature_importance.csv` | Random Forest impurity and permutation importance |
| `logreg_coefficients.csv` | Standardised logistic-regression coefficients |
| `family_detection.csv` | Detection rate per DGA family, including the two held-out families |
| `ablation.csv` | Random Forest trained on all / structural-only / n-gram-only features |
| `error_examples.csv`, `heldout_examples.csv` | Most confident mistakes; held-out samples with scores |
| `summary.json` | All of the above plus configuration and library versions |
| `figures/roc_curves.png` | ROC curves of all models, with an inset for FPR ≤ 5% |
| `figures/confusion_matrix.png` | Confusion-matrix heatmap (selected model) |
| `figures/feature_importance.png` | Impurity and permutation importance |
| `figures/entropy_distribution.png` | Shannon-entropy histogram by class |
| `figures/family_detection.png` | Per-family detection rate at two operating points |

---

## Design

### Dataset (synthetic, documented in `src/dga_detect/data.py`)

* **Legitimate (20,000).** Human-style second-level labels built from an
  embedded 400-word vocabulary: single words, two- or three-word compounds,
  prefix and suffix brands (`getcloud`, `shopify`-style), hyphenated phrases,
  invented pronounceable brands (`zolami`), word+number (`web365`), and
  vowel-dropped brands (`flickr`-style). There are also **random 2–5 letter
  acronyms**, which are deliberate hard negatives because they look like noise.
* **DGA, training families (20,000; 6 families).** Every family is a
  deterministic function of *(family, date, salt)* driven by a 32-bit LCG
  seeded from the date. Bot and botmaster use the same kind of function to
  compute the day's rendezvous list. Families: `short_random` (5–8 chars),
  `conficker_like` (8–11), `cryptolocker_like` (12–18), `uniform_long`
  (19–28), `alnum_random` (letters and digits, 10–20), `hex_hash` (MD5 prefix,
  16–32). They are *inspired by* the named malware, not re-implementations of it.
* **DGA, held-out families (4,000), never used in training.**
  `wordlist_dga` concatenates 2–3 words from a vocabulary that shares no words
  with the benign one (Suppobox/Matsnu-style). `pronounceable_dga` alternates
  consonants and vowels (Symmi-style). These measure how the detector holds up
  against generators designed to look human.
* TLDs are generated for realism, but **only the second-level label is
  featurised**, so the synthetic TLD mix cannot leak the label.

### Features (`src/dga_detect/features.py`)

`length`, `entropy` (Shannon), `vowel_ratio`, `vowel_consonant_ratio`,
`digit_ratio`, `unique_char_ratio`, `max_consonant_run`, `consonant_run_frac`,
`max_digit_run`, `hyphen_count`, plus three **n-gram rarity** features:
`bigram_logprob` and `trigram_logprob` (mean add-one-smoothed conditional
log-probability of each character under a character model of *benign*
names), and `rare_bigram_frac`.

The n-gram reference model is fitted **inside the scikit-learn `Pipeline`
on the benign rows of the training fold only**. Test data and other CV folds
never contribute to it, so it cannot leak.

### Protocol

Stratified (by family) 80/20 train/test split. 5-fold stratified CV on the
training split. The model is **selected by CV ROC-AUC, never on test**, and
operating thresholds (1% FPR) are calibrated on the **out-of-fold CV scores**,
then applied unchanged to test and to the held-out families.

## Threat model

* **Setting.** A defender watches DNS queries leaving an enterprise network
  (resolver logs, a DNS firewall, or a proxy) and wants to flag queries for
  names that infected hosts generate to find their command-and-control (C2)
  server. Lexical scoring needs no WHOIS, no passive-DNS history and no
  response data, so it can run inline before resolution.
* **Adversary.** Malware authors who control the DGA. A naive adversary uses
  a uniform-random or hash-based generator (the training families). An
  **adaptive adversary** knows that lexical detectors exist and generates
  names that look benign, for example by concatenating dictionary words or
  producing pronounceable strings, or by training a generator against the
  detector (DeepDGA). The adversary does not have query access to our
  particular model in our evaluation, but they can imitate benign lexical
  statistics.
* **What the defender wins.** High-recall, low-cost triage of high-entropy
  DGA traffic (≥ 98.6% detection at 1% FPR for all training families except
  short names).
* **What the defender does not win.** Wordlist DGAs largely evade (17–33%
  detected), and short random names collide with real acronyms. Lexical
  scores should therefore be one signal among several (NXDOMAIN bursts, query
  timing, domain age, passive DNS), not the only gate. See the paper's
  *Discussion & Limitations*.
* **Out of scope.** The dataset is synthetic. Absolute numbers on real
  traffic (e.g. Tranco vs. DGArchive) will differ, and the paper says so.

## License

MIT (see `LICENSE`). Synthetic data only; no real malware code or live
indicators are included.
