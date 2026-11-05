"""Model zoo: every model is a full scikit-learn ``Pipeline`` whose first step
is the lexical feature extractor, so ``model.fit(domains, labels)`` and
``model.predict_proba(domains)`` operate directly on raw domain strings.
"""

from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import DomainFeatureExtractor


def random_forest(seed: int = 42, features: list[str] | None = None) -> Pipeline:
    """Random Forest pipeline, optionally restricted to a feature subset."""
    return Pipeline([
        ("features", DomainFeatureExtractor(features=features)),
        ("clf", RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                       n_jobs=-1, random_state=seed)),
    ])


def build_models(seed: int = 42) -> dict[str, Pipeline]:
    """Return the models compared in the paper, keyed by display name.

    * Entropy baseline - logistic regression on Shannon entropy alone, the
      classic "high-entropy name" heuristic used by many SOC rules.
    * Logistic Regression - linear model on all standardised features.
    * Random Forest - bagged decision trees (non-linear, interpretable via
      impurity / permutation importance).
    * Gradient Boosting - histogram-based gradient-boosted trees.
    """
    return {
        "Entropy baseline": Pipeline([
            ("features", DomainFeatureExtractor(features=["entropy"])),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(random_state=seed)),
        ]),
        "Logistic Regression": Pipeline([
            ("features", DomainFeatureExtractor()),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=2000, random_state=seed)),
        ]),
        "Random Forest": random_forest(seed),
        "Gradient Boosting": Pipeline([
            ("features", DomainFeatureExtractor()),
            ("clf", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1,
                                                   random_state=seed)),
        ]),
    }
