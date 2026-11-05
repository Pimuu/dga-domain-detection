"""End-to-end experiment: data -> features -> models -> metrics -> figures.

Run from the repository root:

    python run_experiment.py            # full run (~1 min on a laptop)
    python run_experiment.py --n 5000   # smaller, faster smoke run

Every random choice is seeded (``--seed``, default 42), so two runs with the
same arguments produce identical CSVs.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, train_test_split

from . import plots
from .data import HELDOUT_DGA_FAMILIES, build_dataset
from .evaluate import (classification_metrics, detection_by_family,
                       threshold_at_fpr, tpr_at_fpr)
from .features import FEATURE_NAMES, NGRAM_FEATURES, shannon_entropy
from .models import build_models, random_forest

ROOT = Path(__file__).resolve().parents[2]


def _log(msg: str) -> None:
    print(f"[dga] {msg}", flush=True)


def run(n_per_class: int = 20000, seed: int = 42, test_size: float = 0.2,
        cv_folds: int = 5, out_dir: Path | None = None,
        data_dir: Path | None = None) -> dict:
    t_start = time.perf_counter()
    out_dir = Path(out_dir or ROOT / "results")
    data_dir = Path(data_dir or ROOT / "data")
    fig_dir = out_dir / "figures"
    for d in (out_dir, fig_dir, data_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 1. Data ---------------------------------------------------------------
    _log(f"generating synthetic dataset ({n_per_class} per class, seed={seed})")
    main, held = build_dataset(n_per_class=n_per_class,
                               n_heldout_per_family=max(n_per_class // 10, 200),
                               seed=seed)
    main.to_csv(data_dir / "domains.csv", index=False)
    held.to_csv(data_dir / "heldout_dga.csv", index=False)
    _log(f"main set: {len(main)} rows "
         f"({(main.label == 0).sum()} legit / {(main.label == 1).sum()} dga); "
         f"held-out DGA: {len(held)} rows")

    # 2. Stratified train / test split (stratified by family) --------------
    train, test = train_test_split(main, test_size=test_size, random_state=seed,
                                   stratify=main["family"])
    X_tr, y_tr = train["domain"].to_numpy(), train["label"].to_numpy()
    X_te, y_te = test["domain"].to_numpy(), test["label"].to_numpy()

    models = build_models(seed)
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    cv_rows, test_rows, test_scores, oof_scores, fitted = [], [], {}, {}, {}
    for name, model in models.items():
        # 3. Cross-validation on the training split (out-of-fold scores are
        #    also used to calibrate operating thresholds without touching test)
        oof = np.zeros(len(y_tr))
        for fold, (a, b) in enumerate(skf.split(X_tr, train["family"])):
            m = clone(model).fit(X_tr[a], y_tr[a])
            oof[b] = m.predict_proba(X_tr[b])[:, 1]
            cv_rows.append({"model": name, "fold": fold,
                            **classification_metrics(y_tr[b], oof[b])})
        oof_scores[name] = oof
        thr_1 = threshold_at_fpr(y_tr, oof, 0.01)

        # 4. Fit on the full training split, evaluate once on test ----------
        t0 = time.perf_counter()
        m = clone(model).fit(X_tr, y_tr)
        fit_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        s = m.predict_proba(X_te)[:, 1]
        pred_s = time.perf_counter() - t0
        fitted[name], test_scores[name] = m, s
        at1 = classification_metrics(y_te, s, thr_1)
        test_rows.append({
            "model": name, **classification_metrics(y_te, s),
            "tpr_at_fpr_1pct_test": tpr_at_fpr(y_te, s, 0.01),
            "tpr_at_fpr_0.1pct_test": tpr_at_fpr(y_te, s, 0.001),
            "thr_cv_fpr_1pct": thr_1, "recall_at_cv_thr": at1["recall"],
            "fpr_at_cv_thr": at1["fpr"],
            "fit_seconds": fit_s,
            "us_per_domain": 1e6 * pred_s / len(X_te),
        })
        _log(f"{name:20s} test AUC={test_rows[-1]['roc_auc']:.4f} "
             f"F1={test_rows[-1]['f1']:.4f} (fit {fit_s:.1f}s)")

    metrics = pd.DataFrame(test_rows)
    metrics.to_csv(out_dir / "metrics.csv", index=False, float_format="%.6f")
    cv = pd.DataFrame(cv_rows)
    cv.to_csv(out_dir / "cv_folds.csv", index=False, float_format="%.6f")
    cv_summary = (cv.groupby("model", sort=False)[["accuracy", "precision", "recall",
                                                   "f1", "roc_auc"]]
                  .agg(["mean", "std"]))
    cv_summary.columns = [f"{a}_{b}" for a, b in cv_summary.columns]
    cv_summary.reset_index().to_csv(out_dir / "cv_summary.csv", index=False,
                                    float_format="%.6f")

    # Model selection uses CV ROC-AUC only (never the test split).
    best = cv_summary["roc_auc_mean"].idxmax()
    _log(f"selected model by CV ROC-AUC: {best}")
    bm, bs = fitted[best], test_scores[best]
    best_row = metrics.set_index("model").loc[best]

    # 5. Confusion matrix of the selected model ----------------------------
    cm = np.array([[best_row.tn, best_row.fp], [best_row.fn, best_row.tp]], dtype=int)
    pd.DataFrame(cm, index=["true_legit", "true_dga"],
                 columns=["pred_legit", "pred_dga"]).to_csv(out_dir / "confusion_matrix.csv")

    # 6. Feature importance (Random Forest: impurity + permutation on test) -
    rf = fitted["Random Forest"]
    F_te = rf.named_steps["features"].transform(X_te)
    perm = permutation_importance(rf.named_steps["clf"], F_te, y_te, scoring="roc_auc",
                                  n_repeats=10, random_state=seed, n_jobs=-1)
    imp = pd.DataFrame({
        "feature": FEATURE_NAMES,
        "impurity_importance": rf.named_steps["clf"].feature_importances_,
        "perm_importance_mean": perm.importances_mean,
        "perm_importance_std": perm.importances_std,
    }).sort_values("perm_importance_mean", ascending=False)
    imp.to_csv(out_dir / "feature_importance.csv", index=False, float_format="%.6f")
    lr = fitted["Logistic Regression"].named_steps["clf"]
    pd.DataFrame({"feature": FEATURE_NAMES, "std_coefficient": lr.coef_[0]}) \
        .sort_values("std_coefficient").to_csv(out_dir / "logreg_coefficients.csv",
                                               index=False, float_format="%.6f")

    # 7. Per-family detection incl. held-out (evasive) families ------------
    thr_1 = float(best_row.thr_cv_fpr_1pct)
    thresholds = {"0.5": 0.5, "FPR1%": thr_1}
    fam_test = detection_by_family(test["family"], bs, thresholds)
    fam_test["split"] = np.where(fam_test["family"] == "legit", "test-benign", "test")
    held_scores = bm.predict_proba(held["domain"].to_numpy())[:, 1]
    fam_held = detection_by_family(held["family"], held_scores, thresholds)
    fam_held["split"] = "held-out"
    fam = pd.concat([fam_test, fam_held], ignore_index=True)
    fam.insert(0, "model", best)
    fam.to_csv(out_dir / "family_detection.csv", index=False, float_format="%.6f")

    # 8. Error analysis: most confident mistakes on test --------------------
    err = test.assign(score=bs)
    fps = err[(err.label == 0) & (err.score >= 0.5)].nlargest(15, "score")
    fns = err[(err.label == 1) & (err.score < 0.5)].nsmallest(15, "score")
    pd.concat([fps.assign(error="false_positive"), fns.assign(error="false_negative")]) \
        .to_csv(out_dir / "error_examples.csv", index=False, float_format="%.4f")
    held_ex = held.assign(score=held_scores).groupby("family").head(5)
    held_ex.to_csv(out_dir / "heldout_examples.csv", index=False, float_format="%.4f")

    # 9. Feature-group ablation (Random Forest) ------------------------------
    #    Does the detector rely on the benign n-gram model or on simple
    #    structural statistics, and which generalises to unseen families?
    groups = {"all features": FEATURE_NAMES,
              "structural only (no n-gram)": [f for f in FEATURE_NAMES
                                              if f not in NGRAM_FEATURES],
              "n-gram only": NGRAM_FEATURES}
    abl_rows = []
    for gname, feats in groups.items():
        m = random_forest(seed, feats).fit(X_tr, y_tr)
        s = m.predict_proba(X_te)[:, 1]
        hs = m.predict_proba(held["domain"].to_numpy())[:, 1]
        row = {"feature_set": gname, "n_features": len(feats),
               **{k: v for k, v in classification_metrics(y_te, s).items()
                  if k in ("accuracy", "f1", "roc_auc", "fpr")}}
        for fam_name in HELDOUT_DGA_FAMILIES:
            row[f"heldout_{fam_name}_detect@0.5"] = float(
                (hs[held["family"].to_numpy() == fam_name] >= 0.5).mean())
        abl_rows.append(row)
    ablation = pd.DataFrame(abl_rows)
    ablation.to_csv(out_dir / "ablation.csv", index=False, float_format="%.6f")

    # 10. Figures -----------------------------------------------------------
    plots.plot_roc(y_te, test_scores, metrics.set_index("model")["roc_auc"].to_dict(),
                   fig_dir / "roc_curves.png")
    plots.plot_confusion(cm, best, fig_dir / "confusion_matrix.png")
    plots.plot_feature_importance(imp, fig_dir / "feature_importance.png")
    ent = main["sld"].map(shannon_entropy)
    plots.plot_entropy_distribution(ent[main.label == 0].to_numpy(),
                                    ent[main.label == 1].to_numpy(),
                                    fig_dir / "entropy_distribution.png")
    plots.plot_family_detection(fam, fig_dir / "family_detection.png", best)

    # 11. Summary -----------------------------------------------------------
    runtime = time.perf_counter() - t_start
    summary = {
        "seed": seed, "n_per_class": n_per_class, "test_size": test_size,
        "cv_folds": cv_folds, "n_train": int(len(train)), "n_test": int(len(test)),
        "n_heldout": int(len(held)), "heldout_families": HELDOUT_DGA_FAMILIES,
        "selected_model": best,
        "test_metrics": metrics.set_index("model").round(6).to_dict(orient="index"),
        "family_detection": fam.round(6).to_dict(orient="records"),
        "ablation": ablation.round(6).to_dict(orient="records"),
        "entropy_median": {"legit": float(ent[main.label == 0].median()),
                           "dga": float(ent[main.label == 1].median())},
        "runtime_seconds": round(runtime, 1),
        "environment": {"python": platform.python_version(),
                        "numpy": np.__version__, "pandas": pd.__version__,
                        "scikit-learn": sklearn.__version__,
                        "matplotlib": matplotlib.__version__},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    _log(f"done in {runtime:.1f}s - results in {out_dir}")
    return summary


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=20000, help="domains per class")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--cv", type=int, default=5, help="cross-validation folds")
    p.add_argument("--out", type=Path, default=None, help="results directory")
    p.add_argument("--data-dir", type=Path, default=None, help="dataset directory")
    a = p.parse_args(argv)
    run(a.n, a.seed, a.test_size, a.cv, a.out, a.data_dir)


if __name__ == "__main__":
    main()
