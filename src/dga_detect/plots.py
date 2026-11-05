"""Matplotlib figures for the results section.

Colour roles are fixed and validated for colour-vision deficiency (blue /
orange / aqua categorical slots; a single-hue blue ramp for magnitudes):
benign = blue, DGA = orange everywhere, and model identity uses the same
slot order in every chart.  Every multi-series chart has a legend so colour
is never the only identifier.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
MODEL_COLORS = {"Entropy baseline": GRAY, "Logistic Regression": BLUE,
                "Random Forest": ORANGE, "Gradient Boosting": AQUA}
CLASS_COLORS = {"legit": BLUE, "dga": ORANGE}
BLUE_RAMP = LinearSegmentedColormap.from_list(
    "blue_seq", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "axes.edgecolor": GRID,
    "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "axes.titlecolor": INK, "axes.titleweight": "bold",
    "axes.titlesize": 12, "axes.titlelocation": "left", "font.size": 10,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.dpi": 100, "savefig.dpi": 200,
})


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_roc(y_true, scores: dict[str, np.ndarray], aucs: dict[str, float], path):
    """ROC curves for all models; inset zooms into the low-FPR region."""
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    axin = ax.inset_axes([0.42, 0.12, 0.54, 0.48])
    for name, s in scores.items():
        fpr, tpr, _ = roc_curve(y_true, s)
        kw = dict(color=MODEL_COLORS.get(name, INK), lw=2,
                  ls="--" if name == "Entropy baseline" else "-")
        ax.plot(fpr, tpr, label=f"{name} (AUC = {aucs[name]:.4f})", **kw)
        axin.plot(fpr, tpr, **kw)
    ax.plot([0, 1], [0, 1], color=GRID, lw=1, ls=":")
    axin.set_xlim(0, 0.05)
    axin.set_ylim(0.80, 1.0)
    axin.set_title("FPR ≤ 5 %", fontsize=9, color=INK_2, loc="left", fontweight="normal")
    axin.tick_params(labelsize=8)
    ax.set_xlabel("False-positive rate (benign flagged)")
    ax.set_ylabel("True-positive rate (DGA detected)")
    ax.set_title("ROC curves on the held-out test split")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.13), ncol=2, fontsize=9)
    _save(fig, path)


def plot_confusion(cm: np.ndarray, model_name: str, path):
    """Confusion-matrix heatmap with counts and row-normalised percentages."""
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    row_pct = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(row_pct, cmap=BLUE_RAMP, vmin=0, vmax=1)
    labels = ["Legit", "DGA"]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}\n{row_pct[i, j]:.2%}", ha="center",
                    va="center", fontsize=11,
                    color="white" if row_pct[i, j] > 0.5 else INK)
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.grid(False)
    ax.set_title(f"Confusion matrix - {model_name}")
    _save(fig, path)


def plot_feature_importance(imp_df, path):
    """Two panels sharing the feature order: impurity importance (MDI, from
    training) and permutation importance (mean drop in test ROC-AUC, +/- std).
    Permutation importance under-credits correlated features (shuffling one
    n-gram feature leaves the others intact), so both views are shown."""
    d = imp_df.sort_values("impurity_importance")
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 5.0), sharey=True)
    axes[0].barh(d["feature"], d["impurity_importance"], color=BLUE, height=0.65)
    axes[0].set_xlabel("Mean decrease in impurity (sums to 1)")
    axes[0].set_title("Impurity importance")
    axes[1].barh(d["feature"], d["perm_importance_mean"], xerr=d["perm_importance_std"],
                 color=BLUE, height=0.65,
                 error_kw=dict(ecolor=INK_2, lw=1, capsize=2))
    axes[1].set_xlabel("Mean decrease in test ROC-AUC when shuffled")
    axes[1].set_title("Permutation importance (test split)")
    for ax in axes:
        ax.grid(axis="y", visible=False)
    fig.suptitle("Feature importance - Random Forest", x=0.01, ha="left",
                 fontweight="bold", fontsize=12)
    _save(fig, path)


def plot_entropy_distribution(entropy_legit, entropy_dga, path):
    """Overlaid density histograms of Shannon entropy by class."""
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    bins = np.linspace(0, max(entropy_legit.max(), entropy_dga.max()) + 0.1, 50)
    for vals, name, key in [(entropy_legit, "Legitimate", "legit"),
                            (entropy_dga, "DGA (training families)", "dga")]:
        ax.hist(vals, bins=bins, density=True, alpha=0.55, color=CLASS_COLORS[key],
                label=f"{name} (median {np.median(vals):.2f} bits)",
                edgecolor=SURFACE, linewidth=0.5)
    ax.set_xlabel("Shannon entropy of second-level label (bits)")
    ax.set_ylabel("Density")
    ax.set_title("Entropy distribution by class")
    ax.legend(loc="upper left")
    _save(fig, path)


def plot_family_detection(df, path, model_name: str):
    """Detection rate per DGA family (training and held-out) at two operating
    points; bars share one axis (0-100 %)."""
    d = df[df["family"] != "legit"].reset_index(drop=True)
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    h = 0.38
    ax.barh(y - h / 2, d["flagged@0.5"] * 100, height=h - 0.04, color=BLUE,
            label="threshold 0.5")
    ax.barh(y + h / 2, d["flagged@FPR1%"] * 100, height=h - 0.04, color=ORANGE,
            label="threshold at 1 % FPR")
    labels = [f"{f}{'  (held-out)' if s == 'held-out' else ''}"
              for f, s in zip(d["family"], d["split"])]
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    ax.set_xlabel("Domains flagged as DGA (%)")
    ax.set_title(f"Detection rate by DGA family ({model_name})")
    ax.grid(axis="y", visible=False)
    for yi, v in zip(y + h / 2, d["flagged@FPR1%"] * 100):
        ax.text(v + 1, yi, f"{v:.1f}%", va="center", fontsize=8, color=INK_2)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.16), ncol=2)
    _save(fig, path)
