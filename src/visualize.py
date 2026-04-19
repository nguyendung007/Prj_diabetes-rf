"""
src/visualize.py
Tất cả hàm vẽ biểu đồ cho dự án.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


SAVE_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(SAVE_DIR, exist_ok=True)

PALETTE = "Set2"
plt.rcParams.update({"figure.dpi": 120, "font.family": "DejaVu Sans"})


# ── EDA ───────────────────────────────────────────────────────────────────────

def plot_class_distribution(y, save: bool = True):
    """Biểu đồ phân phối nhãn (có / không tiểu đường)."""
    fig, ax = plt.subplots(figsize=(5, 4))
    labels = ["No Diabetes (0)", "Diabetes (1)"]
    counts = [int((y == 0).sum()), int((y == 1).sum())]
    colors = ["#4CAF50", "#F44336"]
    bars = ax.bar(labels, counts, color=colors, edgecolor="white", linewidth=1.5)
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(cnt), ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.set_title("Phân phối nhãn (Class Distribution)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Số lượng")
    ax.set_ylim(0, max(counts) * 1.2)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(SAVE_DIR, "01_class_distribution.png"))
    plt.show()
    plt.close()


def plot_histograms(df, save: bool = True):
    """Histogram của tất cả features."""
    features = [c for c in df.columns if c != "Outcome"]
    n = len(features)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 3.5))
    axes = axes.flatten()

    colors = sns.color_palette(PALETTE, n)
    for i, feat in enumerate(features):
        axes[i].hist(df[feat], bins=30, color=colors[i], edgecolor="white", alpha=0.85)
        axes[i].set_title(feat, fontsize=11, fontweight="bold")
        axes[i].spines[["top", "right"]].set_visible(False)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Phân phối các Features", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(SAVE_DIR, "02_histograms.png"), bbox_inches="tight")
    plt.show()
    plt.close()


def plot_correlation_heatmap(df, save: bool = True):
    """Correlation heatmap."""
    fig, ax = plt.subplots(figsize=(9, 7))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
        linewidths=0.5, ax=ax, square=True, cbar_kws={"shrink": 0.8}
    )
    ax.set_title("Correlation Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(SAVE_DIR, "03_correlation_heatmap.png"))
    plt.show()
    plt.close()


def plot_boxplots(df, save: bool = True):
    """Boxplot features theo nhóm Outcome."""
    features = [c for c in df.columns if c != "Outcome"]
    n = len(features)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 3.5))
    axes = axes.flatten()

    for i, feat in enumerate(features):
        sns.boxplot(
            data=df, x="Outcome", y=feat, ax=axes[i],
            palette=["#4CAF50", "#F44336"], hue="Outcome", legend=False
        )
        axes[i].set_title(feat, fontsize=11, fontweight="bold")
        axes[i].set_xlabel("Outcome (0=No, 1=Yes)")
        axes[i].spines[["top", "right"]].set_visible(False)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Boxplot Features theo nhóm Outcome", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(SAVE_DIR, "04_boxplots.png"), bbox_inches="tight")
    plt.show()
    plt.close()


# ── Model results ──────────────────────────────────────────────────────────────

def plot_confusion_matrix(cm, model_name: str = "Random Forest", save: bool = True):
    """Vẽ Confusion Matrix."""
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["No Diabetes", "Diabetes"],
        yticklabels=["No Diabetes", "Diabetes"],
        linewidths=0.5
    )
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual", fontsize=11)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save:
        fname = model_name.replace(" ", "_").lower()
        plt.savefig(os.path.join(SAVE_DIR, f"05_confusion_{fname}.png"))
    plt.show()
    plt.close()


def plot_feature_importance(model, feature_names: list, top_n: int = 8, save: bool = True):
    """Biểu đồ Feature Importance của Random Forest."""
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = sns.color_palette("Blues_r", top_n)
    ax.barh(
        [feature_names[i] for i in idx[::-1]],
        importances[idx[::-1]],
        color=colors[::-1], edgecolor="white"
    )
    ax.set_xlabel("Tầm quan trọng (Gini Importance)")
    ax.set_title("Feature Importance — Random Forest", fontsize=14, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(SAVE_DIR, "06_feature_importance.png"))
    plt.show()
    plt.close()


def plot_roc_curves(results: list, save: bool = True):
    """
    Vẽ ROC Curve cho nhiều mô hình.

    Parameters
    ----------
    results : list of dict — mỗi dict gồm 'name', 'fpr', 'tpr', 'auc'
    """
    from sklearn.metrics import auc as sklearn_auc

    fig, ax = plt.subplots(figsize=(7, 6))
    colors = sns.color_palette(PALETTE, len(results))
    for res, color in zip(results, colors):
        auc_val = sklearn_auc(res["fpr"], res["tpr"])
        ax.plot(
            res["fpr"], res["tpr"], color=color, lw=2,
            label=f"{res['name']} (AUC = {auc_val:.3f})"
        )
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curves — So sánh mô hình", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(SAVE_DIR, "07_roc_curves.png"))
    plt.show()
    plt.close()


def plot_model_comparison(eval_results: list, save: bool = True):
    """Bar chart so sánh Accuracy, F1, AUC của các mô hình."""
    names = [r["name"] for r in eval_results]
    metrics = {
        "Accuracy": [r["accuracy"] for r in eval_results],
        "F1-Score": [r["f1"] for r in eval_results],
        "ROC-AUC": [r["auc"] if r["auc"] else 0 for r in eval_results],
    }

    x = np.arange(len(names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#2196F3", "#4CAF50", "#FF9800"]

    for i, (metric, vals) in enumerate(metrics.items()):
        bars = ax.bar(x + i * width, vals, width, label=metric, color=colors[i], alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8
            )

    ax.set_xticks(x + width)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("So sánh hiệu suất các mô hình", fontsize=14, fontweight="bold")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(SAVE_DIR, "08_model_comparison.png"))
    plt.show()
    plt.close()
