"""Model evaluation: classification report, confusion matrix, per-class
ROC and precision-recall curves (one-vs-rest) with micro/macro averages, and headline
statistical parameters (accuracy, balanced accuracy, MCC, Cohen's kappa).
"""
from __future__ import annotations

import json
import logging
from itertools import cycle
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    auc,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    hamming_loss,
    matthews_corrcoef,
    multilabel_confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from .config import Config, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


def compute_statistics(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]) -> dict:
    report = classification_report(
        y_true, y_pred, labels=list(range(len(class_names))),
        target_names=class_names, output_dict=True, zero_division=0
    )
    stats = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "matthews_corrcoef": matthews_corrcoef(y_true, y_pred),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "per_class_report": report,
    }
    return stats


def save_statistics(stats: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(stats, fh, indent=2, default=float)
    logger.info("Saved statistics to %s", path)


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str], path: Path, normalize: str = "true"
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))), normalize=normalize)
    fig, ax = plt.subplots(figsize=(max(10, len(class_names) * 0.55), max(9, len(class_names) * 0.5)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=90, colorbar=True, values_format=".2f")
    ax.set_title("Confusion matrix (row-normalized)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved confusion matrix plot to %s", path)


def _roc_from_binary(y_bin: np.ndarray, y_proba: np.ndarray, class_names: List[str]):
    """Core one-vs-rest ROC/AUC computation shared by single-label
    (after ``label_binarize``) and multilabel (already binary) callers.
    """
    classes = list(range(len(class_names)))

    fpr: Dict = {}
    tpr: Dict = {}
    roc_auc: Dict = {}

    for i, name in enumerate(class_names):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc[i] = float(auc(fpr[i], tpr[i]))
        roc_auc[name] = roc_auc[i]

    fpr["micro"], tpr["micro"], _ = roc_curve(y_bin.ravel(), y_proba.ravel())
    roc_auc["micro"] = float(auc(fpr["micro"], tpr["micro"]))

    all_fpr = np.unique(np.concatenate([fpr[i] for i in classes]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in classes:
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= len(classes)
    fpr["macro"], tpr["macro"] = all_fpr, mean_tpr
    roc_auc["macro"] = float(auc(fpr["macro"], tpr["macro"]))

    return fpr, tpr, roc_auc


def _compute_roc_curves(y_true: np.ndarray, y_proba: np.ndarray, class_names: List[str]):
    """One-vs-rest fpr/tpr/AUC per class plus micro/macro averages, for a
    single-label (1D class-index) ``y_true``.
    """
    classes = list(range(len(class_names)))
    y_bin = np.eye(len(class_names), dtype=int)[np.asarray(y_true)]
    return _roc_from_binary(y_bin, y_proba, class_names)


def plot_roc_curves(
    y_true: np.ndarray, y_proba: np.ndarray, class_names: List[str], path: Path
) -> Dict[str, float]:
    """All classes' one-vs-rest ROC curves overlaid on one plot, plus micro/macro averages.

    Returns a dict of class_name -> AUC (plus 'micro' and 'macro').
    """
    classes = list(range(len(class_names)))
    fpr, tpr, roc_auc = _compute_roc_curves(y_true, y_proba, class_names)

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.plot(fpr["micro"], tpr["micro"], label=f"micro-average (AUC={roc_auc['micro']:.3f})",
            color="deeppink", linestyle=":", linewidth=3)
    ax.plot(fpr["macro"], tpr["macro"], label=f"macro-average (AUC={roc_auc['macro']:.3f})",
            color="navy", linestyle=":", linewidth=3)

    color_cycle = cycle(plt.cm.tab20.colors)
    for i, name in zip(classes, cycle(class_names)):
        if i >= len(class_names):
            break
        ax.plot(fpr[i], tpr[i], color=next(color_cycle), linewidth=1,
                label=f"{class_names[i]} (AUC={roc_auc[class_names[i]]:.2f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("One-vs-Rest ROC curves per odor class")
    ax.legend(loc="lower right", fontsize=6, ncol=2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved ROC plot to %s", path)
    return {**{name: roc_auc[name] for name in class_names}, "micro": roc_auc["micro"], "macro": roc_auc["macro"]}


def plot_roc_curves_per_class(
    y_true: np.ndarray, y_proba: np.ndarray, class_names: List[str], path: Path, n_cols: int = 4
) -> None:
    """Small-multiples grid: one subplot per class, each with its own ROC curve and AUC.

    Complements ``plot_roc_curves`` (all classes overlaid) -- with 15-20+
    classes, overlaid curves are hard to read individually; a separate
    panel per class makes each one legible on its own.
    """
    fpr, tpr, roc_auc = _compute_roc_curves(y_true, y_proba, class_names)

    n = len(class_names)
    n_cols = min(n_cols, n)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 3.0 * n_rows), squeeze=False)

    for idx, name in enumerate(class_names):
        ax = axes[idx // n_cols][idx % n_cols]
        ax.plot(fpr[idx], tpr[idx], color="tab:blue", linewidth=1.5)
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
        ax.fill_between(fpr[idx], tpr[idx], alpha=0.1, color="tab:blue")
        ax.set_title(f"{name}\nAUC={roc_auc[idx]:.3f}", fontsize=9)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.tick_params(labelsize=7)

    for idx in range(n, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")

    fig.suptitle("Per-class one-vs-rest ROC curves", y=1.0)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved per-class ROC grid to %s", path)


def _pr_from_binary(y_bin: np.ndarray, y_proba: np.ndarray, class_names: List[str]):
    """Compute one-vs-rest PR curves and Average Precision (AP).

    AP is the non-interpolated, step-wise area under the PR curve (reported
    as PR-AUC throughout this app), rather than trapezoidal PR area. Undefined
    classes (only positives or only negatives in this evaluation split) are
    excluded from the macro average and recorded as ``None`` in the report.
    """
    y_bin = np.asarray(y_bin)
    y_proba = np.asarray(y_proba)
    if y_bin.ndim != 2 or y_bin.shape != y_proba.shape or y_bin.shape[1] != len(class_names):
        raise ValueError("PR-AUC requires matching (n_samples, n_classes) truth/probability matrices")

    precision, recall, scores = {}, {}, {}
    defined_scores = []
    for i, name in enumerate(class_names):
        labels, probabilities = y_bin[:, i], y_proba[:, i]
        if not (np.any(labels == 1) and np.any(labels == 0)):
            scores[name] = None
            logger.warning("PR-AUC for %s is undefined: this split has only one class", name)
            continue
        precision[i], recall[i], _ = precision_recall_curve(labels, probabilities)
        scores[name] = float(average_precision_score(labels, probabilities))
        defined_scores.append(scores[name])

    flat_true, flat_proba = y_bin.ravel(), y_proba.ravel()
    if np.any(flat_true == 1) and np.any(flat_true == 0):
        precision["micro"], recall["micro"], _ = precision_recall_curve(flat_true, flat_proba)
        scores["micro"] = float(average_precision_score(flat_true, flat_proba))
    else:
        scores["micro"] = None
    scores["macro"] = float(np.mean(defined_scores)) if defined_scores else None
    return precision, recall, scores


def compute_pr_auc(
    y_true: np.ndarray, y_proba: np.ndarray, class_names: List[str], *, multilabel: bool = False
) -> dict:
    """Return per-class, micro and macro PR-AUC (Average Precision) scores.

    Single-label targets are class indices; multi-label targets are binary
    indicator matrices. Scores always use probabilities, not hard predictions.
    """
    if multilabel:
        y_bin = np.asarray(y_true)
    else:
        y = np.asarray(y_true)
        y_bin = np.eye(len(class_names), dtype=int)[y]
    return _pr_from_binary(y_bin, y_proba, class_names)[2]


def plot_pr_curves(
    y_true: np.ndarray, y_proba: np.ndarray, class_names: List[str], path: Path, *, multilabel: bool = False
) -> dict:
    """Plot PR curves, and return Average Precision scores for the metrics report."""
    y_bin = np.asarray(y_true) if multilabel else np.eye(len(class_names), dtype=int)[np.asarray(y_true)]
    precision, recall, pr_auc = _pr_from_binary(y_bin, y_proba, class_names)
    fig, ax = plt.subplots(figsize=(9, 8))
    if "micro" in precision:
        ax.step(recall["micro"], precision["micro"], where="post", color="deeppink", linewidth=2.5,
                label=f"micro-average (AP={pr_auc['micro']:.3f})")
        ax.axhline(float(np.mean(y_bin)), color="gray", linestyle="--", linewidth=1,
                   label="micro prevalence baseline")
    colors = cycle(plt.cm.tab20.colors)
    for i, name in enumerate(class_names):
        color = next(colors)
        if i in precision:
            ax.step(recall[i], precision[i], where="post", linewidth=1, color=color,
                    label=f"{name} (AP={pr_auc[name]:.2f})")
    macro_text = "n/a" if pr_auc["macro"] is None else f"{pr_auc['macro']:.3f}"
    ax.set(xlim=(0, 1), ylim=(0, 1.05), xlabel="Recall", ylabel="Precision",
           title=f"One-vs-rest precision-recall curves (macro PR-AUC / AP={macro_text})")
    ax.legend(loc="lower left", fontsize=6, ncol=2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved precision-recall plot to %s", path)
    return pr_auc


def plot_pr_curves_per_class(
    y_true: np.ndarray, y_proba: np.ndarray, class_names: List[str], path: Path,
    n_cols: int = 4, *, multilabel: bool = False,
) -> None:
    """Small-multiples PR curves; the dashed line is each class's prevalence."""
    y_bin = np.asarray(y_true) if multilabel else np.eye(len(class_names), dtype=int)[np.asarray(y_true)]
    precision, recall, pr_auc = _pr_from_binary(y_bin, y_proba, class_names)
    n = len(class_names)
    n_cols = min(n_cols, n)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 3.0 * n_rows), squeeze=False)
    for i, name in enumerate(class_names):
        ax = axes[i // n_cols][i % n_cols]
        if i in precision:
            ax.step(recall[i], precision[i], where="post", color="tab:blue", linewidth=1.5)
            ax.axhline(float(np.mean(y_bin[:, i])), color="gray", linestyle="--", linewidth=0.8)
            ax.fill_between(recall[i], precision[i], step="post", color="tab:blue", alpha=0.1)
            ax.set_title(f"{name}\nPR-AUC (AP)={pr_auc[name]:.3f}", fontsize=9)
        else:
            ax.text(0.5, 0.5, "Undefined on this split", ha="center", va="center", fontsize=8)
            ax.set_title(f"{name}\nPR-AUC (AP)=n/a", fontsize=9)
        ax.set(xlim=(0, 1), ylim=(0, 1.05))
        ax.tick_params(labelsize=7)
    for i in range(n, n_rows * n_cols):
        axes[i // n_cols][i % n_cols].axis("off")
    fig.suptitle("Per-class precision-recall curves", y=1.0)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved per-class precision-recall grid to %s", path)


def plot_model_comparison(val_results: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(val_results["model"], val_results["val_f1_macro"])
    ax.set_ylabel("Validation macro-F1")
    ax.set_title("Model comparison (held-out validation split)")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved model comparison plot to %s", path)


def build_per_class_table(stats: dict, class_names: List[str]) -> pd.DataFrame:
    """Per-class precision/recall/F1/support + ROC-AUC/PR-AUC + a plain-language
    quality flag, sorted weakest-first so the classes needing attention are
    listed first.
    """
    report = stats["per_class_report"]
    roc_auc = stats.get("roc_auc", {})
    pr_auc = stats.get("pr_auc", {})
    rows = []
    for name in class_names:
        row = report.get(name, {})
        f1 = row.get("f1-score", 0.0)
        if f1 >= 0.6:
            quality = "Good"
        elif f1 >= 0.35:
            quality = "Fair"
        else:
            quality = "Poor"
        rows.append({
            "class": name,
            "support": int(row.get("support", 0)),
            "precision": row.get("precision", 0.0),
            "recall": row.get("recall", 0.0),
            "f1": f1,
            "roc_auc": roc_auc.get(name, float("nan")),
            "pr_auc": pr_auc.get(name, float("nan")),
            "quality": quality,
        })
    table = pd.DataFrame(rows).sort_values("f1", ascending=True).reset_index(drop=True)
    return table


def log_and_save_per_class_table(
    stats: dict, class_names: List[str], config: Config, prefix: str = "test"
) -> pd.DataFrame:
    table = build_per_class_table(stats, class_names)
    display = table.copy()
    for col in ("precision", "recall", "f1", "roc_auc", "pr_auc"):
        display[col] = display[col].map(lambda v: f"{v:.3f}" if v == v else "n/a")
    logger.info(
        "Per-class statistical evaluation on the %s set (weakest first):\n%s",
        prefix, display.to_string(index=False),
    )
    n_poor = (table["quality"] == "Poor").sum()
    n_fair = (table["quality"] == "Fair").sum()
    n_good = (table["quality"] == "Good").sum()
    logger.info("Class quality summary: %d good, %d fair, %d poor (F1 < 0.35).", n_good, n_fair, n_poor)
    if n_poor:
        weak = ", ".join(table.loc[table["quality"] == "Poor", "class"])
        logger.info("Classes needing attention (lowest F1): %s", weak)

    path = config.reports_dir / f"{prefix}_per_class_stats.csv"
    table.to_csv(path, index=False)
    logger.info("Saved per-class statistics to %s", path)
    return table


def full_evaluation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    class_names: List[str],
    config: Config = DEFAULT_CONFIG,
    prefix: str = "test",
) -> dict:
    stats = compute_statistics(y_true, y_pred, class_names)
    roc_auc = plot_roc_curves(y_true, y_proba, class_names, config.plots_dir / f"{prefix}_roc_curves.png")
    plot_roc_curves_per_class(y_true, y_proba, class_names, config.plots_dir / f"{prefix}_roc_curves_per_class.png")
    plot_confusion_matrix(y_true, y_pred, class_names, config.plots_dir / f"{prefix}_confusion_matrix.png")
    stats["roc_auc"] = roc_auc
    stats["pr_auc"] = plot_pr_curves(
        y_true, y_proba, class_names, config.plots_dir / f"{prefix}_pr_curves.png"
    )
    plot_pr_curves_per_class(
        y_true, y_proba, class_names, config.plots_dir / f"{prefix}_pr_curves_per_class.png"
    )
    save_statistics(stats, config.reports_dir / f"{prefix}_metrics.json")
    log_and_save_per_class_table(stats, class_names, config, prefix=prefix)
    return stats


# ---------------------------------------------------------------------------
# Multi-label evaluation. y_true/y_pred here are (n_samples, n_labels)
# binary indicator matrices, not 1D class indices. Reuses the same core
# ROC computation (``_roc_from_binary``) and the same per-class table
# builder as single-label (``classification_report`` and the roc_auc dict
# it produces have an identical shape either way) -- only the headline
# statistics (Hamming loss, subset accuracy instead of MCC/kappa, which
# don't have a standard multilabel generalization) and the plotting
# functions that need genuinely different inputs are duplicated.
# ---------------------------------------------------------------------------

def compute_statistics_multilabel(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]) -> dict:
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    return {
        "subset_accuracy": accuracy_score(y_true, y_pred),
        "hamming_loss": hamming_loss(y_true, y_pred),
        "macro_f1": report["macro avg"]["f1-score"],
        "micro_f1": report["micro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "avg_labels_per_compound_true": float(np.asarray(y_true).sum(axis=1).mean()),
        "avg_labels_per_compound_pred": float(np.asarray(y_pred).sum(axis=1).mean()),
        "per_class_report": report,
    }


def plot_roc_curves_multilabel(
    y_true: np.ndarray, y_proba: np.ndarray, class_names: List[str], path: Path
) -> Dict[str, float]:
    """Same overlay-all-classes ROC plot as ``plot_roc_curves``, sourced
    from an already-binary multilabel ``y_true`` instead of 1D class indices.
    """
    classes = list(range(len(class_names)))
    fpr, tpr, roc_auc = _roc_from_binary(y_true, y_proba, class_names)

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.plot(fpr["micro"], tpr["micro"], label=f"micro-average (AUC={roc_auc['micro']:.3f})",
            color="deeppink", linestyle=":", linewidth=3)
    ax.plot(fpr["macro"], tpr["macro"], label=f"macro-average (AUC={roc_auc['macro']:.3f})",
            color="navy", linestyle=":", linewidth=3)

    color_cycle = cycle(plt.cm.tab20.colors)
    for i, name in zip(classes, cycle(class_names)):
        if i >= len(class_names):
            break
        ax.plot(fpr[i], tpr[i], color=next(color_cycle), linewidth=1,
                label=f"{class_names[i]} (AUC={roc_auc[class_names[i]]:.2f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Per-label ROC curves (multi-label)")
    ax.legend(loc="lower right", fontsize=6, ncol=2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved multilabel ROC plot to %s", path)
    return {**{name: roc_auc[name] for name in class_names}, "micro": roc_auc["micro"], "macro": roc_auc["macro"]}


def plot_roc_curves_per_class_multilabel(
    y_true: np.ndarray, y_proba: np.ndarray, class_names: List[str], path: Path, n_cols: int = 4
) -> None:
    fpr, tpr, roc_auc = _roc_from_binary(y_true, y_proba, class_names)

    n = len(class_names)
    n_cols = min(n_cols, n)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 3.0 * n_rows), squeeze=False)

    for idx, name in enumerate(class_names):
        ax = axes[idx // n_cols][idx % n_cols]
        ax.plot(fpr[idx], tpr[idx], color="tab:blue", linewidth=1.5)
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
        ax.fill_between(fpr[idx], tpr[idx], alpha=0.1, color="tab:blue")
        ax.set_title(f"{name}\nAUC={roc_auc[idx]:.3f}", fontsize=9)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.tick_params(labelsize=7)

    for idx in range(n, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")

    fig.suptitle("Per-label one-vs-rest ROC curves (multi-label)", y=1.0)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved multilabel per-class ROC grid to %s", path)


def plot_confusion_matrices_multilabel(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str], path: Path, n_cols: int = 4
) -> None:
    """One small 2x2 confusion matrix per label (present/absent), since a
    single N x N matrix doesn't make sense when labels aren't mutually
    exclusive.
    """
    mats = multilabel_confusion_matrix(y_true, y_pred)
    n = len(class_names)
    n_cols = min(n_cols, n)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.6 * n_cols, 2.6 * n_rows), squeeze=False)

    for idx, name in enumerate(class_names):
        ax = axes[idx // n_cols][idx % n_cols]
        disp = ConfusionMatrixDisplay(mats[idx], display_labels=["absent", "present"])
        disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
        ax.set_title(name, fontsize=9)
        ax.tick_params(labelsize=7)
        ax.set_xlabel("")
        ax.set_ylabel("")

    for idx in range(n, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")

    fig.suptitle("Per-label confusion matrices (multi-label)", y=1.0)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved multilabel confusion matrix grid to %s", path)


def full_evaluation_multilabel(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    class_names: List[str],
    config: Config = DEFAULT_CONFIG,
    prefix: str = "test",
) -> dict:
    stats = compute_statistics_multilabel(y_true, y_pred, class_names)
    roc_auc = plot_roc_curves_multilabel(y_true, y_proba, class_names, config.plots_dir / f"{prefix}_roc_curves.png")
    plot_roc_curves_per_class_multilabel(
        y_true, y_proba, class_names, config.plots_dir / f"{prefix}_roc_curves_per_class.png"
    )
    plot_confusion_matrices_multilabel(
        y_true, y_pred, class_names, config.plots_dir / f"{prefix}_confusion_matrix.png"
    )
    stats["roc_auc"] = roc_auc
    stats["pr_auc"] = plot_pr_curves(
        y_true, y_proba, class_names, config.plots_dir / f"{prefix}_pr_curves.png", multilabel=True
    )
    plot_pr_curves_per_class(
        y_true, y_proba, class_names, config.plots_dir / f"{prefix}_pr_curves_per_class.png",
        multilabel=True,
    )
    save_statistics(stats, config.reports_dir / f"{prefix}_metrics.json")
    # classification_report has the same shape as single-label, so the same
    # per-class table builder works unchanged.
    log_and_save_per_class_table(stats, class_names, config, prefix=prefix)
    return stats
