import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
    roc_curve
)
import seaborn as sns


def plot_classification_metrics(model, X_test, y_test, title="Classification Metrics"):
    """
    Computes and plots performance metrics and confusion matrix for a classifier.

    Parameters:
        model: Trained classifier with predict and optionally predict_proba
        X_test: Test features
        y_test: True labels
        title: Optional title for the plots
    """

    # Palette colors
    light_red = "#FFE4E4"
    dark_red = "#BD8686"
    light_blue = "#E2FFFF"
    dark_blue = "#91C4C4"

    # Predict labels and probabilities
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)
    mcc = matthews_corrcoef(y_test, y_pred)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)

    # Create figure with subplots
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(title, fontsize=16, color=dark_red)

    # --- Confusion Matrix Heatmap ---
    sns.heatmap(cm, annot=True, fmt="d", cmap=sns.light_palette(dark_red, as_cmap=True), ax=axs[0],
                cbar=False, linecolor=dark_red, linewidth=1)
    axs[0].set_title("Confusion Matrix", color=dark_red)
    axs[0].set_xlabel("Predicted Labels", color=dark_red)
    axs[0].set_ylabel("True Labels", color=dark_red)
    axs[0].tick_params(axis='x', colors=dark_red)
    axs[0].tick_params(axis='y', colors=dark_red)
    axs[0].set_xticklabels(['Legit', 'Fraud'])
    axs[0].set_yticklabels(['Legit', 'Fraud'])

    # --- ROC Curve ---
    axs[1].plot(fpr, tpr, color=dark_blue, lw=3, label=f"AUC = {auc:.4f}")
    axs[1].plot([0, 1], [0, 1], color=dark_red, lw=1.5, linestyle="--")
    axs[1].set_xlim([0.0, 1.0])
    axs[1].set_ylim([0.0, 1.05])
    axs[1].set_xlabel("False Positive Rate", color=dark_red)
    axs[1].set_ylabel("True Positive Rate", color=dark_red)
    axs[1].set_title("ROC Curve", color=dark_red)
    axs[1].tick_params(colors=dark_red)
    axs[1].legend(loc="lower right", facecolor=light_blue, edgecolor=dark_red)

    # --- Metrics Display Below ---
    metrics_text = (
        f"Accuracy:  {accuracy:.4f}\n"
        f"Precision: {precision:.4f}\n"
        f"Recall:    {recall:.4f}\n"
        f"F1 Score:  {f1:.4f}\n"
        f"AUC-ROC:   {auc:.4f}\n"
        f"MCC:       {mcc:.4f}"
    )
    plt.figtext(0.5, 0.03, metrics_text, ha="center", fontsize=12,
                bbox={"facecolor": light_blue, "alpha": 0.7, "pad": 8, "edgecolor": dark_red})

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.show()
    plt.savefig("classification_metrics.png", dpi=300)
