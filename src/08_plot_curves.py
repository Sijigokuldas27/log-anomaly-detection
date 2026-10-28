import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

# draws roc and pr curves for the 3 models, using the test scores saved by 05, 06, 07
# run: python src/08_plot_curves.py

models = [
    ("isolation_forest_normal", "Isolation Forest", "#2a78d6", "-"),
    ("lstm", "LSTM", "#eb6834", "--"),
    ("transformer", "Transformer", "#1baf7a", ":"),
]

for dataset in ["hdfs", "bgl"]:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    for file, label, color, style in models:
        df = pd.read_csv("results/" + dataset + "_" + file + "_scores.csv")
        y, s = df["label"], df["score"]

        fpr, tpr, _ = roc_curve(y, s)
        auc = roc_auc_score(y, s)
        ax1.plot(fpr, tpr, color=color, linestyle=style, linewidth=2,
                 label=label + " (AUC " + str(round(auc, 3)) + ")")

        prec, rec, _ = precision_recall_curve(y, s)
        ap = average_precision_score(y, s)
        ax2.plot(rec, prec, color=color, linestyle=style, linewidth=2,
                 label=label + " (AP " + str(round(ap, 3)) + ")")

    # random guess line, only for roc
    ax1.plot([0, 1], [0, 1], color="gray", linewidth=1, linestyle="-", alpha=0.5)
    ax1.set_xlabel("False positive rate")
    ax1.set_ylabel("True positive rate")
    ax1.set_title(dataset.upper() + " - ROC curve")

    # for pr curve the random line is the anomaly rate
    rate = y.mean()
    ax2.axhline(rate, color="gray", linewidth=1, alpha=0.5)
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title(dataset.upper() + " - Precision-Recall curve")

    for ax in (ax1, ax2):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="lower right" if ax is ax1 else "lower left", frameon=False)

    plt.tight_layout()
    plt.savefig("results/" + dataset + "_curves.png", dpi=150)
    plt.close()
    print("saved results/" + dataset + "_curves.png")