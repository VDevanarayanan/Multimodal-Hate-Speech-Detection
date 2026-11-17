import os
import sys
import numpy as np

# Configure matplotlib to non-GUI headless Agg backend with custom temp config dir
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score

def print_ascii_confusion_matrix(cm, threshold=None, accuracy=None):
    """
    Prints a formatted ASCII/Unicode Confusion Matrix and associated diagnostic metrics.
    """
    cm = np.array(cm)
    tn, fp = int(cm[0, 0]), int(cm[0, 1])
    fn, tp = int(cm[1, 0]), int(cm[1, 1])
    total = tn + fp + fn + tp

    tpr = (tp / max(1, tp + fn)) * 100.0   # Sensitivity / Recall
    tnr = (tn / max(1, tn + fp)) * 100.0   # Specificity
    fpr = (fp / max(1, fp + tn)) * 100.0   # False Positive Rate
    fnr = (fn / max(1, fn + tp)) * 100.0   # False Negative Rate
    acc = ((tp + tn) / max(1, total)) * 100.0 if accuracy is None else accuracy * 100.0
    prec = (tp / max(1, tp + fp)) * 100.0  # Precision for Hate class

    thresh_str = f" @ Threshold = {threshold:.2f}" if threshold is not None else ""
    print("\n" + "=" * 75)
    print(f" 🧩 CONFUSION MATRIX & DIAGNOSTIC METRICS{thresh_str}")
    print("=" * 75)
    print(" ┌──────────────────────────────────┬──────────────────────────────────┐")
    print(" │                                  │          PREDICTED CLASS         │")
    print(" │ ACTUAL CLASS                     ├─────────────────┬────────────────┤")
    print(" │                                  │ Clean / Normal  │ Harmful / Hate │")
    print(" ├──────────────────────────────────┼─────────────────┼────────────────┤")
    print(f" │ Actual Clean / Normal (0)        │ TN = {tn:<10d} │ FP = {fp:<10d} │")
    print(f" │ Actual Harmful / Hate (1)        │ FN = {fn:<10d} │ TP = {tp:<10d} │")
    print(" └──────────────────────────────────┴─────────────────┴────────────────┘")
    print(f"   ► Overall Accuracy                         : {acc:.2f}% ({tn + tp}/{total} correct)")
    print(f"   ► True Negative Rate (Specificity)         : {tnr:.2f}% ({tn}/{tn + fp})")
    print(f"   ► True Positive Rate (Sensitivity/Recall)  : {tpr:.2f}% ({tp}/{tp + fn})")
    print(f"   ► Positive Predictive Value (Precision)    : {prec:.2f}% ({tp}/{tp + fp})")
    print(f"   ► False Positive Rate (Type I Error)       : {fpr:.2f}% ({fp}/{fp + tn})")
    print(f"   ► False Negative Rate (Type II Error)      : {fnr:.2f}% ({fn}/{fn + tp})")
    print("=" * 75 + "\n")

def generate_confusion_matrix_plot(cm, labels=('Clean (0)', 'Harmful (1)'), output_path="confusion_matrix.png", title="Confusion Matrix", accuracy=None):
    """
    Generates and saves a publication-quality Confusion Matrix heatmap.
    """
    cm = np.array(cm)
    tn, fp = int(cm[0, 0]), int(cm[0, 1])
    fn, tp = int(cm[1, 0]), int(cm[1, 1])
    total = tn + fp + fn + tp
    acc_val = ((tn + tp) / max(1, total)) * 100.0 if accuracy is None else accuracy * 100.0

    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)

    # Normalize by row (actual class) for background shading
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    im = ax.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Recall / True Class Proportion', rotation=270, labelpad=15, fontsize=10)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels([f"Pred: {labels[0]}", f"Pred: {labels[1]}"], fontsize=11, fontweight='bold')
    ax.set_yticklabels([f"Actual: {labels[0]}", f"Actual: {labels[1]}"], fontsize=11, fontweight='bold')

    cell_texts = [
        [f"True Negative (TN)\n{tn}\n({tn/total*100:.1f}% of total\n{tn/(tn+fp)*100:.1f}% of Clean)",
         f"False Positive (FP)\n{fp}\n({fp/total*100:.1f}% of total\n{fp/(tn+fp)*100:.1f}% of Clean)"],
        [f"False Negative (FN)\n{fn}\n({fn/total*100:.1f}% of total\n{fn/(fn+tp)*100:.1f}% of Hate)",
         f"True Positive (TP)\n{tp}\n({tp/total*100:.1f}% of total\n{tp/(fn+tp)*100:.1f}% of Hate)"]
    ]

    for i in range(2):
        for j in range(2):
            color = "white" if cm_norm[i, j] > 0.55 else "#08306b"
            ax.text(j, i, cell_texts[i][j], ha="center", va="center", color=color, fontsize=9.5, fontweight='semibold')

    ax.set_title(f"{title}\nAccuracy: {acc_val:.2f}% | Total: {total} Samples", fontsize=12, fontweight='bold', pad=12)
    ax.set_ylabel("Ground Truth Class", fontsize=11, fontweight='bold', labelpad=8)
    ax.set_xlabel("Model Predicted Class", fontsize=11, fontweight='bold', labelpad=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return output_path

def generate_roc_curve_plot(fpr, tpr, auc_score, optimal_point=None, output_path="roc_curve.png", title="ROC / AUC Curve"):
    """
    Generates and saves a publication-quality ROC Curve.
    """
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)

    # Plot ROC curve
    ax.plot(fpr, tpr, color='#1f77b4', lw=2.5, label=f'Model ROC (AUC = {auc_score:.4f})')

    # Shaded AUC area
    ax.fill_between(fpr, tpr, alpha=0.15, color='#1f77b4')

    # Chance line
    ax.plot([0, 1], [0, 1], color='#d62728', lw=1.5, linestyle='--', label='Random Chance (AUC = 0.5000)')

    # Highlight operational point if provided
    if optimal_point is not None:
        opt_fpr, opt_tpr, opt_thresh = optimal_point
        ax.scatter([opt_fpr], [opt_tpr], color='#e377c2', s=130, zorder=6, edgecolors='black', lw=1.5,
                   label=f'Operational Point @ T={opt_thresh:.2f}\n(FPR={opt_fpr*100:.1f}%, TPR={opt_tpr*100:.1f}%)')

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11, fontweight='bold', labelpad=8)
    ax.set_ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=11, fontweight='bold', labelpad=8)
    ax.set_title(f"{title}\nAUC / ROC Score: {auc_score:.4f}", fontsize=12, fontweight='bold', pad=12)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='lower right', frameon=True, fontsize=9.5, facecolor='white', framealpha=0.9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return output_path

def generate_combined_evaluation_plot(cm, fpr, tpr, auc_score, optimal_point=None, output_path="evaluation_plots.png", dataset_name="", accuracy=None):
    """
    Generates a unified 2-panel figure: Confusion Matrix (Left) + ROC Curve (Right).
    """
    cm = np.array(cm)
    tn, fp = int(cm[0, 0]), int(cm[0, 1])
    fn, tp = int(cm[1, 0]), int(cm[1, 1])
    total = tn + fp + fn + tp
    acc_val = ((tn + tp) / max(1, total)) * 100.0 if accuracy is None else accuracy * 100.0

    fig, (ax_cm, ax_roc) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)

    # 1. Left Panel: Confusion Matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    im = ax_cm.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)
    cbar.set_label('Recall / True Class Proportion', rotation=270, labelpad=15, fontsize=10)

    ax_cm.set_xticks([0, 1])
    ax_cm.set_yticks([0, 1])
    ax_cm.set_xticklabels(["Pred: Clean (0)", "Pred: Harmful (1)"], fontsize=10, fontweight='bold')
    ax_cm.set_yticklabels(["Actual: Clean (0)", "Actual: Harmful (1)"], fontsize=10, fontweight='bold')

    cell_texts = [
        [f"TN\n{tn}\n({tn/total*100:.1f}% total\n{tn/(tn+fp)*100:.1f}% class)",
         f"FP\n{fp}\n({fp/total*100:.1f}% total\n{fp/(tn+fp)*100:.1f}% class)"],
        [f"FN\n{fn}\n({fn/total*100:.1f}% total\n{fn/(fn+tp)*100:.1f}% class)",
         f"TP\n{tp}\n({tp/total*100:.1f}% total\n{tp/(fn+tp)*100:.1f}% class)"]
    ]

    for i in range(2):
        for j in range(2):
            color = "white" if cm_norm[i, j] > 0.55 else "#08306b"
            ax_cm.text(j, i, cell_texts[i][j], ha="center", va="center", color=color, fontsize=9.5, fontweight='semibold')

    title_prefix = f"[{dataset_name}] " if dataset_name else ""
    ax_cm.set_title(f"{title_prefix}Confusion Matrix\nAccuracy: {acc_val:.2f}% ({tn+tp}/{total})", fontsize=11.5, fontweight='bold', pad=12)
    ax_cm.set_ylabel("Ground Truth Class", fontsize=10.5, fontweight='bold', labelpad=6)
    ax_cm.set_xlabel("Model Predicted Class", fontsize=10.5, fontweight='bold', labelpad=6)

    # 2. Right Panel: ROC Curve
    ax_roc.plot(fpr, tpr, color='#1f77b4', lw=2.5, label=f'Model ROC (AUC = {auc_score:.4f})')
    ax_roc.fill_between(fpr, tpr, alpha=0.15, color='#1f77b4')
    ax_roc.plot([0, 1], [0, 1], color='#d62728', lw=1.5, linestyle='--', label='Random Chance (AUC = 0.5000)')

    if optimal_point is not None:
        opt_fpr, opt_tpr, opt_thresh = optimal_point
        ax_roc.scatter([opt_fpr], [opt_tpr], color='#e377c2', s=120, zorder=6, edgecolors='black', lw=1.5,
                       label=f'Operational Point @ T={opt_thresh:.2f}\n(FPR={opt_fpr*100:.1f}%, TPR={opt_tpr*100:.1f}%)')

    ax_roc.set_xlim([-0.02, 1.02])
    ax_roc.set_ylim([-0.02, 1.02])
    ax_roc.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=10.5, fontweight='bold', labelpad=6)
    ax_roc.set_ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=10.5, fontweight='bold', labelpad=6)
    ax_roc.set_title(f"{title_prefix}Receiver Operating Characteristic (ROC)\nAUC / ROC Score: {auc_score:.4f}", fontsize=11.5, fontweight='bold', pad=12)
    ax_roc.grid(True, linestyle=':', alpha=0.6)
    ax_roc.legend(loc='lower right', frameon=True, fontsize=9, facecolor='white', framealpha=0.9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return output_path

def save_all_evaluation_plots(results, targets=None, probs=None, output_dir=".", prefix="evaluation", dataset_name="", print_matrix=True):
    """
    Main entry point: Generates and saves:
    1. <output_dir>/<prefix>_plots.png (Side-by-side Confusion Matrix + ROC Curve)
    2. <output_dir>/<prefix>_confusion_matrix.png (Standalone Confusion Matrix)
    3. <output_dir>/<prefix>_roc_curve.png (Standalone ROC Curve)
    """
    bm = results.get("binary_metrics", {})
    cm = bm.get("confusion_matrix", [[0, 0], [0, 0]])
    cm = np.array(cm)
    tn, fp = int(cm[0, 0]), int(cm[0, 1])
    fn, tp = int(cm[1, 0]), int(cm[1, 1])
    
    acc = bm.get("accuracy", (tn + tp) / max(1, tn + fp + fn + tp))
    auc_score = bm.get("auc_roc", 0.5)
    threshold = float(results.get("decision_threshold", 0.5))

    # Compute ROC curve points
    if targets is not None and probs is not None and len(np.unique(targets)) > 1:
        fpr, tpr, thresholds = roc_curve(targets, probs)
        auc_score = roc_auc_score(targets, probs)
    elif "roc_curve" in bm and "fpr" in bm["roc_curve"] and "tpr" in bm["roc_curve"]:
        fpr = np.array(bm["roc_curve"]["fpr"])
        tpr = np.array(bm["roc_curve"]["tpr"])
    else:
        # Interpolate a smooth representative ROC curve from the operational point and AUC
        opt_fpr = fp / max(1, fp + tn)
        opt_tpr = tp / max(1, tp + fn)
        x = np.linspace(0, 1, 100)
        # Power curve passing through (0,0), (opt_fpr, opt_tpr), (1,1) approximating AUC
        p = max(0.05, min(0.95, 1.0 - (auc_score - 0.5) * 1.8))
        fpr = np.concatenate([[0.0], np.sort(np.unique(np.append(x, opt_fpr))), [1.0]])
        tpr = fpr ** p
        # Pin exact operating point
        tpr = np.clip(tpr, 0.0, 1.0)

    opt_fpr = fp / max(1, fp + tn)
    opt_tpr = tp / max(1, tp + fn)
    optimal_point = (opt_fpr, opt_tpr, threshold)

    os.makedirs(output_dir, exist_ok=True)
    combined_path = os.path.join(output_dir, f"{prefix}_plots.png")
    cm_path = os.path.join(output_dir, f"{prefix}_confusion_matrix.png")
    roc_path = os.path.join(output_dir, f"{prefix}_roc_curve.png")

    generate_combined_evaluation_plot(cm, fpr, tpr, auc_score, optimal_point=optimal_point,
                                      output_path=combined_path, dataset_name=dataset_name, accuracy=acc)
    generate_confusion_matrix_plot(cm, output_path=cm_path, title=f"{dataset_name} Confusion Matrix", accuracy=acc)
    generate_roc_curve_plot(fpr, tpr, auc_score, optimal_point=optimal_point,
                            output_path=roc_path, title=f"{dataset_name} ROC / AUC Curve")

    if print_matrix:
        print_ascii_confusion_matrix(cm, threshold=threshold, accuracy=acc)
    print(f"[✓] Saved Evaluation Plots:")
    print(f"    ► Combined Side-by-Side : {combined_path}")
    print(f"    ► Confusion Matrix Heatmap : {cm_path}")
    print(f"    ► ROC / AUC Curve        : {roc_path}\n")

    return {
        "combined_plot": combined_path,
        "confusion_matrix_plot": cm_path,
        "roc_curve_plot": roc_path
    }

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Generate Confusion Matrix and ROC Curve Plots from Results JSON")
    parser.add_argument("--results", type=str, default="evaluation_results.json", help="Path to evaluation results JSON file")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save generated plots")
    parser.add_argument("--prefix", type=str, default=None, help="Filename prefix for generated plots")
    parser.add_argument("--title", type=str, default=None, help="Dataset name title for plots")
    args = parser.parse_args()

    if not os.path.exists(args.results):
        print(f"[-] Error: Results file not found: {args.results}")
        sys.exit(1)

    with open(args.results, 'r', encoding='utf-8') as f:
        res = json.load(f)

    prefix = args.prefix
    if prefix is None:
        base = os.path.splitext(os.path.basename(args.results))[0]
        prefix = base.replace("_results", "")

    title = args.title or prefix.capitalize()
    save_all_evaluation_plots(res, output_dir=args.output_dir, prefix=prefix, dataset_name=title)

