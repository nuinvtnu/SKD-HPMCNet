import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)
from units import *
from units_ablation import *

if __name__ == '__main__':
    path_data = "dataset/"
    path_result = "result_ablation/"
    path_model = "Model_ablation/"
    print(" LOAD TEST DATA")
    X_test, y_test = load_dataset(path_data, "Test_Methylation.csv", 1)
    print(f"Shape of test: X_test = {X_test.shape}, y_test = {y_test.shape}\n")

    y_true_classes = np.argmax(y_test, axis=1)

    # model of PTM (Methylation)
    model_files = [
        "Methylation_best_model_1cnn.keras",
        "Methylation_best_model_2cnn.keras",
        "Methylation_best_model_3cnn.keras",
        "Methylation_best_model_lstm.keras",
        "Methylation_best_model_nocnn.keras"
    ]

    summary_metrics = []
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(16, 7))
    print("Test all PTM")
    for model_name in model_files:
        full_model_path = os.path.join(path_model, model_name)

        # check and load file
        if not os.path.exists(full_model_path):
            print(f"Warning: File not found {full_model_path}, bỏ qua...")
            continue

        print(f"\n Processing model: {model_name}...")

        # Load model
        model = keras.models.load_model(full_model_path, compile=False)

        # Predict
        y_pred_probs = model.predict(X_test, verbose=0)
        y_pred_classes = np.argmax(y_pred_probs, axis=1)

        # Metrics
        tn, fp, fn, tp = confusion_matrix(y_true_classes, y_pred_classes).ravel()
        auc_score = roc_auc_score(y_true_classes, y_pred_probs[:, 1])
        aupr_score = average_precision_score(y_true_classes, y_pred_probs[:, 1])
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        # save the results to the list
        summary_metrics.append({
            'Model': model_name.replace('.keras', ''),
            'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn,
            'Accuracy': round(accuracy, 4),
            'Sensitivity': round(sensitivity, 4),
            'Specificity': round(specificity, 4),
            'AUC': round(auc_score, 4),
            'AUPR': round(aupr_score, 4)
        })

        # --- ROC FOR MODEL ---
        fpr, tpr, _ = roc_curve(y_true_classes, y_pred_probs[:, 1])
        ax_roc.plot(fpr, tpr, lw=2, label=f'{model_name.split(".")[0]} (AUC = {auc_score:.4f})')

        # --- PR_Recall---
        precision, recall, _ = precision_recall_curve(y_true_classes, y_pred_probs[:, 1])
        ax_pr.plot(recall, precision, lw=2, label=f'{model_name.split(".")[0]} (AUPR = {aupr_score:.4f})')

    # Roc
    ax_roc.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.05])
    ax_roc.set_xlabel('False Positive Rate (1 - Specificity)')
    ax_roc.set_ylabel('True Positive Rate (Sensitivity)')
    ax_roc.set_title('ROC Curves Comparison')
    ax_roc.legend(loc="lower right")
    ax_roc.grid(True)

    # PR
    ax_pr.set_xlim([0.0, 1.0])
    ax_pr.set_ylim([0.0, 1.05])
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-Recall Curves Comparison")
    ax_pr.legend(loc="lower left")
    ax_pr.grid(True)

    # Save
    fig.tight_layout()
    fig.savefig(path_result + "comparison_models_curves.png")
    print(f"\n Comparison plots for all models have been saved to: {path_result}comparison_models_curves.png")

    # --- OUTPUT RESULTS TO FILE EXCEL ---
    df_result = pd.DataFrame(summary_metrics)
    print("\n" + "=" * 50 + "\nSUMMARY TABLE OF COMPARATIVE RESULTS:\n" + "=" * 50)
    print(df_result.to_string(index=False))

    # Save to file excel
    excel_path = path_result + "comparison_all_models.xlsx"
    df_result.to_excel(excel_path, index=False)
    print(f"\n Detailed comparison table exported to file: {excel_path}")
    plt.show()