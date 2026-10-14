import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_recall_curve, auc, matthews_corrcoef
from units import *
from units import create_model_cnn357_BiLSTM
if __name__ == '__main__':
    path_data = "dataset/"
    path_result = "result_SKD_HPMCNet/"
    path_model = "model_SKD_HPMCNet/"

    os.makedirs(path_result, exist_ok=True)

    # 1. Define the PTM Type and Test Dataset
    ptm_name = "Sumoylation"
    test_filename = f"Test_{ptm_name}.csv"

    print(f"====Test: {ptm_name} =======")

    # 2. Load dataset test
    Xtest, ytest = load_dataset(path_data, test_filename, 1)

    max_sequence_length = Xtest.shape[1]
    word_index = twoTupleDic1()
    vocab_size = len(word_index)

    # 3. Initialize Model Architecture & Load Weights from the best model
    model_file = f"{ptm_name}_SKD_HPMCNet_fold5.weights.h5"
    model_path = os.path.join(path_model, model_file)

    # create model from
    model = create_model_cnn357_BiLSTM(vocab_size=vocab_size, max_sequence_length=max_sequence_length)

    # Load weights to the model
    model.load_weights(model_path)
    print(f"[+] Model initialized and weights successfully loaded from: {model_file}")

    # 4. Predict
    raw_logits = model.predict(Xtest)
    y_pred_probs = tf.nn.softmax(raw_logits, axis=-1).numpy()

    # 5. Processing Ground-Truth and Predicted Labels
    if ytest.ndim > 1 and ytest.shape[1] > 1:
        y_true_classes = np.argmax(ytest, axis=1)
    else:
        y_true_classes = ytest.ravel()

    y_scores = y_pred_probs[:, 1]
    y_pred_classes = np.argmax(y_pred_probs, axis=1)

    # 6. Metrics
    tn, fp, fn, tp = confusion_matrix(y_true_classes, y_pred_classes).ravel()

    auc_roc = roc_auc_score(y_true_classes, y_scores)

    precision, recall, _ = precision_recall_curve(y_true_classes, y_scores)
    auc_pr = auc(recall, precision)

    sn = tp / (tp + fn) if (tp + fn) > 0 else 0
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    mcc = matthews_corrcoef(y_true_classes, y_pred_classes)

    print("\n RESULTS")
    print(f"TP: {tp} | FP: {fp} | TN: {tn} | FN: {fn}")
    print(f"Sensitivity (Sn): {sn:.4f} | Specificity (Sp): {sp:.4f}")
    print(f"Accuracy (Acc)  : {acc:.4f} | MCC: {mcc:.4f}")
    print(f"ROC-AUC         : {auc_roc:.4f} | PR-AUC: {auc_pr:.4f}")

    # 7. Save and ouput the reslts to the Excel file
    results_df = pd.DataFrame({
        'Metric': ['TP', 'FP', 'TN', 'FN', 'Sensitivity (Sn)', 'Specificity (Sp)', 'Accuracy', 'MCC', 'ROC-AUC',
                   'PR-AUC'],
        'Value': [tp, fp, tn, fn, sn, sp, acc, mcc, auc_roc, auc_pr]
    })

    excel_path = os.path.join(path_result, f"predict_{ptm_name}_model_SKD_HPMCNet.xlsx")
    results_df.to_excel(excel_path, index=False)
    print(f"\nSave results to the Excel file: {excel_path}")

    # 8. Draw and save Precision-Recall Curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='red', lw=2, label=f'PR curve (AUC = {auc_pr:.4f})')
    plt.xlabel('Recall (Sensitivity)')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - Fold 5 ({ptm_name})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc="lower left")
    plt.grid(True)

    fig_path = os.path.join(path_result, f"{ptm_name}_model_SKD_HPMCNet.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"PR curve saved to: {fig_path}")