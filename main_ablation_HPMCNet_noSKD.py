import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc,
    average_precision_score
)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from units import *
from units_ablation import create_model_1cnn

if __name__ == '__main__':
    path_data = "dataset/"
    path_result = "result_KD/"
    path_model = "model_KD/"
    # LOAD DATA
    X, y = load_dataset(path_data, "Train_Methylation.csv", 1)
    X_independent, y_independent = load_dataset(path_data, "Test_Methylation.csv", 1)
    MAX_SEQUENCE_LENGTH = len(X[1])
    word_index1 = twoTupleDic1()
    vocab_size = len(word_index1)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    all_metrics = []
      fig, ax = plt.subplots(figsize=(8, 6))
    best_auc = 0.0
    best_weight_path = ""
    fold = 1
    for train_index, test_index in skf.split(X, np.argmax(y, axis=1)):
        print(f"\n==== Fold {fold} ====")
        Xtrain, Xtest = X[train_index], X[test_index]
        ytrain, ytest = y[train_index], y[test_index]

        early_stopping = EarlyStopping(
            monitor='val_student_loss',
            patience=20,
            restore_best_weights=False,
            mode='min'
        )

        # ==========================================================
        # Self Knowledge Distillation
        # ==========================================================
        student = create_model_3cnn_357(vocab_size, MAX_SEQUENCE_LENGTH)
        teacher = create_model_3cnn_357(vocab_size, MAX_SEQUENCE_LENGTH)

        student(Xtrain[:1])
        teacher(Xtrain[:1])
        teacher.set_weights(student.get_weights())
        teacher.trainable = False
        class SaveStudentWeightsCallback(keras.callbacks.Callback):
            def __init__(self, filepath, monitor='val_student_loss'):
                super().__init__()
                self.filepath = filepath
                self.monitor = monitor
                self.best_loss = float('inf')

            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                current_loss = logs.get(self.monitor)
                if current_loss is not None:
                    if current_loss < self.best_loss:
                        self.best_loss = current_loss
                        self.model.student.save_weights(self.filepath)
                        print(
                            f"\nEpoch {epoch + 1}: {self.monitor} improved to {current_loss:.4f}, saving student weights.")
                else:
                    print(f"\nWarning: Custom callback không tìm thấy metric {self.monitor} trong logs!")


        weight_filepath = path_model + f"model357_selfkd_fold{fold}.weights.h5"
        checkpoint = SaveStudentWeightsCallback(
            filepath=weight_filepath,
            monitor="val_student_loss"
        )

        distiller = Distiller(
            student=student,
            teacher=teacher,
            temperature=3,
            alpha=0.7
        )
        distiller.compile(
            optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-4)
        )

        distiller.fit(
            Xtrain, ytrain,
            epochs=100, batch_size=16, validation_split=0.1,
            callbacks=[early_stopping, checkpoint],
            verbose=1
        )

        print(f"\nLoading best student weights from {weight_filepath} for evaluation...")
        distiller.student.load_weights(weight_filepath)
        teacher.set_weights(distiller.student.get_weights())
        # Dự đoán validation fold
        y_pred_logits = distiller.student.predict(Xtest)
        y_pred_probs = tf.nn.softmax(y_pred_logits, axis=-1).numpy()
        y_pred_classes = np.argmax(y_pred_probs, axis=1)
        y_true_classes = np.argmax(ytest, axis=1)
        tn, fp, fn, tp = confusion_matrix(y_true_classes, y_pred_classes).ravel()
        auc_score = roc_auc_score(y_true_classes, y_pred_probs[:, 1])
        aupr_score = average_precision_score(y_true_classes, y_pred_probs[:, 1])
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        ba_score = (sensitivity + specificity) / 2
        print(f"Fold {fold} - BA: {ba_score:.4f} | AUC: {auc_score:.4f} | AUPR: {aupr_score:.4f}")
        all_metrics.append([tp, fp, tn, fn, ba_score, auc_score, aupr_score])

        if auc_score > best_auc:
            best_auc = auc_score
            best_weight_path = weight_filepath

        fpr, tpr, _ = roc_curve(y_true_classes, y_pred_probs[:, 1])
        ax.plot(fpr, tpr, lw=2, label=f'Fold {fold} (AUC = {auc_score:.2f})')

        fold += 1

    metrics_df = pd.DataFrame(all_metrics, columns=['TP', 'FP', 'TN', 'FN', 'BA', 'AUC', 'AUPR'])
    metrics_df.loc['Mean'] = metrics_df.mean()
    excel_file = path_result + "10fold_kd357_0.3_7.xlsx"
    metrics_df.to_excel(excel_file, index=False)
    print(f"\n Cross-validation results to the file {excel_file}")

    # ROC 10 Folds
    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves for Each Fold - Student Model')
    ax.legend(loc="lower right")
    ax.grid(True)
    fig.savefig(path_result + "10fold_roc_curve357_0.3_7.png")
    plt.close(fig)

    # ==========================================================
    # INDEPENDENT TEST
    # ==========================================================
    print(f"\n==================================================")
    print(f"INDEPENDENT TEST")
    print(f"Best weight from: {best_weight_path} (AUC: {best_auc:.4f})")
    print(f"==================================================")

    final_student = create_model_3cnn_357(vocab_size, MAX_SEQUENCE_LENGTH)
    final_student(X_independent[:1])  # Build mô hình
    final_student.load_weights(best_weight_path)

    # predict
    ind_pred_logits = final_student.predict(X_independent)
    ind_pred_probs = tf.nn.softmax(ind_pred_logits, axis=-1).numpy()
    ind_pred_classes = np.argmax(ind_pred_probs, axis=1)
    ind_true_classes = np.argmax(y_independent, axis=1)
    # Metrics on the Test Set
    ind_tn, ind_fp, ind_fn, ind_tp = confusion_matrix(ind_true_classes, ind_pred_classes).ravel()
    ind_auc = roc_auc_score(ind_true_classes, ind_pred_probs[:, 1])
    ind_aupr = average_precision_score(ind_true_classes, ind_pred_probs[:, 1])

    ind_sensitivity = ind_tp / (ind_tp + ind_fn) if (ind_tp + ind_fn) > 0 else 0
    ind_specificity = ind_tn / (ind_tn + ind_fp) if (ind_tn + ind_fp) > 0 else 0
    ind_ba = (ind_sensitivity + ind_specificity) / 2

    # Print the results
    print("\n--- Result of Independent test ---")
    print(f"TP: {ind_tp} | FP: {ind_fp} | TN: {ind_tn} | FN: {ind_fn}")
    print(f"Balanced Accuracy (BA): {ind_ba:.4f}")
    print(f"AUC: {ind_auc:.4f}")
    print(f"AUPR: {ind_aupr:.4f}")

    independent_metrics = {
        'Metric': ['TP', 'FP', 'TN', 'FN', 'BA', 'AUC', 'AUPR'],
        'Value': [ind_tp, ind_fp, ind_tn, ind_fn, ind_ba, ind_auc, ind_aupr]
    }
    ind_df = pd.DataFrame(independent_metrics)
    ind_excel_file = path_result + "independent_test_result.xlsx"
    ind_df.to_excel(ind_excel_file, index=False)
    print(f"Kết quả bộ test độc lập đã được xuất ra: {ind_excel_file}")

    # ROC
    fig_ind, ax_ind = plt.subplots(figsize=(8, 6))
    ind_fpr, ind_tpr, _ = roc_curve(ind_true_classes, ind_pred_probs[:, 1])
    ax_ind.plot(ind_fpr, ind_tpr, color='darkorange', lw=2, label=f'Independent Test (AUC = {ind_auc:.4f})')
    ax_ind.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    ax_ind.set_xlim([0.0, 1.0])
    ax_ind.set_ylim([0.0, 1.05])
    ax_ind.set_xlabel('False Positive Rate')
    ax_ind.set_ylabel('True Positive Rate')
    ax_ind.set_title('ROC Curve - Independent Test')
    ax_ind.legend(loc="lower right")
    ax_ind.grid(True)
    fig_ind.savefig(path_result + "independent_roc_curve.png")
    plt.show()