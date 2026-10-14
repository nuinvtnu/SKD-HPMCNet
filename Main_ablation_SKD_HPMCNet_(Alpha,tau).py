import os
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
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from units import *

if __name__ == '__main__':
    PTM_TYPE = "Methylation"  # <-- ĐỔI THÀNH "Acetylation", "Nlink", v.v.
    ALPHA_LIST = [0.1]#, 0.5, 0.7]
    TAU_LIST =   [1.0, 3.0]
    path_data = "dataset/"
    path_result = "result_Alpha_Tau/"
    path_model = "model_Alpha_Tau/"
    os.makedirs(path_result, exist_ok=True)
    os.makedirs(path_model, exist_ok=True)

    # Load full dataset
    file_data_name = f"Train_{PTM_TYPE}.csv"
    print(f"Runing Grid Search cho PTM: {PTM_TYPE}")
    print(f"File input: {path_data}{file_data_name}")

    X, y = load_dataset(path_data, file_data_name, 1)

    MAX_SEQUENCE_LENGTH = len(X[1])
    word_index1 = twoTupleDic1()
    vocab_size = len(word_index1)
    summary_results = []

    # ==========================================================
    # ALPHA AND TAU
    # ==========================================================
    for alpha in ALPHA_LIST:
        for tau in TAU_LIST:
            print("\n" + "="*70)
            print(f"  RUNING: ALPHA = {alpha} | TAU (Temperature) = {tau}")
            print("="*70)

            skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
            all_metrics = []


            fig, ax = plt.subplots(figsize=(8, 6))
            fold = 1
            for train_index, test_index in skf.split(X, np.argmax(y, axis=1)):
                print(f"\n---> [Alpha: {alpha} | Tau: {tau}] Fold {fold}/10 ({PTM_TYPE})")
                Xtrain, Xtest = X[train_index], X[test_index]
                ytrain, ytest = y[train_index], y[test_index]

                # ----------------------------------------------------------
                # Class Weight  Fold base on ytrain
                # ----------------------------------------------------------
                y_train_classes = np.argmax(ytrain, axis=1)
                classes = np.unique(y_train_classes)
                weights = compute_class_weight(
                    class_weight='balanced',
                    classes=classes,
                    y=y_train_classes
                )
                class_weight_dict = dict(zip(classes, weights))

                early_stopping = EarlyStopping(
                    monitor='val_student_loss',
                    patience=20,
                    restore_best_weights=False,
                    mode='min'
                )

                # ==========================================================
                # Student & Teacher Models
                # ==========================================================
                student = create_model_cnn357_BiLSTM(vocab_size, MAX_SEQUENCE_LENGTH)
                teacher = create_model_cnn357_BiLSTM(vocab_size, MAX_SEQUENCE_LENGTH)

                # Build  sample dummy
                student(Xtrain[:1])
                teacher(Xtrain[:1])

                # Synchronize initial weights
                teacher.set_weights(student.get_weights())
                teacher.trainable = False

                # Custom Callback save best weights
                class SaveStudentWeightsCallback(keras.callbacks.Callback):
                    def __init__(self, filepath, monitor='val_student_loss'):
                        super().__init__()
                        self.filepath = filepath
                        self.monitor = monitor
                        self.best_loss = float('inf')

                    def on_epoch_end(self, epoch, logs=None):
                        logs = logs or {}
                        current_loss = logs.get(self.monitor)
                        if current_loss is not None and current_loss < self.best_loss:
                            self.best_loss = current_loss
                            self.model.student.save_weights(self.filepath)

                # file name
                weight_filepath = os.path.join(
                    path_model,
                    f"{PTM_TYPE}_a{alpha}_t{tau}_fold{fold}.weights.h5"
                )
                checkpoint = SaveStudentWeightsCallback(
                    filepath=weight_filepath,
                    monitor="val_student_loss"
                )

                # alpha, tau to Distiller
                distiller = Distiller(
                    student=student,
                    teacher=teacher,
                    temperature=tau,
                    alpha=alpha
                )

                distiller.compile(
                    optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-4)
                )

                # Train model
                distiller.fit(
                    Xtrain, ytrain,
                    epochs=100, batch_size=16, validation_split=0.1,
                    callbacks=[early_stopping, checkpoint],
                    class_weight=class_weight_dict,
                    verbose=0  # Để verbose=0 cho gọn console khi chạy grid search
                )

                # Load the best weights
                distiller.student.load_weights(weight_filepath)
                teacher.set_weights(distiller.student.get_weights())

                # Predict
                y_pred_logits = distiller.student.predict(Xtest, verbose=0)
                y_pred_probs = tf.nn.softmax(y_pred_logits, axis=-1).numpy()

                y_pred_classes = np.argmax(y_pred_probs, axis=1)
                y_true_classes = np.argmax(ytest, axis=1)

                # Confusion Matrix
                tn, fp, fn, tp = confusion_matrix(y_true_classes, y_pred_classes).ravel()
                auc_score = roc_auc_score(y_true_classes, y_pred_probs[:, 1])
                aupr_score = average_precision_score(y_true_classes, y_pred_probs[:, 1])

                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                ba_score = (sensitivity + specificity) / 2

                # F1-score and MCC
                f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
                mcc_denom = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
                mcc = ((tp * tn) - (fp * fn)) / mcc_denom if mcc_denom > 0 else 0

                print(f"Fold {fold} -> SEN: {sensitivity:.4f} | SPE: {specificity:.4f} | BA: {ba_score:.4f} | MCC: {mcc:.4f} | AUC: {auc_score:.4f} | AUPR: {aupr_score:.4f}")

                all_metrics.append([tp, fp, tn, fn, sensitivity, specificity, ba_score, mcc, f1, auc_score, aupr_score])

                # ROC Curve
                fpr, tpr, _ = roc_curve(y_true_classes, y_pred_probs[:, 1])
                ax.plot(fpr, tpr, lw=1, alpha=0.7, label=f'Fold {fold} (AUC = {auc_score:.2f})')

                fold += 1

            # ==========================================================
            # SAVE OUTPUT (ALPHA, TAU)
            # ==========================================================
            metrics_df = pd.DataFrame(
                all_metrics,
                columns=['TP', 'FP', 'TN', 'FN', 'SEN', 'SPE', 'BA', 'MCC', 'F1', 'AUC', 'AUPR']
            )
            mean_row = metrics_df.mean()
            metrics_df.loc['Mean'] = mean_row

            excel_file = os.path.join(path_result, f"{PTM_TYPE}_10fold_alpha{alpha}_tau{tau}.xlsx")
            metrics_df.to_excel(excel_file, index=False)
            print(f"\n Saved results of 10-fold: {excel_file}")
            summary_results.append({
                'Alpha': alpha,
                'Tau': tau,
                'SEN': mean_row['SEN'],
                'SPE': mean_row['SPE'],
                'BA': mean_row['BA'],
                'MCC': mean_row['MCC'],
                'F1': mean_row['F1'],
                'AUC': mean_row['AUC'],
                'AUPR': mean_row['AUPR']
            })

            #ROC
            ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'ROC Curves ({PTM_TYPE}) - Alpha: {alpha}, Tau: {tau}')
            ax.legend(loc="lower right", fontsize='small')
            ax.grid(True)
            roc_png_file = os.path.join(path_result, f"{PTM_TYPE}_roc_alpha{alpha}_tau{tau}.png")
            fig.savefig(roc_png_file)
            plt.close(fig)
            tf.keras.backend.clear_session()

    # ==========================================================
    # Ouput file include all ALPHA and TAU
    # ==========================================================
    summary_df = pd.DataFrame(summary_results)
    summary_df = summary_df.sort_values(by='AUPR', ascending=False)
    summary_excel_file = os.path.join(path_result, f"{PTM_TYPE}_GRID_SEARCH_SUMMARY.xlsx")
    summary_df.to_excel(summary_excel_file, index=False)
    print("\n" + "="*70)
    print(f"COMPLETE GRID SEARCH FOR {PTM_TYPE}!")
    print(f"Table of Alpha/Tau saved at:")
    print(f" {summary_excel_file}")
    print("="*70)
    print("\nTop 5 of AUPR:")
    print(summary_df.head(5).to_string(index=False))