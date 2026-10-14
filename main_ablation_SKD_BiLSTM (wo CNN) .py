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
    PTM_TYPE = "Methylation"  # "Nlink", "Methylation", v.v.
    path_data = "dataset/"
    path_result = "result_ablation/"
    path_model = "Model_ablation/"
    os.makedirs(path_result, exist_ok=True)
    os.makedirs(path_model, exist_ok=True)
    file_data_name = f"Train_{PTM_TYPE}.csv"
    print(f" Runing Self-KD for PTM: {PTM_TYPE}")
    print(f"File input: {path_data}{file_data_name}")
    # Load full dataset
    X, y = load_dataset(path_data, file_data_name, 1)
    MAX_SEQUENCE_LENGTH = len(X[1])
    word_index1 = twoTupleDic1()
    vocab_size = len(word_index1)

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    all_metrics = []
    fig, ax = plt.subplots(figsize=(8, 6))

    fold = 1
    for train_index, test_index in skf.split(X, np.argmax(y, axis=1)):
        print(f"\n==== Fold {fold} ({PTM_TYPE}) ====")
        Xtrain, Xtest = X[train_index], X[test_index]
        ytrain, ytest = y[train_index], y[test_index]
        y_train_classes = np.argmax(ytrain, axis=1)
        classes = np.unique(y_train_classes)
        weights = compute_class_weight(
            class_weight='balanced',
            classes=classes,
            y=y_train_classes
        )
        class_weight_dict = dict(zip(classes, weights))
        print(f"Class weights computed for Fold {fold}: {class_weight_dict}")
        early_stopping = EarlyStopping(
            monitor='val_student_loss',
            patience=20,
            restore_best_weights=False,
            mode='min'
        )

        # ==========================================================
        # Self Knowledge Distillation
        # ==========================================================
        student = create_model_bilstm(vocab_size, MAX_SEQUENCE_LENGTH)
        teacher = create_model_bilstm(vocab_size, MAX_SEQUENCE_LENGTH)
        student(Xtrain[:1])
        teacher(Xtrain[:1])
        teacher.set_weights(student.get_weights())
        teacher.trainable = False
        #  Student
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

        weight_filepath = os.path.join(path_model, f"{PTM_TYPE}_model_abso_BiLSTM_fold{fold}.weights.h5")
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

        # train
        distiller.fit(
            Xtrain, ytrain,
            epochs=100, batch_size=16, validation_split=0.1,
            callbacks=[early_stopping, checkpoint],
            class_weight=class_weight_dict,
            verbose=1
        )

        # Load the best weight to Student
        print(f"\nLoading best student weights from {weight_filepath} for evaluation...")
        distiller.student.load_weights(weight_filepath)

        # update Teacher from Student
        teacher.set_weights(distiller.student.get_weights())

        # Predict
        y_pred_logits = distiller.student.predict(Xtest)
        y_pred_probs = tf.nn.softmax(y_pred_logits, axis=-1).numpy()

        y_pred_classes = np.argmax(y_pred_probs, axis=1)
        y_true_classes = np.argmax(ytest, axis=1)

        # Confusion Matrix
        tn, fp, fn, tp = confusion_matrix(y_true_classes, y_pred_classes).ravel()

        # AUC and AUPR
        auc_score = roc_auc_score(y_true_classes, y_pred_probs[:, 1])
        aupr_score = average_precision_score(y_true_classes, y_pred_probs[:, 1])

        #  Balanced Accuracy (BA)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        ba_score = (sensitivity + specificity) / 2

        print(f"Fold {fold} - BA: {ba_score:.4f} | AUC: {auc_score:.4f} | AUPR: {aupr_score:.4f}")

        all_metrics.append([tp, fp, tn, fn, ba_score, auc_score, aupr_score])

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true_classes, y_pred_probs[:, 1])
        ax.plot(fpr, tpr, lw=2, label=f'Fold {fold} (AUC = {auc_score:.2f})')

        fold += 1

    # ==========================================================
    # OUTPUT THE RESULT TO FILE EXCEL
    # ==========================================================
    metrics_df = pd.DataFrame(
        all_metrics,
        columns=['TP', 'FP', 'TN', 'FN', 'BA', 'AUC', 'AUPR']
    )
    metrics_df.loc['Mean'] = metrics_df.mean()

    excel_file = os.path.join(path_result, f"{PTM_TYPE}_10fold_abso_BiLSTM.xlsx")
    metrics_df.to_excel(excel_file, index=False)
    print(f"\n Cross-validation of {PTM_TYPE} đã lưu vào file {excel_file}")

    #  ROC 10 Folds
    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curves for Each Fold ({PTM_TYPE}) - Student Model')
    ax.legend(loc="lower right")
    ax.grid(True)

    roc_png_file = os.path.join(path_result, f"{PTM_TYPE}_10fold_roc_abso_BiLSTM.png")
    fig.savefig(roc_png_file)
    plt.show()