import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
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
    path_data = "dataset/"
    path_result = "2.result_wo_KD/"
    path_model = "2.model_wo_KD/"
    # Load full dataset
    X, y = load_dataset(path_data, "Train_Methylation.csv", 1)

    MAX_SEQUENCE_LENGTH = len(X[1])
    word_index1 = twoTupleDic1()
    vocab_size = len(word_index1)

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    all_metrics = []

    #  ROC curve
    fig, ax = plt.subplots(figsize=(8, 6))

    fold = 1
    for train_index, test_index in skf.split(X, np.argmax(y, axis=1)):
        print(f"\n==== Fold {fold} ====")
        Xtrain, Xtest = X[train_index], X[test_index]
        ytrain, ytest = y[train_index], y[test_index]

        # ----------------------------------------------------------
        # Class Weight for Fold base on ytrain
        # ----------------------------------------------------------
        y_train_classes = np.argmax(ytrain, axis=1)
        classes = np.unique(y_train_classes)
        weights = compute_class_weight(
            class_weight='balanced',
            classes=classes,
            y=y_train_classes
        )
        class_weight_dict = dict(zip(classes, weights))
        print(f"Class weights computed for Fold {fold}: {class_weight_dict}")
        # ----------------------------------------------------------
        # Callbacks baseline model
        weight_filepath = path_model + f"model357_baseline_fold{fold}.weights.h5"

        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,  # Bật restore_best_weights bình thường cho mô hình chuẩn
            mode='min'
        )

        checkpoint = ModelCheckpoint(
            filepath=weight_filepath,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True,
            mode='min',
            verbose=0
        )

        # ==========================================================
        # Init Baseline Model
        # ==========================================================
        model = create_model_3cnn_357(vocab_size, MAX_SEQUENCE_LENGTH)
        model.compile(
            optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-4),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        model.fit(
            Xtrain, ytrain,
            epochs=100,
            batch_size=16,
            validation_split=0.1,
            callbacks=[early_stopping, checkpoint],
            class_weight=class_weight_dict,
            verbose=1
        )
        print(f"\nLoading best weights from {weight_filepath} for evaluation...")
        model.load_weights(weight_filepath)

        # Predict
        y_pred_probs = model.predict(Xtest)
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
        # FPR, TPR
        fpr, tpr, _ = roc_curve(y_true_classes, y_pred_probs[:, 1])
        ax.plot(fpr, tpr, lw=2, label=f'Fold {fold} (AUC = {auc_score:.2f})')

        fold += 1

    # ==========================================================
    # OUTPUT RSULTS TO FILE EXCEL
    # ==========================================================
    metrics_df = pd.DataFrame(
        all_metrics,
        columns=['TP', 'FP', 'TN', 'FN', 'BA', 'AUC', 'AUPR']
    )
    metrics_df.loc['Mean'] = metrics_df.mean()

    excel_file = path_result + "10fold_baseline357.xlsx"
    metrics_df.to_excel(excel_file, index=False)
    print(f"\nKết quả đánh giá chéo đã lưu vào file {excel_file}")

    # ROC 10 Folds
    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves for Each Fold - Baseline Model')
    ax.legend(loc="lower right")
    ax.grid(True)

    fig.savefig(path_result + "10fold_roc_curve357_baseline.png")
    plt.show()