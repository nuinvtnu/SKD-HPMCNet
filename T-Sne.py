import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from sklearn.manifold import TSNE
from units import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

if __name__ == '__main__':
    # PTM list
    PTM_LIST = [
        #"N-linkedGlycosylation",
        #"Sumoylation",
        #"Hydroxylation",
        "Methylation",
        #"Glutarylation",
        #"Malonylations",
        #"Glutathionylation",
        #"Glutarylation"
    ]


    path_data = "dataset/"
    path_result = "result_SKD_HPMCNet/"
    path_model = "model_SKD_HPMCNet/"

    os.makedirs(path_result, exist_ok=True)

    # Lặp qua tất cả loại PTM
    for ptm_type in PTM_LIST:
        print(f"\n=====================")
        print(f"--> t-SNE PTM: {ptm_type}")
        print(f"==========================")

        file_data_name = f"Test_{ptm_type}.csv"
        file_weight_name = f"{ptm_type}_SKD_HPMCNet_fold4.weights.h5" # Mặc định lấy Fold 1 (hoặc đổi fold khác tùy chọn)

        path_csv = os.path.join(path_data, file_data_name)
        path_weight = os.path.join(path_model, file_weight_name)

        #
        if not os.path.exists(path_csv):
            print(f"[SKIP] Dont find the data: {path_csv}")
            continue
        if not os.path.exists(path_weight):
            print(f"[SKIP] Dont find the file weights: {path_weight}")
            continue

        # 1. Load Data
        X, y = load_dataset(path_data, file_data_name, 1)
        X_vis = X
        y_vis = np.argmax(y, axis=1)

        MAX_SEQUENCE_LENGTH = len(X[1])
        word_index1 = twoTupleDic1()
        vocab_size = len(word_index1)

        # 2. Init Model and Load Weights
        model = create_model_cnn357_BiLSTM(vocab_size, MAX_SEQUENCE_LENGTH)
        model(X_vis[:1]) # Build graph
        model.load_weights(path_weight)
        print(f" Weights loaded successfully from the file: {path_weight}")

        colors = ['#1f77b4', '#e377c2'] # Xanh da trời (Non-PTM) và Hồng (PTM)
        labels_map = {0: 'Non-PTM', 1: 'PTM Site'}

        # ==========================================================
        # === 1. Visualize BEFORE DISTILLATION (Input Features / Layer 1) ===
        # ==========================================================
        layer_input_idx = 1
        model_input_features = Model(
            inputs=model.inputs,
            outputs=model.get_layer(index=layer_input_idx).output
        )

        features_input = model_input_features.predict(X_vis, verbose=0)
        if len(features_input.shape) == 3:
            features_input = np.mean(features_input, axis=1)

        features_input_2d = TSNE(
            n_components=2,
            perplexity=30,
            max_iter=1000,
            random_state=42
        ).fit_transform(features_input)

        fig_in, ax_in = plt.subplots(figsize=(7, 6))
        for label_val in np.unique(y_vis):
            idx = (y_vis == label_val)
            ax_in.scatter(
                features_input_2d[idx, 0],
                features_input_2d[idx, 1],
                label=labels_map[label_val],
                alpha=0.6,
                s=25,
                color=colors[label_val % 2]
            )

        ax_in.set_title(f"t-SNE Input Feature Space ({ptm_type})", fontsize=12, fontweight='bold')
        ax_in.legend(loc='upper right')
        ax_in.grid(True, linestyle=':', alpha=0.5)

        # Save image from Input Embedding
        img_input_path = os.path.join(path_result, f"{ptm_type}_tsne_input_test_features.png")
        fig_in.savefig(img_input_path, dpi=300, bbox_inches='tight')
        plt.close(fig_in)
        print(f"--> [Saved] Image t-SNE Input: {img_input_path}")

        # ==========================================================
        # === 2. Visualize AFTER SKD (Pre-logits / High-level Features) ===
        # ==========================================================
        # Get the layer closest to the output
        model_output_features = Model(
            inputs=model.inputs,
            outputs=model.get_layer(index=-2).output
        )

        features_output = model_output_features.predict(X_vis, verbose=0)
        if len(features_output.shape) == 3:
            features_output = np.mean(features_output, axis=1)

        features_output_2d = TSNE(
            n_components=2,
            perplexity=30,
            max_iter=1000,
            random_state=42
        ).fit_transform(features_output)

        fig_out, ax_out = plt.subplots(figsize=(7, 6))
        for label_val in np.unique(y_vis):
            idx = (y_vis == label_val)
            ax_out.scatter(
                features_output_2d[idx, 0],
                features_output_2d[idx, 1],
                label=labels_map[label_val],
                alpha=0.6,
                s=25,
                color=colors[label_val % 2]
            )

        ax_out.set_title(f"t-SNE Class Separability with SKD_HPMCNet ({ptm_type})", fontsize=12, fontweight='bold')
        ax_out.legend(loc='upper right')
        ax_out.grid(True, linestyle=':', alpha=0.5)

        # save images of Separability Features
        img_output_path = os.path.join(path_result, f"{ptm_type}_tsne_learned_features_test.png")
        fig_out.savefig(img_output_path, dpi=300, bbox_inches='tight')
        plt.close(fig_out)
        print(f"[Saved] Image of t-SNE Learned Features: {img_output_path}")

    print("[COMPLETED] All t-SNE plots have been successfully saved in the result_SKD_HPMCNet directory!")