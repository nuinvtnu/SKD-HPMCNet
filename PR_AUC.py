import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
from tensorflow.keras import models
from units import *
import os
# ===== 1. Load all trained models =====
if __name__ == '__main__':
    path_data = "dataset/"
    path_result = "result/"
    path_model = "model/"
    #model_sumo  = models.load_model(path_model + "SKD_Sumo.h5")
    #model_hybrid_sumo = models.load_model(path_model + "Hybrid_DL_Sumo.h5")
    #model_methy   = models.load_model(path_model + "SKD_Methylation.h5")
    #model_hybrid_methy= models.load_model(path_model + "Hybrid_DL_Methylation.h5")
    #model_N_link   = models.load_model(path_model + "SKD_N-linkedGlycosylation.h5")
    #model_hybrid_N_link = models.load_model(path_model + "Hybrid_DL_N-linkedGlycosylation.h5")
    #model_Glu  = models.load_model(path_model + "SKD_Glutarylation.h5")
    #model_hybrid_Glu = models.load_model(path_model + "Hybrid_DL_Glutarylation.h5")
    #model_Glutathio = models.load_model(path_model + "SKD_Glutathionylation.h5")
    #model_hybrid_Glutathio = models.load_model(path_model + "Hybrid_DL_Glutathionylation.h5")
    #model_Hydro = models.load_model(path_model + "SKD_Hydroxylation.h5")
    #model_hybrid_Hydro = models.load_model(path_model + "Hybrid_DL_Hydroxylation.h5")
    model_Malony = models.load_model(path_model + "SKD_Malonylation.h5")
    model_hybrid_Malony = models.load_model(path_model + "Hybrid_DL_Malonylation.h5")


    # ===== 2. Predict probabilities on the test set =====
    models_dict = {
        #"Self-KD-SUMO": model_sumo,
        #"Baseline Hybrid SUMO": model_hybrid_sumo,
        #"SKD-Methylation": model_methy,
        #"Baseline Hybrid-Methylation": model_hybrid_methy,
        #"Self-KD-N-linkedGlycosylation": model_N_link ,
        #"Baseline Hybrid-N-linkedGlycosylation": model_hybrid_N_link,
        #"SKD-Glutarylation": model_Glu,
        #"Baseline Hybrid - Glutarylation": model_hybrid_Glu,
        #"SKD-Glutathionylation": model_Glutathio,
        #"Baseline Hybrid - Glutathionylation": model_hybrid_Glutathio,

        "SKD_Malonylation": model_Malony,
        "Baseline Hybrid_Malonylation": model_hybrid_Malony,
        #"SKD-Malonylation": model_Malony,

    }
    Xtest, ytest = load_dataset(path_data, "Test_Malonylation.csv", 1)
    MAX_SEQUENCE_LENGTH = len(Xtest[1])
    word_index1 = twoTupleDic1()
    vocab_size = len(word_index1)
    y_true = np.array(ytest)  # true labels (one-hot or binary)
    if y_true.ndim > 1:
        y_true = np.argmax(y_true, axis=1)

    # ===== 3. Compute Precision, Recall, and Average Precision (AP) =====
    pr_curves = {}
    for name, model in models_dict.items():
        y_prob = model.predict(Xtest)
        y_score = y_prob[:, 1] if y_prob.shape[1] > 1 else y_prob.ravel()

        precision, recall, _ = precision_recall_curve(y_true, y_score)
        avg_prec = average_precision_score(y_true, y_score)
        pr_curves[name] = (precision, recall, avg_prec)

    # ===== 4. Plot all Precision–Recall Curves =====
    plt.figure(figsize=(8, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(pr_curves)))

    for (name, (precision, recall, avg_prec)), color in zip(pr_curves.items(), colors):
        if "SKD" in name:
            plt.plot(recall, precision, color='red', lw=3.5,
                     label=f'{name}, AP = {avg_prec:.3f}', zorder=10)
        else:
            plt.plot(recall, precision, color=color, lw=1.8,
                     label=f'{name}, AP = {avg_prec:.3f}')

    # Baseline: random classifier
    plt.plot([0, 1], [0.5, 0.5], color='gray', linestyle='--', lw=1.5)

    # ===== 5. Chart formatting =====
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision–Recall Curves of All Model on the test Set',
              fontsize=14, fontweight='bold')

    legend = plt.legend(loc="lower left", fontsize=11)
    for text in legend.get_texts():
        if 'SKD' in text.get_text():
            text.set_fontweight('bold')  # in đậm tên mô hình đề xuất


    plt.grid(alpha=0.3)

    # ===== Save figure =====
    save_path = os.path.join(path_result, "PR_curve_Malonylation.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    print(f"PR curve saved to: {save_path}")

    plt.show()