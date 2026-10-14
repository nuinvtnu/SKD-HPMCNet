**SKD-HPMCNet:** An Efficient Self-Knowledge Distillation Framework with Class Weighting for Protein Post-Translational Modification Site Prediction.

**Abstract:**
Post-translational modifications (PTMs) are essential regulatory mechanisms that influence protein stability, interactions, and cellular signaling. Reliable prediction of PTM sites, however, remains difficult due to the limited availability of labeled data and the inherent class imbalance in most PTM datasets. In this study, we propose a Self-Knowledge Distillation (SKD-HPMCNet) framework for PTM site prediction, in which the teacher and student share the same architecture and parameters. Unlike conventional knowledge distillation approaches that require large teacher models, SKD-HPMCNet enables the model to refine its own predictions by jointly learning from hard labels and soft probability distributions. To further address the impact of imbalanced data, class weighting was incorporated into the loss function. Experimental evaluation was performed using curated benchmark datasets from dbPTM (accessed 20 August 2025), focusing on PTMs with mild class imbalance. The proposed framework is expected to improve generalization, reduce overfitting on small or imbalanced datasets, and provide a benchmark-driven approach for systematically evaluating PTM prediction tools.

## Project Structure
The repository is organized as follows:
```text
Selt_KD_PTM/
│
├── dataset/                          # Training and validation datasets
├── Dataset_test_Tools/               # Independent benchmark test datasets
│
├── Model_ablation/                   # Trained models for ablation studies
├── model_Alpha_Tau/                  # Models with different alpha/tau settings
├── model_SKD_HPMCNet/                # Trained SKD-HPMCNet model weights
│
├── result_ablation/                  # Ablation study results and plots
├── result_Alpha_Tau/                 # Hyperparameter optimization results
├── result_HPMCNet (wo SKD)/          # Baseline results without SKD
├── result_SKD_HPMCNet/               # Main evaluation results
│
├── units.py                          # Core utilities and HPMCNet architecture
├── units_ablation.py                 # Ablation model architectures
│
├── main_SKD_HPMCNet.py               # Main SKD training with 10-fold CV
├── SKD_HPMCNet_predict.py            # Independent test evaluation
│
├── HPMCNet_Class Weight (wo SKD).py  # Baseline training without SKD
├── main_ablation_HPMCNet_noSKD.py    # HPMCNet without self-distillation
├── main_ablation_SKD-Parallel CNN (wo BiLSTM).py
│                                     # CNN-only ablation experiment
├── main_ablation_SKD_BiLSTM (wo CNN) .py
│                                     # BiLSTM-only ablation experiment
│
├── Main_ablation_SKD_HPMCNet_(Alpha,tau).py
│                                     # Alpha/tau hyperparameter search
├── ablation_predict.py               # Ablation model evaluation and plots
├── PR_AUC.py                         # Precision–Recall curve visualization
└── T-Sne.py                          # t-SNE feature visualization
```

**Requirements & Installation:** tensorflow>=2.15.0,numpy, pandas, matplotlib, scikit-learn

**Usage Guide:**
1. Training SKD-HPMCNet: To train the main SKD-HPMCNet model with 10-fold Stratified Cross-Validation
    Code: main_SKD_HPMCNet.py
    Results: result_SKD_HPMCNet
2. Model Evaluation: To evaluate a trained SKD-HPMCNet model checkpoint on independent test data
    python SKD_HPMCNet_predict.py
    Feature Visualization: python T-Sne.py
    Results: result_SKD_HPMCNet
3. Ablation Studies & Model Comparison: To run training and comparison scripts for ablation studies (testing individual components such as CNN-only, BiLSTM-only, or no-SKD baselines)
    Code: HPMCNet_Class Weight (wo SKD).py; main_ablation_HPMCNet_noSKD.py; main_ablation_SKD-Parallel CNN (wo BiLSTM).py; main_ablation_SKD_BiLSTM (wo CNN) .py; 
    The results save in folders: Model_ablaiton;result_ablation
4. Hyperparameter Search (Alpha & Tau): 
    code: Main_ablation_SKD_HPMCNet_(Alpha,tau).py
    The results save in folder: result_Alpha_tau

**Framework Highlights:**
- Self-Knowledge Distillation (SKD): Teacher and student models share identical architecture, refining prediction distributions without relying on a larger teacher model.
- Class Weighting Strategy: Cost-sensitive loss adjustment mitigates severe class imbalance in PTM site datasets.
- Hybrid Multi-Scale Feature Extraction: Parallel 1D CNNs with varying kernel sizes (3, 5, 7) combined with BiLSTM for extracting local sequence motifs and long-range contextual dependencies.
    
