**SKD-HPMCNet:** An Efficient Self-Knowledge Distillation Framework with Class Weighting for Protein Post-Translational Modification Site Prediction.

**Abstract:**
Post-translational modifications (PTMs) are essential regulatory mechanisms that influence protein stability, interactions, and cellular signaling. Reliable prediction of PTM sites, however, remains difficult due to the limited availability of labeled data and the inherent class imbalance in most PTM datasets. In this study, we propose a Self-Knowledge Distillation (SKD-HPMCNet) framework for PTM site prediction, in which the teacher and student share the same architecture and parameters. Unlike conventional knowledge distillation approaches that require large teacher models, SKD-HPMCNet enables the model to refine its own predictions by jointly learning from hard labels and soft probability distributions. To further address the impact of imbalanced data, class weighting was incorporated into the loss function. Experimental evaluation was performed using curated benchmark datasets from dbPTM (accessed 20 August 2025), focusing on PTMs with mild class imbalance. The proposed framework is expected to improve generalization, reduce overfitting on small or imbalanced datasets, and provide a benchmark-driven approach for systematically evaluating PTM prediction tools.

Project Structure:
**Selt_KD_PTM/**
├── dataset/                         # Training and validation datasets
├── Dataset_test_Tools/              # Independent test datasets for benchmarking tools
├── Model_ablation/                  # Saved models for ablation studies (.keras and .h5)
├── model_Alpha_Tau/                 # Saved models with varying Alpha and Tau parameters
├── model_SKD_HPMCNet/               # Saved main SKD-HPMCNet model weights (.weights.h5)
├── result_ablation/                 # Evaluation metrics, ROC & PR curves for ablation study
├── result_Alpha_Tau/                # Hyperparameter tuning results (Alpha & Tau)
├── result_HPMCNet (wo SKD)/         # Baseline results without Self-Knowledge Distillation
├── result_SKD_HPMCNet/              # Main prediction results, ROC & PR curves
│
├── units.py                         # Core utility functions & main HPMCNet architecture
├── units_ablation.py                # Helper functions for ablation model architectures
│
├── main_SKD_HPMCNet.py              # Main training script using SKD with 10-fold CV
├── SKD_HPMCNet_predict.py           # Evaluation script for predicting on test dataset
│
├── HPMCNet_Class Weight (wo SKD).py # Baseline training script without SKD
├── main_ablation_HPMCNet_noSKD.py   # Ablation study: HPMCNet without SKD framework
├── main_ablation_SKD-Parallel CNN (wo BiLSTM).py # Ablation study: SKD without BiLSTM layer
├── main_ablation_SKD_BiLSTM (wo CNN) .py        # Ablation study: SKD without CNN layers
├── Main_ablation_SKD_HPMCNet_(Alpha,tau).py     # Hyperparameter grid search for Alpha and Tau
├── ablation_predict.py              # Script to compare and plot performance of all ablation models
│
├── PR_AUC.py                        # Precision-Recall curve generation and plotting
├── T-Sne.py                         # t-SNE feature representation visualization

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
    
