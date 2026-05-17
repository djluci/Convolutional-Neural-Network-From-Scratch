# 🧠 Single-Layer Neural Networks: ADALINE, Perceptron & Logistic Regression

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-From%20Scratch-lightblue?logo=numpy&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![Course](https://img.shields.io/badge/CS343-Neural%20Networks-purple)

> From-scratch NumPy implementations of ADALINE, Perceptron, and Logistic Regression single-layer neural networks — applied to binary classification and regression with full training analysis, decision boundary visualization, and a gated feature selection extension.

**Authors:** Duilio Lucio & Vivian Hu — CS343: Neural Networks, Spring 2026

---

## 📌 Table of Contents
- [Overview](#-overview)
- [Repository Structure](#-repository-structure)
- [Datasets](#-datasets)
- [Model Implementations](#-model-implementations)
- [Notebooks](#-notebooks)
- [Extension: Gated Logistic Regression](#-extension-gated-logistic-regression)
- [Key Findings](#-key-findings)
- [Requirements](#-requirements)
- [Usage](#-usage)

---

## 🔍 Overview

This project explores the foundations of neural networks through single-layer architectures — the same fundamental building blocks found in deep networks. All models are implemented from scratch in NumPy with no ML libraries used for training.

The project covers:
- Binary classification with ADALINE and Perceptron on real datasets
- Linear regression using the same ADALINE architecture
- Logistic regression with sigmoid activation and cross-entropy loss
- The effect of feature scaling and learning rate on training stability
- Decision boundary and class boundary visualization
- **Extension:** Learnable per-feature gates for automatic feature selection + 5-fold stratified cross-validation

---

## 📂 Repository Structure

```
.
├── adaline.py                        # ADALINE and Perceptron implementations
├── adaline_logistic.py               # AdalineLogistic subclass (sigmoid + cross-entropy)
├── adaline_logistic_regression.py    # Extension: AdalineLogistic + AdalineGatedLogistic
├── k_fold.py                         # Stratified K-fold cross-validation
├── binary_classification.ipynb       # Tasks 1–3 & 5: classification analysis
├── regression.ipynb                  # Task 4: ADALINE for linear regression
├── extension.ipynb                   # Gated logistic regression + cross-validation
├── old_faithful.csv                  # Old Faithful geyser dataset
├── ionosphere.csv                    # JHU Ionosphere radar dataset
├── ionosphere_info.txt               # Dataset documentation
├── extension.mp4                     # Video walkthrough of extension
└── README.md
```

---

## 📊 Datasets

### Old Faithful Geyser (`old_faithful.csv`)
| Detail | Value |
|---|---|
| Samples | 272 |
| Features | `eruptions` (duration in mins), `waiting` (time to next eruption) |
| Task | Binary classification — `severe` (+1) vs not severe (−1) |
| Also used for | Linear regression: predict `waiting` from `eruptions` |

### JHU Ionosphere (`ionosphere.csv`)
| Detail | Value |
|---|---|
| Samples | 351 |
| Features | 34 continuous radar signal attributes |
| Task | Binary classification — `good` vs `bad` radar returns |
| Source | Space Physics Group, Johns Hopkins University (1989) |

---

## ⚙️ Model Implementations

All models share an ADALINE-style API: `fit`, `predict`, `activation`, `loss`, `net_input`.

### `Adaline` — `adaline.py`
- **Activation:** Identity — `f(x) = x`
- **Loss:** Sum of Squared Error (SSE) — `½ Σ(y − ŷ)²`
- **Update rule:** Delta rule (gradient descent)
- **Classes:** −1 / +1

### `Perceptron` — `adaline.py` *(subclass of Adaline)*
- **Activation:** Step function — `+1 if net_in ≥ 0, else −1`
- Everything else inherited from Adaline

### `AdalineLogistic` — `adaline_logistic.py` *(subclass of Adaline)*
- **Activation:** Sigmoid — `σ(z) = 1 / (1 + e⁻ᶻ)`
- **Loss:** Binary cross-entropy — `−Σ[y·log(ẑ) + (1−y)·log(1−ẑ)]`
- **Classes:** 0 / 1 (threshold at 0.5)

### `AdalineGatedLogistic` — `adaline_logistic_regression.py` *(extension, subclass of AdalineLogistic)*
- Adds a learnable per-feature gate vector `gates = σ(g_raw) ∈ (0, 1)`
- Modifies net input: `net_in = (X * gates) @ wts + b`
- Optional L1 penalty on gates drives sparse feature selection
- Gates and weights are jointly optimized via gradient descent

### `kfold_cv` — `k_fold.py`
- Stratified K-fold cross-validation for any ADALINE-style model
- Standardizes features using train-fold statistics only (no data leakage)
- Accepts any model class and keyword arguments via `**model_kwargs`

---

## 📓 Notebooks

### `binary_classification.ipynb`
Tasks 1–3 and 5 on binary classification:

| Task | Description |
|---|---|
| **1** | Implement and test ADALINE; train on Old Faithful with min-max normalization |
| **2** | Compare feature scaling strategies (none, min-max, standardization); analyze learning rate effects; visualize decision boundaries |
| **3** | Implement Perceptron; compare ADALINE vs Perceptron on Ionosphere (34 features); analyze misclassification scatter plots |
| **5** | Implement `AdalineLogistic`; train on Old Faithful; compute class probabilities for test points; visualize decision boundary |

### `regression.ipynb`
**Task 4** — ADALINE as a linear regressor:
- Predict `waiting` time from `eruptions` using the same ADALINE architecture with no code changes
- Fit and plot a regression line through the Old Faithful clusters
- Demonstrates that the identity activation + SSE loss naturally gives linear regression

### `extension.ipynb`
Extension on the Ionosphere dataset — see [Extension section](#-extension-gated-logistic-regression) below.

---

## 🔬 Extension: Gated Logistic Regression

**Goal:** Extend logistic regression with learnable per-feature gates that automatically down-weight or suppress uninformative features, making the model more interpretable.

**Architecture:**
```
gates = σ(g_raw)            # per-feature weights in (0, 1)
net_in = (X * gates) @ wts + b
net_act = σ(net_in)         # predicted probability
```

**Training:** Gates, weights, and bias are all updated jointly via gradient descent. An optional L1 penalty on `gates` encourages sparsity.

**Evaluated with 5-fold stratified cross-validation** against a baseline `AdalineLogistic` on the Ionosphere dataset. Results compared using accuracy boxplots across folds.

**Top gated features** — the 8 features assigned the highest gate values — are reported, providing a built-in feature importance ranking.

---

## 💡 Key Findings

- **Feature scaling is critical** — training on raw (unscaled) Old Faithful features causes numerical overflow and NaN loss; standardization converges faster than min-max normalization.
- **Learning rate sensitivity** — too large causes oscillation/divergence; too small causes slow but stable convergence. ADALINE (continuous SSE loss) is more sensitive to large learning rates than the Perceptron (step-based updates).
- **ADALINE vs Perceptron on Ionosphere** — both achieve ~90–91% accuracy; ADALINE's smoother loss curve reflects continuous optimization while the Perceptron's accuracy is more jagged.
- **Regression reuse** — the same ADALINE code (identity activation + SSE) performs linear regression without modification, demonstrating architectural generality.
- **Gated extension** — learnable gates identify a small subset of the 34 Ionosphere features as most informative, matching the known structure of the dataset.

---

## ⚙️ Requirements

```bash
pip install numpy pandas matplotlib
```

| Package | Purpose |
|---|---|
| `numpy` | All model math and array operations |
| `pandas` | Dataset loading |
| `matplotlib` | Training curves, decision boundaries, scatter plots |

---

## 🚀 Usage

1. Clone the repo and place `old_faithful.csv` and `ionosphere.csv` in a `Data/` subdirectory
2. Install dependencies
3. Run any notebook

```bash
git clone <your-repo-url>
cd <repo-folder>
pip install numpy pandas matplotlib

jupyter notebook binary_classification.ipynb   # Classification tasks
jupyter notebook regression.ipynb              # Regression task
jupyter notebook extension.ipynb               # Gated logistic + K-fold CV
```

> **Note:** All `.py` source files must be in the same directory as the notebooks.

---

<p align="center">Made with 🧠 and NumPy</p>
