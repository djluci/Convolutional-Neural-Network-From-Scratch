# 🌐 Convolutional Neural Networks: From Scratch to JAX-Accelerated

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-From%20Scratch-lightblue?logo=numpy&logoColor=white)
![JAX](https://img.shields.io/badge/JAX-Accelerated-purple)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![Course](https://img.shields.io/badge/CS343-Neural%20Networks-purple)

> A full from-scratch implementation of a Convolutional Neural Network — covering 2D convolution, max pooling, backpropagation, four gradient descent optimizers, and a dropout extension — trained and evaluated on the STL-10 image dataset with a JAX-accelerated fast path for production-speed training.

**Authors:** Duilio Lucio & Vivian Hu — CS343: Neural Networks, Spring 2026

---

## 📌 Table of Contents
- [Overview](#-overview)
- [Repository Structure](#-repository-structure)
- [Network Architecture](#-network-architecture)
- [Dataset](#-dataset)
- [Implementation Modules](#-implementation-modules)
- [Notebooks](#-notebooks)
- [Optimizers](#-optimizers)
- [Key Findings](#-key-findings)
- [Requirements](#-requirements)
- [Usage](#-usage)
- [References](#-references)

---

## 🔍 Overview

This project builds a full Convolutional Neural Network (CNN) from the ground up in NumPy, then accelerates the most expensive operations with JAX. It is organized into four progressive milestones:

1. **Convolution & Pooling** — Implement and validate the core operations from scratch
2. **Build the CNN** — Assemble modular layer classes with forward and backward passes
3. **Optimizers & Training** — Implement SGD, Momentum, and Adam; overfit-test on STL-10
4. **Dropout & AdamW** — Extend the network with regularization and run controlled experiments

---

## 📂 Repository Structure

```
.
├── filter_ops.py                  # 2D convolution and max pooling (pure NumPy)
├── layer.py                       # Layer base class + Conv2D, MaxPool2D, Flatten, Dense, Dropout
├── accelerated_layer.py           # JAX-accelerated Conv2DAccel and MaxPool2DAccel (im2col)
├── network.py                     # ConvNet4, ConvNet4Accel, ConvNet4AccelV2
├── optimizer.py                   # SGD, SGD_Momentum, Adam, AdamW
├── preprocess_data.py             # STL-10 preprocessing pipeline (NCHW format)
├── load_stl10_dataset.py          # STL-10 download, extraction, and resizing
├── convolution_and_pooling.ipynb  # Tasks 1–3: implement and test filter ops
├── convolutional_net.ipynb        # Task 4: build and test CNN layers + forward/backward
├── optimize_and_train_cnn.ipynb   # Tasks 5–7: optimizers, training, overfit test
├── convnet4v2.ipynb               # Tasks 8–9: Dropout, AdamW, ConvNet4AccelV2 experiments
└── README.md
```

---

## 🏗️ Network Architecture

### ConvNet4 / ConvNet4Accel

```
Input (N, 3, H, W)
    │
    ▼
Conv2D      → ReLU        [learnable filters, 'same' padding]
    │
    ▼
MaxPool2D   → Linear      [spatial downsampling]
    │
    ▼
Flatten     → Linear      [reshape to 1D per sample]
    │
    ▼
Dense       → ReLU        [fully connected hidden layer]
    │
    ▼
Dense       → Softmax     [output: class probabilities]
```

### ConvNet4AccelV2 *(Task 9 extension)*

Adds a **Dropout** layer before the output:

```
... → Dense (ReLU) → Dropout → Dense (Softmax)
```

---

## 📊 Dataset

### STL-10 Image Dataset
| Detail | Value |
|---|---|
| Source | [Stanford STL-10](https://cs.stanford.edu/~acoates/stl10/) |
| Classes | 10 — airplane, bird, car, cat, deer, dog, horse, monkey, ship, truck |
| Original resolution | 96×96 RGB |
| Working resolutions | 32×32 (main), 16×16 (overfit test) |
| Preprocessing | float64, per-pixel standardization, NCHW dimension order `(N, C, H, W)` |

**Splits used:**

| Split | Samples |
|---|---|
| Train | 4,000 |
| Test | 500 |
| Validation | 499 |
| Development | 1 |

> **Note:** The STL-10 dataset is automatically downloaded and cached as NumPy arrays. If the download is slow, place `stl10_binary.tar.gz` manually in a `data/` subfolder.

---

## 🔧 Implementation Modules

### `filter_ops.py` — Convolution & Max Pooling (Pure NumPy)

| Function | Description |
|---|---|
| `conv2_gray` | Single-channel 2D convolution with 'same' padding |
| `conv2` | Multi-channel 2D convolution (RGB support) |
| `conv2nn` | Mini-batch 2D convolution with bias (neural network layer) |
| `max_pool` | Single-image 2D max pooling with configurable stride |
| `max_poolnn` | Mini-batch 2D max pooling |
| `get_pooling_out_shape` | Output dimension formula for max pooling |

All implementations use explicit nested loops with 'same' padding and kernel flipping for correctness.

### `layer.py` — Neural Network Layers

| Class | Net-in type | Activation | Notes |
|---|---|---|---|
| `Layer` | — | linear / relu / softmax | Base class; handles forward pass dispatch |
| `Conv2D` | 2D convolution | relu | Weights: `(n_kers, n_chans, ker_sz, ker_sz)` |
| `MaxPool2D` | 2D max pooling | linear | Stores reshaped input for backward pass |
| `Flatten` | Reshape | linear | `(N, C, H, W)` → `(N, C*H*W)` |
| `Dense` | Matrix multiply | relu / softmax | Standard fully connected layer |
| `Dropout` | Pass-through | linear | Inverted dropout; behaves differently in train vs test mode |

### `accelerated_layer.py` — JAX-Accelerated Layers

| Class | Description |
|---|---|
| `Conv2DAccel` | im2col algorithm via `lax.conv_general_dilated_patches`; transposed convolution for backward pass |
| `MaxPool2DAccel` | Strategic reshape + `max` for forward; masked gradient propagation for backward |

Both classes are drop-in replacements for their NumPy counterparts via inheritance.

### `optimizer.py` — Weight Update Algorithms

See [Optimizers](#-optimizers) section below.

---

## 📓 Notebooks

### `convolution_and_pooling.ipynb` — Tasks 1–3
- Unit tests for `conv2_gray` against `scipy.signal.convolve2d`
- Multi-kernel Gabor filter generation and visualization on a clownfish image
- Gabor filters at 4 orientations (horizontal, −45°, vertical, 45°) applied to grayscale images
- RGB convolution tests (`conv2`) and mini-batch convolution tests (`conv2nn`)
- Max pooling tests for single images and mini-batches

### `convolutional_net.ipynb` — Task 4
- Implements and unit-tests all layer classes in `layer.py`
- Tests: linear/ReLU/softmax activations, cross-entropy loss, Conv2D forward pass, full network forward pass with and without regularization, backward pass gradients
- All test outputs validated against expected values

### `optimize_and_train_cnn.ipynb` — Tasks 5–7
- Implements and tests SGD, SGD_Momentum, and Adam in `optimizer.py`
- Implements `predict` and `fit` in `network.py` with validation accuracy tracking per epoch
- **Overfit test:** trains `ConvNet4` on 20 STL-10 training samples at 16×16 resolution to verify backprop is working (should achieve ~100% training accuracy)
- **Full STL-10 training:** optimizer comparison on 4,000 training samples at 32×32

### `convnet4v2.ipynb` — Tasks 8–9
- Implements `Dropout` layer with inverted dropout and train/test mode switching
- Implements `ConvNet4AccelV2` (`Conv2D → MaxPool2D → Flatten → Dense → Dropout → Dense`)
- Implements `AdamW` optimizer (Adam + decoupled weight decay)
- **Dropout experiment:** trains 4 `ConvNet4AccelV2` nets with `dropout_rate ∈ {0.0, 0.1, 0.5, 0.9}` — plots training loss curves and final test accuracy vs dropout rate

---

## ⚡ Optimizers

All implemented in `optimizer.py` with a factory method `Optimizer.create_optimizer(name, ...)`.

| Optimizer | Update Rule | Key Parameters |
|---|---|---|
| **SGD** | `w = w − lr · dw` | `lr` |
| **SGD_Momentum** | `v = m·v + dw; w = w − lr·v` | `lr`, `m=0.9` |
| **Adam** | Bias-corrected first + second moment estimates | `lr`, `β₁=0.9`, `β₂=0.999`, `ε=1e-8` |
| **AdamW** | Adam + decoupled weight decay: `w = w − lr·reg·w` | All Adam params + `reg` |

---

## 💡 Key Findings

- **Convolution correctness** — `conv2_gray` results match `scipy.signal.convolve2d` exactly; Gabor filters produce biologically-inspired orientation-selective responses on real images.
- **Overfit test** — training a large ConvNet4 on only 20 samples drives training accuracy to ~100%, confirming forward and backward passes are correct before scaling up.
- **Optimizer comparison** — Adam converges faster and reaches higher accuracy than SGD and SGD_Momentum on STL-10; AdamW with weight decay improves generalization further.
- **Dropout effect** — moderate dropout rates (0.1–0.5) improve test accuracy over the no-dropout baseline; high dropout (0.9) degrades performance due to excessive information loss.
- **JAX speedup** — `Conv2DAccel` and `MaxPool2DAccel` are dramatically faster than the pure NumPy implementations, making full STL-10 training feasible without a GPU.

---

## ⚙️ Requirements

```bash
pip install numpy matplotlib pillow jax jaxlib scipy
```

| Package | Purpose |
|---|---|
| `numpy` | Core model math |
| `matplotlib` | Training curves, filter visualizations, accuracy plots |
| `pillow` | Image loading and resizing |
| `jax` / `jaxlib` | Accelerated convolution and max pooling |
| `scipy` | Ground truth comparison for convolution unit tests |

---

## 🚀 Usage

1. Clone the repo
2. Install dependencies
3. Run notebooks in order

```bash
git clone <your-repo-url>
cd <repo-folder>
pip install numpy matplotlib pillow jax jaxlib scipy

# Task 1–3: Convolution and pooling operations
jupyter notebook convolution_and_pooling.ipynb

# Task 4: Build CNN layers
jupyter notebook convolutional_net.ipynb

# Task 5–7: Optimizers and training
jupyter notebook optimize_and_train_cnn.ipynb

# Task 8–9: Dropout and AdamW
jupyter notebook convnet4v2.ipynb
```

> **Note:** All `.py` source files must be in the same directory as the notebooks. The clownfish image (`clownfish.png`) should be placed in an `images/` subfolder.

---

## 📚 References

- [STL-10 Dataset — Coates et al. (2011)](https://cs.stanford.edu/~acoates/stl10/)
- [Adam Optimizer — Kingma & Ba (2014)](https://arxiv.org/abs/1412.6980)
- [Dropout — Srivastava et al. (2014)](https://jmlr.org/papers/v15/srivastava14a.html)
- [JAX Documentation](https://jax.readthedocs.io/en/latest/)
- [Gabor Filters — Lee (1996)](http://leelab.cnbc.cmu.edu/publication/assets/links/ImageRepre.pdf)

---

<p align="center">Made with 🌐 and NumPy</p>
