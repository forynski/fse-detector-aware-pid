# Machine Learning Particle Identification for Run 3 Pb-Pb Collisions in ALICE

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-0.4.0+-orange.svg)](https://github.com/google/jax)
[![Flax](https://img.shields.io/badge/Flax-0.6.0+-green.svg)](https://github.com/google/flax)
[![ALICE](https://img.shields.io/badge/ALICE-O2Physics-red.svg)](https://alice.cern/)

**High-performance JAX/Flax neural networks for particle identification in ALICE Run 3**

Includes five complementary architectures: **SimpleNN**, **DNN**, **FSE + Attention**, **LightGBM**, and **XGBoost**

**JAX | Production-ready with XLA compilation | Focal Loss for Neural Networks | Default Class Weighting for Tree Models | DPG Track Selections**

</div>

---

## Overview

**FSE + Attention** is a machine learning framework for **particle identification (PID)** in ALICE Run 3 Pb-Pb collisions, designed to compare feature-based neural networks and tree-based models across several momentum regions, with and without DPG track selections. It is worth mentioning that the baseline **FSE + Attention** model already learns detector information implicitly: attention mechanisms are sufficient to capture detector-dependent behaviour without explicit conditioning, and explicit detector-aware modelling does not significantly outperform the baseline in this dataset.

The framework compares three JAX/Flax neural network variants — **SimpleNN**, **DNN**, and **FSE + Attention** — against two tree-based baselines, **LightGBM** and **XGBoost**. The models are trained on simulated Run 3 Pb-Pb collision data with detector effects, then evaluated over the **full spectrum** and in three momentum slices: **0-1 GeV/c**, **0.7-1.5 GeV/c**, and **1-3 GeV/c**.

### Key Finding: Tree-Based Models Perform Slightly Better , FSE + Attention Outperforms Other Neural Networks

**XGBoost** and **LightGBM** achieve the best test accuracy in most settings, while **FSE + Attention** remains the strongest neural-network approach. The attention model is consistently competitive, especially after DPG cuts, but the tree-based methods still lead by a small margin.

| Momentum Range | SimpleNN | DNN | FSE + Attention | LightGBM | XGBoost | Best NN Gap |
|---|---:|---:|---:|---:|---:|---:|
| Full Spectrum (No DPG cuts) | 86.45% | 86.70% | 87.25% | 87.44% | 87.24% | +0.19% |
| Full Spectrum (With DPG cuts) | 90.92% | 91.22% | 91.41% | 91.59% | 91.63% | +0.22% |
| 0-1 GeV/c (No DPG cuts) | 88.40% | 88.62% | 89.12% | 89.22% | 89.03% | +0.10% |
| 0-1 GeV/c (With DPG cuts) | 93.54% | 93.50% | 93.59% | 93.73% | 93.75% | +0.16% |
| 0.7-1.5 GeV/c (No DPG cuts) | 86.13% | 85.95% | 86.49% | 86.76% | 86.70% | +0.27% |
| 0.7-1.5 GeV/c (With DPG cuts) | 87.16% | 87.19% | 87.42% | 87.90% | 87.96% | +0.54% |
| 1-3 GeV/c (No DPG cuts) | 79.13% | 78.52% | 79.74% | 80.29% | 80.21% | +0.55% |
| 1-3 GeV/c (With DPG cuts) | 82.78% | 82.70% | 83.28% | 83.67% | 83.76% | +0.48% |

### Why the Gain from DPG Cuts Matters

DPG cuts improve every model across every momentum range. The improvement is especially strong in the full spectrum and 0-1 GeV/c samples, where cleaner track selections significantly increase the share of tracks with useful detector information and make the classification task easier.

| Momentum Range | Train Tracks, No Cuts | Train Tracks, DPG Cuts | Reduction | Test Tracks, No Cuts | Test Tracks, DPG Cuts | Reduction |
|---|---:|---:|---:|---:|---:|---:|
| Full Spectrum | 14,080,636 | 3,370,807 | 76.06% | 3,520,159 | 842,702 | 76.06% |
| 0-1 GeV/c | 11,562,280 | 2,683,544 | 76.79% | 2,890,570 | 670,887 | 76.79% |
| 0.7-1.5 GeV/c | 2,992,188 | 917,955 | 69.32% | 748,047 | 229,489 | 69.32% |
| 1-3 GeV/c | 2,247,708 | 648,089 | 71.17% | 561,927 | 162,023 | 71.17% |

### Bayesian PID Availability: The Data Crisis

Real Bayesian PID coverage is very sparse in the full dataset. In the full spectrum sample without DPG cuts, only **8.21 per cent** of tracks have complete Bayesian values. DPG cuts improve this significantly, but the dataset still contains large detector-availability gaps.

| Momentum Range | Total Tracks | Complete Bayesian | Complete % |
|---|---:|---:|---:|
| Full Spectrum (No DPG cuts) | 20,027,670 | 1,644,658 | 8.21% |
| 0-1 GeV/c (No DPG cuts) | 16,816,404 | 1,101,459 | 6.55% |
| 0.7-1.5 GeV/c (No DPG cuts) | 3,854,909 | 749,728 | 19.45% |
| 1-3 GeV/c (No DPG cuts) | 2,865,692 | 510,107 | 17.80% |
| Full Spectrum (With DPG cuts) | 4,319,844 | 1,035,194 | 23.96% |
| 0-1 GeV/c (With DPG cuts) | 3,453,565 | 650,629 | 18.84% |
| 0.7-1.5 GeV/c (With DPG cuts) | 1,159,394 | 488,148 | 42.10% |
| 1-3 GeV/c (With DPG cuts) | 816,622 | 361,446 | 44.26% |

---

## Dataset Summary

**Dataset: Monte Carlo Reconstructed, LHC25f6, Run 544122**

The study uses simulated Run 3 Pb-Pb collisions with detector effects, containing about **20 million tracks** before DPG selections and **4.3 million tracks** after DPG cuts in the full-spectrum case. The task is four-class PID for **pions, kaons, protons, and electrons**.

The data are evaluated in four momentum regions: **full spectrum**, **0-1 GeV/c**, **0.7-1.5 GeV/c**, and **1-3 GeV/c**, both with and without DPG-recommended selections. The DPG cuts make the samples cleaner and improve detector availability, which is reflected in the performance gains.

### Data Splits

| Momentum Range | Train | Test |
|---|---:|---:|
| Full Spectrum (No DPG cuts) | 14,080,636 | 3,520,159 |
| 0-1 GeV/c (No DPG cuts) | 11,562,280 | 2,890,570 |
| 0.7-1.5 GeV/c (No DPG cuts) | 2,992,188 | 748,047 |
| 1-3 GeV/c (No DPG cuts) | 2,247,708 | 561,927 |
| Full Spectrum (With DPG cuts) | 3,370,807 | 842,702 |
| 0-1 GeV/c (With DPG cuts) | 2,683,544 | 670,887 |
| 0.7-1.5 GeV/c (With DPG cuts) | 917,955 | 229,489 |
| 1-3 GeV/c (With DPG cuts) | 648,089 | 162,023 |

### Particle Composition

| Momentum Range | Pion | Kaon | Proton | Electron |
|---|---:|---:|---:|---:|
| Full Spectrum (No DPG cuts) | 69.52% | 4.86% | 12.08% | 13.54% |
| 0-1 GeV/c (No DPG cuts) | 69.22% | 3.25% | 11.82% | 15.71% |
| 0.7-1.5 GeV/c (No DPG cuts) | 72.70% | 8.95% | 13.69% | 4.66% |
| 1-3 GeV/c (No DPG cuts) | 71.51% | 11.90% | 13.02% | 3.57% |
| Full Spectrum (With DPG cuts) | 85.50% | 8.60% | 3.94% | 1.96% |
| 0-1 GeV/c (With DPG cuts) | 88.56% | 6.70% | 2.37% | 2.37% |
| 0.7-1.5 GeV/c (With DPG cuts) | 80.26% | 12.91% | 6.35% | 0.48% |
| 1-3 GeV/c (With DPG cuts) | 74.08% | 15.74% | 9.82% | 0.36% |

---

## Momentum Ranges

The analysis is performed in fixed momentum windows, with and without DPG selections. The no-cuts and with-cuts settings are always shown together to make the effect of track-quality selections explicit.

| Key | Name | Min | Max | DPG Cuts |
|---|---|---:|---:|---|
| full | Full Spectrum (0.1-∞ GeV/c) | 0.1 | ∞ | No |
| 0.1-1 | 0-1 GeV/c | 0.1 | 1.0 | No |
| 0.7-1.5 | 0.7-1.5 GeV/c | 0.7 | 1.5 | No |
| 1-3 | 1-3 GeV/c | 1.0 | 3.0 | No |
| full_DPG | Full Spectrum (0.1-∞ GeV/c, DPG cuts) | 0.1 | ∞ | Yes |
| 0.1-1_DPG | 0-1 GeV/c (DPG cuts) | 0.1 | 1.0 | Yes |
| 0.7-1.5_DPG | 0.7-1.5 GeV/c (DPG cuts) | 0.7 | 1.5 | Yes |
| 1-3_DPG | 1-3 GeV/c (DPG cuts) | 1.0 | 3.0 | Yes |

The range definitions are important because performance changes substantially with momentum. The 0-1 GeV/c region is dominated by dense low-momentum tracks, while 1-3 GeV/c is more difficult because detector separation becomes weaker and kaon identification is especially challenging.

---

## Track Selections

The analysis uses DPG-recommended selections when the DPG-cuts branch is enabled. These selections reduce contamination and improve the fraction of tracks with usable detector information.

| Selection Group | Cut |
|---|---|
| Event           | $\|v_z\| < 10$ cm |
| Kinematics      | $-0.8 < \eta < 0.8$ |
| DCA             | $DCA_{xy} < 0.105$ cm, $DCA_z < 0.12$ cm |
| TPC             | TPC clusters $\geq 70$ |

These cuts are especially important for the DPG branch, where cleaner selections strongly improve the distribution of detector modes and the resulting PID performance.

---

## Features and Detector Groups

The model inputs are built from track kinematics, detector response variables, Bayesian PID values, and detector-availability flags. The feature set is intentionally designed to let attention-based models infer detector-dependent behaviour directly from the inputs.

### Training Features

| Group | Features |
|---|---|
| Kinematics | pt, eta, phi, dca_xy, dca_z |
| TPC | tpc_signal, tpc_nsigma_pi, tpc_nsigma_ka, tpc_nsigma_pr, tpc_nsigma_el |
| TOF | tof_beta, tof_nsigma_pi, tof_nsigma_ka, tof_nsigma_pr, tof_nsigma_el |
| Bayesian | bayes_prob_pi, bayes_prob_ka, bayes_prob_pr, bayes_prob_el |
| Detector Flags | has_tpc, has_tof |

### Detector Groups

| Group | Contents |
|---|---|
| tpc | tpc_signal, tpc_nsigma_pi, tpc_nsigma_ka, tpc_nsigma_pr, tpc_nsigma_el |
| tof | tof_beta, tof_nsigma_pi, tof_nsigma_ka, tof_nsigma_pr, tof_nsigma_el |
| bayes | bayes_prob_pi, bayes_prob_ka, bayes_prob_pr, bayes_prob_el |
| kinematics | pt, eta, phi, dca_xy, dca_z |

The detector-availability information is central to the analysis. In practice, the baseline FSE + Attention model already captures this structure implicitly, which is why explicit conditioning does not provide a meaningful gain.

---

## Model Architecture

The repository compares five model families:
- **JAX_SimpleNN**
- **JAX_DNN**
- **JAX_FSE_Attention**
- **LightGBM**
- **XGBoost**

The JAX models are trained with focal loss to reduce the impact of class imbalance. The tree-based models are trained with default class weighting.

### JAX Hyperparameters

| Model | Hidden Dims | Dropout | Learning Rate | Batch Size | Epochs | Patience |
|---|---|---:|---:|---:|---:|---:|
| JAX_SimpleNN | [512, 256, 128, 64] | 0.5 | 1e-4 | 256 | 100 | 30 |
| JAX_DNN | [1024, 512, 256, 128, 64] | 0.5 | 5e-5 | 256 | 100 | 30 |
| JAX_FSE_Attention | 128 hidden dim, 8 heads | 0.4 | 1e-4 | 256 | 100 | 30 |

### Tree Hyperparameters

| Model | n_estimators | learning_rate | depth / leaves | subsample | colsample_bytree |
|---|---:|---:|---|---:|---:|
| LightGBM | 500 | 0.05 | num_leaves = 64 | 0.8 | 0.8 |
| XGBoost | 500 | 0.05 | max_depth = 6 | 0.8 | 0.8 |

These settings provide a strong and consistent baseline across all momentum ranges. The JAX models are optimised for reproducible GPU-friendly training, while the tree models are tuned for strong tabular performance.

---

## Performance Summary

### Test Accuracy, No DPG Cuts

| Model | Full Spectrum | 0-1 GeV/c | 0.7-1.5 GeV/c | 1-3 GeV/c |
|---|---:|---:|---:|---:|
| SimpleNN | 0.8645 | 0.8840 | 0.8613 | 0.7913 |
| DNN | 0.8670 | 0.8862 | 0.8595 | 0.7852 |
| FSE + Attention | 0.8725 | 0.8912 | 0.8649 | 0.7974 |
| LightGBM | 0.8744 | 0.8922 | 0.8676 | 0.8029 |
| XGBoost | 0.8724 | 0.8903 | 0.8670 | 0.8021 |

### Test Accuracy, With DPG Cuts

| Model | Full Spectrum | 0-1 GeV/c | 0.7-1.5 GeV/c | 1-3 GeV/c |
|---|---:|---:|---:|---:|
| SimpleNN | 0.9092 | 0.9354 | 0.8716 | 0.8278 |
| DNN | 0.9122 | 0.9350 | 0.8719 | 0.8270 |
| FSE + Attention | 0.9141 | 0.9359 | 0.8742 | 0.8328 |
| LightGBM | 0.9159 | 0.9373 | 0.8790 | 0.8367 |
| XGBoost | 0.9163 | 0.9375 | 0.8796 | 0.8376 |

---

## Detector Mode Analysis

The detector-mode study shows that performance depends strongly on the available detector information. Tracks with TPC+TOF information are always classified much more accurately than tracks with TPC-only or no detector information.

### Detector Composition, No DPG Cuts

| Momentum Range | NONE | TPC_ONLY | TPC_TOF |
|---|---:|---:|---:|
| Full Spectrum | 9.52% | 81.55% | 8.93% |
| 0-1 GeV/c | 8.54% | 84.26% | 7.20% |
| 0.7-1.5 GeV/c | 11.97% | 68.64% | 19.40% |
| 1-3 GeV/c | 13.86% | 68.37% | 17.77% |

### Detector Composition, With DPG Cuts

| Momentum Range | NONE | TPC_ONLY | TPC_TOF |
|---|---:|---:|---:|
| Full Spectrum | 18.54% | 57.31% | 24.15% |
| 0-1 GeV/c | 19.39% | 61.62% | 18.99% |
| 0.7-1.5 GeV/c | 15.44% | 42.56% | 42.00% |
| 1-3 GeV/c | 15.06% | 40.78% | 44.16% |

### TPC-Only Accuracy, No DPG Cuts

| Model | Full Spectrum | 0-1 GeV/c | 0.7-1.5 GeV/c | 1-3 GeV/c |
|---|---:|---:|---:|---:|
| SimpleNN | 0.8640 | 0.8836 | 0.8549 | 0.7711 |
| DNN | 0.8666 | 0.8858 | 0.8521 | 0.7633 |
| FSE + Attention | 0.8729 | 0.8917 | 0.8587 | 0.7791 |
| LightGBM | 0.8746 | 0.8923 | 0.8608 | 0.7843 |
| XGBoost | 0.8722 | 0.8901 | 0.8601 | 0.7833 |

### TPC-Only Accuracy, With DPG Cuts

| Model | Full Spectrum | 0-1 GeV/c | 0.7-1.5 GeV/c | 1-3 GeV/c |
|---|---:|---:|---:|---:|
| SimpleNN | 0.9208 | 0.9548 | 0.8366 | 0.7510 |
| DNN | 0.9242 | 0.9542 | 0.8349 | 0.7492 |
| FSE + Attention | 0.9267 | 0.9554 | 0.8380 | 0.7599 |
| LightGBM | 0.9287 | 0.9568 | 0.8438 | 0.7654 |
| XGBoost | 0.9289 | 0.9570 | 0.8444 | 0.7653 |

### TPC+TOF Accuracy, No DPG Cuts

| Model | Full Spectrum | 0-1 GeV/c | 0.7-1.5 GeV/c | 1-3 GeV/c |
|---|---:|---:|---:|---:|
| SimpleNN | 0.9497 | 0.9657 | 0.9495 | 0.9304 |
| DNN | 0.9541 | 0.9706 | 0.9497 | 0.9255 |
| FSE + Attention | 0.9581 | 0.9718 | 0.9542 | 0.9335 |
| LightGBM | 0.9642 | 0.9778 | 0.9605 | 0.9441 |
| XGBoost | 0.9633 | 0.9777 | 0.9601 | 0.9436 |

### TPC+TOF Accuracy, With DPG Cuts

| Model | Full Spectrum | 0-1 GeV/c | 0.7-1.5 GeV/c | 1-3 GeV/c |
|---|---:|---:|---:|---:|
| SimpleNN | 0.9645 | 0.9820 | 0.9610 | 0.9514 |
| DNN | 0.9690 | 0.9820 | 0.9628 | 0.9514 |
| FSE + Attention | 0.9706 | 0.9825 | 0.9659 | 0.9546 |
| LightGBM | 0.9737 | 0.9854 | 0.9708 | 0.9585 |
| XGBoost | 0.9747 | 0.9862 | 0.9718 | 0.9605 |

The detector-mode tables make the core message clear: performance is dominated by detector availability, and the attention model already learns that structure well. Explicit detector-aware conditioning is therefore unnecessary for this dataset.

---

## ROC AUC Summary

### Macro AUC, No DPG Cuts

| Model | Full Spectrum | 0-1 GeV/c | 0.7-1.5 GeV/c | 1-3 GeV/c |
|---|---:|---:|---:|---:|
| SimpleNN | 0.942500 | 0.955871 | 0.907654 | 0.855537 |
| DNN | 0.945440 | 0.958050 | 0.906403 | 0.848973 |
| FSE + Attention | 0.949269 | 0.961487 | 0.917188 | 0.863022 |
| LightGBM | 0.955015 | 0.965795 | 0.927422 | 0.877502 |
| XGBoost | 0.953948 | 0.965070 | 0.926668 | 0.876580 |

### Macro AUC, With DPG Cuts

| Model | Full Spectrum | 0-1 GeV/c | 0.7-1.5 GeV/c | 1-3 GeV/c |
|---|---:|---:|---:|---:|
| SimpleNN | 0.941434 | 0.961211 | 0.912777 | 0.887791 |
| DNN | 0.948302 | 0.960287 | 0.910123 | 0.890487 |
| FSE + Attention | 0.951069 | 0.961822 | 0.922923 | 0.896254 |
| LightGBM | 0.957622 | 0.965824 | 0.933563 | 0.910825 |
| XGBoost | 0.957785 | 0.966037 | 0.934836 | 0.913394 |

---

## Efficiency, Purity, and F1

### Full Spectrum, No DPG Cuts

| Model | Pion F1 | Kaon F1 | Proton F1 | Electron F1 |
|---|---:|---:|---:|---:|
| SimpleNN | 0.9151 | 0.3385 | 0.8704 | 0.6860 |
| DNN | 0.9169 | 0.3677 | 0.8738 | 0.6861 |
| FSE + Attention | 0.9194 | 0.3822 | 0.8802 | 0.7145 |
| LightGBM | 0.9202 | 0.4177 | 0.8838 | 0.7246 |
| XGBoost | 0.9191 | 0.4114 | 0.8827 | 0.7156 |

### Full Spectrum, With DPG Cuts

| Model | Pion F1 | Kaon F1 | Proton F1 | Electron F1 |
|---|---:|---:|---:|---:|
| SimpleNN | 0.9511 | 0.5383 | 0.7132 | 0.3227 |
| DNN | 0.9529 | 0.5545 | 0.7215 | 0.3694 |
| FSE + Attention | 0.9538 | 0.5653 | 0.7328 | 0.4299 |
| LightGBM | 0.9546 | 0.5789 | 0.7409 | 0.5114 |
| XGBoost | 0.9549 | 0.5788 | 0.7415 | 0.5103 |

### 0-1 GeV/c, No DPG Cuts

| Model | Pion F1 | Kaon F1 | Proton F1 | Electron F1 |
|---|---:|---:|---:|---:|
| SimpleNN | 0.9256 | 0.4444 | 0.9442 | 0.6966 |
| DNN | 0.9270 | 0.4294 | 0.9471 | 0.7004 |
| FSE + Attention | 0.9295 | 0.4632 | 0.9491 | 0.7231 |
| LightGBM | 0.9298 | 0.4838 | 0.9508 | 0.7281 |
| XGBoost | 0.9286 | 0.4800 | 0.9513 | 0.7187 |

### 0-1 GeV/c, With DPG Cuts

| Model | Pion F1 | Kaon F1 | Proton F1 | Electron F1 |
|---|---:|---:|---:|---:|
| SimpleNN | 0.9659 | 0.6043 | 0.8276 | 0.4696 |
| DNN | 0.9657 | 0.6087 | 0.8285 | 0.4030 |
| FSE + Attention | 0.9661 | 0.6118 | 0.8298 | 0.4771 |
| LightGBM | 0.9668 | 0.6166 | 0.8315 | 0.5182 |
| XGBoost | 0.9669 | 0.6158 | 0.8334 | 0.5137 |

### 0.7-1.5 GeV/c, No DPG Cuts

| Model | Pion F1 | Kaon F1 | Proton F1 | Electron F1 |
|---|---:|---:|---:|---:|
| SimpleNN | 0.9220 | 0.2569 | 0.8039 | 0.7053 |
| DNN | 0.9213 | 0.2243 | 0.8023 | 0.6727 |
| FSE + Attention | 0.9232 | 0.2711 | 0.8153 | 0.7149 |
| LightGBM | 0.9246 | 0.2854 | 0.8213 | 0.7341 |
| XGBoost | 0.9244 | 0.2811 | 0.8187 | 0.7317 |

### 0.7-1.5 GeV/c, With DPG Cuts

| Model | Pion F1 | Kaon F1 | Proton F1 | Electron F1 |
|---|---:|---:|---:|---:|
| SimpleNN | 0.9271 | 0.4314 | 0.7565 | 0.0162 |
| DNN | 0.9268 | 0.4135 | 0.7616 | 0.1320 |
| FSE + Attention | 0.9286 | 0.4444 | 0.7696 | 0.2616 |
| LightGBM | 0.9307 | 0.4594 | 0.7997 | 0.4394 |
| XGBoost | 0.9311 | 0.4588 | 0.7991 | 0.4879 |

### 1-3 GeV/c, No DPG Cuts

| Model | Pion F1 | Kaon F1 | Proton F1 | Electron F1 |
|---|---:|---:|---:|---:|
| SimpleNN | 0.8795 | 0.2979 | 0.5134 | 0.6568 |
| DNN | 0.8756 | 0.2473 | 0.5027 | 0.5655 |
| FSE + Attention | 0.8824 | 0.3250 | 0.5433 | 0.6640 |
| LightGBM | 0.8847 | 0.3370 | 0.5673 | 0.6985 |
| XGBoost | 0.8844 | 0.3356 | 0.5632 | 0.6937 |

### 1-3 GeV/c, With DPG Cuts

| Model | Pion F1 | Kaon F1 | Proton F1 | Electron F1 |
|---|---:|---:|---:|---:|
| SimpleNN | 0.8967 | 0.4703 | 0.6168 | 0.0000 |
| DNN | 0.8959 | 0.4598 | 0.6195 | 0.0000 |
| FSE + Attention | 0.9002 | 0.5094 | 0.6251 | 0.1169 |
| LightGBM | 0.9024 | 0.5159 | 0.6548 | 0.2829 |
| XGBoost | 0.9028 | 0.5177 | 0.6547 | 0.3686 |

---

## Conclusions

Machine-learning classifiers can provide high-quality particle identification across the momentum ranges studied here. The key finding of this repository is that the baseline **FSE + Attention** already learns detector information implicitly, so attention mechanisms alone are sufficient to capture detector-dependent behaviour without explicit conditioning.

DPG cuts improve all models and all momentum slices. The strongest results are still achieved by **LightGBM** and **XGBoost**, while **FSE + Attention** remains the best neural approach and a very strong benchmark for future studies.

### Practical Takeaways

- Use **XGBoost** or **LightGBM** if the goal is maximum accuracy.
- Use **FSE + Attention** if the goal is a strong neural baseline with detector structure learned implicitly.
- Use **focal loss** for neural networks and **default class weighting** for tree-based models.
- DPG cuts improve both performance and detector availability, especially in the full-spectrum and low-momentum regions.

---

## Reproducibility

The repository includes the full preprocessing pipeline, model definitions, training functions, evaluation routines, and analysis code needed to reproduce the reported results. The notebook is structured so models can be trained from scratch or loaded from saved checkpoints, which makes comparison across momentum ranges straightforward.

The project is intended as a reproducible machine-learning study for particle identification under ALICE Run 3 conditions, with emphasis on detector-dependent behaviour and the effect of DPG track selections.

---

## Citation

If you use this work, please cite the repository and the associated presentation or conference contribution. The project is intended as a reproducible machine-learning study for particle identification under ALICE Run 3 conditions.

---

## Contact and Support

- **Email:** [robert.forynski@cern.ch](mailto:robert.forynski@cern.ch)
- **GitHub Issues:** Report bugs
- **Institution:** CERN, ALICE Collaboration

---

## Citation

```bibtex
@software{fse-detector-aware-pid_2026,
  title={Machine Learning Particle Identification for Run 3 Pb-Pb Collisions in ALICE},
  author={Forynski, Robert},
  year={2026},
  month={April},
  url={https://github.com/forynski/fse-detector-aware-pid}
}
```

---
