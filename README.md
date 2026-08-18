# Detector-Aware Feature-Set-Embedding Attention Networks for Particle Identification in ALICE Run 3

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![JAX](https://img.shields.io/badge/JAX%2FFlax-DNN-9cf.svg)
![ONNX](https://img.shields.io/badge/ONNX-deployable-005CED.svg)
![Experiment](https://img.shields.io/badge/ALICE-Run%203%20Pb%E2%80%93Pb-e4002b.svg)
![Status](https://img.shields.io/badge/status-research-orange.svg)

Machine-learning particle identification (PID) for ALICE Run 3 Pb–Pb collisions, based on a
**detector-aware Feature-Set-Embedding (FSE) model with group-wise multi-head self-attention**.
This repository contains the training and benchmarking code accompanying the proceedings paper
*Detector-Aware Feature-Set-Embedding Attention Networks for Particle Identification in ALICE
Run 3 and Run 4 Pb–Pb Collisions* (R. Forynski).

> **Scope.** This repository provides the **training and benchmarking** notebook. The ONNX
> inference task and configurable analysis wrapper have been **merged into O²Physics**
> as `PidFeatureExtractor` and `PidOnnxInference` — see [O²Physics tasks](#o²physics-tasks) below.

---

## Overview

Particle identification in ALICE combines several sub-detectors (TPC, TOF, TRD, ITS, EMCal,
HMPID), but only a small fraction of tracks are measured by all of them — in Run 3 most tracks
carry **incomplete** PID information. The traditional flat-prior Bayesian estimator degrades
sharply in this regime.

This project trains a single neural network that:

- represents each PID sub-detector as a **token** (a group of features),
- **masks** the tokens of detectors that did not measure a given track,
- applies **multi-head self-attention** restricted by that mask, so information flows only
  between detectors that are genuinely present,

so that one set of weights realises a different effective classifier for every detector
configuration — without training a separate model per configuration, and without imputing the
signals of absent detectors.

The model classifies **pions, kaons, protons and electrons**. (Muons are not treated as a
separate class, being nearly indistinguishable from pions in the central barrel without
dedicated topology information.)

---

## Model

- **Input:** 34 features in **seven detector groups**
  - **TPC** (dE/dx, four nσ, cluster count, χ²)
  - **TOF** (β, mass, four nσ)
  - **TRD** (signal, χ², number of tracklets)
  - **ITS** (cluster count, χ²)
  - **EMCal** (η, φ extrapolated to the calorimeter surface)
  - **HMPID** (Cherenkov signal, MIP charge)
  - **Kinematics / event** (pT, η, φ, DCAxy, DCAz, centrality, vz)
- **Availability mask:** a binary length-7 group mask encodes which detectors measured each track.
- **Architecture:** per-group linear projection to a `d = 128` embedding → masked multi-head
  self-attention (`h = 8`) → per-token layer-normalised, sigmoid-gated transform → masked mean
  pooling over present groups → MLP (128 → 64, ReLU, dropout 0.4) → softmax over 4 classes.
- **Loss:** focal loss (α = 0.25, γ = 2.0) to counter the strong pion dominance.
- **Framework:** JAX/Flax, Adam (lr = 1e-4, gradient-norm clipping, batch size 256, early
  stopping). Layer normalisation is used throughout in preference to batch normalisation, whose
  statistics would be dominated by imputed zeros from absent sparse detectors.

> **Note on the Bayesian baseline.** The Bayesian PID probabilities are **excluded from the
> training features** and retained only as an independent comparison baseline, so the network
> cannot simply reproduce the estimator it is benchmarked against.

### Download the trained model

The trained model is exported to ONNX (`full_DPG_JAX_FSE_Attention.onnx`) and deployed on the
ALICE Condition and Calibration Database (CCDB) at:

```
Users/r/rforynsk/PidFeatureExtractor/model
```

Browse it at [alice-ccdb.cern.ch/browse](http://alice-ccdb.cern.ch/browse/Users/r/rforynsk/PidFeatureExtractor/model),
or fetch it directly:

```
http://alice-ccdb.cern.ch/download/25cf3fbb-9b1f-11f1-bf5f-56153f9e24ee
```

(This direct link points at the current object version; if the model is ever re-uploaded, use
the CCDB path above via `CcdbApi`/`BasicCCDBManager` instead, so you always resolve the latest
version rather than a pinned snapshot.)

This is also the path `PidOnnxInference` reads via O²Physics's `Tools/ML/MlResponse` wrapper
(`modelPathsCcdb` Configurable, with `loadModelFromCcdb=true`) — see
[O²Physics tasks](#o²physics-tasks) below.

---

## Dataset

- **Simulation:** ALICE Run 3 Pb–Pb Monte Carlo production **LHC25f6**, anchored to **run 544122**
  of the 2023 Pb–Pb data-taking, √s_NN = 5.36 TeV. Labels from Monte Carlo truth (PDG codes).
- **Real data:** same run (544122) of the **LHC23zzh** period, used for validation.
- **Size:** full sample 3.08 × 10⁷ tracks; after the standard ALICE **DPG quality selections**
  (|vz| < 10 cm, |η| < 0.8, |DCAxy| < 0.105 cm, |DCAz| < 0.12 cm, ≥ 70 TPC clusters, ≥ 3 ITS
  clusters) → 3.09 × 10⁶ tracks (train/test 2.47 × 10⁶ / 6.17 × 10⁵, stratified 80/20).
- **Class imbalance (DPG):** pions 87.1 %, kaons 7.2 %, protons 4.0 %, electrons 1.7 %.

Continuous features are standardised with per-feature scalers fitted on the training set only;
sparse-detector features use present-only statistics so that imputed zeros are preserved.

---

## Key results (quality-selected / DPG sample)

**Overall (full spectrum):**

| Model | Accuracy | Macro AUC |
|---|---|---|
| Detector-aware FSE + attention | 0.952 | 0.986 |
| Flat-prior Bayesian baseline | 0.407 | — |

By momentum (FSE accuracy / macro AUC): 0.1–1 GeV/c → 0.974 / 0.993; 0.7–1.5 GeV/c → 0.920 /
0.965; 1–3 GeV/c → 0.878 / 0.944.

**Per-species (FSE, full spectrum):**

| Species | Efficiency | Purity | F1 |
|---|---|---|---|
| Pion | 0.991 | 0.961 | 0.975 |
| Kaon | 0.658 | 0.857 | 0.744 |
| Proton | 0.804 | 0.934 | 0.864 |
| Electron | 0.583 | 0.825 | 0.683 |

The network improves the per-species F1 over the flat-prior Bayesian baseline for all four
species, with especially large purity gains for the rarer species (e.g. electron purity
0.03 → 0.83).

**Resilience to detector configuration (accuracy):**

| Classifier | TPC-only | TPC + TOF |
|---|---|---|
| FSE + attention | 0.941 | 0.977 |
| Bayesian | 0.326 | 0.587 |

The detector-aware model already classifies TPC-only tracks at 0.94 accuracy, where the Bayesian
estimator collapses to 0.33 — reliable PID across the whole track population, not only the
minority with complete information.

---

## Comparison with other ML models

On identical tracks and splits, the detector-aware model is benchmarked against a deep
feed-forward network (DNN) on the imputed feature vector and the XGBoost / LightGBM
gradient-boosted trees (full spectrum, DPG):

| Model | Accuracy | Macro AUC | Macro F1 |
|---|---|---|---|
| DNN | 0.953 | 0.986 | 0.818 |
| LightGBM | 0.952 | 0.986 | 0.818 |
| XGBoost | 0.953 | 0.987 | 0.824 |
| FSE + attention | 0.952 | 0.986 | 0.817 |

**All models far exceed the Bayesian baseline and are mutually comparable, with differences of
order 10⁻³ — within run-to-run training variation.** The distinction of the detector-aware model
is **architectural rather than a difference in raw accuracy**: the tree and feed-forward models
require a fixed-length input and therefore depend on **imputing** the signals of absent
detectors, and would in general need separate models per detector configuration. The
detector-aware network instead conditions directly on the per-track detector mask, so a **single
model** covers every configuration — including the incomplete-information tracks that dominate the
sample — and is deployed as a single ONNX model. The FSE model also tends to favour purity over
efficiency and is particularly competitive on the harder electron channel.

## Impact of the additional detectors

Extending the input from a TPC+TOF baseline to the full detector set (adding TRD, EMCal and HMPID)
leaves the overall accuracy almost unchanged (pions, which dominate the sample, are already well
identified) but improves the **macro-averaged F1 by ≈ 0.03 for every model** — the gain
concentrates in the harder channels: kaons and protons by ≈ 0.02 and **electrons by up to ≈ 0.12**
(electron F1 0.59 → 0.68 on the full spectrum), growing towards higher momentum where the TPC/TOF
separation degrades.

---

## Real-data validation

Applied to real Run 3 Pb–Pb data (LHC23zzh, run 544122), the model's predictions populate the
expected PID bands — the TPC dE/dx ordering π < K < p and the TOF velocity ordering π > K > p —
confirming that the network behaves consistently on real data. As real data carry no truth labels,
this is a consistency check rather than a performance measurement.

---

## Deployment

The trained model is exported to **ONNX** and runs within O²Physics via ONNX Runtime (C++),
retrieved from the ALICE CCDB and applied at scale on the GRID / Hyperloop analysis trains.
Because the model accepts the full track population, it can be applied uniformly across an
analysis rather than only to the subset with complete PID information.

### O²Physics tasks

The feature-extraction and inference stages have been merged into O²Physics as two tasks:

- **`PidFeatureExtractor`** — extracts the 34-feature / 7-group PID feature set from AO2D input
  (MC and real data, via a runtime `PROCESS_SWITCH`), with optional DPG cuts, optional Bayesian
  PID, and optional CSV export.
- **`PidOnnxInference`** — loads the trained ONNX model (from CCDB or a local file, via
  O²Physics's `Tools/ML/MlResponse` wrapper) and writes out `PidMlPredictions`, with
  per-detector-group enable/disable options (`useTPC`/`useTOF`/`useTRD`/`useITS`/`useEMCal`/
  `useHMPID`/`useCentrality`).

Merged, in the main O²Physics repository: [`Tools/PIDFeatureExtractor`](https://github.com/AliceO2Group/O2Physics/tree/master/Tools/PIDFeatureExtractor)

---

## Citation

If you use this code, please cite the proceedings paper and the repository:

```bibtex
@software{fse-detector-aware-pid_2026,
  title   = {Detector-Aware Feature-Set-Embedding Attention Networks for Particle
             Identification in ALICE Run 3},
  author  = {Forynski, Robert},
  year    = {2026},
  url     = {https://github.com/forynski/fse-detector-aware-pid}
}
```

## Acknowledgements

This work was supported by the UK Research and Innovation (UKRI) Science and Technology Facilities
Council (STFC) [grant ST/Y509322/1] through a doctoral studentship at the University of Derby, in
association with STFC Daresbury Laboratory, and carried out within the ALICE Collaboration.
