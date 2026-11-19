# Detector-Aware Feature-Set-Embedding Attention Networks for Particle Identification in ALICE Run 3

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![JAX](https://img.shields.io/badge/JAX%2FFlax-DNN-9cf.svg)
![ONNX](https://img.shields.io/badge/ONNX-deployable-005CED.svg)
![Experiment](https://img.shields.io/badge/ALICE-Run%203%20Pb%E2%80%93Pb-e4002b.svg)
![Status](https://img.shields.io/badge/status-research-orange.svg)

Machine-learning particle identification (PID) for ALICE Run 3 Pb–Pb collisions, based on a
**detector-aware Feature-Set-Embedding (FSE) model with group-wise multi-head self-attention**.
This repository contains the training and benchmarking code accompanying the ICNFP 2026 proceedings paper
*Detector-Aware Feature-Set-Embedding Attention Networks for Particle Identification in ALICE
Run 3 Pb–Pb Collisions* (R. Forynski, for the ALICE Collaboration).

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
configuration — without training a separate model per configuration. (Absent-detector values are
imputed with sentinels and accompanied by availability flags; the mask removes the absent groups
from the attention.)

The model classifies **pions, kaons, protons and electrons**. (Muons are not treated as a
separate class, being nearly indistinguishable from pions in the central barrel without
dedicated topology information.)

---

## Model

- **Input:** 34 inputs = 29 measured features in **seven detector groups**, plus 5 binary
  detector-availability flags
  - **TPC** (dE/dx, four nσ, cluster count, χ²)
  - **TOF** (β, mass, four nσ)
  - **TRD** (signal, χ², number of tracklets) — *the TRD signal is a constant placeholder in the current production, see [Known limitations](#known-limitations-of-the-current-dataset)*
  - **ITS** (cluster count, χ²)
  - **EMCal** (η, φ extrapolated to the calorimeter surface — position only, no cluster energy)
  - **HMPID** (Cherenkov signal, MIP charge)
  - **Kinematics / event** (pT, η, φ, DCAxy, DCAz, centrality, vz) — *centrality is constant in the current production*
  - **Detector availability flags** (`has_tpc`, `has_tof`, `has_trd`, `has_emcal`, `has_hmpid`) —
    binary indicators passed alongside the group mask, excluded from feature standardisation
- **Availability mask:** a binary length-7 group mask encodes which detectors measured each track,
  used to gate per-group attention; the 5 availability flags above are a separate, redundant
  encoding of the same information at the feature level.
- **Architecture:** per-group linear projection to a `d = 128` embedding → masked multi-head
  self-attention (`h = 8`) → per-token layer-normalised, sigmoid-gated transform → masked mean
  pooling over present groups → MLP (128 → 64, ReLU, dropout 0.4) → softmax over 4 classes.
- **Loss:** focal loss (α = 0.25, γ = 2.0) to counter the strong pion dominance.
- **Framework:** JAX/Flax, Adam (lr = 1e-4, gradient-norm clipping, batch size 256, early
  stopping on the test set — so quoted performance may be marginally optimistic). Layer
  normalisation is used throughout in preference to batch normalisation, whose statistics would be
  dominated by imputed zeros from absent sparse detectors.

> **Note on the Bayesian baseline.** The Bayesian PID probabilities are **excluded from the
> training features** and retained only as an independent comparison baseline, so the network
> cannot simply reproduce the estimator it is benchmarked against. The baseline is a simplified
> Gaussian-likelihood combination of the TPC and TOF nσ values with flat (or, in one comparison,
> class-proportional) priors — it is **not** the full ALICE Bayesian PID.

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

- **Simulation:** ALICE Run 3 Pb–Pb Monte Carlo production **LHC25f6**, anchored to **apass5** of **run 544122**
  of the 2023 Pb–Pb data-taking, √s_NN = 5.36 TeV. Labels from Monte Carlo truth (PDG codes
  mapped to pion, kaon, proton, electron).
- **Real data:** same run (544122) of the **LHC23zzh** period, used for a consistency check
  (no truth labels are available).
- **Size:** full sample 3.08 × 10⁷ tracks; after the standard ALICE **DPG quality selections**
  (|vz| < 10 cm, |η| < 0.8, |DCAxy| < 0.105 cm, |DCAz| < 0.12 cm, ≥ 70 TPC clusters, ≥ 3 ITS
  clusters) → 3.09 × 10⁶ tracks (train/test 2.47 × 10⁶ / 6.17 × 10⁵, stratified 80/20 at track level).
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
0.03 → 0.83). On this strongly pion-dominated sample, part of the gap to the flat-prior baseline
reflects the network implicitly learning the species abundances.

**Resilience to detector configuration (accuracy):**

| Classifier | TPC-only | TPC + TOF |
|---|---|---|
| FSE + attention | 0.941 | 0.977 |
| Bayesian (flat priors) | 0.326 | 0.587 |

The detector-aware model already classifies TPC-only tracks at 0.94 accuracy, where the flat-prior
Bayesian estimator collapses to 0.33 — reliable PID across the whole track population, not only the
minority with complete information.

---

## Comparison with other ML models

On identical tracks and splits, the detector-aware model is benchmarked against a deep
feed-forward network (DNN) on the imputed feature vector and the XGBoost / LightGBM
gradient-boosted trees (full spectrum, DPG), all receiving the same 34 inputs:

| Model | Accuracy | Macro AUC | Macro F1 |
|---|---|---|---|
| DNN | 0.953 | 0.986 | 0.818 |
| LightGBM | 0.952 | 0.986 | 0.818 |
| XGBoost | 0.953 | 0.987 | 0.824 |
| FSE + attention | 0.952 | 0.986 | 0.817 |

**All models far exceed the Bayesian baseline and are mutually comparable** (differences of at
most 10⁻³ in accuracy and macro AUC and up to 7 × 10⁻³ in macro F1). Each model was trained once
(single seed), so no run-to-run uncertainty is available and the differences should not be read as
a ranking. The distinction of the detector-aware model is **architectural rather than a difference
in raw accuracy**: the tree and feed-forward models require a fixed-length input and therefore
depend on imputing the signals of absent detectors (here supplemented by the availability flags),
or on separate models per detector configuration. The detector-aware network instead conditions
directly on the per-track detector mask, so a **single model** covers every configuration —
including the incomplete-information tracks that dominate the sample — and is deployed as a
single ONNX model.

---

## Known limitations of the current dataset

These apply to the LHC25f6 feature production used for the released model and the paper:

- **`trd_signal` is a constant placeholder** (−1 for every track with TRD information). The TRD
  group therefore contributes only its χ² and number of tracklets, and no transition-radiation
  information.
- **`centrality_ft0c` is constant** in the training sample (100 after clipping to [0, 100]).
- The **EMCal group contains only the extrapolated track position** (no cluster energy), and
  HMPID covers only 0.35 % of the selected tracks.
- Consequently, **the benefit of the TRD, EMCal and HMPID input groups over the TPC and TOF inputs
  has not been established.** An internal controlled check on the same data, split and seed — a
  reduced model with the TRD, ITS, EMCal and HMPID inputs, masks and flags zeroed — gave the same
  FSE performance as the full model (accuracy 0.952, macro F1 0.817 in both cases); this
  comparison is not part of the released notebook or the paper.
- Both constant inputs are kept in the input vector so that the input contract is unchanged. A
  model trained on a production in which they are populated would need retraining and
  validation; do not assume the released model responds meaningfully to these two inputs.
- Species labels come from Monte Carlo truth, so any simulation-versus-data mismatch affects the
  learned boundaries; probability calibration and the interpretability of the attention weights
  have not been separately quantified.

Planned: repopulate the TRD signal and the centrality in the feature production, repeat the study
on larger samples, and validate on real data with samples identified without reference to Monte Carlo
truth (e.g. V0 decay daughters and photon conversions).

---

## Real-data validation

Applied to real Run 3 Pb–Pb data (LHC23zzh, run 544122), the model's predictions populate the
expected PID bands — the TPC dE/dx ordering π < K < p and the TOF velocity ordering π > K > p —
confirming that the network behaves consistently on real data. Since dE/dx and β are themselves
model inputs and real data carry no truth labels, this is a consistency check rather than a
performance measurement or an independent demonstration of generalisation from simulation.

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
