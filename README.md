# xFODE+: Explainable Type-2 Fuzzy Additive ODEs for Uncertainty Quantification

Official MATLAB implementation of the paper:

```bibtex
@inproceedings{kececi2026xfodeplus,
  title     = {xFODE+: Explainable Type-2 Fuzzy Additive ODEs for Uncertainty Quantification},
  author    = {Ke{\c{c}}eci, Ertu{\u{g}}rul and Kumbasar, Tufan},
  booktitle = {2026 IEEE International Conference on Fuzzy Systems (FUZZ-IEEE)},
  year      = {2026}
}
```

Please cite the paper if you use any functions and publish papers about work performed using these codes.

---

## Overview

Recent advances in Deep Learning (DL) have boosted data-driven System Identification (SysID), but reliable use requires Uncertainty Quantification (UQ) alongside accurate predictions. Although UQ-capable models such as Fuzzy ODE (FODE) can produce Prediction Intervals (PIs), they offer limited interpretability. We introduce Explainable Type-2 Fuzzy Additive ODEs for UQ (xFODE+), an interpretable SysID model which produces PIs alongside point predictions while retaining physically meaningful incremental states. xFODE+ implements each fuzzy additive model with Interval Type-2 Fuzzy Logic Systems (IT2-FLSs) and constraints membership functions to the activation of two neighboring rules, limiting overlap and keeping inference locally transparent. The type-reduced sets produced by the IT2-FLSs are aggregated to construct the state update together with the PIs. The model is trained in a DL framework via a composite loss that jointly optimizes prediction accuracy and PI quality. Results on benchmark SysID datasets show that xFODE+ matches FODE in PI quality and achieves comparable accuracy, while providing interpretability.

---

## Requirements

- MATLAB R2023b
- Deep Learning Toolbox
- System Identification Toolbox

---

## Repository Structure

```
xfodeplus/
├── xFODE+/             # Proposed method (xFODE+ PS1/PS2/PS3 and AFODE+)
│   ├── run.m           # Entry point — configure dataset, and PS here
│   └── lib/            # Core library functions
└── IT2-FODE/           # IT2-FODE baseline
    ├── run.m
    └── lib/
```

Datasets are loaded directly from MATLAB's System Identification Toolbox.

---

## Quick Start

1. Open MATLAB and navigate to the model folder (e.g., `xFODE+/`).
2. Open `run.m` and set your configuration:
   ```matlab
   dataset_name          = "MRDamper";     % HairDryer | MRDamper | SteamEngine
   SR_method             = "incremental";  % "incremental" 
   input_membership_type = "trimf";        % "gaussmf" (AFODE+) | "trimf" (PS1) | "gauss2mf" (PS2) | "c-gauss2mf" (PS3)
   number_of_rules       = 5;
   number_of_runs        = 20;
   ```
3. Run `run.m`. Results (mean ± std of RMSE, PICP, and PINAW over `number_of_runs` seeds) are printed to the console.

To reproduce the IT2-FODE baseline, run `IT2-FODE/run.m` with the analogous configuration.

## Baselines

This repository provides only the **xFODE+** method and the **IT2-FODE** baseline. **xFODE**, **NODE**, and **FODE** implementations are available in the following repository:

[https://github.com/ertugrulkececi/xfode](https://github.com/ertugrulkececi/xfode)

---

## Datasets

| Dataset      | Inputs | Outputs | Train samples | Test samples | Source                   |
|--------------|--------|---------|---------------|--------------|--------------------------|
| Hair Dryer   | 1      | 1       | 500           | 500          | Built-in (`dryer2`)      |
| MR Damper    | 1      | 1       | 3000          | 499          | Built-in (`mrdamper`)    |
| Steam Engine | 2      | 2       | 250           | 201          | Built-in (`SteamEng`)    |

All datasets ship with MATLAB's System Identification Toolbox and are loaded by name.

---
