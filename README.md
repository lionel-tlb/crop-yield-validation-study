# Validation Protocol Bias in Agricultural Yield Prediction

**Empirically Quantifying Optimism from Spatial Autocorrelation and Proxy-Variable Leakage**  
_Author:_ **TAKONGMO LIONEL BLERIOT** (<lioneltakoms25@gmail.com>)  
_Affiliation:_ Independent Applied Machine Learning Researcher

[![Paper PDF](<https://img.shields.io/badge/Paper-IEEE%20PDF%20(6%20Pages)-blue?style=flat-square&logo=adobeacrobatreader>)](paper/main.pdf)
[![Executive Summary](https://img.shields.io/badge/Executive%20Summary-2%20Pages%20PDF-orange?style=flat-square&logo=adobeacrobatreader)](paper/executive_summary.pdf)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen?style=flat-square&logo=python)](requirements.txt)
[![Data: FAO & World Bank](https://img.shields.io/badge/Data-FAO%20%26%20World%20Bank-lightgrey?style=flat-square)](data/)

> 📄 **[Read Full Manuscript (6-Page IEEE Format)](paper/main.pdf)** &nbsp;|&nbsp; 📋 **[Read Executive Summary (2-Page Abstract)](paper/executive_summary.pdf)**

---

## Overview

This repository contains the complete, reproducible experimental pipeline and paper source files for the empirical study:

> **"Validation Protocol Bias in Agricultural Yield Prediction: Empirically Quantifying Optimism from Spatial Autocorrelation and Proxy-Variable Leakage"**

We investigate how standard random cross-validation inflates perceived model accuracy relative to structure-aware protocols (Roberts et al., 2017) using the benchmark FAO/World Bank Crop Yield dataset. We isolate the impact of an injective proxy-variable leakage mechanism and compare complex tree models against simple domain baselines.

---

## Core Empirical Findings

1. **The Optimism Gap**: Standard Random CV inflates $R^2$ by **0.38 to 0.44 points** for tree ensembles:
   - **Random Forest**: $R^2 = 0.968 \to 0.528$ ($\Delta = -0.440$)
   - **XGBoost**: $R^2 = 0.945 \to 0.564$ ($\Delta = -0.381$)
2. **Naive Baseline Equivalence**: Under Leave-Country-Out (Group) CV (LCO-CV), Random Forest ($R^2 = 0.528 \pm 0.103$) shows **no detectable difference** from the naive Per-Crop Mean baseline ($R^2 = 0.531 \pm 0.065$; paired per-split $\Delta = -0.004 \pm 0.099$, 95% CI: $[-0.050, 0.042]$, $p = 0.87$). XGBoost ($0.564$) adds only a practically marginal gain ($\Delta = +0.033$).
3. **Proxy Leakage Audit**: `average_rain_fall_mm_per_year` is **time-invariant within 100% of the 101 countries**, providing an injective 1-to-1 country identifier.
4. **Ablation & Permutation Tests**:
   - Removing rainfall narrows the paired optimism gap by **13.4%** for RF and **18.9%** for XGBoost.
   - Permutation importance on held-out test splits reveals that permuting rainfall degrades test $R^2$ by **$0.370 \pm 0.011$** in RF and **$0.279 \pm 0.011$** in XGBoost, placing it among the most influential continuous predictors alongside temperature and pesticides.
5. **Temporal Evaluation Confound**: Forward temporal validation yields intermediate nominal scores ($R^2 \approx 0.86\text{--}0.88$), but because the same countries exist across train and test partitions, the country-identity proxy remains operational.

---

## Repository Structure

```
crop_yield_validation_study/
├── data/
│   ├── yield_df.csv              # Merged clean benchmark (25,932 clean rows)
│   ├── yield.csv                 # Raw FAO yield table
│   ├── rainfall.csv              # Raw World Bank rainfall table
│   ├── pesticides.csv            # Raw FAO pesticide use table
│   └── temp.csv                  # Raw surface temperature table
├── src/
│   ├── 01_data_audit.py          # Data cleaning, deduplication & cardinality audit
│   ├── 02_run_experiments.py     # 450-fit loop (5 models × 3 protocols × 2 conditions)
│   ├── 03_generate_figures.py    # Generates publication-ready figures (Fig 1, 2, 3)
│   └── 04_feature_importance.py  # Computes MDI & Permutation Importance with fixed seed
├── results/
│   ├── audit_table.csv           # Feature cardinality summary
│   ├── raw_fold_metrics.csv      # Metrics for all 450 evaluation fits
│   ├── summary_table_full.csv    # Performance summary (Full Features)
│   ├── summary_table_ablation.csv# Performance summary (No-Rainfall Ablation)
│   ├── feature_importance.csv    # Exact MDI & Permutation importances
│   ├── fig1_protocol_ablation.{pdf,png}
│   ├── fig2_temporal_drift.{pdf,png}
│   └── fig3_error_by_crop.{pdf,png}
├── paper/
│   ├── main.tex                  # IEEE conference paper (6 pages)
│   ├── executive_summary.tex     # 2-page executive summary
│   └── references.bib            # BibTeX bibliography
├── requirements.txt
└── README.md
```

---

## Reproduction Instructions

### 1. Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run Pipeline

```bash
# Step 1: Run data audit & deduplication check
python src/01_data_audit.py

# Step 2: Execute all 450 experimental model fits
python src/02_run_experiments.py

# Step 3: Compute MDI and permutation feature importances
python src/04_feature_importance.py

# Step 4: Generate figures (300 DPI, PDF + PNG)
python src/03_generate_figures.py
```

### 3. Compile LaTeX Manuscripts

Using MiKTeX or TeX Live:

```bash
cd paper
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex

pdflatex -interaction=nonstopmode executive_summary.tex
bibtex executive_summary
pdflatex -interaction=nonstopmode executive_summary.tex
pdflatex -interaction=nonstopmode executive_summary.tex
```

---

## Summary of Results

### $R^2$ Score across Protocols (Full Features, mean $\pm$ SD)

| Model             | Random CV             | Leave-Country-Out CV (LCO-CV) | Optimism Gap ($\Delta$) | Temporal CV (Cutoff Mean) |
| ----------------- | --------------------- | ----------------------------- | ----------------------- | ------------------------- |
| Global Mean       | $-0.000 \pm 0.000$    | $-0.012 \pm 0.013$            | $-0.012$                | $-0.022$                  |
| **Per-Crop Mean** | **$0.599 \pm 0.006$** | **$0.531 \pm 0.065$**         | **$-0.068$**            | **$0.578$**               |
| Ridge Regression  | $0.631 \pm 0.005$     | $0.463 \pm 0.172$             | $-0.168$                | $0.589$                   |
| **Random Forest** | **$0.968 \pm 0.003$** | **$0.528 \pm 0.103$**         | **$-0.440$**            | **$0.876$**               |
| **XGBoost**       | **$0.945 \pm 0.003$** | **$0.564 \pm 0.083$**         | **$-0.381$**            | **$0.858$**               |

---

## Citation

```bibtex
@misc{takongmo2026validation,
  author       = {Takongmo, Lionel Bl{\'e}riot},
  title        = {Validation Protocol Bias in Agricultural Yield Prediction:
                  Empirically Quantifying Optimism from Spatial Autocorrelation
                  and Proxy-Variable Leakage},
  year         = {2026},
  howpublished = {\url{https://github.com/lionel-tlb/crop-yield-validation-study}},
  note         = {Independent Applied Machine Learning Research}
}
```
