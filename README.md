# NIFTY50 Implied Volatility Surface Reconstruction

> Predicting missing implied volatility values across strikes and timestamps using a robust multi-stage ensemble pipeline with strict zero-lookahead bias enforcement — built for the Finance Club, IIT Roorkee Open Project 2026.

---

## Table of Contents

- [Problem Overview](#problem-overview)
- [Approach & Architecture](#approach--architecture)
- [Pipeline Stages](#pipeline-stages)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Usage](#usage)
- [Configuration](#configuration)
- [Key Design Decisions](#key-design-decisions)
- [Output Format](#output-format)

---

## Problem Overview

Options markets quote value through **implied volatility (IV)** rather than price directly. An IV surface describes how IV varies across strike prices and timestamps. In practice, this surface has gaps — from illiquid strikes, sparse trading, or data filtering.

This project reconstructs those missing values from a Nifty50 options dataset using a multi-stage ensemble approach. The evaluation metric is **Mean Squared Error (MSE)** against hidden ground-truth IV values.

**Key Constraint:** Zero Lookahead Bias — all temporal features only use data from strictly prior timestamps, ensuring causality in the reconstruction pipeline.

---

## Approach & Architecture

The solution uses a **five-stage ensemble pipeline** that combines financial interpolation with machine learning:

```
Raw IV Grid (975 timestamps × 28 instruments)
        │
        ▼
┌──────────────────────────┐
│  Stage 1: Base Surface   │  ← Cross-sectional & temporal
│  Model                   │    interpolation across strikes
│  ├─ Linear interp        │
│  ├─ Quadratic fit        │
│  ├─ IDW smoothing        │
│  ├─ Temporal anchor       │
│  └─ Local surface reg.   │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│  Stage 2: 10-Seed        │  ← LightGBM ensemble on 22
│  LightGBM Ensemble       │    financial features (blend=8%)
│  (10 seeds × 800 trees)  │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│  Stage 3: Huber-Clamped  │  ← Per-instrument robust
│  Affine Calibration      │    bias/scale correction
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│  Stage 4: kNN Surface    │  ← Gaussian-kernel weighted
│  Retrieval               │    average of K=40 nearest
│  (K=40, ball tree)       │    observed IV points
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│  Stage 5: Grid-Search    │  ← Optimal α/γ blend of
│  Blend Optimization      │    all three predictors
│  → Final IV Prediction   │
└──────────────────────────┘
```

---

## Pipeline Stages

### Stage 1 — Base Surface Model

The primary predictor for each missing cell `(timestamp, strike)`:

1. **Cross-Sectional Linear Interpolation**
   - Interpolates observed IV across strikes of the same option type (CE/PE) at that timestamp
   - Provides smooth baseline across the moneyness spectrum

2. **Quadratic Refinement** (`quad_alpha = 0.015`)
   - Uses the 7 nearest-strike neighbors to fit a local quadratic function
   - Captures smile curvature near the target strike
   - Blended at 1.5% to preserve interpolation stability

3. **IDW Smoothing** (`idw_smooth = 50`)
   - Inverse-distance-weighted average over all same-type strikes at that timestamp
   - Smoothing factor of 50 prevents extreme local variations
   - Shrinkage weight `0.1%` pulls estimates toward global mean IV

4. **Temporal Anchor** (`temp_beta = 0.01`)
   - Extracts nearest observed IV for the same instrument in past timestamps (strict zero-lookahead)
   - Blends at 1% to anchor the estimate to recent trading history
   - Ensures consistency in the time dimension

5. **Local Surface Regression** (`surface_alpha = 0.02`, `time_window = 6`)
   - Fits weighted least-squares affine surface over the `(ri/N, strike)` neighborhood
   - Uses only observations from strictly prior timestamps
   - Weights by temporal and strike proximity (normalized)
   - Captures local trend in IV surface evolution

### Stage 2 — 10-Seed LightGBM Ensemble

A 10-model ensemble (different random seeds: 42, 123, 456, 789, 2024, 888, 999, 111, 222, 333) trained on 22 cross-sectional and temporal features:

**Feature set includes:**
- Log-moneyness (normalized strike deviation)
- Log-moneyness squared (smile curvature)
- Spot price, time-to-expiry, normalized timestamp
- Mean/std of cross-sectional IV
- Linear interpolated IV, smile slope, quadratic curvature
- Nearest-neighbor IVs (1st, 2nd, 3rd neighbors by strike distance)
- Distance to nearest neighbor
- Temporal predecessors (past IV, time lag, 2nd past IV, temporal mean)
- Strike-spot deviation (absolute and relative)

**Training parameters:**
- 800 estimators per seed (deep trees for variance reduction)
- Learning rate: 0.025 (conservative to avoid overfitting)
- Feature fraction: 0.85, Bagging fraction: 0.9, Bagging freq: 3
- L1/L2 regularization: 0.1 each
- Min child samples: 20 (prevent oversmoothing edge strikes)

**Blending weight:** 8% into base prediction — sufficient to correct systematic errors without introducing noise.

### Stage 3 — Huber-Clamped Affine Calibration

For each of the 28 instruments (strike × type pairs) with ≥30 observed points:

1. **Robust Outlier Removal**
   - Computes residuals: `res = actual_IV - blended_prediction`
   - Keeps points within 4.5σ of the median residual
   - Removes extreme outliers common in deep OTM options

2. **Huber Regression** (ε = 1.35)
   - Fits `y = a + b·x` using robust Huber loss
   - Resistant to remaining outliers while preserving good fits
   - More reliable than OLS on noisy options data

3. **Slope Clamping** (bounds: 0.85 to 1.15)
   - Prevents exploding slopes on edge strikes with few observations
   - Ensures affine correction stays close to identity, preventing extrapolation bias

### Stage 4 — kNN Surface Retrieval

Finds the K=40 nearest observed IV points in a normalized 5-dimensional space:

**Feature space:**
- Log-moneyness (weight: 3.0) — strong emphasis on smile structure
- Time position (ri/N) — emphasizes contemporary data
- Option type indicator (×0.5) — de-emphasizes; same type is implicit
- Relative strike distance (|strike - spot| / spot, weight: 2.0) — moneyness proxy
- Spot level (spot / mean_spot) — ensures similar market regime

**Kernel:**
- Computes distances in normalized feature space using ball tree
- Applies Gaussian kernel: `w_ij = exp(-(d_ij / bandwidth)²)`
- Bandwidth set to median of 2nd-nearest-neighbor distances (adaptive)
- Produces smooth weighted average of actual observed IV values

### Stage 5 — Grid-Search Blend Optimization

Searches over blending weights `α, γ ∈ [0, 1]` in steps of 0.1:

```
final_prediction = (1 - α - γ) · base + α · calibrated + γ · knn
```

Finds the combination that minimizes MSE on all observed points:

```
MSE = mean((observed_IV - blended_prediction)²)
```

Same optimal weights `(α*, γ*)` are then applied to all missing-value predictions.

---

## Key Features

✅ **Strict Zero-Lookahead Bias Enforcement**
- All temporal features use only data from strictly prior timestamps (`i < ri`)
- Ensures the pipeline doesn't use future information to fill past gaps
- Satisfies causality requirements for production pipelines

✅ **Robust Outlier Handling**
- Huber regression with clamped slopes on affine calibration
- Excludes extreme residuals in affine fitting
- Prevents deep OTM options from distorting predictions

✅ **Multi-Scale Ensemble**
- 10 LightGBM models with different seeds reduce variance
- Affine calibration per-instrument removes systematic biases
- kNN layer provides direct anchor to observed data

✅ **Adaptive Smoothing**
- IDW with adaptive smoothing factor prevents noise amplification
- Gaussian-kernel kNN uses adaptive bandwidth based on data density
- Grid-search auto-tunes blend weights for optimal MSE

---

## Project Structure

```
.
├── dataset.csv                         # Input: 975 × 28 IV grid with missing values
├── sandbox_solution.csv                # Input: submission template with IDs
├── FINAL_SUBMISSION_ZERO_LOOKAHEAD.csv # Output: predicted IV values (16 decimals)
└── solution.py                         # Main pipeline
```

---

## Requirements

```
python >= 3.8
numpy >= 1.19
pandas >= 1.1
scipy >= 1.5
scikit-learn >= 0.24
lightgbm >= 3.0
```

Install with:

```bash
pip install numpy pandas scipy scikit-learn lightgbm
```

---

## Usage

Place `dataset.csv` and `sandbox_solution.csv` in the same directory as `solution.py`, then run:

```bash
python solution.py
```

### Program Flow

1. **Load & Preprocess** — Parse instrument names, compute time-to-expiry, sort by timestamp
2. **Extract Base Observations** — Build instrument index and observation 2D arrays
3. **Train Base Model** — Compute base predictions for all observed cells
4. **Train LightGBM Ensemble** — Extract 22 features, train 10 models with different seeds
5. **Blend Predictions** — Combine base and ensemble at 8% weight
6. **Affine Calibration** — Fit per-instrument Huber regressors on blended predictions
7. **kNN Retrieval** — Find K=40 nearest neighbors for all predictions
8. **Optimize Blend** — Grid-search for optimal α/γ on observed cells
9. **Export Submission** — Write final predictions to CSV

**Expected runtime:** 10–30 minutes depending on hardware (the base prediction loop over ~27,000 observed cells is the bottleneck).

---

## Configuration

All tunable parameters are defined at the top of the script:

### Ensemble Configuration

| Parameter | Default | Description |
|---|---|---|
| `BLEND` | `0.08` | LightGBM weight in base blend (8%) |
| `K_NN` | `40` | Number of neighbors for kNN retrieval |
| `SEEDS` | `[42, 123, ..., 333]` | 10 random seeds for ensemble diversity |
| `EXPIRY_DT` | `2026-01-27 15:30:00` | Option expiry date for time-to-expiry calculation |

### Interpolation & Smoothing

| Parameter | Default | Description |
|---|---|---|
| `SHRINK_LAMBDA` | `0.001` | IDW shrinkage weight (0.1%) toward global mean |
| `IDW_SMOOTH` | `50` | IDW distance smoothing constant (strike units) |
| `QUAD_ALPHA` | `0.015` | Quadratic refinement blend weight (1.5%) |
| `TEMP_BETA` | `0.01` | Temporal anchor blend weight (1%) |
| `SURFACE_ALPHA` | `0.02` | Local surface regression blend weight (2%) |
| `TIME_WINDOW` | `6` | ±Timestamp radius for local surface regression |

### Calibration

| Parameter | Default | Description |
|---|---|---|
| Huber ε | `1.35` | Robust regression threshold for outlier detection |
| Slope bounds | `[0.85, 1.15]` | Affine slope clamping to prevent extrapolation |

---

## Key Design Decisions

**Why strict zero-lookahead only?**
This ensures the reconstruction is causally valid. If deployed in production to fill real-time data gaps, the pipeline won't accidentally depend on future information. All temporal features (`past_ti < ri`) and local surface regression use only prior data.

**Why 8% LightGBM blend?**
The base interpolation is near-optimal for smooth IV surfaces on observed cells. Aggressive ML blending (>15%) introduces noise from training patterns that don't generalize to missing cells. The 8% weight lets the ensemble correct systematic errors (e.g., smile curvature near ATM) without overriding the interpolation signal.

**Why per-instrument affine calibration?**
Different strikes and option types exhibit consistent biases: deep OTM IVs are systematically over/underestimated. A per-instrument linear correction `y = a + b·x` removes these biases without requiring complex per-instrument models. Huber regression + outlier removal make this robust to edge cases.

**Why Huber regression with slope clamping?**
- **Huber loss** is more robust than OLS to outliers common in illiquid options
- **Slope clamping [0.85, 1.15]** prevents runaway corrections on strikes with few observations
- Prevents the calibration layer from extrapolating aggressively

**Why kNN over the full surface?**
The kNN component retrieves *actual observed IV values* (not predictions) from geometrically similar positions. This provides a direct anchor to real data and guards against cascade errors (prediction → calibration → extrapolation).

**Why grid-search blend?**
- **Interpretable** — inspect which layers matter (α*, γ*)
- **Robust** — avoids nested hyperparameter tuning
- **Fast** — only 121 grid points
- **Adaptive** — auto-detects the best mix for the specific dataset

---

## Output Format

`FINAL_SUBMISSION_ZERO_LOOKAHEAD.csv` contains two columns:

| Column | Description |
|---|---|
| `id` | Unique identifier matching `sandbox_solution.csv` |
| `value` | Predicted implied volatility (float, 16 decimal places) |

Predictions are clipped to a minimum of `0.001` to avoid non-positive IV values. The file lists predictions for all missing cells in `dataset.csv`.

### Example Output

```
id,value
2026-01-20 09:15||NIFTY22000CE,0.2145678901234567
2026-01-20 09:15||NIFTY22000PE,0.2143567890123456
...
```

---

## Performance Notes

- **Training time:** ~15 min (dominated by base prediction loop over 27K observed cells)
- **Memory usage:** ~500 MB (LightGBM models + feature arrays)
- **Scalability:** Linear in number of observed cells; can handle 50K+ observations

---

## Author Notes

This pipeline balances three principles:
1. **Reliability** — Interpolation-first approach with sparse ML corrections
2. **Robustness** — Outlier handling, slope clamping, affine calibration
3. **Causality** — Strict zero-lookahead ensures production readiness

The 10-seed ensemble + Huber calibration + kNN blending combination has been tuned for variance minimization while preserving the smooth structure of IV surfaces.

