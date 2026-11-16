"""
02_run_experiments.py
=====================
Main experimental loop: 5 models × 3 validation protocols × 2 feature ablations.

Models
------
1. Global Mean (DummyRegressor)
2. Per-Crop Mean (LinearRegression on one-hot Item only)
3. Ridge Regression
4. Random Forest
5. XGBoost

Validation Protocols
--------------------
A. Random CV        : ShuffleSplit(n_splits=20, test_size=0.20, random_state=42)
B. Spatial Group CV : GroupShuffleSplit(n_splits=20, test_size=0.20, random_state=42)
C. Temporal Forward : cutoffs in [2002, 2004, 2006, 2008, 2010]

Feature Conditions
------------------
Full     : Item + rainfall + pesticides + avg_temp
Ablation : Item + pesticides + avg_temp  (no rainfall)

Outputs
-------
- results/raw_fold_metrics.csv
- results/summary_table_full.csv
- results/summary_table_ablation.csv
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import ShuffleSplit, GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1. Load & Clean Data
# ---------------------------------------------------------------------------
print("Loading data ...")
df = pd.read_csv(os.path.join(DATA_DIR, "yield_df.csv"))
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])
df = df.dropna().drop_duplicates().reset_index(drop=True)
print(f"  Clean shape: {df.shape}")

TARGET = "hg/ha_yield"
y = df[TARGET].values

FEATURES_FULL     = ["Item", "average_rain_fall_mm_per_year", "pesticides_tonnes", "avg_temp"]
FEATURES_ABLATION = ["Item", "pesticides_tonnes", "avg_temp"]

NUMERIC_FULL     = ["average_rain_fall_mm_per_year", "pesticides_tonnes", "avg_temp"]
NUMERIC_ABLATION = ["pesticides_tonnes", "avg_temp"]
CAT_FEATURES     = ["Item"]

# ---------------------------------------------------------------------------
# 2. Pipeline Builders
# ---------------------------------------------------------------------------

def build_preprocessor(numeric_features):
    return ColumnTransformer(transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATURES),
    ])


def build_pipelines(numeric_features):
    """Return dict of model_name → sklearn Pipeline."""
    prep = build_preprocessor(numeric_features)

    # For Per-Crop Mean we only need the OHE on Item
    item_only_prep = ColumnTransformer(transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATURES),
    ])

    pipelines = {
        "Global Mean": Pipeline([
            ("prep", build_preprocessor(numeric_features)),
            ("model", DummyRegressor(strategy="mean")),
        ]),
        "Per-Crop Mean": Pipeline([
            ("prep", item_only_prep),
            ("model", LinearRegression()),
        ]),
        "Ridge": Pipeline([
            ("prep", build_preprocessor(numeric_features)),
            ("model", Ridge(alpha=1.0)),
        ]),
        "Random Forest": Pipeline([
            ("prep", build_preprocessor(numeric_features)),
            ("model", RandomForestRegressor(
                n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)),
        ]),
        "XGBoost": Pipeline([
            ("prep", build_preprocessor(numeric_features)),
            ("model", XGBRegressor(
                n_estimators=100, learning_rate=0.1,
                random_state=RANDOM_STATE, n_jobs=-1,
                verbosity=0, eval_metric="rmse")),
        ]),
    }
    return pipelines


# ---------------------------------------------------------------------------
# 3. Metric Helper
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    return mae, rmse, r2


def ci95(values):
    """95% CI half-width via t-distribution."""
    n = len(values)
    if n < 2:
        return np.nan
    se = np.std(values, ddof=1) / np.sqrt(n)
    return se * stats.t.ppf(0.975, df=n - 1)


# ---------------------------------------------------------------------------
# 4. Validation Protocols
# ---------------------------------------------------------------------------

def run_random_cv(X, y_arr, pipelines):
    """ShuffleSplit random CV."""
    ss = ShuffleSplit(n_splits=20, test_size=0.20, random_state=RANDOM_STATE)
    records = []
    for fold_idx, (train_idx, test_idx) in enumerate(ss.split(X)):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y_arr[train_idx], y_arr[test_idx]
        for name, pipe in pipelines.items():
            pipe.fit(X_tr, y_tr)
            mae, rmse, r2 = compute_metrics(y_te, pipe.predict(X_te))
            records.append({"protocol": "Random CV", "fold": fold_idx,
                            "model": name, "MAE": mae, "RMSE": rmse, "R2": r2})
    return records


def run_spatial_cv(X, y_arr, groups, pipelines):
    """GroupShuffleSplit by country."""
    gss = GroupShuffleSplit(n_splits=20, test_size=0.20, random_state=RANDOM_STATE)
    records = []
    for fold_idx, (train_idx, test_idx) in enumerate(gss.split(X, y_arr, groups=groups)):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y_arr[train_idx], y_arr[test_idx]
        for name, pipe in pipelines.items():
            pipe.fit(X_tr, y_tr)
            mae, rmse, r2 = compute_metrics(y_te, pipe.predict(X_te))
            records.append({"protocol": "Spatial CV", "fold": fold_idx,
                            "model": name, "MAE": mae, "RMSE": rmse, "R2": r2})
    return records


def run_temporal_cv(df_full, feature_cols, y_arr, pipelines):
    """Rolling forward temporal split at 5 cutoffs."""
    cutoffs = [2002, 2004, 2006, 2008, 2010]
    records = []
    for cutoff in cutoffs:
        train_mask = df_full["Year"] <= cutoff
        test_mask  = df_full["Year"] > cutoff
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            print(f"  Skipping cutoff {cutoff}: insufficient data.")
            continue
        X_tr = df_full.loc[train_mask, feature_cols]
        X_te = df_full.loc[test_mask,  feature_cols]
        y_tr = y_arr[train_mask.values]
        y_te = y_arr[test_mask.values]
        for name, pipe in pipelines.items():
            pipe.fit(X_tr, y_tr)
            mae, rmse, r2 = compute_metrics(y_te, pipe.predict(X_te))
            records.append({"protocol": "Temporal CV", "fold": cutoff,
                            "model": name, "MAE": mae, "RMSE": rmse, "R2": r2,
                            "cutoff": cutoff})
    return records


# ---------------------------------------------------------------------------
# 5. Summary Table Builder
# ---------------------------------------------------------------------------

def build_summary(records_df, protocols=None):
    if protocols:
        records_df = records_df[records_df["protocol"].isin(protocols)]
    rows = []
    for (protocol, model), grp in records_df.groupby(["protocol", "model"]):
        for metric in ["MAE", "RMSE", "R2"]:
            vals = grp[metric].values
            rows.append({
                "Protocol": protocol,
                "Model": model,
                "Metric": metric,
                "Mean": np.mean(vals),
                "SD": np.std(vals, ddof=1),
                "CI95": ci95(vals),
                "Min": np.min(vals),
                "Max": np.max(vals),
                "N_folds": len(vals),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 6. Main Loop
# ---------------------------------------------------------------------------

all_records = []

for condition_name, feature_cols, numeric_cols in [
    ("Full",     FEATURES_FULL,     NUMERIC_FULL),
    ("Ablation", FEATURES_ABLATION, NUMERIC_ABLATION),
]:
    print(f"\n{'='*70}")
    print(f"CONDITION: {condition_name}  |  Features: {feature_cols}")
    print(f"{'='*70}")

    X = df[feature_cols].copy()
    pipes = build_pipelines(numeric_cols)

    # --- Random CV ---
    print("  Running Random CV (20 folds) ...")
    recs = run_random_cv(X, y, pipes)
    for r in recs:
        r["condition"] = condition_name
    all_records.extend(recs)
    print(f"    Done. {len(recs)} fold-model records.")

    # --- Spatial CV ---
    print("  Running Spatial CV (20 folds) ...")
    groups = df["Area"].values
    recs = run_spatial_cv(X, y, groups, pipes)
    for r in recs:
        r["condition"] = condition_name
    all_records.extend(recs)
    print(f"    Done. {len(recs)} fold-model records.")

    # --- Temporal Forward CV ---
    print("  Running Temporal Forward CV (5 cutoffs) ...")
    recs = run_temporal_cv(df, feature_cols, y, pipes)
    for r in recs:
        r["condition"] = condition_name
    all_records.extend(recs)
    print(f"    Done. {len(recs)} fold-model records.")

# ---------------------------------------------------------------------------
# 7. Save Outputs
# ---------------------------------------------------------------------------
raw_df = pd.DataFrame(all_records)
raw_df.to_csv(os.path.join(RESULTS_DIR, "raw_fold_metrics.csv"), index=False)
print(f"\nSaved raw_fold_metrics.csv  ({len(raw_df)} rows)")

for cond in ["Full", "Ablation"]:
    sub = raw_df[raw_df["condition"] == cond]
    summary = build_summary(sub)
    fname = f"summary_table_{cond.lower()}.csv"
    summary.to_csv(os.path.join(RESULTS_DIR, fname), index=False)
    print(f"Saved {fname}  ({len(summary)} rows)")

# ---------------------------------------------------------------------------
# 8. Console Summary (R² only, for quick review)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("QUICK SUMMARY — R² (Mean ± SD)  [Full Features]")
print("=" * 70)
sub = raw_df[raw_df["condition"] == "Full"]
piv = sub.groupby(["protocol", "model"])["R2"].agg(["mean", "std"]).round(4)
print(piv.to_string())

print("\n" + "=" * 70)
print("QUICK SUMMARY — R² (Mean ± SD)  [No-Rainfall Ablation]")
print("=" * 70)
sub = raw_df[raw_df["condition"] == "Ablation"]
piv = sub.groupby(["protocol", "model"])["R2"].agg(["mean", "std"]).round(4)
print(piv.to_string())

print("\n" + "=" * 70)
print("TEMPORAL DRIFT — RMSE by cutoff year  [Full Features]")
print("=" * 70)
temp_full = raw_df[(raw_df["condition"] == "Full") & (raw_df["protocol"] == "Temporal CV")]
piv2 = temp_full.pivot_table(index="cutoff", columns="model", values="RMSE", aggfunc="mean")
print(piv2.round(0).to_string())

print("\nAll experiments complete.")
