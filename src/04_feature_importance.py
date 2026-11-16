"""
04_feature_importance.py
========================
Computes Mean Decrease in Impurity (MDI) and Permutation Importance
on held-out test splits under Random CV (Protocol A) with fixed random_state.
Saves results to results/feature_importance.csv.
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

df = pd.read_csv(os.path.join(DATA_DIR, "yield_df.csv")).drop(columns=["Unnamed: 0"], errors="ignore").dropna().drop_duplicates()

FEATURES = ["Item", "average_rain_fall_mm_per_year", "pesticides_tonnes", "avg_temp"]
NUMERIC  = ["average_rain_fall_mm_per_year", "pesticides_tonnes", "avg_temp"]
X = df[FEATURES]
y = df["hg/ha_yield"].values

prep = ColumnTransformer([
    ("num", StandardScaler(), NUMERIC),
    ("cat", OneHotEncoder(sparse_output=False), ["Item"])
])
X_trans = prep.fit_transform(X)
cat_names = prep.named_transformers_["cat"].get_feature_names_out(["Item"])
feat_names = NUMERIC + list(cat_names)

ss = ShuffleSplit(n_splits=5, test_size=0.20, random_state=42)
rf_mdi_list, rf_perm_list = [], []
xgb_mdi_list, xgb_perm_list = [], []

print("Computing feature importances across 5 splits...")
for split_idx, (train_idx, test_idx) in enumerate(ss.split(X)):
    print(f"  Split {split_idx + 1}/5 ...")
    # RF
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_trans[train_idx], y[train_idx])
    rf_mdi_list.append(rf.feature_importances_)
    perm_rf = permutation_importance(rf, X_trans[test_idx], y[test_idx], n_repeats=5, random_state=42, n_jobs=-1)
    rf_perm_list.append(perm_rf.importances_mean)

    # XGB
    xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1, verbosity=0)
    xgb.fit(X_trans[train_idx], y[train_idx])
    xgb_mdi_list.append(xgb.feature_importances_)
    perm_xgb = permutation_importance(xgb, X_trans[test_idx], y[test_idx], n_repeats=5, random_state=42, n_jobs=-1)
    xgb_perm_list.append(perm_xgb.importances_mean)

df_rf_mdi  = pd.DataFrame(rf_mdi_list, columns=feat_names)
df_rf_perm = pd.DataFrame(rf_perm_list, columns=feat_names)
df_xgb_mdi = pd.DataFrame(xgb_mdi_list, columns=feat_names)
df_xgb_perm= pd.DataFrame(xgb_perm_list, columns=feat_names)

out = []
for f in feat_names:
    out.append({
        "feature": f,
        "rf_mdi_mean": df_rf_mdi[f].mean() * 100,
        "rf_mdi_sd": df_rf_mdi[f].std(ddof=1) * 100,
        "rf_perm_mean": df_rf_perm[f].mean(),
        "rf_perm_sd": df_rf_perm[f].std(ddof=1),
        "xgb_mdi_mean": df_xgb_mdi[f].mean() * 100,
        "xgb_mdi_sd": df_xgb_mdi[f].std(ddof=1) * 100,
        "xgb_perm_mean": df_xgb_perm[f].mean(),
        "xgb_perm_sd": df_xgb_perm[f].std(ddof=1),
    })

res_df = pd.DataFrame(out)
csv_path = os.path.join(RESULTS_DIR, "feature_importance.csv")
res_df.to_csv(csv_path, index=False)
print(f"Saved -> {csv_path}")

# Print summary table for LaTeX
for row in out:
    f = row["feature"]
    if f in NUMERIC or f in ["Item_Potatoes", "Item_Cassava", "Item_Sweet potatoes"]:
        print(f"{f}: RF MDI={row['rf_mdi_mean']:.1f}+-{row['rf_mdi_sd']:.1f}%, "
              f"RF Perm={row['rf_perm_mean']:.3f}+-{row['rf_perm_sd']:.3f}, "
              f"XGB MDI={row['xgb_mdi_mean']:.1f}+-{row['xgb_mdi_sd']:.1f}%, "
              f"XGB Perm={row['xgb_perm_mean']:.3f}+-{row['xgb_perm_sd']:.3f}")
