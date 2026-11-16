"""
03_generate_figures.py
======================
Produces 3 publication-quality figures from the experiment outputs.

Fig 1 — Protocol × Ablation grouped bar chart (R²)
Fig 2 — Temporal drift: RMSE and R² across cutoff years by model
Fig 3 — Residual error by crop type under Random vs Leave-Country-Out CV
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
sns.set_theme(style="whitegrid", font_scale=1.15)
plt.rcParams.update({
    "font.family"      : "DejaVu Sans",
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "figure.dpi"       : 150,
})

MODEL_ORDER    = ["Global Mean", "Per-Crop Mean", "Ridge", "Random Forest", "XGBoost"]
PROTOCOL_ORDER = ["Random CV", "Leave-Country-Out CV (LCO-CV)", "Temporal CV"]
PROTOCOL_COLOR = {
    "Random CV": "#2196F3",
    "Leave-Country-Out CV (LCO-CV)": "#FF9800",
    "Temporal CV": "#4CAF50"
}

DPI = 300

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
raw = pd.read_csv(os.path.join(RESULTS_DIR, "raw_fold_metrics.csv"))
# Remap Spatial CV to Leave-Country-Out CV (LCO-CV)
raw["protocol"] = raw["protocol"].replace({"Spatial CV": "Leave-Country-Out CV (LCO-CV)"})
print(f"Loaded raw_fold_metrics.csv: {raw.shape}")

# ---------------------------------------------------------------------------
# Fig 1 — Grouped bar: R² across protocols × ablation condition
# ---------------------------------------------------------------------------
print("Generating Fig 1 ...")

fig1_data = (
    raw
    .groupby(["condition", "protocol", "model"])["R2"]
    .agg(mean="mean", sd="std", ci=lambda x: x.sem() * 1.96)
    .reset_index()
)

fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), sharey=True)
fig.suptitle(
    "Fig. 1 — Optimism Bias: $R^2$ by Validation Protocol and Feature Condition",
    fontsize=14, fontweight="bold", y=1.02,
)

for ax, cond in zip(axes, ["Full", "Ablation"]):
    sub = fig1_data[fig1_data["condition"] == cond]
    x   = np.arange(len(MODEL_ORDER))
    width = 0.26
    offsets = [-width, 0, width]

    for i, proto in enumerate(PROTOCOL_ORDER):
        s = sub[sub["protocol"] == proto].set_index("model")
        means = [s.loc[m, "mean"] if m in s.index else np.nan for m in MODEL_ORDER]
        # Error bars only for Random CV and LCO-CV (20 repeated splits)
        # Omit error bars for Temporal CV (mean of deterministic cutoff point estimates)
        if proto == "Temporal CV":
            yerrs = None
        else:
            yerrs = [s.loc[m, "ci"] if m in s.index else np.nan for m in MODEL_ORDER]

        bars = ax.bar(
            x + offsets[i], means,
            width=width - 0.02,
            color=PROTOCOL_COLOR[proto],
            edgecolor="black", linewidth=0.6,
            yerr=yerrs, capsize=3,
            label=proto,
            error_kw={"elinewidth": 1.0, "ecolor": "black"},
        )

    ax.set_title(
        f"{'Full Features' if cond == 'Full' else 'No-Rainfall Ablation'}",
        fontsize=12, fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_ORDER, rotation=25, ha="right", fontsize=10)
    ax.set_ylabel("$R^2$ (mean $\\pm$ 95% CI)" if cond == "Full" else "", fontsize=11)
    ax.set_xlabel("Model", fontsize=11)
    ax.set_ylim(-0.15, 1.05)
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.05))

    if cond == "Ablation":
        ax.legend(
            title="Validation Protocol",
            loc="lower right", fontsize=9, title_fontsize=10,
        )

plt.tight_layout()
for ext in ["pdf", "png"]:
    p = os.path.join(RESULTS_DIR, f"fig1_protocol_ablation.{ext}")
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    print(f"  Saved → {p}")
plt.close()

# ---------------------------------------------------------------------------
# Fig 2 — Temporal drift: RMSE and R² by cutoff year
# ---------------------------------------------------------------------------
print("Generating Fig 2 ...")

temp_data = raw[(raw["protocol"] == "Temporal CV") & (raw["condition"] == "Full")]
temp_agg  = temp_data.groupby(["cutoff", "model"])[["RMSE", "R2"]].mean().reset_index()

MODEL_COLORS = {
    "Global Mean"  : "#9E9E9E",
    "Per-Crop Mean": "#795548",
    "Ridge"        : "#3F51B5",
    "Random Forest": "#4CAF50",
    "XGBoost"      : "#F44336",
}
MODEL_MARKERS = {
    "Global Mean"  : "s",
    "Per-Crop Mean": "D",
    "Ridge"        : "^",
    "Random Forest": "o",
    "XGBoost"      : "*",
}

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle(
    "Fig. 2 — Temporal Drift: Model Performance Across Forward Temporal Cutoffs\n"
    "(Full Features; trained on Year $\\leq$ cutoff, tested on Year > cutoff)",
    fontsize=13, fontweight="bold",
)

for ax, metric, ylabel in zip(
    axes,
    ["RMSE", "R2"],
    ["RMSE (hg/ha)", "$R^2$"],
):
    for model in MODEL_ORDER:
        sub = temp_agg[temp_agg["model"] == model]
        ax.plot(
            sub["cutoff"], sub[metric],
            marker=MODEL_MARKERS[model],
            color=MODEL_COLORS[model],
            linewidth=2.0, markersize=8,
            label=model,
        )
    ax.set_xlabel("Training Cutoff Year ($c$)", fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    ax.set_xticks([2002, 2004, 2006, 2008, 2010])
    ax.tick_params(axis="both", labelsize=13)
    ax.set_title(f"{ylabel} vs. Forward Cutoff Year", fontsize=15, fontweight="bold")
    ax.legend(fontsize=12, loc="best")

plt.tight_layout()
for ext in ["pdf", "png"]:
    p = os.path.join(RESULTS_DIR, f"fig2_temporal_drift.{ext}")
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    print(f"  Saved → {p}")
plt.close()

# ---------------------------------------------------------------------------
# Fig 3 — Residual error by crop type: Random vs Leave-Country-Out CV
# ---------------------------------------------------------------------------
print("Generating Fig 3 ...")

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import ShuffleSplit, GroupShuffleSplit
from xgboost import XGBRegressor

df = pd.read_csv(os.path.join(DATA_DIR, "yield_df.csv"))
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])
df = df.dropna().drop_duplicates().reset_index(drop=True)

FEATURES = ["Item", "average_rain_fall_mm_per_year", "pesticides_tonnes", "avg_temp"]
NUMERIC  = ["average_rain_fall_mm_per_year", "pesticides_tonnes", "avg_temp"]
X = df[FEATURES]
y_arr = df["hg/ha_yield"].values

prep = ColumnTransformer([
    ("num", StandardScaler(), NUMERIC),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["Item"]),
])

xgb_pipe_rand = Pipeline([("prep", prep), ("model", XGBRegressor(
    n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1,
    verbosity=0, eval_metric="rmse"))])
xgb_pipe_spatial = Pipeline([("prep", ColumnTransformer([
    ("num", StandardScaler(), NUMERIC),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["Item"]),
])), ("model", XGBRegressor(
    n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1,
    verbosity=0, eval_metric="rmse"))])

residual_rows = []

# Random CV — first fold
ss = ShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
for train_idx, test_idx in ss.split(X):
    xgb_pipe_rand.fit(X.iloc[train_idx], y_arr[train_idx])
    preds = xgb_pipe_rand.predict(X.iloc[test_idx])
    for i, idx in enumerate(test_idx):
        residual_rows.append({
            "protocol": "Random CV",
            "Item": df.loc[idx, "Item"],
            "y_true": y_arr[idx],
            "y_pred": preds[i],
            "AbsError": abs(y_arr[idx] - preds[i]),
        })

# Leave-Country-Out CV — first fold
gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
groups = df["Area"].values
for train_idx, test_idx in gss.split(X, y_arr, groups=groups):
    xgb_pipe_spatial.fit(X.iloc[train_idx], y_arr[train_idx])
    preds = xgb_pipe_spatial.predict(X.iloc[test_idx])
    for i, idx in enumerate(test_idx):
        residual_rows.append({
            "protocol": "Leave-Country-Out CV (LCO-CV)",
            "Item": df.loc[idx, "Item"],
            "y_true": y_arr[idx],
            "y_pred": preds[i],
            "AbsError": abs(y_arr[idx] - preds[i]),
        })

res_df = pd.DataFrame(residual_rows)

# Sort crops by median error (Random CV)
crop_order = (
    res_df[res_df["protocol"] == "Random CV"]
    .groupby("Item")["AbsError"]
    .median()
    .sort_values()
    .index.tolist()
)

fig, ax = plt.subplots(figsize=(16, 8.5))
fig.suptitle(
    "Fig. 3 — Absolute Prediction Error by Crop Type\n"
    "XGBoost (Full Features): Random CV vs. Leave-Country-Out (Group) CV",
    fontsize=14, fontweight="bold",
)

palette = {
    "Random CV": "#2196F3",
    "Leave-Country-Out CV (LCO-CV)": "#FF9800"
}
sns.boxplot(
    data=res_df,
    x="Item", y="AbsError",
    hue="protocol",
    order=crop_order,
    palette=palette,
    linewidth=1.2,
    flierprops={"marker": ".", "markersize": 3, "alpha": 0.35},
    ax=ax,
)
ax.set_xlabel("Crop Type (Item)", fontsize=15)
ax.set_ylabel("Absolute Error (hg/ha)", fontsize=15)
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=13)
ax.tick_params(axis="y", labelsize=13)
ax.legend(title="Validation Protocol", fontsize=13, title_fontsize=13)
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
    lambda x, _: f"{int(x):,}"))

plt.tight_layout()
for ext in ["pdf", "png"]:
    p = os.path.join(RESULTS_DIR, f"fig3_error_by_crop.{ext}")
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    print(f"  Saved → {p}")
plt.close()

print("\nAll 3 figures generated successfully with updated labels.")
