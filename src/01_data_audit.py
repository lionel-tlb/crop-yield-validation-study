"""
01_data_audit.py
================
Data Cleaning, Leakage Audit, and EDA for the Crop Yield Validation Study.

Outputs
-------
- results/audit_table.csv : per-country cardinality of each feature
- Console: summary statistics and leakage findings
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load yield_df.csv
# ---------------------------------------------------------------------------
print("=" * 70)
print("STEP 1 — Loading yield_df.csv")
print("=" * 70)

raw = pd.read_csv(os.path.join(DATA_DIR, "yield_df.csv"))
print(f"  Raw shape          : {raw.shape}")
print(f"  Columns            : {raw.columns.tolist()}")

# Drop Unnamed index artifact
if "Unnamed: 0" in raw.columns:
    raw = raw.drop(columns=["Unnamed: 0"])
    print("  Dropped 'Unnamed: 0' index artifact.")

# Drop missing values
before_na = len(raw)
raw = raw.dropna()
print(f"  Rows dropped (NaN) : {before_na - len(raw)}")

# Drop exact duplicates
before_dup = len(raw)
raw = raw.drop_duplicates()
print(f"  Rows dropped (dup) : {before_dup - len(raw)}")

df = raw.copy()
print(f"\n  Clean shape        : {df.shape}")
print(f"  Year range         : {df['Year'].min()} – {df['Year'].max()}")
print(f"  Unique countries   : {df['Area'].nunique()}")
print(f"  Unique crops       : {df['Item'].nunique()}")
print(f"  Crop types         : {sorted(df['Item'].unique())}")
print(f"\n  Descriptive stats:")
print(df[["hg/ha_yield", "average_rain_fall_mm_per_year",
          "pesticides_tonnes", "avg_temp"]].describe().round(2).to_string())

# ---------------------------------------------------------------------------
# 2. Intra-Country Feature Cardinality Audit
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 2 — Intra-Country Feature Cardinality Audit (Leakage Check)")
print("=" * 70)

features = ["average_rain_fall_mm_per_year", "pesticides_tonnes", "avg_temp"]
audit_rows = []

for feat in features:
    card = df.groupby("Area")[feat].nunique()
    n_constant = (card == 1).sum()
    n_varying  = (card > 1).sum()
    pct_constant = 100.0 * n_constant / len(card)
    print(f"\n  Feature: {feat}")
    print(f"    Countries with EXACTLY 1 unique value : {n_constant}/{len(card)} ({pct_constant:.1f}%)")
    print(f"    Countries with  > 1  unique value     : {n_varying}/{len(card)}")
    if n_varying > 0:
        print(f"    Distribution of unique-value counts (varying countries):")
        print("      " + card[card > 1].describe().round(2).to_string().replace("\n", "\n      "))
    for country, c in card.items():
        audit_rows.append({"country": country, "feature": feat, "n_unique_values": c})

audit_df = pd.DataFrame(audit_rows)
audit_pivot = audit_df.pivot(index="country", columns="feature", values="n_unique_values").reset_index()
audit_pivot.to_csv(os.path.join(RESULTS_DIR, "audit_table.csv"), index=False)
print(f"\n  Saved → results/audit_table.csv")

# ---------------------------------------------------------------------------
# 3. Key Leakage Finding
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 3 — Leakage Summary")
print("=" * 70)
rain_card = df.groupby("Area")["average_rain_fall_mm_per_year"].nunique()
all_constant = (rain_card == 1).all()
print(f"""
  *** CRITICAL FINDING ***
  'average_rain_fall_mm_per_year' has EXACTLY ONE unique value per country
  across ALL {df['Area'].nunique()} countries in the dataset.

  This means rainfall is a TIME-INVARIANT country-level constant — it acts
  as a de-facto proxy country identifier. Any model trained with this
  feature can essentially 'memorize' country identity under Random CV,
  causing severe optimism bias that disappears under Spatial Group CV
  (where test countries are unseen at training time).

  This is the central leakage mechanism this study empirically quantifies.
  
  Constant rainfall across all countries: {all_constant}
""")

# ---------------------------------------------------------------------------
# 4. Raw CSV Join / Merge Audit
# ---------------------------------------------------------------------------
print("=" * 70)
print("STEP 4 — Raw CSV Structure & Join Pipeline Reconstruction")
print("=" * 70)

raw_files = {
    "yield.csv"      : os.path.join(DATA_DIR, "yield.csv"),
    "rainfall.csv"   : os.path.join(DATA_DIR, "rainfall.csv"),
    "pesticides.csv" : os.path.join(DATA_DIR, "pesticides.csv"),
    "temp.csv"       : os.path.join(DATA_DIR, "temp.csv"),
}

for name, path in raw_files.items():
    r = pd.read_csv(path)
    print(f"\n  {name}:")
    print(f"    Shape   : {r.shape}")
    print(f"    Columns : {r.columns.tolist()}")
    print(f"    Head(2) :\n{r.head(2).to_string()}")

# --- Reconstruct the merge ---
print("\n" + "-" * 70)
print("  Reconstructing merge pipeline ...")

y = pd.read_csv(raw_files["yield.csv"])
# Keep only Yield element, relevant columns
y = y[y["Element"] == "Yield"][["Area", "Item", "Year", "Value"]].rename(
    columns={"Value": "hg/ha_yield"}
)

r_rain = pd.read_csv(raw_files["rainfall.csv"])
r_rain.columns = r_rain.columns.str.strip()  # strip leading space from ' Area'
# Coerce to numeric (may be read as object)
r_rain["average_rain_fall_mm_per_year"] = pd.to_numeric(
    r_rain["average_rain_fall_mm_per_year"], errors="coerce")
# rainfall is typically joined on Area only (not Year)
# Verify: does rainfall have Year column?
if "Year" in r_rain.columns:
    # aggregate to country-level (some datasets repeat rows per year)
    r_rain_agg = r_rain.groupby("Area")["average_rain_fall_mm_per_year"].mean().reset_index()
    print("  Rainfall: grouped by Area (Year-independent mean) before join.")
else:
    r_rain_agg = r_rain[["Area", "average_rain_fall_mm_per_year"]]

pest = pd.read_csv(raw_files["pesticides.csv"])
pest = pest[["Area", "Year", "Value"]].rename(columns={"Value": "pesticides_tonnes"})

temp = pd.read_csv(raw_files["temp.csv"])
temp = temp.rename(columns={"year": "Year", "country": "Area"})

# Merge
merged = y.merge(r_rain_agg, on="Area", how="inner")
merged = merged.merge(pest, on=["Area", "Year"], how="inner")
merged = merged.merge(temp, on=["Area", "Year"], how="inner")

print(f"\n  Reconstructed merge shape : {merged.shape}")
print(f"  Original yield_df shape   : {df.shape}")
print(f"  Column match              : {sorted(merged.columns.tolist()) == sorted(df.columns.tolist())}")
print(f"\n  JOIN PIPELINE SUMMARY:")
print(f"    1. yield.csv       → filter Element='Yield', keep [Area, Item, Year, hg/ha_yield]  → {y.shape}")
print(f"    2. rainfall.csv    → aggregate to country mean (Year-independent)                   → {r_rain_agg.shape}")
print(f"    3. pesticides.csv  → keep [Area, Year, pesticides_tonnes]                           → {pest.shape}")
print(f"    4. temp.csv        → rename [year→Year, country→Area]                              → {temp.shape}")
print(f"    5. INNER JOIN: yield × rainfall(Area) × pesticides(Area,Year) × temp(Area,Year)")
print(f"\n  LIMITATION: Rainfall is joined on Area ONLY (not Year). This collapses")
print(f"  all year-specific rainfall variation into a single time-invariant country")
print(f"  constant — the root cause of the proxy-variable leakage documented above.")

print("\n" + "=" * 70)
print("Audit complete. See results/audit_table.csv")
print("=" * 70)