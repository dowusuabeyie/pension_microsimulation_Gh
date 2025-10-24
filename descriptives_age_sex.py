# === descriptives_age_sex_wide_all_years_pooled.py ===
import numpy as np
import pandas as pd
from pathlib import Path
from utils import load_config, resolve_path

# ======================
# Load config and paths
# ======================
cfg = load_config("config.yaml")
root = Path(cfg["project"]["root_dir"])
out_dir = resolve_path(root, cfg["project"]["output_dir"])
desc_dir = root / "descriptives"
desc_dir.mkdir(exist_ok=True)

years = list(range(int(cfg["years"]["start"]), int(cfg["years"]["end"]) + 1))

# ======================
# Weighted percentile function
# ======================
def weighted_percentile(values, weights, percentiles):
    values, weights = np.array(values), np.array(weights)
    mask = np.isfinite(values) & np.isfinite(weights)
    if not mask.any():
        return [np.nan] * len(percentiles)
    values, weights = values[mask], weights[mask]
    sorter = np.argsort(values)
    values, weights = values[sorter], weights[sorter]
    cum_weights = np.cumsum(weights)
    cum_weights /= cum_weights[-1]
    return np.interp(np.array(percentiles) / 100.0, cum_weights, values)

# ======================
# Compute stats for one dataset
# ======================
def compute_stats(df, year):
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(0)
    df["ie"] = pd.to_numeric(df["ie"], errors="coerce").fillna(0)
    df["contribution"] = pd.to_numeric(df["contribution"], errors="coerce").fillna(0)
    df["pension"] = pd.to_numeric(df["pension"], errors="coerce").fillna(0)
    df["age_min"] = pd.to_numeric(df["age_min"], errors="coerce").fillna(0)

    # Filter invalid combinations:
    df = df[
        ((df["age_min"] < 60) & ((df["ie"] > 0) | (df["contribution"] > 0))) |
        ((df["age_min"] >= 60) & (df["pension"] > 0))
    ].copy()

    results = []
    for (age, sex), g in df.groupby(["age_grp", "sex"]):
        w = g["weight"].to_numpy()
        if w.sum() == 0:
            continue

        def stats(col):
            p10, p90 = weighted_percentile(g[col], w, [10, 90])
            mean = np.average(g[col], weights=w)
            return p10, mean, p90

        ie_p10, ie_mean, ie_p90 = stats("ie")
        contr_p10, contr_mean, contr_p90 = stats("contribution")
        pens_p10, pens_mean, pens_p90 = stats("pension")

        results.append({
            "year": year,
            "age_grp": age,
            "sex": sex,
            "ie_P10": ie_p10, "ie_mean": ie_mean, "ie_P90": ie_p90,
            "contr_P10": contr_p10, "contr_mean": contr_mean, "contr_P90": contr_p90,
            "pens_P10": pens_p10, "pens_mean": pens_mean, "pens_P90": pens_p90
        })
    return pd.DataFrame(results)

# ======================
# Build all datasets
# ======================
all_results = []
for y in years:
    fp = out_dir / f"micro_{y}.csv"
    if not fp.exists():
        print(f"⚠️ Skipping {y} (file not found)")
        continue
    print(f"Processing {y}...")
    df = pd.read_csv(fp)
    df_stats = compute_stats(df, y)
    all_results.append(df_stats)

df_all = pd.concat(all_results, ignore_index=True)

# ======================
# Function to reshape wide layout
# ======================
def reshape_for_display(df):
    rows = []
    for (year, age) in sorted(df.groupby(["year", "age_grp"]).groups.keys(), key=lambda x: (x[0], x[1])):
        sub = df[(df["year"] == year) & (df["age_grp"] == age)]
        for label, suffix in [("P10", "_P10"), ("Mean", "_mean"), ("P90", "_P90")]:
            row = {"Year": year, "Age Group": age, "Moment": label}
            for sex in ["M", "F"]:
                sdata = sub[sub["sex"] == sex]
                row[f"{sex}_ie"] = sdata[f"ie{suffix}"].squeeze() if not sdata.empty else np.nan
                row[f"{sex}_contr"] = sdata[f"contr{suffix}"].squeeze() if not sdata.empty else np.nan
                row[f"{sex}_pens"] = sdata[f"pens{suffix}"].squeeze() if not sdata.empty else np.nan
            rows.append(row)
    wide = pd.DataFrame(rows)
    wide = wide.rename(columns={
        "M_ie": "ie (M)", "M_contr": "contribution (M)", "M_pens": "pension (M)",
        "F_ie": "ie (F)", "F_contr": "contribution (F)", "F_pens": "pension (F)"
    })
    return wide

# ======================
# Pooled 2015–2024 dataset
# ======================
print("Computing pooled 2015–2024 averages...")

# Load and combine all microdata for pooled analysis
micro_pooled = []
for y in years:
    fp = out_dir / f"micro_{y}.csv"
    if fp.exists():
        df = pd.read_csv(fp)
        df["year"] = y
        micro_pooled.append(df)
micro_pooled = pd.concat(micro_pooled, ignore_index=True)

df_pooled_stats = compute_stats(micro_pooled, year="2015–2024")

# Create pooled wide table (no Year column in results)
def reshape_pooled(df):
    rows = []
    for age in sorted(df["age_grp"].unique()):
        sub = df[df["age_grp"] == age]
        for label, suffix in [("P10", "_P10"), ("Mean", "_mean"), ("P90", "_P90")]:
            row = {"Age Group": age, "Moment": label}
            for sex in ["M", "F"]:
                sdata = sub[sub["sex"] == sex]
                row[f"{sex}_ie"] = sdata[f"ie{suffix}"].squeeze() if not sdata.empty else np.nan
                row[f"{sex}_contr"] = sdata[f"contr{suffix}"].squeeze() if not sdata.empty else np.nan
                row[f"{sex}_pens"] = sdata[f"pens{suffix}"].squeeze() if not sdata.empty else np.nan
            rows.append(row)
    wide = pd.DataFrame(rows)
    wide = wide.rename(columns={
        "M_ie": "ie (M)", "M_contr": "contribution (M)", "M_pens": "pension (M)",
        "F_ie": "ie (F)", "F_contr": "contribution (F)", "F_pens": "pension (F)"
    })
    return wide

df_wide_all = reshape_for_display(df_all)
df_wide_2023 = reshape_for_display(df_all[df_all["year"] == 2023])
df_wide_2024 = reshape_for_display(df_all[df_all["year"] == 2024])
df_wide_pooled = reshape_pooled(df_pooled_stats)

# ======================
# Export all four files
# ======================
out_all = desc_dir / "micro_age_sex_table_2015_2024.xlsx"
out_2023 = desc_dir / "micro_age_sex_table_2023.xlsx"
out_2024 = desc_dir / "micro_age_sex_table_2024.xlsx"
out_pooled = desc_dir / "micro_age_sex_table_pooled_2015_2024.xlsx"

with pd.ExcelWriter(out_all, engine="openpyxl") as writer:
    df_wide_all.to_excel(writer, index=False, sheet_name="AllYears")

with pd.ExcelWriter(out_2023, engine="openpyxl") as writer:
    df_wide_2023.to_excel(writer, index=False, sheet_name="2023")

with pd.ExcelWriter(out_2024, engine="openpyxl") as writer:
    df_wide_2024.to_excel(writer, index=False, sheet_name="2024")

with pd.ExcelWriter(out_pooled, engine="openpyxl") as writer:
    df_wide_pooled.to_excel(writer, index=False, sheet_name="Pooled_2015_2024")

print("\n All descriptive tables saved successfully:")
print(f" → {out_all}")
print(f" → {out_2023}")
print(f" → {out_2024}")
print(f" → {out_pooled}")
