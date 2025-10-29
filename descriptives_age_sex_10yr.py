# === descriptives_age_sex_wide_pooled_10yr.py ===
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
# Age re-grouping
# ======================
def parse_age_min_max(age_str):
    """Parse age group like '15–19' or '95+'."""
    if pd.isna(age_str):
        return (None, None)
    s = str(age_str).replace("–", "-")
    if "+" in s:
        val = int(s.replace("+", "").strip())
        return (val, 120)
    try:
        a, b = s.split("-")
        return int(a), int(b)
    except Exception:
        return (None, None)

def collapse_age_groups(df):
    """Collapse 5-year bands into custom 10-year groups."""
    def to_group(age_min):
        if age_min < 15:
            return None
        elif 15 <= age_min <= 24:
            return "15–24"
        elif 25 <= age_min <= 34:
            return "25–34"
        elif 35 <= age_min <= 44:
            return "35–44"
        elif 45 <= age_min <= 54:
            return "45–54"
        elif 55 <= age_min <= 59:
            return "55–59"
        elif 60 <= age_min <= 69:
            return "60–69"
        elif 70 <= age_min <= 79:
            return "70–79"
        elif 80 <= age_min <= 89:
            return "80–89"
        else:
            return "90+"
    df["age_group_10yr"] = df["age_min"].apply(to_group)
    df = df[df["age_group_10yr"].notna()]
    return df

# ======================
# Compute weighted stats
# ======================
def compute_stats(df):
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(0)
    for c in ["ie", "contribution", "pension"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    df["age_min"] = pd.to_numeric(df["age_min"], errors="coerce").fillna(0)

    # --- Apply filter (contributors and retirees)
    df = df[
        ((df["age_min"] < 60) & ((df["ie"] > 0) | (df["contribution"] > 0))) |
        ((df["age_min"] >= 60) & (df["pension"] > 0))
    ].copy()

    # --- Collapse into 10-year age groups
    df = collapse_age_groups(df)

    results = []
    for (age, sex), g in df.groupby(["age_group_10yr", "sex"]):
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
            "age_group": age,
            "sex": sex,
            "ie_P10": ie_p10, "ie_mean": ie_mean, "ie_P90": ie_p90,
            "contr_P10": contr_p10, "contr_mean": contr_mean, "contr_P90": contr_p90,
            "pens_P10": pens_p10, "pens_mean": pens_mean, "pens_P90": pens_p90
        })
    return pd.DataFrame(results)

# ======================
# Load all years and compute pooled stats
# ======================
print("🔹 Computing pooled descriptive statistics (2015–2024, 10-year groups)...")

micro_all = []
for y in years:
    fp = out_dir / f"micro_{y}.csv"
    if fp.exists():
        df = pd.read_csv(fp)
        df["year"] = y
        micro_all.append(df)

micro_pooled = pd.concat(micro_all, ignore_index=True)
df_stats = compute_stats(micro_pooled)

# ======================
# Reshape for presentation
# ======================
def reshape_pooled(df):
    rows = []
    for age in sorted(df["age_group"].unique(), key=lambda x: (int(x.split("–")[0].replace("+","")))):
        sub = df[df["age_group"] == age]
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

df_wide_pooled = reshape_pooled(df_stats)

# ======================
# Save Excel output
# ======================
out_pooled = desc_dir / "micro_age_sex_table_pooled_10yr_2015_2024.xlsx"
with pd.ExcelWriter(out_pooled, engine="openpyxl") as writer:
    df_wide_pooled.to_excel(writer, index=False, sheet_name="Pooled_10yr")

print(f"\n Pooled descriptive table (10-year groups) saved → {out_pooled}")
