# descriptives_micro.py
import os
import pandas as pd
import numpy as np
from pathlib import Path
from utils import load_config, resolve_path, load_demog_for_year

# ================================
# Setup paths and configuration
# ================================
cfg = load_config("config.yaml")
root = Path(cfg["project"]["root_dir"])
out_dir = resolve_path(root, cfg["project"]["output_dir"])
desc_dir = root / "descriptives" 
desc_dir.mkdir(exist_ok=True)

years = list(range(int(cfg["years"]["start"]), int(cfg["years"]["end"]) + 1))

# ================================
# Initialize results
# ================================
records = []

for year in years:
    fp = out_dir / f"micro_{year}.csv"
    if not fp.exists():
        print(f"[{year}] micro file missing — skipping.")
        continue

    m = pd.read_csv(fp)
    m["weight"]       = pd.to_numeric(m["weight"], errors="coerce").fillna(0)
    m["contribution"] = pd.to_numeric(m["contribution"], errors="coerce").fillna(0)
    m["pension"]      = pd.to_numeric(m["pension"], errors="coerce").fillna(0)
    m["regr_0_ind"]   = pd.to_numeric(m["regr_0_ind"], errors="coerce").fillna(0)
    m["retir_0_ind"]  = pd.to_numeric(m["retir_0_ind"], errors="coerce").fillna(0)

    # Core micro aggregates
    contr     = ((m["sch_grp"] == "C") * m["weight"]).sum()
    retir     = ((m["sch_grp"] == "R") * m["weight"]).sum()
    regr_0    = (m["regr_0_ind"] * m["weight"]).sum()
    retir_0   = (m["retir_0_ind"] * m["weight"]).sum()
    contr_amt = (m["contribution"] * m["weight"]).sum()
    ben_amt   = (m["pension"] * m["weight"]).sum()

    # Derived metrics
    p_avg  = ben_amt / retir if retir > 0 else np.nan
    ie_avg = contr_amt / (0.11 * contr) if contr > 0 else np.nan

    # Coverage (contributors / employed pop)
    demog = load_demog_for_year(cfg, year)
    cD = cfg["columns"]["demog"]

    possible_emp_cols = [
        cD.get("emp", ""), "emp", "employment", "employed", "emp_total", "emp_rate"
    ]
    emp_col = next((col for col in possible_emp_cols if col in demog.columns), None)

    if emp_col is not None:
        emp_total = pd.to_numeric(demog[emp_col], errors="coerce").sum()
    else:
        emp_total = np.nan
        print(f"[{year}] ⚠️ No employment column found in demographic data. Coverage skipped.")

    cov = contr / emp_total * 100 if emp_total > 0 else np.nan

    records.append({
        "cal_yr": year,
        "contr": contr,
        "retir": retir,
        "retir_0": retir_0,
        "regr_0": regr_0,
        "contr_amt": contr_amt / 1e9, 
        "ben_amt": ben_amt / 1e9,
        "p_avg": p_avg,
        "ie_avg": ie_avg,
        "cov": cov,
    })

# ===============================
# Convert to DataFrame
# ===============================
df = pd.DataFrame(records)

# compute summary statistics
summary = pd.DataFrame({
    "mean": df.mean(numeric_only=True),
    "median": df.median(numeric_only=True),
    "std": df.std(numeric_only=True, ddof=1)
}).T

# ===============================
# Export to Excel
# ===============================
out_xlsx = desc_dir / "micro_descriptives.xlsx"
with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name="yearly_micro", index=False)
    summary.to_excel(writer, sheet_name="summary_stats")

print(f"\n Descriptive statistics exported here {out_xlsx}")
