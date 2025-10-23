import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from utils import load_config, resolve_path, load_macro

cfg = load_config("config.yaml")
root = cfg["project"]["root_dir"]
out_dir = resolve_path(root, cfg["project"]["output_dir"])

years = list(range(int(cfg["years"]["start"]), int(cfg["years"]["end"]) + 1))

# load macro
macro = load_macro(cfg).set_index(cfg["columns"]["macro"]["year"])

# Multiply macro monetary amounts by multiplier (billions → base unit)
mult_contr = cfg.get("macro_units", {}).get("contributions_total_multiplier", 1.0)
mult_ben   = cfg.get("macro_units", {}).get("benefits_total_multiplier", 1.0)
macro["contr_amt_scaled"] = macro[cfg["columns"]["macro"]["contributions_total"]] * mult_contr
macro["ben_amt_scaled"]   = macro[cfg["columns"]["macro"]["benefits_total"]] * mult_ben

micro_sums = {"year": [], "micro_contr_amt": [], "micro_ben_amt": [], "micro_contributors": [], "micro_retirees": []}

for y in years:
    fp = out_dir / f"micro_{y}.csv"
    if not fp.exists():
        continue
    m = pd.read_csv(fp)
    # ensure numeric
    m["weight"]        = pd.to_numeric(m["weight"], errors="coerce").fillna(0)
    m["contribution"]  = pd.to_numeric(m["contribution"], errors="coerce").fillna(0)
    m["pension"]       = pd.to_numeric(m["pension"], errors="coerce").fillna(0)

    micro_sums["year"].append(y)
    micro_sums["micro_contr_amt"].append((m["contribution"] * m["weight"]).sum())
    micro_sums["micro_ben_amt"].append((m["pension"] * m["weight"]).sum())
    micro_sums["micro_contributors"].append(((m["sch_grp"] == "C") * m["weight"]).sum())
    micro_sums["micro_retirees"].append(((m["sch_grp"] == "R") * m["weight"]).sum())

df_s = pd.DataFrame(micro_sums).set_index("year")

fig, ax = plt.subplots(figsize=(9,5))
ax.plot(df_s.index, df_s["micro_contr_amt"] / 1e9, marker='o', label="Micro contributions (weighted, bn GHS)")
ax.plot(df_s.index, df_s["micro_ben_amt"] / 1e9, marker='o', label="Micro benefits (weighted, bn GHS)")
ax.plot(years, macro.loc[years, "contr_amt_scaled"] / 1e9, marker='s', linestyle='--', label="Macro contributions (bn GHS)")
ax.plot(years, macro.loc[years, "ben_amt_scaled"]   / 1e9, marker='s', linestyle='--', label="Macro benefits (bn GHS)")

ax.set_xlabel("Year")
ax.set_ylabel("Amount (Billion GHS)")
ax.set_title("Macro vs Micro monetary totals (weighted micro sums)")
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()
