import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from utils import load_config, resolve_path, load_macro

# Load configuration and data paths
cfg = load_config("config.yaml")
root = cfg["project"]["root_dir"]
out_dir = resolve_path(root, cfg["project"]["output_dir"])
years = list(range(int(cfg["years"]["start"]), int(cfg["years"]["end"]) + 1))

# Load macro data and prepare multipliers
macro = load_macro(cfg).set_index(cfg["columns"]["macro"]["year"])
mult_contr = cfg.get("macro_units", {}).get("contributions_total_multiplier", 1.0)
mult_ben   = cfg.get("macro_units", {}).get("benefits_total_multiplier", 1.0)

# Containers for each comparison
xs_contrib, ys_contrib = [], []   # contributors
xs_ret, ys_ret = [], []           # retirees
xs_contr_amt, ys_contr_amt = [], []  # contributions
xs_ben_amt, ys_ben_amt = [], []      # benefits

for y in years:
    fp = out_dir / f"micro_{y}.csv"
    if not fp.exists():
        continue

    m = pd.read_csv(fp)
    m["weight"]       = pd.to_numeric(m["weight"], errors="coerce").fillna(0)
    m["contribution"] = pd.to_numeric(m["contribution"], errors="coerce").fillna(0)
    m["pension"]      = pd.to_numeric(m["pension"], errors="coerce").fillna(0)

    # --- Micro weighted totals ---
    micro_contributors = ((m["sch_grp"] == "C") * m["weight"]).sum()
    micro_retirees     = ((m["sch_grp"] == "R") * m["weight"]).sum()
    micro_contr_amt    = (m["contribution"] * m["weight"]).sum()
    micro_ben_amt      = (m["pension"] * m["weight"]).sum()

    # --- Macro totals ---
    macro_contributors = float(macro.loc[y, cfg["columns"]["macro"]["contributors"]])
    macro_retirees     = float(macro.loc[y, cfg["columns"]["macro"]["retirees"]])
    macro_contr_amt    = float(macro.loc[y, cfg["columns"]["macro"]["contributions_total"]]) * mult_contr
    macro_ben_amt      = float(macro.loc[y, cfg["columns"]["macro"]["benefits_total"]]) * mult_ben

    # --- Append (billions for money) ---
    xs_contrib.append(macro_contributors)
    ys_contrib.append(micro_contributors)
    xs_ret.append(macro_retirees)
    ys_ret.append(micro_retirees)
    xs_contr_amt.append(macro_contr_amt / 1e9)
    ys_contr_amt.append(micro_contr_amt / 1e9)
    xs_ben_amt.append(macro_ben_amt / 1e9)
    ys_ben_amt.append(micro_ben_amt / 1e9)

# ---- Plot all four comparisons ----
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

def plot_scatter(ax, x, y, xlabel, ylabel, color):
    if len(x) == 0 or len(y) == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=10)
        return
    ax.scatter(x, y, color=color, alpha=0.8)
    mn, mx = min(min(x), min(y)), max(max(x), max(y))
    ax.plot([mn, mx], [mn, mx], linestyle='--', color='grey', alpha=0.7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle=':', linewidth=0.7)
    
    # Add R² annotation
    r2 = np.corrcoef(x, y)[0, 1] ** 2 if len(x) > 1 else np.nan
    ax.text(0.05, 0.92, f"$R^2$ = {r2:.3f}", transform=ax.transAxes,
            fontsize=10, color="black", bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))

# Top row
plot_scatter(axes[0, 0], xs_contrib, ys_contrib,
             "Macro contributors", "Micro contributors", "tab:blue")

plot_scatter(axes[0, 1], xs_ret, ys_ret,
             "Macro retirees", "Micro retirees", "tab:orange")

# Bottom row
plot_scatter(axes[1, 0], xs_contr_amt, ys_contr_amt,
             "Macro contributions (bn GHS)", "Micro contributions (bn GHS)", "tab:green")

plot_scatter(axes[1, 1], xs_ben_amt, ys_ben_amt,
             "Macro benefits (bn GHS)", "Micro benefits (bn GHS)", "tab:red")

plt.suptitle("Micro vs Macro — Contributors, Retirees, Contributions, and Benefits", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()
