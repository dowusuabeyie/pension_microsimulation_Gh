# === micro_vs_macro_science_split.py ===
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from utils import load_config, resolve_path, load_macro
import scienceplots

# Apply SciencePlots style
plt.style.use(['science', 'scatter'])

# ==============================
# Directory setup
# ==============================
# Create 'figures' folder if it does not exist
if not os.path.exists("./figures"):
    os.makedirs("figures")

# ==============================
# Load configuration and data
# ==============================
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

# ==============================
# Load each year's microdata
# ==============================
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

# ==============================
# Define plotting helper
# ==============================
def plot_scatter(x, y, xlabel, ylabel, color, fname):
    """Draws and saves a SciencePlots-styled scatter plot."""
    fig, ax = plt.subplots(figsize=(5, 5))

    if len(x) == 0 or len(y) == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=10)
    else:
        ax.scatter(x, y, color=color, s=35, alpha=0.9, edgecolor='none')
        mn, mx = min(min(x), min(y)), max(max(x), max(y))
        ax.plot([mn, mx], [mn, mx], linestyle='--', color='grey', alpha=0.7)

        # Axis labels
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        # Inward ticks on all sides
        ax.tick_params(direction='in', length=4, width=0.8, top=True, right=True)
        ax.set_box_aspect(1)

        # Light grid
        #ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)

        # Add R² annotation
        if len(x) > 1:
            r2 = np.corrcoef(x, y)[0, 1] ** 2
            ax.text(
                0.05, 0.92, f"$R^2$ = {r2:.3f}",
                transform=ax.transAxes,
                fontsize=9,
                color="black",
                bbox=dict(facecolor="white", alpha=0.65, edgecolor="none")
            )

    # Title
    # ax.set_title(f"{ylabel} vs {xlabel}", fontsize=11, weight='bold')
    # plt.tight_layout()

    # Save high-quality PDF
    fig.savefig(f"figures/{fname}.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)

# ==============================
# Generate & save four plots
# ==============================
plot_scatter(xs_contrib, ys_contrib,
             "No. of contributors (macrodata)", "No. of contributors (microdata)", "tab:blue",
             "targetline-contr")

plot_scatter(xs_ret, ys_ret,
             "No. of retirees (macrodata)", "No. of retirees (microdata)", "tab:orange",
             "targetline-retir")

plot_scatter(xs_contr_amt, ys_contr_amt,
             "Contributions (bn GHS, macrodata)", "Contributions (bn GHS, microdata)", "tab:green",
             "targetline-contr-amt")

plot_scatter(xs_ben_amt, ys_ben_amt,
             "Benefits (bn GHS, macrodata)", "Benefits (bn GHS, microdata)", "tab:red",
             "targetline-ben-amt")

print("All SciencePlots saved to ./figures as PDF.")
