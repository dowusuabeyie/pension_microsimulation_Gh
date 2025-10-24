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
    xs_contr_amt.append(macro_contr_amt)
    ys_contr_amt.append(micro_contr_amt)
    xs_ben_amt.append(macro_ben_amt)
    ys_ben_amt.append(micro_ben_amt)

# ==============================
# Define plotting helper
# ==============================
def plot_scatter(x, y, xlabel, ylabel, color, fname, scientific=False, xlim=None, ylim=None):
    """Draws and saves a SciencePlots-styled scatter plot."""
    fig, ax = plt.subplots(figsize=(3, 3))

    if len(x) == 0 or len(y) == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=10)
    else:
        ax.scatter(x, y, color=color, s=35, alpha=0.9, edgecolor='none')
        mn, mx = min(min(x), min(y)), max(max(x), max(y))
        ax.plot([mn, mx], [mn, mx], linestyle='--', color='grey', alpha=0.7)

        # Axis labels
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        # Custom limits if provided
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

        # Scientific notation option (for retirees)
        if scientific:
            ax.ticklabel_format(style='sci', axis='both', scilimits=(5, 5))

        # Inward ticks on all sides
        ax.tick_params(direction='in', length=5, width=0.8, top=True, right=True)
        ax.set_box_aspect(1)

        # Add R-squared, RMSE, and MAPE annotations
        if len(x) > 1:
            r2 = np.corrcoef(x, y)[0, 1] ** 2
            rmse = np.sqrt(np.mean((np.array(y) - np.array(x)) ** 2))
            mape = np.mean(np.abs((np.array(y) - np.array(x)) / np.array(x))) * 100

            def sci_not(val):
                """Format number in scientific notation like ×10^{power}."""
                if val == 0:
                    return "0"
                exp = int(np.floor(np.log10(abs(val))))
                base = val / (10 ** exp)
                return fr"{base:.2f}\times10^{{{exp}}}"

            text_eq = (
                fr"$\mathrm{{RMSE}} = {sci_not(rmse)}$" "\n"
                fr"$\mathrm{{MAPE}} = {sci_not(mape)}$" "\n"
                fr"$R^2 = {r2:.2f}$" 
            )

            ax.text(
                0.05, 0.8,
                text_eq,
                transform=ax.transAxes,
                fontsize=9,
                color="black",
                bbox=dict(facecolor="white", alpha=0.65, edgecolor="none")
            )


    # Save high-quality PDF
    fig.savefig(f"figures/{fname}.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)

# ==============================
# Generate & save four plots
# ==============================
plot_scatter(xs_contrib, ys_contrib,
             "Contributors (macrodata)", "Contributors (microdata)",
             "tab:blue", "targetline-contr",
             xlim=(1200000, 2100000), ylim=(1200000, 2100000))

plot_scatter(xs_ret, ys_ret,
             "Retirees (macrodata)", "Retirees (microdata)",
             "tab:orange", "targetline-retir",
             scientific=True,
             xlim=(150000, 260000), ylim=(150000, 260000))

plot_scatter(xs_contr_amt, ys_contr_amt,
             "Contribution (macrodata)", "Contribution (microdata)",
             "tab:green", "targetline-contr-amt",
             xlim=(1000000000, 9000000000), ylim=(1000000000, 9000000000))

plot_scatter(xs_ben_amt, ys_ben_amt,
             "Benefit (macrodata)", "Benefit (microdata)",
             "tab:red", "targetline-ben-amt",
             xlim=(1000000000, 7000000000), ylim=(1000000000, 7000000000))

print("All SciencePlots saved to ./figures as PDF.")

