import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from utils import load_config, resolve_path, load_macro
import scienceplots
plt.style.use(['science']) #science or ieee

# Load configuration and data paths
cfg = load_config("config.yaml")
root = cfg["project"]["root_dir"]
out_dir = resolve_path(root, cfg["project"]["output_dir"])
years = list(range(int(cfg["years"]["start"]), int(cfg["years"]["end"]) + 1))

# Load macro data and multipliers
macro = load_macro(cfg).set_index(cfg["columns"]["macro"]["year"])
mult_contr = cfg.get("macro_units", {}).get("contributions_total_multiplier", 1.0)
mult_ben   = cfg.get("macro_units", {}).get("benefits_total_multiplier", 1.0)

# Containers for mean & diff values
means_contrib, diffs_contrib = [], []
means_ret, diffs_ret = [], []
means_contr_amt, diffs_contr_amt = [], []
means_ben_amt, diffs_ben_amt = [], []

# ---- Collect yearly data ----
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

    # --- Calculate means and diffs ---
    def add_stats(micro, macro, mean_list, diff_list, scale=1.0):
        mean_list.append(0.5 * (micro + macro) / scale)
        diff_list.append((micro - macro) / scale)

    add_stats(micro_contributors, macro_contributors, means_contrib, diffs_contrib)
    add_stats(micro_retirees, macro_retirees, means_ret, diffs_ret)
    add_stats(micro_contr_amt, macro_contr_amt, means_contr_amt, diffs_contr_amt, scale=1e9)
    add_stats(micro_ben_amt, macro_ben_amt, means_ben_amt, diffs_ben_amt, scale=1e9)

# ---- Plotting function ----
def plot_bland_altman(ax, means, diffs, xlabel, ylabel, color):
    if len(means) == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=10)
        return

    means = np.array(means)
    diffs = np.array(diffs)
    md = np.mean(diffs)
    sd = np.std(diffs, ddof=1)
    upper = md + 2.58 * sd
    lower = md - 2.58 * sd

    ax.scatter(means, diffs, color=color, alpha=0.6)
    ax.axhline(upper, color='grey', linestyle='--')
    ax.axhline(lower, color='grey', linestyle='--')
    ax.axhline(0, color='black', linestyle=':', alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle=':', linewidth=0.7)

    #ax.text(0.05, 0.93,
    #        f"Bias={md:.2f}\n±2.58SD=({lower:.2f}, {upper:.2f})",
    #        transform=ax.transAxes, fontsize=9,
    #        bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))
    #ax.legend(loc="lower right", fontsize=8)

# ---- Draw 2×2 panel ----
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

plot_bland_altman(axes[0, 0], means_contrib, diffs_contrib,
                  "Mean of (micro, macro) contributors",
                  "Difference (micro - macro) contributors", "tab:blue")

plot_bland_altman(axes[0, 1], means_ret, diffs_ret,
                  "Mean of (micro, macro) retirees",
                  "Difference (micro - macro) retirees", "tab:orange")

plot_bland_altman(axes[1, 0], means_contr_amt, diffs_contr_amt,
                  "Mean of (micro, macro) contributions (bn GHS)",
                  "Difference (micro - macro) (bn GHS)", "tab:green")

plot_bland_altman(axes[1, 1], means_ben_amt, diffs_ben_amt,
                  "Mean of (micro, macro) benefits (bn GHS)",
                  "Difference (micro - macro) (bn GHS)", "tab:red")

plt.suptitle("Bland–Altman Plots — Micro vs Macro Comparisons", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()
