# === bland_altman_science_split.py ===
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from utils import load_config, resolve_path, load_macro
import scienceplots

plt.style.use(['science', 'scatter'])

# ==============================
# Directory setup
# ==============================
if not os.path.exists("./figures"):
    os.makedirs("figures")

# ==============================
# Load configuration and data
# ==============================
cfg = load_config("config.yaml")
root = cfg["project"]["root_dir"]
out_dir = resolve_path(root, cfg["project"]["output_dir"])
years = list(range(int(cfg["years"]["start"]), int(cfg["years"]["end"]) + 1))

# Load macro data and multipliers
macro = load_macro(cfg).set_index(cfg["columns"]["macro"]["year"])
mult_contr = cfg.get("macro_units", {}).get("contributions_total_multiplier", 1.0)
mult_ben   = cfg.get("macro_units", {}).get("benefits_total_multiplier", 1.0)

# ==============================
# Containers
# ==============================
xs_contrib, ys_contrib = [], []
xs_ret, ys_ret = [], []
xs_contr_amt, ys_contr_amt = [], []
xs_ben_amt, ys_ben_amt = [], []

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

    micro_contributors = ((m["sch_grp"] == "C") * m["weight"]).sum()
    micro_retirees     = ((m["sch_grp"] == "R") * m["weight"]).sum()
    micro_contr_amt    = (m["contribution"] * m["weight"]).sum()
    micro_ben_amt      = (m["pension"] * m["weight"]).sum()

    macro_contributors = float(macro.loc[y, cfg["columns"]["macro"]["contributors"]])
    macro_retirees     = float(macro.loc[y, cfg["columns"]["macro"]["retirees"]])
    macro_contr_amt    = float(macro.loc[y, cfg["columns"]["macro"]["contributions_total"]]) * mult_contr
    macro_ben_amt      = float(macro.loc[y, cfg["columns"]["macro"]["benefits_total"]]) * mult_ben

    xs_contrib.append(macro_contributors)
    ys_contrib.append(micro_contributors)
    xs_ret.append(macro_retirees)
    ys_ret.append(micro_retirees)
    xs_contr_amt.append(macro_contr_amt)
    ys_contr_amt.append(micro_contr_amt)
    xs_ben_amt.append(macro_ben_amt)
    ys_ben_amt.append(micro_ben_amt)

# ==============================
# Bland–Altman helper
# ==============================
def bland_altman_plot(x, y, xlabel, ylabel, color, fname, xlim=None, ylim=None, scientific=False):
    """Draw Bland–Altman plot (mean vs. difference)."""
    fig, ax = plt.subplots(figsize=(3, 3))
    if len(x) == 0 or len(y) == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
    else:
        mean = np.mean([x, y], axis=0)
        diff = np.array(y) - np.array(x)
        md = np.mean(diff)
        sd = np.std(diff)

        ax.scatter(mean, diff, color=color, s=35, alpha=0.9, edgecolor='none', marker='2')
        ax.axhline(md, color='grey', linestyle='--', lw=1)
        ax.axhline(md + 2.58*sd, color='grey', linestyle=':', lw=1)
        ax.axhline(md - 2.58*sd, color='grey', linestyle=':', lw=1)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        # === Scientific notation option (for retirees) ===
        if scientific:
            ax.ticklabel_format(style='sci', axis='x', scilimits=(5, 5))

        ax.tick_params(direction='in', top=True, right=True)
        ax.set_box_aspect(1)

    fig.savefig(f"figures/{fname}.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)

# ==============================
# Generate and save
# ==============================
bland_altman_plot(xs_contrib, ys_contrib,
                  "$(\\text{Contr}_{\\text{micro}}+\\text{Contr}_{\\text{macro}})/2$", "$\\text{d}=(\\text{Contr}_{\\text{micro}}−\\text{Contr}_{\\text{macro}})$",
                  "tab:blue", "bland-altman-contr")

bland_altman_plot(xs_ret, ys_ret,
                  "$(\\text{Retir}_{\\text{micro}}+\\text{Retir}_{\\text{macro}})/2$", "$\\text{d}=(\\text{Retir}_{\\text{micro}}−\\text{Retir}_{\\text{macro}})$",
                  "tab:orange", "bland-altman-retir",
                  scientific=True) 
bland_altman_plot(xs_contr_amt, ys_contr_amt,
                  "$(\\text{CAmt}_{\\text{micro}}+\\text{CAmt}_{\\text{macro}})/2$", "$\\text{d}=(\\text{CAmt}_{\\text{micro}}−\\text{CAmt}_{\\text{macro}})$",
                  "tab:green", "bland-altman-contr-amt")

bland_altman_plot(xs_ben_amt, ys_ben_amt,
                  "$(\\text{Ben}_{\\text{micro}}+\\text{Ben}_{\\text{macro}})/2$", "$\\text{d}=(\\text{Ben}_{\\text{micro}}−\\text{Ben}_{\\text{macro}})$",
                  "tab:red", "bland-altman-ben-amt")

print("All Bland–Altman plots saved to ./figures as PDF.")
