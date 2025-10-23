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

# Containers for both contribution and benefit comparisons
xs_contr, ys_contr = [], []
xs_ben, ys_ben = [], []

for y in years:
    fp = out_dir / f"micro_{y}.csv"
    if not fp.exists():
        continue

    m = pd.read_csv(fp)
    m["weight"]       = pd.to_numeric(m["weight"], errors="coerce").fillna(0)
    m["contribution"] = pd.to_numeric(m["contribution"], errors="coerce").fillna(0)
    m["pension"]      = pd.to_numeric(m["pension"], errors="coerce").fillna(0)

    # Compute micro totals
    micro_contr = (m["contribution"] * m["weight"]).sum()
    micro_ben   = (m["pension"] * m["weight"]).sum()

    # Corresponding macro totals
    macro_contr = float(macro.loc[y, cfg["columns"]["macro"]["contributions_total"]]) * mult_contr
    macro_ben   = float(macro.loc[y, cfg["columns"]["macro"]["benefits_total"]]) * mult_ben

    # Convert to billions
    xs_contr.append(macro_contr / 1e9)
    ys_contr.append(micro_contr / 1e9)
    xs_ben.append(macro_ben / 1e9)
    ys_ben.append(micro_ben / 1e9)

# ---- Plot both comparisons ----
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Contributions plot
ax = axes[0]
ax.scatter(xs_contr, ys_contr, color="tab:blue", label="Contributions")
mn = min(min(xs_contr), min(ys_contr))
mx = max(max(xs_contr), max(ys_contr))
ax.plot([mn, mx], [mn, mx], linestyle='--', color='grey')
ax.set_xlabel("Macro contributions (bn GHS)")
ax.set_ylabel("Micro contributions (bn GHS)")
ax.set_title("Micro vs Macro: Contributions")
ax.grid(True)

# Benefits plot
ax = axes[1]
ax.scatter(xs_ben, ys_ben, color="tab:green", label="Benefits")
mn = min(min(xs_ben), min(ys_ben))
mx = max(max(xs_ben), max(ys_ben))
ax.plot([mn, mx], [mn, mx], linestyle='--', color='grey')
ax.set_xlabel("Macro benefits (bn GHS)")
ax.set_ylabel("Micro benefits (bn GHS)")
ax.set_title("Micro vs Macro: Benefits")
ax.grid(True)

plt.suptitle("Observed (micro) vs Macro (target) — Contributions & Benefits", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
