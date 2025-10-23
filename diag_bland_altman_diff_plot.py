import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from utils import load_config, resolve_path, load_macro

cfg = load_config("config.yaml")
out_dir = resolve_path(cfg["project"]["root_dir"], cfg["project"]["output_dir"])
years = list(range(int(cfg["years"]["start"]), int(cfg["years"]["end"]) + 1))
macro = load_macro(cfg).set_index(cfg["columns"]["macro"]["year"])

mult_contr = cfg.get("macro_units", {}).get("contributions_total_multiplier", 1.0)

means, diffs = [], []

for y in years:
    fp = out_dir / f"micro_{y}.csv"
    if not fp.exists():
        continue
    m = pd.read_csv(fp)
    m["weight"]       = pd.to_numeric(m["weight"], errors="coerce").fillna(0)
    m["contribution"] = pd.to_numeric(m["contribution"], errors="coerce").fillna(0)
    micro_total = (m["contribution"] * m["weight"]).sum()

    macro_total = float(macro.loc[y, cfg["columns"]["macro"]["contributions_total"]]) * mult_contr
    mean = 0.5 * (micro_total + macro_total)
    diff = micro_total - macro_total
    means.append(mean)
    diffs.append(diff)

means = np.array(means)
diffs = np.array(diffs)
md = np.mean(diffs)
sd = np.std(diffs, ddof=1)
upper = md + 1.96 * sd
lower = md - 1.96 * sd

fig, ax = plt.subplots(figsize=(7,5))
ax.scatter(means / 1e9, diffs / 1e9)   # scale to billions for readability
ax.axhline(md / 1e9, linestyle='-')
ax.axhline(upper / 1e9, linestyle='--')
ax.axhline(lower / 1e9, linestyle='--')
ax.set_xlabel("Mean of (micro, macro) contributors (bn GHS)")
ax.set_ylabel("Difference (micro - macro) (bn GHS)")
ax.set_title("Bland-Altman: micro vs macro contributions")
ax.grid(True)
plt.tight_layout()
plt.show()
