import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from utils import load_config, resolve_path, load_macro

cfg = load_config("config.yaml")
root = cfg["project"]["root_dir"]
out_dir = resolve_path(root, cfg["project"]["output_dir"])
years = list(range(int(cfg["years"]["start"]), int(cfg["years"]["end"]) + 1))
macro = load_macro(cfg).set_index(cfg["columns"]["macro"]["year"])

mult_contr = cfg.get("macro_units", {}).get("contributions_total_multiplier", 1.0)

xs, ys = [], []
for y in years:
    fp = out_dir / f"micro_{y}.csv"
    if not fp.exists():
        continue
    m = pd.read_csv(fp)
    m["weight"]       = pd.to_numeric(m["weight"], errors="coerce").fillna(0)
    m["contribution"] = pd.to_numeric(m["contribution"], errors="coerce").fillna(0)
    micro_total = (m["contribution"] * m["weight"]).sum()
    macro_total = float(macro.loc[y, cfg["columns"]["macro"]["contributions_total"]]) * mult_contr
    xs.append(macro_total / 1e9)
    ys.append(micro_total / 1e9)

fig, ax = plt.subplots(figsize=(6,6))
ax.scatter(xs, ys)
mn = min(min(xs), min(ys))
mx = max(max(xs), max(ys))
ax.plot([mn, mx], [mn, mx], linestyle='--', color='grey')
ax.set_xlabel("Macro contributions (bn GHS)")
ax.set_ylabel("Micro contributions (bn GHS)")
ax.set_title("Observed (micro) vs Macro (target)")
ax.grid(True)
plt.tight_layout()
plt.show()
