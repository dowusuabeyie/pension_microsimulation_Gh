import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from utils import load_config, resolve_path, load_demog_for_year

cfg = load_config("config.yaml")
out_dir = resolve_path(cfg["project"]["root_dir"], cfg["project"]["output_dir"])
year = int(cfg["years"]["start"])
micro = pd.read_csv(out_dir / f"micro_{year}.csv")
demog = load_demog_for_year(cfg, year)

age_order = cfg.get("age_groups", {}).get("order")
cD = cfg["columns"]["demog"]

# ensure age_order exists
if age_order is None:
    age_order = sorted(demog[cD["age_group"]].unique())

micro["weight"] = pd.to_numeric(micro["weight"], errors="coerce").fillna(0)

micro_agg = (micro
             .groupby([cD["age_group"], "sex"])["weight"]
             .sum()
             .unstack(fill_value=0)
             .reindex(index=age_order))

demog_agg = (demog
             .groupby([cD["age_group"], cD["sex"]])[cD["population"]]
             .sum()
             .unstack(fill_value=0)
             .reindex(index=age_order))

micro_pct = micro_agg.div(micro_agg.sum().sum()) * 100
demog_pct = demog_agg.div(demog_agg.sum().sum()) * 100

y = np.arange(len(age_order))
fig, axes = plt.subplots(1,2, figsize=(10,8), sharey=True)

axes[0].barh(y, -micro_pct.get("M", pd.Series(0, index=age_order)), align='center')
axes[0].barh(y,  micro_pct.get("F", pd.Series(0, index=age_order)), align='center')
axes[0].set_title("Micro % by age-sex")

axes[1].barh(y, -demog_pct.get("M", pd.Series(0, index=age_order)), align='center')
axes[1].barh(y,  demog_pct.get("F", pd.Series(0, index=age_order)), align='center')
axes[1].set_title("Demographic % by age-sex")

for ax in axes:
    ax.set_yticks(y)
    ax.set_yticklabels(age_order)
    ax.invert_yaxis()
    
plt.suptitle(f"Population pyramid comparison — {year}")
plt.tight_layout()
plt.show()
