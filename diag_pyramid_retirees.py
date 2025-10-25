# === diag_pyramid_retirees.py ===
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import load_config, resolve_path, load_sample_for_year, load_demog_for_year
import scienceplots

plt.style.use(['science', 'scatter'])
FIG_DIR = Path("./figures")
FIG_DIR.mkdir(exist_ok=True)

cfg = load_config("config.yaml")
root = cfg["project"]["root_dir"]
out_dir = resolve_path(root, cfg["project"]["output_dir"])
years = list(range(int(cfg["years"]["start"]), int(cfg["years"]["end"]) + 1))

cD = cfg["columns"]["demog"]
cS = cfg["columns"]["sample"]
age_order = cfg.get("age_groups", {}).get("order")
sex_order = cfg.get("sex_values", ["M", "F"])
AGE_LABEL_FONTSIZE = 7

def make_retiree_pyramid(year):
    micro_fp = out_dir / f"micro_{year}.csv"
    if not micro_fp.exists():
        print(f"[{year}] micro file not found, skipping.")
        return
    micro = pd.read_csv(micro_fp)
    sample = load_sample_for_year(cfg, year)
    micro["weight"] = pd.to_numeric(micro["weight"], errors="coerce").fillna(0)

    # restrict to ages 60+
    micro = micro[micro["age_min"] >= 60]
    sample = sample[sample[cS["age_group"]].apply(lambda x: str(x).startswith("60") or str(x).startswith("65") or "+" in str(x))]

    if age_order is None:
        age_order_local = sorted(sample[cS["age_group"]].astype(str).unique())
    else:
        age_order_local = [a for a in age_order if a.startswith("60") or "+" in a]

    micro_agg = (
        micro.groupby(["age_grp", "sex"])["weight"].sum()
        .unstack(fill_value=0).reindex(index=age_order_local)
    )
    sample_pop = (
        sample.groupby([cS["age_group"], cS["sex"]])[cS["group_total"]]
        .sum().unstack(fill_value=0).reindex(index=age_order_local)
    )

    for df in (micro_agg, sample_pop):
        for s in sex_order:
            if s not in df.columns:
                df[s] = 0.0

    micro_pct = micro_agg.div(micro_agg.sum().sum()) * 100
    sample_pct = sample_pop.div(sample_pop.sum().sum()) * 100

    y = np.arange(len(age_order_local))
    fig, axes = plt.subplots(1, 2, figsize=(5, 3), sharey=True)
    color_m, color_f = "#1f77b4", "#ff7f0e"

    axes[0].barh(y, micro_pct.get("M", 0), color=color_m, alpha=0.8)
    axes[0].barh(y, micro_pct.get("F", 0), color=color_f, alpha=0.8)
    axes[1].barh(y, sample_pct.get("M", 0), color=color_m, alpha=0.8)
    axes[1].barh(y, sample_pct.get("F", 0), color=color_f, alpha=0.8)

    def add_labels(ax, m, f):
        for i, (lv, rv) in enumerate(zip(m, f)):
            if lv > 0:
                ax.text(lv + 0.15, i, f"{lv:.2f}%", va="center", ha="left", fontsize=6)
            if rv > 0:
                ax.text(rv + 0.15, i, f"{rv:.2f}%", va="center", ha="left", fontsize=6)

    add_labels(axes[0], micro_pct["M"], micro_pct["F"])
    add_labels(axes[1], sample_pct["M"], sample_pct["F"])

    for ax in axes:
        ax.set_yticks(y)
        ax.set_yticklabels(age_order_local, fontsize=AGE_LABEL_FONTSIZE)
        ax.invert_yaxis()
        ax.tick_params(direction="in", length=5, width=0.8, top=True, right=True)
        ax.set_xlim(0, max(micro_pct.max().max(), sample_pct.max().max()) * 1.3)
        ax.set_xlabel("% of population")

    axes[1].yaxis.tick_right()
    axes[1].yaxis.set_label_position("right")

    for ax, label in zip(axes, ["Micro-structure", "Macro-structure"]):
        ax.text(0.5, -0.08, label, transform=ax.transAxes,
                fontsize=9, ha="center", va="top", color="black")

    for ax in axes:
        ax.text(ax.get_xlim()[1]*0.5, -1, "Male", color=color_m, fontsize=8, ha="center")
        ax.text(ax.get_xlim()[1]*0.9, -1, "Female", color=color_f, fontsize=8, ha="center")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    outp = FIG_DIR / f"pyramid_retir_{year}.pdf"
    fig.savefig(outp, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[{year}] retirees saved → {outp}")

for y in years:
    make_retiree_pyramid(y)
print("All retiree pyramids saved to ./figures.")