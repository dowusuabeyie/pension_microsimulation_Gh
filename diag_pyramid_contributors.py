import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import load_config, resolve_path, load_sample_for_year
import scienceplots

plt.style.use(['science', 'scatter'])

FIG_DIR = Path("./figures")
FIG_DIR.mkdir(exist_ok=True)

cfg = load_config("config.yaml")
root = cfg["project"]["root_dir"]
out_dir = resolve_path(root, cfg["project"]["output_dir"])
years = list(range(int(cfg["years"]["start"]), int(cfg["years"]["end"]) + 1))
cS = cfg["columns"]["sample"]
sex_order = cfg.get("sex_values", ["M", "F"])
AGE_LABEL_FONTSIZE = 7

def make_contributor_pyramid(year):
    micro_fp = out_dir / f"micro_{year}.csv"
    if not micro_fp.exists():
        print(f"[{year}] micro file not found, skipping.")
        return

    micro = pd.read_csv(micro_fp)
    sample = load_sample_for_year(cfg, year)
    micro["weight"] = pd.to_numeric(micro["weight"], errors="coerce").fillna(0)

    # --- filter: contributors (<60)
    if "age_min" in micro.columns:
        micro = micro[micro["age_min"] < 60]
    if "age_min" in sample.columns:
        sample = sample[sample["age_min"] < 60]
    else:
        # fallback if not numeric, check string prefix
        sample = sample[sample[cS["age_group"]].apply(lambda x: not str(x).startswith("60"))]

    # determine age order
    age_order_local = sorted(sample[cS["age_group"]].astype(str).unique())

    # --- aggregate by age/sex
    micro_agg = micro.groupby(["age_grp", "sex"])["weight"].sum().unstack(fill_value=0).reindex(index=age_order_local)
    sample_agg = sample.groupby([cS["age_group"], cS["sex"]])[cS["group_total"]].sum().unstack(fill_value=0).reindex(index=age_order_local)

    for df in (micro_agg, sample_agg):
        for s in sex_order:
            if s not in df.columns:
                df[s] = 0.0

    micro_pct = micro_agg.div(micro_agg.sum().sum()) * 100
    sample_pct = sample_agg.div(sample_agg.sum().sum()) * 100

    y = np.arange(len(age_order_local))
    fig, axes = plt.subplots(1, 2, figsize=(6, 3), sharey=True)
    color_m, color_f = "#4582b4", "#ee798a"

    # left: Micro
    axes[0].barh(y, -micro_pct["M"], color=color_m, alpha=0.85)
    axes[0].barh(y,  micro_pct["F"], color=color_f, alpha=0.85)
    # right: Macro
    axes[1].barh(y, -sample_pct["M"], color=color_m, alpha=0.85)
    axes[1].barh(y,  sample_pct["F"], color=color_f, alpha=0.85)

    # add labels
    def add_labels(ax, m, f):
        for i, (lv, rv) in enumerate(zip(m, f)):
            if lv > 0:
                ax.text(-lv - 0.15, i, f"{lv:.2f}%", va="center", ha="right", fontsize=6)
            if rv > 0:
                ax.text(rv + 0.15, i, f"{rv:.2f}%", va="center", ha="left", fontsize=6)

    add_labels(axes[0], micro_pct["M"], micro_pct["F"])
    add_labels(axes[1], sample_pct["M"], sample_pct["F"])

    for ax in axes:
        ax.set_yticks(y)
        ax.set_yticklabels(age_order_local, fontsize=AGE_LABEL_FONTSIZE)
        ax.invert_yaxis()
        ax.tick_params(length=0)
        ax.set_xlim(-max(micro_pct.max().max(), sample_pct.max().max()) * 1.2,
                     max(micro_pct.max().max(), sample_pct.max().max()) * 1.2)
        ax.set_xlabel("% of population")
        ticks = ax.get_xticks()
        ax.set_xticklabels([f"{abs(t):.0f}%" for t in ticks])

    # duplicate age labels on right
    axes[1].yaxis.tick_right()
    axes[1].set_yticklabels(age_order_local, fontsize=AGE_LABEL_FONTSIZE)
    axes[1].yaxis.set_label_position("right")

    # sex labels
    for ax in axes:
        ax.text(-ax.get_xlim()[1]*0.8, -1, "Male", color=color_m, fontsize=8, ha="center")
        ax.text(ax.get_xlim()[1]*0.8, -1, "Female", color=color_f, fontsize=8, ha="center")

    # titles under plots
    for ax, label in zip(axes, ["Micro-structure", "Macro-structure"]):
        ax.text(0.5, -0.08, label, transform=ax.transAxes, fontsize=9, ha="center", va="top")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    outp = FIG_DIR / f"pyramid_contr_{year}.pdf"
    fig.savefig(outp, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[{year}] Contributors pyramid saved → {outp}")

for y in years:
    make_contributor_pyramid(y)
