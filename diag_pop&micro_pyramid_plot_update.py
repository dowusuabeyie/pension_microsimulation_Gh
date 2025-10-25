# === diag_pop&micro_pyramid_plot_updated.py ===
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import load_config, resolve_path, load_sample_for_year, load_demog_for_year
import scienceplots

plt.style.use(['science', 'scatter'])

# output folder
FIG_DIR = Path("./figures")
FIG_DIR.mkdir(exist_ok=True)

cfg = load_config("config.yaml")
root = cfg["project"]["root_dir"]
out_dir = resolve_path(root, cfg["project"]["output_dir"])
years = list(range(int(cfg["years"]["start"]), int(cfg["years"]["end"]) + 1))

# config column names
cD = cfg["columns"]["demog"]
cS = cfg["columns"]["sample"]

age_order = cfg.get("age_groups", {}).get("order")
sex_order = cfg.get("sex_values", ["M", "F"])

def make_pyramid_for_year(year):
    micro_fp = out_dir / f"micro_{year}.csv"
    if not micro_fp.exists():
        print(f"[{year}] micro file not found, skipping.")
        return

    micro = pd.read_csv(micro_fp)
    sample = load_sample_for_year(cfg, year)

    micro["weight"] = pd.to_numeric(micro["weight"], errors="coerce").fillna(0)

    if age_order is None:
        if cS["age_group"] in sample.columns:
            age_order_local = sorted(sample[cS["age_group"]].astype(str).unique())
        else:
            demog = load_demog_for_year(cfg, year)
            age_order_local = sorted(demog[cD["age_group"]].astype(str).unique())
    else:
        age_order_local = age_order

    # --- aggregate micro and sample
    micro_agg = (
        micro.groupby([cS["age_group"] if cS["age_group"] in micro.columns else cD["age_group"], "sex"], dropna=False)["weight"]
        .sum().unstack(fill_value=0).reindex(index=age_order_local)
    )
    sample_pop = (
        sample.groupby([cS["age_group"], cS["sex"]], dropna=False)[cS["group_total"]]
        .sum().unstack(fill_value=0).reindex(index=age_order_local)
    )

    # ensure both sexes present
    for df in (micro_agg, sample_pop):
        for s in sex_order:
            if s not in df.columns:
                df[s] = 0.0

    micro_agg = micro_agg[sex_order].fillna(0)
    sample_pop = sample_pop[sex_order].fillna(0)

    # normalize to percentages
    micro_pct = micro_agg.div(micro_agg.sum().sum()) * 100
    sample_pct = sample_pop.div(sample_pop.sum().sum()) * 100

    y = np.arange(len(age_order_local))
    fig, axes = plt.subplots(1, 2, figsize=(5, 3), sharey=True)

    # --- colors
    color_m, color_f = "#644f88", "#bbaed1"

    # left panel: Micro
    axes[0].barh(y, -micro_pct.get(sex_order[0], 0), color=color_m, alpha=0.8, label="Male")
    axes[0].barh(y,  micro_pct.get(sex_order[1], 0), color=color_f, alpha=0.8, label="Female")

    # right panel: Macro
    axes[1].barh(y, -sample_pct.get(sex_order[0], 0), color=color_m, alpha=0.8)
    axes[1].barh(y,  sample_pct.get(sex_order[1], 0), color=color_f, alpha=0.8)

    # --- Add percentage labels
    def add_labels(ax, data_left, data_right):
        for i, (lv, rv) in enumerate(zip(data_left, data_right)):
            if lv != 0:
                ax.text(-lv - 0.15, i, f"{lv:.2f}%", va="center", ha="right", fontsize=6)
            if rv != 0:
                ax.text(rv + 0.15, i, f"{rv:.2f}%", va="center", ha="left", fontsize=6)

    add_labels(axes[0], micro_pct[sex_order[0]], micro_pct[sex_order[1]])
    add_labels(axes[1], sample_pct[sex_order[0]], sample_pct[sex_order[1]])

    # --- Axis and labels
    for ax in axes:
        ax.set_yticks(y)
        ax.set_yticklabels(age_order_local)
        ax.invert_yaxis()
        ax.tick_params(direction="in", length=5, width=0.8, top=True, right=True)
        ax.set_xlim(-max(micro_pct.max().max(), sample_pct.max().max()) * 1.2,
                     max(micro_pct.max().max(), sample_pct.max().max()) * 1.2)
        ax.set_xlabel("% of population")

    # Duplicate age labels on both sides
    axes[1].yaxis.tick_right()
    axes[1].yaxis.set_label_position("right")

    # --- Place sex labels on chart
    axes[0].text(-axes[0].get_xlim()[1] * 0.8, -1, "Male", color=color_m, fontsize=8, ha="center")
    axes[0].text( axes[0].get_xlim()[1] * 0.8, -1, "Female", color=color_f, fontsize=8, ha="center")

    axes[1].text(-axes[1].get_xlim()[1] * 0.8, -1, "Male", color=color_m, fontsize=8, ha="center")
    axes[1].text( axes[1].get_xlim()[1] * 0.8, -1, "Female", color=color_f, fontsize=8, ha="center")

    # --- Subplot titles BELOW each chart
    for ax, label in zip(axes, ["Micro-structure", "Macro-structure"]):
        ax.text(0.5, -0.08, label, transform=ax.transAxes,
                fontsize=9, ha="center", va="top", color="black")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    # plt.suptitle(f"Population pyramid comparison — {year}", fontsize=13)

    outp = FIG_DIR / f"pyramid_{year}.pdf"
    fig.savefig(outp, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[{year}] saved → {outp}")

# Run for all years
for y in years:
    make_pyramid_for_year(y)

print("✅ All pyramids saved to ./figures.")
