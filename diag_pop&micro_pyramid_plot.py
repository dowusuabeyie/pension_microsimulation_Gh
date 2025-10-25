# diag_pop&micro_pyramid_plot.py
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import load_config, resolve_path, load_sample_for_year, load_demog_for_year


try:
    import scienceplots
    plt.style.use(['science', 'scatter'])
except Exception:
    
    pass

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
    # load micro and sample for this year
    micro_fp = out_dir / f"micro_{year}.csv"
    if not micro_fp.exists():
        print(f"[{year}] micro file not found, skipping.")
        return

    micro = pd.read_csv(micro_fp)
    sample = load_sample_for_year(cfg, year)

    # ensure numeric weights
    micro["weight"] = pd.to_numeric(micro["weight"], errors="coerce").fillna(0)

    # Ensure categories / ordering
    if age_order is None:
        # fallback: derive age order from sample (or demog)
        if cS["age_group"] in sample.columns:
            age_order_local = sorted(sample[cS["age_group"]].astype(str).unique())
        else:
            demog = load_demog_for_year(cfg, year)
            age_order_local = sorted(demog[cD["age_group"]].astype(str).unique())
    else:
        age_order_local = age_order

    # Aggregate micro: weighted counts by age_grp × sex
    micro_agg = (
        micro
        .groupby([cS["age_group"] if cS["age_group"] in micro.columns else cD["age_group"], "sex"], dropna=False)["weight"]
        .sum()
        .unstack(fill_value=0)
        .reindex(index=age_order_local)
    )

    # Aggregate macro (sample): use N_g_h as population totals
    # sample uses columns cS["age_group"], cS["sex"], cS["group_total"]
    sample_pop = (
        sample
        .groupby([cS["age_group"], cS["sex"]], dropna=False)[cS["group_total"]]
        .sum()
        .unstack(fill_value=0)
        .reindex(index=age_order_local)
    )

    # If some sex columns missing, ensure both M/F exist
    for df in (micro_agg, sample_pop):
        for s in sex_order:
            if s not in df.columns:
                df[s] = 0.0
        # reorder columns
        df = df[sex_order]

    # convert to DataFrames with the right column order
    micro_agg = micro_agg[sex_order].fillna(0)
    sample_pop = sample_pop[sex_order].fillna(0)

    # normalize to percentages (so pyramids compare shape not absolute size)
    micro_pct = micro_agg.div(micro_agg.sum().sum()).fillna(0) * 100
    sample_pct = sample_pop.div(sample_pop.sum().sum()).fillna(0) * 100

    # y positions
    y = np.arange(len(age_order_local))

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(10, 8), sharey=True)

    # left: micro
    axes[0].barh(y, -micro_pct.get(sex_order[0], pd.Series(0, index=age_order_local)), align='center')
    axes[0].barh(y,  micro_pct.get(sex_order[1], pd.Series(0, index=age_order_local)), align='center')
    axes[0].set_title(f"Micro % by age-sex\n({year})", fontsize=11)

    # right: sample (macro-pop from sample N_g_h)
    axes[1].barh(y, -sample_pct.get(sex_order[0], pd.Series(0, index=age_order_local)), align='center')
    axes[1].barh(y,  sample_pct.get(sex_order[1], pd.Series(0, index=age_order_local)), align='center')
    axes[1].set_title(f"Sample/pop % by age-sex\n({year})", fontsize=11)

    # y-ticks
    for ax in axes:
        ax.set_yticks(y)
        ax.set_yticklabels(age_order_local)
        ax.invert_yaxis()
        ax.tick_params(direction='in', length=5, width=0.8, top=True, right=True)
        ax.set_box_aspect(1)

    # add a small shared x-labels
    axes[0].set_xlabel(f"% population (Micro)")
    axes[1].set_xlabel(f"% population (Sample N_g_h)")

    plt.suptitle(f"Population pyramid comparison — {year}", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    outp = FIG_DIR / f"pyramid_{year}.pdf"
    fig.savefig(outp, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[{year}] saved → {outp}")

# Run for all years
for y in years:
    make_pyramid_for_year(y)

print("All pyramids saved to ./figures.")
