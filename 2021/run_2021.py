# run_2021.py
# Load config + CSVs, align demog with provided sample (C,R,N),
# rescale C and R totals to match macro aggregates if needed,
# and write tidy strata + diagnostics.

from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
from utils import (
    load_config, resolve_path, load_macro, load_demog_for_year, load_sample_for_year,
    align_demog_and_sample, quick_totals, scale_macro_currency
)


def rescale_domain_totals(sample: pd.DataFrame, grp_col: str, N_col: str, sch: str, target: float) -> tuple[pd.DataFrame, float]:
    """
    If sum(N_g_h for sch_grp==sch) != target, scale N_g_h proportionally for that sch.
    Returns (updated_sample, scale_factor).
    """
    mask = sample[grp_col] == sch
    current = sample.loc[mask, N_col].sum()
    if pd.isna(current) or current == 0:
        return sample, np.nan
    sf = float(target) / float(current)
    if not np.isfinite(sf) or sf <= 0:
        return sample, np.nan
    sample.loc[mask, N_col] = sample.loc[mask, N_col] * sf
    return sample, sf


def main():
    cfg = load_config("config.yaml")

    # Decide year (single-year run for now)
    mode = cfg["years"]["mode"]
    years = list(range(int(cfg["years"]["start"]), int(cfg["years"]["end"]) + 1)) if mode == "range" else list(cfg["years"]["list"])
    if not years:
        raise ValueError("No years specified in config.yaml")
    year = years[0]

    # Paths
    root = cfg["project"]["root_dir"]
    out_dir = resolve_path(root, cfg["project"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load
    macro = load_macro(cfg)
    demog = load_demog_for_year(cfg, year)
    sample = load_sample_for_year(cfg, year)

    # Macro controls for this year (with billions -> absolute scaling)
    cM = cfg["columns"]["macro"]
    mrow = macro[macro[cM["year"]] == year]
    if mrow.empty:
        raise ValueError(f"Year {year} not found in macro.csv")
    macro_targets = scale_macro_currency(cfg, mrow.iloc[0])

    # Optional proportional rescale: make sum N_g_h for C and R match macro stocks exactly
    cS = cfg["columns"]["sample"]
    sample, sf_C = rescale_domain_totals(sample, cS["scheme_group"], cS["group_total"], "C", macro_targets["contributors_total"])
    sample, sf_R = rescale_domain_totals(sample, cS["scheme_group"], cS["group_total"], "R", macro_targets["retirees_total"])

    # Align demog + sample (now contains C, R, N)
    strata = align_demog_and_sample(cfg, demog, sample)

    # Write strata
    strata_out = out_dir / f"strata_{year}.csv"
    strata.to_csv(strata_out, index=False, encoding="utf-8-sig")

    # Diagnostics
    diag = quick_totals(cfg, demog, sample)
    for k, v in macro_targets.items():
        diag[k] = v
    diag["scale_factor_C"] = sf_C
    diag["scale_factor_R"] = sf_R

    diag_out = out_dir / f"diagnostics_{year}.csv"
    diag.to_csv(diag_out, index=False, encoding="utf-8-sig")

    print("Done.")
    print(f"→ {strata_out}")
    print(f"→ {diag_out}")
    print("Notes: If scale_factor_C/R differ noticeably from 1.0, your N_g_h were adjusted to hit macro totals exactly.")

if __name__ == "__main__":
    main()
