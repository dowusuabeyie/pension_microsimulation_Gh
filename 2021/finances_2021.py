# finances_2021.py
# Assign earnings, contributions, and pensions to 2021 microdata,
# and scale them to match macro totals exactly.

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from utils import (
    load_config, resolve_path, load_macro, scale_macro_currency
)

def main():
    cfg = load_config("config.yaml")
    year = int(cfg["years"]["start"])  # single-year flow for now

    # Paths
    root = cfg["project"]["root_dir"]
    out_dir = resolve_path(root, cfg["project"]["output_dir"])
    micro_path = out_dir / f"micro_{year}.csv"
    fin_path   = out_dir / f"micro_fin_{year}.csv"
    diag_path  = out_dir / f"diagnostics_fin_{year}.csv"

    # Config params
    kappa = float(cfg["finance"]["kappa_contribution_rate"])
    sigma_C = float(cfg["design"].get("lognormal_sigma_contributors", 0.85))
    sigma_R = float(cfg["design"].get("lognormal_sigma_retirees", 0.55))
    rng_seed = int(cfg["design"].get("rng_seed", 2021))

    # Load micro + macro
    micro = pd.read_csv(micro_path)
    macro = load_macro(cfg)

    cM = cfg["columns"]["macro"]
    mrow = macro[macro[cM["year"]] == year]
    if mrow.empty:
        raise ValueError(f"Year {year} not found in macro.csv")
    macro_targets = scale_macro_currency(cfg, mrow.iloc[0])

    ie_avg = float(mrow.iloc[0][cM["insurable_earnings_avg"]])
    p_avg  = float(mrow.iloc[0][cM["pension_avg"]])

    # Split groups
    C = micro[micro["sch_grp"] == "C"].copy()
    R = micro[micro["sch_grp"] == "R"].copy()
    N = micro[micro["sch_grp"] == "N"].copy()

    # Deterministic random factors for heterogeneity
    rng = np.random.default_rng(rng_seed)

    # --- Contributors: earnings & contributions
    # Base earnings anchored at ie_avg, with lognormal heterogeneity.
    # Note: lognormal with mean=1 (so scale happens in calibration step).
    # ln factor ~ Normal( -0.5*sigma^2, sigma^2 ) -> E[exp(...)] = 1
    mu_C = -0.5 * (sigma_C ** 2)
    C["earnings"] = ie_avg * np.exp(mu_C + sigma_C * rng.standard_normal(len(C)))
    C["contribution"] = kappa * C["earnings"]

    # --- Retirees: pensions
    mu_R = -0.5 * (sigma_R ** 2)
    R["pension"] = p_avg * np.exp(mu_R + sigma_R * rng.standard_normal(len(R)))

    # --- Non-members: zero for both
    N["earnings"] = 0.0
    N["contribution"] = 0.0
    N["pension"] = 0.0

    # --- Scale to macro totals (using weights)
    # contributions_total and benefits_total are already scaled to currency units
    target_contr = macro_targets["contributions_total"]
    target_benef = macro_targets["benefits_total"]

    # Weighted sums (use 0 where weight is NaN)
    Cw = C["weight"].fillna(0).to_numpy()
    Rw = R["weight"].fillna(0).to_numpy()

    sum_contrib = float(np.dot(C["contribution"].to_numpy(), Cw))
    sum_benef   = float(np.dot(R["pension"].to_numpy(), Rw))

    # Safe guards: if current sum is ~0, avoid division by zero
    sf_C = (target_contr / sum_contrib) if sum_contrib > 0 else 1.0
    sf_R = (target_benef / sum_benef)   if sum_benef   > 0 else 1.0

    C["contribution"] *= sf_C
    C["earnings"]     *= sf_C / kappa  # keep contribution = kappa * earnings after scaling

    R["pension"]      *= sf_R

    # Reassemble micro with finance variables
    keep_cols = ["id_person","cal_yr","age_grp","sex","sch_grp","weight"]
    C = C[keep_cols + ["earnings","contribution"]].assign(pension=0.0)
    R = R[keep_cols + ["pension"]].assign(earnings=0.0, contribution=0.0)
    N = N[keep_cols + ["earnings","contribution","pension"]]

    micro_fin = pd.concat([C, R, N], ignore_index=True)
    micro_fin = micro_fin.sort_values(["sch_grp","sex","age_grp","id_person"]).reset_index(drop=True)

    # Diagnostics
    diag = pd.DataFrame([{
        "year": year,
        "kappa": kappa,
        "sigma_C": sigma_C,
        "sigma_R": sigma_R,
        "scale_factor_contributions": sf_C,
        "scale_factor_benefits": sf_R,
        "macro_contributions_total": target_contr,
        "macro_benefits_total": target_benef,
        "contrib_weighted_sum": float(np.dot(micro_fin.loc[micro_fin["sch_grp"]=="C","contribution"], 
                                             micro_fin.loc[micro_fin["sch_grp"]=="C","weight"].fillna(0))),
        "benefit_weighted_sum": float(np.dot(micro_fin.loc[micro_fin["sch_grp"]=="R","pension"], 
                                             micro_fin.loc[micro_fin["sch_grp"]=="R","weight"].fillna(0))),
        "rows": len(micro_fin)
    }])

    # Write outputs
    micro_fin.to_csv(fin_path, index=False, encoding="utf-8-sig")
    diag.to_csv(diag_path, index=False, encoding="utf-8-sig")

    print("Done.")
    print(f"→ {fin_path}")
    print(f"→ {diag_path}")
    print("Totals in diag should match macro contributions_total and benefits_total.")

if __name__ == "__main__":
    main()
