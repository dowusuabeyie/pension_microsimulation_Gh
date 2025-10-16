# transitions_2021.py
# Tag 2021 flow indicators on the microdata:
#  - regr_0_ind: new registrants (N -> C) during year t
#  - retir_0_ind: new retirees during year t
#  - death_ind: deaths during year t (via qx by age-sex)
#
# Targets (weighted by 'weight'):
#  - Sum(regr_0_ind * w)  ~= macro regr_0
#  - Sum(retir_0_ind * w) ~= macro retir_0
#  - Sum(death_ind * w)   ~= Sum(qx_h * w_h)
#
# Pools:
#  - regr_0: sch_grp == 'N' & age < 60
#  - retir_0: sch_grp == 'C' & age >= 60
#  - deaths: everyone, with p = qx(age, sex)
#
# Notes:
#  - 2021 treated as cross-section augmented with flow flags.
#  - Randomness controlled by config.design.rng_seed.

from __future__ import annotations
from pathlib import Path
import re
import numpy as np
import pandas as pd
from utils import (
    load_config, resolve_path, load_macro
)

# ---------- helpers ----------

def parse_age_min(age_label: str) -> int:
    """Extract lower bound from labels like '15–19', '60–64', '95+'."""
    if not isinstance(age_label, str):
        return -1
    m = re.search(r"(\d+)", age_label)
    return int(m.group(1)) if m else -1


def bernoulli_with_target(weights: np.ndarray, base_probs: np.ndarray, target: float, rng: np.random.Generator) -> np.ndarray:
    """
    Draw Bernoulli indicators with probabilities scaled to hit a weighted target.
    p' = min(1, s * p) where s = target / sum(w * p). Then draw Bernoulli(p').
    """
    w = np.asarray(weights, float)
    p = np.clip(np.asarray(base_probs, float), 0.0, 1.0)
    expected = float(np.dot(w, p))
    if expected <= 0 or target <= 0:
        return np.zeros_like(p, dtype=bool)

    s = target / expected
    p_scaled = np.minimum(1.0, p * s)
    draw = rng.random(len(p_scaled)) < p_scaled
    return draw


def proportional_bernoulli_by_pool(pool_mask: np.ndarray, weights: np.ndarray, target: float, rng: np.random.Generator) -> np.ndarray:
    """
    Within a pool, use a single probability p = target / sum(w_pool) and draw Bernoulli(p).
    Scales naturally to hit the target in expectation.
    """
    mask = np.asarray(pool_mask, bool)
    w_pool = np.asarray(weights, float)[mask]
    tot = float(w_pool.sum())
    out = np.zeros_like(mask, dtype=bool)
    if tot <= 0 or target <= 0:
        return out
    p = min(1.0, target / tot)
    probs = np.zeros_like(weights, dtype=float)
    probs[mask] = p
    return bernoulli_with_target(np.asarray(weights, float), probs, target, rng)


def main():
    cfg = load_config("config.yaml")
    year = int(cfg["years"]["start"])  # single-year flow for now

    # Paths
    root = cfg["project"]["root_dir"]
    out_dir = resolve_path(root, cfg["project"]["output_dir"])
    micro_fin_path = out_dir / f"micro_fin_{year}.csv"
    strata_path     = out_dir / f"strata_{year}.csv"
    out_micro       = out_dir / f"micro_transitions_{year}.csv"
    out_diag        = out_dir / f"diagnostics_transitions_{year}.csv"

    # RNG
    rng_seed = int(cfg["design"].get("rng_seed", 2021))
    rng = np.random.default_rng(rng_seed)

    # ---- Load data
    micro = pd.read_csv(micro_fin_path)
    strata = pd.read_csv(strata_path)

    # Normalize headers
    micro.columns = micro.columns.str.strip()
    strata.columns = strata.columns.str.strip()

    # Recover likely variants for micro columns
    ren = {}
    for c in micro.columns:
        key = c.replace(" ", "").replace("\u2009", "").replace("\u202f", "").lower()
        if key in {"agegrp", "age_group", "agegrp.", "agegroup", "age_grp"}:
            ren[c] = "age_grp"
        if key in {"gender", "sex", "sex."}:
            ren[c] = "sex"
    if ren:
        micro = micro.rename(columns=ren)

    required_micro_cols = {"age_grp", "sex", "weight", "sch_grp"}
    miss = [c for c in required_micro_cols if c not in micro.columns]
    if miss:
        raise KeyError(f"Missing columns in micro_fin_{year}.csv: {miss}. Found: {list(micro.columns)}")

    # ---- Merge qx by (age_grp, sex) onto micro (preserving keys)
    cD = cfg["columns"]["demog"]
    qx_by_cell = (
        strata[[cD["age_group"], cD["sex"], cD["mortality_rate"]]]
        .drop_duplicates(subset=[cD["age_group"], cD["sex"]])
        .rename(columns={
            cD["age_group"]: "age_grp",
            cD["sex"]: "sex",
            cD["mortality_rate"]: "qx"
        })
    )
    micro = micro.merge(qx_by_cell, on=["age_grp", "sex"], how="left")

    # ---- Eligibility flags
    micro["age_min"] = micro["age_grp"].astype(str).map(parse_age_min)
    w = micro["weight"].fillna(0.0).to_numpy()

    is_N = (micro["sch_grp"] == "N")
    is_C = (micro["sch_grp"] == "C")

    pool_regr = is_N & (micro["age_min"] < 60)   # N, working-age
    pool_ret0 = is_C & (micro["age_min"] >= 60)  # C, 60+

    # ---- Macro flow targets (counts)
    macro = load_macro(cfg)
    cM = cfg["columns"]["macro"]
    mrow = macro.loc[macro[cM["year"]] == year]
    if mrow.empty:
        raise ValueError(f"Year {year} not found in macro.csv")
    regr0_target = float(mrow.iloc[0][cM["new_registrants"]])  # regr_0
    retir0_target = float(mrow.iloc[0][cM["new_retirees"]])    # retir_0

    # ---- Draw indicators
    regr_ind  = proportional_bernoulli_by_pool(pool_regr.to_numpy(), w, regr0_target, rng)
    retir_ind = proportional_bernoulli_by_pool(pool_ret0.to_numpy(), w, retir0_target, rng)

    # Deaths via qx
    qx = micro["qx"].fillna(0.0).to_numpy()
    expected_deaths = float(np.dot(qx, w))
    death_ind = bernoulli_with_target(w, qx, expected_deaths, rng)

    micro["regr_0_ind"]  = regr_ind.astype(int)
    micro["retir_0_ind"] = retir_ind.astype(int)
    micro["death_ind"]   = death_ind.astype(int)

    # ---- Diagnostics
    diag = {
        "year": year,
        "regr_0_target": regr0_target,
        "retir_0_target": retir0_target,
        "regr_0_weighted": float(np.dot(micro["regr_0_ind"].to_numpy(), w)),
        "retir_0_weighted": float(np.dot(micro["retir_0_ind"].to_numpy(), w)),
        "deaths_expected_weighted": expected_deaths,
        "deaths_achieved_weighted": float(np.dot(micro["death_ind"].to_numpy(), w)),
        "pool_regr_weight": float(w[pool_regr.to_numpy()].sum()),
        "pool_ret0_weight": float(w[pool_ret0.to_numpy()].sum()),
        "n_individuals": int(len(micro)),
    }
    df_diag = pd.DataFrame([diag])

    # ---- Write
    micro.to_csv(out_micro, index=False, encoding="utf-8-sig")
    df_diag.to_csv(out_diag, index=False, encoding="utf-8-sig")

    print("Done.")
    print(f"→ {out_micro}")
    print(f"→ {out_diag}")
    print("Check: regr_0_weighted ≈ regr_0_target; retir_0_weighted ≈ retir_0_target; deaths_expected ≈ deaths_achieved.")


if __name__ == "__main__":
    main()
