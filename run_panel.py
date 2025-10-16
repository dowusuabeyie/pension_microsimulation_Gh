# run_panel.py
from __future__ import annotations
from pathlib import Path
import re
import numpy as np
import pandas as pd

from utils import (
    load_config, resolve_path, load_macro, load_demog_for_year, load_sample_for_year,
    align_demog_and_sample, quick_totals, scale_macro_currency
)

# ---------- helpers ----------
def parse_age_bounds(lbl: str) -> tuple[int, int]:
    """
    Extract (age_min, age_max) from labels like '15–19', '60-64', '95+'.
    For '95+' returns (95, 99). If parsing fails, returns (-1, -1).
    """
    if not isinstance(lbl, str):
        return -1, -1
    # normalize en-dash/emdash to hyphen
    s = lbl.replace("–", "-").replace("—", "-").strip()
    if re.match(r"^\d+\+$", s):
        lo = int(re.findall(r"\d+", s)[0])
        return lo, 99
    nums = re.findall(r"\d+", s)
    if len(nums) == 2:
        lo, hi = int(nums[0]), int(nums[1])
        return lo, hi
    if len(nums) == 1:
        x = int(nums[0]);  return x, x
    return -1, -1

def rescale_domain_totals(sample: pd.DataFrame, grp_col: str, N_col: str, sch: str, target: float):
    """If sum N for sch != target, scale N proportionally within sch."""
    mask = sample[grp_col] == sch
    cur = sample.loc[mask, N_col].sum()
    if pd.isna(cur) or cur == 0:
        return sample, np.nan
    sf = float(target) / float(cur)
    if np.isfinite(sf) and sf > 0:
        sample.loc[mask, N_col] *= sf
        return sample, sf
    return sample, np.nan

def expand_strata_to_micro(strata: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    cS, cD = cfg["columns"]["sample"], cfg["columns"]["demog"]
    rows, uid = [], 1
    for _, r in strata.iterrows():
        n = int(r[cS["group_sample"]]) if pd.notna(r[cS["group_sample"]]) else 0
        if n <= 0:
            continue
        N = float(r[cS["group_total"]]) if pd.notna(r[cS["group_total"]]) else None
        w = (N / n) if (N is not None and n > 0) else None
        age_min, age_max = parse_age_bounds(str(r[cD["age_group"]]))
        for _ in range(n):
            rows.append({
                "id_synt": uid,
                "cal_yr": r[cD["year"]],
                "age_min": age_min,
                "age_max": age_max,
                "age_grp": r[cD["age_group"]],
                "sex": r[cD["sex"]],
                "sch_grp": r[cS["scheme_group"]],
                "weight": w
            })
            uid += 1
    return pd.DataFrame(rows)

def bernoulli_with_target(weights: np.ndarray, base_probs: np.ndarray, target: float, rng: np.random.Generator) -> np.ndarray:
    w = np.asarray(weights, float)
    p = np.clip(np.asarray(base_probs, float), 0.0, 1.0)
    expected = float(np.dot(w, p))
    if expected <= 0 or target <= 0:
        return np.zeros_like(p, dtype=bool)
    s = target / expected
    p_scaled = np.minimum(1.0, p * s)
    return rng.random(len(p_scaled)) < p_scaled

def proportional_pool(mask: np.ndarray, weights: np.ndarray, target: float, rng: np.random.Generator) -> np.ndarray:
    m = np.asarray(mask, bool)
    w_pool = np.asarray(weights, float)[m]
    tot = float(w_pool.sum())
    out = np.zeros_like(m, dtype=bool)
    if tot <= 0 or target <= 0:
        return out
    p = min(1.0, target / tot)
    probs = np.zeros_like(weights, dtype=float); probs[m] = p
    return bernoulli_with_target(weights, probs, target, rng)

# ---------- main pipeline ----------
def main():
    cfg = load_config("config.yaml")
    mode = cfg["years"]["mode"]
    years = list(range(int(cfg["years"]["start"]), int(cfg["years"]["end"])+1)) if mode=="range" else list(cfg["years"]["list"])
    if not years:
        raise ValueError("No years specified in config.yaml")

    root = cfg["project"]["root_dir"]
    out_dir = resolve_path(root, cfg["project"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    stage_dir = out_dir / "_staging"
    stage_dir.mkdir(parents=True, exist_ok=True)

    macro = load_macro(cfg)
    cM, cD, cS = cfg["columns"]["macro"], cfg["columns"]["demog"], cfg["columns"]["sample"]

    # finance params
    kappa = float(cfg["finance"]["kappa_contribution_rate"])
    sigma_C = float(cfg["design"].get("lognormal_sigma_contributors", 0.85))
    sigma_R = float(cfg["design"].get("lognormal_sigma_retirees", 0.55))
    rng_seed = int(cfg["design"].get("rng_seed", 2021))

    for year in years:
        ystage = stage_dir / str(year)
        ystage.mkdir(parents=True, exist_ok=True)

        # ---- load data
        demog  = load_demog_for_year(cfg, year)
        sample = load_sample_for_year(cfg, year)
        mrow = macro.loc[macro[cM["year"]] == year]
        if mrow.empty:
            raise ValueError(f"Year {year} not found in macro.csv")
        macro_targets = scale_macro_currency(cfg, mrow.iloc[0])
        contr_cnt = macro_targets["contributors_total"]
        retir_cnt = macro_targets["retirees_total"]
        contr_amt = macro_targets["contributions_total"]
        benef_amt = macro_targets["benefits_total"]
        ie_avg = float(mrow.iloc[0][cM["insurable_earnings_avg"]])
        p_avg  = float(mrow.iloc[0][cM["pension_avg"]])

        # ---- rescale C/R N_g_h to match macro stocks
        sample, sf_C = rescale_domain_totals(sample, cS["scheme_group"], cS["group_total"], "C", contr_cnt)
        sample, sf_R = rescale_domain_totals(sample, cS["scheme_group"], cS["group_total"], "R", retir_cnt)

        # ---- strata & save (staging)
        strata = align_demog_and_sample(cfg, demog, sample)
        strata_path = ystage / "strata.csv"
        strata.to_csv(strata_path, index=False, encoding="utf-8-sig")

        # ---- expand to micro (id_synt, age_min/max, weight)
        micro = expand_strata_to_micro(strata, cfg)
        micro_base_path = ystage / "micro_base.csv"
        micro.to_csv(micro_base_path, index=False, encoding="utf-8-sig")

        # ---- finances (ie, contribution, pension) scaled to macro money
        rng = np.random.default_rng(rng_seed)
        Cmask = (micro["sch_grp"] == "C")
        Rmask = (micro["sch_grp"] == "R")

        # lognormal factors with mean 1
        muC = -0.5 * (sigma_C**2); muR = -0.5 * (sigma_R**2)

        # IE for contributors; 0 otherwise
        ie = np.zeros(len(micro), float)
        ie[Cmask.values] = ie_avg * np.exp(muC + sigma_C * rng.standard_normal(Cmask.sum()))
        # contributions = kappa * ie (identity BEFORE scaling)
        contrib = kappa * ie

        # pensions for retirees; 0 otherwise
        pens = np.zeros(len(micro), float)
        pens[Rmask.values] = p_avg * np.exp(muR + sigma_R * rng.standard_normal(Rmask.sum()))

        # ---- scale to macro totals (weighted) — CORRECTED
        w = micro["weight"].fillna(0.0).to_numpy()
        cur_contr = float(np.dot(contrib, w))
        cur_benef = float(np.dot(pens, w))

        sf_money_C = (contr_amt / cur_contr) if cur_contr > 0 else 1.0
        sf_money_R = (benef_amt / cur_benef) if cur_benef > 0 else 1.0

        # Scale BOTH ie and contribution by the SAME factor so contribution == kappa * ie holds
        ie      *= sf_money_C
        contrib *= sf_money_C

        # Scale pensions to macro
        pens    *= sf_money_R
  
        # ---- transitions (regr_0, retir_0, deaths)
        regr0_target  = float(mrow.iloc[0][cM["new_registrants"]])
        retir0_target = float(mrow.iloc[0][cM["new_retirees"]])

        # q_as from strata (age_grp, sex)
        qx_map = (
            strata[[cD["age_group"], cD["sex"], cD["mortality_rate"]]]
            .drop_duplicates(subset=[cD["age_group"], cD["sex"]])
            .rename(columns={cD["age_group"]:"age_grp", cD["sex"]:"sex", cD["mortality_rate"]:"q_as"})
        )
        micro = micro.merge(qx_map, on=["age_grp","sex"], how="left")
        micro["q_as"] = micro["q_as"].fillna(0.0)
        micro["p_as"] = 1.0 - micro["q_as"]

        under60   = (micro["age_min"] < 60)
        band55_59 = (micro["age_min"] >= 55) & (micro["age_min"] < 60)
        band60_64 = (micro["age_min"] >= 60) & (micro["age_min"] < 65)

        pool_regr = (micro["sch_grp"]=="N") & under60
        pool_ret1 = (micro["sch_grp"]=="C") & band55_59
        pool_ret2 = (micro["sch_grp"]=="C") & band60_64

        regr_0_ind = proportional_pool(pool_regr.to_numpy(), w, regr0_target, rng)
        # retirees: primary then top-up
        retir_0_ind = np.zeros(len(micro), dtype=bool)
        draw1 = proportional_pool(pool_ret1.to_numpy(), w, retir0_target, rng)
        retir_0_ind |= draw1
        achieved1 = float(np.dot(draw1.astype(float), w))
        need = max(0.0, retir0_target - achieved1)
        if need > 0:
            pool2_eff = pool_ret2 & (~retir_0_ind)
            draw2 = proportional_pool(pool2_eff.to_numpy(), w, need, rng)
            retir_0_ind |= draw2

        # deaths via q_as
        death_ind = bernoulli_with_target(w, micro["q_as"].to_numpy(), float(np.dot(micro["q_as"].to_numpy(), w)), rng)

        # ---- finalize variables
        micro["ie"] = ie
        micro["contribution"] = contrib
        micro["pension"] = pens
        micro["regr_0_ind"]  = regr_0_ind.astype(int)
        micro["retir_0_ind"] = retir_0_ind.astype(int)
        micro["death_ind"]   = death_ind.astype(int)
        micro["sch_status"]  = micro["sch_grp"]  # snapshot status (panel update will evolve it)

        # reorder & write final
        final_cols = [
            "id_synt","cal_yr","age_min","age_max","age_grp","sex","sch_grp","weight",
            "ie","sch_status","contribution","pension","q_as","p_as",
            "regr_0_ind","retir_0_ind","death_ind"
        ]
        micro_final = micro[final_cols].copy()

        # write finals
        final_path = out_dir / f"micro_{year}.csv"
        micro_final.to_csv(final_path, index=False, encoding="utf-8-sig")

        # write year diagnostics (final)
        diag_basic = quick_totals(cfg, demog, sample)
        diag_basic["contributors_total"] = contr_cnt
        diag_basic["retirees_total"] = retir_cnt
        diag_basic["contributions_total"] = contr_amt
        diag_basic["benefits_total"] = benef_amt
        diag_basic["N_scale_C"] = sf_C
        diag_basic["N_scale_R"] = sf_R
        diag_basic["money_scale_C"] = sf_money_C
        diag_basic["money_scale_R"] = sf_money_R
        diag_basic["regr_0_target"] = regr0_target
        diag_basic["retir_0_target"] = retir0_target
        diag_basic["regr_0_weighted"]  = float(np.dot(micro["regr_0_ind"].to_numpy(), w))
        diag_basic["retir_0_weighted"] = float(np.dot(micro["retir_0_ind"].to_numpy(), w))
        diag_basic["deaths_expected_w"] = float(np.dot(micro["q_as"].to_numpy(), w))
        diag_basic["deaths_achieved_w"] = float(np.dot(micro["death_ind"].to_numpy(), w))
        (out_dir / f"diagnostics_{year}.csv").write_text(diag_basic.to_csv(index=False), encoding="utf-8-sig")

        # write intermediates in staging
        (ystage / "micro_fin.csv").write_text(
            pd.DataFrame({"ie":ie, "contribution":contrib, "pension":pens}).to_csv(index=False),
            encoding="utf-8-sig"
        )
        micro.assign(q_as=micro["q_as"], p_as=micro["p_as"],
                     regr_0_ind=micro["regr_0_ind"], retir_0_ind=micro["retir_0_ind"], death_ind=micro["death_ind"])\
             .to_csv(ystage / "micro_transitions.csv", index=False, encoding="utf-8-sig")

        print(f"[{year}] final micro & diagnostics written. Intermediates → {ystage}")

       
if __name__ == "__main__":
    main()
