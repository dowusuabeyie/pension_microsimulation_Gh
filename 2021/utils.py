# utils.py
# Minimal helpers for config/CSV loading and simple validation

from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, List
import yaml
import pandas as pd
import numpy as np


# ---------- Config ----------
def load_config(config_path: str | Path = "config.yaml") -> dict:
    cfg_path = Path(config_path).expanduser().resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"config.yaml not found at: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def resolve_path(root_dir: str | Path, rel_path: str) -> Path:
    """Join root_dir and a (possibly nested) relative path using OS separators."""
    root = Path(root_dir)
    parts = Path(rel_path).parts
    return (root.joinpath(*parts)).resolve()


def path_for_year(pattern: str, year: int) -> str:
    """Fill {year} placeholder in a config path pattern."""
    return pattern.format(year=year)


# ---------- CSV utilities ----------
def read_csv_smart(path: Path) -> pd.DataFrame:
    """
    Try multiple encodings so we can read CSVs saved from Excel/Windows.
    Order: utf-8-sig -> cp1252 -> utf-16.
    """
    encodings = ["utf-8-sig", "cp1252", "utf-16"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError as e:
            last_err = e
            continue
    raise UnicodeDecodeError(
        f"Could not decode {path} with encodings {encodings}. "
        f"Last error: {last_err}"
    )


def _coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Coerce given columns to numeric (strip thousands separators, spaces)."""
    for c in cols:
        if c in df.columns:
            df[c] = (
                df[c]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.strip()
                .replace({"": np.nan})
            )
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_macro(cfg: dict) -> pd.DataFrame:
    cols = cfg["columns"]["macro"]
    macro_path = resolve_path(cfg["project"]["root_dir"], cfg["files"]["macro"])
    df = read_csv_smart(macro_path)
    num_cols = [
        cols["contributors"], cols["retirees"], cols["new_retirees"],
        cols["new_registrants"], cols["contributions_total"],
        cols["benefits_total"], cols["pension_avg"], cols["insurable_earnings_avg"],
        cols["coverage_pct"], cols["indexation_pct"],
        cols["establishments_new"], cols["establishments_active"],
    ]
    df = _coerce_numeric(df, num_cols)
    required = set(cols.values())
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"[macro.csv] missing columns: {missing}")
    return df


def load_demog_for_year(cfg: dict, year: int) -> pd.DataFrame:
    cols = cfg["columns"]["demog"]
    patt = cfg["files"]["demog_by_year_pattern"]
    demog_path = resolve_path(cfg["project"]["root_dir"], path_for_year(patt, year))
    df = read_csv_smart(demog_path)
    df = _coerce_numeric(df, [cols["population"], cols["employed"],
                              cols["employment_rate_pct"], cols["mortality_rate"]])
    required = [cols["year"], cols["age_group"], cols["sex"],
                cols["population"], cols["employed"],
                cols["employment_rate_pct"], cols["mortality_rate"]]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"[demog_{year}.csv] missing columns: {missing}")
    if cols["year"] in df.columns:
        df = df[df[cols["year"]] == year].copy()
    return df


def load_sample_for_year(cfg: dict, year: int) -> pd.DataFrame:
    cols = cfg["columns"]["sample"]
    patt = cfg["files"]["sample_by_year_pattern"]
    sample_path = resolve_path(cfg["project"]["root_dir"], path_for_year(patt, year))
    df = read_csv_smart(sample_path)
    df = _coerce_numeric(df, [cols["group_total"], cols["group_sample"]])
    required = [cols["year"], cols["age_group"], cols["sex"],
                cols["scheme_group"], cols["group_total"], cols["group_sample"]]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"[sample_{year}.csv] missing columns: {missing}")
    if cols["year"] in df.columns:
        df = df[df[cols["year"]] == year].copy()
    # Allow C, R, N
    ok = {"C", "R", "N"}
    bad = set(df[cols["scheme_group"]].astype(str).unique()) - ok
    if bad:
        raise ValueError(f"[sample_{year}.csv] invalid {cols['scheme_group']} values: {bad} (allowed: {ok})")
    return df


# ---------- Alignment & simple diagnostics ----------
def align_demog_and_sample(cfg: dict, demog: pd.DataFrame, sample: pd.DataFrame) -> pd.DataFrame:
    cD, cS = cfg["columns"]["demog"], cfg["columns"]["sample"]
    key = [cD["year"], cD["age_group"], cD["sex"]]
    merged = demog.merge(
        sample[[cS["year"], cS["age_group"], cS["sex"], cS["scheme_group"], cS["group_total"], cS["group_sample"]]],
        left_on=key, right_on=[cS["year"], cS["age_group"], cS["sex"]],
        how="left", suffixes=("", "_s")
    )
    sex_order = cfg.get("sex_values", ["M", "F"])
    age_order = cfg.get("age_groups", {}).get("order", None)
    merged[cD["sex"]] = pd.Categorical(merged[cD["sex"]], categories=sex_order, ordered=True)
    if age_order:
        merged[cD["age_group"]] = pd.Categorical(merged[cD["age_group"]], categories=age_order, ordered=True)
    merged = merged.sort_values([cD["sex"], cD["age_group"], cS["scheme_group"]]).reset_index(drop=True)
    return merged


def quick_totals(cfg: dict, demog: pd.DataFrame, sample_all: pd.DataFrame) -> pd.DataFrame:
    """
    Totals by sch_grp (C, R, N): sum of N_g_h, sum of n_g_h, # strata rows.
    """
    cS = cfg["columns"]["sample"]
    grp_col = cS["scheme_group"]
    N_col = cS["group_total"]
    n_col = cS["group_sample"]

    agg = (
        sample_all
        .groupby(grp_col, dropna=False)
        .agg(
            N_total=(N_col, "sum"),
            n_total=(n_col, "sum"),
        )
    )
    sizes = sample_all.groupby(grp_col, dropna=False).size().rename("strata")
    out = (
        agg.join(sizes)
           .reset_index()
           .rename(columns={grp_col: "sch_grp"})
    )
    order = ["C", "R", "N"]
    out["sch_grp"] = pd.Categorical(out["sch_grp"], categories=order, ordered=True)
    out = out.sort_values("sch_grp").reset_index(drop=True)
    return out


def scale_macro_currency(cfg: dict, macro_row: pd.Series) -> Dict[str, float]:
    cM = cfg["columns"]["macro"]
    multC = cfg.get("macro_units", {}).get("contributions_total_multiplier", 1.0)
    multB = cfg.get("macro_units", {}).get("benefits_total_multiplier", 1.0)
    return {
        "contributors_total": float(macro_row[cM["contributors"]]),
        "retirees_total": float(macro_row[cM["retirees"]]),
        "contributions_total": float(macro_row[cM["contributions_total"]]) * float(multC),
        "benefits_total": float(macro_row[cM["benefits_total"]]) * float(multB),
    }
