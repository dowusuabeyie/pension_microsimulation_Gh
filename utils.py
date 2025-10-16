# utils.py
# Helpers for config, robust CSV loading, validation, alignment, and diagnostics.

from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import yaml
import pandas as pd
import numpy as np

# ---------- Config ----------
def load_config(config_path: str | Path = "config.yaml") -> dict:
    p = Path(config_path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"config.yaml not found at: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def resolve_path(root_dir: str | Path, rel_path: str) -> Path:
    root = Path(root_dir)
    parts = Path(rel_path).parts
    return (root.joinpath(*parts)).resolve()

def path_for_year(pattern: str, year: int) -> str:
    return pattern.format(year=year)

# ---------- Robust CSV reading ----------
def read_csv_smart(path: Path) -> pd.DataFrame:
    """
    Try multiple encodings so CSVs saved from Excel/Windows load cleanly.
    """
    encs = ["utf-8-sig", "cp1252", "utf-16"]
    last = None
    for e in encs:
        try:
            return pd.read_csv(path, encoding=e)
        except UnicodeDecodeError as err:
            last = err
            continue
    raise UnicodeDecodeError(f"Could not decode {path} with {encs}. Last: {last}")

def _coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = (
                df[c].astype(str)
                .str.replace(",", "", regex=False)
                .str.strip()
                .replace({"": np.nan})
            )
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# ---------- Loaders ----------
def load_macro(cfg: dict) -> pd.DataFrame:
    cols = cfg["columns"]["macro"]
    p = resolve_path(cfg["project"]["root_dir"], cfg["files"]["macro"])
    df = read_csv_smart(p)
    num_cols = [
        cols["contributors"], cols["retirees"], cols["new_retirees"],
        cols["new_registrants"], cols["contributions_total"], cols["benefits_total"],
        cols["pension_avg"], cols["insurable_earnings_avg"], cols["coverage_pct"],
        cols["indexation_pct"], cols["establishments_new"], cols["establishments_active"],
    ]
    df = _coerce_numeric(df, num_cols)
    missing = [c for c in set(cols.values()) if c not in df.columns]
    if missing:
        raise ValueError(f"[macro.csv] missing columns: {missing}")
    return df

def _read_demog_path(cfg: dict, year: int) -> Path:
    root = cfg["project"]["root_dir"]
    patt = cfg["files"].get("demog_by_year_pattern")
    if patt:
        p = resolve_path(root, path_for_year(patt, year))
        if p.exists():
            return p
    return resolve_path(root, cfg["files"]["demog_all"])

def load_demog_for_year(cfg: dict, year: int) -> pd.DataFrame:
    cols = cfg["columns"]["demog"]
    p = _read_demog_path(cfg, year)
    df = read_csv_smart(p)
    df = _coerce_numeric(df, [cols["population"], cols["employed"],
                              cols["employment_rate_pct"], cols["mortality_rate"]])
    need = [cols["year"], cols["age_group"], cols["sex"],
            cols["population"], cols["employed"],
            cols["employment_rate_pct"], cols["mortality_rate"]]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"[{p.name}] missing columns: {miss}")
    df = df[df[cols["year"]] == year].copy()
    return df

def _read_sample_path(cfg: dict, year: int) -> Path:
    root = cfg["project"]["root_dir"]
    patt = cfg["files"].get("sample_by_year_pattern")
    if patt:
        p = resolve_path(root, path_for_year(patt, year))
        if p.exists():
            return p
    return resolve_path(root, cfg["files"]["sample_all"])

def load_sample_for_year(cfg: dict, year: int) -> pd.DataFrame:
    cols = cfg["columns"]["sample"]
    p = _read_sample_path(cfg, year)
    df = read_csv_smart(p)
    df = _coerce_numeric(df, [cols["group_total"], cols["group_sample"]])
    need = [cols["year"], cols["age_group"], cols["sex"],
            cols["scheme_group"], cols["group_total"], cols["group_sample"]]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"[{p.name}] missing columns: {miss}")
    df = df[df[cols["year"]] == year].copy()
    ok = {"C", "R", "N"}
    bad = set(df[cols["scheme_group"]].astype(str).unique()) - ok
    if bad:
        raise ValueError(f"[{p.name}] invalid {cols['scheme_group']} values: {bad} (allowed: {ok})")
    return df

# ---------- Alignment & diagnostics ----------
def align_demog_and_sample(cfg: dict, demog: pd.DataFrame, sample: pd.DataFrame) -> pd.DataFrame:
    cD, cS = cfg["columns"]["demog"], cfg["columns"]["sample"]
    keyL = [cD["year"], cD["age_group"], cD["sex"]]
    keyR = [cS["year"], cS["age_group"], cS["sex"]]
    merged = demog.merge(
        sample[[cS["year"], cS["age_group"], cS["sex"], cS["scheme_group"],
                cS["group_total"], cS["group_sample"]]],
        left_on=keyL, right_on=keyR, how="left", suffixes=("", "_s")
    )
    sex_order = cfg.get("sex_values", ["M", "F"])
    age_order = cfg.get("age_groups", {}).get("order", None)
    merged[cD["sex"]] = pd.Categorical(merged[cD["sex"]], categories=sex_order, ordered=True)
    if age_order:
        merged[cD["age_group"]] = pd.Categorical(merged[cD["age_group"]], categories=age_order, ordered=True)
    merged = merged.sort_values([cD["sex"], cD["age_group"], cS["scheme_group"]]).reset_index(drop=True)
    return merged

def quick_totals(cfg: dict, demog: pd.DataFrame, sample_all: pd.DataFrame) -> pd.DataFrame:
    cS = cfg["columns"]["sample"]
    grp, Ncol, ncol = cS["scheme_group"], cS["group_total"], cS["group_sample"]
    agg = sample_all.groupby(grp, dropna=False).agg(
        N_total=(Ncol, "sum"),
        n_total=(ncol, "sum"),
    )
    sizes = sample_all.groupby(grp, dropna=False).size().rename("strata")
    out = agg.join(sizes).reset_index().rename(columns={grp: "sch_grp"})
    out["sch_grp"] = pd.Categorical(out["sch_grp"], categories=["C", "R", "N"], ordered=True)
    return out.sort_values("sch_grp").reset_index(drop=True)

def scale_macro_currency(cfg: dict, mrow: pd.Series) -> Dict[str, float]:
    cM = cfg["columns"]["macro"]
    mC = cfg.get("macro_units", {}).get("contributions_total_multiplier", 1.0)
    mB = cfg.get("macro_units", {}).get("benefits_total_multiplier", 1.0)
    return {
        "contributors_total": float(mrow[cM["contributors"]]),
        "retirees_total": float(mrow[cM["retirees"]]),
        "contributions_total": float(mrow[cM["contributions_total"]]) * float(mC),
        "benefits_total": float(mrow[cM["benefits_total"]]) * float(mB),
    }
