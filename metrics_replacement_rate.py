# metrics_replacement_rate.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

# ----------------- utilities -----------------
def wmean(series: pd.Series, weights: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce")
    w = pd.to_numeric(weights, errors="coerce").clip(lower=0)
    m = s.notna() & w.notna() & (w > 0)
    if not m.any():
        return np.nan
    return float(np.sum(s[m] * w[m]) / np.sum(w[m]))

def _load_year(micro_dir: str | Path, year: int) -> pd.DataFrame:
    p = Path(micro_dir) / f"micro_{year}.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")
    df = pd.read_csv(p)
    if "cal_yr" not in df.columns:
        df["cal_yr"] = year
    return df

# ----------------- RR #1: by retiree age bands (>=60) over C 55–59 -----------------
def rr_bands_60plus_vs_C55_59(years: list[int], micro_dir: str | Path = "output") -> pd.DataFrame:
    """
    For each year and sex:
      RR(age_band>=60) = mean_w(pension | R, age_band) / mean_w(ie | C, 55–59).
    Returns a tidy DataFrame with columns: cal_yr, sex, age_band, rr_band.
    """
    out_rows = []
    for year in years:
        df = _load_year(micro_dir, year)

        # masks
        C = df["sch_grp"] == "C"
        R = df["sch_grp"] == "R"
        w = df["weight"]

        # contributors 55–59 denominator (by sex)
        denom = {}
        for sex in ["M", "F"]:
            mC_55_59 = C & (df["sex"] == sex) & (df["age_min"] >= 55) & (df["age_min"] < 60)
            denom[sex] = wmean(df.loc[mC_55_59, "ie"], w.loc[mC_55_59])

        # retiree bands >=60 present in the data
        bands = (
            df.loc[R & (df["age_min"] >= 60), "age_grp"]
            .astype(str)
            .dropna()
            .unique()
        )
        # sort by the lower bound embedded in age_min
        bands = (
            df.loc[R & (df["age_min"] >= 60), ["age_grp", "age_min"]]
            .drop_duplicates()
            .sort_values("age_min")["age_grp"]
            .tolist()
        )

        for sex in ["M", "F"]:
            d = denom[sex]
            for band in bands:
                mR_band = R & (df["sex"] == sex) & (df["age_grp"] == band)
                top = wmean(df.loc[mR_band, "pension"], w.loc[mR_band])
                rr = float(top / d) if (d is not None and d > 0) else np.nan
                out_rows.append({"cal_yr": year, "sex": sex, "age_band": band, "rr_band": rr})

    return pd.DataFrame(out_rows).sort_values(["cal_yr", "sex", "age_band"]).reset_index(drop=True)

# ----------------- RR #2: overall retirees / overall contributors -----------------
def rr_overall_retirees_vs_contributors(years: list[int], micro_dir: str | Path = "output") -> pd.DataFrame:
    """
    For each year and sex:
      RR_overall = mean_w(pension | R, all ages) / mean_w(ie | C, all ages).
    Returns DataFrame: cal_yr, sex, rr_overall.
    """
    out_rows = []
    for year in years:
        df = _load_year(micro_dir, year)
        w = df["weight"]
        for sex in ["M", "F"]:
            top = wmean(df.loc[(df["sch_grp"] == "R") & (df["sex"] == sex), "pension"],
                        w.loc[(df["sch_grp"] == "R") & (df["sex"] == sex)])
            bot = wmean(df.loc[(df["sch_grp"] == "C") & (df["sex"] == sex), "ie"],
                        w.loc[(df["sch_grp"] == "C") & (df["sex"] == sex)])
            rr = float(top / bot) if (bot is not None and bot > 0) else np.nan
            out_rows.append({"cal_yr": year, "sex": sex, "rr_overall": rr})
    return pd.DataFrame(out_rows).sort_values(["cal_yr", "sex"]).reset_index(drop=True)

# ----------------- convenience: run examples -----------------
if __name__ == "__main__":
    years = [2020, 2021]
    print("\nRR by retiree bands (>=60) vs contributors 55–59:")
    print(rr_bands_60plus_vs_C55_59(years, micro_dir="output"))

    print("\nRR overall (all retirees / all contributors):")
    print(rr_overall_retirees_vs_contributors(years, micro_dir="output"))

# in a Python REPL
import metrics_replacement_rate as m
years = [2020, 2021]
m.rr_bands_60plus_vs_C55_59(years).to_csv("output/rr_bands_60plus.csv", index=False)
m.rr_overall_retirees_vs_contributors(years).to_csv("output/rr_overall.csv", index=False)
