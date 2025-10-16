# expand_micro_2021.py
# Expand strata table (C, R, N) into individual-level microdata for 2021.

from pathlib import Path
import pandas as pd
from utils import load_config, resolve_path

def expand_strata(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    cS = cfg["columns"]["sample"]
    cD = cfg["columns"]["demog"]

    rows = []
    uid = 1
    for _, r in df.iterrows():
        n = int(r[cS["group_sample"]]) if pd.notna(r[cS["group_sample"]]) else 0
        if n <= 0:
            continue
        N = float(r[cS["group_total"]]) if pd.notna(r[cS["group_total"]]) else None
        weight = (N / n) if (N is not None and n > 0) else None
        for _ in range(n):
            rows.append({
                "id_person": uid,
                "cal_yr": r[cD["year"]],
                "age_grp": r[cD["age_group"]],
                "sex": r[cD["sex"]],
                "sch_grp": r[cS["scheme_group"]],  # C, R, or N
                "weight": weight
            })
            uid += 1
    return pd.DataFrame(rows)


def main():
    cfg = load_config("config.yaml")
    year = cfg["years"]["start"]  # single year for now

    # Paths
    root = cfg["project"]["root_dir"]
    out_dir = resolve_path(root, cfg["project"]["output_dir"])
    strata_path = out_dir / f"strata_{year}.csv"
    out_path = out_dir / f"micro_{year}.csv"

    strata = pd.read_csv(strata_path)
    micro = expand_strata(strata, cfg)
    micro.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Microdata written: {out_path} (rows={len(micro):,})")

if __name__ == "__main__":
    main()
