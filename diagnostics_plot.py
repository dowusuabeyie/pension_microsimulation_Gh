# diagnostics_plot.py
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from utils import load_config, read_csv_smart

def make_diagnostics_plots(cfg_path="config.yaml"):
    cfg = load_config(cfg_path)
    root = Path(cfg["project"]["root_dir"])
    out_dir = root / cfg["project"]["output_dir"]
    macro = read_csv_smart(root / cfg["files"]["macro"])
    cM = cfg["columns"]["macro"]

    # Scale macro to base GHS (from billions)
    macro["contr_amt_scaled"] = macro[cM["contributions_total"]] * 1e9
    macro["ben_amt_scaled"] = macro[cM["benefits_total"]] * 1e9

    # Load micro summaries across all years
    micro_summaries = []
    for year in range(cfg["years"]["start"], cfg["years"]["end"] + 1):
        p = out_dir / f"micro_{year}.csv"
        if not p.exists():
            continue
        micro = read_csv_smart(p)
        micro["weight"] = pd.to_numeric(micro["weight"], errors="coerce")

        # Weighted totals by scheme
        agg = (
            micro.groupby("sch_grp", dropna=False)
            .agg(weighted_pop=("weight", "sum"))
            .reset_index()
        )
        agg["cal_yr"] = year
        micro_summaries.append(agg)

    if not micro_summaries:
        raise RuntimeError("No micro_*.csv found in output folder")

    micro_all = pd.concat(micro_summaries, ignore_index=True)

    # Merge macro totals
    compare = (
        macro.rename(columns={
            cM["contributors"]: "macro_C",
            cM["retirees"]: "macro_R"
        })
        [["cal_yr", "macro_C", "macro_R", "contr_amt_scaled", "ben_amt_scaled"]]
    )

    # Pivot micro for easy comparison
    micro_pivot = (
        micro_all.pivot(index="cal_yr", columns="sch_grp", values="weighted_pop")
        .rename_axis(None, axis=1)
        .rename(columns={"C": "micro_C", "R": "micro_R"})
        .reset_index()
    )

    merged = compare.merge(micro_pivot, on="cal_yr", how="left")

    # === Plot 1: Contributors and Retirees totals ===
    plt.figure(figsize=(9, 5))
    plt.plot(merged["cal_yr"], merged["macro_C"], "o-", label="Macro Contributors")
    plt.plot(merged["cal_yr"], merged["micro_C"], "s--", label="Micro Contributors (weighted)")
    plt.plot(merged["cal_yr"], merged["macro_R"], "o-", label="Macro Retirees")
    plt.plot(merged["cal_yr"], merged["micro_R"], "s--", label="Micro Retirees (weighted)")
    plt.title("Validation of Synthetic Microdata against Macro Totals")
    plt.xlabel("Year")
    plt.ylabel("Number of Persons")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "diag_population_validation.png", dpi=300)

    # === Plot 2: Contributions and Benefits totals (GHS billions) ===
    plt.figure(figsize=(9, 5))
    plt.plot(merged["cal_yr"], merged["contr_amt_scaled"] / 1e9, "o-", label="Macro Contributions (bn GHS)")
    plt.plot(merged["cal_yr"], merged["ben_amt_scaled"] / 1e9, "s--", label="Macro Benefits (bn GHS)")
    plt.title("Macro Financial Totals (Contributions & Benefits)")
    plt.xlabel("Year")
    plt.ylabel("Amount (Billion GHS)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "diag_financial_validation.png", dpi=300)

    print("✅ Diagnostics plots generated in:", out_dir)

if __name__ == "__main__":
    make_diagnostics_plots()
