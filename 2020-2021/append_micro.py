# append_micro.py
# Append per-year micro files into a single panel CSV.
# Default kind = "micro" (finals from run_panel.py).
# You can also run with --kind micro_transitions or micro_fin if you ever need.

from __future__ import annotations
from pathlib import Path
import argparse
import pandas as pd
from utils import load_config, resolve_path

def main():
    ap = argparse.ArgumentParser(description="Append yearly micro files into one panel CSV.")
    ap.add_argument("--kind", default="micro", help="base name: micro | micro_fin | micro_transitions")
    args = ap.parse_args()

    cfg = load_config("config.yaml")
    mode = cfg["years"]["mode"]
    years = list(range(int(cfg["years"]["start"]), int(cfg["years"]["end"]) + 1)) if mode == "range" else list(cfg["years"]["list"])
    if not years:
        raise ValueError("No years specified in config.yaml")

    root = cfg["project"]["root_dir"]
    out_dir = resolve_path(root, cfg["project"]["output_dir"])

    frames = []
    for y in years:
        fp = out_dir / f"{args.kind}_{y}.csv"
        if fp.exists():
            frames.append(pd.read_csv(fp))
        else:
            print(f"Warning: {fp.name} not found; skipping.")

    if not frames:
        raise FileNotFoundError("No yearly files found to append.")

    panel = pd.concat(frames, ignore_index=True)
    out_path = out_dir / f"{args.kind}_panel.csv"
    panel.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Panel written: {out_path} (rows={len(panel):,})")

if __name__ == "__main__":
    main()
