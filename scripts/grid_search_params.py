#!/usr/bin/env python3
"""Grid search for optimal model parameters (min_stage_cycles, pipeline_overhead_cycles, occupancy_alpha).

Usage:
    PYTHONPATH=src python3 scripts/grid_search_params.py \
        --measured outputs/a40/measured_speedup.csv,outputs/h100/measured_speedup_h100.csv,outputs/l40s/measured_speedup_l40s.csv
"""

import argparse
import csv
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from predictor import predict_one
from gpu_specs import GPU_SPECS


def load_measured(paths):
    rows = []
    for p in paths.split(","):
        p = p.strip()
        if not p:
            continue
        with open(p) as f:
            for row in csv.DictReader(f):
                rows.append(row)
    return rows


def compute_mape(measured, min_cyc, overhead, alpha):
    # Temporarily patch all GPU specs
    saved = {}
    for gpu_name in GPU_SPECS:
        saved[gpu_name] = (
            GPU_SPECS[gpu_name].get("occupancy_alpha"),
            GPU_SPECS[gpu_name].get("min_stage_cycles"),
            GPU_SPECS[gpu_name].get("pipeline_overhead_cycles"),
        )
        GPU_SPECS[gpu_name]["occupancy_alpha"] = alpha
        GPU_SPECS[gpu_name]["min_stage_cycles"] = float(min_cyc)
        GPU_SPECS[gpu_name]["pipeline_overhead_cycles"] = float(overhead)

    apes = []
    for row in measured:
        gpu = row["gpu"]
        if gpu not in GPU_SPECS:
            continue
        try:
            pred = predict_one(
                row["workload"],
                gpu,
                int(row["problem_size"]),
                int(row["stage"]),
                tile_size=int(row["tile_size"]),
            )
            if pred["valid"]:
                m = float(row["measured_speedup"])
                p = pred["pred_speedup"]
                if m > 0:
                    apes.append(abs(p - m) / m)
        except Exception:
            pass

    # Restore original specs
    for gpu_name in saved:
        orig_alpha, orig_min, orig_overhead = saved[gpu_name]
        if orig_alpha is not None:
            GPU_SPECS[gpu_name]["occupancy_alpha"] = orig_alpha
        if orig_min is not None:
            GPU_SPECS[gpu_name]["min_stage_cycles"] = orig_min
        if orig_overhead is not None:
            GPU_SPECS[gpu_name]["pipeline_overhead_cycles"] = orig_overhead

    if not apes:
        return 999.0
    return sum(apes) / len(apes) * 100.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--measured",
        required=True,
        help="Comma-separated measured_speedup CSVs",
    )
    args = parser.parse_args()

    measured = load_measured(args.measured)
    print("Loaded {} measured rows".format(len(measured)))

    # Grid search ranges
    min_stage_values = [150, 180, 200, 220, 240, 260, 280, 300]
    overhead_values = [0, 10, 20, 30, 40, 50, 75, 100]
    alpha_values = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

    results = []
    total = len(min_stage_values) * len(overhead_values) * len(alpha_values)
    count = 0

    for min_cyc in min_stage_values:
        for overhead in overhead_values:
            for alpha in alpha_values:
                count += 1
                mape = compute_mape(measured, min_cyc, overhead, alpha)
                results.append((mape, min_cyc, overhead, alpha))
                if count % 50 == 0:
                    print("  progress: {}/{}".format(count, total))

    results.sort()

    print("\nTop 10 combinations (MAPE, min_stage, overhead, alpha):")
    for mape, mc, ov, al in results[:10]:
        print(
            "  MAPE={:.2f}%  min_stage={}  overhead={}  alpha={}".format(
                mape, mc, ov, al
            )
        )

    print("\nCurrent params (230, 25, 0.3):")
    for mape, mc, ov, al in results:
        if mc == 220 and ov == 20 and abs(al - 0.3) < 0.01:
            print("  MAPE={:.2f}%  (closest grid point)".format(mape))
            break

    print("\nWorst 5 combinations:")
    for mape, mc, ov, al in results[-5:]:
        print(
            "  MAPE={:.2f}%  min_stage={}  overhead={}  alpha={}".format(
                mape, mc, ov, al
            )
        )


if __name__ == "__main__":
    main()
