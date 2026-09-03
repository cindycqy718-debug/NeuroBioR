#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import numpy as np

from neurobior_repro import METRIC_NAMES, load_data_pack, score_probability_matrix


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-score retained probability matrices.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, default=Path("results/selected_75_runs"))
    parser.add_argument("--output-root", type=Path, default=Path("reproduced/scored_runs"))
    parser.add_argument("--dataset", action="append", choices=("APAVA", "ADFTD", "TDBRAIN"))
    args = parser.parse_args()

    selected = set(args.dataset or ("APAVA", "ADFTD", "TDBRAIN"))
    rows: list[dict[str, object]] = []
    skipped: list[str] = []
    for dataset in sorted(selected):
        try:
            pack = load_data_pack(args.data_root, dataset)
        except FileNotFoundError:
            skipped.append(dataset)
            continue
        for matrix_path in sorted(args.results_root.glob(f"*/{dataset}/seed*/probability_matrix.npy")):
            run_dir = matrix_path.parent
            model_tag = run_dir.parents[1].name
            request_index = int(run_dir.name.removeprefix("seed"))
            matrix = np.load(matrix_path, allow_pickle=False)
            metrics = score_probability_matrix(pack.targets, matrix, class_ids=pack.class_ids)
            rows.append({
                "model_tag": model_tag,
                "dataset": dataset,
                "request_index": request_index,
                **{metric: metrics[metric] for metric in METRIC_NAMES},
            })

    if not rows:
        raise RuntimeError("No retained run could be scored with the supplied data")
    args.output_root.mkdir(parents=True, exist_ok=True)
    per_run_path = args.output_root / "per_run_metrics.csv"
    with per_run_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    aggregates = []
    groups = sorted({(str(row["model_tag"]), str(row["dataset"])) for row in rows})
    for model_tag, dataset in groups:
        group = [row for row in rows if row["model_tag"] == model_tag and row["dataset"] == dataset]
        record: dict[str, object] = {"model_tag": model_tag, "dataset": dataset, "n": len(group)}
        for metric in METRIC_NAMES:
            values = [float(row[metric]) * 100 for row in group]
            record[f"{metric}_mean_percent"] = statistics.mean(values)
            record[f"{metric}_sample_sd_percent"] = statistics.stdev(values) if len(values) > 1 else None
        aggregates.append(record)
    aggregate_path = args.output_root / "mean_sample_sd.csv"
    with aggregate_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregates[0]))
        writer.writeheader()
        writer.writerows(aggregates)
    summary = {
        "status": "PASS",
        "scored_run_count": len(rows),
        "skipped_data_not_in_bundle": skipped,
        "per_run_table": str(per_run_path),
        "aggregate_table": str(aggregate_path),
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

