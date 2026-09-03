#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from neurobior_repro import METRIC_NAMES, load_data_pack, score_probability_matrix, validate_data_pack


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the public NeuroBioR release.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, default=Path("results/selected_75_runs"))
    parser.add_argument("--output", type=Path, default=Path("reproduced/validation_report.json"))
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    results_root = args.results_root.resolve()
    report: dict[str, object] = {"data": {}, "results": {}, "status": "PASS"}

    available = []
    for dataset in ("APAVA", "ADFTD", "TDBRAIN"):
        if (data_root / dataset / "public" / "manifest.json").is_file() and (
            data_root / dataset / "private" / "manifest.json"
        ).is_file():
            available.append(dataset)
            report["data"][dataset] = validate_data_pack(data_root, dataset, load_all=True)

    run_dirs = sorted(path.parent for path in results_root.glob("*/*/seed*/score.json"))
    if len(run_dirs) != 75:
        raise ValueError(f"Expected 75 retained runs, found {len(run_dirs)}")
    counts = {dataset: 0 for dataset in ("APAVA", "ADFTD", "TDBRAIN")}
    rescored = 0
    largest_metric_difference = 0.0
    packs = {dataset: load_data_pack(data_root, dataset) for dataset in available}
    for run_dir in run_dirs:
        dataset = run_dir.parent.name
        counts[dataset] += 1
        matrix_path = run_dir / "probability_matrix.npy"
        predictor_path = run_dir / "predictor.py"
        if not matrix_path.is_file() or not predictor_path.is_file():
            raise FileNotFoundError(f"Incomplete retained run: {run_dir}")
        if dataset not in packs:
            continue
        matrix = np.load(matrix_path, allow_pickle=False)
        metrics = score_probability_matrix(
            packs[dataset].targets,
            matrix,
            class_ids=packs[dataset].class_ids,
        )
        archived = json.loads((run_dir / "score.json").read_text(encoding="utf-8"))["metrics"]
        for metric in METRIC_NAMES:
            difference = abs(float(metrics[metric]) - float(archived[metric]))
            largest_metric_difference = max(largest_metric_difference, difference)
        rescored += 1
    if counts != {"APAVA": 25, "ADFTD": 25, "TDBRAIN": 25}:
        raise ValueError(f"Unexpected retained-run distribution: {counts}")
    if largest_metric_difference > 1e-12:
        raise ValueError(f"Archived metrics do not reproduce: {largest_metric_difference}")

    report["results"] = {
        "retained_run_count": len(run_dirs),
        "runs_by_dataset": counts,
        "rescored_with_available_data": rescored,
        "largest_metric_difference": largest_metric_difference,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

