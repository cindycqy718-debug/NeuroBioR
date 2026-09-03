#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from neurobior_repro import load_data_pack, load_predictor, score_probability_matrix
from neurobior_repro.core import load_windows


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay retained deterministic predictors on aligned TEST data.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, default=Path("results/selected_75_runs"))
    parser.add_argument("--output", type=Path, default=Path("reproduced/predictor_replay.json"))
    parser.add_argument("--dataset", required=True, choices=("APAVA", "ADFTD", "TDBRAIN"))
    parser.add_argument("--model-tag", required=True)
    parser.add_argument("--request-index", required=True, type=int, choices=(41, 42, 43, 44, 45))
    args = parser.parse_args()

    run_dir = args.results_root / args.model_tag / args.dataset / f"seed{args.request_index}"
    pack = load_data_pack(args.data_root, args.dataset)
    windows = load_windows(pack)
    predictor = load_predictor(run_dir / "predictor.py")
    started = time.perf_counter()
    replayed = np.stack([
        np.asarray(predictor.predict(window), dtype=np.float64)
        for window in windows
    ])
    elapsed = time.perf_counter() - started
    archived = np.load(run_dir / "probability_matrix.npy", allow_pickle=False)
    metrics = score_probability_matrix(pack.targets, replayed, class_ids=pack.class_ids)
    hard_equal = np.argmax(replayed, axis=1) == np.argmax(archived, axis=1)
    report = {
        "status": "PASS",
        "model_tag": args.model_tag,
        "dataset": args.dataset,
        "request_index": args.request_index,
        "sample_count": len(windows),
        "hard_label_match_count": int(hard_equal.sum()),
        "hard_label_match_rate": float(hard_equal.mean()),
        "hard_labels_identical": bool(hard_equal.all()),
        "largest_probability_difference": float(np.max(np.abs(replayed - archived))),
        "metrics": metrics,
        "elapsed_seconds": elapsed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

