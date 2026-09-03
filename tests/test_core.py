from __future__ import annotations

import numpy as np

from neurobior_repro.core import score_probability_matrix, validate_probability_matrix


def test_probability_contract() -> None:
    matrix = np.asarray([[0.8, 0.2], [0.1, 0.9]], dtype=np.float64)
    actual = validate_probability_matrix(matrix, sample_count=2, class_count=2)
    assert np.array_equal(actual, matrix)


def test_metrics_perfect_binary() -> None:
    targets = np.asarray([0, 1, 0, 1])
    matrix = np.asarray([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]])
    metrics = score_probability_matrix(targets, matrix, class_ids=(0, 1))
    for key in ("Accuracy", "Precision", "Recall", "F1", "AUROC", "AUPRC"):
        assert metrics[key] == 1.0


def test_metrics_multiclass_shapes() -> None:
    targets = np.asarray([0, 1, 2, 0, 1, 2])
    matrix = np.asarray(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.6, 0.3, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7],
        ]
    )
    metrics = score_probability_matrix(targets, matrix, class_ids=(0, 1, 2))
    assert metrics["sample_count"] == 6
    assert np.asarray(metrics["confusion"]).shape == (3, 3)

