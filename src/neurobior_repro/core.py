from __future__ import annotations

import ast
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Sequence

import numpy as np


METRIC_NAMES = ("Accuracy", "Precision", "Recall", "F1", "AUROC", "AUPRC")
ALLOWED_IMPORT_ROOTS = {"math", "numpy", "scipy"}


@dataclass(frozen=True)
class DataPack:
    root: Path
    dataset: str
    public: dict[str, Any]
    private: dict[str, Any]
    sample_ids: tuple[str, ...]
    targets: np.ndarray
    class_ids: tuple[int, ...]


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return payload


def load_data_pack(data_root: str | Path, dataset: str) -> DataPack:
    root = Path(data_root).resolve() / dataset.upper()
    public_path = root / "public" / "manifest.json"
    private_path = root / "private" / "manifest.json"
    if not public_path.is_file() or not private_path.is_file():
        raise FileNotFoundError(
            f"{dataset.upper()} requires public/manifest.json and private/manifest.json"
        )
    public = _read_json(public_path)
    private = _read_json(private_path)
    public_ids = tuple(
        sample_id
        for batch in public.get("batches", [])
        for sample_id in batch.get("sample_ids", [])
    )
    private_rows = private.get("samples", [])
    private_ids = tuple(str(row["sample_id"]) for row in private_rows)
    if public_ids != private_ids:
        raise ValueError(f"Public/private sample order differs for {dataset.upper()}")
    if len(public_ids) != int(public.get("sample_count", -1)):
        raise ValueError(f"Manifest sample count differs for {dataset.upper()}")
    class_ids = tuple(sorted(int(value) for value in public["class_names"]))
    targets = np.asarray([int(row["class_id"]) for row in private_rows], dtype=np.int64)
    if set(np.unique(targets)) != set(class_ids):
        raise ValueError(f"Target classes differ from the public class contract for {dataset.upper()}")
    if public.get("split") != "TEST" or public.get("test_only") is not True:
        raise ValueError(f"{dataset.upper()} is not marked as an official TEST-only pack")
    if public.get("ground_truth_present") is not False:
        raise ValueError(f"Public manifest unexpectedly exposes ground truth for {dataset.upper()}")
    return DataPack(
        root=root,
        dataset=str(public["dataset"]),
        public=public,
        private=private,
        sample_ids=public_ids,
        targets=targets,
        class_ids=class_ids,
    )


def iter_batches(pack: DataPack) -> Iterable[np.ndarray]:
    for batch in pack.public["batches"]:
        path = pack.root / "public" / batch["input_path"]
        array = np.load(path, allow_pickle=False)
        expected = tuple(int(value) for value in batch["input_shape"])
        if array.shape != expected:
            raise ValueError(f"Unexpected shape {array.shape} for {path}; expected {expected}")
        if array.dtype != np.float32:
            raise ValueError(f"Unexpected dtype {array.dtype} for {path}; expected float32")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"Non-finite EEG values in {path}")
        yield array


def load_windows(pack: DataPack) -> np.ndarray:
    batches = list(iter_batches(pack))
    if not batches:
        raise ValueError(f"No EEG batches found for {pack.dataset}")
    windows = np.concatenate(batches, axis=0)
    expected_shape = (len(pack.sample_ids), *map(int, pack.public["window_shape"]))
    if windows.shape != expected_shape:
        raise ValueError(f"Unexpected complete tensor shape {windows.shape}; expected {expected_shape}")
    return windows


def validate_data_pack(data_root: str | Path, dataset: str, *, load_all: bool = True) -> dict[str, Any]:
    pack = load_data_pack(data_root, dataset)
    total = 0
    if load_all:
        for batch in iter_batches(pack):
            total += len(batch)
    else:
        total = sum(int(batch["input_shape"][0]) for batch in pack.public["batches"])
    if total != len(pack.sample_ids):
        raise ValueError(f"Batch total differs from manifest for {pack.dataset}")
    return {
        "dataset": pack.dataset,
        "split": "TEST",
        "sample_count": len(pack.sample_ids),
        "batch_count": len(pack.public["batches"]),
        "window_shape": list(map(int, pack.public["window_shape"])),
        "class_names": pack.public["class_names"],
        "status": "PASS",
    }


def validate_probability_matrix(
    probabilities: np.ndarray,
    *,
    sample_count: int,
    class_count: int,
) -> np.ndarray:
    matrix = np.asarray(probabilities, dtype=np.float64)
    if matrix.shape != (sample_count, class_count):
        raise ValueError(
            f"Expected probability matrix {(sample_count, class_count)}, found {matrix.shape}"
        )
    if np.any(~np.isfinite(matrix)) or np.any(matrix < 0):
        raise ValueError("Probabilities must be finite and nonnegative")
    if not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-6, rtol=0):
        raise ValueError("Every probability row must sum to one")
    return matrix


def _binary_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    binary = np.asarray(labels, dtype=np.int8)
    values = np.asarray(scores, dtype=np.float64)
    positives = int(binary.sum())
    negatives = len(binary) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("Binary AUROC requires positive and negative examples")
    order = np.argsort(values, kind="stable")
    sorted_scores = values[order]
    ranks = np.arange(1, len(values) + 1, dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        ranks[start:stop] = float(np.mean(ranks[start:stop]))
        start = stop
    original_ranks = np.empty_like(ranks)
    original_ranks[order] = ranks
    rank_sum = float(original_ranks[binary == 1].sum())
    return float((rank_sum - positives * (positives + 1) / 2) / (positives * negatives))


def _binary_auprc(labels: np.ndarray, scores: np.ndarray) -> float:
    binary = np.asarray(labels, dtype=np.int8)
    values = np.asarray(scores, dtype=np.float64)
    positives = int(binary.sum())
    negatives = len(binary) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("Binary AUPRC requires positive and negative examples")
    order = np.argsort(-values, kind="stable")
    sorted_scores = values[order]
    sorted_labels = binary[order]
    true_positives = 0
    previous_recall = 0.0
    average_precision = 0.0
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        true_positives += int(sorted_labels[start:stop].sum())
        recall = true_positives / positives
        precision = true_positives / stop
        average_precision += (recall - previous_recall) * precision
        previous_recall = recall
        start = stop
    return float(average_precision)


def score_probability_matrix(
    targets: Sequence[int] | np.ndarray,
    probabilities: np.ndarray,
    *,
    class_ids: Sequence[int],
) -> dict[str, Any]:
    target_array = np.asarray(targets, dtype=np.int64)
    ordered_classes = tuple(int(value) for value in class_ids)
    matrix = validate_probability_matrix(
        probabilities,
        sample_count=len(target_array),
        class_count=len(ordered_classes),
    )
    predicted = np.asarray(
        [ordered_classes[index] for index in matrix.argmax(axis=1)],
        dtype=np.int64,
    )
    class_index = {class_id: index for index, class_id in enumerate(ordered_classes)}
    confusion = np.zeros((len(ordered_classes), len(ordered_classes)), dtype=np.int64)
    for target, prediction in zip(target_array, predicted):
        confusion[class_index[int(target)], class_index[int(prediction)]] += 1
    precision_values: list[float] = []
    recall_values: list[float] = []
    f1_values: list[float] = []
    per_class_auroc: dict[str, float] = {}
    per_class_auprc: dict[str, float] = {}
    for index, class_id in enumerate(ordered_classes):
        true_positive = int(confusion[index, index])
        predicted_positive = int(confusion[:, index].sum())
        actual_positive = int(confusion[index, :].sum())
        precision = true_positive / predicted_positive if predicted_positive else 0.0
        recall = true_positive / actual_positive if actual_positive else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)
        binary_targets = (target_array == class_id).astype(np.int8)
        per_class_auroc[str(class_id)] = _binary_auroc(binary_targets, matrix[:, index])
        per_class_auprc[str(class_id)] = _binary_auprc(binary_targets, matrix[:, index])
    return {
        "Accuracy": float(np.mean(predicted == target_array)),
        "Precision": float(np.mean(precision_values)),
        "Recall": float(np.mean(recall_values)),
        "F1": float(np.mean(f1_values)),
        "AUROC": float(np.mean(list(per_class_auroc.values()))),
        "AUPRC": float(np.mean(list(per_class_auprc.values()))),
        "PerClassAUROC": per_class_auroc,
        "PerClassAUPRC": per_class_auprc,
        "confusion": confusion.tolist(),
        "sample_count": int(len(target_array)),
    }


def _validate_predictor_source(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots = {alias.name.split(".", 1)[0] for alias in node.names}
            if not roots <= ALLOWED_IMPORT_ROOTS:
                raise ValueError(f"Disallowed predictor import in {path}: {sorted(roots)}")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", 1)[0]
            if root not in ALLOWED_IMPORT_ROOTS:
                raise ValueError(f"Disallowed predictor import in {path}: {root}")
        elif isinstance(node, (ast.With, ast.AsyncWith, ast.ClassDef, ast.Lambda)):
            raise ValueError(f"Disallowed predictor construct {type(node).__name__} in {path}")
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if not any(node.name == "predict" for node in functions):
        raise ValueError(f"Missing predict(window) in {path}")


def load_predictor(path: str | Path) -> ModuleType:
    predictor_path = Path(path).resolve()
    _validate_predictor_source(predictor_path)
    module_name = "neurobior_frozen_predictor_" + predictor_path.parent.name
    spec = importlib.util.spec_from_file_location(module_name, predictor_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import predictor: {predictor_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "predict", None)):
        raise TypeError(f"Predictor does not define callable predict(window): {predictor_path}")
    return module

