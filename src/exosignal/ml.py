"""Small, reproducible TIC-grouped benchmark and probability-ranking models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import BenchmarkConfig
from .features import MODEL_FEATURES


def _target_labels(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = frame.groupby("tic_id")["label"].nunique()
    inconsistent = grouped[grouped != 1]
    if not inconsistent.empty:
        raise ValueError("Each TIC must have exactly one externally supplied ground-truth label.")
    return frame[["tic_id", "label"]].drop_duplicates().sort_values("tic_id")


def split_by_tic(frame: pd.DataFrame, config: BenchmarkConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create a 70/15/15 target-level split without candidate leakage."""
    targets = _target_labels(frame)
    counts = targets["label"].value_counts()
    if len(counts) != 2 or counts.min() < 6:
        raise ValueError("At least six TICs in each class are required for a 70/15/15 grouped benchmark.")
    rng = np.random.default_rng(config.random_seed)
    memberships: dict[int, str] = {}
    for label, group in targets.groupby("label", sort=True):
        ids = group["tic_id"].to_numpy(copy=True)
        rng.shuffle(ids)
        n = len(ids)
        n_train = max(1, round(n * config.train_fraction))
        n_validation = max(1, round(n * config.validation_fraction))
        # Retain at least one target in final test split.
        if n_train + n_validation >= n:
            n_train = n - 2
            n_validation = 1
        for tic_id in ids[:n_train]:
            memberships[int(tic_id)] = "train"
        for tic_id in ids[n_train:n_train + n_validation]:
            memberships[int(tic_id)] = "validation"
        for tic_id in ids[n_train + n_validation:]:
            memberships[int(tic_id)] = "test"
    annotated = frame.assign(split=frame["tic_id"].map(memberships))
    return tuple(annotated.loc[annotated["split"] == name].copy() for name in ("train", "validation", "test"))


def _metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
    prediction = (probabilities >= threshold).astype(int)
    result: dict[str, Any] = {
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, prediction, labels=[0, 1]).tolist(),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
    }
    result["pr_auc"] = float(average_precision_score(y_true, probabilities)) if len(np.unique(y_true)) == 2 else None
    result["roc_auc"] = float(roc_auc_score(y_true, probabilities)) if len(np.unique(y_true)) == 2 else None
    if len(np.unique(y_true)) == 2:
        observed, predicted = calibration_curve(y_true, probabilities, n_bins=min(8, len(y_true)), strategy="quantile")
        result["calibration_curve"] = {"mean_predicted_probability": predicted.tolist(), "fraction_positive": observed.tolist()}
    else:
        result["calibration_curve"] = None
    return result


def train_models(frame: pd.DataFrame, output_directory: Path | str, config: BenchmarkConfig | None = None) -> dict[str, Any]:
    """Fit a baseline and one RF, calibrating RF only on held-out validation.

    The test partition is not inspected until this function's final reporting
    block.  Missing numeric values are median-imputed *within each fitted
    pipeline*, preserving a real NaN pixel offset rather than treating it as
    a zero-offset centroid.
    """
    config = config or BenchmarkConfig()
    missing = set(["tic_id", "label"]) - set(frame.columns)
    if missing:
        raise ValueError(f"Benchmark feature table is missing {sorted(missing)}.")
    frame = frame.dropna(subset=["label"]).copy()
    frame["label"] = frame["label"].astype(int)
    train, validation, test = split_by_tic(frame, config)
    for name, partition in (("train", train), ("validation", validation), ("test", test)):
        if partition["label"].nunique() != 2:
            raise ValueError(f"{name} split lacks both classes; add more labelled TICs.")
    features = [name for name in MODEL_FEATURES if name in frame.columns]
    if len(features) != len(MODEL_FEATURES):
        raise ValueError("Benchmark table is missing one or more documented ExoSignal features.")
    x_train, y_train = train[features], train["label"].to_numpy()
    x_validation, y_validation = validation[features], validation["label"].to_numpy()
    x_test, y_test = test[features], test["label"].to_numpy()
    baseline = Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler()), ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=config.random_seed))])
    baseline.fit(x_train, y_train)
    forest = Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)), ("model", RandomForestClassifier(n_estimators=config.random_forest_trees, class_weight="balanced", min_samples_leaf=2, random_state=config.random_seed, n_jobs=-1))])
    forest.fit(x_train, y_train)
    # ``cv='prefit'`` means the sole calibration set is validation, never test.
    calibrated = CalibratedClassifierCV(forest, method=config.calibration_method, cv="prefit")
    calibrated.fit(x_validation, y_validation)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(baseline, output / "logistic_regression.joblib")
    joblib.dump(calibrated, output / "calibrated_random_forest.joblib")
    pd.concat([train, validation, test], ignore_index=True).to_csv(output / "benchmark_features_with_splits.csv", index=False)
    result = {
        "features": features,
        "config": config.to_dict(),
        "calibration": "sigmoid calibration fitted on validation TICs only after fitting the Random Forest on training TICs.",
        "split_counts": {name: {str(label): int(count) for label, count in part.groupby("label")["tic_id"].nunique().items()} for name, part in (("train", train), ("validation", validation), ("test", test))},
        "logistic_regression_test": _metrics(y_test, baseline.predict_proba(x_test)[:, 1], config.probability_threshold),
        "calibrated_random_forest_test": _metrics(y_test, calibrated.predict_proba(x_test)[:, 1], config.probability_threshold),
    }
    (output / "model_report.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def rank_features(feature_rows: pd.DataFrame, model_path: Path | str) -> pd.DataFrame:
    """Rank candidates using only the saved catalogue-blind feature table."""
    model = joblib.load(model_path)
    frame = feature_rows.copy()
    frame["planet_like_probability"] = model.predict_proba(frame[MODEL_FEATURES])[:, 1]
    return frame.sort_values("planet_like_probability", ascending=False, ignore_index=True)
