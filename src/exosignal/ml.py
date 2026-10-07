"""Small, reproducible TIC-grouped benchmark and probability-ranking models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, confusion_matrix, precision_score,
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
        "accuracy": float(accuracy_score(y_true, prediction)),
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


def strong_class_thresholds(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    minimum_purity: float = 0.80,
) -> dict[str, Any]:
    """Find maximum-coverage strong classes from validation predictions.

    ``minimum_purity`` is the minimum observed class agreement within each
    strong region. The resulting boundaries are data-derived and must never
    use final test labels. Scores between them remain explicitly uncertain.
    """
    if not 0.5 < minimum_purity < 1:
        raise ValueError("Strong-class validation purity must be between 0.5 and 1.")
    frame = pd.DataFrame({"label": y_true.astype(int), "score": probabilities.astype(float)}).sort_values("score")
    lower_candidates: list[tuple[float, int, float]] = []
    upper: tuple[float, int, float] | None = None
    for threshold in frame["score"].drop_duplicates().to_numpy():
        lower_group = frame.loc[frame["score"] <= threshold]
        fp_eb_fraction = float((lower_group["label"] == 0).mean())
        if fp_eb_fraction >= minimum_purity:
            lower_candidates.append((float(threshold), len(lower_group), fp_eb_fraction))
        upper_group = frame.loc[frame["score"] >= threshold]
        planet_fraction = float((upper_group["label"] == 1).mean())
        if upper is None and planet_fraction >= minimum_purity:
            # First qualifying threshold gives the widest strong-planet band.
            upper = (float(threshold), len(upper_group), planet_fraction)
    # Require a real uncertain interval. A low-score subset can remain
    # negative-dominated even after it begins to include high-score objects,
    # so choose the widest qualifying lower band strictly below the upper one.
    lower = max((candidate for candidate in lower_candidates if upper is not None and candidate[0] < upper[0]), default=None)
    if lower is None or upper is None:
        raise ValueError("Validation predictions cannot support separate strong and uncertain classes at the requested purity.")
    return {
        "source": "validation TICs only",
        "minimum_observed_validation_purity": minimum_purity,
        "strong_fp_eb_max_score": lower[0],
        "strong_planet_like_min_score": upper[0],
        "strong_fp_eb_validation_examples": lower[1],
        "strong_fp_eb_validation_purity": lower[2],
        "strong_planet_like_validation_examples": upper[1],
        "strong_planet_like_validation_purity": upper[2],
    }


def _plot_evaluation(
    y_true: np.ndarray,
    baseline: np.ndarray,
    calibrated: np.ndarray,
    selected_model: str,
    threshold: float,
    path: Path,
) -> None:
    """Persist a compact held-out confusion/calibration artifact."""
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    selected_probabilities = baseline if selected_model == "logistic_regression" else calibrated
    selected_label = "Logistic regression" if selected_model == "logistic_regression" else "Calibrated Random Forest"
    matrix = confusion_matrix(y_true, selected_probabilities >= threshold, labels=[0, 1])
    image = axes[0].imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axes[0], fraction=0.046)
    axes[0].set(xticks=[0, 1], yticks=[0, 1], xticklabels=["FP/EB-like", "planet-like"], yticklabels=["FP/EB-like", "planet-like"], xlabel="Predicted", ylabel="Ground truth", title=f"{selected_label}: held-out test")
    for (row, column), value in np.ndenumerate(matrix):
        axes[0].text(column, row, str(value), ha="center", va="center")
    for probabilities, label, color in ((baseline, "Logistic regression", "tab:gray"), (calibrated, "Calibrated Random Forest", "tab:blue")):
        observed, predicted = calibration_curve(y_true, probabilities, n_bins=min(8, len(y_true)), strategy="quantile")
        axes[1].plot(predicted, observed, marker="o", label=label, color=color)
    axes[1].plot([0, 1], [0, 1], "k--", linewidth=1, label="ideal")
    axes[1].set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted probability", ylabel="Observed positive fraction", title="Held-out calibration")
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def train_models(
    frame: pd.DataFrame,
    output_directory: Path | str,
    config: BenchmarkConfig | None = None,
    feature_names: list[str] | tuple[str, ...] | None = None,
    strong_class_minimum_purity: float | None = None,
) -> dict[str, Any]:
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
    validation_counts = validation.groupby("label")["tic_id"].nunique()
    if validation_counts.min() < config.minimum_calibration_per_class:
        raise ValueError(
            "Validation calibration population is too small for defensible probability calibration: "
            f"need at least {config.minimum_calibration_per_class} TICs per class, got {validation_counts.to_dict()}."
        )
    declared_features = list(feature_names or MODEL_FEATURES)
    missing_features = [name for name in declared_features if name not in frame.columns]
    if missing_features:
        raise ValueError(f"Benchmark table is missing documented ExoSignal features: {missing_features}.")
    all_unavailable = [
        name for name in declared_features
        if not pd.to_numeric(frame[name], errors="coerce").notna().any()
    ]
    # A column with no measurement in any benchmark TIC is not a model feature:
    # passing it through an imputer only hides that it supplied no information.
    features = [name for name in declared_features if name not in all_unavailable]
    if not features:
        raise ValueError("No documented ExoSignal feature has an observed value in this benchmark.")
    x_train, y_train = train[features], train["label"].to_numpy()
    x_validation, y_validation = validation[features], validation["label"].to_numpy()
    x_test, y_test = test[features], test["label"].to_numpy()
    baseline = Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler()), ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=config.random_seed))])
    baseline.fit(x_train, y_train)
    forest = Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)), ("model", RandomForestClassifier(n_estimators=config.random_forest_trees, class_weight="balanced", min_samples_leaf=2, random_state=config.random_seed, n_jobs=-1))])
    forest.fit(x_train, y_train)
    # FrozenEstimator means the sole calibration set is validation, never test.
    calibrated = CalibratedClassifierCV(FrozenEstimator(forest), method=config.calibration_method, cv="prefit")
    calibrated.fit(x_validation, y_validation)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    # Select the model family before any test probabilities are inspected.
    # The Random Forest is compared pre-calibration here because its sigmoid
    # calibration is fitted on this very validation partition.
    baseline_validation = baseline.predict_proba(x_validation)[:, 1]
    forest_validation = forest.predict_proba(x_validation)[:, 1]
    baseline_validation_metrics = _metrics(y_validation, baseline_validation, config.probability_threshold)
    forest_validation_metrics = _metrics(y_validation, forest_validation, config.probability_threshold)
    def selection_key(metrics: dict[str, Any]) -> tuple[float, float, float, float]:
        return (
            float(metrics["pr_auc"] if metrics["pr_auc"] is not None else -np.inf),
            float(metrics["roc_auc"] if metrics["roc_auc"] is not None else -np.inf),
            -float(metrics["brier_score"]),
            float(metrics["precision"]),
        )
    selected = "logistic_regression" if selection_key(baseline_validation_metrics) >= selection_key(forest_validation_metrics) else "calibrated_random_forest"
    selected_validation = baseline_validation if selected == "logistic_regression" else calibrated.predict_proba(x_validation)[:, 1]
    strong_bands = (
        strong_class_thresholds(y_validation, selected_validation, strong_class_minimum_purity)
        if strong_class_minimum_purity is not None else None
    )

    baseline_test = baseline.predict_proba(x_test)[:, 1]
    calibrated_test = calibrated.predict_proba(x_test)[:, 1]
    joblib.dump(baseline, output / "logistic_regression.joblib")
    joblib.dump(calibrated, output / "calibrated_random_forest.joblib")
    (output / "model_features.json").write_text(
        json.dumps({"features": features, "excluded_all_unavailable_features": all_unavailable}, indent=2) + "\n",
        encoding="utf-8",
    )
    pd.concat([train, validation, test], ignore_index=True).to_csv(output / "benchmark_features_with_splits.csv", index=False)
    test.assign(logistic_probability=baseline_test, calibrated_random_forest_probability=calibrated_test).to_csv(output / "held_out_test_predictions.csv", index=False)
    _plot_evaluation(y_test, baseline_test, calibrated_test, selected, config.probability_threshold, output / "held_out_model_evaluation.png")
    baseline_metrics = _metrics(y_test, baseline_test, config.probability_threshold)
    forest_metrics = _metrics(y_test, calibrated_test, config.probability_threshold)
    result = {
        "features": features,
        "excluded_all_unavailable_features": all_unavailable,
        "selected_ranking_model": selected,
        "selection_rationale": "Selected from validation metrics before final test evaluation, in this order: PR-AUC, ROC-AUC, lower Brier score, then precision.",
        "config": config.to_dict(),
        "calibration": "sigmoid calibration fitted on validation TICs only after fitting the Random Forest on training TICs.",
        "calibration_population_tics": {str(label): int(count) for label, count in validation_counts.items()},
        "split_counts": {name: {str(label): int(count) for label, count in part.groupby("label")["tic_id"].nunique().items()} for name, part in (("train", train), ("validation", validation), ("test", test))},
        "logistic_regression_validation": baseline_validation_metrics,
        "random_forest_validation_precalibration": forest_validation_metrics,
        "logistic_regression_test": baseline_metrics,
        "calibrated_random_forest_test": forest_metrics,
    }
    if strong_bands is not None:
        result["validation_strong_class_thresholds"] = strong_bands
    (output / "model_report.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def rank_features(feature_rows: pd.DataFrame, model_path: Path | str) -> pd.DataFrame:
    """Rank candidates using only the saved catalogue-blind feature table."""
    model_path = Path(model_path)
    model = joblib.load(model_path)
    schema_path = model_path.parent / "model_features.json"
    features = json.loads(schema_path.read_text(encoding="utf-8"))["features"] if schema_path.exists() else MODEL_FEATURES
    frame = feature_rows.copy()
    frame["planet_like_probability"] = model.predict_proba(frame[features])[:, 1]
    return frame.sort_values("planet_like_probability", ascending=False, ignore_index=True)
