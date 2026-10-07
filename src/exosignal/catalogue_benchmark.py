"""A reproducible, catalogue-only NASA TOI benchmark.

This module deliberately does not retrieve light curves. It evaluates whether
numeric values already published in the NASA Exoplanet Archive TOI table can
separate trusted planet and false-positive dispositions. It is therefore a
catalogue-feature classifier, not an independent transit-detection benchmark.
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import quote_plus

import numpy as np
import pandas as pd

from .config import BenchmarkConfig
from .errors import ExoSignalError
from .ml import rank_features, train_models


NASA_TOI_TAP = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
FEATURE_COLUMNS = (
    "pl_orbper", "pl_trandurh", "pl_trandep", "st_tmag",
    "st_dist", "st_teff", "st_logg", "st_rad",
)
POSITIVE_DISPOSITIONS = {"CP", "KP"}
NEGATIVE_DISPOSITIONS = {"FP", "FA"}
# "Strong" is reserved for a region with at least 85% observed agreement in
# validation. The actual score boundaries are still learned from validation,
# rather than hard-coded score cutoffs.
STRONG_CLASS_MINIMUM_VALIDATION_PURITY = 0.85


def _catalogue_value(value: object) -> object:
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _classification(score: float, thresholds: dict[str, object]) -> str:
    if score >= float(thresholds["strong_planet_like_min_score"]):
        return "STRONGLY_PLANET_LIKE"
    if score <= float(thresholds["strong_fp_eb_max_score"]):
        return "STRONGLY_FP_EB_LIKE"
    return "UNCERTAIN"


def _candidate_record(scored_row: pd.Series, source_row: pd.Series, thresholds: dict[str, object]) -> dict[str, object]:
    score = float(scored_row["planet_like_probability"])
    return {
        "toi": float(scored_row["toi"]),
        "tic_id": int(scored_row["tic_id"]),
        "planet_like_benchmark_score": score,
        "benchmark_class": _classification(score, thresholds),
        "published_measurements": {feature: (float(scored_row[feature]) if pd.notna(scored_row[feature]) else None) for feature in FEATURE_COLUMNS},
        # Returned only after scoring for transparency; never a model input.
        "actual_catalogue_disposition": str(source_row["tfopwg_disp"]),
        "nasa_toi_row": {str(column): _catalogue_value(value) for column, value in source_row.items()},
    }


def download_nasa_toi_table(cache_path: Path | str) -> pd.DataFrame:
    """Download and cache NASA's public TOI table with its original columns."""
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return pd.read_csv(path)
    query = quote_plus("select * from toi")
    frame = pd.read_csv(f"{NASA_TOI_TAP}?query={query}&format=csv")
    frame.to_csv(path, index=False)
    return frame


def _representative_sample(frame: pd.DataFrame, count: int, seed: int) -> pd.DataFrame:
    """Seeded sampling across log-period bins, avoiding a short-period head."""
    if len(frame) < count:
        raise ValueError(f"Need {count} eligible targets, but source has only {len(frame)}.")
    work = frame.copy()
    work["_period_bin"] = pd.qcut(
        np.log10(work["pl_orbper"].astype(float)), q=min(8, count, len(work)), duplicates="drop"
    )
    groups = [group for _, group in work.groupby("_period_bin", observed=True, sort=True)]
    rng = np.random.default_rng(seed)
    target_per_group = [count // len(groups) + (index < count % len(groups)) for index in range(len(groups))]
    selected: list[pd.DataFrame] = []
    remainder: list[pd.DataFrame] = []
    for group, requested in zip(groups, target_per_group):
        order = rng.permutation(len(group))
        selected.append(group.iloc[order[:requested]])
        remainder.append(group.iloc[order[requested:]])
    result = pd.concat(selected, ignore_index=True)
    if len(result) < count:
        pool = pd.concat(remainder, ignore_index=True)
        result = pd.concat([result, pool.iloc[rng.permutation(len(pool))[: count - len(result)]]], ignore_index=True)
    return result.drop(columns="_period_bin").sort_values(["tid", "toi"], kind="stable", ignore_index=True)


def select_labelled_toi_targets(
    catalogue: pd.DataFrame,
    per_class: int | None = 500,
    seed: int = 20261006,
    excluded_tic_ids: set[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return separate label manifest and raw-name numerical feature table.

    The target unit is one TIC. A TIC with conflicting trusted dispositions is
    excluded rather than allowed to leak contradictory labels across splits.
    """
    required = {"tid", "toi", "tfopwg_disp", *FEATURE_COLUMNS}
    missing = required - set(catalogue.columns)
    if missing:
        raise ValueError(f"NASA TOI table is missing expected columns: {sorted(missing)}")
    work = catalogue.copy()
    work["tid"] = pd.to_numeric(work["tid"], errors="coerce")
    work["pl_orbper"] = pd.to_numeric(work["pl_orbper"], errors="coerce")
    disposition = work["tfopwg_disp"].astype(str).str.upper().str.strip()
    work["label"] = np.where(disposition.isin(POSITIVE_DISPOSITIONS), 1, np.where(disposition.isin(NEGATIVE_DISPOSITIONS), 0, np.nan))
    work = work.dropna(subset=["tid", "toi", "label", "pl_orbper"])
    work = work.loc[work["pl_orbper"] > 0].copy()
    work["tid"] = work["tid"].astype(int)
    conflicting = work.groupby("tid")["label"].nunique()
    work = work.loc[~work["tid"].isin(conflicting[conflicting > 1].index)].copy()
    if excluded_tic_ids:
        work = work.loc[~work["tid"].isin(excluded_tic_ids)].copy()
    # One deterministically chosen candidate row per TIC prevents stellar
    # properties from a multi-TOI host appearing in more than one split.
    work = work.sort_values(["tid", "toi"], kind="stable").drop_duplicates("tid", keep="first")
    positives = work.loc[work["label"] == 1].copy() if per_class is None else _representative_sample(work.loc[work["label"] == 1], per_class, seed)
    negatives = work.loc[work["label"] == 0].copy() if per_class is None else _representative_sample(work.loc[work["label"] == 0], per_class, seed + 1)
    selected = pd.concat([positives, negatives], ignore_index=True).sort_values(["label", "tid"], kind="stable", ignore_index=True)
    manifest = selected[["tid", "toi", "tfopwg_disp", "label"]].rename(columns={"tid": "tic_id"})
    manifest["label_source"] = "NASA Exoplanet Archive TOI tfopwg_disp"
    features = selected[["tid", "toi", *FEATURE_COLUMNS, "label"]].rename(columns={"tid": "tic_id"})
    return manifest, features


def build_catalogue_benchmark(
    output_directory: Path | str,
    per_class: int | None = 500,
    excluded_tic_ids: set[int] | None = None,
) -> dict[str, object]:
    """Create a NASA-only benchmark and evaluate a fresh held-out TIC split."""
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    source = download_nasa_toi_table(directory / "nasa_toi_snapshot.csv")
    manifest, features = select_labelled_toi_targets(source, per_class, excluded_tic_ids=excluded_tic_ids)
    manifest.to_csv(directory / "catalogue_manifest.csv", index=False)
    features.to_csv(directory / "catalogue_features.csv", index=False)
    (directory / "feature_schema.json").write_text(json.dumps({
        "source_table": "NASA Exoplanet Archive TOI table",
        "feature_columns": list(FEATURE_COLUMNS),
        "excluded_from_features": ["tid", "toi", "tfopwg_disp", "toidisplay", "toipfx", "ctoi_alias", "toi_created", "rowupdate", "release_date", "pl_pnum"],
        "label_definition": {"planet_like": sorted(POSITIVE_DISPOSITIONS), "false_positive_like": sorted(NEGATIVE_DISPOSITIONS)},
        "caveat": "This is a classifier of published catalogue measurements, not independent light-curve transit recovery or planet confirmation.",
    }, indent=2) + "\n", encoding="utf-8")
    model_report = train_models(
        features,
        directory / "models",
        BenchmarkConfig(),
        feature_names=FEATURE_COLUMNS,
        strong_class_minimum_purity=STRONG_CLASS_MINIMUM_VALIDATION_PURITY,
    )
    thresholds = model_report["validation_strong_class_thresholds"]
    selected_name = str(model_report["selected_ranking_model"])
    scored = rank_features(features.drop(columns="label"), directory / "models" / f"{selected_name}.joblib")
    source_index = source.copy()
    source_index["tid"] = pd.to_numeric(source_index["tid"], errors="coerce")
    source_index["toi"] = pd.to_numeric(source_index["toi"], errors="coerce")
    source_by_key = {
        (int(row.tid), float(row.toi)): row
        for _, row in source_index.dropna(subset=["tid", "toi"]).iterrows()
    }
    index_targets = [
        _candidate_record(row, source_by_key[(int(row.tic_id), float(row.toi))], thresholds)
        for _, row in scored.iterrows()
    ]
    target_index = {
        "source": "NASA Exoplanet Archive TOI table",
        "selected_model": selected_name,
        "strong_classification_thresholds": thresholds,
        "interpretation": "The score estimates agreement with the CP/KP side of this catalogue benchmark. It is not a physical confirmation probability and does not establish that a candidate is an exoplanet.",
        "targets": index_targets,
    }
    (directory / "catalogue_target_index.json").write_text(json.dumps(target_index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    pd.DataFrame([{
        "tic_id": target["tic_id"], "toi": target["toi"],
        "benchmark_class": target["benchmark_class"],
        "planet_like_benchmark_score": target["planet_like_benchmark_score"],
        "actual_catalogue_disposition": target["actual_catalogue_disposition"],
    } for target in index_targets]).to_csv(directory / "catalogue_scoreboard.csv", index=False)
    report = {
        "source": "NASA Exoplanet Archive TOI table",
        "source_query": "select * from toi",
        "unit": "one deterministic TOI row per unique TIC target",
        "selection": "all eligible TICs after exclusions" if per_class is None else f"{per_class} deterministic period-stratified TICs per class",
        "targets": len(features),
        "positive_targets": int((features["label"] == 1).sum()),
        "negative_targets": int((features["label"] == 0).sum()),
        "feature_columns": list(FEATURE_COLUMNS),
        "label_handling": "tfopwg_disp is stored only in the separate manifest; it is not a model feature and is revealed only during evaluation.",
        "caveat": "Held-out metrics measure agreement with trusted NASA dispositions using catalogue values, not independent planet detection from photometry.",
        "excluded_prior_test_tics": len(excluded_tic_ids or set()),
        "model_report": model_report,
    }
    (directory / "catalogue_benchmark_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def assess_catalogue_tic(tic_id: int, benchmark_directory: Path | str = "outputs/catalogue_benchmark_final") -> dict[str, object]:
    """Score published TOI measurements for a TIC without consulting its label.

    A TIC can have multiple TOI rows, so the response retains one assessment
    per published candidate. ``tfopwg_disp`` is deliberately absent from the
    returned data and is never passed into the model.
    """
    directory = Path(benchmark_directory)
    source_path = directory / "nasa_toi_snapshot.csv"
    report_path = directory / "models" / "model_report.json"
    if not source_path.exists() or not report_path.exists():
        raise ExoSignalError("Catalogue benchmark outputs are missing; run `exosignal catalogue-benchmark` first.")
    index_path = directory / "catalogue_target_index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        candidates = [candidate for candidate in index["targets"] if int(candidate["tic_id"]) == int(tic_id)]
        if not candidates:
            raise ExoSignalError(f"TIC {tic_id} is not in this fixed benchmark catalogue.")
        assessment = {
            "tic_id": int(tic_id), "source": index["source"],
            "selected_model": index["selected_model"],
            "strong_classification_thresholds": index["strong_classification_thresholds"],
            "interpretation": index["interpretation"], "candidates": candidates,
        }
        output = directory / "assessments" / str(tic_id)
        output.mkdir(parents=True, exist_ok=True)
        (output / "catalogue_assessment.json").write_text(json.dumps(assessment, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return {**assessment, "output": str(output / "catalogue_assessment.json")}
    source = pd.read_csv(source_path)
    rows = source.loc[pd.to_numeric(source["tid"], errors="coerce") == int(tic_id)].copy()
    rows = rows.dropna(subset=["toi"])
    if rows.empty:
        raise ExoSignalError(f"TIC {tic_id} has no public NASA TOI row. A catalogue-only classifier cannot assess a star without published transit-candidate measurements.")
    features = rows[["tid", "toi", *FEATURE_COLUMNS]].rename(columns={"tid": "tic_id"})
    # ``rank_features`` sorts and resets its DataFrame index, so retain the
    # original catalogue location as data for the post-score provenance reveal.
    features["catalogue_source_index"] = rows.index.to_numpy()
    model_report = json.loads(report_path.read_text(encoding="utf-8"))
    selected_name = str(model_report["selected_ranking_model"])
    model_path = directory / "models" / f"{selected_name}.joblib"
    scored = rank_features(features, model_path)
    thresholds = model_report.get("validation_strong_class_thresholds")
    if not isinstance(thresholds, dict):
        raise ExoSignalError("Catalogue model is missing validation-derived strong-class thresholds; rebuild the benchmark.")
    candidates = [_candidate_record(row, rows.loc[int(row["catalogue_source_index"])], thresholds) for _, row in scored.iterrows()]
    assessment = {
        "tic_id": int(tic_id),
        "source": "NASA Exoplanet Archive TOI table",
        "selected_model": selected_name,
        "strong_classification_thresholds": thresholds,
        "interpretation": "The score estimates agreement with the CP/KP side of this catalogue benchmark. The three UI classes use validation-derived strong-class thresholds. It is not a physical confirmation probability and does not establish that a candidate is an exoplanet.",
        "candidates": candidates,
    }
    output = directory / "assessments" / str(tic_id)
    output.mkdir(parents=True, exist_ok=True)
    (output / "catalogue_assessment.json").write_text(json.dumps(assessment, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {**assessment, "output": str(output / "catalogue_assessment.json")}
