"""Trusted-label benchmark construction with explicit target attrition."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .catalogue import reliable_label_manifest
from .config import PreprocessingConfig, SearchConfig, VettingConfig
from .errors import NoTessDataError
from .features import features_from_target_run
from .pipeline import analyse_target


def _representative_period_sample(frame: pd.DataFrame, maximum: int, seed: int) -> pd.DataFrame:
    """Deterministically sample across log-period quantiles, not shortest P."""
    rng = np.random.default_rng(seed)
    frame = frame.copy()
    frame["_log_period"] = np.log10(frame["catalogue_period_days"])
    bins = min(6, len(frame), frame["_log_period"].nunique())
    if bins < 2:
        return frame.sample(n=min(maximum, len(frame)), random_state=seed)
    frame["_period_bin"] = pd.qcut(frame["_log_period"], q=bins, duplicates="drop")
    selected: list[pd.DataFrame] = []
    per_bin, remainder = divmod(maximum, frame["_period_bin"].nunique())
    for _, group in frame.groupby("_period_bin", observed=True):
        take = min(len(group), per_bin + (1 if remainder > 0 else 0))
        remainder = max(0, remainder - 1)
        selected.append(group.iloc[rng.permutation(len(group))[:take]])
    result = pd.concat(selected, ignore_index=True)
    if len(result) < maximum:
        remaining = frame.loc[~frame["tic_id"].isin(result["tic_id"])]
        take = min(maximum - len(result), len(remaining))
        result = pd.concat([result, remaining.iloc[rng.permutation(len(remaining))[:take]]], ignore_index=True)
    return result.drop(columns=["_log_period", "_period_bin"], errors="ignore")


def benchmark_manifest(catalogue: pd.DataFrame, maximum_per_class: int, seed: int = 20261005) -> pd.DataFrame:
    """Create a fixed, period-representative label manifest for the benchmark."""
    labels = reliable_label_manifest(catalogue, None)
    periods = catalogue[["TIC ID", "Period (days)"]].copy()
    periods["Period (days)"] = pd.to_numeric(periods["Period (days)"], errors="coerce")
    periods = periods.loc[periods["Period (days)"] > 0].drop_duplicates("TIC ID")
    result = labels.merge(periods, left_on="tic_id", right_on="TIC ID", how="inner").drop(columns="TIC ID")
    result = result.rename(columns={"Period (days)": "catalogue_period_days"})
    selected = pd.concat(
        [_representative_period_sample(group, maximum_per_class, seed + int(label))
         for label, group in result.groupby("label", sort=True)],
        ignore_index=True,
    )
    # Interleave the independently sorted classes so every deterministic batch
    # has comparable positive/negative representation; this affects retrieval
    # order only, never labels, splits, or model inputs.
    groups = [group.reset_index(drop=True) for _, group in selected.groupby("label", sort=True)]
    rows = []
    for index in range(max(len(group) for group in groups)):
        for group in groups:
            if index < len(group):
                rows.append(group.iloc[index].to_dict())
    return pd.DataFrame(rows)


def supplemental_negative_manifest(
    catalogue: pd.DataFrame, existing_tics: set[int], maximum: int, seed: int = 20261006
) -> pd.DataFrame:
    """Select additional, non-overlapping reliable FP/FA TICs by period strata."""
    labels = reliable_label_manifest(catalogue, None)
    periods = catalogue[["TIC ID", "Period (days)"]].copy()
    periods["Period (days)"] = pd.to_numeric(periods["Period (days)"], errors="coerce")
    periods = periods.loc[periods["Period (days)"] > 0].drop_duplicates("TIC ID")
    candidates = labels.merge(periods, left_on="tic_id", right_on="TIC ID", how="inner").drop(columns="TIC ID")
    candidates = candidates.loc[(candidates["label"] == 0) & ~candidates["tic_id"].astype(int).isin(existing_tics)]
    result = _representative_period_sample(candidates.rename(columns={"Period (days)": "catalogue_period_days"}), maximum, seed)
    return result.sort_values("tic_id", kind="stable", ignore_index=True)


def _period_match(found: float, truth: float, tolerance: float = 0.01) -> tuple[str, float] | None:
    """Return exact/half/double match using one fractional-error definition."""
    if not np.isfinite(found) or not np.isfinite(truth) or truth <= 0:
        return None
    for kind, expected, priority in (("exact", truth, 0), ("half_period", truth / 2, 1), ("double_period", truth * 2, 2)):
        error = abs(found - expected) / expected
        if error <= tolerance:
            return kind, error
    return None


def build_benchmark(
    manifest: pd.DataFrame,
    output_directory: Path | str,
    preprocessing_config: PreprocessingConfig | None = None,
    search_config: SearchConfig | None = None,
    vetting_config: VettingConfig | None = None,
    exclude_tic_ids: set[int] | None = None,
    maximum_targets: int | None = None,
    target_offset: int = 0,
    workers: int = 1,
) -> dict[str, Any]:
    """Process reliable labelled TICs and retain only recovered labelled signals.

    The public label is used here solely to evaluate whether ExoSignal recovered
    that known signal. It is appended after numeric feature extraction and is
    not part of the feature schema in :mod:`exosignal.features`.
    """
    exclude_tic_ids = exclude_tic_ids or set()
    manifest = manifest.loc[~manifest["tic_id"].astype(int).isin(exclude_tic_ids)].copy()
    manifest = manifest.iloc[target_offset:].copy()
    if maximum_targets is not None:
        manifest = manifest.head(maximum_targets).copy()
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    existing_features_path = directory / "benchmark_features.csv"
    existing_attrition_path = directory / "benchmark_attrition.csv"
    state_features = pd.read_csv(existing_features_path) if existing_features_path.exists() else pd.DataFrame()
    state_attrition = pd.read_csv(existing_attrition_path) if existing_attrition_path.exists() else pd.DataFrame()
    prior_outcomes = {
        int(record["tic_id"]): record
        for record in state_attrition.to_dict("records")
    }

    def checkpoint(feature: pd.DataFrame | None, outcome: dict[str, Any]) -> None:
        """Durably merge one finished TIC before the next worker result."""
        nonlocal state_features, state_attrition
        tic_id = int(outcome["tic_id"])
        if not state_features.empty:
            state_features = state_features.loc[state_features["tic_id"].astype(int) != tic_id]
        if feature is not None:
            state_features = pd.concat([state_features, feature], ignore_index=True)
        if not state_features.empty:
            state_features = state_features.sort_values(["tic_id", "candidate_id"], kind="stable", ignore_index=True)
        if not state_attrition.empty:
            state_attrition = state_attrition.loc[state_attrition["tic_id"].astype(int) != tic_id]
        state_attrition = pd.concat([state_attrition, pd.DataFrame([outcome])], ignore_index=True)
        state_attrition = state_attrition.sort_values("tic_id", kind="stable", ignore_index=True)
        state_features.to_csv(existing_features_path, index=False)
        state_attrition.to_csv(existing_attrition_path, index=False)

    def process_one(item: dict[str, Any]) -> tuple[pd.DataFrame | None, dict[str, Any]]:
        tic_id = int(item["tic_id"])
        # A completed target is immutable on resume. This prevents a transient
        # later MAST/TPF issue from replacing an already valid measurement.
        previous = prior_outcomes.get(tic_id)
        if previous is not None and previous.get("status") in {"usable", "no_candidate_recovered", "no_suitable_tess_data"}:
            existing = state_features.loc[state_features["tic_id"].astype(int) == tic_id].copy() if not state_features.empty else pd.DataFrame()
            return (existing if not existing.empty else None), previous
        try:
            candidate_directory = directory / "targets" / str(tic_id) / "candidates"
            candidate_file = candidate_directory / "candidates.json"
            vetted = False
            if candidate_file.exists():
                records = json.loads(candidate_file.read_text(encoding="utf-8"))
                vetted = all((candidate_directory / f"{record['candidate_id'].lower()}_vetting.json").exists() for record in records)
            if not candidate_file.exists() or not vetted:
                analyse_target(
                    tic_id, directory / "targets", preprocessing_config,
                    search_config or SearchConfig(tls_enabled=False, max_candidates=3), True,
                    vetting_config or VettingConfig(generate_pixel_plots=False, retrieve_pixel_data=True),
                    generate_plots=False,
                )
            features = features_from_target_run(tic_id, directory / "targets")
            matches = []
            for index, candidate in features.iterrows():
                match = _period_match(float(candidate["period_days"]), float(item["catalogue_period_days"]))
                if match is not None:
                    kind, error = match
                    matches.append(({"exact": 0, "half_period": 1, "double_period": 2}[kind], error, -float(candidate["bls_snr"]), str(candidate["candidate_id"]), index, kind))
            recovered = features.loc[[item[4] for item in sorted(matches)]].copy() if matches else pd.DataFrame()
            if recovered.empty:
                return None, {"tic_id": tic_id, "label": item["label"], "status": "no_candidate_recovered"}
            recovered = recovered.iloc[[0]].copy()
            selected = sorted(matches)[0]
            recovered["label"] = int(item["label"])
            return recovered, {"tic_id": tic_id, "label": item["label"], "status": "usable", "match_type": selected[5], "period_fractional_error": selected[1], "selected_candidate_id": selected[3]}
        except NoTessDataError as error:
            return None, {"tic_id": tic_id, "label": item["label"], "status": "no_suitable_tess_data", "reason": str(error)}
        except Exception as error:
            return None, {"tic_id": tic_id, "label": item["label"], "status": "pipeline_failure", "reason": str(error)}

    records = manifest.to_dict("records")
    if workers <= 1:
        results = map(process_one, records)
    else:
        # Independent TIC directories and per-target exception handling make a
        # modest thread pool safe for network-bound MAST retrieval. The sample,
        # labels, preprocessing, and search settings remain unchanged.
        executor = ThreadPoolExecutor(max_workers=workers)
        futures = [executor.submit(process_one, item) for item in records]
        results = (future.result() for future in as_completed(futures))
    for feature, outcome in results:
        checkpoint(feature, outcome)
    if workers > 1:
        executor.shutdown(wait=True)
    features = state_features
    attrition = state_attrition.to_dict("records")
    statuses = state_attrition["status"] if not state_attrition.empty else pd.Series(dtype=str)
    successfully_processed = int(statuses.isin(["usable", "no_candidate_recovered"]).sum())
    report = {
        "source_labelled_tics": len(attrition),
        "labelled_targets_requested": len(attrition),
        "current_batch_targets": len(manifest),
        "suitable_tess_data_found": successfully_processed,
        "successfully_processed": successfully_processed,
        "explicitly_excluded_tics": sorted(exclude_tic_ids),
        "usable_positive_examples": int((features.get("label", pd.Series(dtype=int)) == 1).sum()),
        "usable_negative_examples": int((features.get("label", pd.Series(dtype=int)) == 0).sum()),
        "attrition": statuses.value_counts().to_dict(),
        "detection_performance": {
            "definition": "A known labelled signal is recovered when an ExoSignal candidate has an exact, P/2, or 2P period within the same 1% fractional error tolerance. This is separate from ML classification.",
            "recovered_known_signals": int((statuses == "usable").sum()),
            "known_signal_not_recovered": int((statuses == "no_candidate_recovered").sum()),
            "exact_recoveries": int(state_attrition.get("match_type", pd.Series(dtype=str)).eq("exact").sum()),
            "half_period_recoveries": int(state_attrition.get("match_type", pd.Series(dtype=str)).eq("half_period").sum()),
            "double_period_recoveries": int(state_attrition.get("match_type", pd.Series(dtype=str)).eq("double_period").sum()),
        },
        "feature_table": str(directory / "benchmark_features.csv"),
    }
    (directory / "benchmark_attrition.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
