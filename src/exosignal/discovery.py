"""Reproducible, catalogue-blind TESS-sector discovery runs."""

from __future__ import annotations

import json
import re
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .config import PreprocessingConfig, SearchConfig, VettingConfig
from .features import features_from_target_run
from .ml import rank_features
from .pipeline import analyse_target


def target_tic_id(value: object) -> int | None:
    """Parse both current bare MAST TICs and legacy ``TIC 123`` names."""
    text = str(value).strip()
    match = re.fullmatch(r"(\d+)", text) or re.search(r"TIC\s*(\d+)", text, re.IGNORECASE)
    return int(match.group(1)) if match else None


def select_sector_tics(sector: int, maximum_targets: int, tmag_limit: float | None = None, excluded_tic_ids: set[int] | None = None) -> tuple[list[int], dict[str, Any]]:
    """Select TICs deterministically from public MAST TESS observations.

    This does not inspect ExoFOP, known planet lists, or labels. Magnitude
    filtering is intentionally explicit: if requested, TIC metadata is queried
    and targets lacking a usable TESS magnitude are excluded.
    """
    from astroquery.mast import Catalogs, Observations

    observations = Observations.query_criteria(obs_collection="TESS", sequence_number=int(sector), dataproduct_type="timeseries")
    names = observations["target_name"] if "target_name" in observations.colnames else []
    candidates = set()
    for name in names:
        # MAST presently returns bare numeric TIC IDs for this endpoint, but
        # older product listings may use "TIC 123".  Accept both forms.
        tic_id = target_tic_id(name)
        if tic_id is not None:
            candidates.add(tic_id)
    excluded_tic_ids = excluded_tic_ids or set()
    selected: list[int] = []
    for tic_id in sorted(candidates):
        if tic_id in excluded_tic_ids:
            continue
        if tmag_limit is not None:
            try:
                tic = Catalogs.query_criteria(catalog="Tic", ID=tic_id)
                tmag = float(tic[0]["Tmag"])
            except Exception:
                continue
            if tmag > tmag_limit:
                continue
        selected.append(tic_id)
        if len(selected) >= maximum_targets:
            break
    return selected, {"mast_observations_returned": len(observations), "unique_tics_before_filter": len(candidates), "training_tics_excluded": len(excluded_tic_ids)}


def run_discovery(
    sector: int,
    maximum_targets: int,
    model_path: Path | str,
    output_directory: Path | str = "outputs/discovery",
    tmag_limit: float | None = None,
    preprocessing_config: PreprocessingConfig | None = None,
    search_config: SearchConfig | None = None,
    vetting_config: VettingConfig | None = None,
    additional_excluded_tic_ids: set[int] | None = None,
) -> dict[str, Any]:
    """Search unseen TICs, freeze ranking, and deliberately do no catalogue lookup."""
    directory = Path(output_directory) / f"sector_{int(sector):03d}"
    directory.mkdir(parents=True, exist_ok=True)
    split_file = Path(model_path).parent / "benchmark_features_with_splits.csv"
    training_tics: set[int] = set()
    if split_file.exists():
        training_tics = set(pd.read_csv(split_file)["tic_id"].astype(int))
    manifest_tics = additional_excluded_tic_ids or set()
    excluded_tics = training_tics | manifest_tics
    selected, selection = select_sector_tics(sector, maximum_targets, tmag_limit, excluded_tics)
    parameters = {
        "sector": int(sector), "maximum_targets": int(maximum_targets), "tmag_limit": tmag_limit,
        "selection_order": "ascending TIC ID from public MAST timeseries records",
        "product": "one SPOC PDCSAP product per sector, preferring 120-second cadence",
        "catalogue_lookup_before_freeze": False,
        "experiment_type": "blind discovery workflow demonstration; not a discovery-performance benchmark",
        "unseen_target_safeguard": "All TICs in the model's persisted benchmark split and supplied labelled manifests are excluded.",
        "label_manifest_tics_excluded": len(manifest_tics),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        **selection,
    }
    manifest: list[dict[str, Any]] = []
    features: list[pd.DataFrame] = []
    for tic_id in selected:
        try:
            summary = analyse_target(tic_id, directory / "targets", preprocessing_config, search_config or SearchConfig(tls_enabled=False), True, vetting_config or VettingConfig(generate_pixel_plots=False))
            rows = features_from_target_run(tic_id, directory / "targets")
            if rows.empty:
                manifest.append({"tic_id": tic_id, "status": "no_candidate_recovered", "sectors": summary["processed_sectors"]})
            else:
                features.append(rows)
                manifest.append({"tic_id": tic_id, "status": "success", "sectors": summary["processed_sectors"], "candidates": len(rows)})
        except Exception as error:
            manifest.append({"tic_id": tic_id, "status": "failed", "reason": str(error)})
    pd.DataFrame(manifest).to_csv(directory / "target_manifest.csv", index=False)
    all_features = pd.concat(features, ignore_index=True) if features else pd.DataFrame()
    if all_features.empty:
        ranking = all_features
    else:
        ranking = rank_features(all_features, model_path)
        candidate_rows = []
        for tic_id in ranking["tic_id"].unique():
            candidates = json.loads((directory / "targets" / str(tic_id) / "candidates" / "candidates.json").read_text())
            candidate_rows.extend({"tic_id": int(tic_id), "candidate_id": item["candidate_id"], "epoch_btjd": item["epoch_btjd"]} for item in candidates)
        ranking = ranking.merge(pd.DataFrame(candidate_rows), on=["tic_id", "candidate_id"], how="left")
    # This is the immutable pre-catalogue evidence record. Crossmatching is a
    # separate command/function and never mutates this file. Its SHA-256
    # fingerprint records exactly what was frozen before any reveal.
    ranking_path = directory / "pre_crossmatch_ranking.csv"
    ranking.to_csv(ranking_path, index=False)
    fingerprint = hashlib.sha256(ranking_path.read_bytes()).hexdigest()
    (directory / "pre_crossmatch_ranking.sha256").write_text(f"{fingerprint}  {ranking_path.name}\n", encoding="utf-8")
    parameters["targets_attempted"] = len(selected)
    parameters["signals_before_catalogue_lookup"] = len(ranking)
    parameters["pre_crossmatch_ranking_sha256"] = fingerprint
    (directory / "discovery_parameters.json").write_text(json.dumps(parameters, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"directory": str(directory), "parameters": parameters, "manifest": manifest, "pre_crossmatch_ranking": str(ranking_path)}
