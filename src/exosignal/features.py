"""Numeric, catalogue-blind features measured only by ExoSignal."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .search import Candidate


# This is intentionally a short, inspectable list.  Catalogue disposition,
# names, TOI values, and vetting PASS/WARNING/FAIL strings are never features.
MODEL_FEATURES = [
    "period_days", "duration_hours", "depth", "depth_snr", "bls_snr",
    "observed_transits", "tls_sde", "tls_snr", "event_fractional_depth_scatter",
    "event_weak_fraction", "odd_even_depth_difference", "odd_even_difference_sigma",
    "secondary_depth", "secondary_sigma", "secondary_to_primary_ratio",
    "sector_max_depth_difference", "sector_difference_sigma", "out_of_transit_rms",
    "crowdsap", "pixel_available", "pixel_offset_pixels",
]


def _number(value: Any) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan
    return value if np.isfinite(value) else np.nan


def _values(dossier: dict[str, Any], name: str) -> dict[str, Any]:
    return dict(dossier.get(name, {}).get("values", {}))


def feature_row(tic_id: int, candidate: Candidate | dict[str, Any], dossier: dict[str, Any]) -> dict[str, Any]:
    """Return one numeric feature record; absent pixel values remain NaN.

    ``pixel_available`` is a measured availability indicator. The companion
    numerical offset remains missing when no centroid could be calculated and
    is median-imputed inside each model pipeline, never encoded as zero.
    """
    candidate_dict = candidate.to_dict() if isinstance(candidate, Candidate) else candidate
    individual = _values(dossier, "individual_transit_consistency")
    odd_even = _values(dossier, "odd_even")
    secondary = _values(dossier, "secondary_eclipse")
    sector = _values(dossier, "sector_consistency")
    variability = _values(dossier, "out_of_transit_variability")
    crowding = _values(dossier, "crowding")
    pixel = _values(dossier, "pixel_source_check")
    sector_results = pixel.get("sector_results", [])
    offsets = [_number(item.get("offset_pixels")) for item in sector_results]
    offsets = [value for value in offsets if np.isfinite(value)]
    depth = _number(candidate_dict.get("depth"))
    depth_err = _number(candidate_dict.get("depth_err"))
    secondary_depth = _number(secondary.get("secondary_depth"))
    return {
        "tic_id": int(tic_id),
        "candidate_id": str(candidate_dict.get("candidate_id", "")),
        "period_days": _number(candidate_dict.get("period_days")),
        "duration_hours": _number(candidate_dict.get("duration_hours")),
        "depth": depth,
        "depth_snr": depth / depth_err if np.isfinite(depth_err) and depth_err > 0 else np.nan,
        "bls_snr": _number(candidate_dict.get("bls_snr")),
        "observed_transits": _number(candidate_dict.get("observed_transits")),
        "tls_sde": _number(candidate_dict.get("tls_sde")),
        "tls_snr": _number(candidate_dict.get("tls_snr")),
        "event_fractional_depth_scatter": _number(individual.get("fractional_depth_scatter")),
        "event_weak_fraction": _number(individual.get("weak_or_absent_fraction")),
        "odd_even_depth_difference": _number(odd_even.get("depth_difference")),
        "odd_even_difference_sigma": _number(odd_even.get("difference_sigma")),
        "secondary_depth": secondary_depth,
        "secondary_sigma": _number(secondary.get("secondary_sigma")),
        "secondary_to_primary_ratio": secondary_depth / depth if np.isfinite(depth) and depth > 0 else np.nan,
        "sector_max_depth_difference": _number(sector.get("max_depth_difference")),
        "sector_difference_sigma": _number(sector.get("difference_sigma")),
        "out_of_transit_rms": _number(variability.get("out_of_transit_rms")),
        "crowdsap": _number(crowding.get("crowdsap")),
        "pixel_available": int(bool(offsets)),
        "pixel_offset_pixels": float(np.median(offsets)) if offsets else np.nan,
    }


def features_from_target_run(tic_id: int, output_root: Path | str = "outputs/targets") -> pd.DataFrame:
    """Read pipeline products and produce candidate-level numeric features."""
    directory = Path(output_root) / str(tic_id) / "candidates"
    candidates = json.loads((directory / "candidates.json").read_text(encoding="utf-8"))
    rows = []
    for candidate in candidates:
        path = directory / f"{candidate['candidate_id'].lower()}_vetting.json"
        if path.exists():
            rows.append(feature_row(tic_id, candidate, json.loads(path.read_text(encoding="utf-8"))))
    return pd.DataFrame(rows, columns=["tic_id", "candidate_id", *MODEL_FEATURES])
