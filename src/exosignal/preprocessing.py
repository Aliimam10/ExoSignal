"""Sector-aware, transit-conservative preprocessing of PDCSAP measurements."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

from .config import PreprocessingConfig
from .errors import UnusableLightCurveError
from .models import ProcessedSector


def _robust_sigma(values: np.ndarray) -> float:
    median = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - median))
    return float(1.4826 * mad)


def _segments_from_gaps(time: np.ndarray, gap_break_days: float) -> list[slice]:
    """Return continuous pieces; detrending must never bridge major gaps."""
    boundaries = np.flatnonzero(np.diff(time) > gap_break_days) + 1
    starts = np.r_[0, boundaries]
    stops = np.r_[boundaries, len(time)]
    return [slice(int(start), int(stop)) for start, stop in zip(starts, stops)]


def _odd_at_least(value: int) -> int:
    return value if value % 2 else value + 1


def _running_median_trend(
    time: np.ndarray, flux: np.ndarray, config: PreprocessingConfig
) -> np.ndarray:
    """Estimate a long baseline within continuous pieces of one sector."""
    trend = np.empty_like(flux, dtype=float)
    for segment in _segments_from_gaps(time, config.gap_break_days):
        segment_time = time[segment]
        segment_flux = flux[segment]
        if len(segment_flux) < config.min_segment_points:
            trend[segment] = np.nanmedian(segment_flux)
            continue

        cadence_days = np.nanmedian(np.diff(segment_time))
        window_points = _odd_at_least(
            max(3, int(round(config.detrend_window_days / cadence_days)))
        )
        # A segment shorter than the intended window is assigned its robust
        # baseline rather than filtering across an observational boundary.
        if window_points >= len(segment_flux):
            trend[segment] = np.nanmedian(segment_flux)
        else:
            trend[segment] = median_filter(
                segment_flux, size=window_points, mode="nearest"
            )
    return trend


def process_sector(
    *,
    sector: int,
    time: Iterable[float],
    flux: Iterable[float],
    flux_err: Iterable[float] | None,
    quality: Iterable[int] | None,
    config: PreprocessingConfig | None = None,
    transit_mask: Iterable[bool] | None = None,
) -> ProcessedSector:
    """Clean and detrend exactly one SPOC PDCSAP sector.

    ``transit_mask`` is reserved for the transit-aware second detrending pass
    planned after a search algorithm exists; passing it currently raises an
    error rather than silently claiming masked processing occurred.
    """
    config = config or PreprocessingConfig()
    if transit_mask is not None:
        raise NotImplementedError("Transit masking will be added with Commit 2 search.")

    time_array = np.asarray(time, dtype=float)
    flux_array = np.asarray(flux, dtype=float)
    if len(time_array) != len(flux_array):
        raise ValueError("time and flux must have the same length")
    if flux_err is None:
        error_array = np.full(len(time_array), np.nan)
    else:
        error_array = np.asarray(flux_err, dtype=float)
        if len(error_array) != len(time_array):
            raise ValueError("flux_err must match time length")
    if quality is None:
        quality_array = np.zeros(len(time_array), dtype=int)
    else:
        quality_array = np.asarray(quality)
        if len(quality_array) != len(time_array):
            raise ValueError("quality must match time length")

    original_count = len(time_array)
    finite = np.isfinite(time_array) & np.isfinite(flux_array)
    quality_ok = quality_array == 0
    keep = finite & quality_ok
    removed_nonfinite = int((~finite).sum())
    removed_quality = int((finite & ~quality_ok).sum())

    cleaned = pd.DataFrame(
        {
            "time": time_array[keep],
            "pdcsap_flux": flux_array[keep],
            "pdcsap_flux_err": error_array[keep],
        }
    ).sort_values("time", ignore_index=True)
    if len(cleaned) < config.min_segment_points:
        raise UnusableLightCurveError(
            f"Sector {sector} has only {len(cleaned)} usable cadences after filtering."
        )

    median_flux = float(np.nanmedian(cleaned["pdcsap_flux"]))
    if not np.isfinite(median_flux) or median_flux <= 0:
        raise UnusableLightCurveError(f"Sector {sector} has an invalid PDCSAP baseline.")

    normalized = cleaned["pdcsap_flux"].to_numpy() / median_flux
    sigma = _robust_sigma(normalized)
    if not np.isfinite(sigma) or sigma == 0:
        # Perfectly flat test/engineering data have zero MAD. Falling back to
        # standard deviation still permits an obvious positive discontinuity to
        # be excluded, while the deliberately high 8-sigma threshold remains
        # conservative for real light curves.
        sigma = float(np.nanstd(normalized))
    # Deliberately retain downward excursions: at this stage they may be real
    # transits. Strong upward one-cadence events are generally instrumental and
    # can bias the long-timescale baseline.
    if np.isfinite(sigma) and sigma > 0:
        positive_outlier = normalized - np.nanmedian(normalized) > (
            config.positive_outlier_sigma * sigma
        )
    else:
        positive_outlier = np.zeros(len(cleaned), dtype=bool)
    cleaned = cleaned.loc[~positive_outlier].reset_index(drop=True)
    if len(cleaned) < config.min_segment_points:
        raise UnusableLightCurveError(
            f"Sector {sector} has too few cadences after conservative outlier filtering."
        )

    values = cleaned["pdcsap_flux"].to_numpy()
    trend = _running_median_trend(cleaned["time"].to_numpy(), values, config)
    valid_trend = np.isfinite(trend) & (trend > 0)
    cleaned = cleaned.loc[valid_trend].reset_index(drop=True)
    trend = trend[valid_trend]
    if len(cleaned) < config.min_segment_points:
        raise UnusableLightCurveError(f"Sector {sector} has an unusable trend estimate.")

    raw_normalized = cleaned["pdcsap_flux"].to_numpy() / median_flux - 1.0
    relative_flux = cleaned["pdcsap_flux"].to_numpy() / trend - 1.0
    relative_error = cleaned["pdcsap_flux_err"].to_numpy() / trend
    data = pd.DataFrame(
        {
            "time": cleaned["time"],
            "flux": relative_flux,
            "flux_err": relative_error,
            "raw_normalized_flux": raw_normalized,
            "sector": int(sector),
        }
    )
    diagnostics = {
        "sector": int(sector),
        "input_cadences": original_count,
        "removed_nonfinite": removed_nonfinite,
        "removed_nonzero_quality": removed_quality,
        "removed_positive_outliers": int(positive_outlier.sum()),
        "output_cadences": len(data),
        "sector_median_pdcsap_flux": median_flux,
        "continuous_segments": len(_segments_from_gaps(data["time"].to_numpy(), config.gap_break_days)),
    }
    return ProcessedSector(sector=int(sector), data=data, diagnostics=diagnostics)


def combine_sectors(sectors: Iterable[ProcessedSector]) -> pd.DataFrame:
    """Combine already processed sectors without erasing sector identity."""
    frames = [item.data for item in sectors]
    if not frames:
        return pd.DataFrame(
            columns=["time", "flux", "flux_err", "raw_normalized_flux", "sector"]
        )
    return pd.concat(frames, ignore_index=True).sort_values("time", ignore_index=True)
