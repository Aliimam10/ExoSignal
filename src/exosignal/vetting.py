"""Compact, evidence-preserving astrophysical vetting for transit candidates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from contextlib import contextmanager
from pathlib import Path
import signal
import threading
from typing import Any

import numpy as np
import pandas as pd

from .config import VettingConfig
from .search import Candidate, candidate_transit_mask
from .tess import selected_product_indices


@dataclass
class Diagnostic:
    """A verdict always accompanied by its measurement and criterion."""

    status: str
    values: dict[str, object]
    criterion: str
    explanation: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class PixelSectorMeasurement:
    sector: int
    status: str
    interpretation: str
    expected_position_xy: tuple[float, float] | None
    source_position_xy: tuple[float, float] | None
    offset_pixels: float | None
    offset_arcsec: float | None
    offset_significance: float | None
    in_cadences: int
    out_cadences: int
    reason: str | None = None
    out_image: np.ndarray | None = None
    in_image: np.ndarray | None = None
    difference_image: np.ndarray | None = None

    def summary(self) -> dict[str, object]:
        result = asdict(self)
        for image_key in ("out_image", "in_image", "difference_image"):
            result.pop(image_key)
        return result


def _robust_std(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return np.nan
    return float(1.4826 * np.median(np.abs(values - np.median(values))))


def _event_depth(inner: np.ndarray, outer: np.ndarray) -> tuple[float, float, float]:
    """Return local depth, conservative uncertainty, and SNR."""
    if len(inner) < 2 or len(outer) < 2:
        return np.nan, np.nan, np.nan
    depth = float(np.nanmedian(outer) - np.nanmedian(inner))
    scatter = _robust_std(np.r_[inner, outer])
    uncertainty = scatter * np.sqrt(1 / len(inner) + 1 / len(outer))
    if np.isfinite(uncertainty) and uncertainty > 0:
        snr = depth / uncertainty
    elif np.isfinite(depth) and depth > 0:
        snr = np.inf
    else:
        snr = np.nan
    return depth, uncertainty, snr


def individual_transit_events(
    data: pd.DataFrame, candidate: Candidate, config: VettingConfig
) -> pd.DataFrame:
    """Measure predicted events without treating inter-sector gaps as misses.

    An epoch wholly outside the local observing windows is recorded for audit,
    but excluded from every event-consistency denominator.  An epoch that
    overlaps observations yet lacks enough usable cadences is distinct: that
    is an ``insufficient_usable_cadence`` event, not evidence against a
    transit.
    """
    time = data["time"].to_numpy(dtype=float)
    event_start = int(np.ceil((time.min() - candidate.epoch_btjd) / candidate.period_days))
    event_stop = int(np.floor((time.max() - candidate.epoch_btjd) / candidate.period_days))
    records: list[dict[str, object]] = []
    duration_days = candidate.duration_hours / 24.0
    for number in range(event_start, event_stop + 1):
        midtime = candidate.epoch_btjd + number * candidate.period_days
        relative = time - midtime
        inner_mask = np.abs(relative) <= duration_days / 2
        outer_mask = (np.abs(relative) >= duration_days) & (np.abs(relative) <= 3 * duration_days)
        local_coverage = np.abs(relative) <= 3 * duration_days
        inner = data.loc[inner_mask, "flux"].to_numpy(dtype=float)
        outer = data.loc[outer_mask, "flux"].to_numpy(dtype=float)
        inner_count, outer_count = len(inner), len(outer)
        sector = None
        if local_coverage.any():
            sector_values = pd.to_numeric(data.loc[local_coverage, "sector"], errors="coerce")
            sector_values = sector_values.loc[np.isfinite(sector_values)]
            if not sector_values.empty:
                mode = float(sector_values.mode().iloc[0])
                if mode.is_integer():
                    sector = int(mode)
        if not local_coverage.any():
            category = "outside_observing_coverage"
            depth = uncertainty = snr = np.nan
        elif inner_count < config.minimum_event_inner_points or outer_count < config.minimum_event_outer_points:
            category = "insufficient_usable_cadence"
            depth = uncertainty = snr = np.nan
        else:
            depth, uncertainty, snr = _event_depth(inner, outer)
            if np.isfinite(snr) and snr >= config.event_detection_snr:
                category = "detected"
            elif np.isfinite(depth) and depth > 0:
                category = "noisy_consistent"
            else:
                category = "weak_or_absent"
        records.append(
            {
                "event_number": number,
                "predicted_midtime_btjd": midtime,
                "sector": sector,
                "inner_cadences": inner_count,
                "outer_cadences": outer_count,
                "depth": depth,
                "depth_uncertainty": uncertainty,
                "depth_snr": snr,
                "category": category,
            }
        )
    return pd.DataFrame(records)


def individual_consistency(events: pd.DataFrame, config: VettingConfig) -> Diagnostic:
    measurable_categories = {"detected", "noisy_consistent", "weak_or_absent"}
    observable = events.loc[events["category"].isin(measurable_categories)].copy()
    outside_coverage = int((events["category"] == "outside_observing_coverage").sum())
    insufficient = int((events["category"] == "insufficient_usable_cadence").sum())
    weak = int((observable["category"] == "weak_or_absent").sum())
    detected = int((observable["category"] == "detected").sum())
    noisy = int((observable["category"] == "noisy_consistent").sum())
    if len(observable) == 0:
        return Diagnostic(
            "NOT_AVAILABLE",
            {"predicted_events": len(events), "outside_observing_coverage_events": outside_coverage,
             "insufficient_usable_cadence_events": insufficient, "observable_events": 0},
            "At least one adequately sampled predicted transit is required.",
            "Every predicted event falls in a data gap or lacks sufficient local baseline.",
        )
    depths = observable["depth"].to_numpy(dtype=float)
    median_depth = float(np.nanmedian(depths))
    depth_scatter = _robust_std(depths)
    fractional_scatter = depth_scatter / median_depth if median_depth > 0 else np.nan
    weak_fraction = weak / len(observable)
    values = {
        "predicted_events": len(events), "observable_events": len(observable),
        "outside_observing_coverage_events": outside_coverage,
        "insufficient_usable_cadence_events": insufficient, "detected_events": detected,
        "noisy_consistent_events": noisy, "weak_or_absent_events": weak,
        "weak_or_absent_fraction": weak_fraction, "median_event_depth": median_depth,
        "event_depth_scatter": depth_scatter, "fractional_depth_scatter": fractional_scatter,
    }
    if weak_fraction >= config.weak_event_warning_fraction:
        return Diagnostic(
            "WARNING", values,
            f"Warning when weak/absent events are at least {config.weak_event_warning_fraction:.0%} of observable events.",
            "Many adequately observed predicted events are weak or absent; non-covered and insufficient-cadence events were not penalized.",
        )
    return Diagnostic(
        "PASS", values,
        f"Warning when weak/absent events are at least {config.weak_event_warning_fraction:.0%} of observable events.",
        "Observable events are predominantly detected or noisy but consistent; non-covered and insufficient-cadence events are excluded.",
    )


def odd_even_diagnostic(events: pd.DataFrame, config: VettingConfig) -> Diagnostic:
    observed = events.loc[events["category"].isin({"detected", "noisy_consistent", "weak_or_absent"})].copy()
    groups = []
    for parity, label in ((0, "even"), (1, "odd")):
        values = observed.loc[observed["event_number"] % 2 == parity, "depth"].dropna().to_numpy()
        if len(values) < 2:
            return Diagnostic(
                "NOT_AVAILABLE", {"available_events": len(observed), "missing_group": label},
                "At least two measurable odd and even events are required.",
                "There are insufficient measurable events to compare odd and even depths.",
            )
        groups.append((label, float(np.nanmedian(values)), _robust_std(values) / np.sqrt(len(values)), len(values)))
    even, odd = groups
    difference = abs(even[1] - odd[1])
    uncertainty = np.sqrt(even[2] ** 2 + odd[2] ** 2)
    significance = difference / uncertainty if uncertainty > 0 else np.nan
    values = {
        "even_depth": even[1], "odd_depth": odd[1], "depth_difference": difference,
        "difference_sigma": significance, "even_events": even[3], "odd_events": odd[3],
    }
    if not np.isfinite(significance):
        status = "NOT_AVAILABLE"
    elif significance >= config.odd_even_fail_sigma:
        status = "FAIL"
    elif significance >= config.odd_even_warning_sigma:
        status = "WARNING"
    else:
        status = "PASS"
    return Diagnostic(
        status, values,
        f"WARNING at {config.odd_even_warning_sigma:g}σ; FAIL at {config.odd_even_fail_sigma:g}σ.",
        "A significant odd/even depth difference can indicate an eclipsing binary; agreement does not confirm a planet.",
    )


def secondary_eclipse_diagnostic(data: pd.DataFrame, candidate: Candidate, config: VettingConfig) -> Diagnostic:
    time = data["time"].to_numpy(dtype=float)
    phase = (time - candidate.epoch_btjd) % candidate.period_days / candidate.period_days
    duration_phase = (candidate.duration_hours / 24.0) / candidate.period_days
    distance = np.abs(phase - 0.5)
    inner = data.loc[distance <= duration_phase / 2, "flux"].to_numpy(dtype=float)
    outer = data.loc[(distance >= duration_phase) & (distance <= 3 * duration_phase), "flux"].to_numpy(dtype=float)
    depth, uncertainty, significance = _event_depth(inner, outer)
    values = {"secondary_depth": depth, "secondary_depth_uncertainty": uncertainty, "secondary_sigma": significance, "in_cadences": len(inner), "out_cadences": len(outer)}
    if len(inner) < config.minimum_event_inner_points or len(outer) < config.minimum_event_outer_points:
        status = "NOT_AVAILABLE"
    elif significance >= config.secondary_fail_sigma:
        status = "FAIL"
    elif significance >= config.secondary_warning_sigma:
        status = "WARNING"
    else:
        status = "PASS"
    return Diagnostic(
        status, values,
        f"WARNING at {config.secondary_warning_sigma:g}σ; FAIL at {config.secondary_fail_sigma:g}σ for a phase-0.5 flux dip.",
        "A measurable secondary eclipse can indicate stellar self-luminosity or an eclipsing binary; non-detection is not proof of planetary nature.",
    )


def sector_consistency(events: pd.DataFrame, config: VettingConfig) -> Diagnostic:
    observable = events.loc[
        events["category"].isin({"detected", "noisy_consistent", "weak_or_absent"}) & events["sector"].notna()
    ].copy()
    groups = []
    for sector, group in observable.groupby("sector"):
        depths = group["depth"].dropna().to_numpy(dtype=float)
        if len(depths) >= 2:
            groups.append((int(sector), float(np.median(depths)), _robust_std(depths) / np.sqrt(len(depths)), len(depths)))
    if len(groups) < 2:
        return Diagnostic(
            "NOT_AVAILABLE", {"sectors_with_measurable_events": len(groups)},
            "At least two sectors with two measurable events each are required.",
            "Multi-sector depth consistency cannot be assessed from the available events.",
        )
    depths = np.array([group[1] for group in groups])
    uncertainties = np.array([group[2] for group in groups])
    difference = float(depths.max() - depths.min())
    combined_uncertainty = float(np.sqrt(uncertainties[np.argmax(depths)] ** 2 + uncertainties[np.argmin(depths)] ** 2))
    significance = difference / combined_uncertainty if combined_uncertainty > 0 else np.nan
    values = {
        "sector_depths": [{"sector": item[0], "depth": item[1], "depth_uncertainty": item[2], "events": item[3]} for item in groups],
        "max_depth_difference": difference, "difference_sigma": significance,
    }
    status = "PASS"
    if not np.isfinite(significance):
        status = "NOT_AVAILABLE"
    elif significance >= config.sector_fail_sigma:
        status = "FAIL"
    elif significance >= config.sector_warning_sigma:
        status = "WARNING"
    return Diagnostic(
        status, values,
        f"WARNING at {config.sector_warning_sigma:g}σ; FAIL at {config.sector_fail_sigma:g}σ for the largest sector depth difference.",
        "Depth agreement across sectors supports repeatability; disagreement can reflect contamination, variability, or systematics.",
    )


def variability_diagnostic(data: pd.DataFrame, candidate: Candidate, config: VettingConfig) -> Diagnostic:
    mask = candidate_transit_mask(data["time"].to_numpy(), candidate.period_days, candidate.epoch_btjd, candidate.duration_hours, 1.5)
    oot = data.loc[~mask, "flux"].to_numpy(dtype=float)
    rms = float(np.sqrt(np.nanmean((oot - np.nanmedian(oot)) ** 2)))
    status = "PASS"
    if rms >= config.variability_fail_rms:
        status = "FAIL"
    elif rms >= config.variability_warning_rms:
        status = "WARNING"
    return Diagnostic(
        status, {"out_of_transit_rms": rms, "out_of_transit_cadences": len(oot)},
        f"WARNING at RMS {config.variability_warning_rms:g}; FAIL at RMS {config.variability_fail_rms:g}.",
        "Large out-of-transit variability can complicate a transit interpretation; it is not independently diagnostic of candidate type.",
    )


def crowding_diagnostic(metadata: dict[str, object], config: VettingConfig) -> Diagnostic:
    value = metadata.get("crowdsap")
    try:
        crowding = float(value)
    except (TypeError, ValueError):
        crowding = np.nan
    if not np.isfinite(crowding):
        return Diagnostic("NOT_AVAILABLE", {"crowdsap": None}, "SPOC CROWDSAP header value required.", "No SPOC crowding metric is available.")
    if crowding < config.crowding_fail_minimum:
        status = "FAIL"
    elif crowding < config.crowding_warning_minimum:
        status = "WARNING"
    else:
        status = "PASS"
    return Diagnostic(
        status, {"crowdsap": crowding, "estimated_contaminating_fraction": 1 - crowding},
        f"WARNING below CROWDSAP {config.crowding_warning_minimum:g}; FAIL below {config.crowding_fail_minimum:g}.",
        "CROWDSAP estimates the target's fraction of flux in the aperture; lower values imply stronger dilution/contamination risk.",
    )


def compute_pixel_measurement(
    *, sector: int, time: np.ndarray, flux_cube: np.ndarray, quality: np.ndarray | None,
    candidate: Candidate, expected_position_xy: tuple[float, float] | None, config: VettingConfig,
) -> PixelSectorMeasurement:
    """Calculate a compact difference-image source-location measurement."""
    valid = np.isfinite(time) & np.all(np.isfinite(flux_cube), axis=(1, 2))
    if quality is not None:
        valid &= np.asarray(quality) == 0
    in_mask = valid & candidate_transit_mask(time, candidate.period_days, candidate.epoch_btjd, candidate.duration_hours)
    near_mask = candidate_transit_mask(time, candidate.period_days, candidate.epoch_btjd, candidate.duration_hours, 3.0)
    out_mask = valid & ~near_mask
    if in_mask.sum() < config.pixel_minimum_in_cadences or out_mask.sum() < config.pixel_minimum_out_cadences:
        return PixelSectorMeasurement(sector, "NOT_AVAILABLE", "not available", expected_position_xy, None, None, None, None, int(in_mask.sum()), int(out_mask.sum()), "Insufficient usable in-/out-of-transit TPF cadences.")
    out_image = np.nanmean(flux_cube[out_mask], axis=0)
    in_image = np.nanmean(flux_cube[in_mask], axis=0)
    difference = out_image - in_image
    weights = np.clip(difference, 0, None)
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0:
        return PixelSectorMeasurement(sector, "NOT_AVAILABLE", "inconclusive", expected_position_xy, None, None, None, None, int(in_mask.sum()), int(out_mask.sum()), "Difference image has no positive flux-loss centroid.", out_image, in_image, difference)
    y, x = np.indices(weights.shape)
    source = (float((x * weights).sum() / total), float((y * weights).sum() / total))
    if expected_position_xy is None:
        interpretation, status, offset, offset_arcsec = "inconclusive", "WARNING", None, None
    else:
        offset = float(np.hypot(source[0] - expected_position_xy[0], source[1] - expected_position_xy[1]))
        offset_arcsec = offset * 21.0  # TESS pixels are approximately 21 arcsec.
        if offset <= config.pixel_consistent_offset_pixels:
            interpretation, status = "consistent with target", "PASS"
        elif offset >= config.pixel_off_target_offset_pixels:
            interpretation, status = "possible off-target contamination", "WARNING"
        else:
            interpretation, status = "inconclusive", "WARNING"
    return PixelSectorMeasurement(sector, status, interpretation, expected_position_xy, source, offset, offset_arcsec, None, int(in_mask.sum()), int(out_mask.sum()), None, out_image, in_image, difference)


def summarise_pixel_measurements(measurements: list[PixelSectorMeasurement]) -> Diagnostic:
    available = [measurement for measurement in measurements if measurement.status != "NOT_AVAILABLE"]
    values = {"sector_results": [measurement.summary() for measurement in measurements]}
    if not available:
        return Diagnostic("NOT_AVAILABLE", values, "Suitable SPOC TPF data and cadence coverage required.", "Pixel source check: NOT AVAILABLE.")
    if any(item.interpretation == "possible off-target contamination" for item in available):
        status, explanation = "WARNING", "At least one difference image is offset from the expected target location. This is not a confirmation of contamination."
    elif all(item.interpretation == "consistent with target" for item in available):
        status, explanation = "PASS", "Available difference-image centroids are consistent with the expected target position; this does not confirm planetary nature."
    else:
        status, explanation = "WARNING", "The available pixel results are inconclusive."
    return Diagnostic(status, values, "Offsets use 0.5 pixel for consistency and 1.0 pixel for possible off-target warning; no formal centroid significance is claimed.", explanation)


def _expected_tpf_position(tpf: Any) -> tuple[float, float] | None:
    """Prefer the TIC sky position projected through the TPF WCS."""
    try:
        from astropy import units as u
        from astropy.coordinates import SkyCoord

        ra = tpf.meta.get("RA_OBJ", tpf.meta.get("RA"))
        dec = tpf.meta.get("DEC_OBJ", tpf.meta.get("DEC"))
        if ra is not None and dec is not None and getattr(tpf, "wcs", None) is not None:
            x, y = tpf.wcs.world_to_pixel(SkyCoord(float(ra) * u.deg, float(dec) * u.deg))
            shape = tpf.shape
            if np.isfinite(x) and np.isfinite(y) and 0 <= x < shape[-1] and 0 <= y < shape[-2]:
                return float(x), float(y)
    except Exception:
        pass
    # Header WCS can be incomplete for some products; the SPOC aperture centre
    # is a labelled fallback rather than a claim of astrometric precision.
    mask = getattr(tpf, "pipeline_mask", None)
    if mask is not None and np.any(mask):
        y, x = np.where(mask)
        return float(np.mean(x)), float(np.mean(y))
    shape = getattr(tpf, "shape", (0, 0, 0))
    return ((shape[-1] - 1) / 2, (shape[-2] - 1) / 2) if len(shape) == 3 else None


@contextmanager
def _mast_timeout(seconds: float):
    """Interrupt a main-thread MAST operation instead of blocking a benchmark."""
    if seconds <= 0 or threading.current_thread() is not threading.main_thread():
        yield
        return
    def expired(_signum: int, _frame: object) -> None:
        raise TimeoutError(f"MAST TPF operation exceeded {seconds:g} seconds.")
    old_handler = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def retrieve_spoc_tpfs(tic_id: int, sectors: set[int], cache_directory: Path, timeout_seconds: float = 120.0) -> tuple[dict[int, Any], list[str]]:
    """Retrieve/cache at most one normal SPOC TPF per requested sector."""
    try:
        import lightkurve as lk

        with _mast_timeout(timeout_seconds):
            result = lk.search_targetpixelfile(f"TIC {tic_id}", mission="TESS", author="SPOC")
        tpfs: dict[int, Any] = {}
        errors: list[str] = []
        for index in selected_product_indices(result):
            row = result.table[index]
            sector = int(row["sequence_number"])
            if sector not in sectors:
                continue
            try:
                with _mast_timeout(timeout_seconds):
                    tpf = result[index].download(download_dir=str(cache_directory))
                if tpf is not None:
                    tpfs[sector] = tpf
            except Exception as error:
                errors.append(f"sector {sector}: {error}")
        return tpfs, errors
    except Exception as error:
        return {}, [f"TPF query unavailable: {error}"]


def pixel_source_diagnostic(
    candidate: Candidate, tpfs: dict[int, Any], config: VettingConfig
) -> tuple[Diagnostic, list[PixelSectorMeasurement]]:
    measurements: list[PixelSectorMeasurement] = []
    for sector, tpf in sorted(tpfs.items()):
        time = np.asarray(tpf.time.value, dtype=float)
        flux = np.asarray(getattr(tpf.flux, "value", tpf.flux), dtype=float)
        quality = np.asarray(getattr(tpf.quality, "value", tpf.quality)) if hasattr(tpf, "quality") else None
        measurements.append(compute_pixel_measurement(
            sector=sector, time=time, flux_cube=flux, quality=quality, candidate=candidate,
            expected_position_xy=_expected_tpf_position(tpf), config=config,
        ))
    return summarise_pixel_measurements(measurements), measurements


def candidate_vetting(data: pd.DataFrame, metadata: dict[str, object], candidate: Candidate, config: VettingConfig, pixel: Diagnostic) -> tuple[dict[str, Diagnostic], pd.DataFrame]:
    events = individual_transit_events(data, candidate, config)
    diagnostics = {
        "individual_transit_consistency": individual_consistency(events, config),
        "odd_even": odd_even_diagnostic(events, config),
        "secondary_eclipse": secondary_eclipse_diagnostic(data, candidate, config),
        "sector_consistency": sector_consistency(events, config),
        "out_of_transit_variability": variability_diagnostic(data, candidate, config),
        "crowding": crowding_diagnostic(metadata, config),
        "pixel_source_check": pixel,
    }
    statuses = [diagnostic.status for diagnostic in diagnostics.values() if diagnostic.status != "NOT_AVAILABLE"]
    overall = "FAIL" if "FAIL" in statuses else "WARNING" if "WARNING" in statuses else "PASS"
    diagnostics["overall"] = Diagnostic(overall, {"status_counts": {status: statuses.count(status) for status in ("PASS", "WARNING", "FAIL")}}, "FAIL if any available diagnostic fails; otherwise WARNING if any warns.", "Combined vetting result; it is not a planet confirmation.")
    return diagnostics, events
