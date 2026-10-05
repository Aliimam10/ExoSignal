"""BLS-led transit-like signal search and TLS refinement for Commit 2."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd
from astropy.timeseries import BoxLeastSquares

from .config import PreprocessingConfig, SearchConfig
from .preprocessing import combine_sectors, process_sector


@dataclass(frozen=True)
class Candidate:
    """A measured transit-like periodic signal, not a planet classification."""

    candidate_id: str
    period_days: float
    epoch_btjd: float
    duration_hours: float
    depth: float
    depth_err: float | None
    radius_ratio: float
    approximate_planet_radius_earth: float | None
    bls_power: float
    bls_snr: float
    observed_transits: int
    tls_period_days: float | None = None
    tls_sde: float | None = None
    tls_snr: float | None = None
    tls_error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class BlsResult:
    candidate: Candidate
    periods: np.ndarray
    power: np.ndarray


def candidate_transit_mask(
    time: np.ndarray, period_days: float, epoch_btjd: float, duration_hours: float, factor: float = 1.0
) -> np.ndarray:
    """Mark predicted windows around transit centres without assuming continuity."""
    half_window = duration_hours / 24.0 * factor / 2.0
    phase_time = (time - epoch_btjd + period_days / 2.0) % period_days - period_days / 2.0
    return np.abs(phase_time) <= half_window


def _period_limits(time: np.ndarray, config: SearchConfig, centre: float | None = None) -> tuple[float, float]:
    baseline = float(np.nanmax(time) - np.nanmin(time))
    maximum = min(config.max_period_days, baseline / config.minimum_transits)
    minimum = config.min_period_days
    if centre is not None:
        minimum = max(minimum, centre * 0.97)
        maximum = min(maximum, centre * 1.03)
    if maximum <= minimum:
        raise ValueError("The usable baseline is too short for the configured BLS period range.")
    return minimum, maximum


def _frequency_factor(
    time: np.ndarray, minimum_period: float, maximum_period: float, config: SearchConfig
) -> float:
    """Cap global grid size while retaining the default fine grid when cheap."""
    baseline = float(np.nanmax(time) - np.nanmin(time))
    minimum_duration = config.min_duration_hours / 24.0
    native_samples = 1 + (
        (1 / minimum_period - 1 / maximum_period) * baseline**2 / minimum_duration
    )
    return max(1.0, native_samples / config.max_bls_period_samples)


def _independent_of(period: float, accepted_periods: list[float]) -> bool:
    """Reject near-identical peaks and common simple harmonic aliases."""
    aliases = (0.5, 1.0, 2.0, 1 / 3, 3.0)
    for other in accepted_periods:
        ratio = period / other
        if any(abs(ratio - alias) / alias < 0.02 for alias in aliases):
            return False
    return True


def _observed_transits(time: np.ndarray, candidate: Candidate) -> int:
    mask = candidate_transit_mask(
        time, candidate.period_days, candidate.epoch_btjd, candidate.duration_hours
    )
    event = np.rint((time[mask] - candidate.epoch_btjd) / candidate.period_days).astype(int)
    return len(np.unique(event))


def _bin_for_bls(frame: pd.DataFrame, bin_minutes: float) -> pd.DataFrame:
    """Bin only the BLS calculation within sectors, never across gaps/sectors."""
    if bin_minutes <= 0:
        return frame
    bin_days = bin_minutes / (24 * 60)
    binned: list[pd.DataFrame] = []
    for _, sector_data in frame.groupby("sector", sort=False):
        sector_data = sector_data.sort_values("time").copy()
        origin = float(sector_data["time"].iloc[0])
        sector_data["_bin"] = np.floor((sector_data["time"] - origin) / bin_days).astype(int)
        rows = []
        for _, group in sector_data.groupby("_bin", sort=True):
            errors = group["flux_err"].to_numpy(dtype=float)
            finite_errors = errors[np.isfinite(errors) & (errors > 0)]
            rows.append(
                {
                    "time": group["time"].mean(),
                    "flux": group["flux"].median(),
                    "flux_err": (
                        float(np.sqrt(np.sum(finite_errors**2)) / len(finite_errors))
                        if len(finite_errors)
                        else np.nan
                    ),
                    "sector": group["sector"].iloc[0],
                }
            )
        binned.append(pd.DataFrame(rows))
    return pd.concat(binned, ignore_index=True).sort_values("time", ignore_index=True)


def run_bls(
    data: pd.DataFrame,
    config: SearchConfig,
    excluded_mask: np.ndarray | None = None,
    period_centre: float | None = None,
) -> BlsResult | None:
    """Run a BLS search and return the strongest independent meaningful peak."""
    frame = data.copy()
    if excluded_mask is not None:
        frame = frame.loc[~excluded_mask].copy()
    finite = np.isfinite(frame["time"]) & np.isfinite(frame["flux"])
    frame = frame.loc[finite]
    frame = _bin_for_bls(frame, config.bls_bin_minutes)
    if len(frame) < 100:
        return None
    time = frame["time"].to_numpy(dtype=float)
    flux = frame["flux"].to_numpy(dtype=float)
    errors = frame["flux_err"].to_numpy(dtype=float)
    dy = errors if np.all(np.isfinite(errors) & (errors > 0)) else None
    minimum_period, maximum_period = _period_limits(time, config, period_centre)
    durations = np.linspace(config.min_duration_hours / 24.0, config.max_duration_hours / 24.0, 10)
    periodogram = BoxLeastSquares(time, flux, dy=dy).autopower(
        durations,
        minimum_period=minimum_period,
        maximum_period=maximum_period,
        objective="snr",
        method="fast",
        oversample=config.bls_oversample,
        frequency_factor=_frequency_factor(time, minimum_period, maximum_period, config),
    )
    power = np.asarray(periodogram.power, dtype=float)
    periods = np.asarray(periodogram.period, dtype=float)
    accepted_periods: list[float] = []
    for index in np.argsort(power)[::-1]:
        if not np.isfinite(power[index]) or not _independent_of(periods[index], accepted_periods):
            continue
        duration_days = float(periodogram.duration[index])
        depth = float(periodogram.depth[index])
        depth_err = float(periodogram.depth_err[index])
        candidate = Candidate(
            candidate_id="",
            period_days=float(periods[index]),
            epoch_btjd=float(periodogram.transit_time[index]),
            duration_hours=duration_days * 24.0,
            depth=depth,
            depth_err=depth_err if np.isfinite(depth_err) else None,
            radius_ratio=float(np.sqrt(max(depth, 0.0))),
            approximate_planet_radius_earth=None,
            bls_power=float(power[index]),
            bls_snr=float(power[index]),
            observed_transits=0,
        )
        candidate = replace(candidate, observed_transits=_observed_transits(time, candidate))
        if candidate.observed_transits < config.minimum_transits:
            accepted_periods.append(candidate.period_days)
            continue
        if candidate.bls_snr < config.minimum_snr:
            return None
        return BlsResult(candidate=candidate, periods=periods, power=power)
    return None


def redetrend_with_transit_masks(
    data: pd.DataFrame,
    candidates: list[Candidate],
    preprocessing_config: PreprocessingConfig,
    search_config: SearchConfig,
) -> pd.DataFrame:
    """Recalculate each sector trend while protecting predicted transit windows."""
    processed = []
    for sector, group in data.groupby("sector", sort=True):
        group = group.sort_values("time")
        time = group["time"].to_numpy(dtype=float)
        mask = np.zeros(len(group), dtype=bool)
        for candidate in candidates:
            mask |= candidate_transit_mask(
                time,
                candidate.period_days,
                candidate.epoch_btjd,
                candidate.duration_hours,
                search_config.transit_mask_duration_factor,
            )
        raw_flux = group["raw_normalized_flux"].to_numpy(dtype=float) + 1.0
        raw_error = group["raw_normalized_flux_err"].to_numpy(dtype=float)
        processed.append(
            process_sector(
                sector=int(sector),
                time=time,
                flux=raw_flux,
                flux_err=raw_error,
                quality=None,
                config=preprocessing_config,
                transit_mask=mask,
            )
        )
    return combine_sectors(processed)


def refine_with_tls(data: pd.DataFrame, candidate: Candidate, config: SearchConfig) -> Candidate:
    """Use TLS only to refine a promising BLS signal; failure stays explicit."""
    if not config.tls_enabled:
        return candidate
    try:
        from transitleastsquares import transitleastsquares

        frame = data.loc[np.isfinite(data["time"]) & np.isfinite(data["flux"])].copy()
        frame = _bin_for_bls(frame, config.tls_bin_minutes)
        errors = frame["flux_err"].to_numpy(dtype=float)
        if not np.all(np.isfinite(errors) & (errors > 0)):
            errors = None
        model = transitleastsquares(
            frame["time"].to_numpy(dtype=float), frame["flux"].to_numpy(dtype=float) + 1.0, errors
        )
        result = model.power(
            period_min=max(
                config.min_period_days,
                candidate.period_days * (1 - config.tls_period_window_fraction),
            ),
            period_max=min(
                config.max_period_days,
                candidate.period_days * (1 + config.tls_period_window_fraction),
            ),
            use_threads=config.tls_threads,
            show_progress_bar=False,
        )
        return replace(
            candidate,
            tls_period_days=float(result.period),
            tls_sde=float(result.SDE),
            tls_snr=float(result.snr),
        )
    except Exception as error:  # TLS is an optional refinement, never a hidden fallback.
        return replace(candidate, tls_error=str(error))


def search_transits(
    data: pd.DataFrame,
    preprocessing_config: PreprocessingConfig | None = None,
    search_config: SearchConfig | None = None,
) -> tuple[list[Candidate], pd.DataFrame, list[BlsResult]]:
    """Detect up to three independent BLS candidates with masked re-detrending."""
    preprocessing_config = preprocessing_config or PreprocessingConfig()
    search_config = search_config or SearchConfig()
    candidates: list[Candidate] = []
    periodograms: list[BlsResult] = []
    working = data
    for number in range(1, search_config.max_candidates + 1):
        excluded = np.zeros(len(working), dtype=bool)
        for candidate in candidates:
            excluded |= candidate_transit_mask(
                working["time"].to_numpy(),
                candidate.period_days,
                candidate.epoch_btjd,
                candidate.duration_hours,
                search_config.transit_mask_duration_factor,
            )
        preliminary = run_bls(working, search_config, excluded)
        if preliminary is None:
            break
        refined = redetrend_with_transit_masks(
            data, candidates + [preliminary.candidate], preprocessing_config, search_config
        )
        remasked = np.zeros(len(refined), dtype=bool)
        for candidate in candidates:
            remasked |= candidate_transit_mask(
                refined["time"].to_numpy(),
                candidate.period_days,
                candidate.epoch_btjd,
                candidate.duration_hours,
                search_config.transit_mask_duration_factor,
            )
        measured = run_bls(
            refined, search_config, remasked, period_centre=preliminary.candidate.period_days
        )
        result = measured or preliminary
        candidate = replace(result.candidate, candidate_id=f"C{number:02d}")
        candidate = refine_with_tls(refined, candidate, search_config)
        candidates.append(candidate)
        periodograms.append(BlsResult(candidate=candidate, periods=result.periods, power=result.power))
        working = refined
    return candidates, working, periodograms


def add_stellar_radius_interpretation(candidate: Candidate, stellar_radius_rsun: object | None) -> Candidate:
    """Attach a deliberately approximate radius estimate when stellar radius exists.

    This uses the small-planet depth relation only. Dilution, impact parameter,
    limb darkening, and stellar-radius uncertainty are not modelled here.
    """
    try:
        radius = float(stellar_radius_rsun)
    except (TypeError, ValueError):
        return candidate
    if not np.isfinite(radius) or radius <= 0:
        return candidate
    return replace(
        candidate,
        approximate_planet_radius_earth=candidate.radius_ratio * radius * 109.076,
    )
