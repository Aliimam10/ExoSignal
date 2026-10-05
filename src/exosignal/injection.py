"""Compact transit injection/recovery in real TESS cadence/noise data."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .config import PreprocessingConfig, SearchConfig
from .preprocessing import combine_sectors, process_sector
from .search import Candidate, candidate_transit_mask, search_transits


@dataclass(frozen=True)
class Injection:
    period_days: float
    depth: float
    duration_hours: float
    epoch_btjd: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def inject_transit(data: pd.DataFrame, injection: Injection) -> pd.DataFrame:
    """Inject a box transit into pre-detrending, per-sector-normalised data.

    ``raw_normalized_flux`` is the saved PDCSAP flux after mandatory quality
    filtering and per-sector scale normalisation but before long-timescale
    detrending. Thus this experiment exercises the trend estimate and BLS
    search, rather than adding dips to the already-final ``flux`` column.
    """
    result = data.copy()
    mask = candidate_transit_mask(result["time"].to_numpy(), injection.period_days, injection.epoch_btjd, injection.duration_hours)
    raw = result["raw_normalized_flux"].to_numpy(dtype=float)
    result.loc[mask, "raw_normalized_flux"] = (raw[mask] + 1.0) * (1.0 - injection.depth) - 1.0
    return result


def preprocess_injected(data: pd.DataFrame, config: PreprocessingConfig) -> pd.DataFrame:
    processed = []
    for sector, group in data.groupby("sector", sort=True):
        raw = group["raw_normalized_flux"].to_numpy(dtype=float) + 1.0
        errors = group.get("raw_normalized_flux_err", group["flux_err"]).to_numpy(dtype=float)
        processed.append(process_sector(sector=int(sector), time=group["time"], flux=raw, flux_err=errors, quality=None, config=config))
    return combine_sectors(processed)


def recovery_kind(injected_period: float, recovered_period: float | None, tolerance: float = 0.01) -> str:
    """Classify exact and P/2, 2P recovery explicitly (fractional tolerance)."""
    if recovered_period is None or not np.isfinite(recovered_period):
        return "not_recovered"
    ratio = recovered_period / injected_period
    if abs(ratio - 1.0) <= tolerance:
        return "exact_period"
    if abs(ratio - 0.5) <= tolerance or abs(ratio - 2.0) <= 2 * tolerance:
        return "simple_harmonic"
    return "other_period"


def inject_and_recover(data: pd.DataFrame, injection: Injection, preprocessing_config: PreprocessingConfig | None = None, search_config: SearchConfig | None = None) -> dict[str, object]:
    """Run real-data injection through normal detrending and BLS detection."""
    preprocessing_config = preprocessing_config or PreprocessingConfig()
    search_config = search_config or SearchConfig(tls_enabled=False, max_candidates=1)
    injected = inject_transit(data, injection)
    processed = preprocess_injected(injected, preprocessing_config)
    candidates, _, _ = search_transits(processed, preprocessing_config, search_config)
    recovered: Candidate | None = candidates[0] if candidates else None
    return {
        "injection": injection.to_dict(),
        "recovered": recovered.to_dict() if recovered else None,
        "recovery": recovery_kind(injection.period_days, recovered.period_days if recovered else None),
        "period_fractional_error": (recovered.period_days / injection.period_days - 1.0) if recovered else None,
        "processed_cadences": len(processed),
    }


def compact_grid(data: pd.DataFrame, periods_days: list[float], depths: list[float], duration_hours: float, epoch_btjd: float, preprocessing_config: PreprocessingConfig | None = None, search_config: SearchConfig | None = None) -> pd.DataFrame:
    """Run a controlled period/depth grid and retain exact/harmonic outcomes."""
    rows = []
    for period in periods_days:
        for depth in depths:
            result = inject_and_recover(data, Injection(period, depth, duration_hours, epoch_btjd), preprocessing_config, search_config)
            row = dict(result["injection"])
            row.update({key: value for key, value in result.items() if key not in {"injection", "recovered"}})
            row["recovered_period_days"] = result["recovered"]["period_days"] if result["recovered"] else np.nan
            rows.append(row)
    return pd.DataFrame(rows)
