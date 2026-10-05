import numpy as np
import pandas as pd

from exosignal.config import SearchConfig
from exosignal.preprocessing import process_sector
from exosignal.search import (
    _independent_of,
    add_stellar_radius_interpretation,
    candidate_transit_mask,
    run_bls,
    search_transits,
)


def _injected_lightcurve(period=3.2, depth=0.01, duration_days=0.14):
    rng = np.random.default_rng(7)
    time = np.arange(0.0, 30.0, 0.02)
    mask = candidate_transit_mask(time, period, 0.5, duration_days * 24)
    flux = rng.normal(0.0, 3e-4, len(time)) - depth * mask
    return time, flux, mask


def test_bls_recovers_deterministic_injected_period_without_period_hint():
    time, flux, _ = _injected_lightcurve()
    data = pd.DataFrame(
        {"time": time, "flux": flux, "flux_err": 3e-4, "sector": 1}
    )
    result = run_bls(
        data,
        SearchConfig(
            min_period_days=1.0,
            max_period_days=5.0,
            min_duration_hours=1.0,
            max_duration_hours=6.0,
            minimum_snr=4.0,
        ),
    )
    assert result is not None
    assert abs(result.candidate.period_days - 3.2) < 0.01
    assert result.candidate.observed_transits >= 3
    assert abs(result.candidate.radius_ratio - 0.1) < 0.01


def test_transit_masked_detrending_preserves_injected_depth():
    time = np.arange(0.0, 30.0, 0.02)
    injected_depth = 0.01
    mask = candidate_transit_mask(time, 3.2, 0.6, 0.12 * 24)
    long_trend = 1 + 0.015 * np.sin(2 * np.pi * time / 9)
    flux = long_trend * (1 - injected_depth * mask)
    result = process_sector(
        sector=1,
        time=time,
        flux=flux,
        flux_err=np.full(len(time), 2e-4),
        quality=np.zeros(len(time)),
        transit_mask=mask,
    )
    recovered_mask = candidate_transit_mask(
        result.data["time"].to_numpy(), 3.2, 0.6, 0.12 * 24
    )
    recovered_depth = -float(result.data.loc[recovered_mask, "flux"].median())
    assert recovered_depth > injected_depth * 0.9
    assert result.diagnostics["protected_transit_cadences"] > 0


def test_iterative_search_returns_candidate_and_refined_lightcurve():
    time, flux, _ = _injected_lightcurve()
    processed = pd.DataFrame(
        {
            "time": time,
            "flux": flux,
            "flux_err": 3e-4,
            "raw_normalized_flux": flux,
            "raw_normalized_flux_err": 3e-4,
            "sector": 1,
        }
    )
    candidates, refined, _ = search_transits(
        processed,
        search_config=SearchConfig(
            min_period_days=1.0,
            max_period_days=5.0,
            min_duration_hours=1.0,
            max_duration_hours=6.0,
            minimum_snr=4.0,
            max_candidates=1,
            tls_enabled=False,
        ),
    )
    assert len(candidates) == 1
    assert candidates[0].candidate_id == "C01"
    assert abs(candidates[0].period_days - 3.2) < 0.03
    assert len(refined) == len(processed)


def test_simple_harmonics_are_not_reported_as_independent_peaks():
    assert not _independent_of(1.6, [3.2])
    assert not _independent_of(6.4, [3.2])
    assert _independent_of(2.7, [3.2])


def test_radius_interpretation_requires_a_valid_stellar_radius():
    time, flux, _ = _injected_lightcurve()
    data = pd.DataFrame({"time": time, "flux": flux, "flux_err": 3e-4, "sector": 1})
    candidate = run_bls(
        data, SearchConfig(min_period_days=1, max_period_days=5, minimum_snr=4)
    ).candidate
    interpreted = add_stellar_radius_interpretation(candidate, 1.0)
    assert interpreted.approximate_planet_radius_earth is not None
    assert 9 < interpreted.approximate_planet_radius_earth < 13
