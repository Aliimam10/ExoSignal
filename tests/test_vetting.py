import numpy as np
import pandas as pd

from exosignal.config import VettingConfig
from exosignal.search import Candidate
from exosignal.vetting import (
    compute_pixel_measurement,
    individual_transit_events,
    odd_even_diagnostic,
    secondary_eclipse_diagnostic,
    sector_consistency,
)


def _candidate():
    return Candidate(
        candidate_id="C01", period_days=2.0, epoch_btjd=0.5, duration_hours=2.4,
        depth=0.01, depth_err=0.001, radius_ratio=0.1, approximate_planet_radius_earth=None,
        bls_power=10.0, bls_snr=10.0, observed_transits=5,
    )


def test_odd_even_depth_difference_is_flagged():
    events = pd.DataFrame({
        "event_number": [0, 2, 4, 1, 3, 5],
        "depth": [0.01, 0.011, 0.009, 0.03, 0.031, 0.029],
        "category": ["detected"] * 6,
    })
    result = odd_even_diagnostic(events, VettingConfig())
    assert result.status == "FAIL"
    assert result.values["difference_sigma"] > 5


def test_secondary_eclipse_depth_is_flagged():
    candidate = _candidate()
    time = np.arange(0, 12, 0.01)
    phase = (time - candidate.epoch_btjd) % candidate.period_days / candidate.period_days
    flux = np.zeros(len(time))
    flux[np.abs(phase - 0.5) < 0.02] = -0.02
    result = secondary_eclipse_diagnostic(pd.DataFrame({"time": time, "flux": flux}), candidate, VettingConfig())
    assert result.status == "FAIL"
    assert result.values["secondary_depth"] > 0


def test_events_distinguish_intersector_gap_from_insufficient_cadence():
    candidate = _candidate()
    # Data cover only one of several predicted transits across the time span.
    time = np.r_[np.arange(0.35, 0.65, 0.01), np.arange(4.35, 4.65, 0.01)]
    flux = np.zeros(len(time))
    data = pd.DataFrame({"time": time, "flux": flux, "sector": 1})
    events = individual_transit_events(data, candidate, VettingConfig())
    assert "outside_observing_coverage" in set(events["category"])
    assert not events.loc[events["category"] == "outside_observing_coverage", "depth"].notna().any()


def test_sector_consistency_compares_measurable_sector_depths():
    events = pd.DataFrame({
        "category": ["detected"] * 4,
        "sector": [1, 1, 2, 2],
        "depth": [0.010, 0.011, 0.0105, 0.0095],
    })
    result = sector_consistency(events, VettingConfig())
    assert result.status == "PASS"
    assert len(result.values["sector_depths"]) == 2


def test_pixel_fixture_recovers_target_centroid_and_unavailable_case():
    candidate = _candidate()
    time = np.arange(0, 10, 0.05)
    cube = np.full((len(time), 5, 5), 100.0)
    in_transit = np.abs((time - candidate.epoch_btjd + 1) % 2 - 1) <= candidate.duration_hours / 48
    cube[in_transit, 2, 2] -= 10
    measurement = compute_pixel_measurement(
        sector=1, time=time, flux_cube=cube, quality=np.zeros(len(time), dtype=int), candidate=candidate,
        expected_position_xy=(2.0, 2.0), config=VettingConfig(),
    )
    assert measurement.interpretation == "consistent with target"
    assert measurement.offset_pixels is not None and measurement.offset_pixels < 0.1
    unavailable = compute_pixel_measurement(
        sector=1, time=time[:3], flux_cube=cube[:3], quality=np.zeros(3, dtype=int), candidate=candidate,
        expected_position_xy=(2.0, 2.0), config=VettingConfig(),
    )
    assert unavailable.status == "NOT_AVAILABLE"
