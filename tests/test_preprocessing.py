import numpy as np
import pytest

from exosignal.config import PreprocessingConfig
from exosignal.errors import UnusableLightCurveError
from exosignal.preprocessing import combine_sectors, process_sector


def test_removes_nonfinite_and_nonzero_quality_cadences():
    result = process_sector(
        sector=1,
        time=np.array([0.0, 0.1, np.nan, 0.3, 0.4, 0.5, 0.6]),
        flux=np.array([100.0, 100.0, 100.0, np.nan, 100.0, 100.0, 100.0]),
        flux_err=np.ones(7),
        quality=np.array([0, 1, 0, 0, 0, 0, 0]),
        config=PreprocessingConfig(min_segment_points=4),
    )
    assert len(result.data) == 4
    assert result.diagnostics["removed_nonfinite"] == 2
    assert result.diagnostics["removed_nonzero_quality"] == 1


def test_normalisation_and_errors_are_preserved():
    result = process_sector(
        sector=2,
        time=np.arange(30) / 10,
        flux=np.full(30, 200.0),
        flux_err=np.full(30, 2.0),
        quality=np.zeros(30, dtype=int),
    )
    assert np.allclose(result.data["flux"], 0.0)
    assert np.allclose(result.data["flux_err"], 0.01)
    assert np.allclose(result.data["raw_normalized_flux"], 0.0)


def test_positive_outlier_removed_but_downward_event_retained():
    flux = np.full(101, 100.0)
    flux[30] = 200.0
    flux[70] = 98.0
    result = process_sector(
        sector=3,
        time=np.arange(101) / 48,
        flux=flux,
        flux_err=np.ones(101),
        quality=np.zeros(101, dtype=int),
    )
    assert result.diagnostics["removed_positive_outliers"] == 1
    assert result.data["flux"].min() < -0.01


def test_combining_preserves_sector_identity_and_time_order():
    first = process_sector(
        sector=1, time=np.arange(30) / 10, flux=np.ones(30), flux_err=None, quality=None
    )
    second = process_sector(
        sector=2, time=10 + np.arange(30) / 10, flux=np.ones(30), flux_err=None, quality=None
    )
    combined = combine_sectors([second, first])
    assert combined["time"].is_monotonic_increasing
    assert set(combined["sector"]) == {1, 2}


def test_unusable_sector_is_explained():
    with pytest.raises(UnusableLightCurveError, match="usable cadences"):
        process_sector(
            sector=1,
            time=np.arange(5),
            flux=np.ones(5),
            flux_err=None,
            quality=np.zeros(5),
        )


def test_duplicate_timestamps_use_a_safe_constant_trend():
    result = process_sector(
        sector=4,
        time=np.zeros(30),
        flux=np.full(30, 100.0),
        flux_err=np.ones(30),
        quality=np.zeros(30),
    )
    assert len(result.data) == 30
    assert np.allclose(result.data["flux"], 0.0)
